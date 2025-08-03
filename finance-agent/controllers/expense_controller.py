"""Expense controller with business logic."""

from typing import List, Optional
from fastapi import HTTPException, status
from loguru import logger
from datetime import datetime

from services.finance_service import FinanceService
from services.ai_finance import AIFinanceService
from models.finance import Expense, ExpenseCreate, ExpenseUpdate
from utils.validators import validate_expense_data
from utils.notifications import send_expense_notification
from shared_libs.intelligence import get_business_rules


class ExpenseController:
    """Controller for expense operations."""
    
    def __init__(self, finance_service: FinanceService, ai_finance: AIFinanceService):
        self.finance_service = finance_service
        self.ai_finance = ai_finance
    
    async def create_expense(self, expense_data: ExpenseCreate, current_user: dict) -> Expense:
        """Create expense with AI-powered validation and auto-approval."""
        try:
            # Validate expense data
            validation_result = await validate_expense_data(expense_data.dict())
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Validation failed: {validation_result.errors}"
                )
            
            # AI-powered expense analysis
            expense_analysis = await self.ai_finance.analyze_expense_legitimacy(
                expense_data.dict(), current_user
            )
            
            # Business rules evaluation
            business_rules = await get_business_rules()
            approval_decision = await business_rules.evaluate_expense_approval(
                {**expense_data.dict(), "employee_level": current_user.get("level", "junior")}
            )
            
            # Create expense
            expense = await self.finance_service.create_expense(expense_data)
            if not expense:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create expense"
                )
            
            # Auto-approve if rules allow
            if approval_decision.get("auto_approve", False):
                await self.finance_service.approve_expense(expense.expense_id, "system_auto_approval")
                
                # Send approval notification
                await send_expense_notification(
                    current_user.get("email", ""),
                    "expense_auto_approved",
                    {
                        "expense_id": expense.expense_id,
                        "amount": expense.amount,
                        "reason": approval_decision.get("reason", "Auto-approved")
                    }
                )
            else:
                # Send for manual approval
                await self._route_for_approval(expense, approval_decision, current_user)
            
            # Flag suspicious expenses
            if expense_analysis.get("risk_score", 0) > 0.7:
                await self._flag_suspicious_expense(expense, expense_analysis)
            
            logger.info(f"Expense created: {expense.expense_id}, auto_approved: {approval_decision.get('auto_approve', False)}")
            
            return expense
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to create expense: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def _route_for_approval(self, expense: Expense, approval_decision: dict, current_user: dict):
        """Route expense for appropriate approval level."""
        try:
            approval_level = approval_decision.get("approval_level", "manager")
            
            # Get approver based on level
            approver_email = await self._get_approver_email(current_user, approval_level)
            
            if approver_email:
                await send_expense_notification(
                    approver_email,
                    "expense_approval_required",
                    {
                        "expense_id": expense.expense_id,
                        "employee_name": current_user.get("name", "Unknown"),
                        "amount": expense.amount,
                        "category": expense.category,
                        "description": expense.description,
                        "approval_level": approval_level
                    }
                )
            
        except Exception as e:
            logger.error(f"Failed to route expense for approval: {e}")
    
    async def _get_approver_email(self, current_user: dict, approval_level: str) -> str:
        """Get approver email based on approval level."""
        # In real implementation, this would query employee hierarchy
        approver_mapping = {
            "manager": "manager@company.com",
            "director": "director@company.com", 
            "ceo": "ceo@company.com"
        }
        return approver_mapping.get(approval_level, "finance@company.com")
    
    async def _flag_suspicious_expense(self, expense: Expense, analysis: dict):
        """Flag suspicious expense for review."""
        try:
            await send_expense_notification(
                "finance-audit@company.com",
                "suspicious_expense_detected",
                {
                    "expense_id": expense.expense_id,
                    "risk_score": analysis.get("risk_score", 0),
                    "risk_factors": analysis.get("risk_factors", []),
                    "amount": expense.amount,
                    "employee_id": expense.employee_id
                }
            )
            
            logger.warning(f"Suspicious expense flagged: {expense.expense_id}")
            
        except Exception as e:
            logger.error(f"Failed to flag suspicious expense: {e}")