"""Finance Service for managing financial operations."""

import uuid
from datetime import datetime, date
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger

from shared_libs.database import get_database
from models.finance import Expense, ExpenseCreate, Invoice, InvoiceCreate, BudgetCategory, PayrollRecord


class FinanceService:
    """Finance service for managing financial operations."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.expenses_collection = "expenses"
        self.invoices_collection = "invoices"
        self.budget_collection = "budget_categories"
        self.payroll_collection = "payroll_records"
    
    async def initialize(self):
        """Initialize the Finance service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.expenses_collection].create_index("expense_id", unique=True)
        await self.db[self.expenses_collection].create_index("employee_id")
        await self.db[self.expenses_collection].create_index("status")
        await self.db[self.expenses_collection].create_index("category")
        
        await self.db[self.invoices_collection].create_index("invoice_id", unique=True)
        await self.db[self.invoices_collection].create_index("client_email")
        await self.db[self.invoices_collection].create_index("status")
        
        await self.db[self.budget_collection].create_index("category_id", unique=True)
        await self.db[self.payroll_collection].create_index("payroll_id", unique=True)
        await self.db[self.payroll_collection].create_index("employee_id")
        
        logger.info("Finance service initialized")
    
    # Expense methods
    async def create_expense(self, expense_data: ExpenseCreate) -> Optional[Expense]:
        """Create a new expense."""
        try:
            expense_id = f"EXP{str(uuid.uuid4())[:8].upper()}"
            
            expense_dict = expense_data.dict()
            expense_dict["expense_id"] = expense_id
            expense_dict["created_at"] = datetime.utcnow()
            expense_dict["updated_at"] = datetime.utcnow()
            
            result = await self.db[self.expenses_collection].insert_one(expense_dict)
            
            if result.inserted_id:
                expense_dict["_id"] = result.inserted_id
                logger.info(f"Expense created: {expense_id}")
                return Expense(**expense_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create expense: {e}")
            return None
    
    async def get_expense(self, expense_id: str) -> Optional[Expense]:
        """Get an expense by ID."""
        try:
            expense_doc = await self.db[self.expenses_collection].find_one(
                {"expense_id": expense_id}
            )
            
            if expense_doc:
                return Expense(**expense_doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get expense {expense_id}: {e}")
            return None
    
    async def list_expenses(self, employee_id: Optional[str] = None, status: Optional[str] = None,
                          category: Optional[str] = None, skip: int = 0, limit: int = 10) -> List[Expense]:
        """List expenses with optional filters."""
        try:
            query = {}
            if employee_id:
                query["employee_id"] = employee_id
            if status:
                query["status"] = status
            if category:
                query["category"] = category
            
            expenses = []
            cursor = self.db[self.expenses_collection].find(query).skip(skip).limit(limit)
            
            async for expense_doc in cursor:
                expenses.append(Expense(**expense_doc))
            
            return expenses
            
        except Exception as e:
            logger.error(f"Failed to list expenses: {e}")
            return []
    
    async def approve_expense(self, expense_id: str, approved_by: str) -> bool:
        """Approve an expense."""
        try:
            result = await self.db[self.expenses_collection].update_one(
                {"expense_id": expense_id},
                {
                    "$set": {
                        "status": "approved",
                        "approved_by": approved_by,
                        "approved_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Expense approved: {expense_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to approve expense {expense_id}: {e}")
            return False
    
    # Invoice methods
    async def create_invoice(self, invoice_data: InvoiceCreate) -> Optional[Invoice]:
        """Create a new invoice."""
        try:
            invoice_id = f"INV{str(uuid.uuid4())[:8].upper()}"
            
            invoice_dict = invoice_data.dict()
            invoice_dict["invoice_id"] = invoice_id
            invoice_dict["created_at"] = datetime.utcnow()
            invoice_dict["updated_at"] = datetime.utcnow()
            
            result = await self.db[self.invoices_collection].insert_one(invoice_dict)
            
            if result.inserted_id:
                invoice_dict["_id"] = result.inserted_id
                logger.info(f"Invoice created: {invoice_id}")
                return Invoice(**invoice_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create invoice: {e}")
            return None
    
    async def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        """Get an invoice by ID."""
        try:
            invoice_doc = await self.db[self.invoices_collection].find_one(
                {"invoice_id": invoice_id}
            )
            
            if invoice_doc:
                return Invoice(**invoice_doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get invoice {invoice_id}: {e}")
            return None
    
    async def list_invoices(self, status: Optional[str] = None, skip: int = 0, limit: int = 10) -> List[Invoice]:
        """List invoices with optional filters."""
        try:
            query = {}
            if status:
                query["status"] = status
            
            invoices = []
            cursor = self.db[self.invoices_collection].find(query).skip(skip).limit(limit)
            
            async for invoice_doc in cursor:
                invoices.append(Invoice(**invoice_doc))
            
            return invoices
            
        except Exception as e:
            logger.error(f"Failed to list invoices: {e}")
            return []
    
    async def mark_invoice_paid(self, invoice_id: str) -> bool:
        """Mark an invoice as paid."""
        try:
            result = await self.db[self.invoices_collection].update_one(
                {"invoice_id": invoice_id},
                {
                    "$set": {
                        "status": "paid",
                        "paid_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Invoice marked as paid: {invoice_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to mark invoice as paid {invoice_id}: {e}")
            return False
    
    # Budget methods
    async def create_budget_category(self, name: str, allocated_amount: float, 
                                   period: str, year: int, month: Optional[int] = None) -> Optional[BudgetCategory]:
        """Create a budget category."""
        try:
            category_id = f"BUD{str(uuid.uuid4())[:8].upper()}"
            
            budget_dict = {
                "category_id": category_id,
                "name": name,
                "allocated_amount": allocated_amount,
                "spent_amount": 0.0,
                "period": period,
                "year": year,
                "month": month,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await self.db[self.budget_collection].insert_one(budget_dict)
            
            if result.inserted_id:
                budget_dict["_id"] = result.inserted_id
                logger.info(f"Budget category created: {category_id}")
                return BudgetCategory(**budget_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create budget category: {e}")
            return None
    
    async def get_budget_summary(self, year: int, month: Optional[int] = None) -> dict:
        """Get budget summary for a period."""
        try:
            query = {"year": year}
            if month:
                query["month"] = month
            
            total_allocated = 0.0
            total_spent = 0.0
            categories = []
            
            cursor = self.db[self.budget_collection].find(query)
            
            async for budget_doc in cursor:
                total_allocated += budget_doc["allocated_amount"]
                total_spent += budget_doc["spent_amount"]
                categories.append({
                    "name": budget_doc["name"],
                    "allocated": budget_doc["allocated_amount"],
                    "spent": budget_doc["spent_amount"],
                    "remaining": budget_doc["allocated_amount"] - budget_doc["spent_amount"]
                })
            
            return {
                "total_allocated": total_allocated,
                "total_spent": total_spent,
                "total_remaining": total_allocated - total_spent,
                "categories": categories
            }
            
        except Exception as e:
            logger.error(f"Failed to get budget summary: {e}")
            return {}
