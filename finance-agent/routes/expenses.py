"""Expense management routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import date

from shared_libs.auth import get_current_user
from models.finance import ExpenseCreate, Expense


router = APIRouter()


@router.post("/", response_model=dict)
async def create_expense(
    expense_data: ExpenseCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new expense."""
    from main import app
    finance_service = app.state.finance_service
    
    expense = await finance_service.create_expense(expense_data)
    
    if not expense:
        raise HTTPException(status_code=400, detail="Failed to create expense")
    
    return {
        "message": "Expense created successfully",
        "expense_id": expense.expense_id,
        "amount": expense.amount,
        "status": expense.status
    }


@router.get("/", response_model=List[dict])
async def list_expenses(
    employee_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """List expenses with filters."""
    from main import app
    finance_service = app.state.finance_service
    
    skip = (page - 1) * limit
    expenses = await finance_service.list_expenses(employee_id, status, category, skip, limit)
    
    return [
        {
            "expense_id": exp.expense_id,
            "employee_id": exp.employee_id,
            "amount": exp.amount,
            "currency": exp.currency,
            "category": exp.category,
            "description": exp.description,
            "expense_date": exp.expense_date.isoformat(),
            "status": exp.status,
            "created_at": exp.created_at.isoformat()
        } for exp in expenses
    ]


@router.get("/{expense_id}", response_model=dict)
async def get_expense(
    expense_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get an expense by ID."""
    from main import app
    finance_service = app.state.finance_service
    
    expense = await finance_service.get_expense(expense_id)
    
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")
    
    return {
        "expense_id": expense.expense_id,
        "employee_id": expense.employee_id,
        "amount": expense.amount,
        "currency": expense.currency,
        "category": expense.category,
        "description": expense.description,
        "expense_date": expense.expense_date.isoformat(),
        "status": expense.status,
        "approved_by": expense.approved_by,
        "approved_at": expense.approved_at.isoformat() if expense.approved_at else None,
        "notes": expense.notes,
        "created_at": expense.created_at.isoformat()
    }


@router.put("/{expense_id}/approve", response_model=dict)
async def approve_expense(
    expense_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Approve an expense."""
    from main import app
    finance_service = app.state.finance_service
    
    success = await finance_service.approve_expense(expense_id, current_user["sub"])
    
    if not success:
        raise HTTPException(status_code=404, detail="Expense not found or already processed")
    
    return {"message": "Expense approved successfully"}


expenses_router = router
