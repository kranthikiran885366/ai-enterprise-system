"""Budget management routes."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from shared_libs.auth import get_current_user


router = APIRouter()


class BudgetCategoryCreate(BaseModel):
    name: str
    allocated_amount: float
    period: str  # monthly, quarterly, yearly
    year: int
    month: Optional[int] = None


@router.post("/categories", response_model=dict)
async def create_budget_category(
    budget_data: BudgetCategoryCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a budget category."""
    from main import app
    finance_service = app.state.finance_service
    
    budget_category = await finance_service.create_budget_category(
        name=budget_data.name,
        allocated_amount=budget_data.allocated_amount,
        period=budget_data.period,
        year=budget_data.year,
        month=budget_data.month
    )
    
    if not budget_category:
        raise HTTPException(status_code=400, detail="Failed to create budget category")
    
    return {
        "message": "Budget category created successfully",
        "category_id": budget_category.category_id,
        "name": budget_category.name,
        "allocated_amount": budget_category.allocated_amount
    }


@router.get("/summary", response_model=dict)
async def get_budget_summary(
    year: int = Query(...),
    month: Optional[int] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Get budget summary for a period."""
    from main import app
    finance_service = app.state.finance_service
    
    summary = await finance_service.get_budget_summary(year, month)
    
    return summary


budget_router = router
