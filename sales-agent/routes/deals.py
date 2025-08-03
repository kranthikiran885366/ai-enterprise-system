"""Deal management routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import date

from shared_libs.auth import get_current_user
from models.sales import DealCreate


router = APIRouter()


@router.post("/", response_model=dict)
async def create_deal(
    deal_data: DealCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new deal."""
    from main import app
    controller = SalesController(app.state.sales_service, app.state.ai_sales_service)
    
    return await controller.create_deal(deal_data.dict(), current_user)


@router.get("/", response_model=List[dict])
async def list_deals(
    stage: Optional[str] = Query(None),
    assigned_to: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """List deals with filters."""
    from main import app
    controller = SalesController(app.state.sales_service, app.state.ai_sales_service)
    
    return await controller.list_deals(stage, assigned_to, page, limit, current_user)


@router.get("/pipeline", response_model=dict)
async def get_pipeline(
    sales_rep: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """Get sales pipeline overview."""
    from main import app
    sales_service = app.state.sales_service
    
    pipeline = await sales_service.get_sales_pipeline(sales_rep)
    return pipeline


@router.put("/{deal_id}/stage", response_model=dict)
async def update_deal_stage(
    deal_id: str,
    stage: str,
    probability: float = Query(..., ge=0, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Update deal stage and probability."""
    from main import app
    sales_service = app.state.sales_service
    
    success = await sales_service.update_deal_stage(deal_id, stage, probability, current_user["sub"])
    
    if not success:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    return {"message": "Deal stage updated successfully"}


deals_router = router