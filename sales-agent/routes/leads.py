"""Lead management routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime

from shared_libs.auth import get_current_user
from models.sales import LeadCreate, Lead
from controllers.sales_controller import SalesController


router = APIRouter()


@router.post("/", response_model=dict)
async def create_lead(
    lead_data: LeadCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new lead."""
    from main import app
    controller = SalesController(app.state.sales_service, app.state.ai_sales_service)
    
    return await controller.create_lead(lead_data.dict(), current_user)


@router.get("/", response_model=List[dict])
async def list_leads(
    status: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    assigned_to: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """List leads with filters."""
    from main import app
    controller = SalesController(app.state.sales_service, app.state.ai_sales_service)
    
    return await controller.list_leads(status, source, assigned_to, page, limit, current_user)


@router.get("/{lead_id}", response_model=dict)
async def get_lead(
    lead_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a lead by ID."""
    from main import app
    controller = SalesController(app.state.sales_service, app.state.ai_sales_service)
    
    return await controller.get_lead(lead_id, current_user)


@router.put("/{lead_id}/status", response_model=dict)
async def update_lead_status(
    lead_id: str,
    status: str,
    current_user: dict = Depends(get_current_user)
):
    """Update lead status."""
    from main import app
    controller = SalesController(app.state.sales_service, app.state.ai_sales_service)
    
    return await controller.update_lead_status(lead_id, status, current_user)


@router.post("/{lead_id}/score", response_model=dict)
async def score_lead(
    lead_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Score lead using AI."""
    from main import app
    controller = SalesController(app.state.sales_service, app.state.ai_sales_service)
    
    return await controller.score_lead(lead_id, current_user)


leads_router = router