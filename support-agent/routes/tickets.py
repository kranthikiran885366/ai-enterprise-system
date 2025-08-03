"""Ticket management routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from shared_libs.auth import get_current_user
from models.support import TicketCreate, TicketUpdate
from controllers.support_controller import SupportController


router = APIRouter()


@router.post("/", response_model=dict)
async def create_ticket(
    ticket_data: TicketCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new support ticket."""
    from main import app
    controller = SupportController(app.state.support_service, app.state.ai_support_service)
    
    return await controller.create_ticket(ticket_data.dict(), current_user)


@router.get("/", response_model=List[dict])
async def list_tickets(
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    assigned_to: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """List tickets with filters."""
    from main import app
    controller = SupportController(app.state.support_service, app.state.ai_support_service)
    
    return await controller.list_tickets(status, priority, category, assigned_to, page, limit, current_user)


@router.get("/{ticket_id}", response_model=dict)
async def get_ticket(
    ticket_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get ticket details."""
    from main import app
    controller = SupportController(app.state.support_service, app.state.ai_support_service)
    
    return await controller.get_ticket(ticket_id, current_user)


@router.put("/{ticket_id}", response_model=dict)
async def update_ticket(
    ticket_id: str,
    update_data: TicketUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update ticket."""
    from main import app
    controller = SupportController(app.state.support_service, app.state.ai_support_service)
    
    return await controller.update_ticket(ticket_id, update_data.dict(), current_user)


@router.post("/{ticket_id}/escalate", response_model=dict)
async def escalate_ticket(
    ticket_id: str,
    escalation_reason: str = Query(...),
    current_user: dict = Depends(get_current_user)
):
    """Escalate ticket to higher support tier."""
    from main import app
    controller = SupportController(app.state.support_service, app.state.ai_support_service)
    
    return await controller.escalate_ticket(ticket_id, escalation_reason, current_user)


tickets_router = router