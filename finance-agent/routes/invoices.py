"""Invoice management routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from shared_libs.auth import get_current_user
from models.finance import InvoiceCreate


router = APIRouter()


@router.post("/", response_model=dict)
async def create_invoice(
    invoice_data: InvoiceCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new invoice."""
    from main import app
    finance_service = app.state.finance_service
    
    invoice = await finance_service.create_invoice(invoice_data)
    
    if not invoice:
        raise HTTPException(status_code=400, detail="Failed to create invoice")
    
    return {
        "message": "Invoice created successfully",
        "invoice_id": invoice.invoice_id,
        "amount": invoice.amount,
        "client_name": invoice.client_name
    }


@router.get("/", response_model=List[dict])
async def list_invoices(
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """List invoices with filters."""
    from main import app
    finance_service = app.state.finance_service
    
    skip = (page - 1) * limit
    invoices = await finance_service.list_invoices(status, skip, limit)
    
    return [
        {
            "invoice_id": inv.invoice_id,
            "client_name": inv.client_name,
            "client_email": inv.client_email,
            "amount": inv.amount,
            "currency": inv.currency,
            "description": inv.description,
            "issue_date": inv.issue_date.isoformat(),
            "due_date": inv.due_date.isoformat(),
            "status": inv.status,
            "created_at": inv.created_at.isoformat()
        } for inv in invoices
    ]


@router.get("/{invoice_id}", response_model=dict)
async def get_invoice(
    invoice_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get an invoice by ID."""
    from main import app
    finance_service = app.state.finance_service
    
    invoice = await finance_service.get_invoice(invoice_id)
    
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    return {
        "invoice_id": invoice.invoice_id,
        "client_name": invoice.client_name,
        "client_email": invoice.client_email,
        "amount": invoice.amount,
        "currency": invoice.currency,
        "description": invoice.description,
        "line_items": invoice.line_items,
        "issue_date": invoice.issue_date.isoformat(),
        "due_date": invoice.due_date.isoformat(),
        "status": invoice.status,
        "paid_at": invoice.paid_at.isoformat() if invoice.paid_at else None,
        "notes": invoice.notes,
        "created_at": invoice.created_at.isoformat()
    }


@router.put("/{invoice_id}/mark-paid", response_model=dict)
async def mark_invoice_paid(
    invoice_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Mark an invoice as paid."""
    from main import app
    finance_service = app.state.finance_service
    
    success = await finance_service.mark_invoice_paid(invoice_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    return {"message": "Invoice marked as paid successfully"}


invoices_router = router
