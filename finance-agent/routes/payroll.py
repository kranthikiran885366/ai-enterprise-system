"""Payroll management routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from datetime import date

from shared_libs.auth import get_current_user


router = APIRouter()


class PayrollCreate(BaseModel):
    employee_id: str
    pay_period_start: date
    pay_period_end: date
    base_salary: float
    overtime_hours: float = 0.0
    overtime_rate: float = 0.0
    bonuses: float = 0.0
    deductions: float = 0.0
    tax_rate: float = 0.25  # 25% default tax rate


@router.post("/", response_model=dict)
async def create_payroll_record(
    payroll_data: PayrollCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a payroll record."""
    # Calculate gross and net pay
    overtime_pay = payroll_data.overtime_hours * payroll_data.overtime_rate
    gross_pay = payroll_data.base_salary + overtime_pay + payroll_data.bonuses
    tax_deductions = gross_pay * payroll_data.tax_rate
    net_pay = gross_pay - tax_deductions - payroll_data.deductions
    
    return {
        "message": "Payroll record created successfully",
        "payroll_id": f"PAY{str(hash(payroll_data.employee_id))[:8].upper()}",
        "employee_id": payroll_data.employee_id,
        "gross_pay": gross_pay,
        "tax_deductions": tax_deductions,
        "net_pay": net_pay,
        "status": "pending"
    }


@router.get("/", response_model=List[dict])
async def list_payroll_records(
    employee_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """List payroll records with filters."""
    # Mock data for demonstration
    records = [
        {
            "payroll_id": "PAY001",
            "employee_id": "EMP001",
            "pay_period_start": "2024-01-01",
            "pay_period_end": "2024-01-15",
            "gross_pay": 5000.0,
            "tax_deductions": 1250.0,
            "net_pay": 3750.0,
            "status": "processed"
        },
        {
            "payroll_id": "PAY002",
            "employee_id": "EMP002",
            "pay_period_start": "2024-01-01",
            "pay_period_end": "2024-01-15",
            "gross_pay": 4500.0,
            "tax_deductions": 1125.0,
            "net_pay": 3375.0,
            "status": "pending"
        }
    ]
    
    # Apply filters
    if employee_id:
        records = [rec for rec in records if rec["employee_id"] == employee_id]
    if status:
        records = [rec for rec in records if rec["status"] == status]
    
    return records


@router.get("/{payroll_id}", response_model=dict)
async def get_payroll_record(
    payroll_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a payroll record by ID."""
    # Mock data for demonstration
    return {
        "payroll_id": payroll_id,
        "employee_id": "EMP001",
        "pay_period_start": "2024-01-01",
        "pay_period_end": "2024-01-15",
        "base_salary": 4000.0,
        "overtime_hours": 10.0,
        "overtime_rate": 50.0,
        "bonuses": 500.0,
        "deductions": 100.0,
        "gross_pay": 5000.0,
        "tax_deductions": 1250.0,
        "net_pay": 3750.0,
        "status": "processed"
    }


payroll_router = router
