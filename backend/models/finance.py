from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ExpenseCategory(str, Enum):
    travel = "travel"
    meals = "meals"
    software = "software"
    hardware = "hardware"
    marketing = "marketing"
    office_supplies = "office_supplies"
    training = "training"
    utilities = "utilities"
    salaries = "salaries"
    misc = "misc"


class ExpenseStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    reimbursed = "reimbursed"


class InvoiceStatus(str, Enum):
    draft = "draft"
    sent = "sent"
    paid = "paid"
    overdue = "overdue"
    cancelled = "cancelled"


class ExpenseCreate(BaseModel):
    employee_id: int
    category: ExpenseCategory
    amount: float = Field(..., gt=0)
    description: str
    expense_date: str
    receipt_url: Optional[str] = None
    project_code: Optional[str] = None
    department: Optional[str] = None


class ExpenseUpdate(BaseModel):
    status: ExpenseStatus
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None


class InvoiceCreate(BaseModel):
    vendor_name: str
    vendor_email: Optional[str] = None
    amount: float = Field(..., gt=0)
    tax_amount: float = 0.0
    due_date: str
    po_number: Optional[str] = None
    line_items: Optional[List[dict]] = []
    notes: Optional[str] = None
    payment_terms: str = "net30"


class BudgetCreate(BaseModel):
    department: str
    period: str
    fiscal_year: int
    allocated_amount: float = Field(..., gt=0)
    category: Optional[str] = None


class BudgetUpdate(BaseModel):
    allocated_amount: Optional[float] = None
    notes: Optional[str] = None


class ForecastRequest(BaseModel):
    period_months: int = Field(default=3, ge=1, le=24)
    include_departments: Optional[List[str]] = None
    scenario: str = "base"


class TransactionCreate(BaseModel):
    type: str
    amount: float
    description: str
    department: str
    category: str
    reference_id: Optional[str] = None
    transaction_date: str
