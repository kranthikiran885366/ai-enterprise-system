"""Finance models for Finance Agent."""

from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel
from enum import Enum
from decimal import Decimal

from shared_libs.models import BaseDocument


class ExpenseCategory(str, Enum):
    TRAVEL = "travel"
    OFFICE_SUPPLIES = "office_supplies"
    MARKETING = "marketing"
    UTILITIES = "utilities"
    SOFTWARE = "software"
    EQUIPMENT = "equipment"
    MEALS = "meals"
    OTHER = "other"


class ExpenseStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PAID = "paid"


class InvoiceStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class Expense(BaseDocument):
    """Expense model."""
    expense_id: str
    employee_id: str
    amount: float
    currency: str = "USD"
    category: ExpenseCategory
    description: str
    receipt_url: Optional[str] = None
    expense_date: date
    status: ExpenseStatus = ExpenseStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    notes: Optional[str] = None


class ExpenseCreate(BaseModel):
    """Expense creation model."""
    employee_id: str
    amount: float
    currency: str = "USD"
    category: ExpenseCategory
    description: str
    receipt_url: Optional[str] = None
    expense_date: date
    notes: Optional[str] = None


class Invoice(BaseDocument):
    """Invoice model."""
    invoice_id: str
    client_name: str
    client_email: str
    amount: float
    currency: str = "USD"
    description: str
    line_items: List[dict] = []
    issue_date: date
    due_date: date
    status: InvoiceStatus = InvoiceStatus.DRAFT
    paid_at: Optional[datetime] = None
    notes: Optional[str] = None


class InvoiceCreate(BaseModel):
    """Invoice creation model."""
    client_name: str
    client_email: str
    amount: float
    currency: str = "USD"
    description: str
    line_items: List[dict] = []
    issue_date: date
    due_date: date
    notes: Optional[str] = None


class BudgetCategory(BaseDocument):
    """Budget category model."""
    category_id: str
    name: str
    description: Optional[str] = None
    allocated_amount: float
    spent_amount: float = 0.0
    currency: str = "USD"
    period: str  # monthly, quarterly, yearly
    year: int
    month: Optional[int] = None
    quarter: Optional[int] = None


class PayrollRecord(BaseDocument):
    """Payroll record model."""
    payroll_id: str
    employee_id: str
    pay_period_start: date
    pay_period_end: date
    base_salary: float
    overtime_hours: float = 0.0
    overtime_rate: float = 0.0
    bonuses: float = 0.0
    deductions: float = 0.0
    gross_pay: float
    tax_deductions: float
    net_pay: float
    currency: str = "USD"
    status: str = "pending"  # pending, processed, paid
