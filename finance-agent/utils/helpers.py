"""Helper utilities for Finance Agent."""

import hashlib
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal
import calendar


def generate_expense_id() -> str:
    """Generate unique expense ID."""
    timestamp = datetime.utcnow().strftime("%y%m%d")
    random_suffix = str(uuid.uuid4())[:6].upper()
    return f"EXP{timestamp}{random_suffix}"


def generate_invoice_id() -> str:
    """Generate unique invoice ID."""
    timestamp = datetime.utcnow().strftime("%y%m%d")
    random_suffix = str(uuid.uuid4())[:6].upper()
    return f"INV{timestamp}{random_suffix}"


def calculate_tax_amount(amount: float, tax_rate: float) -> float:
    """Calculate tax amount."""
    return round(amount * tax_rate, 2)


def calculate_net_amount(gross_amount: float, tax_rate: float, deductions: float = 0) -> float:
    """Calculate net amount after tax and deductions."""
    tax_amount = calculate_tax_amount(gross_amount, tax_rate)
    return round(gross_amount - tax_amount - deductions, 2)


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency."""
    if currency == "USD":
        return f"${amount:,.2f}"
    elif currency == "EUR":
        return f"€{amount:,.2f}"
    elif currency == "GBP":
        return f"£{amount:,.2f}"
    else:
        return f"{currency} {amount:,.2f}"


def calculate_days_overdue(due_date: date) -> int:
    """Calculate days overdue from due date."""
    if due_date >= date.today():
        return 0
    return (date.today() - due_date).days


def calculate_late_fee(amount: float, days_overdue: int, late_fee_rate: float = 0.015) -> float:
    """Calculate late fee based on days overdue."""
    if days_overdue <= 0:
        return 0.0
    
    # 1.5% per month, prorated daily
    monthly_rate = late_fee_rate
    daily_rate = monthly_rate / 30
    
    return round(amount * daily_rate * days_overdue, 2)


def get_fiscal_year(date_obj: date) -> int:
    """Get fiscal year for a given date (assuming April-March fiscal year)."""
    if date_obj.month >= 4:  # April to December
        return date_obj.year
    else:  # January to March
        return date_obj.year - 1


def get_quarter(date_obj: date) -> int:
    """Get quarter for a given date."""
    return (date_obj.month - 1) // 3 + 1


def calculate_budget_utilization(allocated: float, spent: float) -> Dict[str, Any]:
    """Calculate budget utilization metrics."""
    if allocated <= 0:
        return {"utilization_percentage": 0, "remaining": 0, "status": "invalid"}
    
    utilization = (spent / allocated) * 100
    remaining = allocated - spent
    
    if utilization > 100:
        status = "over_budget"
    elif utilization > 90:
        status = "critical"
    elif utilization > 75:
        status = "warning"
    else:
        status = "healthy"
    
    return {
        "utilization_percentage": round(utilization, 2),
        "remaining": round(remaining, 2),
        "status": status,
        "overspend": max(0, spent - allocated)
    }


def calculate_expense_statistics(expenses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate expense statistics."""
    if not expenses:
        return {}
    
    total_amount = sum(exp.get("amount", 0) for exp in expenses)
    avg_amount = total_amount / len(expenses)
    
    # Category breakdown
    category_totals = {}
    for exp in expenses:
        category = exp.get("category", "other")
        category_totals[category] = category_totals.get(category, 0) + exp.get("amount", 0)
    
    # Status breakdown
    status_counts = {}
    for exp in expenses:
        status = exp.get("status", "pending")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Monthly breakdown
    monthly_totals = {}
    for exp in expenses:
        expense_date = exp.get("expense_date")
        if expense_date:
            if isinstance(expense_date, str):
                expense_date = datetime.fromisoformat(expense_date).date()
            month_key = expense_date.strftime("%Y-%m")
            monthly_totals[month_key] = monthly_totals.get(month_key, 0) + exp.get("amount", 0)
    
    return {
        "total_expenses": len(expenses),
        "total_amount": round(total_amount, 2),
        "average_amount": round(avg_amount, 2),
        "category_breakdown": category_totals,
        "status_breakdown": status_counts,
        "monthly_breakdown": monthly_totals
    }


def validate_expense_policy(expense_data: Dict[str, Any], employee_data: Dict[str, Any]) -> List[str]:
    """Validate expense against company policy."""
    violations = []
    
    amount = expense_data.get("amount", 0)
    category = expense_data.get("category", "")
    employee_level = employee_data.get("level", "junior")
    
    # Category-specific limits
    category_limits = {
        "meals": {"junior": 50, "senior": 75, "manager": 100, "director": 150},
        "travel": {"junior": 2000, "senior": 3000, "manager": 5000, "director": 10000},
        "office_supplies": {"junior": 200, "senior": 300, "manager": 500, "director": 1000},
        "software": {"junior": 500, "senior": 1000, "manager": 2000, "director": 5000}
    }
    
    if category in category_limits:
        limit = category_limits[category].get(employee_level, 0)
        if amount > limit:
            violations.append(f"{category} expense exceeds limit for {employee_level} level (${limit})")
    
    # Receipt requirements
    if amount > 25 and not expense_data.get("receipt_url"):
        violations.append("Receipt required for expenses over $25")
    
    # Advance approval requirements
    if amount > 1000 and not expense_data.get("pre_approved"):
        violations.append("Expenses over $1000 require advance approval")
    
    return violations


def calculate_reimbursement_schedule(expenses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate reimbursement schedule for approved expenses."""
    approved_expenses = [exp for exp in expenses if exp.get("status") == "approved"]
    
    if not approved_expenses:
        return {"total_reimbursement": 0, "payment_date": None}
    
    total_amount = sum(exp.get("amount", 0) for exp in approved_expenses)
    
    # Next payment date (assuming bi-weekly payroll)
    today = date.today()
    days_until_friday = (4 - today.weekday()) % 7  # Friday is 4
    if days_until_friday == 0:  # Today is Friday
        days_until_friday = 7  # Next Friday
    
    next_payment_date = today + timedelta(days=days_until_friday)
    
    return {
        "total_reimbursement": round(total_amount, 2),
        "expense_count": len(approved_expenses),
        "payment_date": next_payment_date.isoformat(),
        "expenses": [
            {
                "expense_id": exp.get("expense_id"),
                "amount": exp.get("amount"),
                "description": exp.get("description")
            } for exp in approved_expenses
        ]
    }


def generate_financial_report_data(period_start: date, period_end: date, 
                                 expenses: List[Dict[str, Any]], 
                                 invoices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive financial report data."""
    # Expense analysis
    period_expenses = [
        exp for exp in expenses 
        if period_start <= datetime.fromisoformat(str(exp.get("expense_date", period_start))).date() <= period_end
    ]
    
    # Invoice analysis
    period_invoices = [
        inv for inv in invoices
        if period_start <= datetime.fromisoformat(str(inv.get("issue_date", period_start))).date() <= period_end
    ]
    
    expense_stats = calculate_expense_statistics(period_expenses)
    
    # Invoice statistics
    total_invoiced = sum(inv.get("amount", 0) for inv in period_invoices)
    paid_invoices = [inv for inv in period_invoices if inv.get("status") == "paid"]
    total_collected = sum(inv.get("amount", 0) for inv in paid_invoices)
    
    # Cash flow analysis
    cash_outflow = expense_stats.get("total_amount", 0)
    cash_inflow = total_collected
    net_cash_flow = cash_inflow - cash_outflow
    
    return {
        "period": {
            "start_date": period_start.isoformat(),
            "end_date": period_end.isoformat(),
            "days": (period_end - period_start).days + 1
        },
        "expenses": {
            "total_expenses": expense_stats.get("total_expenses", 0),
            "total_amount": expense_stats.get("total_amount", 0),
            "average_expense": expense_stats.get("average_amount", 0),
            "category_breakdown": expense_stats.get("category_breakdown", {}),
            "status_breakdown": expense_stats.get("status_breakdown", {})
        },
        "invoices": {
            "total_invoices": len(period_invoices),
            "total_invoiced": round(total_invoiced, 2),
            "total_collected": round(total_collected, 2),
            "collection_rate": round((total_collected / total_invoiced) * 100, 2) if total_invoiced > 0 else 0,
            "outstanding_amount": round(total_invoiced - total_collected, 2)
        },
        "cash_flow": {
            "inflow": round(cash_inflow, 2),
            "outflow": round(cash_outflow, 2),
            "net_flow": round(net_cash_flow, 2),
            "flow_status": "positive" if net_cash_flow > 0 else "negative"
        }
    }