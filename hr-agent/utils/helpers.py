"""Helper utilities for HR Agent."""

import hashlib
import secrets
import string
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
import uuid


def generate_employee_id(first_name: str, last_name: str, hire_date: datetime) -> str:
    """Generate unique employee ID."""
    # Create base from name and date
    base = f"{first_name[:2]}{last_name[:2]}{hire_date.strftime('%y%m')}"
    
    # Add random suffix for uniqueness
    suffix = ''.join(secrets.choice(string.digits) for _ in range(3))
    
    return f"EMP{base.upper()}{suffix}"


def generate_secure_password(length: int = 12) -> str:
    """Generate secure temporary password."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    return password


def calculate_tenure(hire_date: datetime) -> Dict[str, int]:
    """Calculate employee tenure."""
    today = datetime.now()
    delta = today - hire_date
    
    years = delta.days // 365
    months = (delta.days % 365) // 30
    days = (delta.days % 365) % 30
    
    return {
        "years": years,
        "months": months,
        "days": days,
        "total_days": delta.days
    }


def calculate_age(birth_date: date) -> int:
    """Calculate age from birth date."""
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))


def format_salary(salary: float, currency: str = "USD") -> str:
    """Format salary for display."""
    if currency == "USD":
        return f"${salary:,.2f}"
    elif currency == "EUR":
        return f"€{salary:,.2f}"
    elif currency == "GBP":
        return f"£{salary:,.2f}"
    else:
        return f"{currency} {salary:,.2f}"


def calculate_working_days(start_date: date, end_date: date, exclude_weekends: bool = True) -> int:
    """Calculate working days between two dates."""
    if start_date > end_date:
        return 0
    
    total_days = (end_date - start_date).days + 1
    
    if not exclude_weekends:
        return total_days
    
    # Count weekdays only
    working_days = 0
    current_date = start_date
    
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
            working_days += 1
        current_date += timedelta(days=1)
    
    return working_days


def get_next_performance_review_date(hire_date: datetime, review_frequency_months: int = 12) -> datetime:
    """Calculate next performance review date."""
    today = datetime.now()
    
    # Calculate months since hire
    months_since_hire = (today.year - hire_date.year) * 12 + (today.month - hire_date.month)
    
    # Calculate next review
    next_review_months = ((months_since_hire // review_frequency_months) + 1) * review_frequency_months
    
    next_review_date = hire_date.replace(
        year=hire_date.year + (next_review_months // 12),
        month=hire_date.month + (next_review_months % 12)
    )
    
    # Handle month overflow
    if next_review_date.month > 12:
        next_review_date = next_review_date.replace(
            year=next_review_date.year + 1,
            month=next_review_date.month - 12
        )
    
    return next_review_date


def anonymize_employee_data(employee_data: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymize employee data for analytics."""
    anonymized = employee_data.copy()
    
    # Remove or hash PII
    pii_fields = ["first_name", "last_name", "email", "phone", "address"]
    
    for field in pii_fields:
        if field in anonymized:
            if field == "email":
                # Keep domain for analytics
                email = anonymized[field]
                domain = email.split("@")[1] if "@" in email else "unknown"
                anonymized[field] = f"user_{hash(email) % 10000}@{domain}"
            else:
                # Hash other PII
                anonymized[field] = hashlib.sha256(str(anonymized[field]).encode()).hexdigest()[:8]
    
    return anonymized


def calculate_leave_balance(employee_id: str, leave_type: str, accrual_rate: float, 
                          hire_date: datetime, used_days: float = 0) -> Dict[str, float]:
    """Calculate leave balance for employee."""
    today = datetime.now()
    
    # Calculate months worked
    months_worked = (today.year - hire_date.year) * 12 + (today.month - hire_date.month)
    
    # Calculate accrued leave
    accrued_days = months_worked * accrual_rate
    
    # Calculate available balance
    available_days = max(0, accrued_days - used_days)
    
    return {
        "accrued_days": round(accrued_days, 2),
        "used_days": round(used_days, 2),
        "available_days": round(available_days, 2),
        "accrual_rate_monthly": accrual_rate
    }


def generate_employee_report_data(employee: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive employee report data."""
    hire_date = employee.get("hire_date")
    if isinstance(hire_date, str):
        hire_date = datetime.fromisoformat(hire_date)
    
    tenure = calculate_tenure(hire_date)
    next_review = get_next_performance_review_date(hire_date)
    
    # Calculate leave balances (example rates)
    leave_balances = {
        "vacation": calculate_leave_balance(
            employee.get("employee_id", ""),
            "vacation",
            2.0,  # 2 days per month
            hire_date
        ),
        "sick": calculate_leave_balance(
            employee.get("employee_id", ""),
            "sick",
            1.0,  # 1 day per month
            hire_date
        )
    }
    
    return {
        "employee_info": {
            "employee_id": employee.get("employee_id"),
            "name": f"{employee.get('first_name', '')} {employee.get('last_name', '')}",
            "department": employee.get("department"),
            "position": employee.get("position"),
            "status": employee.get("status")
        },
        "employment_details": {
            "hire_date": hire_date.isoformat(),
            "tenure": tenure,
            "next_performance_review": next_review.isoformat(),
            "manager_id": employee.get("manager_id")
        },
        "compensation": {
            "salary": employee.get("salary"),
            "formatted_salary": format_salary(employee.get("salary", 0)) if employee.get("salary") else "N/A"
        },
        "leave_balances": leave_balances,
        "skills": employee.get("skills", []),
        "certifications": employee.get("certifications", [])
    }


def validate_business_rules(employee_data: Dict[str, Any], operation: str) -> List[str]:
    """Validate business rules for employee operations."""
    violations = []
    
    if operation == "create":
        # New employee business rules
        hire_date = employee_data.get("hire_date")
        if isinstance(hire_date, str):
            hire_date = datetime.fromisoformat(hire_date)
        
        # Cannot hire for past dates (except within 30 days)
        if hire_date and hire_date.date() < (date.today() - timedelta(days=30)):
            violations.append("Cannot create employee with hire date more than 30 days in the past")
        
        # Salary range validation by department
        salary = employee_data.get("salary", 0)
        department = employee_data.get("department", "")
        
        salary_ranges = {
            "engineering": (60000, 200000),
            "sales": (40000, 150000),
            "marketing": (45000, 120000),
            "hr": (50000, 100000),
            "finance": (55000, 130000)
        }
        
        if department in salary_ranges and salary:
            min_salary, max_salary = salary_ranges[department]
            if salary < min_salary or salary > max_salary:
                violations.append(f"Salary for {department} must be between ${min_salary:,} and ${max_salary:,}")
    
    elif operation == "update":
        # Update business rules
        if "salary" in employee_data:
            # Salary increase limit (50% max increase)
            # This would require current salary comparison in real implementation
            pass
    
    elif operation == "terminate":
        # Termination business rules
        # Check for pending approvals, active projects, etc.
        pass
    
    return violations


def generate_onboarding_checklist(employee_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate onboarding checklist for new employee."""
    department = employee_data.get("department", "")
    position = employee_data.get("position", "")
    
    base_checklist = [
        {
            "task": "Send welcome email",
            "responsible": "HR",
            "due_days": 0,
            "category": "communication"
        },
        {
            "task": "Create IT accounts",
            "responsible": "IT",
            "due_days": 1,
            "category": "setup"
        },
        {
            "task": "Prepare workspace",
            "responsible": "Facilities",
            "due_days": 1,
            "category": "setup"
        },
        {
            "task": "Schedule orientation",
            "responsible": "HR",
            "due_days": 2,
            "category": "training"
        },
        {
            "task": "Assign buddy/mentor",
            "responsible": "Manager",
            "due_days": 3,
            "category": "support"
        },
        {
            "task": "Department introduction",
            "responsible": "Manager",
            "due_days": 5,
            "category": "integration"
        },
        {
            "task": "First week check-in",
            "responsible": "HR",
            "due_days": 7,
            "category": "feedback"
        },
        {
            "task": "30-day review",
            "responsible": "Manager",
            "due_days": 30,
            "category": "evaluation"
        }
    ]
    
    # Add department-specific tasks
    if department == "engineering":
        base_checklist.extend([
            {
                "task": "Setup development environment",
                "responsible": "IT",
                "due_days": 2,
                "category": "setup"
            },
            {
                "task": "Code repository access",
                "responsible": "Engineering Manager",
                "due_days": 3,
                "category": "access"
            }
        ])
    elif department == "sales":
        base_checklist.extend([
            {
                "task": "CRM system training",
                "responsible": "Sales Manager",
                "due_days": 5,
                "category": "training"
            },
            {
                "task": "Product knowledge session",
                "responsible": "Product Team",
                "due_days": 7,
                "category": "training"
            }
        ])
    
    return sorted(base_checklist, key=lambda x: x["due_days"])


def calculate_employee_metrics(employees: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate various employee metrics."""
    if not employees:
        return {}
    
    total_employees = len(employees)
    
    # Department distribution
    dept_distribution = {}
    for emp in employees:
        dept = emp.get("department", "unknown")
        dept_distribution[dept] = dept_distribution.get(dept, 0) + 1
    
    # Status distribution
    status_distribution = {}
    for emp in employees:
        status = emp.get("status", "unknown")
        status_distribution[status] = status_distribution.get(status, 0) + 1
    
    # Average tenure
    total_tenure_days = 0
    tenure_count = 0
    
    for emp in employees:
        hire_date = emp.get("hire_date")
        if hire_date:
            if isinstance(hire_date, str):
                hire_date = datetime.fromisoformat(hire_date)
            tenure = calculate_tenure(hire_date)
            total_tenure_days += tenure["total_days"]
            tenure_count += 1
    
    avg_tenure_days = total_tenure_days / tenure_count if tenure_count > 0 else 0
    
    # Salary statistics
    salaries = [emp.get("salary", 0) for emp in employees if emp.get("salary")]
    avg_salary = sum(salaries) / len(salaries) if salaries else 0
    min_salary = min(salaries) if salaries else 0
    max_salary = max(salaries) if salaries else 0
    
    return {
        "total_employees": total_employees,
        "department_distribution": dept_distribution,
        "status_distribution": status_distribution,
        "average_tenure_days": round(avg_tenure_days),
        "average_tenure_years": round(avg_tenure_days / 365, 1),
        "salary_statistics": {
            "average": round(avg_salary, 2),
            "minimum": min_salary,
            "maximum": max_salary,
            "count": len(salaries)
        }
    }