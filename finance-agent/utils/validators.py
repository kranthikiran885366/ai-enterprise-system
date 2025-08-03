"""Validation utilities for Finance Agent."""

import re
from typing import Dict, Any, List
from datetime import datetime, date
from decimal import Decimal, InvalidOperation


class ValidationResult:
    """Validation result container."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []


async def validate_expense_data(data: Dict[str, Any], is_update: bool = False) -> ValidationResult:
    """Validate expense data."""
    errors = []
    
    # Required fields for creation
    if not is_update:
        required_fields = ["employee_id", "amount", "category", "description", "expense_date"]
        for field in required_fields:
            if not data.get(field):
                errors.append(f"{field} is required")
    
    # Amount validation
    if "amount" in data:
        try:
            amount = float(data["amount"])
            if amount <= 0:
                errors.append("Amount must be greater than zero")
            if amount > 50000:  # Reasonable upper limit
                errors.append("Amount exceeds maximum allowed ($50,000)")
        except (ValueError, TypeError):
            errors.append("Amount must be a valid number")
    
    # Category validation
    valid_categories = ["travel", "office_supplies", "marketing", "utilities", "software", "equipment", "meals", "other"]
    if "category" in data:
        if data["category"] not in valid_categories:
            errors.append(f"Category must be one of: {', '.join(valid_categories)}")
    
    # Description validation
    if "description" in data:
        description = data["description"]
        if len(description.strip()) < 5:
            errors.append("Description must be at least 5 characters")
        if len(description) > 500:
            errors.append("Description is too long (max 500 characters)")
    
    # Date validation
    if "expense_date" in data:
        expense_date = data["expense_date"]
        if isinstance(expense_date, str):
            try:
                expense_date = datetime.fromisoformat(expense_date).date()
            except ValueError:
                errors.append("Invalid expense date format")
        
        if isinstance(expense_date, date):
            if expense_date > date.today():
                errors.append("Expense date cannot be in the future")
            if expense_date < date.today().replace(year=date.today().year - 1):
                errors.append("Expense date cannot be more than 1 year old")
    
    return ValidationResult(len(errors) == 0, errors)


async def validate_invoice_data(data: Dict[str, Any], is_update: bool = False) -> ValidationResult:
    """Validate invoice data."""
    errors = []
    
    # Required fields for creation
    if not is_update:
        required_fields = ["client_name", "client_email", "amount", "description", "issue_date", "due_date"]
        for field in required_fields:
            if not data.get(field):
                errors.append(f"{field} is required")
    
    # Client email validation
    if "client_email" in data:
        email = data["client_email"]
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
            errors.append("Invalid client email format")
    
    # Amount validation
    if "amount" in data:
        try:
            amount = float(data["amount"])
            if amount <= 0:
                errors.append("Invoice amount must be greater than zero")
            if amount > 1000000:  # $1M limit
                errors.append("Invoice amount exceeds maximum allowed")
        except (ValueError, TypeError):
            errors.append("Amount must be a valid number")
    
    return ValidationResult(len(errors) == 0, errors)