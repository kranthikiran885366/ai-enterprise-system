"""Validation utilities for HR Agent."""

import re
from typing import Dict, Any, List
from datetime import datetime, date
from pydantic import BaseModel, ValidationError
from email_validator import validate_email, EmailNotValidError


class ValidationResult:
    """Validation result container."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []


async def validate_employee_data(data: Dict[str, Any], is_update: bool = False) -> ValidationResult:
    """Validate employee data."""
    errors = []
    
    # Required fields for creation
    if not is_update:
        required_fields = ["first_name", "last_name", "email", "department", "position"]
        for field in required_fields:
            if not data.get(field):
                errors.append(f"{field} is required")
    
    # Email validation
    if "email" in data:
        try:
            validate_email(data["email"])
        except EmailNotValidError:
            errors.append("Invalid email format")
    
    # Name validation
    for name_field in ["first_name", "last_name"]:
        if name_field in data:
            name = data[name_field]
            if not isinstance(name, str) or len(name.strip()) < 2:
                errors.append(f"{name_field} must be at least 2 characters")
            if not re.match(r"^[a-zA-Z\s'-]+$", name):
                errors.append(f"{name_field} contains invalid characters")
    
    # Phone validation
    if "phone" in data and data["phone"]:
        phone = data["phone"]
        if not re.match(r"^\+?[\d\s\-\(\)]+$", phone):
            errors.append("Invalid phone number format")
    
    # Department validation
    valid_departments = ["engineering", "sales", "marketing", "hr", "finance", "legal", "admin", "it", "support"]
    if "department" in data:
        if data["department"] not in valid_departments:
            errors.append(f"Department must be one of: {', '.join(valid_departments)}")
    
    # Salary validation
    if "salary" in data and data["salary"] is not None:
        try:
            salary = float(data["salary"])
            if salary < 0:
                errors.append("Salary cannot be negative")
            if salary > 1000000:  # Reasonable upper limit
                errors.append("Salary exceeds maximum allowed")
        except (ValueError, TypeError):
            errors.append("Salary must be a valid number")
    
    # Hire date validation
    if "hire_date" in data:
        hire_date = data["hire_date"]
        if isinstance(hire_date, str):
            try:
                hire_date = datetime.fromisoformat(hire_date)
            except ValueError:
                errors.append("Invalid hire date format")
        
        if isinstance(hire_date, datetime):
            if hire_date.date() > date.today():
                errors.append("Hire date cannot be in the future")
            if hire_date.year < 1900:
                errors.append("Hire date is too old")
    
    # Skills validation
    if "skills" in data and data["skills"]:
        skills = data["skills"]
        if not isinstance(skills, list):
            errors.append("Skills must be a list")
        else:
            for skill in skills:
                if not isinstance(skill, str) or len(skill.strip()) < 2:
                    errors.append("Each skill must be at least 2 characters")
    
    # Emergency contact validation
    if "emergency_contact" in data and data["emergency_contact"]:
        contact = data["emergency_contact"]
        if not isinstance(contact, dict):
            errors.append("Emergency contact must be an object")
        else:
            if "name" in contact and (not contact["name"] or len(contact["name"].strip()) < 2):
                errors.append("Emergency contact name is required")
            if "phone" in contact and contact["phone"]:
                if not re.match(r"^\+?[\d\s\-\(\)]+$", contact["phone"]):
                    errors.append("Invalid emergency contact phone format")
    
    return ValidationResult(len(errors) == 0, errors)


async def validate_job_posting(data: Dict[str, Any]) -> ValidationResult:
    """Validate job posting data."""
    errors = []
    
    # Required fields
    required_fields = ["title", "department", "description", "location"]
    for field in required_fields:
        if not data.get(field):
            errors.append(f"{field} is required")
    
    # Title validation
    if "title" in data:
        title = data["title"]
        if len(title.strip()) < 5:
            errors.append("Job title must be at least 5 characters")
        if len(title) > 100:
            errors.append("Job title is too long (max 100 characters)")
    
    # Description validation
    if "description" in data:
        description = data["description"]
        if len(description.strip()) < 50:
            errors.append("Job description must be at least 50 characters")
        if len(description) > 5000:
            errors.append("Job description is too long (max 5000 characters)")
    
    # Requirements validation
    if "requirements" in data and data["requirements"]:
        requirements = data["requirements"]
        if not isinstance(requirements, list):
            errors.append("Requirements must be a list")
        else:
            for req in requirements:
                if not isinstance(req, str) or len(req.strip()) < 3:
                    errors.append("Each requirement must be at least 3 characters")
    
    # Salary range validation
    if "salary_range" in data and data["salary_range"]:
        salary_range = data["salary_range"]
        if not re.match(r"^\$?\d+k?\s*-\s*\$?\d+k?$", salary_range.replace(",", "")):
            errors.append("Invalid salary range format (e.g., '$50k - $70k')")
    
    # Employment type validation
    valid_types = ["full-time", "part-time", "contract", "internship", "temporary"]
    if "employment_type" in data:
        if data["employment_type"] not in valid_types:
            errors.append(f"Employment type must be one of: {', '.join(valid_types)}")
    
    return ValidationResult(len(errors) == 0, errors)


async def validate_application(data: Dict[str, Any]) -> ValidationResult:
    """Validate job application data."""
    errors = []
    
    # Required fields
    required_fields = ["candidate_name", "candidate_email", "job_id"]
    for field in required_fields:
        if not data.get(field):
            errors.append(f"{field} is required")
    
    # Email validation
    if "candidate_email" in data:
        try:
            validate_email(data["candidate_email"])
        except EmailNotValidError:
            errors.append("Invalid candidate email format")
    
    # Name validation
    if "candidate_name" in data:
        name = data["candidate_name"]
        if len(name.strip()) < 2:
            errors.append("Candidate name must be at least 2 characters")
        if not re.match(r"^[a-zA-Z\s'-]+$", name):
            errors.append("Candidate name contains invalid characters")
    
    # Phone validation
    if "candidate_phone" in data and data["candidate_phone"]:
        phone = data["candidate_phone"]
        if not re.match(r"^\+?[\d\s\-\(\)]+$", phone):
            errors.append("Invalid candidate phone format")
    
    # Resume validation
    if "resume_url" in data and data["resume_url"]:
        resume_url = data["resume_url"]
        if not re.match(r"^https?://", resume_url):
            errors.append("Resume URL must be a valid HTTP/HTTPS URL")
    
    # Cover letter validation
    if "cover_letter" in data and data["cover_letter"]:
        cover_letter = data["cover_letter"]
        if len(cover_letter) > 2000:
            errors.append("Cover letter is too long (max 2000 characters)")
    
    return ValidationResult(len(errors) == 0, errors)


async def validate_attendance_record(data: Dict[str, Any]) -> ValidationResult:
    """Validate attendance record data."""
    errors = []
    
    # Required fields
    required_fields = ["employee_id", "date", "attendance_type"]
    for field in required_fields:
        if not data.get(field):
            errors.append(f"{field} is required")
    
    # Date validation
    if "date" in data:
        record_date = data["date"]
        if isinstance(record_date, str):
            try:
                record_date = datetime.fromisoformat(record_date).date()
            except ValueError:
                errors.append("Invalid date format")
        
        if isinstance(record_date, date):
            if record_date > date.today():
                errors.append("Attendance date cannot be in the future")
            if record_date < date.today().replace(year=date.today().year - 1):
                errors.append("Attendance date is too old")
    
    # Attendance type validation
    valid_types = ["present", "absent", "late", "half_day", "work_from_home"]
    if "attendance_type" in data:
        if data["attendance_type"] not in valid_types:
            errors.append(f"Attendance type must be one of: {', '.join(valid_types)}")
    
    # Time validation
    if "check_in" in data and data["check_in"]:
        try:
            datetime.fromisoformat(str(data["check_in"]))
        except ValueError:
            errors.append("Invalid check-in time format")
    
    if "check_out" in data and data["check_out"]:
        try:
            datetime.fromisoformat(str(data["check_out"]))
        except ValueError:
            errors.append("Invalid check-out time format")
    
    # Validate check-in/check-out logic
    if ("check_in" in data and data["check_in"] and 
        "check_out" in data and data["check_out"]):
        try:
            check_in = datetime.fromisoformat(str(data["check_in"]))
            check_out = datetime.fromisoformat(str(data["check_out"]))
            
            if check_out <= check_in:
                errors.append("Check-out time must be after check-in time")
            
            # Reasonable work hours check
            work_duration = (check_out - check_in).total_seconds() / 3600
            if work_duration > 16:  # More than 16 hours
                errors.append("Work duration exceeds reasonable limits")
                
        except ValueError:
            pass  # Already handled above
    
    return ValidationResult(len(errors) == 0, errors)


async def validate_leave_request(data: Dict[str, Any]) -> ValidationResult:
    """Validate leave request data."""
    errors = []
    
    # Required fields
    required_fields = ["employee_id", "leave_type", "start_date", "end_date", "reason"]
    for field in required_fields:
        if not data.get(field):
            errors.append(f"{field} is required")
    
    # Leave type validation
    valid_leave_types = ["vacation", "sick", "personal", "maternity", "paternity", "bereavement", "unpaid"]
    if "leave_type" in data:
        if data["leave_type"] not in valid_leave_types:
            errors.append(f"Leave type must be one of: {', '.join(valid_leave_types)}")
    
    # Date validation
    start_date = None
    end_date = None
    
    if "start_date" in data:
        try:
            start_date = datetime.fromisoformat(str(data["start_date"])).date()
        except ValueError:
            errors.append("Invalid start date format")
    
    if "end_date" in data:
        try:
            end_date = datetime.fromisoformat(str(data["end_date"])).date()
        except ValueError:
            errors.append("Invalid end date format")
    
    # Date logic validation
    if start_date and end_date:
        if end_date < start_date:
            errors.append("End date must be after start date")
        
        # Check for reasonable leave duration
        leave_duration = (end_date - start_date).days + 1
        if leave_duration > 365:  # More than a year
            errors.append("Leave duration exceeds maximum allowed")
        
        # Check advance notice (except for sick leave)
        if data.get("leave_type") != "sick":
            if start_date <= date.today():
                errors.append("Leave requests must be submitted in advance (except sick leave)")
    
    # Reason validation
    if "reason" in data:
        reason = data["reason"]
        if len(reason.strip()) < 10:
            errors.append("Leave reason must be at least 10 characters")
        if len(reason) > 500:
            errors.append("Leave reason is too long (max 500 characters)")
    
    return ValidationResult(len(errors) == 0, errors)