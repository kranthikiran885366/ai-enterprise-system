"""Employee models for HR Agent."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr
from enum import Enum

from shared_libs.models import BaseDocument


class EmployeeStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TERMINATED = "terminated"
    ON_LEAVE = "on_leave"


class Department(str, Enum):
    ENGINEERING = "engineering"
    SALES = "sales"
    MARKETING = "marketing"
    HR = "hr"
    FINANCE = "finance"
    LEGAL = "legal"
    ADMIN = "admin"
    IT = "it"
    SUPPORT = "support"


class Employee(BaseDocument):
    """Employee model."""
    employee_id: str
    first_name: str
    last_name: str
    email: EmailStr
    phone: Optional[str] = None
    department: Department
    position: str
    manager_id: Optional[str] = None
    hire_date: datetime
    salary: Optional[float] = None
    status: EmployeeStatus = EmployeeStatus.ACTIVE
    address: Optional[str] = None
    emergency_contact: Optional[dict] = None
    skills: List[str] = []
    certifications: List[str] = []


class EmployeeCreate(BaseModel):
    """Employee creation model."""
    first_name: str
    last_name: str
    email: EmailStr
    phone: Optional[str] = None
    department: Department
    position: str
    manager_id: Optional[str] = None
    hire_date: datetime
    salary: Optional[float] = None
    address: Optional[str] = None
    emergency_contact: Optional[dict] = None
    skills: List[str] = []
    certifications: List[str] = []


class EmployeeUpdate(BaseModel):
    """Employee update model."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    department: Optional[Department] = None
    position: Optional[str] = None
    manager_id: Optional[str] = None
    salary: Optional[float] = None
    status: Optional[EmployeeStatus] = None
    address: Optional[str] = None
    emergency_contact: Optional[dict] = None
    skills: Optional[List[str]] = None
    certifications: Optional[List[str]] = None


class EmployeeResponse(BaseModel):
    """Employee response model."""
    id: str
    employee_id: str
    first_name: str
    last_name: str
    email: str
    phone: Optional[str]
    department: str
    position: str
    manager_id: Optional[str]
    hire_date: datetime
    salary: Optional[float]
    status: str
    address: Optional[str]
    emergency_contact: Optional[dict]
    skills: List[str]
    certifications: List[str]
    created_at: datetime
    updated_at: datetime
