from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date, datetime
from enum import Enum


class EmployeeStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    on_leave = "on_leave"
    terminated = "terminated"
    probation = "probation"


class LeaveType(str, Enum):
    annual = "annual"
    sick = "sick"
    maternity = "maternity"
    paternity = "paternity"
    unpaid = "unpaid"
    emergency = "emergency"
    bereavement = "bereavement"


class LeaveStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    cancelled = "cancelled"


class EmployeeCreate(BaseModel):
    employee_id: str
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    department: str
    designation: str
    employment_type: str = "full_time"
    hire_date: str
    salary: float = Field(..., gt=0)
    manager_id: Optional[int] = None
    skills: Optional[List[str]] = []
    address: Optional[str] = None


class EmployeeUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    designation: Optional[str] = None
    salary: Optional[float] = None
    manager_id: Optional[int] = None
    skills: Optional[List[str]] = None
    status: Optional[EmployeeStatus] = None
    address: Optional[str] = None


class LeaveRequestCreate(BaseModel):
    employee_id: int
    leave_type: LeaveType
    start_date: str
    end_date: str
    reason: str = Field(..., min_length=10)
    emergency_contact: Optional[str] = None


class AttendanceLog(BaseModel):
    employee_id: int
    date: str
    check_in: Optional[str] = None
    check_out: Optional[str] = None
    status: str = "present"
    notes: Optional[str] = None


class RecruitmentCreate(BaseModel):
    job_title: str
    department: str
    requirements: str
    salary_min: float
    salary_max: float
    location: str
    job_type: str = "full_time"
    openings: int = 1
    deadline: Optional[str] = None


class CandidateCreate(BaseModel):
    job_id: int
    full_name: str
    email: str
    phone: Optional[str] = None
    resume_url: Optional[str] = None
    experience_years: int = 0
    skills: Optional[List[str]] = []
    current_company: Optional[str] = None
    expected_salary: Optional[float] = None


class PayrollRun(BaseModel):
    period_month: int = Field(..., ge=1, le=12)
    period_year: int = Field(..., ge=2020)
    employee_ids: Optional[List[int]] = None
