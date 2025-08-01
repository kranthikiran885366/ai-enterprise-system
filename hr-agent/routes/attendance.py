"""Attendance management routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime, date
from enum import Enum

from shared_libs.auth import get_current_user


router = APIRouter()


class AttendanceType(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    LATE = "late"
    HALF_DAY = "half_day"
    WORK_FROM_HOME = "work_from_home"


class AttendanceRecord(BaseModel):
    employee_id: str
    date: date
    check_in: Optional[datetime] = None
    check_out: Optional[datetime] = None
    attendance_type: AttendanceType
    notes: Optional[str] = None


class LeaveRequest(BaseModel):
    employee_id: str
    leave_type: str
    start_date: date
    end_date: date
    reason: str
    status: str = "pending"


@router.post("/records", response_model=dict)
async def create_attendance_record(
    attendance_data: AttendanceRecord,
    current_user: dict = Depends(get_current_user)
):
    """Create an attendance record."""
    return {
        "message": "Attendance record created successfully",
        "employee_id": attendance_data.employee_id,
        "date": attendance_data.date.isoformat(),
        "type": attendance_data.attendance_type
    }


@router.get("/records", response_model=List[dict])
async def list_attendance_records(
    employee_id: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """List attendance records with filters."""
    # Mock data for demonstration
    records = [
        {
            "id": "ATT001",
            "employee_id": "EMP001",
            "date": "2024-01-15",
            "check_in": "09:00:00",
            "check_out": "17:30:00",
            "attendance_type": "present",
            "hours_worked": 8.5
        },
        {
            "id": "ATT002",
            "employee_id": "EMP002",
            "date": "2024-01-15",
            "check_in": "09:15:00",
            "check_out": "17:30:00",
            "attendance_type": "late",
            "hours_worked": 8.25
        }
    ]
    
    # Apply filters
    if employee_id:
        records = [rec for rec in records if rec["employee_id"] == employee_id]
    
    return records


@router.post("/leave-requests", response_model=dict)
async def submit_leave_request(
    leave_data: LeaveRequest,
    current_user: dict = Depends(get_current_user)
):
    """Submit a leave request."""
    return {
        "message": "Leave request submitted successfully",
        "request_id": "LEAVE001",
        "employee_id": leave_data.employee_id,
        "status": "pending"
    }


@router.get("/leave-requests", response_model=List[dict])
async def list_leave_requests(
    employee_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """List leave requests with filters."""
    # Mock data for demonstration
    requests = [
        {
            "id": "LEAVE001",
            "employee_id": "EMP001",
            "leave_type": "vacation",
            "start_date": "2024-02-01",
            "end_date": "2024-02-05",
            "status": "approved",
            "requested_at": datetime.utcnow().isoformat()
        },
        {
            "id": "LEAVE002",
            "employee_id": "EMP002",
            "leave_type": "sick",
            "start_date": "2024-01-20",
            "end_date": "2024-01-22",
            "status": "pending",
            "requested_at": datetime.utcnow().isoformat()
        }
    ]
    
    # Apply filters
    if employee_id:
        requests = [req for req in requests if req["employee_id"] == employee_id]
    if status:
        requests = [req for req in requests if req["status"] == status]
    
    return requests


attendance_router = router
