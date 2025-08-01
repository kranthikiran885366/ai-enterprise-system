"""Recruitment management routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

from shared_libs.auth import get_current_user
from shared_libs.models import BaseDocument
from controllers.recruitment_controller import RecruitmentController


router = APIRouter()


class JobStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    ON_HOLD = "on_hold"


class ApplicationStatus(str, Enum):
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEW = "interview"
    OFFER = "offer"
    HIRED = "hired"
    REJECTED = "rejected"


class JobPosting(BaseModel):
    title: str
    department: str
    description: str
    requirements: List[str]
    salary_range: Optional[str] = None
    location: str
    employment_type: str = "full-time"
    status: JobStatus = JobStatus.OPEN


class JobApplication(BaseModel):
    job_id: str
    candidate_name: str
    candidate_email: str
    candidate_phone: Optional[str] = None
    resume_url: Optional[str] = None
    cover_letter: Optional[str] = None
    status: ApplicationStatus = ApplicationStatus.APPLIED


@router.post("/jobs", response_model=dict)
async def create_job_posting(
    job_data: JobPosting,
    current_user: dict = Depends(get_current_user)
):
    """Create a new job posting."""
    from main import app
    controller = RecruitmentController(app.state.ai_recruitment_service)
    
    return await controller.create_job_posting(job_data.dict(), current_user)


@router.get("/jobs", response_model=List[dict])
async def list_job_postings(
    status: Optional[JobStatus] = Query(None),
    department: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """List job postings with filters."""
    # Mock data for demonstration
    jobs = [
        {
            "id": "JOB001",
            "title": "Senior Software Engineer",
            "department": "engineering",
            "status": "open",
            "location": "Remote",
            "created_at": datetime.utcnow().isoformat()
        },
        {
            "id": "JOB002",
            "title": "Marketing Manager",
            "department": "marketing",
            "status": "open",
            "location": "New York",
            "created_at": datetime.utcnow().isoformat()
        }
    ]
    
    # Apply filters
    if status:
        jobs = [job for job in jobs if job["status"] == status]
    if department:
        jobs = [job for job in jobs if job["department"] == department]
    
    return jobs


@router.post("/applications", response_model=dict)
async def submit_application(
    application_data: JobApplication,
    current_user: dict = Depends(get_current_user)
):
    """Submit a job application."""
    from main import app
    controller = RecruitmentController(app.state.ai_recruitment_service)
    
    return await controller.submit_application(application_data.dict(), current_user)


@router.get("/applications", response_model=List[dict])
async def list_applications(
    job_id: Optional[str] = Query(None),
    status: Optional[ApplicationStatus] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """List job applications with filters."""
    # Mock data for demonstration
    applications = [
        {
            "id": "APP001",
            "job_id": "JOB001",
            "candidate_name": "John Doe",
            "candidate_email": "john@example.com",
            "status": "screening",
            "applied_at": datetime.utcnow().isoformat()
        },
        {
            "id": "APP002",
            "job_id": "JOB002",
            "candidate_name": "Jane Smith",
            "candidate_email": "jane@example.com",
            "status": "interview",
            "applied_at": datetime.utcnow().isoformat()
        }
    ]
    
    # Apply filters
    if job_id:
        applications = [app for app in applications if app["job_id"] == job_id]
    if status:
        applications = [app for app in applications if app["status"] == status]
    
    return applications


@router.post("/ai-interview", response_model=dict)
async def conduct_ai_interview(
    candidate_id: str,
    interview_type: str = "technical",
    current_user: dict = Depends(get_current_user)
):
    """Conduct AI-powered interview."""
    from main import app
    controller = RecruitmentController(app.state.ai_recruitment_service)
    
    return await controller.conduct_ai_interview(candidate_id, interview_type, current_user)


@router.get("/analytics", response_model=dict)
async def get_recruitment_analytics(
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(get_current_user)
):
    """Get recruitment analytics."""
    from main import app
    controller = RecruitmentController(app.state.ai_recruitment_service)
    
    return await controller.get_recruitment_analytics(days, current_user)


recruitment_router = router
