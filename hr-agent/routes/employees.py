"""Employee management routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from shared_libs.auth import get_current_user
from models.employee import EmployeeCreate, EmployeeUpdate, EmployeeResponse
from shared_libs.models import PaginatedResponse


router = APIRouter()


@router.post("/", response_model=EmployeeResponse)
async def create_employee(
    employee_data: EmployeeCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new employee."""
    from main import app
    hr_service = app.state.hr_service
    
    employee = await hr_service.create_employee(employee_data)
    
    if not employee:
        raise HTTPException(status_code=400, detail="Failed to create employee")
    
    return EmployeeResponse(
        id=str(employee.id),
        employee_id=employee.employee_id,
        first_name=employee.first_name,
        last_name=employee.last_name,
        email=employee.email,
        phone=employee.phone,
        department=employee.department,
        position=employee.position,
        manager_id=employee.manager_id,
        hire_date=employee.hire_date,
        salary=employee.salary,
        status=employee.status,
        address=employee.address,
        emergency_contact=employee.emergency_contact,
        skills=employee.skills,
        certifications=employee.certifications,
        created_at=employee.created_at,
        updated_at=employee.updated_at
    )


@router.get("/", response_model=PaginatedResponse)
async def list_employees(
    department: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """List employees with pagination and filters."""
    from main import app
    hr_service = app.state.hr_service
    
    skip = (page - 1) * limit
    employees = await hr_service.list_employees(department, status, skip, limit)
    total = await hr_service.get_employee_count(department, status)
    
    employee_responses = [
        EmployeeResponse(
            id=str(emp.id),
            employee_id=emp.employee_id,
            first_name=emp.first_name,
            last_name=emp.last_name,
            email=emp.email,
            phone=emp.phone,
            department=emp.department,
            position=emp.position,
            manager_id=emp.manager_id,
            hire_date=emp.hire_date,
            salary=emp.salary,
            status=emp.status,
            address=emp.address,
            emergency_contact=emp.emergency_contact,
            skills=emp.skills,
            certifications=emp.certifications,
            created_at=emp.created_at,
            updated_at=emp.updated_at
        ) for emp in employees
    ]
    
    return PaginatedResponse(
        items=employee_responses,
        total=total,
        page=page,
        limit=limit,
        pages=(total + limit - 1) // limit
    )


@router.get("/{employee_id}", response_model=EmployeeResponse)
async def get_employee(
    employee_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get an employee by ID."""
    from main import app
    hr_service = app.state.hr_service
    
    employee = await hr_service.get_employee(employee_id)
    
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    return EmployeeResponse(
        id=str(employee.id),
        employee_id=employee.employee_id,
        first_name=employee.first_name,
        last_name=employee.last_name,
        email=employee.email,
        phone=employee.phone,
        department=employee.department,
        position=employee.position,
        manager_id=employee.manager_id,
        hire_date=employee.hire_date,
        salary=employee.salary,
        status=employee.status,
        address=employee.address,
        emergency_contact=employee.emergency_contact,
        skills=employee.skills,
        certifications=employee.certifications,
        created_at=employee.created_at,
        updated_at=employee.updated_at
    )


@router.put("/{employee_id}", response_model=EmployeeResponse)
async def update_employee(
    employee_id: str,
    update_data: EmployeeUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update an employee."""
    from main import app
    hr_service = app.state.hr_service
    
    employee = await hr_service.update_employee(employee_id, update_data)
    
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found or update failed")
    
    return EmployeeResponse(
        id=str(employee.id),
        employee_id=employee.employee_id,
        first_name=employee.first_name,
        last_name=employee.last_name,
        email=employee.email,
        phone=employee.phone,
        department=employee.department,
        position=employee.position,
        manager_id=employee.manager_id,
        hire_date=employee.hire_date,
        salary=employee.salary,
        status=employee.status,
        address=employee.address,
        emergency_contact=employee.emergency_contact,
        skills=employee.skills,
        certifications=employee.certifications,
        created_at=employee.created_at,
        updated_at=employee.updated_at
    )


@router.delete("/{employee_id}")
async def delete_employee(
    employee_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete an employee."""
    from main import app
    hr_service = app.state.hr_service
    
    success = await hr_service.delete_employee(employee_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    return {"message": "Employee deleted successfully"}


@router.get("/search/{search_term}", response_model=List[EmployeeResponse])
async def search_employees(
    search_term: str,
    current_user: dict = Depends(get_current_user)
):
    """Search employees by name, email, or employee ID."""
    from main import app
    hr_service = app.state.hr_service
    
    employees = await hr_service.search_employees(search_term)
    
    return [
        EmployeeResponse(
            id=str(emp.id),
            employee_id=emp.employee_id,
            first_name=emp.first_name,
            last_name=emp.last_name,
            email=emp.email,
            phone=emp.phone,
            department=emp.department,
            position=emp.position,
            manager_id=emp.manager_id,
            hire_date=emp.hire_date,
            salary=emp.salary,
            status=emp.status,
            address=emp.address,
            emergency_contact=emp.emergency_contact,
            skills=emp.skills,
            certifications=emp.certifications,
            created_at=emp.created_at,
            updated_at=emp.updated_at
        ) for emp in employees
    ]


employees_router = router
