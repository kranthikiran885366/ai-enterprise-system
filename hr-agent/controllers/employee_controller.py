"""Employee controller with business logic."""

from typing import List, Optional
from fastapi import HTTPException, status
from loguru import logger

from services.hr_service import HRService
from services.ai_recruitment import AIRecruitmentService
from models.employee import Employee, EmployeeCreate, EmployeeUpdate
from utils.validators import validate_employee_data
from utils.notifications import send_employee_notification


class EmployeeController:
    """Controller for employee operations."""
    
    def __init__(self, hr_service: HRService, ai_recruitment: AIRecruitmentService):
        self.hr_service = hr_service
        self.ai_recruitment = ai_recruitment
    
    async def create_employee(self, employee_data: EmployeeCreate, current_user: dict) -> Employee:
        """Create a new employee with validation and notifications."""
        try:
            # Validate employee data
            validation_result = await validate_employee_data(employee_data.dict())
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Validation failed: {validation_result.errors}"
                )
            
            # Check for duplicate email
            existing_employee = await self.hr_service.get_employee_by_email(employee_data.email)
            if existing_employee:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Employee with this email already exists"
                )
            
            # Create employee
            employee = await self.hr_service.create_employee(employee_data)
            if not employee:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create employee"
                )
            
            # Send welcome notification
            await send_employee_notification(
                employee.email,
                "welcome",
                {
                    "name": f"{employee.first_name} {employee.last_name}",
                    "employee_id": employee.employee_id,
                    "start_date": employee.hire_date.isoformat()
                }
            )
            
            # Create AI-powered onboarding lifecycle bot
            await self.ai_recruitment.create_employee_lifecycle_bot(
                employee.employee_id,
                "onboarding"
            )
            
            # Track mood and performance baseline
            await self.ai_recruitment.track_employee_mood_performance(employee.employee_id)
            
            logger.info(f"Employee created successfully: {employee.employee_id}")
            return employee
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to create employee: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def get_employee(self, employee_id: str, current_user: dict) -> Employee:
        """Get employee by ID with access control."""
        try:
            employee = await self.hr_service.get_employee(employee_id)
            if not employee:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Employee not found"
                )
            
            # Check access permissions
            if not self._can_access_employee(current_user, employee):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
            
            return employee
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get employee {employee_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def update_employee(self, employee_id: str, update_data: EmployeeUpdate, 
                            current_user: dict) -> Employee:
        """Update employee with validation and change tracking."""
        try:
            # Get existing employee
            existing_employee = await self.hr_service.get_employee(employee_id)
            if not existing_employee:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Employee not found"
                )
            
            # Check permissions
            if not self._can_modify_employee(current_user, existing_employee):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
            
            # Validate update data
            update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
            if update_dict:
                validation_result = await validate_employee_data(update_dict, is_update=True)
                if not validation_result.is_valid:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Validation failed: {validation_result.errors}"
                    )
            
            # Track changes for audit
            changes = self._track_employee_changes(existing_employee, update_data)
            
            # Update employee
            updated_employee = await self.hr_service.update_employee(employee_id, update_data)
            if not updated_employee:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update employee"
                )
            
            # Log changes
            if changes:
                logger.info(f"Employee {employee_id} updated: {changes}")
                
                # Send notification for significant changes
                if any(field in changes for field in ['department', 'position', 'manager_id']):
                    await send_employee_notification(
                        updated_employee.email,
                        "profile_update",
                        {
                            "name": f"{updated_employee.first_name} {updated_employee.last_name}",
                            "changes": changes
                        }
                    )
            
            return updated_employee
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update employee {employee_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def list_employees(self, department: Optional[str], status: Optional[str],
                           skip: int, limit: int, current_user: dict) -> List[Employee]:
        """List employees with filtering and access control."""
        try:
            # Apply department filter based on user permissions
            if not self._is_hr_admin(current_user) and department is None:
                # Non-admin users can only see their own department
                department = current_user.get("department")
            
            employees = await self.hr_service.list_employees(department, status, skip, limit)
            
            # Filter based on access permissions
            accessible_employees = []
            for employee in employees:
                if self._can_access_employee(current_user, employee):
                    accessible_employees.append(employee)
            
            return accessible_employees
            
        except Exception as e:
            logger.error(f"Failed to list employees: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def delete_employee(self, employee_id: str, current_user: dict) -> bool:
        """Soft delete employee with proper workflow."""
        try:
            # Get employee
            employee = await self.hr_service.get_employee(employee_id)
            if not employee:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Employee not found"
                )
            
            # Check permissions (only HR admin can delete)
            if not self._is_hr_admin(current_user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only HR administrators can delete employees"
                )
            
            # Soft delete (mark as terminated)
            update_data = EmployeeUpdate(status="terminated")
            updated_employee = await self.hr_service.update_employee(employee_id, update_data)
            
            if updated_employee:
                # Create offboarding lifecycle bot
                await self.ai_recruitment.create_employee_lifecycle_bot(
                    employee_id,
                    "offboarding"
                )
                
                # Send offboarding notification
                await send_employee_notification(
                    employee.email,
                    "offboarding",
                    {
                        "name": f"{employee.first_name} {employee.last_name}",
                        "employee_id": employee.employee_id,
                        "last_day": updated_employee.updated_at.isoformat()
                    }
                )
                
                logger.info(f"Employee terminated: {employee_id}")
                return True
            
            return False
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete employee {employee_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def search_employees(self, search_term: str, current_user: dict) -> List[Employee]:
        """Search employees with access control."""
        try:
            employees = await self.hr_service.search_employees(search_term)
            
            # Filter based on access permissions
            accessible_employees = []
            for employee in employees:
                if self._can_access_employee(current_user, employee):
                    accessible_employees.append(employee)
            
            return accessible_employees
            
        except Exception as e:
            logger.error(f"Failed to search employees: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    def _can_access_employee(self, current_user: dict, employee: Employee) -> bool:
        """Check if user can access employee data."""
        # HR admin can access all
        if self._is_hr_admin(current_user):
            return True
        
        # Managers can access their direct reports
        if employee.manager_id == current_user.get("employee_id"):
            return True
        
        # Users can access their own data
        if employee.employee_id == current_user.get("employee_id"):
            return True
        
        # Same department access for certain roles
        if (employee.department == current_user.get("department") and 
            current_user.get("role") in ["manager", "team_lead"]):
            return True
        
        return False
    
    def _can_modify_employee(self, current_user: dict, employee: Employee) -> bool:
        """Check if user can modify employee data."""
        # HR admin can modify all
        if self._is_hr_admin(current_user):
            return True
        
        # Managers can modify their direct reports (limited fields)
        if employee.manager_id == current_user.get("employee_id"):
            return True
        
        return False
    
    def _is_hr_admin(self, current_user: dict) -> bool:
        """Check if user is HR administrator."""
        return (current_user.get("department") == "hr" and 
                current_user.get("role") in ["admin", "hr_admin", "manager"])
    
    def _track_employee_changes(self, existing: Employee, update_data: EmployeeUpdate) -> dict:
        """Track changes made to employee."""
        changes = {}
        
        for field, new_value in update_data.dict().items():
            if new_value is not None:
                old_value = getattr(existing, field, None)
                if old_value != new_value:
                    changes[field] = {"old": old_value, "new": new_value}
        
        return changes