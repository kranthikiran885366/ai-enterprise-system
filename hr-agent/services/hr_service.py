"""HR Service for managing HR operations."""

import uuid
from datetime import datetime
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger

from shared_libs.database import get_database
from models.employee import Employee, EmployeeCreate, EmployeeUpdate


class HRService:
    """HR service for managing employees and HR operations."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.employees_collection = "employees"
        self.recruitment_collection = "recruitment"
        self.attendance_collection = "attendance"
    
    async def initialize(self):
        """Initialize the HR service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.employees_collection].create_index("employee_id", unique=True)
        await self.db[self.employees_collection].create_index("email", unique=True)
        await self.db[self.employees_collection].create_index("department")
        await self.db[self.employees_collection].create_index("status")
        
        logger.info("HR service initialized")
    
    async def create_employee(self, employee_data: EmployeeCreate) -> Optional[Employee]:
        """Create a new employee."""
        try:
            # Generate employee ID
            employee_id = f"EMP{str(uuid.uuid4())[:8].upper()}"
            
            employee_dict = employee_data.dict()
            employee_dict["employee_id"] = employee_id
            employee_dict["created_at"] = datetime.utcnow()
            employee_dict["updated_at"] = datetime.utcnow()
            
            result = await self.db[self.employees_collection].insert_one(employee_dict)
            
            if result.inserted_id:
                employee_dict["_id"] = result.inserted_id
                logger.info(f"Employee created: {employee_id}")
                return Employee(**employee_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create employee: {e}")
            return None
    
    async def get_employee(self, employee_id: str) -> Optional[Employee]:
        """Get an employee by ID."""
        try:
            employee_doc = await self.db[self.employees_collection].find_one(
                {"employee_id": employee_id}
            )
            
            if employee_doc:
                return Employee(**employee_doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get employee {employee_id}: {e}")
            return None
    
    async def update_employee(self, employee_id: str, update_data: EmployeeUpdate) -> Optional[Employee]:
        """Update an employee."""
        try:
            update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
            update_dict["updated_at"] = datetime.utcnow()
            
            result = await self.db[self.employees_collection].update_one(
                {"employee_id": employee_id},
                {"$set": update_dict}
            )
            
            if result.modified_count > 0:
                updated_employee = await self.get_employee(employee_id)
                logger.info(f"Employee updated: {employee_id}")
                return updated_employee
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to update employee {employee_id}: {e}")
            return None
    
    async def delete_employee(self, employee_id: str) -> bool:
        """Delete an employee."""
        try:
            result = await self.db[self.employees_collection].delete_one(
                {"employee_id": employee_id}
            )
            
            if result.deleted_count > 0:
                logger.info(f"Employee deleted: {employee_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete employee {employee_id}: {e}")
            return False
    
    async def list_employees(self, department: Optional[str] = None, status: Optional[str] = None, 
                           skip: int = 0, limit: int = 10) -> List[Employee]:
        """List employees with optional filters."""
        try:
            query = {}
            if department:
                query["department"] = department
            if status:
                query["status"] = status
            
            employees = []
            cursor = self.db[self.employees_collection].find(query).skip(skip).limit(limit)
            
            async for employee_doc in cursor:
                employees.append(Employee(**employee_doc))
            
            return employees
            
        except Exception as e:
            logger.error(f"Failed to list employees: {e}")
            return []
    
    async def get_employee_count(self, department: Optional[str] = None, status: Optional[str] = None) -> int:
        """Get total count of employees with optional filters."""
        try:
            query = {}
            if department:
                query["department"] = department
            if status:
                query["status"] = status
            
            count = await self.db[self.employees_collection].count_documents(query)
            return count
            
        except Exception as e:
            logger.error(f"Failed to get employee count: {e}")
            return 0
    
    async def search_employees(self, search_term: str) -> List[Employee]:
        """Search employees by name or email."""
        try:
            query = {
                "$or": [
                    {"first_name": {"$regex": search_term, "$options": "i"}},
                    {"last_name": {"$regex": search_term, "$options": "i"}},
                    {"email": {"$regex": search_term, "$options": "i"}},
                    {"employee_id": {"$regex": search_term, "$options": "i"}}
                ]
            }
            
            employees = []
            cursor = self.db[self.employees_collection].find(query)
            
            async for employee_doc in cursor:
                employees.append(Employee(**employee_doc))
            
            return employees
            
        except Exception as e:
            logger.error(f"Failed to search employees: {e}")
            return []
