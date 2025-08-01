"""Validation middleware for HR Agent."""

from fastapi import Request, HTTPException, status
from typing import Dict, Any
import json
from loguru import logger

from utils.validators import (
    validate_employee_data, 
    validate_job_posting, 
    validate_application,
    validate_attendance_record,
    validate_leave_request
)


class ValidationMiddleware:
    """Request validation middleware."""
    
    def __init__(self):
        self.validation_rules = {
            "/api/hr/employees": {
                "POST": validate_employee_data,
                "PUT": lambda data: validate_employee_data(data, is_update=True)
            },
            "/api/hr/recruitment/jobs": {
                "POST": validate_job_posting
            },
            "/api/hr/recruitment/applications": {
                "POST": validate_application
            },
            "/api/hr/attendance/records": {
                "POST": validate_attendance_record
            },
            "/api/hr/attendance/leave-requests": {
                "POST": validate_leave_request
            }
        }
    
    async def __call__(self, request: Request, call_next):
        """Validate request data."""
        # Only validate POST and PUT requests with JSON body
        if request.method not in ["POST", "PUT"]:
            return await call_next(request)
        
        # Check if path needs validation
        path = request.url.path
        validator = None
        
        for rule_path, methods in self.validation_rules.items():
            if path.startswith(rule_path) and request.method in methods:
                validator = methods[request.method]
                break
        
        if not validator:
            return await call_next(request)
        
        try:
            # Read and parse request body
            body = await request.body()
            if not body:
                return await call_next(request)
            
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON in request body"
                )
            
            # Validate data
            validation_result = await validator(data)
            
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "message": "Validation failed",
                        "errors": validation_result.errors
                    }
                )
            
            # Store validated data in request state
            request.state.validated_data = data
            
            # Continue with request
            response = await call_next(request)
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Validation middleware error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Validation service error"
            )


async def get_validated_data(request: Request) -> Dict[str, Any]:
    """Get validated data from request state."""
    if hasattr(request.state, 'validated_data'):
        return request.state.validated_data
    
    # Fallback to parsing body again
    body = await request.body()
    if body:
        return json.loads(body)
    
    return {}