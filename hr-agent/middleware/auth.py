"""Authentication middleware for HR Agent."""

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from loguru import logger

from shared_libs.auth import verify_token


security = HTTPBearer()


class HRAuthMiddleware:
    """HR-specific authentication middleware."""
    
    def __init__(self):
        self.public_endpoints = [
            "/",
            "/health",
            "/docs",
            "/openapi.json"
        ]
    
    async def __call__(self, request: Request, call_next):
        """Process authentication for HR endpoints."""
        # Skip auth for public endpoints
        if request.url.path in self.public_endpoints:
            return await call_next(request)
        
        # Check for authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authorization header"
            )
        
        try:
            # Extract and verify token
            token = auth_header.split(" ")[1]
            payload = verify_token(token)
            
            # Add user info to request state
            request.state.current_user = payload
            
            # Check HR-specific permissions
            if not self._has_hr_access(payload, request.url.path):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions for HR operations"
                )
            
            response = await call_next(request)
            return response
            
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error"
            )
    
    def _has_hr_access(self, user_payload: dict, path: str) -> bool:
        """Check if user has access to HR operations."""
        user_department = user_payload.get("department", "")
        user_role = user_payload.get("role", "")
        
        # HR department has full access
        if user_department == "hr":
            return True
        
        # Managers can access employee data in their department
        if user_role in ["manager", "director"] and "/employees" in path:
            return True
        
        # Users can access their own data
        if "/employees/" in path and user_payload.get("employee_id") in path:
            return True
        
        # Admin users have full access
        if user_role in ["admin", "super_admin"]:
            return True
        
        return False


async def get_current_user_from_request(request: Request) -> dict:
    """Get current user from request state."""
    if hasattr(request.state, 'current_user'):
        return request.state.current_user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="User not authenticated"
    )


def require_hr_admin(user: dict) -> bool:
    """Check if user is HR administrator."""
    return (user.get("department") == "hr" and 
            user.get("role") in ["admin", "hr_admin", "manager"])


def require_manager_or_hr(user: dict) -> bool:
    """Check if user is manager or HR."""
    return (user.get("role") in ["manager", "director"] or 
            user.get("department") == "hr")


def can_access_employee_data(user: dict, employee_id: str) -> bool:
    """Check if user can access specific employee data."""
    # HR can access all
    if user.get("department") == "hr":
        return True
    
    # Users can access their own data
    if user.get("employee_id") == employee_id:
        return True
    
    # Managers can access their direct reports
    # This would require checking manager-employee relationships in real implementation
    
    return False