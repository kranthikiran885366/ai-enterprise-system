"""Logging middleware for HR Agent."""

import time
import uuid
from fastapi import Request, Response
from loguru import logger
import json


class HRLoggingMiddleware:
    """HR-specific logging middleware."""
    
    def __init__(self):
        self.sensitive_fields = [
            "password", "ssn", "social_security", "tax_id", 
            "bank_account", "credit_card", "salary"
        ]
    
    async def __call__(self, request: Request, call_next):
        """Log HR operations with sensitive data protection."""
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        # Add correlation ID to request
        request.state.correlation_id = correlation_id
        
        # Log request (with sensitive data filtering)
        await self._log_request(request, correlation_id)
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            await self._log_response(request, response, process_time, correlation_id)
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Log error
            await self._log_error(request, e, process_time, correlation_id)
            raise
    
    async def _log_request(self, request: Request, correlation_id: str):
        """Log incoming request."""
        try:
            # Get user info if available
            user_info = getattr(request.state, 'current_user', {})
            user_id = user_info.get('sub', 'anonymous')
            user_role = user_info.get('role', 'unknown')
            
            # Get request body (filtered)
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body_bytes = await request.body()
                    if body_bytes:
                        body = json.loads(body_bytes)
                        body = self._filter_sensitive_data(body)
                except:
                    body = "Unable to parse body"
            
            logger.info(
                "HR Request",
                correlation_id=correlation_id,
                method=request.method,
                url=str(request.url),
                user_id=user_id,
                user_role=user_role,
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                body=body
            )
            
        except Exception as e:
            logger.error(f"Failed to log request: {e}")
    
    async def _log_response(self, request: Request, response: Response, 
                          process_time: float, correlation_id: str):
        """Log response."""
        try:
            user_info = getattr(request.state, 'current_user', {})
            user_id = user_info.get('sub', 'anonymous')
            
            logger.info(
                "HR Response",
                correlation_id=correlation_id,
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=round(process_time, 4),
                user_id=user_id
            )
            
            # Log specific HR operations
            if request.method == "POST" and "/employees" in str(request.url):
                logger.info(
                    "Employee Created",
                    correlation_id=correlation_id,
                    user_id=user_id,
                    operation="create_employee"
                )
            elif request.method == "PUT" and "/employees" in str(request.url):
                logger.info(
                    "Employee Updated",
                    correlation_id=correlation_id,
                    user_id=user_id,
                    operation="update_employee"
                )
            elif request.method == "DELETE" and "/employees" in str(request.url):
                logger.info(
                    "Employee Deleted",
                    correlation_id=correlation_id,
                    user_id=user_id,
                    operation="delete_employee"
                )
            
        except Exception as e:
            logger.error(f"Failed to log response: {e}")
    
    async def _log_error(self, request: Request, error: Exception, 
                        process_time: float, correlation_id: str):
        """Log error."""
        try:
            user_info = getattr(request.state, 'current_user', {})
            user_id = user_info.get('sub', 'anonymous')
            
            logger.error(
                "HR Request Error",
                correlation_id=correlation_id,
                method=request.method,
                url=str(request.url),
                error=str(error),
                error_type=type(error).__name__,
                process_time=round(process_time, 4),
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    def _filter_sensitive_data(self, data):
        """Filter sensitive data from logs."""
        if isinstance(data, dict):
            filtered = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                    filtered[key] = "[FILTERED]"
                elif isinstance(value, (dict, list)):
                    filtered[key] = self._filter_sensitive_data(value)
                else:
                    filtered[key] = value
            return filtered
        elif isinstance(data, list):
            return [self._filter_sensitive_data(item) for item in data]
        else:
            return data