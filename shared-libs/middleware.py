"""Shared middleware for all services."""

import time
import uuid
from typing import Callable
from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger
import redis.asyncio as redis


# Rate limiter
limiter = Limiter(key_func=get_remote_address)


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Logging middleware to track requests."""
    start_time = time.time()
    correlation_id = str(uuid.uuid4())
    
    # Add correlation ID to request state
    request.state.correlation_id = correlation_id
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        correlation_id=correlation_id,
        client_ip=request.client.host if request.client else None
    )
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time,
            correlation_id=correlation_id
        )
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        
        logger.error(
            "Request failed",
            method=request.method,
            url=str(request.url),
            error=str(e),
            process_time=process_time,
            correlation_id=correlation_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "correlation_id": correlation_id
            },
            headers={"X-Correlation-ID": correlation_id}
        )


async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """Add security headers to responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response


def setup_cors(app):
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_rate_limiting(app):
    """Setup rate limiting."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "message": "Request failed",
                "correlation_id": correlation_id
            },
            headers={"X-Correlation-ID": correlation_id}
        )
    
    logger.error(
        "Unhandled exception",
        error=str(exc),
        correlation_id=correlation_id,
        url=str(request.url)
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "correlation_id": correlation_id
        },
        headers={"X-Correlation-ID": correlation_id}
    )
