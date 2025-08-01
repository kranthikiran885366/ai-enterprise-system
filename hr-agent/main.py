"""HR Agent - Human Resources Management Microservice."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from shared_libs.database import db_manager
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from routes.employees import employees_router
from routes.recruitment import recruitment_router
from routes.attendance import attendance_router
from services.hr_service import HRService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting HR Agent...")
    await db_manager.connect()
    
    # Initialize HR service
    app.state.hr_service = HRService()
    await app.state.hr_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down HR Agent...")
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - HR Agent",
    description="Human Resources Management Microservice",
    version="1.0.0",
    lifespan=lifespan
)

# Setup middleware
app.middleware("http")(logging_middleware)
app.middleware("http")(security_headers_middleware)
setup_cors(app)
setup_rate_limiting(app)

# Add exception handler
app.add_exception_handler(Exception, exception_handler)

# Include routers
app.include_router(employees_router, prefix="/api/hr/employees", tags=["Employees"])
app.include_router(recruitment_router, prefix="/api/hr/recruitment", tags=["Recruitment"])
app.include_router(attendance_router, prefix="/api/hr/attendance", tags=["Attendance"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "HR Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "hr-agent",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=True
    )
