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
from services.ai_recruitment import AIRecruitmentService
from config.database import hr_db_manager
from middleware.auth import HRAuthMiddleware
from middleware.validation import ValidationMiddleware
from middleware.logging import HRLoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting HR Agent...")
    await hr_db_manager.connect()
    
    # Initialize HR service
    app.state.hr_service = HRService()
    await app.state.hr_service.initialize()
    
    # Initialize AI Recruitment service
    app.state.ai_recruitment_service = AIRecruitmentService()
    await app.state.ai_recruitment_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down HR Agent...")
    await hr_db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - HR Agent",
    description="Human Resources Management Microservice",
    version="1.0.0",
    lifespan=lifespan
)

# Setup middleware
app.middleware("http")(HRLoggingMiddleware())
app.middleware("http")(ValidationMiddleware())
app.middleware("http")(HRAuthMiddleware())
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
    # Check database health
    db_healthy = await hr_db_manager.health_check()
    
    health_status = {
        "status": "healthy",
        "service": "hr-agent",
        "version": "1.0.0",
        "database": "healthy" if db_healthy else "unhealthy",
        "ai_features": "enabled" if hasattr(app.state, 'ai_recruitment_service') else "disabled"
    }
    
    if not db_healthy:
        health_status["status"] = "unhealthy"
    
    return health_status


@app.get("/capabilities")
async def get_capabilities():
    """Get HR agent capabilities."""
    return {
        "service": "hr-agent",
        "capabilities": [
            "employee_management",
            "recruitment",
            "attendance_tracking",
            "ai_resume_analysis",
            "ai_interviews",
            "performance_tracking",
            "leave_management"
        ],
        "ai_features": {
            "resume_analysis": True,
            "ai_interviews": True,
            "mood_tracking": True,
            "attrition_prediction": True
        }
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=True
    )
