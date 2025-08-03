"""Admin Agent - Administrative operations and announcements microservice."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from shared_libs.database import db_manager
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from routes.announcements import announcements_router
from routes.policies import policies_router
from routes.permissions import permissions_router
from routes.notifications import notifications_router
from services.admin_service import AdminService
from services.ai_admin import AIAdminService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Admin Agent...")
    await db_manager.connect()
    
    # Initialize services
    app.state.admin_service = AdminService()
    await app.state.admin_service.initialize()
    
    app.state.ai_admin_service = AIAdminService()
    await app.state.ai_admin_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Admin Agent...")
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - Admin Agent",
    description="Administrative operations and announcements microservice",
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
app.include_router(announcements_router, prefix="/api/admin/announcements", tags=["Announcements"])
app.include_router(policies_router, prefix="/api/admin/policies", tags=["Policies"])
app.include_router(permissions_router, prefix="/api/admin/permissions", tags=["Permissions"])
app.include_router(notifications_router, prefix="/api/admin/notifications", tags=["Notifications"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Admin Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "admin-agent",
        "version": "1.0.0"
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get admin agent capabilities."""
    return {
        "service": "admin-agent",
        "capabilities": [
            "announcement_management",
            "policy_management",
            "permission_management",
            "notification_system",
            "user_management",
            "system_configuration"
        ],
        "ai_features": {
            "smart_notifications": True,
            "policy_analysis": True,
            "user_behavior_analysis": True,
            "automated_announcements": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8006)),
        reload=True
    )