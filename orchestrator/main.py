"""Central Orchestrator - API Gateway and Service Registry."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
from loguru import logger

from shared_libs.database import db_manager, get_database
from shared_libs.auth import get_current_user, create_access_token, verify_password, get_password_hash
from shared_libs.models import HealthResponse, ServiceRegistration, InterServiceRequest
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from services.service_registry import ServiceRegistry
from services.auth_service import AuthService
from routes.auth import auth_router
from routes.services import services_router
from routes.proxy import proxy_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Central Orchestrator...")
    await db_manager.connect()
    
    # Initialize service registry
    app.state.service_registry = ServiceRegistry()
    await app.state.service_registry.initialize()
    
    # Initialize auth service
    app.state.auth_service = AuthService()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Central Orchestrator...")
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - Central Orchestrator",
    description="Central API Gateway and Service Registry for AI Enterprise System",
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
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(services_router, prefix="/services", tags=["Service Registry"])
app.include_router(proxy_router, prefix="/api", tags=["API Proxy"])


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "service": "AI Enterprise System - Central Orchestrator",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db = await get_database()
        await db.command("ping")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    return HealthResponse(
        service="orchestrator",
        dependencies={
            "database": db_status
        }
    )


@app.get("/system/status")
async def system_status(current_user: dict = Depends(get_current_user)):
    """Get system-wide status."""
    service_registry = app.state.service_registry
    services = await service_registry.get_all_services()
    
    system_health = {
        "orchestrator": "healthy",
        "total_services": len(services),
        "active_services": len([s for s in services if s.status == "active"]),
        "services": {}
    }
    
    # Check health of all registered services
    async with httpx.AsyncClient() as client:
        for service in services:
            try:
                response = await client.get(f"{service.url}{service.health_endpoint}", timeout=5.0)
                system_health["services"][service.name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "last_check": service.last_heartbeat.isoformat()
                }
            except Exception:
                system_health["services"][service.name] = {
                    "status": "unreachable",
                    "last_check": service.last_heartbeat.isoformat()
                }
    
    return system_health


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
