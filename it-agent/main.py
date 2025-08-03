"""IT Agent - Infrastructure and asset management microservice."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from shared_libs.database import db_manager
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from routes.assets import assets_router
from routes.tickets import tickets_router
from routes.infrastructure import infrastructure_router
from routes.security import security_router
from services.it_service import ITService
from services.ai_infrastructure import AIInfrastructureService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting IT Agent...")
    await db_manager.connect()
    
    # Initialize services
    app.state.it_service = ITService()
    await app.state.it_service.initialize()
    
    app.state.ai_infrastructure_service = AIInfrastructureService()
    await app.state.ai_infrastructure_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down IT Agent...")
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - IT Agent",
    description="Infrastructure and asset management microservice",
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
app.include_router(assets_router, prefix="/api/it/assets", tags=["Assets"])
app.include_router(tickets_router, prefix="/api/it/tickets", tags=["IT Tickets"])
app.include_router(infrastructure_router, prefix="/api/it/infrastructure", tags=["Infrastructure"])
app.include_router(security_router, prefix="/api/it/security", tags=["Security"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "IT Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "it-agent",
        "version": "1.0.0"
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get IT agent capabilities."""
    return {
        "service": "it-agent",
        "capabilities": [
            "asset_management",
            "infrastructure_monitoring",
            "security_management",
            "ticket_management",
            "automated_provisioning",
            "compliance_monitoring"
        ],
        "ai_features": {
            "predictive_maintenance": True,
            "security_threat_detection": True,
            "automated_incident_response": True,
            "capacity_planning": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8005)),
        reload=True
    )