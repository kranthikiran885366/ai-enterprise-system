"""Marketing Agent - AI-powered marketing automation and analytics."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from shared_libs.database import db_manager
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from routes.campaigns import campaigns_router
from routes.analytics import analytics_router
from routes.content import content_router
from routes.leads import leads_router
from services.marketing_service import MarketingService
from services.ai_marketing import AIMarketingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Marketing Agent...")
    await db_manager.connect()
    
    # Initialize services
    app.state.marketing_service = MarketingService()
    await app.state.marketing_service.initialize()
    
    app.state.ai_marketing_service = AIMarketingService()
    await app.state.ai_marketing_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Marketing Agent...")
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - Marketing Agent",
    description="AI-powered marketing automation and analytics",
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
app.include_router(campaigns_router, prefix="/api/marketing/campaigns", tags=["Campaigns"])
app.include_router(analytics_router, prefix="/api/marketing/analytics", tags=["Analytics"])
app.include_router(content_router, prefix="/api/marketing/content", tags=["Content"])
app.include_router(leads_router, prefix="/api/marketing/leads", tags=["Lead Generation"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Marketing Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "marketing-agent",
        "version": "1.0.0"
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get marketing agent capabilities."""
    return {
        "service": "marketing-agent",
        "capabilities": [
            "campaign_management",
            "lead_generation",
            "content_optimization",
            "analytics_reporting",
            "ai_content_generation",
            "audience_segmentation",
            "performance_tracking"
        ],
        "ai_features": {
            "content_generation": True,
            "audience_targeting": True,
            "campaign_optimization": True,
            "sentiment_analysis": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8004)),
        reload=True
    )