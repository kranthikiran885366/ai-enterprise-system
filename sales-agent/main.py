"""Sales Agent - AI-powered sales management and automation."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from shared_libs.database import db_manager
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from routes.leads import leads_router
from routes.deals import deals_router
from routes.forecasting import forecasting_router
from routes.analytics import analytics_router
from services.sales_service import SalesService
from services.ai_sales import AISalesService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Sales Agent...")
    await db_manager.connect()
    
    # Initialize services
    app.state.sales_service = SalesService()
    await app.state.sales_service.initialize()
    
    app.state.ai_sales_service = AISalesService()
    await app.state.ai_sales_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sales Agent...")
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - Sales Agent",
    description="AI-powered sales management and automation",
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
app.include_router(leads_router, prefix="/api/sales/leads", tags=["Leads"])
app.include_router(deals_router, prefix="/api/sales/deals", tags=["Deals"])
app.include_router(forecasting_router, prefix="/api/sales/forecasting", tags=["Forecasting"])
app.include_router(analytics_router, prefix="/api/sales/analytics", tags=["Analytics"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Sales Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "sales-agent",
        "version": "1.0.0"
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get sales agent capabilities."""
    return {
        "service": "sales-agent",
        "capabilities": [
            "lead_management",
            "deal_tracking",
            "sales_forecasting",
            "ai_lead_scoring",
            "churn_prediction",
            "sales_analytics",
            "pipeline_management"
        ],
        "ai_features": {
            "lead_scoring": True,
            "churn_prediction": True,
            "sales_forecasting": True,
            "deal_probability": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8003)),
        reload=True
    )