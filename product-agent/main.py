"""Product/Engineering Agent - AI-powered development and project management."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from shared_libs.database import db_manager
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from routes.sprints import sprints_router
from routes.stories import stories_router
from routes.bugs import bugs_router
from routes.estimates import estimates_router
from services.ai_development import AIDevelopmentService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Product/Engineering Agent...")
    await db_manager.connect()
    
    # Initialize AI Development service
    app.state.ai_development_service = AIDevelopmentService()
    await app.state.ai_development_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Product/Engineering Agent...")
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - Product/Engineering Agent",
    description="AI-powered development and project management",
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
app.include_router(sprints_router, prefix="/api/product/sprints", tags=["Sprints"])
app.include_router(stories_router, prefix="/api/product/stories", tags=["Stories"])
app.include_router(bugs_router, prefix="/api/product/bugs", tags=["Bugs"])
app.include_router(estimates_router, prefix="/api/product/estimates", tags=["Estimates"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Product/Engineering Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "product-agent",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8003)),
        reload=True
    )
