"""QA Agent - AI-powered quality assurance and testing automation."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from shared_libs.database import db_manager
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from routes.test_cases import test_cases_router
from routes.test_execution import test_execution_router
from routes.coverage import coverage_router
from routes.regression import regression_router
from services.ai_testing import AITestingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting QA Agent...")
    await db_manager.connect()
    
    # Initialize AI Testing service
    app.state.ai_testing_service = AITestingService()
    await app.state.ai_testing_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down QA Agent...")
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - QA Agent",
    description="AI-powered quality assurance and testing automation",
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
app.include_router(test_cases_router, prefix="/api/qa/test-cases", tags=["Test Cases"])
app.include_router(test_execution_router, prefix="/api/qa/execution", tags=["Test Execution"])
app.include_router(coverage_router, prefix="/api/qa/coverage", tags=["Coverage"])
app.include_router(regression_router, prefix="/api/qa/regression", tags=["Regression"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "QA Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "qa-agent",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8004)),
        reload=True
    )
