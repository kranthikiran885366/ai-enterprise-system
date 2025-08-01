"""Finance Agent - Financial Management Microservice."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from shared_libs.database import db_manager
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from routes.expenses import expenses_router
from routes.invoices import invoices_router
from routes.budget import budget_router
from routes.payroll import payroll_router
from services.finance_service import FinanceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Finance Agent...")
    await db_manager.connect()
    
    # Initialize Finance service
    app.state.finance_service = FinanceService()
    await app.state.finance_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Finance Agent...")
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - Finance Agent",
    description="Financial Management Microservice",
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
app.include_router(expenses_router, prefix="/api/finance/expenses", tags=["Expenses"])
app.include_router(invoices_router, prefix="/api/finance/invoices", tags=["Invoices"])
app.include_router(budget_router, prefix="/api/finance/budget", tags=["Budget"])
app.include_router(payroll_router, prefix="/api/finance/payroll", tags=["Payroll"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Finance Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "finance-agent",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8002)),
        reload=True
    )
