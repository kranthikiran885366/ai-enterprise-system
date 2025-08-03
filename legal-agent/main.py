"""Legal Agent - Legal compliance and document management microservice."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from shared_libs.database import db_manager
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from routes.contracts import contracts_router
from routes.compliance import compliance_router
from routes.cases import cases_router
from routes.documents import documents_router
from services.legal_service import LegalService
from services.ai_legal import AILegalService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Legal Agent...")
    await db_manager.connect()
    
    # Initialize services
    app.state.legal_service = LegalService()
    await app.state.legal_service.initialize()
    
    app.state.ai_legal_service = AILegalService()
    await app.state.ai_legal_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Legal Agent...")
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - Legal Agent",
    description="Legal compliance and document management microservice",
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
app.include_router(contracts_router, prefix="/api/legal/contracts", tags=["Contracts"])
app.include_router(compliance_router, prefix="/api/legal/compliance", tags=["Compliance"])
app.include_router(cases_router, prefix="/api/legal/cases", tags=["Legal Cases"])
app.include_router(documents_router, prefix="/api/legal/documents", tags=["Documents"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Legal Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "legal-agent",
        "version": "1.0.0"
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get legal agent capabilities."""
    return {
        "service": "legal-agent",
        "capabilities": [
            "contract_management",
            "compliance_monitoring",
            "legal_case_tracking",
            "document_analysis",
            "risk_assessment",
            "regulatory_compliance"
        ],
        "ai_features": {
            "contract_analysis": True,
            "compliance_monitoring": True,
            "risk_assessment": True,
            "document_classification": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8007)),
        reload=True
    )