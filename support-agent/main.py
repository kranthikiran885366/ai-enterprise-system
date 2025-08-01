"""Customer Support Agent - AI-powered customer support and ticket management."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from shared_libs.database import db_manager
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from routes.tickets import tickets_router
from routes.chatbot import chatbot_router
from routes.knowledge_base import knowledge_base_router
from routes.escalation import escalation_router
from services.ai_support import AISupportService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Customer Support Agent...")
    await db_manager.connect()
    
    # Initialize AI Support service
    app.state.ai_support_service = AISupportService()
    await app.state.ai_support_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Customer Support Agent...")
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - Customer Support Agent",
    description="AI-powered customer support and ticket management",
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
app.include_router(tickets_router, prefix="/api/support/tickets", tags=["Tickets"])
app.include_router(chatbot_router, prefix="/api/support/chatbot", tags=["Chatbot"])
app.include_router(knowledge_base_router, prefix="/api/support/knowledge", tags=["Knowledge Base"])
app.include_router(escalation_router, prefix="/api/support/escalation", tags=["Escalation"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Customer Support Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "support-agent",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8005)),
        reload=True
    )
