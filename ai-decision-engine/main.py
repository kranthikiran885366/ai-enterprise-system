"""AI Decision Engine - Core AI Logic and Rule-Based Decision Making."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from shared_libs.database import db_manager
from shared_libs.messaging import message_broker
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from routes.rules import rules_router
from routes.recommendations import recommendations_router
from routes.analytics import analytics_router
from services.decision_engine import DecisionEngine
from services.rule_engine import RuleEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AI Decision Engine...")
    await db_manager.connect()
    await message_broker.connect()
    
    # Initialize services
    app.state.rule_engine = RuleEngine()
    await app.state.rule_engine.initialize()
    
    app.state.decision_engine = DecisionEngine()
    await app.state.decision_engine.initialize()
    
    # Start background tasks
    await app.state.decision_engine.start_monitoring()
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Decision Engine...")
    await app.state.decision_engine.stop_monitoring()
    await message_broker.disconnect()
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - AI Decision Engine",
    description="Core AI Logic and Rule-Based Decision Making",
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
app.include_router(rules_router, prefix="/api/ai/rules", tags=["Rules"])
app.include_router(recommendations_router, prefix="/api/ai/recommendations", tags=["Recommendations"])
app.include_router(analytics_router, prefix="/api/ai/analytics", tags=["Analytics"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AI Decision Engine",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ai-decision-engine",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8009)),
        reload=True
    )
