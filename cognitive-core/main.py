"""Cognitive Core Engine - Central AI Brain for Phase 3+ Features."""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
from loguru import logger

from shared_libs.database import db_manager, get_database
from shared_libs.auth import get_current_user
from shared_libs.models import HealthResponse
from shared_libs.middleware import (
    logging_middleware, security_headers_middleware, setup_cors, 
    setup_rate_limiting, exception_handler
)
from services.intelligence_mesh import IntelligenceMesh
from services.predictive_engine import PredictiveEngine
from services.auto_resolver import AutoResolver
from services.optimization_engine import OptimizationEngine
from routes.intelligence import intelligence_router
from routes.predictions import predictions_router
from routes.optimization import optimization_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Cognitive Core Engine...")
    await db_manager.connect()
    
    # Initialize core services
    app.state.intelligence_mesh = IntelligenceMesh()
    await app.state.intelligence_mesh.initialize()
    
    app.state.predictive_engine = PredictiveEngine()
    await app.state.predictive_engine.initialize()
    
    app.state.auto_resolver = AutoResolver()
    await app.state.auto_resolver.initialize()
    
    app.state.optimization_engine = OptimizationEngine()
    await app.state.optimization_engine.initialize()
    
    # Start background tasks
    asyncio.create_task(app.state.intelligence_mesh.start_monitoring())
    asyncio.create_task(app.state.predictive_engine.start_prediction_cycles())
    asyncio.create_task(app.state.auto_resolver.start_resolution_monitoring())
    asyncio.create_task(app.state.optimization_engine.start_optimization_cycles())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Cognitive Core Engine...")
    await app.state.intelligence_mesh.stop_monitoring()
    await app.state.predictive_engine.stop_prediction_cycles()
    await app.state.auto_resolver.stop_resolution_monitoring()
    await app.state.optimization_engine.stop_optimization_cycles()
    await db_manager.disconnect()


# Create FastAPI app
app = FastAPI(
    title="AI Enterprise System - Cognitive Core Engine",
    description="Central AI Brain for Autonomous Enterprise Operations",
    version="3.0.0",
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
app.include_router(intelligence_router, prefix="/intelligence", tags=["Intelligence Mesh"])
app.include_router(predictions_router, prefix="/predictions", tags=["Predictive Engine"])
app.include_router(optimization_router, prefix="/optimization", tags=["Optimization Engine"])


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "service": "AI Enterprise System - Cognitive Core Engine",
        "version": "3.0.0",
        "status": "running",
        "capabilities": [
            "Intelligence Mesh Coordination",
            "Predictive Analytics",
            "Auto-Resolution",
            "System Optimization",
            "Cross-Agent Learning"
        ],
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
    
    # Check core services
    services_status = {}
    try:
        services_status["intelligence_mesh"] = "healthy" if app.state.intelligence_mesh else "unhealthy"
        services_status["predictive_engine"] = "healthy" if app.state.predictive_engine else "unhealthy"
        services_status["auto_resolver"] = "healthy" if app.state.auto_resolver else "unhealthy"
        services_status["optimization_engine"] = "healthy" if app.state.optimization_engine else "unhealthy"
    except Exception:
        services_status = {"all_services": "unhealthy"}
    
    return HealthResponse(
        service="cognitive-core",
        dependencies={
            "database": db_status,
            **services_status
        }
    )


@app.get("/system/intelligence-status")
async def intelligence_status(current_user: dict = Depends(get_current_user)):
    """Get system-wide intelligence status."""
    try:
        intelligence_mesh = app.state.intelligence_mesh
        predictive_engine = app.state.predictive_engine
        auto_resolver = app.state.auto_resolver
        optimization_engine = app.state.optimization_engine
        
        # Get status from each service
        mesh_status = await intelligence_mesh.get_mesh_status()
        prediction_status = await predictive_engine.get_prediction_status()
        resolver_status = await auto_resolver.get_resolver_status()
        optimization_status = await optimization_engine.get_optimization_status()
        
        return {
            "cognitive_core_status": "active",
            "intelligence_mesh": mesh_status,
            "predictive_engine": prediction_status,
            "auto_resolver": resolver_status,
            "optimization_engine": optimization_status,
            "system_intelligence_level": await _calculate_system_intelligence_level(
                mesh_status, prediction_status, resolver_status, optimization_status
            ),
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get intelligence status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve intelligence status")


async def _calculate_system_intelligence_level(mesh_status: dict, prediction_status: dict, 
                                             resolver_status: dict, optimization_status: dict) -> dict:
    """Calculate overall system intelligence level."""
    try:
        # Calculate intelligence metrics
        mesh_score = mesh_status.get("coordination_efficiency", 0.5)
        prediction_score = prediction_status.get("prediction_accuracy", 0.5)
        resolver_score = resolver_status.get("resolution_success_rate", 0.5)
        optimization_score = optimization_status.get("optimization_effectiveness", 0.5)
        
        overall_score = (mesh_score + prediction_score + resolver_score + optimization_score) / 4
        
        # Determine intelligence level
        if overall_score >= 0.9:
            level = "autonomous"
            description = "System operates with full autonomy and high intelligence"
        elif overall_score >= 0.75:
            level = "advanced"
            description = "System demonstrates advanced intelligence with minimal supervision"
        elif overall_score >= 0.6:
            level = "intermediate"
            description = "System shows good intelligence with moderate supervision"
        elif overall_score >= 0.4:
            level = "basic"
            description = "System has basic intelligence capabilities"
        else:
            level = "learning"
            description = "System is in learning phase"
        
        return {
            "level": level,
            "score": round(overall_score, 3),
            "description": description,
            "component_scores": {
                "intelligence_mesh": mesh_score,
                "predictive_engine": prediction_score,
                "auto_resolver": resolver_score,
                "optimization_engine": optimization_score
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate system intelligence level: {e}")
        return {"level": "unknown", "score": 0.5}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 9000)),
        reload=True
    )
