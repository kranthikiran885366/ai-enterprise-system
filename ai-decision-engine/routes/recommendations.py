"""Recommendations management routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from shared_libs.auth import get_current_user
from models.decision import RecommendationCreate


router = APIRouter()


class RecommendationStatusUpdate(BaseModel):
    status: str  # pending, accepted, rejected, implemented


@router.post("/", response_model=dict)
async def create_recommendation(
    recommendation_data: RecommendationCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new recommendation."""
    from main import app
    decision_engine = app.state.decision_engine
    
    recommendation = await decision_engine.create_recommendation(recommendation_data)
    
    if not recommendation:
        raise HTTPException(status_code=400, detail="Failed to create recommendation")
    
    return {
        "message": "Recommendation created successfully",
        "recommendation_id": recommendation.recommendation_id,
        "title": recommendation.title,
        "category": recommendation.category
    }


@router.get("/", response_model=List[dict])
async def list_recommendations(
    department: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """List recommendations with filters."""
    from main import app
    decision_engine = app.state.decision_engine
    
    skip = (page - 1) * limit
    recommendations = await decision_engine.get_recommendations(department, category, status, skip, limit)
    
    return [
        {
            "recommendation_id": rec.recommendation_id,
            "title": rec.title,
            "description": rec.description,
            "category": rec.category,
            "department": rec.department,
            "priority": rec.priority,
            "confidence_score": rec.confidence_score,
            "status": rec.status,
            "created_for": rec.created_for,
            "created_at": rec.created_at.isoformat()
        } for rec in recommendations
    ]


@router.get("/{recommendation_id}", response_model=dict)
async def get_recommendation(
    recommendation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific recommendation."""
    from main import app
    decision_engine = app.state.decision_engine
    
    recommendations = await decision_engine.get_recommendations()
    recommendation = next((r for r in recommendations if r.recommendation_id == recommendation_id), None)
    
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    
    return {
        "recommendation_id": recommendation.recommendation_id,
        "title": recommendation.title,
        "description": recommendation.description,
        "category": recommendation.category,
        "department": recommendation.department,
        "priority": recommendation.priority,
        "confidence_score": recommendation.confidence_score,
        "data_sources": recommendation.data_sources,
        "suggested_actions": recommendation.suggested_actions,
        "status": recommendation.status,
        "created_for": recommendation.created_for,
        "created_at": recommendation.created_at.isoformat(),
        "updated_at": recommendation.updated_at.isoformat()
    }


@router.put("/{recommendation_id}/status", response_model=dict)
async def update_recommendation_status(
    recommendation_id: str,
    status_update: RecommendationStatusUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update recommendation status."""
    from main import app
    decision_engine = app.state.decision_engine
    
    success = await decision_engine.update_recommendation_status(
        recommendation_id, 
        status_update.status, 
        current_user["sub"]
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    
    return {"message": f"Recommendation status updated to {status_update.status}"}


recommendations_router = router
