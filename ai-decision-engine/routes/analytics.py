"""Analytics and insights routes."""

from typing import Optional
from fastapi import APIRouter, Depends, Query

from shared_libs.auth import get_current_user


router = APIRouter()


@router.get("/summary", response_model=dict)
async def get_analytics_summary(
    days: int = Query(7, ge=1, le=90),
    current_user: dict = Depends(get_current_user)
):
    """Get analytics summary for the specified number of days."""
    from main import app
    decision_engine = app.state.decision_engine
    
    summary = await decision_engine.get_analytics_summary(days)
    
    return summary


@router.get("/dashboard", response_model=dict)
async def get_dashboard_data(
    current_user: dict = Depends(get_current_user)
):
    """Get dashboard data for AI insights."""
    from main import app
    decision_engine = app.state.decision_engine
    rule_engine = app.state.rule_engine
    
    # Get recent recommendations
    recent_recommendations = await decision_engine.get_recommendations(limit=5)
    
    # Get active alerts
    active_alerts = await rule_engine.get_alerts(status="active", limit=10)
    
    # Get analytics summary
    analytics = await decision_engine.get_analytics_summary(7)
    
    return {
        "recent_recommendations": [
            {
                "recommendation_id": rec.recommendation_id,
                "title": rec.title,
                "category": rec.category,
                "priority": rec.priority,
                "confidence_score": rec.confidence_score,
                "status": rec.status,
                "created_at": rec.created_at.isoformat()
            } for rec in recent_recommendations
        ],
        "active_alerts": [
            {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "severity": alert.severity,
                "department": alert.department,
                "created_at": alert.created_at.isoformat()
            } for alert in active_alerts
        ],
        "analytics_summary": analytics,
        "system_status": {
            "total_rules": len(await rule_engine.list_rules(limit=1000)),
            "active_rules": len(await rule_engine.list_rules(status="active", limit=1000)),
            "pending_recommendations": len(await decision_engine.get_recommendations(status="pending", limit=1000)),
            "active_alerts": len(active_alerts)
        }
    }


analytics_router = router
