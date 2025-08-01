"""Rules management routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from shared_libs.auth import get_current_user
from models.decision import RuleCreate, Rule


router = APIRouter()


@router.post("/", response_model=dict)
async def create_rule(
    rule_data: RuleCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new business rule."""
    from main import app
    rule_engine = app.state.rule_engine
    
    rule = await rule_engine.create_rule(rule_data, current_user["sub"])
    
    if not rule:
        raise HTTPException(status_code=400, detail="Failed to create rule")
    
    return {
        "message": "Rule created successfully",
        "rule_id": rule.rule_id,
        "name": rule.name,
        "status": rule.status
    }


@router.get("/", response_model=List[dict])
async def list_rules(
    department: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """List business rules with filters."""
    from main import app
    rule_engine = app.state.rule_engine
    
    skip = (page - 1) * limit
    rules = await rule_engine.list_rules(department, status, skip, limit)
    
    return [
        {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "description": rule.description,
            "rule_type": rule.rule_type,
            "priority": rule.priority,
            "status": rule.status,
            "department": rule.department,
            "created_by": rule.created_by,
            "trigger_count": rule.trigger_count,
            "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
            "created_at": rule.created_at.isoformat()
        } for rule in rules
    ]


@router.get("/{rule_id}", response_model=dict)
async def get_rule(
    rule_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific rule."""
    from main import app
    rule_engine = app.state.rule_engine
    
    rule = await rule_engine.get_rule(rule_id)
    
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    return {
        "rule_id": rule.rule_id,
        "name": rule.name,
        "description": rule.description,
        "rule_type": rule.rule_type,
        "conditions": rule.conditions,
        "actions": rule.actions,
        "priority": rule.priority,
        "status": rule.status,
        "department": rule.department,
        "created_by": rule.created_by,
        "trigger_count": rule.trigger_count,
        "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
        "created_at": rule.created_at.isoformat(),
        "updated_at": rule.updated_at.isoformat()
    }


@router.get("/alerts/", response_model=List[dict])
async def list_alerts(
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """List alerts with filters."""
    from main import app
    rule_engine = app.state.rule_engine
    
    skip = (page - 1) * limit
    alerts = await rule_engine.get_alerts(status, severity, department, skip, limit)
    
    return [
        {
            "alert_id": alert.alert_id,
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity,
            "department": alert.department,
            "rule_id": alert.rule_id,
            "status": alert.status,
            "acknowledged_by": alert.acknowledged_by,
            "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            "created_at": alert.created_at.isoformat()
        } for alert in alerts
    ]


@router.put("/alerts/{alert_id}/acknowledge", response_model=dict)
async def acknowledge_alert(
    alert_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Acknowledge an alert."""
    from main import app
    rule_engine = app.state.rule_engine
    
    success = await rule_engine.acknowledge_alert(alert_id, current_user["sub"])
    
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return {"message": "Alert acknowledged successfully"}


rules_router = router
