"""Decision and rule models for AI Decision Engine."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from enum import Enum

from shared_libs.models import BaseDocument


class RuleType(str, Enum):
    THRESHOLD = "threshold"
    CONDITION = "condition"
    PATTERN = "pattern"
    ANOMALY = "anomaly"


class RuleStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"


class ActionType(str, Enum):
    ALERT = "alert"
    NOTIFICATION = "notification"
    AUTOMATION = "automation"
    ESCALATION = "escalation"


class Rule(BaseDocument):
    """Business rule model."""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: int = 1  # 1-10, 10 being highest
    status: RuleStatus = RuleStatus.ACTIVE
    department: Optional[str] = None
    created_by: str
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


class Decision(BaseDocument):
    """Decision record model."""
    decision_id: str
    rule_id: str
    rule_name: str
    triggered_by: Dict[str, Any]
    decision_data: Dict[str, Any]
    actions_taken: List[Dict[str, Any]]
    confidence_score: float
    department: Optional[str] = None
    status: str = "executed"  # executed, pending, failed


class Recommendation(BaseDocument):
    """AI recommendation model."""
    recommendation_id: str
    title: str
    description: str
    category: str
    department: Optional[str] = None
    priority: int = 1
    confidence_score: float
    data_sources: List[str]
    suggested_actions: List[Dict[str, Any]]
    status: str = "pending"  # pending, accepted, rejected, implemented
    created_for: Optional[str] = None


class Alert(BaseDocument):
    """Alert model."""
    alert_id: str
    title: str
    message: str
    severity: str  # low, medium, high, critical
    department: Optional[str] = None
    rule_id: Optional[str] = None
    data: Dict[str, Any] = {}
    status: str = "active"  # active, acknowledged, resolved
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class RuleCreate(BaseModel):
    """Rule creation model."""
    name: str
    description: str
    rule_type: RuleType
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: int = 1
    department: Optional[str] = None


class RecommendationCreate(BaseModel):
    """Recommendation creation model."""
    title: str
    description: str
    category: str
    department: Optional[str] = None
    priority: int = 1
    confidence_score: float
    data_sources: List[str]
    suggested_actions: List[Dict[str, Any]]
    created_for: Optional[str] = None
