"""Admin models for Admin Agent."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr
from enum import Enum

from shared_libs.models import BaseDocument


class AnnouncementType(str, Enum):
    GENERAL = "general"
    POLICY_UPDATE = "policy_update"
    SYSTEM_MAINTENANCE = "system_maintenance"
    EMERGENCY = "emergency"
    CELEBRATION = "celebration"
    TRAINING = "training"


class AnnouncementPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyType(str, Enum):
    HR_POLICY = "hr_policy"
    IT_POLICY = "it_policy"
    SECURITY_POLICY = "security_policy"
    FINANCIAL_POLICY = "financial_policy"
    GENERAL_POLICY = "general_policy"


class PermissionLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class Announcement(BaseDocument):
    """Company announcement model."""
    announcement_id: str
    title: str
    content: str
    announcement_type: AnnouncementType
    priority: AnnouncementPriority = AnnouncementPriority.MEDIUM
    target_audience: List[str] = []  # departments, roles, or "all"
    author: str
    published: bool = False
    publish_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    read_by: List[str] = []
    acknowledgment_required: bool = False
    acknowledged_by: List[str] = []


class Policy(BaseDocument):
    """Company policy model."""
    policy_id: str
    title: str
    description: str
    policy_type: PolicyType
    content: str
    version: str = "1.0"
    status: str = "draft"  # draft, review, approved, active, archived
    effective_date: Optional[date] = None
    review_date: Optional[date] = None
    author: str
    approver: Optional[str] = None
    approval_date: Optional[datetime] = None
    applicable_departments: List[str] = []
    compliance_requirements: List[str] = []


class Permission(BaseDocument):
    """User permission model."""
    permission_id: str
    user_id: str
    resource: str
    permission_level: PermissionLevel
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = {}


class SystemNotification(BaseDocument):
    """System notification model."""
    notification_id: str
    recipient_id: str
    recipient_email: EmailStr
    title: str
    message: str
    notification_type: str  # email, sms, push, in_app
    priority: str = "medium"
    sent: bool = False
    sent_at: Optional[datetime] = None
    read: bool = False
    read_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class UserRole(BaseDocument):
    """User role model."""
    role_id: str
    role_name: str
    description: str
    permissions: List[str] = []
    department: Optional[str] = None
    level: int = 1  # 1=basic, 5=admin
    created_by: str


class AnnouncementCreate(BaseModel):
    """Announcement creation model."""
    title: str
    content: str
    announcement_type: AnnouncementType
    priority: AnnouncementPriority = AnnouncementPriority.MEDIUM
    target_audience: List[str] = []
    publish_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    acknowledgment_required: bool = False


class PolicyCreate(BaseModel):
    """Policy creation model."""
    title: str
    description: str
    policy_type: PolicyType
    content: str
    effective_date: Optional[date] = None
    applicable_departments: List[str] = []
    compliance_requirements: List[str] = []