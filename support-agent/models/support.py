"""Support models for Support Agent."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr
from enum import Enum

from shared_libs.models import BaseDocument


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketCategory(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    GENERAL = "general"


class CustomerTier(str, Enum):
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class Ticket(BaseDocument):
    """Support ticket model."""
    ticket_id: str
    customer_id: str
    customer_email: EmailStr
    customer_name: str
    subject: str
    description: str
    category: TicketCategory
    priority: TicketPriority = TicketPriority.MEDIUM
    status: TicketStatus = TicketStatus.OPEN
    assigned_to: Optional[str] = None
    customer_tier: CustomerTier = CustomerTier.BASIC
    tags: List[str] = []
    attachments: List[str] = []
    resolution: Optional[str] = None
    satisfaction_score: Optional[int] = None
    first_response_time: Optional[int] = None  # minutes
    resolution_time: Optional[int] = None  # minutes
    escalated: bool = False
    escalated_at: Optional[datetime] = None


class KnowledgeArticle(BaseDocument):
    """Knowledge base article model."""
    article_id: str
    title: str
    content: str
    category: str
    tags: List[str] = []
    author: str
    status: str = "published"  # draft, published, archived
    views: int = 0
    helpful_votes: int = 0
    unhelpful_votes: int = 0
    last_updated: datetime


class ChatSession(BaseDocument):
    """Chat session model."""
    session_id: str
    customer_id: str
    customer_email: EmailStr
    messages: List[Dict[str, Any]] = []
    status: str = "active"  # active, ended
    agent_id: Optional[str] = None
    ai_handled: bool = True
    satisfaction_score: Optional[int] = None
    escalated_to_human: bool = False


class FAQ(BaseDocument):
    """FAQ model."""
    faq_id: str
    question: str
    answer: str
    category: str
    tags: List[str] = []
    views: int = 0
    helpful_votes: int = 0
    unhelpful_votes: int = 0


class TicketCreate(BaseModel):
    """Ticket creation model."""
    customer_email: EmailStr
    customer_name: str
    subject: str
    description: str
    category: TicketCategory
    priority: TicketPriority = TicketPriority.MEDIUM
    customer_tier: CustomerTier = CustomerTier.BASIC
    tags: List[str] = []


class TicketUpdate(BaseModel):
    """Ticket update model."""
    status: Optional[TicketStatus] = None
    priority: Optional[TicketPriority] = None
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None
    tags: Optional[List[str]] = None


class ChatMessage(BaseModel):
    """Chat message model."""
    message: str
    sender: str  # customer, agent, ai
    timestamp: datetime = None