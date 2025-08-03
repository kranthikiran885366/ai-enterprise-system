"""Sales models for Sales Agent."""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr
from enum import Enum

from shared_libs.models import BaseDocument


class LeadStatus(str, Enum):
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class LeadSource(str, Enum):
    WEBSITE = "website"
    REFERRAL = "referral"
    COLD_OUTREACH = "cold_outreach"
    SOCIAL_MEDIA = "social_media"
    EVENT = "event"
    ADVERTISEMENT = "advertisement"
    PARTNER = "partner"


class DealStage(str, Enum):
    PROSPECTING = "prospecting"
    QUALIFICATION = "qualification"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class Lead(BaseDocument):
    """Lead model."""
    lead_id: str
    company_name: str
    contact_name: str
    contact_email: EmailStr
    contact_phone: Optional[str] = None
    job_title: Optional[str] = None
    company_size: Optional[int] = None
    industry: Optional[str] = None
    budget: Optional[float] = None
    source: LeadSource
    status: LeadStatus = LeadStatus.NEW
    assigned_to: Optional[str] = None
    ai_score: Optional[float] = None
    notes: List[Dict[str, Any]] = []
    tags: List[str] = []
    last_contacted: Optional[datetime] = None


class Deal(BaseDocument):
    """Deal model."""
    deal_id: str
    lead_id: Optional[str] = None
    company_name: str
    contact_name: str
    contact_email: EmailStr
    deal_value: float
    currency: str = "USD"
    stage: DealStage
    probability: float = 0.0
    expected_close_date: Optional[date] = None
    actual_close_date: Optional[date] = None
    assigned_to: str
    products: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []
    activities: List[Dict[str, Any]] = []


class SalesActivity(BaseDocument):
    """Sales activity model."""
    activity_id: str
    lead_id: Optional[str] = None
    deal_id: Optional[str] = None
    activity_type: str  # call, email, meeting, demo
    subject: str
    description: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_to: str
    outcome: Optional[str] = None
    next_action: Optional[str] = None


class SalesTarget(BaseDocument):
    """Sales target model."""
    target_id: str
    sales_rep: str
    period: str  # monthly, quarterly, yearly
    year: int
    month: Optional[int] = None
    quarter: Optional[int] = None
    target_amount: float
    achieved_amount: float = 0.0
    target_deals: int
    achieved_deals: int = 0


class LeadCreate(BaseModel):
    """Lead creation model."""
    company_name: str
    contact_name: str
    contact_email: EmailStr
    contact_phone: Optional[str] = None
    job_title: Optional[str] = None
    company_size: Optional[int] = None
    industry: Optional[str] = None
    budget: Optional[float] = None
    source: LeadSource
    notes: Optional[str] = None
    tags: List[str] = []


class DealCreate(BaseModel):
    """Deal creation model."""
    lead_id: Optional[str] = None
    company_name: str
    contact_name: str
    contact_email: EmailStr
    deal_value: float
    currency: str = "USD"
    stage: DealStage
    expected_close_date: Optional[date] = None
    products: List[Dict[str, Any]] = []
    notes: Optional[str] = None


class ActivityCreate(BaseModel):
    """Activity creation model."""
    lead_id: Optional[str] = None
    deal_id: Optional[str] = None
    activity_type: str
    subject: str
    description: Optional[str] = None
    scheduled_at: Optional[datetime] = None