"""Campaign data models for marketing operations."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class CampaignStatus(str, Enum):
    """Campaign status enumeration."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CampaignType(str, Enum):
    """Campaign type enumeration."""
    EMAIL = "email"
    SOCIAL = "social"
    CONTENT = "content"
    PAID_ADS = "paid_ads"
    INFLUENCER = "influencer"


class CampaignCreate(BaseModel):
    """Campaign creation model."""
    name: str = Field(..., min_length=1, max_length=255)
    description: str
    campaign_type: str
    owner: str
    target_audience: Dict[str, Any]
    start_date: datetime
    end_date: datetime
    budget: float = Field(..., gt=0)
    goals: List[str] = []

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Q1 Email Campaign",
                "description": "Quarterly email marketing campaign",
                "campaign_type": "email",
                "owner": "john_doe",
                "target_audience": {
                    "industry": "technology",
                    "company_size": "enterprise",
                    "region": "US"
                },
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-03-31T23:59:59",
                "budget": 50000.0,
                "goals": ["increase_leads", "improve_engagement"]
            }
        }


class CampaignUpdate(BaseModel):
    """Campaign update model."""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    budget: Optional[float] = None
    goals: Optional[List[str]] = None


class Campaign(BaseModel):
    """Campaign model."""
    campaign_id: str
    name: str
    description: str
    campaign_type: str
    owner: str
    status: str
    target_audience: Dict[str, Any]
    start_date: datetime
    end_date: datetime
    budget: float
    spent: float
    goals: List[str]
    content: List[Dict[str, Any]] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "campaign_id": "CAMPABCD1234",
                "name": "Q1 Email Campaign",
                "description": "Quarterly email marketing campaign",
                "campaign_type": "email",
                "owner": "john_doe",
                "status": "active",
                "target_audience": {
                    "industry": "technology",
                    "company_size": "enterprise",
                    "region": "US"
                },
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-03-31T23:59:59",
                "budget": 50000.0,
                "spent": 12500.0,
                "goals": ["increase_leads", "improve_engagement"],
                "content": [],
                "created_at": "2024-01-01T10:00:00",
                "updated_at": "2024-01-15T14:30:00"
            }
        }


class ContentCreate(BaseModel):
    """Campaign content creation model."""
    campaign_id: str
    content_type: str
    content: str
    channel: Optional[str] = None


class Content(BaseModel):
    """Campaign content model."""
    content_id: str
    campaign_id: str
    content_type: str
    content: str
    channel: Optional[str] = None
    created_at: datetime


class AudienceSegmentCreate(BaseModel):
    """Audience segment creation model."""
    name: str = Field(..., min_length=1, max_length=255)
    description: str
    criteria: Dict[str, Any]
    size_estimate: int = Field(..., ge=0)


class AudienceSegment(BaseModel):
    """Audience segment model."""
    segment_id: str
    name: str
    description: str
    criteria: Dict[str, Any]
    size_estimate: int
    created_at: datetime
    updated_at: datetime


class CampaignMetrics(BaseModel):
    """Campaign metrics model."""
    campaign_id: str
    date: str
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spent: float = 0.0
    revenue: float = 0.0
    ctr: float = 0.0
    conversion_rate: float = 0.0
    roi: float = 0.0
