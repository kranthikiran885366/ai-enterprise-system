"""Marketing models for Marketing Agent."""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr
from enum import Enum

from shared_libs.models import BaseDocument


class CampaignType(str, Enum):
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    CONTENT = "content"
    PAID_ADS = "paid_ads"
    WEBINAR = "webinar"
    EVENT = "event"


class CampaignStatus(str, Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ContentType(str, Enum):
    BLOG_POST = "blog_post"
    EMAIL_TEMPLATE = "email_template"
    SOCIAL_POST = "social_post"
    LANDING_PAGE = "landing_page"
    VIDEO = "video"
    INFOGRAPHIC = "infographic"


class Campaign(BaseDocument):
    """Marketing campaign model."""
    campaign_id: str
    name: str
    description: str
    campaign_type: CampaignType
    status: CampaignStatus = CampaignStatus.DRAFT
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    budget: Optional[float] = None
    target_audience: Dict[str, Any] = {}
    goals: List[str] = []
    metrics: Dict[str, Any] = {}
    content_ids: List[str] = []
    created_by: str


class Content(BaseDocument):
    """Marketing content model."""
    content_id: str
    title: str
    content_type: ContentType
    content_body: str
    metadata: Dict[str, Any] = {}
    tags: List[str] = []
    status: str = "draft"  # draft, published, archived
    author: str
    ai_generated: bool = False
    performance_metrics: Dict[str, Any] = {}


class Lead(BaseDocument):
    """Marketing lead model."""
    lead_id: str
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    phone: Optional[str] = None
    source: str  # campaign_id, organic, referral
    score: Optional[float] = None
    status: str = "new"  # new, qualified, converted, lost
    tags: List[str] = []
    interactions: List[Dict[str, Any]] = []


class EmailCampaign(BaseDocument):
    """Email campaign model."""
    campaign_id: str
    subject_line: str
    email_content: str
    recipient_list: List[str] = []
    send_date: Optional[datetime] = None
    sent_count: int = 0
    opened_count: int = 0
    clicked_count: int = 0
    unsubscribed_count: int = 0
    bounced_count: int = 0


class SocialMediaPost(BaseDocument):
    """Social media post model."""
    post_id: str
    platform: str  # facebook, twitter, linkedin, instagram
    content: str
    media_urls: List[str] = []
    scheduled_time: Optional[datetime] = None
    posted_time: Optional[datetime] = None
    engagement_metrics: Dict[str, Any] = {}
    hashtags: List[str] = []


class CampaignCreate(BaseModel):
    """Campaign creation model."""
    name: str
    description: str
    campaign_type: CampaignType
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    budget: Optional[float] = None
    target_audience: Dict[str, Any] = {}
    goals: List[str] = []


class ContentCreate(BaseModel):
    """Content creation model."""
    title: str
    content_type: ContentType
    content_body: str
    metadata: Dict[str, Any] = {}
    tags: List[str] = []