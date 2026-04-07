from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class LeadStatus(str, Enum):
    new = "new"
    contacted = "contacted"
    qualified = "qualified"
    proposal = "proposal"
    negotiation = "negotiation"
    won = "won"
    lost = "lost"
    inactive = "inactive"


class DealStage(str, Enum):
    prospect = "prospect"
    qualified = "qualified"
    proposal = "proposal"
    negotiation = "negotiation"
    closed_won = "closed_won"
    closed_lost = "closed_lost"


class LeadSource(str, Enum):
    web = "web"
    referral = "referral"
    campaign = "campaign"
    cold_outreach = "cold_outreach"
    partner = "partner"
    event = "event"
    social = "social"


class LeadCreate(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    company: str
    job_title: Optional[str] = None
    source: LeadSource = LeadSource.web
    estimated_value: float = 0.0
    industry: Optional[str] = None
    company_size: Optional[str] = None
    notes: Optional[str] = None


class LeadUpdate(BaseModel):
    status: Optional[LeadStatus] = None
    score: Optional[int] = None
    assigned_to: Optional[int] = None
    estimated_value: Optional[float] = None
    notes: Optional[str] = None


class DealCreate(BaseModel):
    lead_id: int
    title: str
    value: float = Field(..., gt=0)
    stage: DealStage = DealStage.prospect
    probability: int = Field(default=10, ge=0, le=100)
    expected_close_date: str
    rep_id: int
    product: Optional[str] = None
    notes: Optional[str] = None


class DealUpdate(BaseModel):
    stage: Optional[DealStage] = None
    value: Optional[float] = None
    probability: Optional[int] = None
    expected_close_date: Optional[str] = None
    notes: Optional[str] = None
    lost_reason: Optional[str] = None


class QuoteCreate(BaseModel):
    deal_id: int
    line_items: List[dict]
    discount_percent: float = 0.0
    tax_percent: float = 0.0
    valid_until: str
    payment_terms: str = "net30"
    notes: Optional[str] = None


class CommissionCalculate(BaseModel):
    rep_id: int
    period_month: int = Field(..., ge=1, le=12)
    period_year: int
    commission_rate: float = Field(default=0.05, ge=0, le=1)
