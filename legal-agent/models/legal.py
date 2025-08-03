"""Legal models for Legal Agent."""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr
from enum import Enum

from shared_libs.models import BaseDocument


class ContractType(str, Enum):
    EMPLOYMENT = "employment"
    VENDOR = "vendor"
    CLIENT = "client"
    NDA = "nda"
    SERVICE_AGREEMENT = "service_agreement"
    LEASE = "lease"
    PARTNERSHIP = "partnership"


class ContractStatus(str, Enum):
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    PENDING_SIGNATURE = "pending_signature"
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class CaseStatus(str, Enum):
    OPEN = "open"
    INVESTIGATION = "investigation"
    LITIGATION = "litigation"
    SETTLEMENT = "settlement"
    CLOSED = "closed"


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    REMEDIATION_REQUIRED = "remediation_required"


class Contract(BaseDocument):
    """Legal contract model."""
    contract_id: str
    title: str
    contract_type: ContractType
    status: ContractStatus = ContractStatus.DRAFT
    parties: List[Dict[str, Any]] = []
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    renewal_date: Optional[date] = None
    auto_renewal: bool = False
    value: Optional[float] = None
    currency: str = "USD"
    terms: Dict[str, Any] = {}
    clauses: List[Dict[str, Any]] = []
    attachments: List[str] = []
    assigned_lawyer: Optional[str] = None
    risk_score: Optional[float] = None
    compliance_requirements: List[str] = []


class LegalCase(BaseDocument):
    """Legal case model."""
    case_id: str
    case_number: Optional[str] = None
    title: str
    description: str
    case_type: str  # litigation, compliance, employment, contract_dispute
    status: CaseStatus = CaseStatus.OPEN
    priority: str = "medium"  # low, medium, high, critical
    plaintiff: Optional[str] = None
    defendant: Optional[str] = None
    assigned_lawyer: str
    court: Optional[str] = None
    filing_date: Optional[date] = None
    hearing_dates: List[date] = []
    estimated_cost: Optional[float] = None
    actual_cost: Optional[float] = None
    documents: List[str] = []
    timeline: List[Dict[str, Any]] = []


class ComplianceCheck(BaseDocument):
    """Compliance check model."""
    check_id: str
    regulation_name: str
    regulation_type: str  # gdpr, sox, hipaa, pci_dss
    department: str
    status: ComplianceStatus
    last_check_date: datetime
    next_check_date: datetime
    findings: List[Dict[str, Any]] = []
    remediation_actions: List[Dict[str, Any]] = []
    assigned_to: str
    risk_level: str = "medium"


class LegalDocument(BaseDocument):
    """Legal document model."""
    document_id: str
    title: str
    document_type: str  # contract, policy, procedure, legal_opinion
    content: Optional[str] = None
    file_url: Optional[str] = None
    version: str = "1.0"
    status: str = "draft"  # draft, review, approved, archived
    author: str
    reviewer: Optional[str] = None
    approval_date: Optional[datetime] = None
    tags: List[str] = []
    related_cases: List[str] = []
    related_contracts: List[str] = []


class ContractCreate(BaseModel):
    """Contract creation model."""
    title: str
    contract_type: ContractType
    parties: List[Dict[str, Any]] = []
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    value: Optional[float] = None
    currency: str = "USD"
    terms: Dict[str, Any] = {}
    assigned_lawyer: Optional[str] = None


class CaseCreate(BaseModel):
    """Legal case creation model."""
    title: str
    description: str
    case_type: str
    priority: str = "medium"
    plaintiff: Optional[str] = None
    defendant: Optional[str] = None
    assigned_lawyer: str
    estimated_cost: Optional[float] = None