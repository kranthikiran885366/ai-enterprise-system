"""IT infrastructure models."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from enum import Enum

from shared_libs.models import BaseDocument


class AssetType(str, Enum):
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    SERVER = "server"
    NETWORK_DEVICE = "network_device"
    MOBILE_DEVICE = "mobile_device"
    SOFTWARE_LICENSE = "software_license"
    PRINTER = "printer"
    MONITOR = "monitor"


class AssetStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    RETIRED = "retired"
    LOST = "lost"
    STOLEN = "stolen"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Asset(BaseDocument):
    """IT Asset model."""
    asset_id: str
    asset_tag: str
    name: str
    asset_type: AssetType
    brand: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    purchase_date: Optional[datetime] = None
    purchase_cost: Optional[float] = None
    warranty_expiry: Optional[datetime] = None
    assigned_to: Optional[str] = None
    location: Optional[str] = None
    status: AssetStatus = AssetStatus.ACTIVE
    specifications: Dict[str, Any] = {}
    maintenance_schedule: List[Dict[str, Any]] = []
    depreciation_rate: Optional[float] = None


class ITTicket(BaseDocument):
    """IT Support ticket model."""
    ticket_id: str
    employee_id: str
    employee_email: str
    subject: str
    description: str
    category: str  # hardware, software, network, security, access
    priority: TicketPriority = TicketPriority.MEDIUM
    status: TicketStatus = TicketStatus.OPEN
    assigned_to: Optional[str] = None
    asset_id: Optional[str] = None
    resolution: Optional[str] = None
    resolution_time: Optional[int] = None  # minutes
    escalated: bool = False


class SecurityIncident(BaseDocument):
    """Security incident model."""
    incident_id: str
    incident_type: str  # malware, phishing, data_breach, unauthorized_access
    severity: str  # low, medium, high, critical
    description: str
    affected_systems: List[str] = []
    affected_users: List[str] = []
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    status: str = "open"  # open, investigating, contained, resolved
    response_actions: List[Dict[str, Any]] = []


class NetworkDevice(BaseDocument):
    """Network device model."""
    device_id: str
    device_name: str
    device_type: str  # router, switch, firewall, access_point
    ip_address: str
    mac_address: Optional[str] = None
    location: str
    status: str = "active"  # active, inactive, maintenance
    last_ping: Optional[datetime] = None
    uptime_percentage: Optional[float] = None
    configuration: Dict[str, Any] = {}


class SoftwareLicense(BaseDocument):
    """Software license model."""
    license_id: str
    software_name: str
    license_type: str  # perpetual, subscription, volume
    license_key: Optional[str] = None
    seats_total: int
    seats_used: int = 0
    purchase_date: datetime
    expiry_date: Optional[datetime] = None
    cost: float
    vendor: str
    assigned_users: List[str] = []


class AssetCreate(BaseModel):
    """Asset creation model."""
    name: str
    asset_type: AssetType
    brand: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    purchase_date: Optional[datetime] = None
    purchase_cost: Optional[float] = None
    warranty_expiry: Optional[datetime] = None
    assigned_to: Optional[str] = None
    location: Optional[str] = None
    specifications: Dict[str, Any] = {}


class ITTicketCreate(BaseModel):
    """IT ticket creation model."""
    employee_email: str
    subject: str
    description: str
    category: str
    priority: TicketPriority = TicketPriority.MEDIUM
    asset_id: Optional[str] = None