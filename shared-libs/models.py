"""Shared Pydantic models across all services."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class BaseDocument(BaseModel):
    """Base document model."""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    service: str
    version: str = "1.0.0"
    dependencies: Dict[str, str] = {}


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: Optional[str] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""
    
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=10, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """Paginated response model."""
    
    items: List[Any]
    total: int
    page: int
    limit: int
    pages: int


class ServiceRegistration(BaseModel):
    """Service registration model."""
    
    name: str
    url: str
    health_endpoint: str
    status: str = "active"
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}


class InterServiceRequest(BaseModel):
    """Inter-service communication request model."""
    
    source_service: str
    target_service: str
    action: str
    data: Dict[str, Any] = {}
    correlation_id: Optional[str] = None


class InterServiceResponse(BaseModel):
    """Inter-service communication response model."""
    
    success: bool
    data: Dict[str, Any] = {}
    error: Optional[str] = None
    correlation_id: Optional[str] = None
