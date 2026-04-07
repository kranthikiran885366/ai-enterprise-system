from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Any, Dict
from datetime import datetime
from enum import Enum


class Role(str, Enum):
    admin = "admin"
    manager = "manager"
    agent = "agent"
    readonly = "readonly"


class Department(str, Enum):
    hr = "hr"
    finance = "finance"
    sales = "sales"
    marketing = "marketing"
    it = "it"
    admin = "admin"
    legal = "legal"
    support = "support"
    analytics = "analytics"
    audit = "audit"
    executive = "executive"
    all = "all"


class TokenData(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: int
    username: str
    role: str
    department: str


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=6)
    role: Role = Role.readonly
    department: Department = Department.all
    full_name: Optional[str] = None


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: str
    department: str
    full_name: Optional[str]
    is_active: bool
    created_at: str


class LoginRequest(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
