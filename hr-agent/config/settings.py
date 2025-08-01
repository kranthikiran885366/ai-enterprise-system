"""Configuration settings for HR Agent."""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator


class HRSettings(BaseSettings):
    """HR Agent configuration settings."""
    
    # Service settings
    service_name: str = "hr-agent"
    service_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8001
    
    # Database settings
    mongodb_url: str = "mongodb://localhost:27017/enterprise"
    database_name: str = "enterprise"
    
    # Authentication settings
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 30
    
    # Email settings
    email_service: str = "log"  # log, smtp, sendgrid
    smtp_server: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    sendgrid_api_key: str = ""
    from_email: str = "noreply@company.com"
    
    # Admin settings
    admin_emails: List[str] = ["admin@company.com"]
    
    # HR-specific settings
    default_leave_accrual_rate: float = 2.0  # days per month
    max_leave_days_per_request: int = 30
    advance_notice_days: int = 7
    probation_period_months: int = 6
    performance_review_frequency_months: int = 12
    
    # File upload settings
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = ["pdf", "doc", "docx", "txt"]
    upload_directory: str = "/tmp/hr-uploads"
    
    # AI settings
    openai_api_key: Optional[str] = None
    enable_ai_features: bool = True
    ai_confidence_threshold: float = 0.7
    
    # Security settings
    enable_audit_logging: bool = True
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    
    # Integration settings
    orchestrator_url: str = "http://localhost:8000"
    finance_agent_url: str = "http://localhost:8002"
    
    # Notification settings
    enable_notifications: bool = True
    notification_batch_size: int = 100
    notification_retry_attempts: int = 3
    
    @validator('admin_emails', pre=True)
    def parse_admin_emails(cls, v):
        if isinstance(v, str):
            return [email.strip() for email in v.split(',')]
        return v
    
    @validator('mongodb_url')
    def validate_mongodb_url(cls, v):
        if not v.startswith('mongodb://') and not v.startswith('mongodb+srv://'):
            raise ValueError('MongoDB URL must start with mongodb:// or mongodb+srv://')
        return v
    
    class Config:
        env_file = ".env"
        env_prefix = "HR_"


class DevelopmentSettings(HRSettings):
    """Development environment settings."""
    debug: bool = True
    enable_ai_features: bool = False  # Disable AI in dev to avoid API costs


class ProductionSettings(HRSettings):
    """Production environment settings."""
    debug: bool = False
    enable_audit_logging: bool = True
    jwt_expiration_minutes: int = 15  # Shorter expiration in production


class TestingSettings(HRSettings):
    """Testing environment settings."""
    debug: bool = True
    mongodb_url: str = "mongodb://localhost:27017/enterprise_test"
    database_name: str = "enterprise_test"
    enable_notifications: bool = False
    enable_ai_features: bool = False


def get_settings() -> HRSettings:
    """Get settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Global settings instance
settings = get_settings()