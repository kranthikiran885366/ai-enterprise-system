import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI Enterprise System"
    app_version: str = "2.0.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"

    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "enterprise-ai-super-secret-key-2025-change-in-prod")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    jwt_refresh_token_expire_days: int = 7

    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./enterprise.db")
    db_file: str = os.path.join(os.path.dirname(__file__), "enterprise.db")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    cors_origins: list = ["*"]
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.path.join(os.path.dirname(__file__), "logs", "enterprise.log")

    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")

    default_admin_username: str = "admin"
    default_admin_password: str = "admin123"
    default_admin_email: str = "admin@enterprise.ai"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
