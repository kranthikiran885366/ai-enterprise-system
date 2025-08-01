"""Authentication service for the orchestrator."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger

from shared_libs.database import get_database
from shared_libs.auth import verify_password, get_password_hash, create_access_token
from shared_libs.models import BaseDocument


class User(BaseDocument):
    """User model."""
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    roles: list = []


class AuthService:
    """Authentication service."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.collection_name = "users"
    
    async def initialize(self):
        """Initialize the auth service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.collection_name].create_index("username", unique=True)
        await self.db[self.collection_name].create_index("email", unique=True)
        
        # Create default admin user if not exists
        await self.create_default_admin()
        
        logger.info("Auth service initialized")
    
    async def create_default_admin(self):
        """Create default admin user."""
        try:
            existing_admin = await self.db[self.collection_name].find_one({"username": "admin"})
            
            if not existing_admin:
                admin_user = {
                    "username": "admin",
                    "email": "admin@enterprise.ai",
                    "hashed_password": get_password_hash("admin123"),
                    "is_active": True,
                    "is_admin": True,
                    "roles": ["admin", "user"],
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                await self.db[self.collection_name].insert_one(admin_user)
                logger.info("Default admin user created")
                
        except Exception as e:
            logger.error(f"Failed to create default admin user: {e}")
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user."""
        try:
            user_doc = await self.db[self.collection_name].find_one({"username": username})
            
            if not user_doc:
                return None
            
            if not verify_password(password, user_doc["hashed_password"]):
                return None
            
            if not user_doc.get("is_active", True):
                return None
            
            # Remove sensitive data
            user_doc.pop("hashed_password", None)
            user_doc["_id"] = str(user_doc["_id"])
            
            return user_doc
            
        except Exception as e:
            logger.error(f"Failed to authenticate user {username}: {e}")
            return None
    
    async def create_user(self, username: str, email: str, password: str, is_admin: bool = False) -> bool:
        """Create a new user."""
        try:
            user_data = {
                "username": username,
                "email": email,
                "hashed_password": get_password_hash(password),
                "is_active": True,
                "is_admin": is_admin,
                "roles": ["admin", "user"] if is_admin else ["user"],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            await self.db[self.collection_name].insert_one(user_data)
            logger.info(f"User created: {username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            return False
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        try:
            user_doc = await self.db[self.collection_name].find_one({"username": username})
            
            if user_doc:
                user_doc["_id"] = str(user_doc["_id"])
                user_doc.pop("hashed_password", None)
                return user_doc
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user {username}: {e}")
            return None
