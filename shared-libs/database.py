"""Database utilities shared across all services."""

import os
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
from loguru import logger


class DatabaseManager:
    """Database connection manager."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.database_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/enterprise")
    
    async def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(self.database_url)
            # Test connection
            await self.client.admin.command('ping')
            
            # Extract database name from URL
            db_name = self.database_url.split('/')[-1].split('?')[0]
            self.database = self.client[db_name]
            
            logger.info(f"Connected to MongoDB: {db_name}")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    def get_database(self) -> AsyncIOMotorDatabase:
        """Get database instance."""
        if not self.database:
            raise RuntimeError("Database not connected")
        return self.database


# Global database manager instance
db_manager = DatabaseManager()


async def get_database() -> AsyncIOMotorDatabase:
    """Dependency to get database instance."""
    return db_manager.get_database()
