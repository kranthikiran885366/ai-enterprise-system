"""Database configuration for HR Agent."""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
from loguru import logger
from typing import Optional

from config.settings import settings


class HRDatabaseManager:
    """HR-specific database manager."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(settings.mongodb_url)
            
            # Test connection
            await self.client.admin.command('ping')
            
            # Get database
            self.database = self.client[settings.database_name]
            
            # Create indexes
            await self._create_indexes()
            
            logger.info(f"Connected to MongoDB: {settings.database_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self):
        """Create database indexes for HR collections."""
        try:
            # Employee indexes
            await self.database.employees.create_index("employee_id", unique=True)
            await self.database.employees.create_index("email", unique=True)
            await self.database.employees.create_index([("department", 1), ("status", 1)])
            await self.database.employees.create_index("manager_id")
            await self.database.employees.create_index("hire_date")
            
            # Attendance indexes
            await self.database.attendance_records.create_index([("employee_id", 1), ("date", -1)])
            await self.database.attendance_records.create_index("date")
            
            # Leave request indexes
            await self.database.leave_requests.create_index("employee_id")
            await self.database.leave_requests.create_index([("status", 1), ("start_date", 1)])
            
            # Job posting indexes
            await self.database.job_postings.create_index("job_id", unique=True)
            await self.database.job_postings.create_index([("department", 1), ("status", 1)])
            await self.database.job_postings.create_index("created_at")
            
            # Application indexes
            await self.database.job_applications.create_index("application_id", unique=True)
            await self.database.job_applications.create_index("job_id")
            await self.database.job_applications.create_index("candidate_email")
            
            # Performance review indexes
            await self.database.performance_reviews.create_index("employee_id")
            await self.database.performance_reviews.create_index("review_date")
            
            # Audit log indexes
            await self.database.audit_logs.create_index([("timestamp", -1)])
            await self.database.audit_logs.create_index("user_id")
            await self.database.audit_logs.create_index("operation")
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database indexes: {e}")
            raise
    
    def get_database(self) -> AsyncIOMotorDatabase:
        """Get database instance."""
        if not self.database:
            raise RuntimeError("Database not connected")
        return self.database
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            if not self.client:
                return False
            
            await self.client.admin.command('ping')
            return True
            
        except Exception:
            return False


# Global database manager instance
hr_db_manager = HRDatabaseManager()


async def get_hr_database() -> AsyncIOMotorDatabase:
    """Get HR database instance."""
    return hr_db_manager.get_database()