"""Service Registry for managing microservices."""

from datetime import datetime, timedelta
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger

from shared_libs.database import get_database
from shared_libs.models import ServiceRegistration


class ServiceRegistry:
    """Service registry for managing microservices."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.collection_name = "service_registry"
    
    async def initialize(self):
        """Initialize the service registry."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.collection_name].create_index("name", unique=True)
        await self.db[self.collection_name].create_index("status")
        
        logger.info("Service registry initialized")
    
    async def register_service(self, service: ServiceRegistration) -> bool:
        """Register a new service."""
        try:
            service_dict = service.dict()
            service_dict["last_heartbeat"] = datetime.utcnow()
            
            # Upsert service registration
            result = await self.db[self.collection_name].update_one(
                {"name": service.name},
                {"$set": service_dict},
                upsert=True
            )
            
            logger.info(f"Service registered: {service.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service.name}: {e}")
            return False
    
    async def unregister_service(self, service_name: str) -> bool:
        """Unregister a service."""
        try:
            result = await self.db[self.collection_name].delete_one({"name": service_name})
            
            if result.deleted_count > 0:
                logger.info(f"Service unregistered: {service_name}")
                return True
            else:
                logger.warning(f"Service not found for unregistration: {service_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister service {service_name}: {e}")
            return False
    
    async def update_heartbeat(self, service_name: str) -> bool:
        """Update service heartbeat."""
        try:
            result = await self.db[self.collection_name].update_one(
                {"name": service_name},
                {"$set": {"last_heartbeat": datetime.utcnow(), "status": "active"}}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for {service_name}: {e}")
            return False
    
    async def get_service(self, service_name: str) -> Optional[ServiceRegistration]:
        """Get a specific service."""
        try:
            service_doc = await self.db[self.collection_name].find_one({"name": service_name})
            
            if service_doc:
                return ServiceRegistration(**service_doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get service {service_name}: {e}")
            return None
    
    async def get_all_services(self) -> List[ServiceRegistration]:
        """Get all registered services."""
        try:
            services = []
            cursor = self.db[self.collection_name].find({})
            
            async for service_doc in cursor:
                services.append(ServiceRegistration(**service_doc))
            
            return services
            
        except Exception as e:
            logger.error(f"Failed to get all services: {e}")
            return []
    
    async def get_healthy_services(self) -> List[ServiceRegistration]:
        """Get all healthy services."""
        try:
            # Consider services healthy if they had a heartbeat in the last 5 minutes
            cutoff_time = datetime.utcnow() - timedelta(minutes=5)
            
            services = []
            cursor = self.db[self.collection_name].find({
                "status": "active",
                "last_heartbeat": {"$gte": cutoff_time}
            })
            
            async for service_doc in cursor:
                services.append(ServiceRegistration(**service_doc))
            
            return services
            
        except Exception as e:
            logger.error(f"Failed to get healthy services: {e}")
            return []
    
    async def cleanup_stale_services(self):
        """Remove services that haven't sent heartbeat in a while."""
        try:
            # Mark services as inactive if no heartbeat for 10 minutes
            cutoff_time = datetime.utcnow() - timedelta(minutes=10)
            
            result = await self.db[self.collection_name].update_many(
                {"last_heartbeat": {"$lt": cutoff_time}},
                {"$set": {"status": "inactive"}}
            )
            
            if result.modified_count > 0:
                logger.info(f"Marked {result.modified_count} services as inactive")
            
        except Exception as e:
            logger.error(f"Failed to cleanup stale services: {e}")
