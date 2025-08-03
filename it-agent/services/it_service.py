"""IT Service for managing infrastructure and assets."""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger

from shared_libs.database import get_database
from models.infrastructure import Asset, AssetCreate, ITTicket, ITTicketCreate, NetworkDevice, SoftwareLicense


class ITService:
    """IT service for managing assets, infrastructure, and support."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.assets_collection = "it_assets"
        self.tickets_collection = "it_tickets"
        self.network_devices_collection = "network_devices"
        self.software_licenses_collection = "software_licenses"
        self.security_incidents_collection = "security_incidents"
    
    async def initialize(self):
        """Initialize the IT service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.assets_collection].create_index("asset_id", unique=True)
        await self.db[self.assets_collection].create_index("asset_tag", unique=True)
        await self.db[self.assets_collection].create_index("assigned_to")
        await self.db[self.assets_collection].create_index("status")
        await self.db[self.assets_collection].create_index("asset_type")
        
        await self.db[self.tickets_collection].create_index("ticket_id", unique=True)
        await self.db[self.tickets_collection].create_index("employee_email")
        await self.db[self.tickets_collection].create_index("status")
        await self.db[self.tickets_collection].create_index("priority")
        
        await self.db[self.network_devices_collection].create_index("device_id", unique=True)
        await self.db[self.network_devices_collection].create_index("ip_address", unique=True)
        await self.db[self.network_devices_collection].create_index("status")
        
        await self.db[self.software_licenses_collection].create_index("license_id", unique=True)
        await self.db[self.software_licenses_collection].create_index("software_name")
        await self.db[self.software_licenses_collection].create_index("expiry_date")
        
        await self.db[self.security_incidents_collection].create_index("incident_id", unique=True)
        await self.db[self.security_incidents_collection].create_index("severity")
        await self.db[self.security_incidents_collection].create_index("status")
        
        logger.info("IT service initialized")
    
    async def create_asset(self, asset_data: AssetCreate) -> Optional[Asset]:
        """Create a new IT asset."""
        try:
            asset_id = f"AST{str(uuid.uuid4())[:8].upper()}"
            asset_tag = f"TAG{str(uuid.uuid4())[:6].upper()}"
            
            asset_dict = asset_data.dict()
            asset_dict.update({
                "asset_id": asset_id,
                "asset_tag": asset_tag,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            result = await self.db[self.assets_collection].insert_one(asset_dict)
            
            if result.inserted_id:
                asset_dict["_id"] = result.inserted_id
                logger.info(f"Asset created: {asset_id}")
                return Asset(**asset_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create asset: {e}")
            return None
    
    async def assign_asset(self, asset_id: str, employee_id: str) -> bool:
        """Assign asset to employee."""
        try:
            result = await self.db[self.assets_collection].update_one(
                {"asset_id": asset_id},
                {
                    "$set": {
                        "assigned_to": employee_id,
                        "status": "active",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Asset assigned: {asset_id} -> {employee_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to assign asset {asset_id}: {e}")
            return False
    
    async def create_it_ticket(self, ticket_data: ITTicketCreate) -> Optional[ITTicket]:
        """Create a new IT support ticket."""
        try:
            ticket_id = f"IT{str(uuid.uuid4())[:8].upper()}"
            employee_id = f"EMP{str(hash(ticket_data.employee_email))[:8].upper()}"
            
            ticket_dict = ticket_data.dict()
            ticket_dict.update({
                "ticket_id": ticket_id,
                "employee_id": employee_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            result = await self.db[self.tickets_collection].insert_one(ticket_dict)
            
            if result.inserted_id:
                ticket_dict["_id"] = result.inserted_id
                logger.info(f"IT ticket created: {ticket_id}")
                return ITTicket(**ticket_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create IT ticket: {e}")
            return None
    
    async def monitor_network_devices(self) -> Dict[str, Any]:
        """Monitor network device health."""
        try:
            devices = await self.db[self.network_devices_collection].find({}).to_list(None)
            
            monitoring_results = {
                "total_devices": len(devices),
                "active_devices": 0,
                "inactive_devices": 0,
                "devices_needing_attention": [],
                "overall_health": "healthy"
            }
            
            for device in devices:
                device_id = device.get("device_id")
                last_ping = device.get("last_ping")
                
                # Simulate ping check (in real implementation, would actually ping)
                if last_ping and (datetime.utcnow() - last_ping).total_seconds() < 300:  # 5 minutes
                    monitoring_results["active_devices"] += 1
                    # Update uptime
                    await self.db[self.network_devices_collection].update_one(
                        {"device_id": device_id},
                        {"$set": {"status": "active", "last_ping": datetime.utcnow()}}
                    )
                else:
                    monitoring_results["inactive_devices"] += 1
                    monitoring_results["devices_needing_attention"].append({
                        "device_id": device_id,
                        "device_name": device.get("device_name"),
                        "issue": "No response to ping",
                        "last_seen": last_ping.isoformat() if last_ping else "Never"
                    })
                    
                    # Update status
                    await self.db[self.network_devices_collection].update_one(
                        {"device_id": device_id},
                        {"$set": {"status": "inactive"}}
                    )
            
            # Determine overall health
            if monitoring_results["inactive_devices"] > monitoring_results["total_devices"] * 0.1:
                monitoring_results["overall_health"] = "degraded"
            if monitoring_results["inactive_devices"] > monitoring_results["total_devices"] * 0.2:
                monitoring_results["overall_health"] = "critical"
            
            return monitoring_results
            
        except Exception as e:
            logger.error(f"Failed to monitor network devices: {e}")
            return {"overall_health": "unknown"}
    
    async def get_asset_utilization(self) -> Dict[str, Any]:
        """Get asset utilization statistics."""
        try:
            # Asset type distribution
            pipeline = [
                {"$group": {
                    "_id": "$asset_type",
                    "count": {"$sum": 1},
                    "active": {"$sum": {"$cond": [{"$eq": ["$status", "active"]}, 1, 0]}},
                    "total_cost": {"$sum": "$purchase_cost"}
                }}
            ]
            
            asset_stats = {}
            cursor = self.db[self.assets_collection].aggregate(pipeline)
            async for stat in cursor:
                asset_type = stat["_id"]
                asset_stats[asset_type] = {
                    "total": stat["count"],
                    "active": stat["active"],
                    "utilization_rate": round((stat["active"] / stat["count"]) * 100, 2),
                    "total_cost": stat.get("total_cost", 0)
                }
            
            # Assets needing maintenance
            maintenance_due = await self.db[self.assets_collection].find({
                "warranty_expiry": {"$lte": datetime.utcnow() + timedelta(days=30)},
                "status": "active"
            }).to_list(None)
            
            # Unassigned assets
            unassigned = await self.db[self.assets_collection].count_documents({
                "assigned_to": None,
                "status": "active"
            })
            
            return {
                "asset_statistics": asset_stats,
                "maintenance_due": len(maintenance_due),
                "unassigned_assets": unassigned,
                "total_assets": sum(stat["total"] for stat in asset_stats.values()),
                "total_active": sum(stat["active"] for stat in asset_stats.values()),
                "overall_utilization": round(
                    (sum(stat["active"] for stat in asset_stats.values()) / 
                     sum(stat["total"] for stat in asset_stats.values())) * 100, 2
                ) if asset_stats else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get asset utilization: {e}")
            return {}