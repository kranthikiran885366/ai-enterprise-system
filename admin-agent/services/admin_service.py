"""Admin Service for managing administrative operations."""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger

from shared_libs.database import get_database
from models.admin import Announcement, AnnouncementCreate, Policy, PolicyCreate, Permission, SystemNotification


class AdminService:
    """Admin service for managing announcements, policies, and permissions."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.announcements_collection = "announcements"
        self.policies_collection = "policies"
        self.permissions_collection = "permissions"
        self.notifications_collection = "system_notifications"
        self.user_roles_collection = "user_roles"
    
    async def initialize(self):
        """Initialize the Admin service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.announcements_collection].create_index("announcement_id", unique=True)
        await self.db[self.announcements_collection].create_index("announcement_type")
        await self.db[self.announcements_collection].create_index("priority")
        await self.db[self.announcements_collection].create_index("published")
        await self.db[self.announcements_collection].create_index("publish_date")
        
        await self.db[self.policies_collection].create_index("policy_id", unique=True)
        await self.db[self.policies_collection].create_index("policy_type")
        await self.db[self.policies_collection].create_index("status")
        await self.db[self.policies_collection].create_index("effective_date")
        
        await self.db[self.permissions_collection].create_index("permission_id", unique=True)
        await self.db[self.permissions_collection].create_index("user_id")
        await self.db[self.permissions_collection].create_index("resource")
        await self.db[self.permissions_collection].create_index("expires_at")
        
        await self.db[self.notifications_collection].create_index("notification_id", unique=True)
        await self.db[self.notifications_collection].create_index("recipient_id")
        await self.db[self.notifications_collection].create_index("sent")
        await self.db[self.notifications_collection].create_index("created_at")
        
        await self.db[self.user_roles_collection].create_index("role_id", unique=True)
        await self.db[self.user_roles_collection].create_index("role_name", unique=True)
        
        logger.info("Admin service initialized")
    
    async def create_announcement(self, announcement_data: AnnouncementCreate, author: str) -> Optional[Announcement]:
        """Create a new announcement."""
        try:
            announcement_id = f"ANN{str(uuid.uuid4())[:8].upper()}"
            
            announcement_dict = announcement_data.dict()
            announcement_dict.update({
                "announcement_id": announcement_id,
                "author": author,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "read_by": [],
                "acknowledged_by": []
            })
            
            result = await self.db[self.announcements_collection].insert_one(announcement_dict)
            
            if result.inserted_id:
                announcement_dict["_id"] = result.inserted_id
                logger.info(f"Announcement created: {announcement_id}")
                return Announcement(**announcement_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create announcement: {e}")
            return None
    
    async def publish_announcement(self, announcement_id: str) -> bool:
        """Publish an announcement."""
        try:
            result = await self.db[self.announcements_collection].update_one(
                {"announcement_id": announcement_id},
                {
                    "$set": {
                        "published": True,
                        "publish_date": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count > 0:
                # Create notifications for target audience
                await self._create_announcement_notifications(announcement_id)
                logger.info(f"Announcement published: {announcement_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to publish announcement {announcement_id}: {e}")
            return False
    
    async def _create_announcement_notifications(self, announcement_id: str) -> None:
        """Create notifications for announcement target audience."""
        try:
            announcement = await self.db[self.announcements_collection].find_one({"announcement_id": announcement_id})
            if not announcement:
                return
            
            target_audience = announcement.get("target_audience", [])
            
            # Get recipients based on target audience
            recipients = await self._get_announcement_recipients(target_audience)
            
            # Create notifications
            for recipient in recipients:
                notification = {
                    "notification_id": f"NOT{str(uuid.uuid4())[:8].upper()}",
                    "recipient_id": recipient.get("user_id"),
                    "recipient_email": recipient.get("email"),
                    "title": f"New Announcement: {announcement.get('title')}",
                    "message": announcement.get("content")[:200] + "..." if len(announcement.get("content", "")) > 200 else announcement.get("content", ""),
                    "notification_type": "in_app",
                    "priority": announcement.get("priority", "medium"),
                    "metadata": {
                        "announcement_id": announcement_id,
                        "announcement_type": announcement.get("announcement_type")
                    },
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                await self.db[self.notifications_collection].insert_one(notification)
            
            logger.info(f"Created {len(recipients)} notifications for announcement {announcement_id}")
            
        except Exception as e:
            logger.error(f"Failed to create announcement notifications: {e}")
    
    async def _get_announcement_recipients(self, target_audience: List[str]) -> List[Dict[str, Any]]:
        """Get recipients for announcement based on target audience."""
        try:
            recipients = []
            
            for target in target_audience:
                if target == "all":
                    # Get all employees
                    all_employees = await self.db["employees"].find({
                        "status": "active"
                    }).to_list(None)
                    
                    for emp in all_employees:
                        recipients.append({
                            "user_id": emp.get("employee_id"),
                            "email": emp.get("email")
                        })
                else:
                    # Get employees by department or role
                    dept_employees = await self.db["employees"].find({
                        "department": target,
                        "status": "active"
                    }).to_list(None)
                    
                    for emp in dept_employees:
                        recipients.append({
                            "user_id": emp.get("employee_id"),
                            "email": emp.get("email")
                        })
            
            # Remove duplicates
            unique_recipients = []
            seen_emails = set()
            
            for recipient in recipients:
                email = recipient.get("email")
                if email and email not in seen_emails:
                    unique_recipients.append(recipient)
                    seen_emails.add(email)
            
            return unique_recipients
            
        except Exception as e:
            logger.error(f"Failed to get announcement recipients: {e}")
            return []
    
    async def create_policy(self, policy_data: PolicyCreate, author: str) -> Optional[Policy]:
        """Create a new company policy."""
        try:
            policy_id = f"POL{str(uuid.uuid4())[:8].upper()}"
            
            policy_dict = policy_data.dict()
            policy_dict.update({
                "policy_id": policy_id,
                "author": author,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            result = await self.db[self.policies_collection].insert_one(policy_dict)
            
            if result.inserted_id:
                policy_dict["_id"] = result.inserted_id
                logger.info(f"Policy created: {policy_id}")
                return Policy(**policy_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create policy: {e}")
            return None
    
    async def get_active_policies(self, department: Optional[str] = None) -> List[Policy]:
        """Get active policies, optionally filtered by department."""
        try:
            query = {"status": "active"}
            if department:
                query["$or"] = [
                    {"applicable_departments": department},
                    {"applicable_departments": "all"}
                ]
            
            policies = []
            cursor = self.db[self.policies_collection].find(query).sort("effective_date", -1)
            
            async for policy_doc in cursor:
                policies.append(Policy(**policy_doc))
            
            return policies
            
        except Exception as e:
            logger.error(f"Failed to get active policies: {e}")
            return []
    
    async def grant_permission(self, user_id: str, resource: str, permission_level: str, 
                             granted_by: str, expires_at: Optional[datetime] = None) -> bool:
        """Grant permission to a user."""
        try:
            permission_id = f"PERM{str(uuid.uuid4())[:8].upper()}"
            
            permission = {
                "permission_id": permission_id,
                "user_id": user_id,
                "resource": resource,
                "permission_level": permission_level,
                "granted_by": granted_by,
                "granted_at": datetime.utcnow(),
                "expires_at": expires_at,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await self.db[self.permissions_collection].insert_one(permission)
            
            if result.inserted_id:
                logger.info(f"Permission granted: {user_id} -> {resource} ({permission_level})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to grant permission: {e}")
            return False
    
    async def check_user_permission(self, user_id: str, resource: str, required_level: str) -> bool:
        """Check if user has required permission for resource."""
        try:
            permission = await self.db[self.permissions_collection].find_one({
                "user_id": user_id,
                "resource": resource,
                "$or": [
                    {"expires_at": None},
                    {"expires_at": {"$gt": datetime.utcnow()}}
                ]
            })
            
            if not permission:
                return False
            
            # Check permission level hierarchy
            level_hierarchy = {"read": 1, "write": 2, "admin": 3, "super_admin": 4}
            
            user_level = level_hierarchy.get(permission.get("permission_level", "read"), 0)
            required_level_num = level_hierarchy.get(required_level, 0)
            
            return user_level >= required_level_num
            
        except Exception as e:
            logger.error(f"Failed to check user permission: {e}")
            return False
    
    async def send_system_notification(self, recipient_id: str, recipient_email: str, 
                                     title: str, message: str, notification_type: str = "in_app",
                                     priority: str = "medium", metadata: Dict[str, Any] = None) -> bool:
        """Send system notification to user."""
        try:
            notification_id = f"NOT{str(uuid.uuid4())[:8].upper()}"
            
            notification = {
                "notification_id": notification_id,
                "recipient_id": recipient_id,
                "recipient_email": recipient_email,
                "title": title,
                "message": message,
                "notification_type": notification_type,
                "priority": priority,
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await self.db[self.notifications_collection].insert_one(notification)
            
            if result.inserted_id:
                # In real implementation, would actually send notification
                logger.info(f"System notification created: {notification_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send system notification: {e}")
            return False
    
    async def get_user_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        """Get notifications for a user."""
        try:
            query = {"recipient_id": user_id}
            if unread_only:
                query["read"] = False
            
            notifications = []
            cursor = self.db[self.notifications_collection].find(query).sort("created_at", -1).limit(50)
            
            async for notification in cursor:
                notifications.append({
                    "notification_id": notification.get("notification_id"),
                    "title": notification.get("title"),
                    "message": notification.get("message"),
                    "notification_type": notification.get("notification_type"),
                    "priority": notification.get("priority"),
                    "read": notification.get("read", False),
                    "created_at": notification.get("created_at").isoformat() if notification.get("created_at") else None
                })
            
            return notifications
            
        except Exception as e:
            logger.error(f"Failed to get user notifications: {e}")
            return []