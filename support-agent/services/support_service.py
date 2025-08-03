"""Support Service for managing customer support operations."""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger

from shared_libs.database import get_database
from models.support import Ticket, TicketCreate, KnowledgeArticle, FAQ


class SupportService:
    """Support service for managing tickets and knowledge base."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.tickets_collection = "tickets"
        self.knowledge_articles_collection = "knowledge_articles"
        self.faqs_collection = "faqs"
        self.chat_sessions_collection = "chat_sessions"
    
    async def initialize(self):
        """Initialize the Support service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.tickets_collection].create_index("ticket_id", unique=True)
        await self.db[self.tickets_collection].create_index("customer_email")
        await self.db[self.tickets_collection].create_index("status")
        await self.db[self.tickets_collection].create_index("priority")
        await self.db[self.tickets_collection].create_index("category")
        await self.db[self.tickets_collection].create_index("assigned_to")
        await self.db[self.tickets_collection].create_index("created_at")
        
        await self.db[self.knowledge_articles_collection].create_index("article_id", unique=True)
        await self.db[self.knowledge_articles_collection].create_index("category")
        await self.db[self.knowledge_articles_collection].create_index("tags")
        await self.db[self.knowledge_articles_collection].create_index("status")
        
        await self.db[self.faqs_collection].create_index("faq_id", unique=True)
        await self.db[self.faqs_collection].create_index("category")
        await self.db[self.faqs_collection].create_index("tags")
        
        await self.db[self.chat_sessions_collection].create_index("session_id", unique=True)
        await self.db[self.chat_sessions_collection].create_index("customer_id")
        
        logger.info("Support service initialized")
    
    async def create_ticket(self, ticket_data: TicketCreate) -> Optional[Ticket]:
        """Create a new support ticket."""
        try:
            ticket_id = f"TICK{str(uuid.uuid4())[:8].upper()}"
            customer_id = f"CUST{str(hash(ticket_data.customer_email))[:8].upper()}"
            
            ticket_dict = ticket_data.dict()
            ticket_dict.update({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            result = await self.db[self.tickets_collection].insert_one(ticket_dict)
            
            if result.inserted_id:
                ticket_dict["_id"] = result.inserted_id
                logger.info(f"Ticket created: {ticket_id}")
                return Ticket(**ticket_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create ticket: {e}")
            return None
    
    async def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Get a ticket by ID."""
        try:
            ticket_doc = await self.db[self.tickets_collection].find_one({"ticket_id": ticket_id})
            
            if ticket_doc:
                return Ticket(**ticket_doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get ticket {ticket_id}: {e}")
            return None
    
    async def update_ticket(self, ticket_id: str, update_data: Dict[str, Any], updated_by: str) -> bool:
        """Update a ticket."""
        try:
            update_data["updated_at"] = datetime.utcnow()
            update_data["updated_by"] = updated_by
            
            # Calculate resolution time if ticket is being resolved
            if update_data.get("status") == "resolved" and "resolution_time" not in update_data:
                ticket = await self.get_ticket(ticket_id)
                if ticket:
                    resolution_time = (datetime.utcnow() - ticket.created_at).total_seconds() / 60  # minutes
                    update_data["resolution_time"] = int(resolution_time)
            
            result = await self.db[self.tickets_collection].update_one(
                {"ticket_id": ticket_id},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Ticket updated: {ticket_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update ticket {ticket_id}: {e}")
            return False
    
    async def get_customer_tickets(self, customer_email: str) -> List[Ticket]:
        """Get all tickets for a customer."""
        try:
            tickets = []
            cursor = self.db[self.tickets_collection].find({"customer_email": customer_email}).sort("created_at", -1)
            
            async for ticket_doc in cursor:
                tickets.append(Ticket(**ticket_doc))
            
            return tickets
            
        except Exception as e:
            logger.error(f"Failed to get customer tickets for {customer_email}: {e}")
            return []
    
    async def get_support_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get support metrics for specified period."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Total tickets
            total_tickets = await self.db[self.tickets_collection].count_documents({
                "created_at": {"$gte": start_date}
            })
            
            # Resolved tickets
            resolved_tickets = await self.db[self.tickets_collection].count_documents({
                "created_at": {"$gte": start_date},
                "status": "resolved"
            })
            
            # Average resolution time
            resolved_cursor = self.db[self.tickets_collection].find({
                "created_at": {"$gte": start_date},
                "status": "resolved",
                "resolution_time": {"$exists": True}
            })
            
            resolution_times = []
            async for ticket in resolved_cursor:
                resolution_times.append(ticket.get("resolution_time", 0))
            
            avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
            
            # First response time
            first_response_cursor = self.db[self.tickets_collection].find({
                "created_at": {"$gte": start_date},
                "first_response_time": {"$exists": True}
            })
            
            first_response_times = []
            async for ticket in first_response_cursor:
                first_response_times.append(ticket.get("first_response_time", 0))
            
            avg_first_response = sum(first_response_times) / len(first_response_times) if first_response_times else 0
            
            # Customer satisfaction
            satisfaction_cursor = self.db[self.tickets_collection].find({
                "created_at": {"$gte": start_date},
                "satisfaction_score": {"$exists": True}
            })
            
            satisfaction_scores = []
            async for ticket in satisfaction_cursor:
                satisfaction_scores.append(ticket.get("satisfaction_score", 0))
            
            avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0
            
            return {
                "period_days": days,
                "total_tickets": total_tickets,
                "resolved_tickets": resolved_tickets,
                "resolution_rate": round((resolved_tickets / total_tickets) * 100, 2) if total_tickets > 0 else 0,
                "avg_resolution_time_minutes": round(avg_resolution_time, 2),
                "avg_first_response_minutes": round(avg_first_response, 2),
                "avg_satisfaction_score": round(avg_satisfaction, 2),
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get support metrics: {e}")
            return {}