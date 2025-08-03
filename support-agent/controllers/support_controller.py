"""Support controller with business logic."""

from typing import List, Optional, Dict, Any
from fastapi import HTTPException, status
from loguru import logger

from services.support_service import SupportService
from services.ai_support import AISupportService
from models.support import Ticket, TicketCreate


class SupportController:
    """Controller for support operations."""
    
    def __init__(self, support_service: SupportService, ai_support: AISupportService):
        self.support_service = support_service
        self.ai_support = ai_support
    
    async def create_ticket(self, ticket_data: dict, current_user: dict) -> dict:
        """Create ticket with AI analysis."""
        try:
            # AI analyze ticket urgency
            urgency_analysis = await self.ai_support.analyze_ticket_urgency(ticket_data)
            
            # Update ticket data with AI recommendations
            ticket_data["priority"] = urgency_analysis.get("recommended_priority", "medium")
            
            # Create ticket
            ticket = await self.support_service.create_ticket(TicketCreate(**ticket_data))
            if not ticket:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create ticket"
                )
            
            # Auto-escalate if needed
            if urgency_analysis.get("escalate", False):
                await self._auto_escalate_ticket(ticket.ticket_id, urgency_analysis)
            
            # Send auto-response if available
            auto_response = urgency_analysis.get("auto_response", {})
            if auto_response.get("send_immediately", False):
                await self._send_auto_response(ticket, auto_response)
            
            return {
                "message": "Ticket created successfully",
                "ticket_id": ticket.ticket_id,
                "priority": ticket.priority,
                "urgency_score": urgency_analysis.get("urgency_score", 0),
                "escalated": urgency_analysis.get("escalate", False),
                "sla_hours": urgency_analysis.get("sla_hours", 24)
            }
            
        except Exception as e:
            logger.error(f"Failed to create ticket: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def _auto_escalate_ticket(self, ticket_id: str, analysis: Dict[str, Any]) -> None:
        """Auto-escalate ticket based on AI analysis."""
        try:
            escalation_data = {
                "escalated": True,
                "escalated_at": datetime.utcnow(),
                "escalation_reason": "AI auto-escalation",
                "escalation_factors": analysis.get("urgency_factors", [])
            }
            
            await self.support_service.update_ticket(ticket_id, escalation_data)
            logger.info(f"Ticket auto-escalated: {ticket_id}")
            
        except Exception as e:
            logger.error(f"Failed to auto-escalate ticket: {e}")
    
    async def _send_auto_response(self, ticket: Ticket, auto_response: Dict[str, Any]) -> None:
        """Send automatic response to customer."""
        try:
            # In real implementation, this would send email
            logger.info(f"Auto-response sent for ticket {ticket.ticket_id}")
            
        except Exception as e:
            logger.error(f"Failed to send auto-response: {e}")
    
    async def list_tickets(self, status: Optional[str], priority: Optional[str], 
                          category: Optional[str], assigned_to: Optional[str],
                          page: int, limit: int, current_user: dict) -> List[dict]:
        """List tickets with filtering."""
        try:
            query = {}
            if status:
                query["status"] = status
            if priority:
                query["priority"] = priority
            if category:
                query["category"] = category
            if assigned_to:
                query["assigned_to"] = assigned_to
            
            skip = (page - 1) * limit
            
            tickets = await self.support_service.db[self.support_service.tickets_collection].find(
                query
            ).skip(skip).limit(limit).sort("created_at", -1).to_list(None)
            
            return [
                {
                    "ticket_id": ticket.get("ticket_id"),
                    "customer_name": ticket.get("customer_name"),
                    "subject": ticket.get("subject"),
                    "category": ticket.get("category"),
                    "priority": ticket.get("priority"),
                    "status": ticket.get("status"),
                    "assigned_to": ticket.get("assigned_to"),
                    "customer_tier": ticket.get("customer_tier"),
                    "created_at": ticket.get("created_at").isoformat() if ticket.get("created_at") else None
                } for ticket in tickets
            ]
            
        except Exception as e:
            logger.error(f"Failed to list tickets: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def get_ticket(self, ticket_id: str, current_user: dict) -> dict:
        """Get ticket details."""
        try:
            ticket = await self.support_service.get_ticket(ticket_id)
            if not ticket:
                raise HTTPException(status_code=404, detail="Ticket not found")
            
            # Get AI analysis if available
            analysis = await self.support_service.db[self.ai_support.ticket_analysis_collection].find_one(
                {"ticket_id": ticket_id}
            )
            
            return {
                "ticket_id": ticket.ticket_id,
                "customer_id": ticket.customer_id,
                "customer_email": ticket.customer_email,
                "customer_name": ticket.customer_name,
                "subject": ticket.subject,
                "description": ticket.description,
                "category": ticket.category,
                "priority": ticket.priority,
                "status": ticket.status,
                "assigned_to": ticket.assigned_to,
                "customer_tier": ticket.customer_tier,
                "tags": ticket.tags,
                "resolution": ticket.resolution,
                "satisfaction_score": ticket.satisfaction_score,
                "escalated": ticket.escalated,
                "ai_analysis": analysis,
                "created_at": ticket.created_at.isoformat(),
                "updated_at": ticket.updated_at.isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get ticket: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def update_ticket(self, ticket_id: str, update_data: dict, current_user: dict) -> dict:
        """Update ticket."""
        try:
            # Remove None values
            update_data = {k: v for k, v in update_data.items() if v is not None}
            
            if not update_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No valid update data provided"
                )
            
            success = await self.support_service.update_ticket(ticket_id, update_data, current_user["sub"])
            
            if not success:
                raise HTTPException(status_code=404, detail="Ticket not found")
            
            return {"message": "Ticket updated successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update ticket: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def escalate_ticket(self, ticket_id: str, escalation_reason: str, current_user: dict) -> dict:
        """Escalate ticket to higher support tier."""
        try:
            escalation_data = {
                "escalated": True,
                "escalated_at": datetime.utcnow(),
                "escalation_reason": escalation_reason,
                "escalated_by": current_user["sub"]
            }
            
            success = await self.support_service.update_ticket(ticket_id, escalation_data, current_user["sub"])
            
            if not success:
                raise HTTPException(status_code=404, detail="Ticket not found")
            
            return {"message": "Ticket escalated successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to escalate ticket: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )