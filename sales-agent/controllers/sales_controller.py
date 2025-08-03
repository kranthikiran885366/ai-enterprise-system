"""Sales controller with business logic."""

from typing import List, Optional, Dict, Any
from fastapi import HTTPException, status
from loguru import logger

from services.sales_service import SalesService
from services.ai_sales import AISalesService
from models.sales import Lead, Deal


class SalesController:
    """Controller for sales operations."""
    
    def __init__(self, sales_service: SalesService, ai_sales: AISalesService):
        self.sales_service = sales_service
        self.ai_sales = ai_sales
    
    async def create_lead(self, lead_data: dict, current_user: dict) -> dict:
        """Create lead with AI scoring."""
        try:
            # Create lead
            lead = await self.sales_service.create_lead(LeadCreate(**lead_data))
            if not lead:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create lead"
                )
            
            # AI score the lead
            scoring_result = await self.ai_sales.score_lead(lead.lead_id)
            
            # Auto-assign based on score
            if scoring_result.get("lead_score", 0) > 0.8:
                await self._auto_assign_lead(lead.lead_id, "senior_sales")
            elif scoring_result.get("lead_score", 0) > 0.6:
                await self._auto_assign_lead(lead.lead_id, "mid_level_sales")
            
            return {
                "message": "Lead created successfully",
                "lead_id": lead.lead_id,
                "ai_score": scoring_result.get("lead_score", 0),
                "grade": scoring_result.get("grade", "D")
            }
            
        except Exception as e:
            logger.error(f"Failed to create lead: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def _auto_assign_lead(self, lead_id: str, team: str) -> None:
        """Auto-assign lead to appropriate team."""
        # In real implementation, this would query available sales reps
        assignment_map = {
            "senior_sales": "senior_rep@company.com",
            "mid_level_sales": "mid_rep@company.com",
            "junior_sales": "junior_rep@company.com"
        }
        
        assigned_to = assignment_map.get(team, "default_rep@company.com")
        
        await self.sales_service.db[self.sales_service.leads_collection].update_one(
            {"lead_id": lead_id},
            {"$set": {"assigned_to": assigned_to}}
        )
    
    async def score_lead(self, lead_id: str, current_user: dict) -> dict:
        """Score lead using AI."""
        try:
            lead = await self.sales_service.get_lead(lead_id)
            if not lead:
                raise HTTPException(status_code=404, detail="Lead not found")
            
            scoring_result = await self.ai_sales.score_lead(lead_id)
            
            return {
                "lead_id": lead_id,
                "scoring_result": scoring_result
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to score lead: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def list_leads(self, status: Optional[str], source: Optional[str], 
                        assigned_to: Optional[str], page: int, limit: int, 
                        current_user: dict) -> List[dict]:
        """List leads with filtering."""
        try:
            query = {}
            if status:
                query["status"] = status
            if source:
                query["source"] = source
            if assigned_to:
                query["assigned_to"] = assigned_to
            
            skip = (page - 1) * limit
            
            leads = await self.sales_service.db[self.sales_service.leads_collection].find(
                query
            ).skip(skip).limit(limit).to_list(None)
            
            return [
                {
                    "lead_id": lead.get("lead_id"),
                    "company_name": lead.get("company_name"),
                    "contact_name": lead.get("contact_name"),
                    "contact_email": lead.get("contact_email"),
                    "status": lead.get("status"),
                    "source": lead.get("source"),
                    "ai_score": lead.get("ai_score"),
                    "assigned_to": lead.get("assigned_to"),
                    "created_at": lead.get("created_at").isoformat() if lead.get("created_at") else None
                } for lead in leads
            ]
            
        except Exception as e:
            logger.error(f"Failed to list leads: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def get_lead(self, lead_id: str, current_user: dict) -> dict:
        """Get lead details."""
        try:
            lead = await self.sales_service.get_lead(lead_id)
            if not lead:
                raise HTTPException(status_code=404, detail="Lead not found")
            
            return {
                "lead_id": lead.lead_id,
                "company_name": lead.company_name,
                "contact_name": lead.contact_name,
                "contact_email": lead.contact_email,
                "contact_phone": lead.contact_phone,
                "job_title": lead.job_title,
                "company_size": lead.company_size,
                "industry": lead.industry,
                "budget": lead.budget,
                "source": lead.source,
                "status": lead.status,
                "assigned_to": lead.assigned_to,
                "ai_score": lead.ai_score,
                "notes": lead.notes,
                "tags": lead.tags,
                "last_contacted": lead.last_contacted.isoformat() if lead.last_contacted else None,
                "created_at": lead.created_at.isoformat(),
                "updated_at": lead.updated_at.isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get lead: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def update_lead_status(self, lead_id: str, status: str, current_user: dict) -> dict:
        """Update lead status."""
        try:
            success = await self.sales_service.update_lead_status(lead_id, status, current_user["sub"])
            
            if not success:
                raise HTTPException(status_code=404, detail="Lead not found")
            
            return {"message": "Lead status updated successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update lead status: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def create_deal(self, deal_data: dict, current_user: dict) -> dict:
        """Create deal with AI probability calculation."""
        try:
            # Create deal
            deal = await self.sales_service.create_deal(DealCreate(**deal_data))
            if not deal:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create deal"
                )
            
            # Calculate AI probability
            probability_result = await self.ai_sales.calculate_deal_probability(deal.deal_id)
            
            # Update deal with AI probability
            if probability_result.get("probability"):
                await self.sales_service.update_deal_stage(
                    deal.deal_id, 
                    deal.stage, 
                    probability_result["probability"], 
                    "ai_system"
                )
            
            return {
                "message": "Deal created successfully",
                "deal_id": deal.deal_id,
                "ai_probability": probability_result.get("probability", 0),
                "risk_factors": probability_result.get("risk_factors", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to create deal: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def list_deals(self, stage: Optional[str], assigned_to: Optional[str], 
                        page: int, limit: int, current_user: dict) -> List[dict]:
        """List deals with filtering."""
        try:
            query = {}
            if stage:
                query["stage"] = stage
            if assigned_to:
                query["assigned_to"] = assigned_to
            
            skip = (page - 1) * limit
            
            deals = await self.sales_service.db[self.sales_service.deals_collection].find(
                query
            ).skip(skip).limit(limit).to_list(None)
            
            return [
                {
                    "deal_id": deal.get("deal_id"),
                    "company_name": deal.get("company_name"),
                    "contact_name": deal.get("contact_name"),
                    "deal_value": deal.get("deal_value"),
                    "currency": deal.get("currency"),
                    "stage": deal.get("stage"),
                    "probability": deal.get("probability"),
                    "expected_close_date": deal.get("expected_close_date").isoformat() if deal.get("expected_close_date") else None,
                    "assigned_to": deal.get("assigned_to"),
                    "created_at": deal.get("created_at").isoformat() if deal.get("created_at") else None
                } for deal in deals
            ]
            
        except Exception as e:
            logger.error(f"Failed to list deals: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )