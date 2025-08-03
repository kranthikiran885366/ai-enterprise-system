"""Sales Service for managing sales operations."""

import uuid
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger

from shared_libs.database import get_database
from models.sales import Lead, LeadCreate, Deal, DealCreate, SalesActivity, ActivityCreate


class SalesService:
    """Sales service for managing leads, deals, and activities."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.leads_collection = "leads"
        self.deals_collection = "deals"
        self.activities_collection = "sales_activities"
        self.targets_collection = "sales_targets"
        self.forecasts_collection = "sales_forecasts"
    
    async def initialize(self):
        """Initialize the Sales service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.leads_collection].create_index("lead_id", unique=True)
        await self.db[self.leads_collection].create_index("contact_email", unique=True)
        await self.db[self.leads_collection].create_index("status")
        await self.db[self.leads_collection].create_index("assigned_to")
        await self.db[self.leads_collection].create_index("ai_score")
        await self.db[self.leads_collection].create_index("source")
        
        await self.db[self.deals_collection].create_index("deal_id", unique=True)
        await self.db[self.deals_collection].create_index("lead_id")
        await self.db[self.deals_collection].create_index("stage")
        await self.db[self.deals_collection].create_index("assigned_to")
        await self.db[self.deals_collection].create_index("expected_close_date")
        
        await self.db[self.activities_collection].create_index("activity_id", unique=True)
        await self.db[self.activities_collection].create_index("lead_id")
        await self.db[self.activities_collection].create_index("deal_id")
        await self.db[self.activities_collection].create_index("assigned_to")
        await self.db[self.activities_collection].create_index("scheduled_at")
        
        await self.db[self.targets_collection].create_index("target_id", unique=True)
        await self.db[self.targets_collection].create_index("sales_rep")
        await self.db[self.targets_collection].create_index([("year", 1), ("month", 1)])
        
        logger.info("Sales service initialized")
    
    async def create_lead(self, lead_data: LeadCreate) -> Optional[Lead]:
        """Create a new lead."""
        try:
            lead_id = f"LEAD{str(uuid.uuid4())[:8].upper()}"
            
            lead_dict = lead_data.dict()
            lead_dict["lead_id"] = lead_id
            lead_dict["created_at"] = datetime.utcnow()
            lead_dict["updated_at"] = datetime.utcnow()
            
            # Convert notes string to notes list
            if "notes" in lead_dict and isinstance(lead_dict["notes"], str):
                lead_dict["notes"] = [{
                    "note": lead_dict["notes"],
                    "created_at": datetime.utcnow(),
                    "created_by": "system"
                }]
            elif "notes" not in lead_dict:
                lead_dict["notes"] = []
            
            result = await self.db[self.leads_collection].insert_one(lead_dict)
            
            if result.inserted_id:
                lead_dict["_id"] = result.inserted_id
                logger.info(f"Lead created: {lead_id}")
                return Lead(**lead_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create lead: {e}")
            return None
    
    async def get_lead(self, lead_id: str) -> Optional[Lead]:
        """Get a lead by ID."""
        try:
            lead_doc = await self.db[self.leads_collection].find_one({"lead_id": lead_id})
            
            if lead_doc:
                return Lead(**lead_doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get lead {lead_id}: {e}")
            return None
    
    async def update_lead_status(self, lead_id: str, status: str, updated_by: str) -> bool:
        """Update lead status."""
        try:
            result = await self.db[self.leads_collection].update_one(
                {"lead_id": lead_id},
                {
                    "$set": {
                        "status": status,
                        "updated_at": datetime.utcnow(),
                        "updated_by": updated_by
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Lead status updated: {lead_id} -> {status}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update lead status {lead_id}: {e}")
            return False
    
    async def create_deal(self, deal_data: DealCreate) -> Optional[Deal]:
        """Create a new deal."""
        try:
            deal_id = f"DEAL{str(uuid.uuid4())[:8].upper()}"
            
            deal_dict = deal_data.dict()
            deal_dict["deal_id"] = deal_id
            deal_dict["created_at"] = datetime.utcnow()
            deal_dict["updated_at"] = datetime.utcnow()
            
            # Convert notes string to notes list
            if "notes" in deal_dict and isinstance(deal_dict["notes"], str):
                deal_dict["notes"] = [{
                    "note": deal_dict["notes"],
                    "created_at": datetime.utcnow(),
                    "created_by": "system"
                }]
            elif "notes" not in deal_dict:
                deal_dict["notes"] = []
            
            # Initialize activities list
            deal_dict["activities"] = []
            
            result = await self.db[self.deals_collection].insert_one(deal_dict)
            
            if result.inserted_id:
                deal_dict["_id"] = result.inserted_id
                logger.info(f"Deal created: {deal_id}")
                return Deal(**deal_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create deal: {e}")
            return None
    
    async def get_deal(self, deal_id: str) -> Optional[Deal]:
        """Get a deal by ID."""
        try:
            deal_doc = await self.db[self.deals_collection].find_one({"deal_id": deal_id})
            
            if deal_doc:
                return Deal(**deal_doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get deal {deal_id}: {e}")
            return None
    
    async def update_deal_stage(self, deal_id: str, stage: str, probability: float, updated_by: str) -> bool:
        """Update deal stage and probability."""
        try:
            update_data = {
                "stage": stage,
                "probability": probability,
                "updated_at": datetime.utcnow(),
                "updated_by": updated_by
            }
            
            # Set close date if deal is closed
            if stage in ["closed_won", "closed_lost"]:
                update_data["actual_close_date"] = date.today()
            
            result = await self.db[self.deals_collection].update_one(
                {"deal_id": deal_id},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Deal stage updated: {deal_id} -> {stage}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update deal stage {deal_id}: {e}")
            return False
    
    async def create_activity(self, activity_data: ActivityCreate) -> Optional[SalesActivity]:
        """Create a sales activity."""
        try:
            activity_id = f"ACT{str(uuid.uuid4())[:8].upper()}"
            
            activity_dict = activity_data.dict()
            activity_dict["activity_id"] = activity_id
            activity_dict["created_at"] = datetime.utcnow()
            activity_dict["updated_at"] = datetime.utcnow()
            
            result = await self.db[self.activities_collection].insert_one(activity_dict)
            
            if result.inserted_id:
                activity_dict["_id"] = result.inserted_id
                
                # Update related lead/deal with activity
                if activity_data.lead_id:
                    await self._add_activity_to_lead(activity_data.lead_id, activity_dict)
                if activity_data.deal_id:
                    await self._add_activity_to_deal(activity_data.deal_id, activity_dict)
                
                logger.info(f"Activity created: {activity_id}")
                return SalesActivity(**activity_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create activity: {e}")
            return None
    
    async def _add_activity_to_lead(self, lead_id: str, activity: Dict[str, Any]) -> None:
        """Add activity to lead's activity list."""
        try:
            await self.db[self.leads_collection].update_one(
                {"lead_id": lead_id},
                {
                    "$push": {"activities": {
                        "activity_id": activity["activity_id"],
                        "type": activity["activity_type"],
                        "subject": activity["subject"],
                        "created_at": activity["created_at"]
                    }},
                    "$set": {"last_contacted": datetime.utcnow()}
                }
            )
        except Exception as e:
            logger.error(f"Failed to add activity to lead {lead_id}: {e}")
    
    async def _add_activity_to_deal(self, deal_id: str, activity: Dict[str, Any]) -> None:
        """Add activity to deal's activity list."""
        try:
            await self.db[self.deals_collection].update_one(
                {"deal_id": deal_id},
                {
                    "$push": {"activities": {
                        "activity_id": activity["activity_id"],
                        "type": activity["activity_type"],
                        "subject": activity["subject"],
                        "created_at": activity["created_at"]
                    }}
                }
            )
        except Exception as e:
            logger.error(f"Failed to add activity to deal {deal_id}: {e}")
    
    async def get_sales_pipeline(self, sales_rep: Optional[str] = None) -> Dict[str, Any]:
        """Get sales pipeline overview."""
        try:
            query = {}
            if sales_rep:
                query["assigned_to"] = sales_rep
            
            # Get deals by stage
            pipeline = {}
            stages = ["prospecting", "qualification", "proposal", "negotiation"]
            
            for stage in stages:
                stage_query = {**query, "stage": stage}
                deals = await self.db[self.deals_collection].find(stage_query).to_list(None)
                
                total_value = sum(deal.get("deal_value", 0) for deal in deals)
                weighted_value = sum(
                    deal.get("deal_value", 0) * deal.get("probability", 0) / 100
                    for deal in deals
                )
                
                pipeline[stage] = {
                    "deal_count": len(deals),
                    "total_value": round(total_value, 2),
                    "weighted_value": round(weighted_value, 2),
                    "deals": deals
                }
            
            # Calculate totals
            total_deals = sum(stage["deal_count"] for stage in pipeline.values())
            total_value = sum(stage["total_value"] for stage in pipeline.values())
            total_weighted = sum(stage["weighted_value"] for stage in pipeline.values())
            
            return {
                "pipeline": pipeline,
                "totals": {
                    "total_deals": total_deals,
                    "total_value": round(total_value, 2),
                    "weighted_value": round(total_weighted, 2)
                },
                "sales_rep": sales_rep,
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get sales pipeline: {e}")
            return {}
    
    async def get_sales_metrics(self, period_days: int = 30, sales_rep: Optional[str] = None) -> Dict[str, Any]:
        """Get sales metrics for a period."""
        try:
            start_date = datetime.utcnow() - timedelta(days=period_days)
            query = {"created_at": {"$gte": start_date}}
            
            if sales_rep:
                query["assigned_to"] = sales_rep
            
            # Get leads metrics
            total_leads = await self.db[self.leads_collection].count_documents(query)
            qualified_leads = await self.db[self.leads_collection].count_documents({
                **query, "status": {"$in": ["qualified", "proposal", "negotiation"]}
            })
            
            # Get deals metrics
            total_deals = await self.db[self.deals_collection].count_documents(query)
            won_deals = await self.db[self.deals_collection].count_documents({
                **query, "stage": "closed_won"
            })
            lost_deals = await self.db[self.deals_collection].count_documents({
                **query, "stage": "closed_lost"
            })
            
            # Calculate revenue
            won_deals_cursor = self.db[self.deals_collection].find({
                **query, "stage": "closed_won"
            })
            total_revenue = 0
            async for deal in won_deals_cursor:
                total_revenue += deal.get("deal_value", 0)
            
            # Calculate conversion rates
            lead_to_deal_rate = (total_deals / total_leads) if total_leads > 0 else 0
            deal_win_rate = (won_deals / (won_deals + lost_deals)) if (won_deals + lost_deals) > 0 else 0
            lead_qualification_rate = (qualified_leads / total_leads) if total_leads > 0 else 0
            
            return {
                "period_days": period_days,
                "leads": {
                    "total": total_leads,
                    "qualified": qualified_leads,
                    "qualification_rate": round(lead_qualification_rate, 3)
                },
                "deals": {
                    "total": total_deals,
                    "won": won_deals,
                    "lost": lost_deals,
                    "win_rate": round(deal_win_rate, 3)
                },
                "revenue": {
                    "total": round(total_revenue, 2),
                    "average_deal_size": round(total_revenue / won_deals, 2) if won_deals > 0 else 0
                },
                "conversion_rates": {
                    "lead_to_deal": round(lead_to_deal_rate, 3),
                    "deal_win_rate": round(deal_win_rate, 3),
                    "lead_qualification_rate": round(lead_qualification_rate, 3)
                },
                "sales_rep": sales_rep,
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get sales metrics: {e}")
            return {}