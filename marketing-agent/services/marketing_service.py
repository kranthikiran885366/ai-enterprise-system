"""Marketing Service for managing campaigns and marketing operations."""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger

from shared_libs.database import get_database
from shared_libs.ai_providers import get_orchestrator


class CampaignStatus(str, Enum):
    """Campaign status enumeration."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CampaignType(str, Enum):
    """Campaign type enumeration."""
    EMAIL = "email"
    SOCIAL = "social"
    CONTENT = "content"
    PAID_ADS = "paid_ads"
    INFLUENCER = "influencer"


class MarketingService:
    """Marketing service for managing campaigns and content."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.campaigns_collection = "marketing_campaigns"
        self.content_collection = "marketing_content"
        self.analytics_collection = "campaign_analytics"
        self.segments_collection = "audience_segments"
        self.orchestrator = None
    
    async def initialize(self):
        """Initialize the Marketing service."""
        self.db = await get_database()
        self.orchestrator = await get_orchestrator()
        
        # Create indexes
        await self.db[self.campaigns_collection].create_index("campaign_id", unique=True)
        await self.db[self.campaigns_collection].create_index("name")
        await self.db[self.campaigns_collection].create_index("status")
        await self.db[self.campaigns_collection].create_index("campaign_type")
        await self.db[self.campaigns_collection].create_index("owner")
        await self.db[self.campaigns_collection].create_index("created_at")
        
        await self.db[self.content_collection].create_index("content_id", unique=True)
        await self.db[self.content_collection].create_index("campaign_id")
        await self.db[self.content_collection].create_index("content_type")
        
        await self.db[self.analytics_collection].create_index("campaign_id")
        await self.db[self.analytics_collection].create_index([("campaign_id", 1), ("date", 1)])
        
        await self.db[self.segments_collection].create_index("segment_id", unique=True)
        
        logger.info("Marketing service initialized")
    
    async def create_campaign(
        self,
        name: str,
        description: str,
        campaign_type: str,
        owner: str,
        target_audience: Dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        budget: float,
        goals: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Create a new marketing campaign."""
        try:
            campaign_id = f"CAMP{str(uuid.uuid4())[:8].upper()}"
            
            campaign = {
                "campaign_id": campaign_id,
                "name": name,
                "description": description,
                "campaign_type": campaign_type,
                "owner": owner,
                "status": CampaignStatus.DRAFT,
                "target_audience": target_audience,
                "start_date": start_date,
                "end_date": end_date,
                "budget": budget,
                "spent": 0.0,
                "goals": goals,
                "content": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await self.db[self.campaigns_collection].insert_one(campaign)
            
            if result.inserted_id:
                campaign["_id"] = result.inserted_id
                logger.info(f"Campaign created: {campaign_id}")
                return campaign
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create campaign: {e}")
            return None
    
    async def get_campaign(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Get a campaign by ID."""
        try:
            campaign = await self.db[self.campaigns_collection].find_one(
                {"campaign_id": campaign_id}
            )
            return campaign
        except Exception as e:
            logger.error(f"Failed to get campaign {campaign_id}: {e}")
            return None
    
    async def list_campaigns(
        self,
        owner: Optional[str] = None,
        status: Optional[str] = None,
        campaign_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List campaigns with optional filters."""
        try:
            query = {}
            if owner:
                query["owner"] = owner
            if status:
                query["status"] = status
            if campaign_type:
                query["campaign_type"] = campaign_type
            
            campaigns = []
            cursor = self.db[self.campaigns_collection].find(query).skip(skip).limit(limit)
            
            async for campaign in cursor:
                campaigns.append(campaign)
            
            return campaigns
            
        except Exception as e:
            logger.error(f"Failed to list campaigns: {e}")
            return []
    
    async def update_campaign_status(
        self,
        campaign_id: str,
        status: str,
        updated_by: str
    ) -> bool:
        """Update campaign status."""
        try:
            result = await self.db[self.campaigns_collection].update_one(
                {"campaign_id": campaign_id},
                {
                    "$set": {
                        "status": status,
                        "updated_at": datetime.utcnow(),
                        "updated_by": updated_by
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Campaign status updated: {campaign_id} -> {status}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update campaign status: {e}")
            return False
    
    async def generate_campaign_content(
        self,
        campaign_id: str,
        content_type: str,
        target_audience: Dict[str, Any],
        tone: str = "professional",
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate marketing content using AI."""
        try:
            campaign = await self.get_campaign(campaign_id)
            if not campaign:
                logger.error(f"Campaign not found: {campaign_id}")
                return []
            
            prompt = f"""Generate {count} pieces of compelling {content_type} marketing content.

Campaign: {campaign['name']}
Description: {campaign['description']}
Target Audience: {target_audience}
Tone: {tone}
Goals: {', '.join(campaign['goals'])}

For each piece of content, provide:
1. Headline/Subject
2. Body/Copy (2-3 sentences)
3. Call-to-action
4. Key benefits highlighted

Format as numbered list."""
            
            response = await self.orchestrator.complete(prompt, temperature=0.8)
            
            # Parse response and create content records
            content_items = []
            lines = response.split('\n')
            current_content = {}
            
            for line in lines:
                if line.strip() and not line.startswith(('1.', '2.', '3.', '4.')):
                    # This is a new content item
                    if current_content:
                        content_items.append(current_content)
                    current_content = {
                        "content_id": f"CONT{str(uuid.uuid4())[:8].upper()}",
                        "campaign_id": campaign_id,
                        "content_type": content_type,
                        "raw_content": line.strip(),
                        "created_at": datetime.utcnow()
                    }
            
            if current_content:
                content_items.append(current_content)
            
            # Store content
            for item in content_items:
                await self.db[self.content_collection].insert_one(item)
            
            logger.info(f"Generated {len(content_items)} content pieces for campaign {campaign_id}")
            return content_items
            
        except Exception as e:
            logger.error(f"Failed to generate campaign content: {e}")
            return []
    
    async def create_audience_segment(
        self,
        name: str,
        description: str,
        criteria: Dict[str, Any],
        size_estimate: int
    ) -> Optional[Dict[str, Any]]:
        """Create an audience segment."""
        try:
            segment_id = f"SEG{str(uuid.uuid4())[:8].upper()}"
            
            segment = {
                "segment_id": segment_id,
                "name": name,
                "description": description,
                "criteria": criteria,
                "size_estimate": size_estimate,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await self.db[self.segments_collection].insert_one(segment)
            
            if result.inserted_id:
                segment["_id"] = result.inserted_id
                logger.info(f"Audience segment created: {segment_id}")
                return segment
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create audience segment: {e}")
            return None
    
    async def record_campaign_metrics(
        self,
        campaign_id: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """Record campaign performance metrics."""
        try:
            analytics_record = {
                "campaign_id": campaign_id,
                "date": datetime.utcnow().date(),
                "timestamp": datetime.utcnow(),
                "metrics": metrics,
                "created_at": datetime.utcnow()
            }
            
            result = await self.db[self.analytics_collection].insert_one(analytics_record)
            
            if result.inserted_id:
                logger.info(f"Campaign metrics recorded for {campaign_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to record campaign metrics: {e}")
            return False
    
    async def get_campaign_analytics(
        self,
        campaign_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get campaign analytics over a period."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            analytics = await self.db[self.analytics_collection].find({
                "campaign_id": campaign_id,
                "timestamp": {"$gte": start_date}
            }).to_list(None)
            
            if not analytics:
                return {
                    "campaign_id": campaign_id,
                    "impressions": 0,
                    "clicks": 0,
                    "conversions": 0,
                    "roi": 0,
                    "ctr": 0,
                    "conversion_rate": 0
                }
            
            # Aggregate metrics
            total_impressions = sum(m.get("metrics", {}).get("impressions", 0) for m in analytics)
            total_clicks = sum(m.get("metrics", {}).get("clicks", 0) for m in analytics)
            total_conversions = sum(m.get("metrics", {}).get("conversions", 0) for m in analytics)
            total_spent = sum(m.get("metrics", {}).get("spent", 0) for m in analytics)
            total_revenue = sum(m.get("metrics", {}).get("revenue", 0) for m in analytics)
            
            ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
            conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
            roi = ((total_revenue - total_spent) / total_spent * 100) if total_spent > 0 else 0
            
            return {
                "campaign_id": campaign_id,
                "period_days": days,
                "impressions": total_impressions,
                "clicks": total_clicks,
                "conversions": total_conversions,
                "spent": round(total_spent, 2),
                "revenue": round(total_revenue, 2),
                "ctr": round(ctr, 2),
                "conversion_rate": round(conversion_rate, 2),
                "roi": round(roi, 2),
                "data_points": len(analytics)
            }
            
        except Exception as e:
            logger.error(f"Failed to get campaign analytics: {e}")
            return {}
    
    async def optimize_campaign(
        self,
        campaign_id: str
    ) -> Dict[str, Any]:
        """Use AI to analyze and optimize campaign."""
        try:
            campaign = await self.get_campaign(campaign_id)
            if not campaign:
                return {"error": "Campaign not found"}
            
            analytics = await self.get_campaign_analytics(campaign_id)
            
            prompt = f"""Analyze this marketing campaign performance and provide optimization recommendations.

Campaign: {campaign['name']}
Type: {campaign['campaign_type']}
Budget: ${campaign['budget']}

Performance Metrics:
- Impressions: {analytics.get('impressions', 0):,}
- Clicks: {analytics.get('clicks', 0):,}
- CTR: {analytics.get('ctr', 0)}%
- Conversions: {analytics.get('conversions', 0):,}
- Conversion Rate: {analytics.get('conversion_rate', 0)}%
- ROI: {analytics.get('roi', 0)}%

Provide 3-5 specific, actionable recommendations to improve campaign performance."""
            
            recommendations = await self.orchestrator.complete(prompt, temperature=0.7)
            
            return {
                "campaign_id": campaign_id,
                "current_metrics": analytics,
                "recommendations": recommendations,
                "analysis_date": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize campaign: {e}")
            return {"error": str(e)}
