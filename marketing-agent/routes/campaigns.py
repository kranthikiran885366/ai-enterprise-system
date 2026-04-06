"""Marketing campaign routes."""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
from datetime import datetime
from loguru import logger

from models.campaign import (
    CampaignCreate, CampaignUpdate, Campaign, ContentCreate,
    AudienceSegmentCreate, AudienceSegment
)


router = APIRouter()

# This will be injected from main.py
marketing_service = None


async def get_marketing_service():
    """Dependency to get marketing service."""
    global marketing_service
    if not marketing_service:
        raise HTTPException(status_code=500, detail="Marketing service not initialized")
    return marketing_service


@router.post("/", response_model=Campaign, status_code=201)
async def create_campaign(
    campaign_data: CampaignCreate,
    service = Depends(get_marketing_service)
):
    """Create a new marketing campaign."""
    try:
        campaign = await service.create_campaign(
            name=campaign_data.name,
            description=campaign_data.description,
            campaign_type=campaign_data.campaign_type,
            owner=campaign_data.owner,
            target_audience=campaign_data.target_audience,
            start_date=campaign_data.start_date,
            end_date=campaign_data.end_date,
            budget=campaign_data.budget,
            goals=campaign_data.goals
        )
        
        if not campaign:
            raise HTTPException(status_code=400, detail="Failed to create campaign")
        
        return campaign
    except Exception as e:
        logger.error(f"Failed to create campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{campaign_id}", response_model=Campaign)
async def get_campaign(
    campaign_id: str,
    service = Depends(get_marketing_service)
):
    """Get a campaign by ID."""
    try:
        campaign = await service.get_campaign(campaign_id)
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        return campaign
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[Campaign])
async def list_campaigns(
    owner: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    campaign_type: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    service = Depends(get_marketing_service)
):
    """List campaigns with optional filters."""
    try:
        campaigns = await service.list_campaigns(
            owner=owner,
            status=status,
            campaign_type=campaign_type,
            skip=skip,
            limit=limit
        )
        return campaigns
    except Exception as e:
        logger.error(f"Failed to list campaigns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{campaign_id}/status", status_code=200)
async def update_campaign_status(
    campaign_id: str,
    status: str,
    updated_by: str,
    service = Depends(get_marketing_service)
):
    """Update campaign status."""
    try:
        success = await service.update_campaign_status(campaign_id, status, updated_by)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update campaign status")
        
        return {"message": f"Campaign status updated to {status}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update campaign status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{campaign_id}/generate-content", status_code=200)
async def generate_campaign_content(
    campaign_id: str,
    content_type: str,
    target_audience: dict,
    tone: str = "professional",
    count: int = 3,
    service = Depends(get_marketing_service)
):
    """Generate marketing content using AI."""
    try:
        content_items = await service.generate_campaign_content(
            campaign_id=campaign_id,
            content_type=content_type,
            target_audience=target_audience,
            tone=tone,
            count=count
        )
        
        return {
            "campaign_id": campaign_id,
            "content_count": len(content_items),
            "content": content_items
        }
    except Exception as e:
        logger.error(f"Failed to generate campaign content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segments/", response_model=AudienceSegment, status_code=201)
async def create_audience_segment(
    segment_data: AudienceSegmentCreate,
    service = Depends(get_marketing_service)
):
    """Create an audience segment."""
    try:
        segment = await service.create_audience_segment(
            name=segment_data.name,
            description=segment_data.description,
            criteria=segment_data.criteria,
            size_estimate=segment_data.size_estimate
        )
        
        if not segment:
            raise HTTPException(status_code=400, detail="Failed to create segment")
        
        return segment
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{campaign_id}/analytics", status_code=200)
async def get_campaign_analytics(
    campaign_id: str,
    days: int = Query(30, ge=1, le=365),
    service = Depends(get_marketing_service)
):
    """Get campaign analytics over a period."""
    try:
        analytics = await service.get_campaign_analytics(campaign_id, days)
        
        if "error" in analytics:
            raise HTTPException(status_code=400, detail=analytics["error"])
        
        return analytics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{campaign_id}/optimize", status_code=200)
async def optimize_campaign(
    campaign_id: str,
    service = Depends(get_marketing_service)
):
    """Get AI optimization recommendations for a campaign."""
    try:
        recommendations = await service.optimize_campaign(campaign_id)
        
        if "error" in recommendations:
            raise HTTPException(status_code=400, detail=recommendations["error"])
        
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to optimize campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{campaign_id}/metrics", status_code=201)
async def record_campaign_metrics(
    campaign_id: str,
    metrics: dict,
    service = Depends(get_marketing_service)
):
    """Record campaign performance metrics."""
    try:
        success = await service.record_campaign_metrics(campaign_id, metrics)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to record metrics")
        
        return {"message": "Metrics recorded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
