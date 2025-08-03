"""AI-powered admin services for Admin Agent."""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger
import uuid

from shared_libs.database import get_database
from shared_libs.intelligence import get_nlp_processor
from shared_libs.data_lake import get_data_lake


class AIAdminService:
    """AI-powered administrative automation service."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.user_behavior_collection = "user_behavior_analysis"
        self.smart_notifications_collection = "smart_notifications"
        self.policy_analysis_collection = "policy_analysis"
    
    async def initialize(self):
        """Initialize the AI admin service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.user_behavior_collection].create_index("user_id")
        await self.db[self.user_behavior_collection].create_index("analysis_date")
        await self.db[self.user_behavior_collection].create_index("behavior_score")
        
        await self.db[self.smart_notifications_collection].create_index("notification_id", unique=True)
        await self.db[self.smart_notifications_collection].create_index("user_id")
        await self.db[self.smart_notifications_collection].create_index("trigger_type")
        
        await self.db[self.policy_analysis_collection].create_index("analysis_id", unique=True)
        await self.db[self.policy_analysis_collection].create_index("policy_id")
        
        logger.info("AI Admin service initialized")
    
    async def analyze_user_behavior(self, user_id: str) -> Dict[str, Any]:
        """Analyze user behavior patterns for insights."""
        try:
            data_lake = await get_data_lake()
            
            # Get user activity data from data lake
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            
            user_events = await data_lake.get_events(
                entity_type="user",
                entity_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not user_events:
                return {"message": "No user activity data available"}
            
            # Analyze activity patterns
            activity_analysis = await self._analyze_activity_patterns(user_events)
            
            # Analyze system usage
            usage_analysis = await self._analyze_system_usage(user_events)
            
            # Generate behavior insights
            behavior_insights = await self._generate_behavior_insights(activity_analysis, usage_analysis)
            
            # Calculate behavior score
            behavior_score = self._calculate_behavior_score(activity_analysis, usage_analysis)
            
            analysis_result = {
                "analysis_id": f"UBA_{str(uuid.uuid4())[:8].upper()}",
                "user_id": user_id,
                "analysis_period_days": 30,
                "activity_analysis": activity_analysis,
                "usage_analysis": usage_analysis,
                "behavior_insights": behavior_insights,
                "behavior_score": behavior_score,
                "recommendations": await self._generate_user_recommendations(behavior_insights, behavior_score),
                "created_at": datetime.utcnow()
            }
            
            # Store analysis
            await self.db[self.user_behavior_collection].insert_one(analysis_result)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="admin",
                event_type="user_behavior_analyzed",
                entity_type="user",
                entity_id=user_id,
                data={
                    "behavior_score": behavior_score,
                    "insights_count": len(behavior_insights),
                    "analysis_period": 30
                }
            )
            
            logger.info(f"User behavior analyzed: {user_id}, score={behavior_score:.3f}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze user behavior: {e}")
            return {}
    
    async def _analyze_activity_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user activity patterns."""
        try:
            if not events:
                return {"total_activities": 0}
            
            # Activity by hour
            hourly_activity = {}
            daily_activity = {}
            activity_types = {}
            
            for event in events:
                timestamp = event.get("timestamp", datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                hour = timestamp.hour
                day = timestamp.strftime("%A")
                event_type = event.get("event_type", "unknown")
                
                hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
                daily_activity[day] = daily_activity.get(day, 0) + 1
                activity_types[event_type] = activity_types.get(event_type, 0) + 1
            
            # Find peak activity times
            peak_hour = max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else 9
            peak_day = max(daily_activity.items(), key=lambda x: x[1])[0] if daily_activity else "Monday"
            
            return {
                "total_activities": len(events),
                "hourly_distribution": hourly_activity,
                "daily_distribution": daily_activity,
                "activity_types": activity_types,
                "peak_hour": peak_hour,
                "peak_day": peak_day,
                "most_common_activity": max(activity_types.items(), key=lambda x: x[1])[0] if activity_types else "unknown"
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze activity patterns: {e}")
            return {"total_activities": 0}
    
    async def _analyze_system_usage(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze system usage patterns."""
        try:
            if not events:
                return {"systems_used": []}
            
            systems_used = set()
            feature_usage = {}
            
            for event in events:
                # Extract system/service from event
                agent = event.get("agent", "unknown")
                systems_used.add(agent)
                
                # Track feature usage
                event_type = event.get("event_type", "unknown")
                feature_usage[event_type] = feature_usage.get(event_type, 0) + 1
            
            # Calculate usage diversity
            usage_diversity = len(systems_used) / 8  # Assuming 8 total systems
            
            return {
                "systems_used": list(systems_used),
                "system_count": len(systems_used),
                "feature_usage": feature_usage,
                "usage_diversity": round(usage_diversity, 3),
                "most_used_feature": max(feature_usage.items(), key=lambda x: x[1])[0] if feature_usage else "unknown"
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze system usage: {e}")
            return {"systems_used": []}
    
    async def _generate_behavior_insights(self, activity_analysis: Dict[str, Any], 
                                        usage_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from behavior analysis."""
        try:
            insights = []
            
            # Activity insights
            total_activities = activity_analysis.get("total_activities", 0)
            if total_activities > 100:
                insights.append("High system engagement - power user")
            elif total_activities < 20:
                insights.append("Low system engagement - may need training")
            
            peak_hour = activity_analysis.get("peak_hour", 9)
            if peak_hour < 8 or peak_hour > 18:
                insights.append("Works outside standard business hours")
            
            # Usage insights
            system_count = usage_analysis.get("system_count", 0)
            if system_count >= 6:
                insights.append("Uses multiple systems effectively")
            elif system_count <= 2:
                insights.append("Limited system usage - potential for expansion")
            
            usage_diversity = usage_analysis.get("usage_diversity", 0)
            if usage_diversity > 0.7:
                insights.append("High feature adoption across systems")
            
            return insights
            
        
            
        except Exception as e:
            logger.error(f"Failed to generate behavior insights: {e}")
            return []
    
    def _calculate_behavior_score(self, activity_analysis: Dict[str, Any], 
                                usage_analysis: Dict[str, Any]) -> float:
        """Calculate overall behavior score."""
        try:
            score = 0.0
            
            # Activity score (40%)
            total_activities = activity_analysis.get("total_activities", 0)
            activity_score = min(total_activities / 100, 1.0)  # Normalize to 1.0
            score += activity_score * 0.4
            
            # Usage diversity score (30%)
            usage_diversity = usage_analysis.get("usage_diversity", 0)
            score += usage_diversity * 0.3
            
            # Engagement consistency score (30%)
            daily_distribution = activity_analysis.get("daily_distribution", {})
            if daily_distribution:
                # Calculate consistency (lower standard deviation = more consistent)
                daily_counts = list(daily_distribution.values())
                if len(daily_counts) > 1:
                    avg_daily = sum(daily_counts) / len(daily_counts)
                    variance = sum((x - avg_daily) ** 2 for x in daily_counts) / len(daily_counts)
                    consistency = max(0, 1 - (variance / (avg_daily ** 2))) if avg_daily > 0 else 0
                    score += consistency * 0.3
            
            return round(min(score, 1.0), 3)
            
        except Exception as e:
            logger.error(f"Failed to calculate behavior score: {e}")
            return 0.5
    
    async def _generate_user_recommendations(self, insights: List[str], behavior_score: float) -> List[str]:
        """Generate recommendations for user."""
        try:
            recommendations = []
            
            if behavior_score < 0.3:
                recommendations.extend([
                    "Provide additional system training",
                    "Schedule onboarding review session",
                    "Assign system mentor"
                ])
            elif behavior_score > 0.8:
                recommendations.extend([
                    "Consider for advanced feature beta testing",
                    "Potential candidate for system champion role",
                    "Provide advanced training opportunities"
                ])
            
            # Insight-based recommendations
            for insight in insights:
                if "training" in insight.lower():
                    recommendations.append("Schedule targeted training session")
                elif "outside" in insight.lower():
                    recommendations.append("Review work-life balance and flexible work policies")
                elif "limited" in insight.lower():
                    recommendations.append("Introduce additional system features gradually")
            
            return list(set(recommendations))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to generate user recommendations: {e}")
            return []
    
    async def generate_smart_notifications(self) -> Dict[str, Any]:
        """Generate smart notifications based on system events and user behavior."""
        try:
            data_lake = await get_data_lake()
            
            # Get recent system events
            recent_events = await data_lake.get_events(
                start_date=datetime.utcnow() - timedelta(hours=24),
                limit=1000
            )
            
            # Analyze events for notification triggers
            notification_triggers = await self._analyze_notification_triggers(recent_events)
            
            # Generate notifications
            generated_notifications = []
            
            for trigger in notification_triggers:
                notifications = await self._create_smart_notifications(trigger)
                generated_notifications.extend(notifications)
            
            # Store generated notifications
            for notification in generated_notifications:
                await self.db[self.smart_notifications_collection].insert_one(notification)
            
            result = {
                "generation_id": f"SN_{str(uuid.uuid4())[:8].upper()}",
                "notifications_generated": len(generated_notifications),
                "triggers_processed": len(notification_triggers),
                "notifications": generated_notifications,
                "created_at": datetime.utcnow()
            }
            
            logger.info(f"Smart notifications generated: {len(generated_notifications)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate smart notifications: {e}")
            return {}
    
    async def _analyze_notification_triggers(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze events for notification triggers."""
        try:
            triggers = []
            
            # Group events by type
            event_groups = {}
            for event in events:
                event_type = event.get("event_type", "unknown")
                if event_type not in event_groups:
                    event_groups[event_type] = []
                event_groups[event_type].append(event)
            
            # Check for trigger conditions
            for event_type, event_list in event_groups.items():
                if len(event_list) > 10:  # High frequency events
                    triggers.append({
                        "trigger_type": "high_frequency_activity",
                        "event_type": event_type,
                        "count": len(event_list),
                        "description": f"High frequency of {event_type} events detected"
                    })
                
                # Check for error events
                if "error" in event_type or "failed" in event_type:
                    triggers.append({
                        "trigger_type": "error_pattern",
                        "event_type": event_type,
                        "count": len(event_list),
                        "description": f"Error pattern detected: {event_type}"
                    })
            
            return triggers
            
        except Exception as e:
            logger.error(f"Failed to analyze notification triggers: {e}")
            return []
    
    async def _create_smart_notifications(self, trigger: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create smart notifications based on triggers."""
        try:
            notifications = []
            trigger_type = trigger.get("trigger_type")
            
            if trigger_type == "high_frequency_activity":
                # Notify system administrators
                notifications.append({
                    "notification_id": f"SN_{str(uuid.uuid4())[:8].upper()}",
                    "trigger_type": trigger_type,
                    "recipient_role": "system_admin",
                    "title": "High System Activity Detected",
                    "message": trigger.get("description", ""),
                    "priority": "medium",
                    "auto_generated": True,
                    "created_at": datetime.utcnow()
                })
            
            elif trigger_type == "error_pattern":
                # Notify IT team
                notifications.append({
                    "notification_id": f"SN_{str(uuid.uuid4())[:8].upper()}",
                    "trigger_type": trigger_type,
                    "recipient_role": "it_team",
                    "title": "Error Pattern Detected",
                    "message": trigger.get("description", ""),
                    "priority": "high",
                    "auto_generated": True,
                    "created_at": datetime.utcnow()
                })
            
            return notifications
            
        except Exception as e:
            logger.error(f"Failed to create smart notifications: {e}")
            return []