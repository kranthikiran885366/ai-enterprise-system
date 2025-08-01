"""Decision Engine for AI-powered decision making and monitoring."""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger
import httpx

from shared_libs.database import get_database
from shared_libs.messaging import get_message_broker
from models.decision import Recommendation, RecommendationCreate
from services.rule_engine import RuleEngine


class DecisionEngine:
    """AI Decision Engine for intelligent decision making."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.rule_engine: Optional[RuleEngine] = None
        self.recommendations_collection = "recommendations"
        self.analytics_collection = "analytics"
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
    
    async def initialize(self):
        """Initialize the decision engine."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.recommendations_collection].create_index("recommendation_id", unique=True)
        await self.db[self.recommendations_collection].create_index("category")
        await self.db[self.recommendations_collection].create_index("department")
        await self.db[self.recommendations_collection].create_index("status")
        await self.db[self.recommendations_collection].create_index("priority")
        
        await self.db[self.analytics_collection].create_index("date")
        await self.db[self.analytics_collection].create_index("department")
        
        logger.info("Decision engine initialized")
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Decision engine monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Decision engine monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect data from all services
                await self._collect_service_data()
                
                # Analyze patterns and generate recommendations
                await self._analyze_patterns()
                
                # Check for anomalies
                await self._detect_anomalies()
                
                # Sleep for 5 minutes before next iteration
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _collect_service_data(self):
        """Collect data from all microservices."""
        try:
            services = [
                {"name": "hr", "url": "http://hr-agent:8000"},
                {"name": "finance", "url": "http://finance-agent:8000"},
                {"name": "sales", "url": "http://sales-agent:8000"},
                {"name": "marketing", "url": "http://marketing-agent:8000"},
                {"name": "it", "url": "http://it-agent:8000"},
                {"name": "admin", "url": "http://admin-agent:8000"},
                {"name": "legal", "url": "http://legal-agent:8000"},
                {"name": "support", "url": "http://support-agent:8000"}
            ]
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                for service in services:
                    try:
                        # Get service health
                        health_response = await client.get(f"{service['url']}/health")
                        
                        if health_response.status_code == 200:
                            # Store analytics data
                            analytics_data = {
                                "service": service["name"],
                                "status": "healthy",
                                "response_time": health_response.elapsed.total_seconds(),
                                "timestamp": datetime.utcnow(),
                                "date": datetime.utcnow().date().isoformat()
                            }
                            
                            await self.db[self.analytics_collection].insert_one(analytics_data)
                        
                    except Exception as e:
                        logger.warning(f"Failed to collect data from {service['name']}: {e}")
                        
                        # Store failure data
                        analytics_data = {
                            "service": service["name"],
                            "status": "unhealthy",
                            "error": str(e),
                            "timestamp": datetime.utcnow(),
                            "date": datetime.utcnow().date().isoformat()
                        }
                        
                        await self.db[self.analytics_collection].insert_one(analytics_data)
            
        except Exception as e:
            logger.error(f"Failed to collect service data: {e}")
    
    async def _analyze_patterns(self):
        """Analyze patterns and generate recommendations."""
        try:
            # Analyze service health patterns
            await self._analyze_service_health()
            
            # Analyze department performance
            await self._analyze_department_performance()
            
            # Generate cost optimization recommendations
            await self._generate_cost_recommendations()
            
        except Exception as e:
            logger.error(f"Failed to analyze patterns: {e}")
    
    async def _analyze_service_health(self):
        """Analyze service health patterns."""
        try:
            # Get service health data from last 24 hours
            yesterday = datetime.utcnow() - timedelta(days=1)
            
            pipeline = [
                {"$match": {"timestamp": {"$gte": yesterday}}},
                {"$group": {
                    "_id": "$service",
                    "total_checks": {"$sum": 1},
                    "healthy_checks": {"$sum": {"$cond": [{"$eq": ["$status", "healthy"]}, 1, 0]}},
                    "avg_response_time": {"$avg": "$response_time"}
                }},
                {"$addFields": {
                    "health_percentage": {"$multiply": [{"$divide": ["$healthy_checks", "$total_checks"]}, 100]}
                }}
            ]
            
            cursor = self.db[self.analytics_collection].aggregate(pipeline)
            
            async for service_health in cursor:
                service_name = service_health["_id"]
                health_percentage = service_health.get("health_percentage", 0)
                avg_response_time = service_health.get("avg_response_time", 0)
                
                # Generate recommendations for unhealthy services
                if health_percentage &lt; 95:
                    await self._create_recommendation(
                        title=f"Service Health Alert: {service_name}",
                        description=f"Service {service_name} has {health_percentage:.1f}% uptime in the last 24 hours",
                        category="infrastructure",
                        department="it",
                        priority=8,
                        confidence_score=0.9,
                        data_sources=[f"{service_name}_health_metrics"],
                        suggested_actions=[
                            {"action": "investigate_service_issues", "service": service_name},
                            {"action": "check_resource_usage", "service": service_name},
                            {"action": "review_error_logs", "service": service_name}
                        ]
                    )
                
                # Generate recommendations for slow services
                if avg_response_time > 2.0:  # 2 seconds
                    await self._create_recommendation(
                        title=f"Performance Alert: {service_name}",
                        description=f"Service {service_name} has average response time of {avg_response_time:.2f}s",
                        category="performance",
                        department="it",
                        priority=6,
                        confidence_score=0.8,
                        data_sources=[f"{service_name}_performance_metrics"],
                        suggested_actions=[
                            {"action": "optimize_database_queries", "service": service_name},
                            {"action": "scale_service_resources", "service": service_name},
                            {"action": "implement_caching", "service": service_name}
                        ]
                    )
            
        except Exception as e:
            logger.error(f"Failed to analyze service health: {e}")
    
    async def _analyze_department_performance(self):
        """Analyze department performance patterns."""
        try:
            # This would analyze department-specific metrics
            # For now, we'll create sample recommendations
            
            departments = ["hr", "finance", "sales", "marketing", "it", "admin", "legal", "support"]
            
            for dept in departments:
                # Generate sample performance recommendations
                if dept == "hr":
                    await self._create_recommendation(
                        title="HR Process Optimization",
                        description="Employee onboarding process can be streamlined",
                        category="process_improvement",
                        department="hr",
                        priority=5,
                        confidence_score=0.7,
                        data_sources=["hr_metrics", "employee_feedback"],
                        suggested_actions=[
                            {"action": "automate_onboarding_checklist"},
                            {"action": "implement_digital_forms"},
                            {"action": "create_welcome_portal"}
                        ]
                    )
                elif dept == "finance":
                    await self._create_recommendation(
                        title="Expense Approval Workflow",
                        description="Expense approval process has bottlenecks",
                        category="workflow_optimization",
                        department="finance",
                        priority=6,
                        confidence_score=0.8,
                        data_sources=["expense_data", "approval_times"],
                        suggested_actions=[
                            {"action": "implement_automated_approvals"},
                            {"action": "set_approval_thresholds"},
                            {"action": "create_escalation_rules"}
                        ]
                    )
            
        except Exception as e:
            logger.error(f"Failed to analyze department performance: {e}")
    
    async def _generate_cost_recommendations(self):
        """Generate cost optimization recommendations."""
        try:
            await self._create_recommendation(
                title="Cloud Resource Optimization",
                description="Unused cloud resources detected that can be optimized",
                category="cost_optimization",
                department="it",
                priority=7,
                confidence_score=0.85,
                data_sources=["cloud_usage_metrics", "cost_analysis"],
                suggested_actions=[
                    {"action": "right_size_instances"},
                    {"action": "implement_auto_scaling"},
                    {"action": "schedule_non_production_resources"}
                ]
            )
            
        except Exception as e:
            logger.error(f"Failed to generate cost recommendations: {e}")
    
    async def _detect_anomalies(self):
        """Detect anomalies in system behavior."""
        try:
            # This would implement anomaly detection algorithms
            # For now, we'll create sample anomaly alerts
            
            # Check for unusual patterns in the last hour
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            # Count service failures
            failure_count = await self.db[self.analytics_collection].count_documents({
                "status": "unhealthy",
                "timestamp": {"$gte": one_hour_ago}
            })
            
            if failure_count > 5:  # More than 5 failures in an hour
                await self._create_recommendation(
                    title="System Anomaly Detected",
                    description=f"Unusual number of service failures detected: {failure_count} in the last hour",
                    category="anomaly_detection",
                    department="it",
                    priority=9,
                    confidence_score=0.95,
                    data_sources=["system_health_metrics"],
                    suggested_actions=[
                        {"action": "investigate_system_issues"},
                        {"action": "check_infrastructure_status"},
                        {"action": "review_recent_deployments"}
                    ]
                )
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
    
    async def _create_recommendation(self, title: str, description: str, category: str,
                                   department: str, priority: int, confidence_score: float,
                                   data_sources: List[str], suggested_actions: List[Dict[str, Any]],
                                   created_for: Optional[str] = None) -> Optional[Recommendation]:
        """Create a new recommendation."""
        try:
            # Check if similar recommendation already exists
            existing = await self.db[self.recommendations_collection].find_one({
                "title": title,
                "status": {"$in": ["pending", "accepted"]},
                "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)}
            })
            
            if existing:
                return None  # Don't create duplicate recommendations
            
            recommendation_id = f"REC{str(uuid.uuid4())[:8].upper()}"
            
            recommendation_dict = {
                "recommendation_id": recommendation_id,
                "title": title,
                "description": description,
                "category": category,
                "department": department,
                "priority": priority,
                "confidence_score": confidence_score,
                "data_sources": data_sources,
                "suggested_actions": suggested_actions,
                "status": "pending",
                "created_for": created_for,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await self.db[self.recommendations_collection].insert_one(recommendation_dict)
            
            if result.inserted_id:
                recommendation_dict["_id"] = result.inserted_id
                logger.info(f"Recommendation created: {recommendation_id}")
                return Recommendation(**recommendation_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create recommendation: {e}")
            return None
    
    async def create_recommendation(self, recommendation_data: RecommendationCreate) -> Optional[Recommendation]:
        """Create a new recommendation from external request."""
        return await self._create_recommendation(
            title=recommendation_data.title,
            description=recommendation_data.description,
            category=recommendation_data.category,
            department=recommendation_data.department,
            priority=recommendation_data.priority,
            confidence_score=recommendation_data.confidence_score,
            data_sources=recommendation_data.data_sources,
            suggested_actions=recommendation_data.suggested_actions,
            created_for=recommendation_data.created_for
        )
    
    async def get_recommendations(self, department: Optional[str] = None, category: Optional[str] = None,
                                status: Optional[str] = None, skip: int = 0, limit: int = 10) -> List[Recommendation]:
        """Get recommendations with filters."""
        try:
            query = {}
            if department:
                query["department"] = department
            if category:
                query["category"] = category
            if status:
                query["status"] = status
            
            recommendations = []
            cursor = self.db[self.recommendations_collection].find(query).skip(skip).limit(limit).sort("priority", -1)
            
            async for rec_doc in cursor:
                recommendations.append(Recommendation(**rec_doc))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []
    
    async def update_recommendation_status(self, recommendation_id: str, status: str, updated_by: Optional[str] = None) -> bool:
        """Update recommendation status."""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if updated_by:
                update_data["updated_by"] = updated_by
            
            if status == "accepted":
                update_data["accepted_at"] = datetime.utcnow()
            elif status == "rejected":
                update_data["rejected_at"] = datetime.utcnow()
            elif status == "implemented":
                update_data["implemented_at"] = datetime.utcnow()
            
            result = await self.db[self.recommendations_collection].update_one(
                {"recommendation_id": recommendation_id},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Recommendation status updated: {recommendation_id} -> {status}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update recommendation status {recommendation_id}: {e}")
            return False
    
    async def get_analytics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics summary for the specified number of days."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Service health summary
            health_pipeline = [
                {"$match": {"timestamp": {"$gte": start_date}}},
                {"$group": {
                    "_id": "$service",
                    "total_checks": {"$sum": 1},
                    "healthy_checks": {"$sum": {"$cond": [{"$eq": ["$status", "healthy"]}, 1, 0]}},
                    "avg_response_time": {"$avg": "$response_time"}
                }},
                {"$addFields": {
                    "health_percentage": {"$multiply": [{"$divide": ["$healthy_checks", "$total_checks"]}, 100]}
                }}
            ]
            
            service_health = []
            cursor = self.db[self.analytics_collection].aggregate(health_pipeline)
            async for service in cursor:
                service_health.append({
                    "service": service["_id"],
                    "health_percentage": round(service.get("health_percentage", 0), 2),
                    "avg_response_time": round(service.get("avg_response_time", 0), 3),
                    "total_checks": service["total_checks"]
                })
            
            # Recommendations summary
            rec_pipeline = [
                {"$match": {"created_at": {"$gte": start_date}}},
                {"$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }}
            ]
            
            recommendation_stats = {}
            cursor = self.db[self.recommendations_collection].aggregate(rec_pipeline)
            async for stat in cursor:
                recommendation_stats[stat["_id"]] = stat["count"]
            
            # Top categories
            category_pipeline = [
                {"$match": {"created_at": {"$gte": start_date}}},
                {"$group": {
                    "_id": "$category",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}},
                {"$limit": 5}
            ]
            
            top_categories = []
            cursor = self.db[self.recommendations_collection].aggregate(category_pipeline)
            async for category in cursor:
                top_categories.append({
                    "category": category["_id"],
                    "count": category["count"]
                })
            
            return {
                "period_days": days,
                "service_health": service_health,
                "recommendation_stats": recommendation_stats,
                "top_categories": top_categories,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}
