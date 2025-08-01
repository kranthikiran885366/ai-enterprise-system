"""Central Data Lake for unified analytics across all agents."""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger
import pandas as pd
import numpy as np

from shared_libs.database import get_database


class DataLake:
    """Central Data Lake for storing and analyzing data from all agents."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.collections = {
            "events": "data_lake_events",
            "metrics": "data_lake_metrics", 
            "analytics": "data_lake_analytics",
            "insights": "data_lake_insights"
        }
    
    async def initialize(self):
        """Initialize the data lake."""
        self.db = await get_database()
        
        # Create indexes for better performance
        await self.db[self.collections["events"]].create_index([("timestamp", -1)])
        await self.db[self.collections["events"]].create_index([("agent", 1), ("event_type", 1)])
        await self.db[self.collections["events"]].create_index([("entity_type", 1), ("entity_id", 1)])
        
        await self.db[self.collections["metrics"]].create_index([("date", -1)])
        await self.db[self.collections["metrics"]].create_index([("agent", 1), ("metric_name", 1)])
        
        await self.db[self.collections["analytics"]].create_index([("analysis_type", 1), ("created_at", -1)])
        await self.db[self.collections["insights"]].create_index([("insight_type", 1), ("priority", -1)])
        
        logger.info("Data Lake initialized")
    
    async def store_event(self, agent: str, event_type: str, entity_type: str, 
                         entity_id: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store an event in the data lake."""
        try:
            event = {
                "event_id": f"evt_{int(datetime.utcnow().timestamp() * 1000)}_{agent}",
                "timestamp": datetime.utcnow(),
                "agent": agent,
                "event_type": event_type,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "data": data,
                "metadata": metadata or {},
                "processed": False
            }
            
            result = await self.db[self.collections["events"]].insert_one(event)
            
            if result.inserted_id:
                logger.debug(f"Event stored: {event['event_id']}")
                return event["event_id"]
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return ""
    
    async def store_metric(self, agent: str, metric_name: str, value: Union[int, float], 
                          date: datetime, dimensions: Optional[Dict[str, Any]] = None) -> bool:
        """Store a metric in the data lake."""
        try:
            metric = {
                "metric_id": f"met_{int(date.timestamp())}_{agent}_{metric_name}",
                "agent": agent,
                "metric_name": metric_name,
                "value": value,
                "date": date,
                "dimensions": dimensions or {},
                "created_at": datetime.utcnow()
            }
            
            # Upsert to handle duplicate metrics
            await self.db[self.collections["metrics"]].replace_one(
                {"metric_id": metric["metric_id"]},
                metric,
                upsert=True
            )
            
            logger.debug(f"Metric stored: {agent}.{metric_name} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
            return False
    
    async def get_events(self, agent: Optional[str] = None, event_type: Optional[str] = None,
                        entity_type: Optional[str] = None, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve events from the data lake."""
        try:
            query = {}
            
            if agent:
                query["agent"] = agent
            if event_type:
                query["event_type"] = event_type
            if entity_type:
                query["entity_type"] = entity_type
            
            if start_date or end_date:
                query["timestamp"] = {}
                if start_date:
                    query["timestamp"]["$gte"] = start_date
                if end_date:
                    query["timestamp"]["$lte"] = end_date
            
            events = []
            cursor = self.db[self.collections["events"]].find(query).sort("timestamp", -1).limit(limit)
            
            async for event in cursor:
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []
    
    async def get_metrics(self, agent: Optional[str] = None, metric_name: Optional[str] = None,
                         start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Retrieve metrics from the data lake."""
        try:
            query = {}
            
            if agent:
                query["agent"] = agent
            if metric_name:
                query["metric_name"] = metric_name
            
            if start_date or end_date:
                query["date"] = {}
                if start_date:
                    query["date"]["$gte"] = start_date
                if end_date:
                    query["date"]["$lte"] = end_date
            
            metrics = []
            cursor = self.db[self.collections["metrics"]].find(query).sort("date", -1)
            
            async for metric in cursor:
                metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []
    
    async def analyze_agent_performance(self, agent: str, days: int = 30) -> Dict[str, Any]:
        """Analyze agent performance over specified period."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get events for the agent
            events = await self.get_events(agent=agent, start_date=start_date, end_date=end_date, limit=1000)
            
            # Get metrics for the agent
            metrics = await self.get_metrics(agent=agent, start_date=start_date, end_date=end_date)
            
            # Analyze event patterns
            event_analysis = self._analyze_events(events)
            
            # Analyze metric trends
            metric_analysis = self._analyze_metrics(metrics)
            
            # Generate insights
            insights = await self._generate_performance_insights(agent, event_analysis, metric_analysis)
            
            analysis = {
                "agent": agent,
                "period_days": days,
                "analysis_date": datetime.utcnow(),
                "event_analysis": event_analysis,
                "metric_analysis": metric_analysis,
                "insights": insights,
                "recommendations": await self._generate_recommendations(agent, insights)
            }
            
            # Store analysis
            await self.db[self.collections["analytics"]].insert_one({
                **analysis,
                "analysis_type": "agent_performance",
                "created_at": datetime.utcnow()
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze agent performance: {e}")
            return {}
    
    def _analyze_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze event patterns."""
        if not events:
            return {"total_events": 0, "event_types": {}, "hourly_distribution": {}}
        
        event_types = {}
        hourly_distribution = {}
        
        for event in events:
            # Count event types
            event_type = event.get("event_type", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            # Hourly distribution
            hour = event.get("timestamp", datetime.utcnow()).hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        return {
            "total_events": len(events),
            "event_types": event_types,
            "hourly_distribution": hourly_distribution,
            "most_common_event": max(event_types.items(), key=lambda x: x[1])[0] if event_types else None
        }
    
    def _analyze_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze metric trends."""
        if not metrics:
            return {"total_metrics": 0, "metric_trends": {}}
        
        metric_trends = {}
        
        # Group metrics by name
        grouped_metrics = {}
        for metric in metrics:
            name = metric.get("metric_name", "unknown")
            if name not in grouped_metrics:
                grouped_metrics[name] = []
            grouped_metrics[name].append(metric)
        
        # Calculate trends for each metric
        for name, metric_list in grouped_metrics.items():
            if len(metric_list) >= 2:
                # Sort by date
                sorted_metrics = sorted(metric_list, key=lambda x: x.get("date", datetime.min))
                
                values = [m.get("value", 0) for m in sorted_metrics]
                
                # Calculate trend
                if len(values) >= 2:
                    trend = (values[-1] - values[0]) / len(values)
                    avg_value = sum(values) / len(values)
                    
                    metric_trends[name] = {
                        "trend": trend,
                        "average": avg_value,
                        "latest_value": values[-1],
                        "data_points": len(values)
                    }
        
        return {
            "total_metrics": len(metrics),
            "metric_trends": metric_trends
        }
    
    async def _generate_performance_insights(self, agent: str, event_analysis: Dict[str, Any], 
                                           metric_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance insights."""
        insights = []
        
        # Event-based insights
        total_events = event_analysis.get("total_events", 0)
        if total_events > 0:
            most_common_event = event_analysis.get("most_common_event")
            if most_common_event:
                insights.append({
                    "type": "event_pattern",
                    "message": f"Most common activity: {most_common_event}",
                    "priority": "medium"
                })
        
        # Metric-based insights
        metric_trends = metric_analysis.get("metric_trends", {})
        for metric_name, trend_data in metric_trends.items():
            trend = trend_data.get("trend", 0)
            if trend > 0.1:
                insights.append({
                    "type": "positive_trend",
                    "message": f"{metric_name} is trending upward (+{trend:.2f})",
                    "priority": "low"
                })
            elif trend < -0.1:
                insights.append({
                    "type": "negative_trend",
                    "message": f"{metric_name} is trending downward ({trend:.2f})",
                    "priority": "high"
                })
        
        return insights
    
    async def _generate_recommendations(self, agent: str, insights: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on insights."""
        recommendations = []
        
        for insight in insights:
            if insight.get("type") == "negative_trend":
                recommendations.append(f"Investigate declining {insight.get('message', '').split()[0]} metric")
            elif insight.get("type") == "positive_trend":
                recommendations.append(f"Analyze and replicate success factors for {insight.get('message', '').split()[0]}")
        
        # Agent-specific recommendations
        if agent == "hr":
            recommendations.append("Review employee satisfaction surveys for improvement opportunities")
        elif agent == "finance":
            recommendations.append("Analyze expense patterns for cost optimization")
        elif agent == "sales":
            recommendations.append("Review lead conversion funnel for bottlenecks")
        
        return recommendations
    
    async def get_cross_agent_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics across all agents."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get all events
            all_events = await self.get_events(start_date=start_date, end_date=end_date, limit=5000)
            
            # Get all metrics
            all_metrics = await self.get_metrics(start_date=start_date, end_date=end_date)
            
            # Analyze by agent
            agent_activity = {}
            for event in all_events:
                agent = event.get("agent", "unknown")
                if agent not in agent_activity:
                    agent_activity[agent] = {"events": 0, "event_types": set()}
                agent_activity[agent]["events"] += 1
                agent_activity[agent]["event_types"].add(event.get("event_type", "unknown"))
            
            # Convert sets to lists for JSON serialization
            for agent_data in agent_activity.values():
                agent_data["event_types"] = list(agent_data["event_types"])
                agent_data["unique_event_types"] = len(agent_data["event_types"])
            
            # Metric summary by agent
            agent_metrics = {}
            for metric in all_metrics:
                agent = metric.get("agent", "unknown")
                if agent not in agent_metrics:
                    agent_metrics[agent] = {"metrics": 0, "metric_names": set()}
                agent_metrics[agent]["metrics"] += 1
                agent_metrics[agent]["metric_names"].add(metric.get("metric_name", "unknown"))
            
            # Convert sets to lists
            for agent_data in agent_metrics.values():
                agent_data["metric_names"] = list(agent_data["metric_names"])
                agent_data["unique_metrics"] = len(agent_data["metric_names"])
            
            return {
                "period_days": days,
                "total_events": len(all_events),
                "total_metrics": len(all_metrics),
                "agent_activity": agent_activity,
                "agent_metrics": agent_metrics,
                "most_active_agent": max(agent_activity.items(), key=lambda x: x[1]["events"])[0] if agent_activity else None,
                "analysis_date": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get cross-agent analytics: {e}")
            return {}
    
    async def store_insight(self, insight_type: str, title: str, description: str,
                          priority: str, agent: Optional[str] = None, 
                          data: Optional[Dict[str, Any]] = None) -> str:
        """Store an insight in the data lake."""
        try:
            insight = {
                "insight_id": f"ins_{int(datetime.utcnow().timestamp() * 1000)}",
                "insight_type": insight_type,
                "title": title,
                "description": description,
                "priority": priority,
                "agent": agent,
                "data": data or {},
                "created_at": datetime.utcnow(),
                "status": "active"
            }
            
            result = await self.db[self.collections["insights"]].insert_one(insight)
            
            if result.inserted_id:
                logger.info(f"Insight stored: {insight['insight_id']}")
                return insight["insight_id"]
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to store insight: {e}")
            return ""
    
    async def get_insights(self, insight_type: Optional[str] = None, agent: Optional[str] = None,
                          priority: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get insights from the data lake."""
        try:
            query = {"status": "active"}
            
            if insight_type:
                query["insight_type"] = insight_type
            if agent:
                query["agent"] = agent
            if priority:
                query["priority"] = priority
            
            insights = []
            cursor = self.db[self.collections["insights"]].find(query).sort("created_at", -1).limit(limit)
            
            async for insight in cursor:
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get insights: {e}")
            return []


# Global data lake instance
data_lake = DataLake()


async def get_data_lake() -> DataLake:
    """Get data lake instance."""
    return data_lake
