"""Bug pattern recognition service for identifying and analyzing bug patterns."""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict, Counter
import re
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger
import uuid

from shared_libs.database import get_database
from shared_libs.intelligence import get_nlp_processor
from shared_libs.data_lake import get_data_lake


class BugPatternRecognitionService:
    """Service for recognizing and analyzing bug patterns."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.bug_patterns_collection = "bug_patterns"
        self.bug_clusters_collection = "bug_clusters"
        self.pattern_predictions_collection = "pattern_predictions"
    
    async def initialize(self):
        """Initialize the bug pattern recognition service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.bug_patterns_collection].create_index("pattern_id", unique=True)
        await self.db[self.bug_patterns_collection].create_index("pattern_type")
        await self.db[self.bug_patterns_collection].create_index("confidence_score")
        
        await self.db[self.bug_clusters_collection].create_index("cluster_id", unique=True)
        await self.db[self.bug_clusters_collection].create_index("created_at")
        
        await self.db[self.pattern_predictions_collection].create_index("prediction_id", unique=True)
        await self.db[self.pattern_predictions_collection].create_index("predicted_at")
        
        logger.info("Bug Pattern Recognition service initialized")
    
    async def analyze_bug_patterns(self, time_period_days: int = 90) -> Dict[str, Any]:
        """Analyze bug patterns over a specified time period."""
        try:
            data_lake = await get_data_lake()
            
            # Get bugs from the specified time period
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=time_period_days)
            
            bugs = await self.db["bugs"].find({
                "created_at": {"$gte": start_date, "$lte": end_date}
            }).to_list(None)
            
            if not bugs:
                return {"message": "No bugs found in the specified time period"}
            
            # Analyze different pattern types
            patterns = {
                "temporal_patterns": await self._analyze_temporal_patterns(bugs),
                "component_patterns": await self._analyze_component_patterns(bugs),
                "severity_patterns": await self._analyze_severity_patterns(bugs),
                "root_cause_patterns": await self._analyze_root_cause_patterns(bugs),
                "developer_patterns": await self._analyze_developer_patterns(bugs),
                "text_similarity_patterns": await self._analyze_text_similarity_patterns(bugs)
            }
            
            # Generate insights and predictions
            insights = await self._generate_pattern_insights(patterns, bugs)
            predictions = await self._generate_bug_predictions(patterns, bugs)
            
            # Create comprehensive analysis
            analysis_result = {
                "analysis_id": f"BPA_{str(uuid.uuid4())[:8].upper()}",
                "time_period_days": time_period_days,
                "total_bugs_analyzed": len(bugs),
                "patterns_identified": patterns,
                "insights": insights,
                "predictions": predictions,
                "recommendations": await self._generate_pattern_recommendations(patterns, insights),
                "created_at": datetime.utcnow()
            }
            
            # Store analysis
            await self.db[self.bug_patterns_collection].insert_one(analysis_result)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="product",
                event_type="bug_patterns_analyzed",
                entity_type="analysis",
                entity_id=analysis_result["analysis_id"],
                data={
                    "bugs_analyzed": len(bugs),
                    "patterns_found": len([p for p in patterns.values() if p]),
                    "time_period": time_period_days
                }
            )
            
            logger.info(f"Bug pattern analysis completed: {analysis_result['analysis_id']}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze bug patterns: {e}")
            return {}
    
    async def _analyze_temporal_patterns(self, bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in bug occurrence."""
        try:
            temporal_data = {
                "hourly_distribution": defaultdict(int),
                "daily_distribution": defaultdict(int),
                "weekly_distribution": defaultdict(int),
                "monthly_distribution": defaultdict(int)
            }
            
            for bug in bugs:
                created_at = bug.get("created_at", datetime.utcnow())
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)
                
                # Hourly pattern
                temporal_data["hourly_distribution"][created_at.hour] += 1
                
                # Daily pattern
                temporal_data["daily_distribution"][created_at.strftime("%A")] += 1
                
                # Weekly pattern
                week_number = created_at.isocalendar()[1]
                temporal_data["weekly_distribution"][week_number] += 1
                
                # Monthly pattern
                temporal_data["monthly_distribution"][created_at.strftime("%Y-%m")] += 1
            
            # Convert to regular dicts and find peaks
            patterns = {}
            for pattern_type, data in temporal_data.items():
                regular_dict = dict(data)
                patterns[pattern_type] = {
                    "distribution": regular_dict,
                    "peak_times": self._find_peak_times(regular_dict),
                    "total_occurrences": sum(regular_dict.values())
                }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze temporal patterns: {e}")
            return {}
    
    def _find_peak_times(self, distribution: Dict[str, int]) -> List[Dict[str, Any]]:
        """Find peak times in a distribution."""
        if not distribution:
            return []
        
        # Calculate average
        avg_value = sum(distribution.values()) / len(distribution)
        
        # Find values significantly above average (>150% of average)
        peaks = []
        for time_period, count in distribution.items():
            if count > avg_value * 1.5:
                peaks.append({
                    "time_period": time_period,
                    "count": count,
                    "percentage_above_average": ((count - avg_value) / avg_value) * 100
                })
        
        # Sort by count descending
        return sorted(peaks, key=lambda x: x["count"], reverse=True)
    
    async def _analyze_component_patterns(self, bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns by component/module."""
        try:
            component_data = defaultdict(lambda: {
                "bug_count": 0,
                "severity_distribution": defaultdict(int),
                "bug_types": defaultdict(int),
                "resolution_times": []
            })
            
            for bug in bugs:
                component = bug.get("component", "unknown")
                severity = bug.get("severity", "medium")
                bug_type = bug.get("type", "bug")
                
                component_data[component]["bug_count"] += 1
                component_data[component]["severity_distribution"][severity] += 1
                component_data[component]["bug_types"][bug_type] += 1
                
                # Calculate resolution time if available
                created_at = bug.get("created_at")
                resolved_at = bug.get("resolved_at")
                if created_at and resolved_at:
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at)
                    if isinstance(resolved_at, str):
                        resolved_at = datetime.fromisoformat(resolved_at)
                    
                    resolution_time = (resolved_at - created_at).total_seconds() / 3600  # hours
                    component_data[component]["resolution_times"].append(resolution_time)
            
            # Process component data
            component_patterns = {}
            for component, data in component_data.items():
                avg_resolution_time = (
                    sum(data["resolution_times"]) / len(data["resolution_times"])
                    if data["resolution_times"] else 0
                )
                
                component_patterns[component] = {
                    "bug_count": data["bug_count"],
                    "severity_distribution": dict(data["severity_distribution"]),
                    "bug_types": dict(data["bug_types"]),
                    "avg_resolution_time_hours": round(avg_resolution_time, 2),
                    "bug_density": data["bug_count"] / len(bugs) if bugs else 0
                }
            
            # Find problematic components
            sorted_components = sorted(
                component_patterns.items(),
                key=lambda x: x[1]["bug_count"],
                reverse=True
            )
            
            return {
                "component_analysis": component_patterns,
                "most_problematic_components": sorted_components[:5],
                "total_components_affected": len(component_patterns)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze component patterns: {e}")
            return {}
    
    async def _analyze_severity_patterns(self, bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns by bug severity."""
        try:
            severity_data = defaultdict(lambda: {
                "count": 0,
                "components": defaultdict(int),
                "resolution_times": [],
                "escalation_rate": 0
            })
            
            for bug in bugs:
                severity = bug.get("severity", "medium")
                component = bug.get("component", "unknown")
                
                severity_data[severity]["count"] += 1
                severity_data[severity]["components"][component] += 1
                
                # Track resolution time
                created_at = bug.get("created_at")
                resolved_at = bug.get("resolved_at")
                if created_at and resolved_at:
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at)
                    if isinstance(resolved_at, str):
                        resolved_at = datetime.fromisoformat(resolved_at)
                    
                    resolution_time = (resolved_at - created_at).total_seconds() / 3600
                    severity_data[severity]["resolution_times"].append(resolution_time)
                
                # Track escalations
                if bug.get("escalated", False):
                    severity_data[severity]["escalation_rate"] += 1
            
            # Process severity data
            severity_patterns = {}
            for severity, data in severity_data.items():
                avg_resolution_time = (
                    sum(data["resolution_times"]) / len(data["resolution_times"])
                    if data["resolution_times"] else 0
                )
                
                escalation_rate = (
                    data["escalation_rate"] / data["count"]
                    if data["count"] > 0 else 0
                )
                
                severity_patterns[severity] = {
                    "count": data["count"],
                    "percentage": (data["count"] / len(bugs)) * 100 if bugs else 0,
                    "avg_resolution_time_hours": round(avg_resolution_time, 2),
                    "escalation_rate": round(escalation_rate, 3),
                    "top_components": dict(Counter(data["components"]).most_common(3))
                }
            
            return {
                "severity_analysis": severity_patterns,
                "severity_distribution": {k: v["count"] for k, v in severity_patterns.items()}
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze severity patterns: {e}")
            return {}
    
    async def _analyze_root_cause_patterns(self, bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns by root cause."""
        try:
            nlp = await get_nlp_processor()
            
            root_cause_data = defaultdict(lambda: {
                "count": 0,
                "components": defaultdict(int),
                "severities": defaultdict(int),
                "keywords": []
            })
            
            # Extract root causes from bug descriptions and resolution notes
            for bug in bugs:
                root_cause = bug.get("root_cause", "")
                description = bug.get("description", "")
                resolution_notes = bug.get("resolution_notes", "")
                
                # If no explicit root cause, try to infer from text
                if not root_cause:
                    combined_text = f"{description} {resolution_notes}"
                    root_cause = await self._infer_root_cause(combined_text)
                
                if root_cause:
                    component = bug.get("component", "unknown")
                    severity = bug.get("severity", "medium")
                    
                    root_cause_data[root_cause]["count"] += 1
                    root_cause_data[root_cause]["components"][component] += 1
                    root_cause_data[root_cause]["severities"][severity] += 1
                    
                    # Extract keywords
                    keywords = await nlp.extract_keywords(f"{description} {resolution_notes}", 5)
                    root_cause_data[root_cause]["keywords"].extend(keywords)
            
            # Process root cause data
            root_cause_patterns = {}
            for cause, data in root_cause_data.items():
                # Get most common keywords
                keyword_counter = Counter(data["keywords"])
                common_keywords = [kw[0] for kw in keyword_counter.most_common(5)]
                
                root_cause_patterns[cause] = {
                    "count": data["count"],
                    "percentage": (data["count"] / len(bugs)) * 100 if bugs else 0,
                    "top_components": dict(Counter(data["components"]).most_common(3)),
                    "severity_distribution": dict(data["severities"]),
                    "common_keywords": common_keywords
                }
            
            # Sort by frequency
            sorted_causes = sorted(
                root_cause_patterns.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )
            
            return {
                "root_cause_analysis": root_cause_patterns,
                "most_common_causes": sorted_causes[:10]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze root cause patterns: {e}")
            return {}
    
    async def _infer_root_cause(self, text: str) -> str:
        """Infer root cause from bug description and resolution text."""
        try:
            text_lower = text.lower()
            
            # Common root cause patterns
            cause_patterns = {
                "logic_error": ["logic", "algorithm", "calculation", "condition", "if statement"],
                "null_pointer": ["null", "undefined", "nullpointer", "reference"],
                "memory_leak": ["memory", "leak", "garbage", "heap", "stack overflow"],
                "race_condition": ["race", "thread", "concurrent", "synchronization", "deadlock"],
                "configuration": ["config", "setting", "environment", "deployment"],
                "integration": ["api", "service", "integration", "external", "third party"],
                "validation": ["validation", "input", "sanitization", "security"],
                "performance": ["slow", "timeout", "performance", "optimization"],
                "ui_ux": ["ui", "ux", "interface", "display", "rendering"],
                "database": ["database", "sql", "query", "connection", "transaction"]
            }
            
            # Score each potential cause
            cause_scores = {}
            for cause, keywords in cause_patterns.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    cause_scores[cause] = score
            
            # Return the highest scoring cause
            if cause_scores:
                return max(cause_scores.items(), key=lambda x: x[1])[0]
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Failed to infer root cause: {e}")
            return "unknown"
    
    async def _analyze_developer_patterns(self, bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns by developer/assignee."""
        try:
            developer_data = defaultdict(lambda: {
                "bugs_assigned": 0,
                "bugs_created": 0,
                "resolution_times": [],
                "severity_distribution": defaultdict(int),
                "components": defaultdict(int)
            })
            
            for bug in bugs:
                assignee = bug.get("assignee", "unassigned")
                reporter = bug.get("reporter", "unknown")
                severity = bug.get("severity", "medium")
                component = bug.get("component", "unknown")
                
                # Track assignments
                developer_data[assignee]["bugs_assigned"] += 1
                developer_data[assignee]["severity_distribution"][severity] += 1
                developer_data[assignee]["components"][component] += 1
                
                # Track bug creation
                developer_data[reporter]["bugs_created"] += 1
                
                # Track resolution time
                created_at = bug.get("created_at")
                resolved_at = bug.get("resolved_at")
                if created_at and resolved_at and assignee != "unassigned":
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at)
                    if isinstance(resolved_at, str):
                        resolved_at = datetime.fromisoformat(resolved_at)
                    
                    resolution_time = (resolved_at - created_at).total_seconds() / 3600
                    developer_data[assignee]["resolution_times"].append(resolution_time)
            
            # Process developer data
            developer_patterns = {}
            for developer, data in developer_data.items():
                avg_resolution_time = (
                    sum(data["resolution_times"]) / len(data["resolution_times"])
                    if data["resolution_times"] else 0
                )
                
                developer_patterns[developer] = {
                    "bugs_assigned": data["bugs_assigned"],
                    "bugs_created": data["bugs_created"],
                    "avg_resolution_time_hours": round(avg_resolution_time, 2),
                    "severity_distribution": dict(data["severity_distribution"]),
                    "top_components": dict(Counter(data["components"]).most_common(3)),
                    "productivity_score": self._calculate_productivity_score(data)
                }
            
            return {
                "developer_analysis": developer_patterns,
                "top_performers": self._identify_top_performers(developer_patterns),
                "areas_for_improvement": self._identify_improvement_areas(developer_patterns)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze developer patterns: {e}")
            return {}
    
    def _calculate_productivity_score(self, developer_data: Dict[str, Any]) -> float:
        """Calculate a productivity score for a developer."""
        try:
            bugs_assigned = developer_data["bugs_assigned"]
            resolution_times = developer_data["resolution_times"]
            
            if not bugs_assigned or not resolution_times:
                return 0.0
            
            # Base score from number of bugs resolved
            base_score = min(bugs_assigned / 10, 1.0)  # Normalize to max 1.0
            
            # Adjust for resolution speed (lower time = higher score)
            avg_resolution_time = sum(resolution_times) / len(resolution_times)
            speed_factor = max(0.1, 1.0 - (avg_resolution_time / 168))  # 168 hours = 1 week
            
            return round(base_score * speed_factor, 2)
            
        except Exception as e:
            logger.error(f"Failed to calculate productivity score: {e}")
            return 0.0
    
    def _identify_top_performers(self, developer_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify top performing developers."""
        performers = []
        
        for developer, data in developer_patterns.items():
            if developer not in ["unassigned", "unknown"] and data["bugs_assigned"] > 0:
                performers.append({
                    "developer": developer,
                    "productivity_score": data["productivity_score"],
                    "bugs_resolved": data["bugs_assigned"],
                    "avg_resolution_time": data["avg_resolution_time_hours"]
                })
        
        # Sort by productivity score
        return sorted(performers, key=lambda x: x["productivity_score"], reverse=True)[:5]
    
    def _identify_improvement_areas(self, developer_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify areas for improvement."""
        improvement_areas = []
        
        for developer, data in developer_patterns.items():
            if developer not in ["unassigned", "unknown"] and data["bugs_assigned"] > 0:
                issues = []
                
                # High resolution time
                if data["avg_resolution_time_hours"] > 72:  # More than 3 days
                    issues.append("High average resolution time")
                
                # High severity bugs
                high_severity = data["severity_distribution"].get("high", 0) + data["severity_distribution"].get("critical", 0)
                if high_severity > data["bugs_assigned"] * 0.3:  # More than 30% high severity
                    issues.append("High proportion of severe bugs")
                
                if issues:
                    improvement_areas.append({
                        "developer": developer,
                        "issues": issues,
                        "suggestions": self._generate_improvement_suggestions(issues)
                    })
        
        return improvement_areas
    
    def _generate_improvement_suggestions(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on identified issues."""
        suggestions = []
        
        for issue in issues:
            if "resolution time" in issue:
                suggestions.extend([
                    "Provide additional training on debugging techniques",
                    "Implement pair programming for complex bugs",
                    "Review code complexity in assigned components"
                ])
            elif "severe bugs" in issue:
                suggestions.extend([
                    "Increase code review rigor",
                    "Implement additional testing procedures",
                    "Provide training on critical system components"
                ])
        
        return list(set(suggestions))  # Remove duplicates
    
    async def _analyze_text_similarity_patterns(self, bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze text similarity patterns to find duplicate or related bugs."""
        try:
            nlp = await get_nlp_processor()
            
            # Group bugs by similarity
            similarity_clusters = []
            processed_bugs = set()
            
            for i, bug1 in enumerate(bugs):
                if bug1.get("bug_id") in processed_bugs:
                    continue
                
                cluster = [bug1]
                bug1_text = f"{bug1.get('title', '')} {bug1.get('description', '')}"
                
                for j, bug2 in enumerate(bugs[i+1:], i+1):
                    if bug2.get("bug_id") in processed_bugs:
                        continue
                    
                    bug2_text = f"{bug2.get('title', '')} {bug2.get('description', '')}"
                    similarity = await nlp.calculate_similarity(bug1_text, bug2_text)
                    
                    if similarity > 0.7:  # High similarity threshold
                        cluster.append(bug2)
                        processed_bugs.add(bug2.get("bug_id"))
                
                if len(cluster) > 1:  # Only include clusters with multiple bugs
                    similarity_clusters.append({
                        "cluster_id": f"CLUSTER_{len(similarity_clusters)+1}",
                        "bugs": cluster,
                        "bug_count": len(cluster),
                        "similarity_score": 0.7  # Minimum threshold used
                    })
                
                processed_bugs.add(bug1.get("bug_id"))
            
            # Analyze cluster patterns
            cluster_analysis = {
                "total_clusters": len(similarity_clusters),
                "total_similar_bugs": sum(cluster["bug_count"] for cluster in similarity_clusters),
                "largest_cluster_size": max((cluster["bug_count"] for cluster in similarity_clusters), default=0),
                "clusters": similarity_clusters[:10]  # Top 10 clusters
            }
            
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze text similarity patterns: {e}")
            return {}
    
    async def _generate_pattern_insights(self, patterns: Dict[str, Any], 
                                       bugs: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from identified patterns."""
        try:
            insights = []
            
            # Temporal insights
            temporal = patterns.get("temporal_patterns", {})
            if temporal:
                hourly_peaks = temporal.get("hourly_distribution", {}).get("peak_times", [])
                if hourly_peaks:
                    peak_hour = hourly_peaks[0]["time_period"]
                    insights.append(f"Most bugs occur at {peak_hour}:00 - consider deployment timing")
                
                daily_peaks = temporal.get("daily_distribution", {}).get("peak_times", [])
                if daily_peaks:
                    peak_day = daily_peaks[0]["time_period"]
                    insights.append(f"Most bugs occur on {peak_day} - review release schedules")
            
            # Component insights
            component = patterns.get("component_patterns", {})
            if component:
                problematic = component.get("most_problematic_components", [])
                if problematic:
                    top_component = problematic[0][0]
                    bug_count = problematic[0][1]["bug_count"]
                    insights.append(f"Component '{top_component}' has {bug_count} bugs - needs attention")
            
            # Severity insights
            severity = patterns.get("severity_patterns", {})
            if severity:
                severity_dist = severity.get("severity_distribution", {})
                critical_count = severity_dist.get("critical", 0)
                high_count = severity_dist.get("high", 0)
                
                if critical_count + high_count > len(bugs) * 0.3:
                    insights.append("High proportion of severe bugs - review quality processes")
            
            # Root cause insights
            root_cause = patterns.get("root_cause_patterns", {})
            if root_cause:
                common_causes = root_cause.get("most_common_causes", [])
                if common_causes:
                    top_cause = common_causes[0][0]
                    cause_count = common_causes[0][1]["count"]
                    insights.append(f"Most common root cause is '{top_cause}' ({cause_count} bugs)")
            
            # Similarity insights
            similarity = patterns.get("text_similarity_patterns", {})
            if similarity:
                similar_bugs = similarity.get("total_similar_bugs", 0)
                if similar_bugs > 0:
                    insights.append(f"{similar_bugs} bugs appear to be duplicates or highly related")
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate pattern insights: {e}")
            return []
    
    async def _generate_bug_predictions(self, patterns: Dict[str, Any], 
                                      bugs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate predictions based on identified patterns."""
        try:
            predictions = {}
            
            # Predict next bug hotspots
            component_patterns = patterns.get("component_patterns", {})
            if component_patterns:
                component_analysis = component_patterns.get("component_analysis", {})
                hotspots = []
                
                for component, data in component_analysis.items():
                    if data["bug_density"] > 0.1:  # More than 10% of bugs
                        hotspots.append({
                            "component": component,
                            "risk_score": data["bug_density"],
                            "predicted_bugs_next_month": int(data["bug_count"] * 0.3)  # 30% of current rate
                        })
                
                predictions["bug_hotspots"] = sorted(hotspots, key=lambda x: x["risk_score"], reverse=True)[:5]
            
            # Predict resolution times
            severity_patterns = patterns.get("severity_patterns", {})
            if severity_patterns:
                severity_analysis = severity_patterns.get("severity_analysis", {})
                resolution_predictions = {}
                
                for severity, data in severity_analysis.items():
                    resolution_predictions[severity] = {
                        "predicted_resolution_hours": data.get("avg_resolution_time_hours", 24),
                        "confidence": 0.7 if data.get("count", 0) > 5 else 0.4
                    }
                
                predictions["resolution_time_predictions"] = resolution_predictions
            
            # Predict developer workload
            developer_patterns = patterns.get("developer_patterns", {})
            if developer_patterns:
                developer_analysis = developer_patterns.get("developer_analysis", {})
                workload_predictions = []
                
                for developer, data in developer_analysis.items():
                    if developer not in ["unassigned", "unknown"] and data["bugs_assigned"] > 0:
                        predicted_load = int(data["bugs_assigned"] * 0.8)  # 80% of current rate
                        workload_predictions.append({
                            "developer": developer,
                            "predicted_monthly_bugs": predicted_load,
                            "capacity_utilization": min(predicted_load / 20, 1.0)  # Assume 20 bugs/month capacity
                        })
                
                predictions["developer_workload"] = sorted(
                    workload_predictions, 
                    key=lambda x: x["capacity_utilization"], 
                    reverse=True
                )[:10]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to generate bug predictions: {e}")
            return {}
    
    async def _generate_pattern_recommendations(self, patterns: Dict[str, Any], 
                                              insights: List[str]) -> List[str]:
        """Generate actionable recommendations based on patterns and insights."""
        try:
            recommendations = []
            
            # Component-based recommendations
            component_patterns = patterns.get("component_patterns", {})
            if component_patterns:
                problematic = component_patterns.get("most_problematic_components", []) 
                problematic = component_patterns.get("most_problematic_components", [])
                if problematic:
                    for component, data in problematic[:3]:  # Top 3 problematic components
                        recommendations.append(f"Increase code review focus on '{component}' component")
                        recommendations.append(f"Consider refactoring '{component}' to reduce complexity")
            
            # Temporal-based recommendations
            temporal_patterns = patterns.get("temporal_patterns", {})
            if temporal_patterns:
                hourly_peaks = temporal_patterns.get("hourly_distribution", {}).get("peak_times", [])
                if hourly_peaks:
                    recommendations.append("Avoid deployments during peak bug occurrence hours")
                
                daily_peaks = temporal_patterns.get("daily_distribution", {}).get("peak_times", [])
                if daily_peaks:
                    recommendations.append("Implement additional testing before high-bug-occurrence days")
            
            # Root cause-based recommendations
            root_cause_patterns = patterns.get("root_cause_patterns", {})
            if root_cause_patterns:
                common_causes = root_cause_patterns.get("most_common_causes", [])
                for cause, data in common_causes[:3]:  # Top 3 causes
                    if cause == "logic_error":
                        recommendations.append("Implement more comprehensive unit testing")
                        recommendations.append("Conduct algorithm review sessions")
                    elif cause == "null_pointer":
                        recommendations.append("Implement null safety checks")
                        recommendations.append("Use optional types where appropriate")
                    elif cause == "integration":
                        recommendations.append("Improve API testing and mocking")
                        recommendations.append("Implement contract testing")
            
            # Developer-based recommendations
            developer_patterns = patterns.get("developer_patterns", {})
            if developer_patterns:
                improvement_areas = developer_patterns.get("areas_for_improvement", [])
                for area in improvement_areas:
                    recommendations.extend(area.get("suggestions", []))
            
            # Similarity-based recommendations
            similarity_patterns = patterns.get("text_similarity_patterns", {})
            if similarity_patterns:
                similar_bugs = similarity_patterns.get("total_similar_bugs", 0)
                if similar_bugs > 0:
                    recommendations.append("Implement duplicate bug detection in bug tracking system")
                    recommendations.append("Create bug templates to improve consistency")
            
            return list(set(recommendations))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to generate pattern recommendations: {e}")
            return []
