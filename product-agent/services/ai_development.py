"""AI-powered development services for Product/Engineering Agent."""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import openai
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger
import uuid
import re
from collections import defaultdict

from shared_libs.database import get_database
from shared_libs.intelligence import get_nlp_processor, get_ml_predictor
from shared_libs.data_lake import get_data_lake

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY", "")


class AIDevelopmentService:
    """AI-powered development and project management service."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.sprint_plans_collection = "sprint_plans"
        self.story_translations_collection = "story_translations"
        self.bug_patterns_collection = "bug_patterns"
        self.effort_estimates_collection = "effort_estimates"
        self.code_analysis_collection = "code_analysis"
    
    async def initialize(self):
        """Initialize the AI development service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.sprint_plans_collection].create_index("plan_id", unique=True)
        await self.db[self.sprint_plans_collection].create_index("sprint_id")
        await self.db[self.sprint_plans_collection].create_index("created_at")
        
        await self.db[self.story_translations_collection].create_index("translation_id", unique=True)
        await self.db[self.story_translations_collection].create_index("story_id")
        
        await self.db[self.bug_patterns_collection].create_index("pattern_id", unique=True)
        await self.db[self.bug_patterns_collection].create_index("pattern_type")
        await self.db[self.bug_patterns_collection].create_index("confidence_score")
        
        await self.db[self.effort_estimates_collection].create_index("estimate_id", unique=True)
        await self.db[self.effort_estimates_collection].create_index("story_id")
        
        await self.db[self.code_analysis_collection].create_index("analysis_id", unique=True)
        await self.db[self.code_analysis_collection].create_index("repository")
        
        logger.info("AI Development service initialized")
    
    async def generate_ai_sprint_plan(self, sprint_id: str, team_data: Dict[str, Any], 
                                    backlog_items: List[Dict[str, Any]], 
                                    sprint_duration: int = 14) -> Dict[str, Any]:
        """Generate AI-based sprint planning with capacity analysis and story prioritization."""
        try:
            data_lake = await get_data_lake()
            
            # Analyze team capacity
            team_capacity = await self._analyze_team_capacity(team_data, sprint_duration)
            
            # Analyze backlog items
            backlog_analysis = await self._analyze_backlog_items(backlog_items)
            
            # Get historical sprint data for better planning
            historical_data = await self._get_historical_sprint_data(team_data.get("team_id"))
            
            # Generate story point estimates for unestimated items
            estimated_items = []
            for item in backlog_items:
                if not item.get("story_points"):
                    estimate = await self.estimate_story_effort(item.get("story_id", ""), item)
                    item["story_points"] = estimate.get("estimated_points", 5)
                    item["confidence"] = estimate.get("confidence", 0.5)
                estimated_items.append(item)
            
            # Prioritize stories using AI
            prioritized_stories = await self._prioritize_stories_with_ai(estimated_items, team_data, historical_data)
            
            # Create optimal sprint plan
            sprint_plan = await self._create_optimal_sprint_plan(
                prioritized_stories, team_capacity, sprint_duration, historical_data
            )
            
            # Generate risk assessment
            risk_assessment = await self._assess_sprint_risks(sprint_plan, team_capacity, historical_data)
            
            # Generate recommendations
            recommendations = await self._generate_sprint_recommendations(
                sprint_plan, team_capacity, risk_assessment, historical_data
            )
            
            ai_sprint_plan = {
                "plan_id": f"SP_{str(uuid.uuid4())[:8].upper()}",
                "sprint_id": sprint_id,
                "team_capacity": team_capacity,
                "backlog_analysis": backlog_analysis,
                "prioritized_stories": prioritized_stories,
                "sprint_plan": sprint_plan,
                "risk_assessment": risk_assessment,
                "recommendations": recommendations,
                "confidence_score": self._calculate_plan_confidence(sprint_plan, team_capacity, historical_data),
                "created_at": datetime.utcnow(),
                "sprint_duration_days": sprint_duration
            }
            
            # Store sprint plan
            await self.db[self.sprint_plans_collection].insert_one(ai_sprint_plan)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="product",
                event_type="sprint_plan_generated",
                entity_type="sprint",
                entity_id=sprint_id,
                data={
                    "plan_id": ai_sprint_plan["plan_id"],
                    "total_stories": len(sprint_plan.get("selected_stories", [])),
                    "total_points": sprint_plan.get("total_points", 0),
                    "confidence_score": ai_sprint_plan["confidence_score"]
                }
            )
            
            logger.info(f"AI sprint plan generated: {ai_sprint_plan['plan_id']} for sprint {sprint_id}")
            
            return ai_sprint_plan
            
        except Exception as e:
            logger.error(f"Failed to generate AI sprint plan: {e}")
            return {}
    
    async def _analyze_team_capacity(self, team_data: Dict[str, Any], sprint_duration: int) -> Dict[str, Any]:
        """Analyze team capacity for sprint planning."""
        try:
            team_members = team_data.get("members", [])
            
            total_capacity = 0
            member_capacities = []
            
            for member in team_members:
                # Base capacity (hours per day * working days)
                working_hours_per_day = member.get("working_hours_per_day", 8)
                working_days = sprint_duration - 4  # Subtract weekends for 2-week sprint
                
                # Adjust for availability
                availability = member.get("availability", 1.0)  # 0.0 to 1.0
                
                # Adjust for meetings and other commitments
                meeting_overhead = member.get("meeting_overhead", 0.2)  # 20% for meetings
                
                # Calculate effective capacity
                base_capacity = working_hours_per_day * working_days
                effective_capacity = base_capacity * availability * (1 - meeting_overhead)
                
                # Convert to story points (assume 1 story point = 4 hours)
                capacity_points = effective_capacity / 4
                
                member_capacity = {
                    "member_id": member.get("member_id"),
                    "name": member.get("name"),
                    "role": member.get("role"),
                    "base_capacity_hours": base_capacity,
                    "effective_capacity_hours": effective_capacity,
                    "capacity_points": capacity_points,
                    "availability": availability,
                    "skills": member.get("skills", [])
                }
                
                member_capacities.append(member_capacity)
                total_capacity += capacity_points
            
            # Adjust for team dynamics and historical performance
            team_velocity_factor = team_data.get("historical_velocity_factor", 0.8)  # Teams usually deliver 80% of planned
            adjusted_capacity = total_capacity * team_velocity_factor
            
            return {
                "total_capacity_points": adjusted_capacity,
                "raw_capacity_points": total_capacity,
                "velocity_factor": team_velocity_factor,
                "member_capacities": member_capacities,
                "team_size": len(team_members),
                "sprint_duration_days": sprint_duration
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze team capacity: {e}")
            return {"total_capacity_points": 40, "member_capacities": []}
    
    async def _analyze_backlog_items(self, backlog_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze backlog items for planning insights."""
        try:
            nlp = await get_nlp_processor()
            
            analysis = {
                "total_items": len(backlog_items),
                "priority_distribution": defaultdict(int),
                "complexity_distribution": defaultdict(int),
                "category_distribution": defaultdict(int),
                "estimated_items": 0,
                "unestimated_items": 0,
                "blocked_items": 0,
                "ready_items": 0
            }
            
            for item in backlog_items:
                # Priority analysis
                priority = item.get("priority", "medium").lower()
                analysis["priority_distribution"][priority] += 1
                
                # Complexity analysis
                story_points = item.get("story_points", 0)
                if story_points > 0:
                    analysis["estimated_items"] += 1
                    if story_points <= 3:
                        complexity = "simple"
                    elif story_points <= 8:
                        complexity = "medium"
                    else:
                        complexity = "complex"
                    analysis["complexity_distribution"][complexity] += 1
                else:
                    analysis["unestimated_items"] += 1
                
                # Category analysis
                category = item.get("category", "feature").lower()
                analysis["category_distribution"][category] += 1
                
                # Status analysis
                status = item.get("status", "").lower()
                if "blocked" in status:
                    analysis["blocked_items"] += 1
                elif "ready" in status or status == "todo":
                    analysis["ready_items"] += 1
            
            # Convert defaultdicts to regular dicts
            analysis["priority_distribution"] = dict(analysis["priority_distribution"])
            analysis["complexity_distribution"] = dict(analysis["complexity_distribution"])
            analysis["category_distribution"] = dict(analysis["category_distribution"])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze backlog items: {e}")
            return {"total_items": 0}
    
    async def _get_historical_sprint_data(self, team_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get historical sprint data for the team."""
        try:
            # Get last 10 sprints for the team
            sprints = await self.db["sprints"].find({
                "team_id": team_id,
                "status": "completed"
            }).sort("end_date", -1).limit(limit).to_list(None)
            
            if not sprints:
                return {"average_velocity": 30, "completion_rate": 0.8, "sprints_analyzed": 0}
            
            velocities = []
            completion_rates = []
            
            for sprint in sprints:
                planned_points = sprint.get("planned_points", 0)
                completed_points = sprint.get("completed_points", 0)
                
                if planned_points > 0:
                    velocities.append(completed_points)
                    completion_rates.append(completed_points / planned_points)
            
            avg_velocity = sum(velocities) / len(velocities) if velocities else 30
            avg_completion_rate = sum(completion_rates) / len(completion_rates) if completion_rates else 0.8
            
            return {
                "average_velocity": avg_velocity,
                "completion_rate": avg_completion_rate,
                "sprints_analyzed": len(sprints),
                "velocity_trend": self._calculate_velocity_trend(velocities),
                "historical_sprints": sprints
            }
            
        except Exception as e:
            logger.error(f"Failed to get historical sprint data: {e}")
            return {"average_velocity": 30, "completion_rate": 0.8, "sprints_analyzed": 0}
    
    def _calculate_velocity_trend(self, velocities: List[float]) -> str:
        """Calculate velocity trend from historical data."""
        if len(velocities) < 3:
            return "insufficient_data"
        
        recent_avg = sum(velocities[:3]) / 3
        older_avg = sum(velocities[3:6]) / min(3, len(velocities[3:6])) if len(velocities) > 3 else recent_avg
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    async def _prioritize_stories_with_ai(self, stories: List[Dict[str, Any]], 
                                        team_data: Dict[str, Any], 
                                        historical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize stories using AI analysis."""
        try:
            nlp = await get_nlp_processor()
            
            # Score each story
            scored_stories = []
            
            for story in stories:
                score = 0.0
                
                # Business value score
                priority = story.get("priority", "medium").lower()
                if priority == "critical":
                    score += 10
                elif priority == "high":
                    score += 7
                elif priority == "medium":
                    score += 5
                elif priority == "low":
                    score += 2
                
                # Effort vs value ratio
                story_points = story.get("story_points", 5)
                business_value = story.get("business_value", 5)  # 1-10 scale
                if story_points > 0:
                    value_ratio = business_value / story_points
                    score += value_ratio * 2
                
                # Dependencies and blockers
                dependencies = story.get("dependencies", [])
                if not dependencies:
                    score += 2  # Bonus for no dependencies
                
                blocked = story.get("blocked", False)
                if blocked:
                    score -= 5  # Penalty for blocked items
                
                # Team skills alignment
                required_skills = story.get("required_skills", [])
                team_skills = []
                for member in team_data.get("members", []):
                    team_skills.extend(member.get("skills", []))
                
                skill_match = len(set(required_skills) & set(team_skills)) / len(required_skills) if required_skills else 1
                score += skill_match * 3
                
                # Technical debt consideration
                if story.get("category") == "technical_debt":
                    score += 1  # Small bonus for tech debt
                
                # Customer impact
                customer_impact = story.get("customer_impact", "medium").lower()
                if customer_impact == "high":
                    score += 3
                elif customer_impact == "medium":
                    score += 1
                
                story["ai_priority_score"] = score
                scored_stories.append(story)
            
            # Sort by score (descending)
            return sorted(scored_stories, key=lambda x: x.get("ai_priority_score", 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to prioritize stories with AI: {e}")
            return stories
    
    async def _create_optimal_sprint_plan(self, prioritized_stories: List[Dict[str, Any]], 
                                        team_capacity: Dict[str, Any], 
                                        sprint_duration: int, 
                                        historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimal sprint plan using capacity and priorities."""
        try:
            available_capacity = team_capacity.get("total_capacity_points", 40)
            selected_stories = []
            total_points = 0
            
            # Select stories that fit within capacity
            for story in prioritized_stories:
                story_points = story.get("story_points", 5)
                
                # Check if story fits in remaining capacity
                if total_points + story_points <= available_capacity:
                    # Check dependencies
                    dependencies = story.get("dependencies", [])
                    dependencies_met = True
                    
                    for dep_id in dependencies:
                        # Check if dependency is already in selected stories or completed
                        dep_selected = any(s.get("story_id") == dep_id for s in selected_stories)
                        dep_completed = story.get("dependency_status", {}).get(dep_id) == "completed"
                        
                        if not (dep_selected or dep_completed):
                            dependencies_met = False
                            break
                    
                    if dependencies_met and not story.get("blocked", False):
                        selected_stories.append(story)
                        total_points += story_points
            
            # Calculate capacity utilization
            capacity_utilization = total_points / available_capacity if available_capacity > 0 else 0
            
            # Generate daily breakdown
            daily_breakdown = self._generate_daily_breakdown(selected_stories, sprint_duration)
            
            return {
                "selected_stories": selected_stories,
                "total_points": total_points,
                "available_capacity": available_capacity,
                "capacity_utilization": capacity_utilization,
                "stories_count": len(selected_stories),
                "daily_breakdown": daily_breakdown,
                "buffer_capacity": available_capacity - total_points
            }
            
        except Exception as e:
            logger.error(f"Failed to create optimal sprint plan: {e}")
            return {"selected_stories": [], "total_points": 0}
    
    def _generate_daily_breakdown(self, selected_stories: List[Dict[str, Any]], 
                                sprint_duration: int) -> List[Dict[str, Any]]:
        """Generate daily breakdown of sprint work."""
        try:
            daily_breakdown = []
            total_points = sum(story.get("story_points", 0) for story in selected_stories)
            
            # Distribute work across sprint days (excluding weekends)
            working_days = sprint_duration - 4  # Subtract weekends for 2-week sprint
            points_per_day = total_points / working_days if working_days > 0 else 0
            
            current_day = 1
            remaining_points = total_points
            
            for day in range(working_days):
                day_points = min(points_per_day, remaining_points)
                
                daily_breakdown.append({
                    "day": current_day,
                    "planned_points": round(day_points, 1),
                    "cumulative_points": round(total_points - remaining_points + day_points, 1),
                    "completion_percentage": round(((total_points - remaining_points + day_points) / total_points) * 100, 1) if total_points > 0 else 0
                })
                
                remaining_points -= day_points
                current_day += 1
            
            return daily_breakdown
            
        except Exception as e:
            logger.error(f"Failed to generate daily breakdown: {e}")
            return []
    
    async def _assess_sprint_risks(self, sprint_plan: Dict[str, Any], 
                                 team_capacity: Dict[str, Any], 
                                 historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks for the sprint plan."""
        try:
            risks = []
            risk_score = 0.0
            
            # Capacity utilization risk
            utilization = sprint_plan.get("capacity_utilization", 0)
            if utilization > 0.95:
                risks.append({
                    "type": "over_capacity",
                    "severity": "high",
                    "description": f"Sprint is {utilization*100:.1f}% of capacity - very high risk of not completing all work",
                    "mitigation": "Consider removing lower priority stories"
                })
                risk_score += 0.3
            elif utilization > 0.85:
                risks.append({
                    "type": "high_capacity",
                    "severity": "medium",
                    "description": f"Sprint is {utilization*100:.1f}% of capacity - limited buffer for unexpected work",
                    "mitigation": "Monitor progress closely and be prepared to descope"
                })
                risk_score += 0.2
            
            # Team availability risk
            member_capacities = team_capacity.get("member_capacities", [])
            low_availability_members = [m for m in member_capacities if m.get("availability", 1.0) < 0.8]
            if low_availability_members:
                risks.append({
                    "type": "team_availability",
                    "severity": "medium",
                    "description": f"{len(low_availability_members)} team members have reduced availability",
                    "mitigation": "Redistribute work or adjust expectations"
                })
                risk_score += 0.15
            
            # Dependency risk
            selected_stories = sprint_plan.get("selected_stories", [])
            stories_with_deps = [s for s in selected_stories if s.get("dependencies", [])]
            if stories_with_deps:
                risks.append({
                    "type": "dependencies",
                    "severity": "medium",
                    "description": f"{len(stories_with_deps)} stories have dependencies that could cause delays",
                    "mitigation": "Prioritize dependency resolution and have backup stories ready"
                })
                risk_score += 0.1
            
            # Historical performance risk
            completion_rate = historical_data.get("completion_rate", 0.8)
            if completion_rate < 0.7:
                risks.append({
                    "type": "historical_performance",
                    "severity": "high",
                    "description": f"Team historically completes only {completion_rate*100:.1f}% of planned work",
                    "mitigation": "Reduce sprint scope by 20-30%"
                })
                risk_score += 0.25
            
            # Complexity risk
            complex_stories = [s for s in selected_stories if s.get("story_points", 0) > 8]
            if complex_stories:
                risks.append({
                    "type": "complexity",
                    "severity": "medium",
                    "description": f"{len(complex_stories)} stories are highly complex (>8 points)",
                    "mitigation": "Break down complex stories or ensure senior developers are assigned"
                })
                risk_score += 0.1
            
            # Determine overall risk level
            if risk_score >= 0.6:
                risk_level = "high"
            elif risk_score >= 0.3:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "overall_risk_level": risk_level,
                "risk_score": min(risk_score, 1.0),
                "identified_risks": risks,
                "risk_count": len(risks)
            }
            
        except Exception as e:
            logger.error(f"Failed to assess sprint risks: {e}")
            return {"overall_risk_level": "medium", "risk_score": 0.5, "identified_risks": []}
    
    async def _generate_sprint_recommendations(self, sprint_plan: Dict[str, Any], 
                                             team_capacity: Dict[str, Any], 
                                             risk_assessment: Dict[str, Any], 
                                             historical_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for sprint success."""
        try:
            recommendations = []
            
            # Capacity-based recommendations
            utilization = sprint_plan.get("capacity_utilization", 0)
            if utilization > 0.9:
                recommendations.append("Consider reducing scope - sprint is at very high capacity")
            elif utilization < 0.7:
                recommendations.append("Sprint has extra capacity - consider adding more stories")
            
            # Risk-based recommendations
            risk_level = risk_assessment.get("overall_risk_level", "medium")
            if risk_level == "high":
                recommendations.append("High risk sprint - implement daily standups and close monitoring")
                recommendations.append("Prepare backup stories in case primary stories are blocked")
            
            # Historical performance recommendations
            completion_rate = historical_data.get("completion_rate", 0.8)
            if completion_rate < 0.8:
                recommendations.append(f"Team completes {completion_rate*100:.0f}% on average - consider reducing planned work by {(1-completion_rate)*100:.0f}%")
            
            # Team composition recommendations
            member_capacities = team_capacity.get("member_capacities", [])
            if len(member_capacities) < 3:
                recommendations.append("Small team size - ensure knowledge sharing and avoid single points of failure")
            
            # Story-specific recommendations
            selected_stories = sprint_plan.get("selected_stories", [])
            unestimated_stories = [s for s in selected_stories if not s.get("story_points")]
            if unestimated_stories:
                recommendations.append(f"{len(unestimated_stories)} stories lack estimates - conduct estimation session")
            
            # Velocity trend recommendations
            velocity_trend = historical_data.get("velocity_trend", "stable")
            if velocity_trend == "decreasing":
                recommendations.append("Team velocity is decreasing - investigate impediments and team health")
            elif velocity_trend == "increasing":
                recommendations.append("Team velocity is improving - consider gradually increasing capacity")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate sprint recommendations: {e}")
            return []
    
    def _calculate_plan_confidence(self, sprint_plan: Dict[str, Any], 
                                 team_capacity: Dict[str, Any], 
                                 historical_data: Dict[str, Any]) -> float:
        """Calculate confidence score for the sprint plan."""
        try:
            confidence = 0.5  # Base confidence
            
            # Historical data confidence
            sprints_analyzed = historical_data.get("sprints_analyzed", 0)
            if sprints_analyzed >= 5:
                confidence += 0.2
            elif sprints_analyzed >= 3:
                confidence += 0.1
            
            # Capacity utilization confidence
            utilization = sprint_plan.get("capacity_utilization", 0)
            if 0.7 <= utilization <= 0.85:
                confidence += 0.2  # Optimal range
            elif utilization > 0.95:
                confidence -= 0.3  # Over capacity
            
            # Team stability confidence
            member_count = len(team_capacity.get("member_capacities", []))
            if member_count >= 3:
                confidence += 0.1
            
            # Estimation confidence
            selected_stories = sprint_plan.get("selected_stories", [])
            estimated_stories = [s for s in selected_stories if s.get("story_points", 0) > 0]
            if len(estimated_stories) == len(selected_stories) and selected_stories:
                confidence += 0.1
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate plan confidence: {e}")
            return 0.5
    
    async def translate_story_to_code_suggestions(self, story_id: str, 
                                                story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate user story to code implementation suggestions."""
        try:
            data_lake = await get_data_lake()
            
            # Extract story details
            title = story_data.get("title", "")
            description = story_data.get("description", "")
            acceptance_criteria = story_data.get("acceptance_criteria", [])
            
            # Analyze story content using NLP
            nlp = await get_nlp_processor()
            story_text = f"{title} {description} {' '.join(acceptance_criteria)}"
            
            # Extract technical keywords and concepts
            keywords = await nlp.extract_keywords(story_text, 15)
            
            # Generate code suggestions using AI
            code_suggestions = await self._generate_code_suggestions(story_data, keywords)
            
            # Analyze technical requirements
            tech_requirements = await self._analyze_technical_requirements(story_data, keywords)
            
            # Generate API design suggestions
            api_suggestions = await self._generate_api_suggestions(story_data, tech_requirements)
            
            # Generate database schema suggestions
            db_suggestions = await self._generate_database_suggestions(story_data, tech_requirements)
            
            # Generate test case suggestions
            test_suggestions = await self._generate_test_suggestions(story_data, acceptance_criteria)
            
            # Estimate implementation complexity
            complexity_analysis = await self._analyze_implementation_complexity(
                code_suggestions, tech_requirements, story_data
            )
            
            translation_result = {
                "translation_id": f"ST_{str(uuid.uuid4())[:8].upper()}",
                "story_id": story_id,
                "code_suggestions": code_suggestions,
                "technical_requirements": tech_requirements,
                "api_suggestions": api_suggestions,
                "database_suggestions": db_suggestions,
                "test_suggestions": test_suggestions,
                "complexity_analysis": complexity_analysis,
                "implementation_steps": await self._generate_implementation_steps(code_suggestions, tech_requirements),
                "estimated_files_to_modify": complexity_analysis.get("files_to_modify", []),
                "confidence_score": self._calculate_translation_confidence(code_suggestions, tech_requirements),
                "created_at": datetime.utcnow()
            }
            
            # Store translation
            await self.db[self.story_translations_collection].insert_one(translation_result)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="product",
                event_type="story_translated_to_code",
                entity_type="story",
                entity_id=story_id,
                data={
                    "translation_id": translation_result["translation_id"],
                    "complexity_level": complexity_analysis.get("complexity_level", "medium"),
                    "estimated_files": len(complexity_analysis.get("files_to_modify", [])),
                    "confidence_score": translation_result["confidence_score"]
                }
            )
            
            logger.info(f"Story translated to code suggestions: {story_id} -> {translation_result['translation_id']}")
            
            return translation_result
            
        except Exception as e:
            logger.error(f"Failed to translate story to code suggestions: {e}")
            return {}
    
    async def _generate_code_suggestions(self, story_data: Dict[str, Any], 
                                       keywords: List[str]) -> Dict[str, Any]:
        """Generate code implementation suggestions using AI."""
        try:
            if not openai.api_key:
                return self._generate_fallback_code_suggestions(story_data, keywords)
            
            story_text = f"""
            Title: {story_data.get('title', '')}
            Description: {story_data.get('description', '')}
            Acceptance Criteria: {'; '.join(story_data.get('acceptance_criteria', []))}
            """
            
            prompt = f"""
            Analyze this user story and provide code implementation suggestions:
            
            {story_text}
            
            Please provide suggestions in the following JSON format:
            {{
                "frontend_components": ["ComponentName1", "ComponentName2"],
                "backend_endpoints": [
                    {{"method": "GET", "path": "/api/resource", "description": "Get resource"}},
                    {{"method": "POST", "path": "/api/resource", "description": "Create resource"}}
                ],
                "database_operations": ["CREATE", "READ", "UPDATE", "DELETE"],
                "business_logic": ["ValidationService", "ProcessingService"],
                "integration_points": ["ExternalAPI", "EmailService"],
                "security_considerations": ["Authentication", "Authorization"],
                "performance_considerations": ["Caching", "Indexing"],
                "suggested_architecture": "MVC/Microservices/etc"
            }}
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            suggestions = json.loads(response.choices[0].message.content)
            return suggestions
            
        except Exception as e:
            logger.error(f"AI code suggestion generation failed: {e}")
            return self._generate_fallback_code_suggestions(story_data, keywords)
    
    def _generate_fallback_code_suggestions(self, story_data: Dict[str, Any], 
                                          keywords: List[str]) -> Dict[str, Any]:
        """Generate fallback code suggestions using rule-based approach."""
        try:
            suggestions = {
                "frontend_components": [],
                "backend_endpoints": [],
                "database_operations": [],
                "business_logic": [],
                "integration_points": [],
                "security_considerations": [],
                "performance_considerations": [],
                "suggested_architecture": "MVC"
            }
            
            # Analyze keywords for patterns
            title_lower = story_data.get("title", "").lower()
            description_lower = story_data.get("description", "").lower()
            
            # Frontend components
            if any(word in title_lower for word in ["form", "page", "view", "display"]):
                suggestions["frontend_components"].extend(["FormComponent", "PageComponent"])
            if any(word in title_lower for word in ["list", "table", "grid"]):
                suggestions["frontend_components"].append("ListComponent")
            if any(word in title_lower for word in ["modal", "dialog", "popup"]):
                suggestions["frontend_components"].append("ModalComponent")
            
            # Backend endpoints
            if any(word in description_lower for word in ["create", "add", "new"]):
                suggestions["backend_endpoints"].append({"method": "POST", "path": "/api/resource", "description": "Create new resource"})
            if any(word in description_lower for word in ["get", "fetch", "retrieve", "show"]):
                suggestions["backend_endpoints"].append({"method": "GET", "path": "/api/resource", "description": "Get resource"})
            if any(word in description_lower for word in ["update", "edit", "modify"]):
                suggestions["backend_endpoints"].append({"method": "PUT", "path": "/api/resource/:id", "description": "Update resource"})
            if any(word in description_lower for word in ["delete", "remove"]):
                suggestions["backend_endpoints"].append({"method": "DELETE", "path": "/api/resource/:id", "description": "Delete resource"})
            
            # Database operations
            if suggestions["backend_endpoints"]:
                suggestions["database_operations"] = ["CREATE", "READ", "UPDATE", "DELETE"]
            
            # Business logic
            if any(word in description_lower for word in ["validate", "check", "verify"]):
                suggestions["business_logic"].append("ValidationService")
            if any(word in description_lower for word in ["process", "calculate", "compute"]):
                suggestions["business_logic"].append("ProcessingService")
            
            # Security considerations
            if any(word in description_lower for word in ["user", "login", "auth"]):
                suggestions["security_considerations"].extend(["Authentication", "Authorization"])
            
            return suggestions
            
        except
                suggestions["security_considerations"].extend(["Authentication", "Authorization"])
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate fallback code suggestions: {e}")
            return {
                "frontend_components": [],
                "backend_endpoints": [],
                "database_operations": [],
                "business_logic": [],
                "integration_points": [],
                "security_considerations": [],
                "performance_considerations": [],
                "suggested_architecture": "MVC"
            }
    
    async def _analyze_technical_requirements(self, story_data: Dict[str, Any], 
                                            keywords: List[str]) -> Dict[str, Any]:
        """Analyze technical requirements from story data."""
        try:
            requirements = {
                "data_models": [],
                "external_integrations": [],
                "performance_requirements": [],
                "scalability_requirements": [],
                "security_requirements": [],
                "compliance_requirements": []
            }
            
            description = story_data.get("description", "").lower()
            acceptance_criteria = " ".join(story_data.get("acceptance_criteria", [])).lower()
            combined_text = f"{description} {acceptance_criteria}"
            
            # Data models
            if any(word in combined_text for word in ["user", "customer", "account"]):
                requirements["data_models"].append("User")
            if any(word in combined_text for word in ["order", "purchase", "transaction"]):
                requirements["data_models"].append("Order")
            if any(word in combined_text for word in ["product", "item", "catalog"]):
                requirements["data_models"].append("Product")
            if any(word in combined_text for word in ["payment", "billing", "invoice"]):
                requirements["data_models"].append("Payment")
            
            # External integrations
            if any(word in combined_text for word in ["email", "notification", "alert"]):
                requirements["external_integrations"].append("EmailService")
            if any(word in combined_text for word in ["payment", "stripe", "paypal"]):
                requirements["external_integrations"].append("PaymentGateway")
            if any(word in combined_text for word in ["sms", "text message"]):
                requirements["external_integrations"].append("SMSService")
            if any(word in combined_text for word in ["api", "third party", "external"]):
                requirements["external_integrations"].append("ExternalAPI")
            
            # Performance requirements
            if any(word in combined_text for word in ["fast", "quick", "performance", "speed"]):
                requirements["performance_requirements"].append("Response time optimization")
            if any(word in combined_text for word in ["large", "bulk", "many", "thousands"]):
                requirements["performance_requirements"].append("Bulk operation handling")
            
            # Security requirements
            if any(word in combined_text for word in ["secure", "private", "confidential"]):
                requirements["security_requirements"].append("Data encryption")
            if any(word in combined_text for word in ["permission", "role", "access"]):
                requirements["security_requirements"].append("Role-based access control")
            
            return requirements
            
        except Exception as e:
            logger.error(f"Failed to analyze technical requirements: {e}")
            return {}
    
    async def _generate_api_suggestions(self, story_data: Dict[str, Any], 
                                      tech_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate API endpoint suggestions."""
        try:
            api_suggestions = []
            
            # Base resource name from story
            title = story_data.get("title", "").lower()
            resource_name = "resource"  # Default
            
            # Try to extract resource name
            if "user" in title:
                resource_name = "users"
            elif "order" in title:
                resource_name = "orders"
            elif "product" in title:
                resource_name = "products"
            elif "payment" in title:
                resource_name = "payments"
            
            # Generate CRUD endpoints
            api_suggestions.extend([
                {
                    "method": "GET",
                    "path": f"/api/{resource_name}",
                    "description": f"List all {resource_name}",
                    "parameters": ["page", "limit", "filter"],
                    "response": f"Array of {resource_name}"
                },
                {
                    "method": "GET",
                    "path": f"/api/{resource_name}/{{id}}",
                    "description": f"Get specific {resource_name[:-1]}",
                    "parameters": ["id"],
                    "response": f"Single {resource_name[:-1]} object"
                },
                {
                    "method": "POST",
                    "path": f"/api/{resource_name}",
                    "description": f"Create new {resource_name[:-1]}",
                    "parameters": ["body"],
                    "response": f"Created {resource_name[:-1]} object"
                },
                {
                    "method": "PUT",
                    "path": f"/api/{resource_name}/{{id}}",
                    "description": f"Update {resource_name[:-1]}",
                    "parameters": ["id", "body"],
                    "response": f"Updated {resource_name[:-1]} object"
                },
                {
                    "method": "DELETE",
                    "path": f"/api/{resource_name}/{{id}}",
                    "description": f"Delete {resource_name[:-1]}",
                    "parameters": ["id"],
                    "response": "Success confirmation"
                }
            ])
            
            return api_suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate API suggestions: {e}")
            return []
    
    async def _generate_database_suggestions(self, story_data: Dict[str, Any], 
                                           tech_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate database schema suggestions."""
        try:
            db_suggestions = {
                "tables": [],
                "relationships": [],
                "indexes": [],
                "constraints": []
            }
            
            data_models = tech_requirements.get("data_models", [])
            
            for model in data_models:
                table_suggestion = {
                    "name": model.lower() + "s",
                    "columns": self._generate_table_columns(model),
                    "primary_key": "id",
                    "timestamps": True
                }
                db_suggestions["tables"].append(table_suggestion)
            
            # Generate relationships
            if "User" in data_models and "Order" in data_models:
                db_suggestions["relationships"].append({
                    "type": "one_to_many",
                    "from": "users",
                    "to": "orders",
                    "foreign_key": "user_id"
                })
            
            # Generate indexes
            for table in db_suggestions["tables"]:
                if "email" in [col["name"] for col in table["columns"]]:
                    db_suggestions["indexes"].append({
                        "table": table["name"],
                        "columns": ["email"],
                        "unique": True
                    })
            
            return db_suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate database suggestions: {e}")
            return {}
    
    def _generate_table_columns(self, model: str) -> List[Dict[str, Any]]:
        """Generate table columns for a data model."""
        base_columns = [
            {"name": "id", "type": "UUID", "nullable": False, "primary_key": True},
            {"name": "created_at", "type": "TIMESTAMP", "nullable": False},
            {"name": "updated_at", "type": "TIMESTAMP", "nullable": False}
        ]
        
        model_columns = {
            "User": [
                {"name": "email", "type": "VARCHAR(255)", "nullable": False, "unique": True},
                {"name": "first_name", "type": "VARCHAR(100)", "nullable": False},
                {"name": "last_name", "type": "VARCHAR(100)", "nullable": False},
                {"name": "password_hash", "type": "VARCHAR(255)", "nullable": False},
                {"name": "is_active", "type": "BOOLEAN", "nullable": False, "default": True}
            ],
            "Order": [
                {"name": "user_id", "type": "UUID", "nullable": False, "foreign_key": "users.id"},
                {"name": "total_amount", "type": "DECIMAL(10,2)", "nullable": False},
                {"name": "status", "type": "VARCHAR(50)", "nullable": False},
                {"name": "order_date", "type": "TIMESTAMP", "nullable": False}
            ],
            "Product": [
                {"name": "name", "type": "VARCHAR(255)", "nullable": False},
                {"name": "description", "type": "TEXT", "nullable": True},
                {"name": "price", "type": "DECIMAL(10,2)", "nullable": False},
                {"name": "sku", "type": "VARCHAR(100)", "nullable": False, "unique": True},
                {"name": "is_active", "type": "BOOLEAN", "nullable": False, "default": True}
            ],
            "Payment": [
                {"name": "order_id", "type": "UUID", "nullable": False, "foreign_key": "orders.id"},
                {"name": "amount", "type": "DECIMAL(10,2)", "nullable": False},
                {"name": "payment_method", "type": "VARCHAR(50)", "nullable": False},
                {"name": "status", "type": "VARCHAR(50)", "nullable": False},
                {"name": "transaction_id", "type": "VARCHAR(255)", "nullable": True}
            ]
        }
        
        specific_columns = model_columns.get(model, [])
        return base_columns + specific_columns
    
    async def _generate_test_suggestions(self, story_data: Dict[str, Any], 
                                       acceptance_criteria: List[str]) -> Dict[str, Any]:
        """Generate test case suggestions."""
        try:
            test_suggestions = {
                "unit_tests": [],
                "integration_tests": [],
                "e2e_tests": [],
                "test_scenarios": []
            }
            
            # Generate unit tests
            test_suggestions["unit_tests"].extend([
                "Test input validation",
                "Test business logic functions",
                "Test error handling",
                "Test edge cases"
            ])
            
            # Generate integration tests
            test_suggestions["integration_tests"].extend([
                "Test API endpoints",
                "Test database operations",
                "Test external service integrations"
            ])
            
            # Generate E2E tests from acceptance criteria
            for i, criteria in enumerate(acceptance_criteria):
                test_suggestions["e2e_tests"].append(f"Test: {criteria}")
                test_suggestions["test_scenarios"].append({
                    "scenario": f"Acceptance Criteria {i+1}",
                    "given": "User is on the application",
                    "when": f"User performs action described in: {criteria}",
                    "then": "Expected outcome should occur"
                })
            
            return test_suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate test suggestions: {e}")
            return {}
    
    async def _analyze_implementation_complexity(self, code_suggestions: Dict[str, Any], 
                                               tech_requirements: Dict[str, Any], 
                                               story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze implementation complexity."""
        try:
            complexity_score = 0
            complexity_factors = []
            
            # Frontend complexity
            frontend_components = code_suggestions.get("frontend_components", [])
            complexity_score += len(frontend_components) * 2
            if len(frontend_components) > 3:
                complexity_factors.append("Multiple frontend components required")
            
            # Backend complexity
            backend_endpoints = code_suggestions.get("backend_endpoints", [])
            complexity_score += len(backend_endpoints) * 3
            if len(backend_endpoints) > 5:
                complexity_factors.append("Multiple API endpoints required")
            
            # Database complexity
            data_models = tech_requirements.get("data_models", [])
            complexity_score += len(data_models) * 4
            if len(data_models) > 2:
                complexity_factors.append("Multiple data models required")
            
            # Integration complexity
            integrations = tech_requirements.get("external_integrations", [])
            complexity_score += len(integrations) * 5
            if integrations:
                complexity_factors.append("External integrations required")
            
            # Security complexity
            security_reqs = tech_requirements.get("security_requirements", [])
            complexity_score += len(security_reqs) * 3
            if security_reqs:
                complexity_factors.append("Security requirements add complexity")
            
            # Determine complexity level
            if complexity_score <= 10:
                complexity_level = "simple"
            elif complexity_score <= 25:
                complexity_level = "medium"
            elif complexity_score <= 40:
                complexity_level = "complex"
            else:
                complexity_level = "very_complex"
            
            # Estimate files to modify
            files_to_modify = []
            if frontend_components:
                files_to_modify.extend([f"components/{comp}.tsx" for comp in frontend_components])
            if backend_endpoints:
                files_to_modify.extend(["routes/api.py", "controllers/resource_controller.py"])
            if data_models:
                files_to_modify.extend([f"models/{model.lower()}.py" for model in data_models])
            
            return {
                "complexity_level": complexity_level,
                "complexity_score": complexity_score,
                "complexity_factors": complexity_factors,
                "files_to_modify": files_to_modify,
                "estimated_development_days": self._estimate_development_days(complexity_level),
                "recommended_team_size": self._recommend_team_size(complexity_level)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze implementation complexity: {e}")
            return {"complexity_level": "medium", "complexity_score": 20}
    
    def _estimate_development_days(self, complexity_level: str) -> int:
        """Estimate development days based on complexity."""
        estimates = {
            "simple": 2,
            "medium": 5,
            "complex": 10,
            "very_complex": 20
        }
        return estimates.get(complexity_level, 5)
    
    def _recommend_team_size(self, complexity_level: str) -> int:
        """Recommend team size based on complexity."""
        recommendations = {
            "simple": 1,
            "medium": 2,
            "complex": 3,
            "very_complex": 4
        }
        return recommendations.get(complexity_level, 2)
    
    async def _generate_implementation_steps(self, code_suggestions: Dict[str, Any], 
                                           tech_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate implementation steps."""
        try:
            steps = []
            
            # Database setup
            if tech_requirements.get("data_models"):
                steps.append({
                    "step": 1,
                    "phase": "Database Setup",
                    "description": "Create database models and migrations",
                    "estimated_hours": 4,
                    "dependencies": []
                })
            
            # Backend API
            if code_suggestions.get("backend_endpoints"):
                steps.append({
                    "step": 2,
                    "phase": "Backend Development",
                    "description": "Implement API endpoints and business logic",
                    "estimated_hours": 8,
                    "dependencies": [1] if tech_requirements.get("data_models") else []
                })
            
            # Frontend components
            if code_suggestions.get("frontend_components"):
                steps.append({
                    "step": 3,
                    "phase": "Frontend Development",
                    "description": "Create UI components and integrate with API",
                    "estimated_hours": 12,
                    "dependencies": [2] if code_suggestions.get("backend_endpoints") else []
                })
            
            # Testing
            steps.append({
                "step": 4,
                "phase": "Testing",
                "description": "Write and execute unit, integration, and E2E tests",
                "estimated_hours": 6,
                "dependencies": [3] if code_suggestions.get("frontend_components") else [2]
            })
            
            # Integration
            if tech_requirements.get("external_integrations"):
                steps.append({
                    "step": 5,
                    "phase": "External Integrations",
                    "description": "Implement external service integrations",
                    "estimated_hours": 8,
                    "dependencies": [2]
                })
            
            return steps
            
        except Exception as e:
            logger.error(f"Failed to generate implementation steps: {e}")
            return []
    
    def _calculate_translation_confidence(self, code_suggestions: Dict[str, Any], 
                                        tech_requirements: Dict[str, Any]) -> float:
        """Calculate confidence score for story translation."""
        try:
            confidence = 0.5  # Base confidence
            
            # More suggestions = higher confidence
            total_suggestions = (
                len(code_suggestions.get("frontend_components", [])) +
                len(code_suggestions.get("backend_endpoints", [])) +
                len(code_suggestions.get("business_logic", []))
            )
            
            if total_suggestions >= 5:
                confidence += 0.2
            elif total_suggestions >= 3:
                confidence += 0.1
            
            # Clear technical requirements = higher confidence
            if tech_requirements.get("data_models"):
                confidence += 0.1
            if tech_requirements.get("external_integrations"):
                confidence += 0.1
            
            # Well-defined API = higher confidence
            if len(code_suggestions.get("backend_endpoints", [])) >= 3:
                confidence += 0.1
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate translation confidence: {e}")
            return 0.5
    
    async def estimate_story_effort(self, story_id: str, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate story effort using AI and historical data."""
        try:
            data_lake = await get_data_lake()
            
            # Get historical story data for comparison
            historical_stories = await self._get_historical_stories()
            
            # Analyze story characteristics
            story_analysis = await self._analyze_story_characteristics(story_data)
            
            # Find similar stories
            similar_stories = await self._find_similar_stories(story_data, historical_stories)
            
            # Generate effort estimate using multiple methods
            estimates = {
                "historical_comparison": await self._estimate_from_historical_data(similar_stories),
                "complexity_analysis": await self._estimate_from_complexity(story_analysis),
                "ai_estimation": await self._estimate_with_ai(story_data, story_analysis)
            }
            
            # Combine estimates with weights
            final_estimate = (
                estimates["historical_comparison"] * 0.4 +
                estimates["complexity_analysis"] * 0.3 +
                estimates["ai_estimation"] * 0.3
            )
            
            # Round to nearest Fibonacci number (common in story pointing)
            fibonacci_points = [1, 2, 3, 5, 8, 13, 21]
            estimated_points = min(fibonacci_points, key=lambda x: abs(x - final_estimate))
            
            # Calculate confidence based on data quality
            confidence = self._calculate_estimation_confidence(similar_stories, story_analysis, estimates)
            
            # Generate estimation breakdown
            estimation_breakdown = {
                "factors_analyzed": story_analysis.get("factors", []),
                "similar_stories_count": len(similar_stories),
                "estimation_methods": estimates,
                "confidence_factors": self._get_confidence_factors(similar_stories, story_analysis)
            }
            
            effort_estimate = {
                "estimate_id": f"EST_{str(uuid.uuid4())[:8].upper()}",
                "story_id": story_id,
                "estimated_points": estimated_points,
                "confidence": confidence,
                "estimation_range": {
                    "min": max(1, estimated_points - 2),
                    "max": min(21, estimated_points + 3)
                },
                "estimation_breakdown": estimation_breakdown,
                "similar_stories": [s.get("story_id") for s in similar_stories[:5]],
                "complexity_factors": story_analysis.get("complexity_factors", []),
                "risk_factors": story_analysis.get("risk_factors", []),
                "created_at": datetime.utcnow()
            }
            
            # Store estimate
            await self.db[self.effort_estimates_collection].insert_one(effort_estimate)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="product",
                event_type="story_effort_estimated",
                entity_type="story",
                entity_id=story_id,
                data={
                    "estimate_id": effort_estimate["estimate_id"],
                    "estimated_points": estimated_points,
                    "confidence": confidence,
                    "similar_stories_used": len(similar_stories)
                }
            )
            
            logger.info(f"Story effort estimated: {story_id} -> {estimated_points} points (confidence: {confidence:.2f})")
            
            return effort_estimate
            
        except Exception as e:
            logger.error(f"Failed to estimate story effort: {e}")
            return {"estimated_points": 5, "confidence": 0.5}
    
    async def _get_historical_stories(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Get historical stories with actual effort data."""
        try:
            stories = await self.db["stories"].find({
                "status": "completed",
                "actual_points": {"$exists": True, "$gt": 0}
            }).limit(limit).to_list(None)
            
            return stories
            
        except Exception as e:
            logger.error(f"Failed to get historical stories: {e}")
            return []
    
    async def _analyze_story_characteristics(self, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze story characteristics for estimation."""
        try:
            nlp = await get_nlp_processor()
            
            title = story_data.get("title", "")
            description = story_data.get("description", "")
            acceptance_criteria = story_data.get("acceptance_criteria", [])
            
            # Text analysis
            combined_text = f"{title} {description} {' '.join(acceptance_criteria)}"
            keywords = await nlp.extract_keywords(combined_text, 10)
            
            # Complexity factors
            complexity_factors = []
            complexity_score = 0
            
            # Text length indicates complexity
            text_length = len(combined_text)
            if text_length > 500:
                complexity_factors.append("Detailed requirements (long description)")
                complexity_score += 2
            elif text_length > 200:
                complexity_score += 1
            
            # Number of acceptance criteria
            criteria_count = len(acceptance_criteria)
            if criteria_count > 5:
                complexity_factors.append("Many acceptance criteria")
                complexity_score += 2
            elif criteria_count > 2:
                complexity_score += 1
            
            # Technical complexity keywords
            tech_keywords = ["api", "database", "integration", "security", "performance", "migration"]
            tech_mentions = sum(1 for keyword in tech_keywords if keyword in combined_text.lower())
            if tech_mentions > 2:
                complexity_factors.append("High technical complexity")
                complexity_score += 3
            elif tech_mentions > 0:
                complexity_score += 1
            
            # UI complexity keywords
            ui_keywords = ["form", "table", "chart", "dashboard", "responsive", "mobile"]
            ui_mentions = sum(1 for keyword in ui_keywords if keyword in combined_text.lower())
            if ui_mentions > 2:
                complexity_factors.append("Complex UI requirements")
                complexity_score += 2
            elif ui_mentions > 0:
                complexity_score += 1
            
            # Risk factors
            risk_factors = []
            if "new" in combined_text.lower() and "technology" in combined_text.lower():
                risk_factors.append("New technology involved")
            if "external" in combined_text.lower() or "third party" in combined_text.lower():
                risk_factors.append("External dependencies")
            if "migration" in combined_text.lower():
                risk_factors.append("Data migration required")
            
            return {
                "complexity_score": complexity_score,
                "complexity_factors": complexity_factors,
                "risk_factors": risk_factors,
                "keywords": keywords,
                "text_length": text_length,
                "criteria_count": criteria_count,
                "tech_complexity": tech_mentions,
                "ui_complexity": ui_mentions
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze story characteristics: {e}")
            return {"complexity_score": 3}
    
    async def _find_similar_stories(self, story_data: Dict[str, Any], 
                                  historical_stories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find similar stories from historical data."""
        try:
            nlp = await get_nlp_processor()
            
            current_text = f"{story_data.get('title', '')} {story_data.get('description', '')}"
            similar_stories = []
            
            for historical_story in historical_stories:
                historical_text = f"{historical_story.get('title', '')} {historical_story.get('description', '')}"
                
                # Calculate text similarity
                similarity = await nlp.calculate_similarity(current_text, historical_text)
                
                if similarity > 0.3:  # 30% similarity threshold
                    historical_story["similarity_score"] = similarity
                    similar_stories.append(historical_story)
            
            # Sort by similarity and return top matches
            similar_stories.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            return similar_stories[:10]
            
        except Exception as e:
            logger.error(f"Failed to find similar stories: {e}")
            return []
    
    async def _estimate_from_historical_data(self, similar_stories: List[Dict[str, Any]]) -> float:
        """Estimate effort based on similar historical stories."""
        try:
            if not similar_stories:
                return 5.0  # Default estimate
            
            # Weight estimates by similarity
            weighted_sum = 0
            total_weight = 0
            
            for story in similar_stories:
                actual_points = story.get("actual_points", 0)
                similarity = story.get("similarity_score", 0)
                
                if actual_points > 0:
                    weighted_sum += actual_points * similarity
                    total_weight += similarity
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                # Fallback to simple average
                points = [s.get("actual_points", 0) for s in similar_stories if s.get("actual_points", 0) > 0]
                return sum(points) / len(points) if points else 5.0
            
        except Exception as e:
            logger.error(f"Failed to estimate from historical data: {e}")
            return 5.0
    
    async def _estimate_from_complexity(self, story_analysis: Dict[str, Any]) -> float:
        """Estimate effort based on complexity analysis."""
        try:
            base_estimate = 3.0  # Base story points
            complexity_score = story_analysis.get("complexity_score", 0)
            
            # Add points based on complexity
            complexity_points = complexity_score * 0.5
            
            # Add points for risk factors
            risk_factors = story_analysis.get("risk_factors", [])
            risk_points = len(risk_factors) * 1.0
            
            total_estimate = base_estimate + complexity_points + risk_points
            
            # Cap at reasonable maximum
            return min(total_estimate, 21.0)
            
        except Exception as e:
            logger.error(f"Failed to estimate from complexity: {e}")
            return 5.0
    
    async def _estimate_with_ai(self, story_data: Dict[str, Any], 
                              story_analysis: Dict[str, Any]) -> float:
        """Estimate effort using AI."""
        try:
            if not openai.api_key:
                return 5.0  # Fallback estimate
            
            story_text = f"""
            Title: {story_data.get('title', '')}
            Description: {story_data.get('description', '')}
            Acceptance Criteria: {'; '.join(story_data.get('acceptance_criteria', []))}
            Complexity Score: {story_analysis.get('complexity_score', 0)}
            Risk Factors: {', '.join(story_analysis.get('risk_factors', []))}
            """
            
            prompt = f"""
            Estimate the effort for this user story in story points (1, 2, 3, 5, 8, 13, 21).
            
            {story_text}
            
            Consider:
            - Implementation complexity
            - Testing requirements
            - Integration needs
            - Risk factors
            
            Respond with just a number from the Fibonacci sequence: 1, 2, 3, 5, 8, 13, or 21
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            # Extract number from response
            response_text = response.choices[0].message.content.strip()
            try:
                estimate = float(re.findall(r'\d+', response_text)[0])
                # Ensure it's a valid Fibonacci number
                fibonacci_points = [1, 2, 3, 5, 8, 13, 21]
                return min(fibonacci_points, key=lambda x: abs(x - estimate))
            except:
                return 5.0
            
        except Exception as e:
            logger.error(f"AI estimation failed: {e}")
            return 5.0
    
    def _calculate_estimation_confidence(self, similar_stories: List[Dict[str, Any]], 
                                       story_analysis: Dict[str, Any], 
                                       estimates: Dict[str, float]) -> float:
        """Calculate confidence in the estimation."""
        try:
            confidence = 0.5  # Base confidence
            
            # More similar stories = higher confidence
            if len(similar_stories) >= 5:
                confidence += 0.2
            elif len(similar_stories) >= 2:
                confidence += 0.1
            
            # High similarity = higher confidence
            if similar_stories:
                avg_similarity = sum(s.get("similarity_score", 0) for s in similar_stories) / len(similar_stories)
                confidence += avg_similarity * 0.2
            
            # Consistent estimates = higher confidence
            estimate_values = list(estimates.values())
            if estimate_values:
                estimate_variance = max(estimate_values) - min(estimate_values)
                if estimate_variance <= 2:
                    confidence += 0.1
                elif estimate_variance <= 5:
                    confidence += 0.05
            
            # Clear requirements = higher confidence
            if story_analysis.get("criteria_count", 0) >= 3:
                confidence += 0.1
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate estimation confidence: {e}")
            return 0.5
    
    def _get_confidence_factors(self, similar_stories: List[Dict[str, Any]], 
                              story_analysis: Dict[str, Any]) -> List[str]:
        """Get factors that affect estimation confidence."""
        factors = []
        
        if len(similar_stories) >= 5:
            factors.append("Multiple similar stories found")
        elif len(similar_stories) >= 2:
            factors.append("Some similar stories found")
        else:
            factors.append("Limited historical data")
        
        if story_analysis.get("criteria_count", 0) >= 3:
            factors.append("Well-defined acceptance criteria")
        else:
            factors.append("Limited acceptance criteria")
        
        if story_analysis.get("risk_factors"):
            factors.append("Risk factors identified")
        
        return factors
