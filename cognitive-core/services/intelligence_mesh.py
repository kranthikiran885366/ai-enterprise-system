"""Intelligence Mesh - Unified Intelligence coordination across all agents."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger
import httpx
import uuid

from shared_libs.database import get_database
from shared_libs.data_lake import get_data_lake
from shared_libs.messaging import get_message_broker


class IntelligenceMesh:
    """Unified Intelligence Mesh for cross-agent coordination and learning."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.agents_collection = "mesh_agents"
        self.coordination_collection = "agent_coordination"
        self.learning_collection = "cross_agent_learning"
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Agent registry
        self.registered_agents = {}
        self.agent_capabilities = {}
        self.agent_performance_metrics = {}
    
    async def initialize(self):
        """Initialize the Intelligence Mesh."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.agents_collection].create_index("agent_id", unique=True)
        await self.db[self.agents_collection].create_index("status")
        await self.db[self.agents_collection].create_index("last_heartbeat")
        
        await self.db[self.coordination_collection].create_index("coordination_id", unique=True)
        await self.db[self.coordination_collection].create_index("created_at")
        await self.db[self.coordination_collection].create_index("status")
        
        await self.db[self.learning_collection].create_index("learning_id", unique=True)
        await self.db[self.learning_collection].create_index("source_agent")
        await self.db[self.learning_collection].create_index("created_at")
        
        # Discover and register existing agents
        await self._discover_agents()
        
        logger.info("Intelligence Mesh initialized")
    
    async def start_monitoring(self):
        """Start intelligence mesh monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Intelligence Mesh monitoring started")
    
    async def stop_monitoring(self):
        """Stop intelligence mesh monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Intelligence Mesh monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for intelligence mesh."""
        while self.is_monitoring:
            try:
                # Update agent status
                await self._update_agent_status()
                
                # Coordinate cross-agent tasks
                await self._coordinate_cross_agent_tasks()
                
                # Share learning insights
                await self._share_learning_insights()
                
                # Optimize agent interactions
                await self._optimize_agent_interactions()
                
                # Sleep for 30 seconds before next iteration
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in intelligence mesh monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _discover_agents(self):
        """Discover and register available agents."""
        try:
            # Known agent endpoints
            agent_endpoints = {
                "hr": "http://hr-agent:8000",
                "finance": "http://finance-agent:8000",
                "sales": "http://sales-agent:8000",
                "product": "http://product-agent:8000",
                "qa": "http://qa-agent:8000",
                "support": "http://support-agent:8000"
            }
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                for agent_name, endpoint in agent_endpoints.items():
                    try:
                        # Check agent health
                        response = await client.get(f"{endpoint}/health")
                        if response.status_code == 200:
                            # Get agent capabilities
                            capabilities_response = await client.get(f"{endpoint}/capabilities")
                            capabilities = capabilities_response.json() if capabilities_response.status_code == 200 else {}
                            
                            # Register agent
                            await self._register_agent(agent_name, endpoint, capabilities)
                            
                    except Exception as e:
                        logger.warning(f"Failed to discover agent {agent_name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to discover agents: {e}")
    
    async def _register_agent(self, agent_name: str, endpoint: str, capabilities: Dict[str, Any]):
        """Register an agent in the intelligence mesh."""
        try:
            agent_data = {
                "agent_id": agent_name,
                "endpoint": endpoint,
                "capabilities": capabilities,
                "status": "active",
                "last_heartbeat": datetime.utcnow(),
                "registered_at": datetime.utcnow(),
                "performance_metrics": {
                    "response_time": 0,
                    "success_rate": 1.0,
                    "load_factor": 0.0
                }
            }
            
            # Upsert agent registration
            await self.db[self.agents_collection].replace_one(
                {"agent_id": agent_name},
                agent_data,
                upsert=True
            )
            
            # Update local registry
            self.registered_agents[agent_name] = agent_data
            self.agent_capabilities[agent_name] = capabilities
            
            logger.info(f"Agent registered in intelligence mesh: {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_name}: {e}")
    
    async def _update_agent_status(self):
        """Update status of all registered agents."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                for agent_name, agent_data in self.registered_agents.items():
                    try:
                        endpoint = agent_data["endpoint"]
                        start_time = datetime.utcnow()
                        
                        # Health check
                        response = await client.get(f"{endpoint}/health")
                        response_time = (datetime.utcnow() - start_time).total_seconds()
                        
                        if response.status_code == 200:
                            # Update performance metrics
                            await self._update_agent_performance(agent_name, response_time, True)
                            
                            # Update heartbeat
                            await self.db[self.agents_collection].update_one(
                                {"agent_id": agent_name},
                                {
                                    "$set": {
                                        "status": "active",
                                        "last_heartbeat": datetime.utcnow()
                                    }
                                }
                            )
                        else:
                            await self._update_agent_performance(agent_name, response_time, False)
                            
                    except Exception as e:
                        logger.warning(f"Failed to update status for agent {agent_name}: {e}")
                        await self._update_agent_performance(agent_name, 10.0, False)
            
        except Exception as e:
            logger.error(f"Failed to update agent status: {e}")
    
    async def _update_agent_performance(self, agent_name: str, response_time: float, success: bool):
        """Update agent performance metrics."""
        try:
            # Get current metrics
            current_metrics = self.agent_performance_metrics.get(agent_name, {
                "response_time": 0,
                "success_rate": 1.0,
                "total_requests": 0,
                "successful_requests": 0
            })
            
            # Update metrics
            current_metrics["total_requests"] += 1
            if success:
                current_metrics["successful_requests"] += 1
            
            # Calculate running averages
            current_metrics["success_rate"] = current_metrics["successful_requests"] / current_metrics["total_requests"]
            current_metrics["response_time"] = (current_metrics["response_time"] + response_time) / 2
            
            # Store updated metrics
            self.agent_performance_metrics[agent_name] = current_metrics
            
            # Update in database
            await self.db[self.agents_collection].update_one(
                {"agent_id": agent_name},
                {"$set": {"performance_metrics": current_metrics}}
            )
            
        except Exception as e:
            logger.error(f"Failed to update performance for agent {agent_name}: {e}")
    
    async def _coordinate_cross_agent_tasks(self):
        """Coordinate tasks that require multiple agents."""
        try:
            # Get pending coordination requests
            pending_coordinations = await self.db[self.coordination_collection].find({
                "status": "pending"
            }).to_list(None)
            
            for coordination in pending_coordinations:
                await self._execute_coordination(coordination)
            
        except Exception as e:
            logger.error(f"Failed to coordinate cross-agent tasks: {e}")
    
    async def _execute_coordination(self, coordination: Dict[str, Any]):
        """Execute a cross-agent coordination task."""
        try:
            coordination_id = coordination.get("coordination_id")
            task_type = coordination.get("task_type")
            required_agents = coordination.get("required_agents", [])
            task_data = coordination.get("task_data", {})
            
            logger.info(f"Executing coordination {coordination_id}: {task_type}")
            
            # Check if all required agents are available
            available_agents = []
            for agent_name in required_agents:
                if agent_name in self.registered_agents:
                    agent_status = self.registered_agents[agent_name].get("status")
                    if agent_status == "active":
                        available_agents.append(agent_name)
            
            if len(available_agents) != len(required_agents):
                logger.warning(f"Not all required agents available for coordination {coordination_id}")
                return
            
            # Execute coordination based on task type
            if task_type == "employee_onboarding":
                await self._coordinate_employee_onboarding(coordination_id, task_data, available_agents)
            elif task_type == "project_planning":
                await self._coordinate_project_planning(coordination_id, task_data, available_agents)
            elif task_type == "incident_response":
                await self._coordinate_incident_response(coordination_id, task_data, available_agents)
            elif task_type == "budget_planning":
                await self._coordinate_budget_planning(coordination_id, task_data, available_agents)
            else:
                logger.warning(f"Unknown coordination task type: {task_type}")
            
        except Exception as e:
            logger.error(f"Failed to execute coordination: {e}")
    
    async def _coordinate_employee_onboarding(self, coordination_id: str, task_data: Dict[str, Any], agents: List[str]):
        """Coordinate employee onboarding across HR, IT, and Finance agents."""
        try:
            employee_data = task_data.get("employee_data", {})
            
            # Step 1: HR creates employee record
            if "hr" in agents:
                hr_result = await self._call_agent_api("hr", "/employees", "POST", employee_data)
                if not hr_result.get("success"):
                    raise Exception("HR agent failed to create employee record")
                
                employee_id = hr_result.get("employee_id")
                task_data["employee_id"] = employee_id
            
            # Step 2: Finance sets up payroll
            if "finance" in agents and task_data.get("employee_id"):
                payroll_data = {
                    "employee_id": task_data["employee_id"],
                    "salary": employee_data.get("salary"),
                    "start_date": employee_data.get("start_date")
                }
                finance_result = await self._call_agent_api("finance", "/payroll", "POST", payroll_data)
                if not finance_result.get("success"):
                    logger.warning("Finance agent failed to set up payroll")
            
            # Step 3: IT provisions accounts (if IT agent exists)
            # This would be implemented when IT agent is added
            
            # Update coordination status
            await self.db[self.coordination_collection].update_one(
                {"coordination_id": coordination_id},
                {
                    "$set": {
                        "status": "completed",
                        "completed_at": datetime.utcnow(),
                        "result": {"employee_id": task_data.get("employee_id")}
                    }
                }
            )
            
            logger.info(f"Employee onboarding coordination completed: {coordination_id}")
            
        except Exception as e:
            logger.error(f"Failed to coordinate employee onboarding: {e}")
            await self.db[self.coordination_collection].update_one(
                {"coordination_id": coordination_id},
                {"$set": {"status": "failed", "error": str(e)}}
            )
    
    async def _coordinate_project_planning(self, coordination_id: str, task_data: Dict[str, Any], agents: List[str]):
        """Coordinate project planning across Product, Finance, and HR agents."""
        try:
            project_data = task_data.get("project_data", {})
            
            # Step 1: Product agent creates project plan
            if "product" in agents:
                product_result = await self._call_agent_api("product", "/projects", "POST", project_data)
                if product_result.get("success"):
                    project_id = product_result.get("project_id")
                    task_data["project_id"] = project_id
            
            # Step 2: Finance estimates budget
            if "finance" in agents and task_data.get("project_id"):
                budget_request = {
                    "project_id": task_data["project_id"],
                    "scope": project_data.get("scope"),
                    "duration": project_data.get("duration")
                }
                finance_result = await self._call_agent_api("finance", "/budget/estimate", "POST", budget_request)
                if finance_result.get("success"):
                    task_data["estimated_budget"] = finance_result.get("estimated_budget")
            
            # Step 3: HR estimates resource requirements
            if "hr" in agents and task_data.get("project_id"):
                resource_request = {
                    "project_id": task_data["project_id"],
                    "required_skills": project_data.get("required_skills", []),
                    "team_size": project_data.get("team_size", 5)
                }
                hr_result = await self._call_agent_api("hr", "/resources/estimate", "POST", resource_request)
                if hr_result.get("success"):
                    task_data["resource_plan"] = hr_result.get("resource_plan")
            
            # Update coordination status
            await self.db[self.coordination_collection].update_one(
                {"coordination_id": coordination_id},
                {
                    "$set": {
                        "status": "completed",
                        "completed_at": datetime.utcnow(),
                        "result": {
                            "project_id": task_data.get("project_id"),
                            "estimated_budget": task_data.get("estimated_budget"),
                            "resource_plan": task_data.get("resource_plan")
                        }
                    }
                }
            )
            
            logger.info(f"Project planning coordination completed: {coordination_id}")
            
        except Exception as e:
            logger.error(f"Failed to coordinate project planning: {e}")
            await self.db[self.coordination_collection].update_one(
                {"coordination_id": coordination_id},
                {"$set": {"status": "failed", "error": str(e)}}
            )
    
    async def _coordinate_incident_response(self, coordination_id: str, task_data: Dict[str, Any], agents: List[str]):
        """Coordinate incident response across multiple agents."""
        try:
            incident_data = task_data.get("incident_data", {})
            incident_type = incident_data.get("type", "general")
            
            # Determine response strategy based on incident type
            if incident_type == "security":
                # Security incident - involve IT and Legal
                response_agents = ["support", "legal"]
            elif incident_type == "financial":
                # Financial incident - involve Finance and Legal
                response_agents = ["finance", "legal"]
            elif incident_type == "customer":
                # Customer incident - involve Support and Sales
                response_agents = ["support", "sales"]
            else:
                # General incident - involve Support
                response_agents = ["support"]
            
            # Execute response with available agents
            response_results = {}
            for agent in response_agents:
                if agent in agents:
                    try:
                        result = await self._call_agent_api(
                            agent, 
                            "/incidents/respond", 
                            "POST", 
                            incident_data
                        )
                        response_results[agent] = result
                    except Exception as e:
                        logger.error(f"Agent {agent} failed to respond to incident: {e}")
                        response_results[agent] = {"success": False, "error": str(e)}
            
            # Update coordination status
            await self.db[self.coordination_collection].update_one(
                {"coordination_id": coordination_id},
                {
                    "$set": {
                        "status": "completed",
                        "completed_at": datetime.utcnow(),
                        "result": response_results
                    }
                }
            )
            
            logger.info(f"Incident response coordination completed: {coordination_id}")
            
        except Exception as e:
            logger.error(f"Failed to coordinate incident response: {e}")
            await self.db[self.coordination_collection].update_one(
                {"coordination_id": coordination_id},
                {"$set": {"status": "failed", "error": str(e)}}
            )
    
    async def _coordinate_budget_planning(self, coordination_id: str, task_data: Dict[str, Any], agents: List[str]):
        """Coordinate budget planning across Finance and HR agents."""
        try:
            budget_data = task_data.get("budget_data", {})
            
            # Step 1: Finance creates budget framework
            if "finance" in agents:
                finance_result = await self._call_agent_api("finance", "/budget/create", "POST", budget_data)
                if finance_result.get("success"):
                    budget_id = finance_result.get("budget_id")
                    task_data["budget_id"] = budget_id
            
            # Step 2: HR provides personnel cost estimates
            if "hr" in agents and task_data.get("budget_id"):
                hr_request = {
                    "budget_id": task_data["budget_id"],
                    "department": budget_data.get("department"),
                    "period": budget_data.get("period")
                }
                hr_result = await self._call_agent_api("hr", "/budget/personnel-costs", "POST", hr_request)
                if hr_result.get("success"):
                    task_data["personnel_costs"] = hr_result.get("personnel_costs")
            
            # Step 3: Finance finalizes budget
            if "finance" in agents and task_data.get("budget_id"):
                finalize_request = {
                    "budget_id": task_data["budget_id"],
                    "personnel_costs": task_data.get("personnel_costs")
                }
                final_result = await self._call_agent_api("finance", "/budget/finalize", "POST", finalize_request)
                task_data["final_budget"] = final_result.get("final_budget")
            
            # Update coordination status
            await self.db[self.coordination_collection].update_one(
                {"coordination_id": coordination_id},
                {
                    "$set": {
                        "status": "completed",
                        "completed_at": datetime.utcnow(),
                        "result": {
                            "budget_id": task_data.get("budget_id"),
                            "final_budget": task_data.get("final_budget")
                        }
                    }
                }
            )
            
            logger.info(f"Budget planning coordination completed: {coordination_id}")
            
        except Exception as e:
            logger.error(f"Failed to coordinate budget planning: {e}")
            await self.db[self.coordination_collection].update_one(
                {"coordination_id": coordination_id},
                {"$set": {"status": "failed", "error": str(e)}}
            )
    
    async def _call_agent_api(self, agent_name: str, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call an agent's API endpoint."""
        try:
            if agent_name not in self.registered_agents:
                raise Exception(f"Agent {agent_name} not registered")
            
            agent_endpoint = self.registered_agents[agent_name]["endpoint"]
            url = f"{agent_endpoint}{endpoint}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                if method.upper() == "GET":
                    response = await client.get(url, params=data)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data)
                elif method.upper() == "PUT":
                    response = await client.put(url, json=data)
                elif method.upper() == "DELETE":
                    response = await client.delete(url)
                else:
                    raise Exception(f"Unsupported HTTP method: {method}")
                
                if response.status_code == 200:
                    return {"success": True, **response.json()}
                else:
                    return {"success": False, "error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Failed to call agent API {agent_name}{endpoint}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _share_learning_insights(self):
        """Share learning insights between agents."""
        try:
            # Get recent learning insights
            recent_insights = await self.db[self.learning_collection].find({
                "created_at": {"$gte": datetime.utcnow() - timedelta(hours=1)},
                "shared": {"$ne": True}
            }).to_list(None)
            
            for insight in recent_insights:
                await self._distribute_learning_insight(insight)
                
                # Mark as shared
                await self.db[self.learning_collection].update_one(
                    {"learning_id": insight["learning_id"]},
                    {"$set": {"shared": True, "shared_at": datetime.utcnow()}}
                )
            
        except Exception as e:
            logger.error(f"Failed to share learning insights: {e}")
    
    async def _distribute_learning_insight(self, insight: Dict[str, Any]):
        """Distribute a learning insight to relevant agents."""
        try:
            source_agent = insight.get("source_agent")
            insight_type = insight.get("insight_type")
            insight_data = insight.get("insight_data", {})
            
            # Determine target agents based on insight type
            target_agents = []
            if insight_type == "process_optimization":
                target_agents = list(self.registered_agents.keys())  # All agents
            elif insight_type == "customer_behavior":
                target_agents = ["sales", "support", "marketing"]
            elif insight_type == "financial_pattern":
                target_agents = ["finance", "sales", "hr"]
            elif insight_type == "performance_improvement":
                target_agents = ["hr", "product", "qa"]
            
            # Distribute to target agents
            for agent_name in target_agents:
                if agent_name != source_agent and agent_name in self.registered_agents:
                    try:
                        await self._call_agent_api(
                            agent_name,
                            "/learning/insight",
                            "POST",
                            {
                                "source_agent": source_agent,
                                "insight_type": insight_type,
                                "insight_data": insight_data
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to share insight with agent {agent_name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to distribute learning insight: {e}")
    
    async def _optimize_agent_interactions(self):
        """Optimize interactions between agents."""
        try:
            # Analyze agent performance and interaction patterns
            performance_data = {}
            for agent_name, metrics in self.agent_performance_metrics.items():
                performance_data[agent_name] = {
                    "response_time": metrics.get("response_time", 0),
                    "success_rate": metrics.get("success_rate", 1.0),
                    "load_factor": metrics.get("load_factor", 0.0)
                }
            
            # Identify optimization opportunities
            optimizations = []
            
            # High response time optimization
            for agent_name, data in performance_data.items():
                if data["response_time"] > 5.0:  # More than 5 seconds
                    optimizations.append({
                        "type": "response_time",
                        "agent": agent_name,
                        "issue": "High response time",
                        "recommendation": "Consider load balancing or resource scaling"
                    })
            
            # Low success rate optimization
            for agent_name, data in performance_data.items():
                if data["success_rate"] < 0.9:  # Less than 90% success rate
                    optimizations.append({
                        "type": "success_rate",
                        "agent": agent_name,
                        "issue": "Low success rate",
                        "recommendation": "Investigate error patterns and improve error handling"
                    })
            
            # Store optimization recommendations
            if optimizations:
                optimization_record = {
                    "optimization_id": f"OPT_{str(uuid.uuid4())[:8].upper()}",
                    "optimizations": optimizations,
                    "created_at": datetime.utcnow(),
                    "status": "pending"
                }
                
                await self.db["mesh_optimizations"].insert_one(optimization_record)
                logger.info(f"Generated {len(optimizations)} optimization recommendations")
            
        except Exception as e:
            logger.error(f"Failed to optimize agent interactions: {e}")
    
    async def get_mesh_status(self) -> Dict[str, Any]:
        """Get current intelligence mesh status."""
        try:
            # Get agent statistics
            total_agents = len(self.registered_agents)
            active_agents = len([a for a in self.registered_agents.values() if a.get("status") == "active"])
            
            # Calculate coordination efficiency
            recent_coordinations = await self.db[self.coordination_collection].count_documents({
                "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)}
            })
            
            successful_coordinations = await self.db[self.coordination_collection].count_documents({
                "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)},
                "status": "completed"
            })
            
            coordination_efficiency = (successful_coordinations / recent_coordinations) if recent_coordinations > 0 else 1.0
            
            # Calculate average response time
            avg_response_time = 0
            if self.agent_performance_metrics:
                response_times = [metrics.get("response_time", 0) for metrics in self.agent_performance_metrics.values()]
                avg_response_time = sum(response_times) / len(response_times)
            
            return {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "agent_availability": (active_agents / total_agents) if total_agents > 0 else 0,
                "coordination_efficiency": coordination_efficiency,
                "avg_response_time": round(avg_response_time, 3),
                "recent_coordinations": recent_coordinations,
                "successful_coordinations": successful_coordinations,
                "mesh_health": "healthy" if coordination_efficiency > 0.8 else "degraded",
                "last_updated": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get mesh status: {e}")
            return {"mesh_health": "unknown"}
    
    async def create_coordination_request(self, task_type: str, required_agents: List[str], 
                                        task_data: Dict[str, Any]) -> str:
        """Create a new cross-agent coordination request."""
        try:
            coordination_id = f"COORD_{str(uuid.uuid4())[:8].upper()}"
            
            coordination_request = {
                "coordination_id": coordination_id,
                "task_type": task_type,
                "required_agents": required_agents,
                "task_data": task_data,
                "status": "pending",
                "created_at": datetime.utcnow(),
                "priority": task_data.get("priority", "medium")
            }
            
            await self.db[self.coordination_collection].insert_one(coordination_request)
            
            logger.info(f"Coordination request created: {coordination_id} for task {task_type}")
            
            return coordination_id
            
        except Exception as e:
            logger.error(f"Failed to create coordination request: {e}")
            return ""
    
    async def record_learning_insight(self, source_agent: str, insight_type: str, 
                                    insight_data: Dict[str, Any]) -> str:
        """Record a learning insight from an agent."""
        try:
            learning_id = f"LEARN_{str(uuid.uuid4())[:8].upper()}"
            
            learning_record = {
                "learning_id": learning_id,
                "source_agent": source_agent,
                "insight_type": insight_type,
                "insight_data": insight_data,
                "created_at": datetime.utcnow(),
                "shared": False
            }
            
            await self.db[self.learning_collection].insert_one(learning_record)
            
            logger.info(f"Learning insight recorded: {learning_id} from {source_agent}")
            
            return learning_id
            
        except Exception as e:
            logger.error(f"Failed to record learning insight: {e}")
            return ""
