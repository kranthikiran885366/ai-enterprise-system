"""Rule Engine for processing business rules."""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger

from shared_libs.database import get_database
from models.decision import Rule, RuleCreate, Decision, Alert


class RuleEngine:
    """Rule engine for processing business rules."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.rules_collection = "rules"
        self.decisions_collection = "decisions"
        self.alerts_collection = "alerts"
    
    async def initialize(self):
        """Initialize the rule engine."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.rules_collection].create_index("rule_id", unique=True)
        await self.db[self.rules_collection].create_index("status")
        await self.db[self.rules_collection].create_index("department")
        await self.db[self.rules_collection].create_index("priority")
        
        await self.db[self.decisions_collection].create_index("decision_id", unique=True)
        await self.db[self.decisions_collection].create_index("rule_id")
        await self.db[self.decisions_collection].create_index("created_at")
        
        await self.db[self.alerts_collection].create_index("alert_id", unique=True)
        await self.db[self.alerts_collection].create_index("status")
        await self.db[self.alerts_collection].create_index("severity")
        
        # Create default rules
        await self.create_default_rules()
        
        logger.info("Rule engine initialized")
    
    async def create_default_rules(self):
        """Create default business rules."""
        default_rules = [
            {
                "name": "High Expense Alert",
                "description": "Alert when expense exceeds $1000",
                "rule_type": "threshold",
                "conditions": {
                    "field": "expense.amount",
                    "operator": "greater_than",
                    "value": 1000
                },
                "actions": [
                    {
                        "type": "alert",
                        "severity": "high",
                        "message": "High expense detected: ${amount}"
                    }
                ],
                "priority": 8,
                "department": "finance"
            },
            {
                "name": "Employee Onboarding",
                "description": "Trigger onboarding process for new employees",
                "rule_type": "condition",
                "conditions": {
                    "field": "employee.status",
                    "operator": "equals",
                    "value": "new"
                },
                "actions": [
                    {
                        "type": "automation",
                        "action": "start_onboarding_workflow",
                        "parameters": {"employee_id": "{employee.id}"}
                    }
                ],
                "priority": 9,
                "department": "hr"
            },
            {
                "name": "Invoice Overdue",
                "description": "Alert when invoice is overdue",
                "rule_type": "condition",
                "conditions": {
                    "field": "invoice.due_date",
                    "operator": "less_than",
                    "value": "today"
                },
                "actions": [
                    {
                        "type": "notification",
                        "recipient": "finance_team",
                        "message": "Invoice {invoice.id} is overdue"
                    }
                ],
                "priority": 7,
                "department": "finance"
            }
        ]
        
        for rule_data in default_rules:
            existing_rule = await self.db[self.rules_collection].find_one({"name": rule_data["name"]})
            if not existing_rule:
                rule_id = f"RULE{str(uuid.uuid4())[:8].upper()}"
                rule_data.update({
                    "rule_id": rule_id,
                    "status": "active",
                    "created_by": "system",
                    "trigger_count": 0,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                })
                await self.db[self.rules_collection].insert_one(rule_data)
                logger.info(f"Default rule created: {rule_data['name']}")
    
    async def create_rule(self, rule_data: RuleCreate, created_by: str) -> Optional[Rule]:
        """Create a new rule."""
        try:
            rule_id = f"RULE{str(uuid.uuid4())[:8].upper()}"
            
            rule_dict = rule_data.dict()
            rule_dict.update({
                "rule_id": rule_id,
                "created_by": created_by,
                "trigger_count": 0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            result = await self.db[self.rules_collection].insert_one(rule_dict)
            
            if result.inserted_id:
                rule_dict["_id"] = result.inserted_id
                logger.info(f"Rule created: {rule_id}")
                return Rule(**rule_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create rule: {e}")
            return None
    
    async def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID."""
        try:
            rule_doc = await self.db[self.rules_collection].find_one({"rule_id": rule_id})
            
            if rule_doc:
                return Rule(**rule_doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get rule {rule_id}: {e}")
            return None
    
    async def list_rules(self, department: Optional[str] = None, status: Optional[str] = None,
                        skip: int = 0, limit: int = 10) -> List[Rule]:
        """List rules with optional filters."""
        try:
            query = {}
            if department:
                query["department"] = department
            if status:
                query["status"] = status Optional[str] = None, status: Optional[str] = None,
                        skip: int = 0, limit: int = 10) -> List[Rule]:
        """List rules with optional filters."""
        try:
            query = {}
            if department:
                query["department"] = department
            if status:
                query["status"] = status
            
            rules = []
            cursor = self.db[self.rules_collection].find(query).skip(skip).limit(limit).sort("priority", -1)
            
            async for rule_doc in cursor:
                rules.append(Rule(**rule_doc))
            
            return rules
            
        except Exception as e:
            logger.error(f"Failed to list rules: {e}")
            return []
    
    async def evaluate_rule(self, rule: Rule, data: Dict[str, Any]) -> bool:
        """Evaluate if a rule should be triggered based on data."""
        try:
            conditions = rule.conditions
            field = conditions.get("field")
            operator = conditions.get("operator")
            value = conditions.get("value")
            
            # Extract field value from data
            field_value = self._get_nested_value(data, field)
            
            if field_value is None:
                return False
            
            # Evaluate condition
            if operator == "equals":
                return field_value == value
            elif operator == "not_equals":
                return field_value != value
            elif operator == "greater_than":
                return float(field_value) > float(value)
            elif operator == "less_than":
                return float(field_value) &lt; float(value)
            elif operator == "greater_equal":
                return float(field_value) >= float(value)
            elif operator == "less_equal":
                return float(field_value) &lt;= float(value)
            elif operator == "contains":
                return str(value).lower() in str(field_value).lower()
            elif operator == "in":
                return field_value in value if isinstance(value, list) else False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate rule {rule.rule_id}: {e}")
            return False
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested value from data using dot notation."""
        try:
            keys = field_path.split(".")
            value = data
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            return value
            
        except Exception:
            return None
    
    async def trigger_rule(self, rule: Rule, data: Dict[str, Any]) -> Optional[Decision]:
        """Trigger a rule and execute its actions."""
        try:
            decision_id = f"DEC{str(uuid.uuid4())[:8].upper()}"
            actions_taken = []
            
            # Execute each action
            for action in rule.actions:
                action_result = await self._execute_action(action, data, rule)
                actions_taken.append(action_result)
            
            # Create decision record
            decision_dict = {
                "decision_id": decision_id,
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "triggered_by": data,
                "decision_data": {
                    "conditions_met": True,
                    "evaluation_time": datetime.utcnow().isoformat()
                },
                "actions_taken": actions_taken,
                "confidence_score": 1.0,  # Rule-based decisions have 100% confidence
                "department": rule.department,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await self.db[self.decisions_collection].insert_one(decision_dict)
            
            if result.inserted_id:
                decision_dict["_id"] = result.inserted_id
                
                # Update rule trigger count and last triggered
                await self.db[self.rules_collection].update_one(
                    {"rule_id": rule.rule_id},
                    {
                        "$set": {"last_triggered": datetime.utcnow()},
                        "$inc": {"trigger_count": 1}
                    }
                )
                
                logger.info(f"Rule triggered: {rule.rule_id} -> Decision: {decision_id}")
                return Decision(**decision_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to trigger rule {rule.rule_id}: {e}")
            return None
    
    async def _execute_action(self, action: Dict[str, Any], data: Dict[str, Any], rule: Rule) -> Dict[str, Any]:
        """Execute a single action."""
        try:
            action_type = action.get("type")
            
            if action_type == "alert":
                return await self._create_alert(action, data, rule)
            elif action_type == "notification":
                return await self._send_notification(action, data, rule)
            elif action_type == "automation":
                return await self._trigger_automation(action, data, rule)
            elif action_type == "escalation":
                return await self._escalate(action, data, rule)
            
            return {"type": action_type, "status": "unknown_action_type"}
            
        except Exception as e:
            logger.error(f"Failed to execute action {action}: {e}")
            return {"type": action.get("type"), "status": "failed", "error": str(e)}
    
    async def _create_alert(self, action: Dict[str, Any], data: Dict[str, Any], rule: Rule) -> Dict[str, Any]:
        """Create an alert."""
        try:
            alert_id = f"ALERT{str(uuid.uuid4())[:8].upper()}"
            
            # Format message with data
            message = action.get("message", "Alert triggered")
            message = self._format_message(message, data)
            
            alert_dict = {
                "alert_id": alert_id,
                "title": f"Rule Alert: {rule.name}",
                "message": message,
                "severity": action.get("severity", "medium"),
                "department": rule.department,
                "rule_id": rule.rule_id,
                "data": data,
                "status": "active",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            await self.db[self.alerts_collection].insert_one(alert_dict)
            
            logger.info(f"Alert created: {alert_id}")
            return {"type": "alert", "status": "created", "alert_id": alert_id}
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return {"type": "alert", "status": "failed", "error": str(e)}
    
    async def _send_notification(self, action: Dict[str, Any], data: Dict[str, Any], rule: Rule) -> Dict[str, Any]:
        """Send a notification."""
        try:
            recipient = action.get("recipient")
            message = action.get("message", "Notification from rule engine")
            message = self._format_message(message, data)
            
            # In a real implementation, this would send email, SMS, or push notification
            logger.info(f"Notification sent to {recipient}: {message}")
            
            return {"type": "notification", "status": "sent", "recipient": recipient}
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return {"type": "notification", "status": "failed", "error": str(e)}
    
    async def _trigger_automation(self, action: Dict[str, Any], data: Dict[str, Any], rule: Rule) -> Dict[str, Any]:
        """Trigger an automation."""
        try:
            automation_action = action.get("action")
            parameters = action.get("parameters", {})
            
            # Format parameters with data
            formatted_params = {}
            for key, value in parameters.items():
                if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                    field_path = value[1:-1]  # Remove { }
                    formatted_params[key] = self._get_nested_value(data, field_path)
                else:
                    formatted_params[key] = value
            
            # In a real implementation, this would trigger workflow automation
            logger.info(f"Automation triggered: {automation_action} with params: {formatted_params}")
            
            return {"type": "automation", "status": "triggered", "action": automation_action, "parameters": formatted_params}
            
        except Exception as e:
            logger.error(f"Failed to trigger automation: {e}")
            return {"type": "automation", "status": "failed", "error": str(e)}
    
    async def _escalate(self, action: Dict[str, Any], data: Dict[str, Any], rule: Rule) -> Dict[str, Any]:
        """Escalate an issue."""
        try:
            escalation_level = action.get("level", "manager")
            department = action.get("department", rule.department)
            
            # In a real implementation, this would escalate to appropriate personnel
            logger.info(f"Issue escalated to {escalation_level} in {department} department")
            
            return {"type": "escalation", "status": "escalated", "level": escalation_level, "department": department}
            
        except Exception as e:
            logger.error(f"Failed to escalate: {e}")
            return {"type": "escalation", "status": "failed", "error": str(e)}
    
    def _format_message(self, message: str, data: Dict[str, Any]) -> str:
        """Format message with data placeholders."""
        try:
            # Simple placeholder replacement
            import re
            
            def replace_placeholder(match):
                field_path = match.group(1)
                value = self._get_nested_value(data, field_path)
                return str(value) if value is not None else f"{{{field_path}}}"
            
            return re.sub(r'\{([^}]+)\}', replace_placeholder, message)
            
        except Exception:
            return message
    
    async def get_alerts(self, status: Optional[str] = None, severity: Optional[str] = None,
                        department: Optional[str] = None, skip: int = 0, limit: int = 10) -> List[Alert]:
        """Get alerts with filters."""
        try:
            query = {}
            if status:
                query["status"] = status
            if severity:
                query["severity"] = severity
            if department:
                query["department"] = department
            
            alerts = []
            cursor = self.db[self.alerts_collection].find(query).skip(skip).limit(limit).sort("created_at", -1)
            
            async for alert_doc in cursor:
                alerts.append(Alert(**alert_doc))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        try:
            result = await self.db[self.alerts_collection].update_one(
                {"alert_id": alert_id},
                {
                    "$set": {
                        "status": "acknowledged",
                        "acknowledged_by": acknowledged_by,
                        "acknowledged_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False
