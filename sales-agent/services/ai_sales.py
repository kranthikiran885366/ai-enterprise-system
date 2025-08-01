"""AI-powered sales services for Sales Agent."""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import openai
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger
import uuid
import requests
from bs4 import BeautifulSoup

from shared_libs.database import get_database
from shared_libs.intelligence import get_nlp_processor, get_ml_predictor, get_business_rules
from shared_libs.data_lake import get_data_lake

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY", "")


class AISalesService:
    """AI-powered sales automation and prediction service."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.lead_scoring_collection = "lead_scoring"
        self.email_campaigns_collection = "email_campaigns"
        self.market_signals_collection = "market_signals"
        self.churn_predictions_collection = "churn_predictions"
        self.sales_forecasts_collection = "sales_forecasts"
    
    async def initialize(self):
        """Initialize the AI sales service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.lead_scoring_collection].create_index("lead_id")
        await self.db[self.lead_scoring_collection].create_index("score")
        await self.db[self.lead_scoring_collection].create_index("created_at")
        
        await self.db[self.email_campaigns_collection].create_index("campaign_id", unique=True)
        await self.db[self.email_campaigns_collection].create_index("lead_id")
        
        await self.db[self.market_signals_collection].create_index("signal_id", unique=True)
        await self.db[self.market_signals_collection].create_index("signal_type")
        await self.db[self.market_signals_collection].create_index("created_at")
        
        await self.db[self.churn_predictions_collection].create_index("customer_id")
        await self.db[self.churn_predictions_collection].create_index("churn_probability")
        
        await self.db[self.sales_forecasts_collection].create_index("forecast_id", unique=True)
        await self.db[self.sales_forecasts_collection].create_index("forecast_date")
        
        logger.info("AI Sales service initialized")
    
    async def predict_customer_churn(self, customer_id: str) -> Dict[str, Any]:
        """Predict customer churn probability and generate retention strategies."""
        try:
            ml_predictor = await get_ml_predictor()
            data_lake = await get_data_lake()
            
            # Get customer data
            customer = await self.db["customers"].find_one({"customer_id": customer_id})
            if not customer:
                raise ValueError("Customer not found")
            
            # Get customer activity data
            customer_activity = await self._get_customer_activity(customer_id)
            
            # Get support ticket history
            support_tickets = await self._get_customer_support_history(customer_id)
            
            # Get payment history
            payment_history = await self._get_customer_payment_history(customer_id)
            
            # Get usage/engagement data
            usage_data = await self._get_customer_usage_data(customer_id)
            
            # Prepare data for ML prediction
            churn_features = {
                "days_since_last_activity": customer_activity.get("days_since_last_activity", 0),
                "support_tickets": len(support_tickets),
                "payment_delays": payment_history.get("payment_delays", 0),
                "usage_decline": usage_data.get("usage_decline", 0),
                "engagement_score": usage_data.get("engagement_score", 0.5),
                "contract_value": customer.get("contract_value", 0),
                "tenure_months": customer_activity.get("tenure_months", 0),
                "feature_adoption": usage_data.get("feature_adoption", 0.5),
                "support_satisfaction": support_tickets[-1].get("satisfaction_score", 5) if support_tickets else 5
            }
            
            # Get churn prediction
            churn_prediction = await ml_predictor.predict_churn(churn_features)
            
            # Analyze churn factors in detail
            detailed_factors = await self._analyze_churn_factors(customer_id, churn_features, support_tickets, payment_history, usage_data)
            
            # Generate retention strategies
            retention_strategies = await self._generate_retention_strategies(customer_id, churn_prediction, detailed_factors)
            
            # Calculate customer lifetime value at risk
            clv_at_risk = await self._calculate_clv_at_risk(customer, churn_prediction.get("churn_probability", 0))
            
            churn_analysis = {
                "analysis_id": f"CHURN_{str(uuid.uuid4())[:8].upper()}",
                "customer_id": customer_id,
                "churn_probability": churn_prediction.get("churn_probability", 0),
                "risk_level": churn_prediction.get("risk_level", "low"),
                "churn_factors": churn_prediction.get("factors", []),
                "detailed_factors": detailed_factors,
                "retention_strategies": retention_strategies,
                "clv_at_risk": clv_at_risk,
                "recommended_actions": retention_strategies.get("immediate_actions", []),
                "timeline_for_action": retention_strategies.get("timeline", "30_days"),
                "success_probability": retention_strategies.get("success_probability", 0.5),
                "features_analyzed": churn_features,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(days=30)
            }
            
            # Store churn prediction
            await self.db[self.churn_predictions_collection].insert_one(churn_analysis)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="sales",
                event_type="churn_prediction_completed",
                entity_type="customer",
                entity_id=customer_id,
                data={
                    "churn_probability": churn_prediction.get("churn_probability", 0),
                    "risk_level": churn_prediction.get("risk_level", "low"),
                    "clv_at_risk": clv_at_risk
                }
            )
            
            # Auto-execute retention actions for high-risk customers
            if churn_prediction.get("risk_level") in ["high", "critical"]:
                await self._auto_execute_retention_actions(customer_id, retention_strategies)
            
            logger.info(f"Churn prediction completed for customer {customer_id}: {churn_prediction.get('churn_probability', 0):.2f} probability, {churn_prediction.get('risk_level', 'low')} risk")
            
            return churn_analysis
            
        except Exception as e:
            logger.error(f"Failed to predict customer churn: {e}")
            return {}
    
    async def _get_customer_activity(self, customer_id: str) -> Dict[str, Any]:
        """Get customer activity data."""
        try:
            # Get recent activities
            activities = await self.db["customer_activities"].find({
                "customer_id": customer_id
            }).sort("timestamp", -1).limit(100).to_list(None)
            
            if not activities:
                return {"days_since_last_activity": 999, "tenure_months": 0}
            
            # Calculate days since last activity
            last_activity = activities[0].get("timestamp", datetime.utcnow())
            days_since_last = (datetime.utcnow() - last_activity).days
            
            # Calculate tenure
            first_activity = activities[-1].get("timestamp", datetime.utcnow())
            tenure_months = (datetime.utcnow() - first_activity).days / 30
            
            return {
                "days_since_last_activity": days_since_last,
                "tenure_months": tenure_months,
                "total_activities": len(activities)
            }
            
        except Exception as e:
            logger.error(f"Failed to get customer activity: {e}")
            return {"days_since_last_activity": 999, "tenure_months": 0}
    
    async def _get_customer_support_history(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get customer support ticket history."""
        try:
            tickets = await self.db["support_tickets"].find({
                "customer_id": customer_id
            }).sort("created_at", -1).limit(50).to_list(None)
            
            return tickets
            
        except Exception as e:
            logger.error(f"Failed to get customer support history: {e}")
            return []
    
    async def _get_customer_payment_history(self, customer_id: str) -> Dict[str, Any]:
        """Get customer payment history."""
        try:
            payments = await self.db["payments"].find({
                "customer_id": customer_id
            }).sort("payment_date", -1).limit(24).to_list(None)  # Last 24 payments
            
            if not payments:
                return {"payment_delays": 0, "avg_payment_time": 0}
            
            # Count payment delays
            payment_delays = 0
            payment_times = []
            
            for payment in payments:
                due_date = payment.get("due_date")
                paid_date = payment.get("paid_date")
                
                if due_date and paid_date:
                    if isinstance(due_date, str):
                        due_date = datetime.fromisoformat(due_date)
                    if isinstance(paid_date, str):
                        paid_date = datetime.fromisoformat(paid_date)
                    
                    if paid_date > due_date:
                        payment_delays += 1
                    
                    payment_time = (paid_date - due_date).days
                    payment_times.append(payment_time)
            
            avg_payment_time = sum(payment_times) / len(payment_times) if payment_times else 0
            
            return {
                "payment_delays": payment_delays,
                "avg_payment_time": avg_payment_time,
                "total_payments": len(payments)
            }
            
        except Exception as e:
            logger.error(f"Failed to get customer payment history: {e}")
            return {"payment_delays": 0, "avg_payment_time": 0}
    
    async def _get_customer_usage_data(self, customer_id: str) -> Dict[str, Any]:
        """Get customer usage and engagement data."""
        try:
            # Get usage data for last 6 months
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=180)
            
            usage_records = await self.db["usage_analytics"].find({
                "customer_id": customer_id,
                "date": {"$gte": start_date.date(), "$lte": end_date.date()}
            }).sort("date", -1).to_list(None)
            
            if not usage_records:
                return {"usage_decline": 0, "engagement_score": 0.5, "feature_adoption": 0.5}
            
            # Calculate usage decline
            recent_usage = usage_records[:30]  # Last 30 days
            older_usage = usage_records[30:60] if len(usage_records) > 60 else usage_records[30:]
            
            recent_avg = sum(r.get("usage_minutes", 0) for r in recent_usage) / len(recent_usage) if recent_usage else 0
            older_avg = sum(r.get("usage_minutes", 0) for r in older_usage) / len(older_usage) if older_usage else recent_avg
            
            usage_decline = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0
            usage_decline = max(0, min(1, usage_decline))  # Normalize to 0-1
            
            # Calculate engagement score
            total_sessions = sum(r.get("sessions", 0) for r in usage_records)
            total_features_used = len(set(feature for r in usage_records for feature in r.get("features_used", [])))
            avg_session_duration = sum(r.get("avg_session_duration", 0) for r in usage_records) / len(usage_records)
            
            engagement_score = min(1.0, (total_sessions / 100 + total_features_used / 20 + avg_session_duration / 60) / 3)
            
            # Calculate feature adoption
            available_features = 50  # Assume 50 available features
            feature_adoption = min(1.0, total_features_used / available_features)
            
            return {
                "usage_decline": usage_decline,
                "engagement_score": engagement_score,
                "feature_adoption": feature_adoption,
                "total_sessions": total_sessions,
                "avg_session_duration": avg_session_duration
            }
            
        except Exception as e:
            logger.error(f"Failed to get customer usage data: {e}")
            return {"usage_decline": 0, "engagement_score": 0.5, "feature_adoption": 0.5}
    
    async def _analyze_churn_factors(self, customer_id: str, features: Dict[str, Any], 
                                   support_tickets: List[Dict[str, Any]], 
                                   payment_history: Dict[str, Any], 
                                   usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detailed churn factors."""
        try:
            factors = {
                "engagement_factors": [],
                "support_factors": [],
                "payment_factors": [],
                "usage_factors": [],
                "relationship_factors": []
            }
            
            # Engagement factors
            if features.get("days_since_last_activity", 0) > 30:
                factors["engagement_factors"].append(f"No activity for {features['days_since_last_activity']} days")
            
            if usage_data.get("engagement_score", 0.5) < 0.3:
                factors["engagement_factors"].append("Low overall engagement score")
            
            # Support factors
            recent_tickets = [t for t in support_tickets if (datetime.utcnow() - t.get("created_at", datetime.utcnow())).days <= 30]
            if len(recent_tickets) > 3:
                factors["support_factors"].append(f"{len(recent_tickets)} support tickets in last 30 days")
            
            unsatisfied_tickets = [t for t in support_tickets if t.get("satisfaction_score", 5) < 3]
            if len(unsatisfied_tickets) > 1:
                factors["support_factors"].append(f"{len(unsatisfied_tickets)} tickets with low satisfaction")
            
            # Payment factors
            if payment_history.get("payment_delays", 0) > 2:
                factors["payment_factors"].append(f"{payment_history['payment_delays']} payment delays")
            
            if payment_history.get("avg_payment_time", 0) > 10:
                factors["payment_factors"].append(f"Average payment delay of {payment_history['avg_payment_time']:.1f} days")
            
            # Usage factors
            if usage_data.get("usage_decline", 0) > 0.3:
                factors["usage_factors"].append(f"Usage declined by {usage_data['usage_decline']*100:.1f}%")
            
            if usage_data.get("feature_adoption", 0.5) < 0.3:
                factors["usage_factors"].append("Low feature adoption rate")
            
            # Relationship factors
            if features.get("tenure_months", 0) < 6:
                factors["relationship_factors"].append("New customer (less than 6 months)")
            
            return factors
            
        except Exception as e:
            logger.error(f"Failed to analyze churn factors: {e}")
            return {}
    
    async def _generate_retention_strategies(self, customer_id: str, churn_prediction: Dict[str, Any], 
                                           detailed_factors: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized retention strategies."""
        try:
            risk_level = churn_prediction.get("risk_level", "low")
            churn_probability = churn_prediction.get("churn_probability", 0)
            
            strategies = {
                "immediate_actions": [],
                "short_term_actions": [],
                "long_term_actions": [],
                "timeline": "30_days",
                "success_probability": 0.5,
                "estimated_cost": 0,
                "expected_clv_saved": 0
            }
            
            # Immediate actions based on risk level
            if risk_level in ["critical", "high"]:
                strategies["immediate_actions"].extend([
                    "Schedule executive check-in call within 24 hours",
                    "Assign dedicated customer success manager",
                    "Offer immediate support for any issues"
                ])
                strategies["timeline"] = "7_days"
                strategies["success_probability"] = 0.7
            elif risk_level == "medium":
                strategies["immediate_actions"].extend([
                    "Schedule customer success call within 1 week",
                    "Send personalized retention offer",
                    "Provide additional training resources"
                ])
                strategies["timeline"] = "14_days"
                strategies["success_probability"] = 0.6
            
            # Actions based on specific factors
            engagement_factors = detailed_factors.get("engagement_factors", [])
            if engagement_factors:
                strategies["short_term_actions"].extend([
                    "Create personalized onboarding plan",
                    "Offer one-on-one training session",
                    "Send feature adoption guide"
                ])
            
            support_factors = detailed_factors.get("support_factors", [])
            if support_factors:
                strategies["immediate_actions"].extend([
                    "Review and resolve outstanding support issues",
                    "Assign premium support tier",
                    "Schedule technical review meeting"
                ])
            
            payment_factors = detailed_factors.get("payment_factors", [])
            if payment_factors:
                strategies["short_term_actions"].extend([
                    "Offer flexible payment terms",
                    "Discuss budget constraints",
                    "Consider temporary discount"
                ])
            
            usage_factors = detailed_factors.get("usage_factors", [])
            if usage_factors:
                strategies["short_term_actions"].extend([
                    "Conduct usage optimization review",
                    "Provide advanced feature training",
                    "Create custom workflow recommendations"
                ])
            
            # Long-term relationship building
            strategies["long_term_actions"].extend([
                "Establish quarterly business reviews",
                "Create customer advisory board invitation",
                "Develop strategic partnership opportunities",
                "Implement customer feedback loop"
            ])
            
            # Estimate costs and benefits
            if risk_level == "critical":
                strategies["estimated_cost"] = 5000
                strategies["expected_clv_saved"] = 50000
            elif risk_level == "high":
                strategies["estimated_cost"] = 2000
                strategies["expected_clv_saved"] = 25000
            elif risk_level == "medium":
                strategies["estimated_cost"] = 500
                strategies["expected_clv_saved"] = 10000
            
            return strategies
            
        except Exception as e:
            logger.error(f"Failed to generate retention strategies: {e}")
            return {}
    
    async def _calculate_clv_at_risk(self, customer: Dict[str, Any], churn_probability: float) -> float:
        """Calculate customer lifetime value at risk."""
        try:
            # Get customer contract value and tenure
            monthly_value = customer.get("monthly_recurring_revenue", 0)
            contract_length = customer.get("contract_length_months", 12)
            
            # Calculate remaining contract value
            remaining_months = max(0, contract_length - customer.get("months_active", 0))
            remaining_contract_value = monthly_value * remaining_months
            
            # Estimate future value beyond current contract
            estimated_renewal_probability = 0.8  # 80% renewal rate
            estimated_future_contracts = 2  # Average 2 more contract renewals
            future_value = monthly_value * contract_length * estimated_future_contracts * estimated_renewal_probability
            
            # Total CLV
            total_clv = remaining_contract_value + future_value
            
            # CLV at risk
            clv_at_risk = total_clv * churn_probability
            
            return round(clv_at_risk, 2)
            
        except Exception as e:
            logger.error(f"Failed to calculate CLV at risk: {e}")
            return 0
    
    async def _auto_execute_retention_actions(self, customer_id: str, retention_strategies: Dict[str, Any]) -> None:
        """Auto-execute retention actions for high-risk customers."""
        try:
            immediate_actions = retention_strategies.get("immediate_actions", [])
            
            for action in immediate_actions:
                if "schedule" in action.lower() and "call" in action.lower():
                    await self._schedule_retention_call(customer_id, action)
                elif "assign" in action.lower() and "manager" in action.lower():
                    await self._assign_customer_success_manager(customer_id)
                elif "send" in action.lower() and "offer" in action.lower():
                    await self._send_retention_offer(customer_id)
            
            logger.info(f"Auto-executed {len(immediate_actions)} retention actions for customer {customer_id}")
            
        except Exception as e:
            logger.error(f"Failed to auto-execute retention actions: {e}")
    
    async def _schedule_retention_call(self, customer_id: str, action: str) -> None:
        """Schedule retention call."""
        try:
            task = {
                "task_id": f"RETENTION_{str(uuid.uuid4())[:8].upper()}",
                "type": "retention_call",
                "customer_id": customer_id,
                "description": action,
                "priority": "critical",
                "status": "pending",
                "created_at": datetime.utcnow(),
                "assigned_to": "customer_success_team"
            }
            
            await self.db["tasks"].insert_one(task)
            logger.info(f"Retention call scheduled for customer {customer_id}")
            
        except Exception as e:
            logger.error(f"Failed to schedule retention call: {e}")
    
    async def _assign_customer_success_manager(self, customer_id: str) -> None:
        """Assign dedicated customer success manager."""
        try:
            # Update customer record
            await self.db["customers"].update_one(
                {"customer_id": customer_id},
                {
                    "$set": {
                        "dedicated_csm": True,
                        "csm_assigned_at": datetime.utcnow(),
                        "support_tier": "premium"
                    }
                }
            )
            
            logger.info(f"Dedicated CSM assigned to customer {customer_id}")
            
        except Exception as e:
            logger.error(f"Failed to assign customer success manager: {e}")
    
    async def _send_retention_offer(self, customer_id: str) -> None:
        """Send personalized retention offer."""
        try:
            offer = {
                "offer_id": f"OFFER_{str(uuid.uuid4())[:8].upper()}",
                "customer_id": customer_id,
                "offer_type": "retention_discount",
                "discount_percentage": 20,
                "valid_until": datetime.utcnow() + timedelta(days=30),
                "status": "active",
                "created_at": datetime.utcnow()
            }
            
            await self.db["retention_offers"].insert_one(offer)
            logger.info(f"Retention offer sent to customer {customer_id}")
            
        except Exception as e:
            logger.error(f"Failed to send retention offer: {e}")
