"""AI-powered finance services for Finance Agent."""

import os
import json
import asyncio
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional
import openai
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger
import uuid
import numpy as np

from shared_libs.database import get_database
from shared_libs.intelligence import get_nlp_processor, get_ml_predictor
from shared_libs.data_lake import get_data_lake

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY", "")


class AIFinanceService:
    """AI-powered finance automation and analysis service."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.expense_analysis_collection = "expense_analysis"
        self.fraud_detection_collection = "fraud_detection"
        self.budget_forecasts_collection = "budget_forecasts"
        self.cash_flow_predictions_collection = "cash_flow_predictions"
        self.financial_insights_collection = "financial_insights"
    
    async def initialize(self):
        """Initialize the AI finance service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.expense_analysis_collection].create_index("expense_id")
        await self.db[self.expense_analysis_collection].create_index("risk_score")
        await self.db[self.expense_analysis_collection].create_index("created_at")
        
        await self.db[self.fraud_detection_collection].create_index("detection_id", unique=True)
        await self.db[self.fraud_detection_collection].create_index("employee_id")
        await self.db[self.fraud_detection_collection].create_index("risk_level")
        
        await self.db[self.budget_forecasts_collection].create_index("forecast_id", unique=True)
        await self.db[self.budget_forecasts_collection].create_index("department")
        await self.db[self.budget_forecasts_collection].create_index("forecast_date")
        
        await self.db[self.cash_flow_predictions_collection].create_index("prediction_id", unique=True)
        await self.db[self.cash_flow_predictions_collection].create_index("prediction_date")
        
        await self.db[self.financial_insights_collection].create_index("insight_id", unique=True)
        await self.db[self.financial_insights_collection].create_index("insight_type")
        
        logger.info("AI Finance service initialized")
    
    async def analyze_expense_legitimacy(self, expense_data: Dict[str, Any], employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze expense legitimacy using AI and pattern recognition."""
        try:
            data_lake = await get_data_lake()
            nlp = await get_nlp_processor()
            
            # Get employee's expense history
            employee_id = expense_data.get("employee_id")
            expense_history = await self._get_employee_expense_history(employee_id, days=90)
            
            # Analyze expense patterns
            pattern_analysis = await self._analyze_expense_patterns(expense_data, expense_history)
            
            # Sentiment analysis of description
            description = expense_data.get("description", "")
            sentiment_analysis = await nlp.analyze_sentiment(description)
            
            # Anomaly detection
            anomaly_score = await self._detect_expense_anomalies(expense_data, expense_history)
            
            # Policy compliance check
            policy_compliance = await self._check_policy_compliance(expense_data, employee_data)
            
            # Calculate overall risk score
            risk_factors = []
            risk_score = 0.0
            
            # Pattern-based risk factors
            if pattern_analysis.get("unusual_amount", False):
                risk_score += 0.3
                risk_factors.append("Unusual expense amount for this employee")
            
            if pattern_analysis.get("unusual_category", False):
                risk_score += 0.2
                risk_factors.append("Unusual expense category for this employee")
            
            if pattern_analysis.get("frequency_anomaly", False):
                risk_score += 0.25
                risk_factors.append("Unusual expense frequency pattern")
            
            # Sentiment-based risk factors
            if sentiment_analysis.get("classification") == "negative":
                risk_score += 0.1
                risk_factors.append("Negative sentiment in description")
            
            # Anomaly-based risk factors
            if anomaly_score > 0.7:
                risk_score += 0.3
                risk_factors.append("Statistical anomaly detected")
            
            # Policy compliance risk factors
            if not policy_compliance.get("compliant", True):
                risk_score += 0.4
                risk_factors.extend(policy_compliance.get("violations", []))
            
            # Receipt validation
            if expense_data.get("amount", 0) > 25 and not expense_data.get("receipt_url"):
                risk_score += 0.2
                risk_factors.append("Missing receipt for expense over $25")
            
            # Time-based validation
            expense_date = expense_data.get("expense_date")
            if expense_date:
                if isinstance(expense_date, str):
                    expense_date = datetime.fromisoformat(expense_date).date()
                
                days_old = (date.today() - expense_date).days
                if days_old > 30:
                    risk_score += 0.15
                    risk_factors.append(f"Expense submitted {days_old} days after occurrence")
            
            # Cap risk score at 1.0
            risk_score = min(risk_score, 1.0)
            
            # Determine risk level
            if risk_score >= 0.8:
                risk_level = "critical"
                recommended_action = "manual_review_required"
            elif risk_score >= 0.6:
                risk_level = "high"
                recommended_action = "manager_approval_required"
            elif risk_score >= 0.4:
                risk_level = "medium"
                recommended_action = "automated_approval_with_audit"
            else:
                risk_level = "low"
                recommended_action = "automated_approval"
            
            analysis_result = {
                "analysis_id": f"EA_{str(uuid.uuid4())[:8].upper()}",
                "expense_id": expense_data.get("expense_id", "unknown"),
                "employee_id": employee_id,
                "risk_score": round(risk_score, 3),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommended_action": recommended_action,
                "pattern_analysis": pattern_analysis,
                "sentiment_analysis": sentiment_analysis,
                "anomaly_score": round(anomaly_score, 3),
                "policy_compliance": policy_compliance,
                "confidence": 0.85,
                "created_at": datetime.utcnow()
            }
            
            # Store analysis
            await self.db[self.expense_analysis_collection].insert_one(analysis_result)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="finance",
                event_type="expense_analyzed",
                entity_type="expense",
                entity_id=expense_data.get("expense_id", "unknown"),
                data={
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "recommended_action": recommended_action
                }
            )
            
            # Generate fraud alert if high risk
            if risk_level in ["critical", "high"]:
                await self._generate_fraud_alert(expense_data, analysis_result)
            
            logger.info(f"Expense legitimacy analyzed: {expense_data.get('expense_id', 'unknown')}, risk_score={risk_score:.3f}, level={risk_level}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze expense legitimacy: {e}")
            return {"risk_score": 0.5, "risk_level": "medium", "recommended_action": "manual_review"}
    
    async def _get_employee_expense_history(self, employee_id: str, days: int = 90) -> List[Dict[str, Any]]:
        """Get employee's expense history."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            expenses = await self.db["expenses"].find({
                "employee_id": employee_id,
                "created_at": {"$gte": cutoff_date}
            }).sort("created_at", -1).to_list(None)
            
            return expenses
            
        except Exception as e:
            logger.error(f"Failed to get expense history for {employee_id}: {e}")
            return []
    
    async def _analyze_expense_patterns(self, current_expense: Dict[str, Any], 
                                      expense_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze expense patterns for anomalies."""
        try:
            if not expense_history:
                return {"unusual_amount": False, "unusual_category": False, "frequency_anomaly": False}
            
            current_amount = current_expense.get("amount", 0)
            current_category = current_expense.get("category", "")
            
            # Amount pattern analysis
            historical_amounts = [exp.get("amount", 0) for exp in expense_history]
            if historical_amounts:
                avg_amount = sum(historical_amounts) / len(historical_amounts)
                std_amount = np.std(historical_amounts) if len(historical_amounts) > 1 else 0
                
                # Check if current amount is more than 2 standard deviations from mean
                unusual_amount = abs(current_amount - avg_amount) > (2 * std_amount) if std_amount > 0 else False
            else:
                unusual_amount = False
            
            # Category pattern analysis
            category_counts = {}
            for exp in expense_history:
                cat = exp.get("category", "other")
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Check if current category is rarely used (less than 10% of expenses)
            total_expenses = len(expense_history)
            category_frequency = category_counts.get(current_category, 0) / total_expenses if total_expenses > 0 else 0
            unusual_category = category_frequency < 0.1 and total_expenses > 10
            
            # Frequency pattern analysis
            recent_expenses = [
                exp for exp in expense_history
                if (datetime.utcnow() - exp.get("created_at", datetime.utcnow())).days <= 7
            ]
            frequency_anomaly = len(recent_expenses) > 10
            
            return {
                "unusual_amount": unusual_amount,
                "unusual_category": unusual_category,
                "frequency_anomaly": frequency_anomaly,
                "historical_avg_amount": round(avg_amount, 2) if historical_amounts else 0,
                "category_frequency": round(category_frequency, 3),
                "recent_expense_count": len(recent_expenses)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze expense patterns: {e}")
            return {"unusual_amount": False, "unusual_category": False, "frequency_anomaly": False}
    
    async def _detect_expense_anomalies(self, expense_data: Dict[str, Any], 
                                      expense_history: List[Dict[str, Any]]) -> float:
        """Detect anomalies in expense data using statistical methods."""
        try:
            if not expense_history:
                return 0.0
            
            anomaly_score = 0.0
            
            # Amount anomaly
            amounts = [exp.get("amount", 0) for exp in expense_history]
            if amounts:
                current_amount = expense_data.get("amount", 0)
                mean_amount = np.mean(amounts)
                std_amount = np.std(amounts)
                
                if std_amount > 0:
                    z_score = abs(current_amount - mean_amount) / std_amount
                    if z_score > 2:
                        anomaly_score += min(0.5, z_score / 4)
            
            # Time pattern anomaly
            expense_times = []
            for exp in expense_history:
                created_at = exp.get("created_at", datetime.utcnow())
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)
                expense_times.append(created_at.hour)
            
            if expense_times:
                current_hour = datetime.utcnow().hour
                hour_frequency = expense_times.count(current_hour) / len(expense_times)
                if hour_frequency < 0.05:
                    anomaly_score += 0.1
            
            return min(anomaly_score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to detect expense anomalies: {e}")
            return 0.0
    
    async def _check_policy_compliance(self, expense_data: Dict[str, Any], 
                                     employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check expense policy compliance."""
        try:
            violations = []
            compliant = True
            
            amount = expense_data.get("amount", 0)
            category = expense_data.get("category", "")
            employee_level = employee_data.get("level", "junior")
            
            # Category-specific policy checks
            policy_rules = {
                "meals": {
                    "max_amount": {"junior": 50, "senior": 75, "manager": 100, "director": 150},
                    "requires_receipt": True,
                    "business_justification_required": False
                },
                "travel": {
                    "max_amount": {"junior": 2000, "senior": 3000, "manager": 5000, "director": 10000},
                    "requires_receipt": True,
                    "business_justification_required": True
                }
            }
            
            if category in policy_rules:
                rules = policy_rules[category]
                
                # Amount limit check
                max_amount = rules["max_amount"].get(employee_level, 0)
                if amount > max_amount:
                    violations.append(f"Amount exceeds {employee_level} limit for {category} (${max_amount})")
                    compliant = False
                
                # Receipt requirement check
                if rules["requires_receipt"] and amount > 25 and not expense_data.get("receipt_url"):
                    violations.append("Receipt required for this expense category")
                    compliant = False
            
            return {"compliant": compliant, "violations": violations}
            
        except Exception as e:
            logger.error(f"Failed to check policy compliance: {e}")
            return {"compliant": True, "violations": []}
    
    async def _generate_fraud_alert(self, expense_data: Dict[str, Any], analysis_result: Dict[str, Any]) -> None:
        """Generate fraud alert for suspicious expenses."""
        try:
            alert = {
                "alert_id": f"FRAUD_{str(uuid.uuid4())[:8].upper()}",
                "type": "expense_fraud",
                "expense_id": expense_data.get("expense_id"),
                "employee_id": expense_data.get("employee_id"),
                "risk_score": analysis_result.get("risk_score"),
                "risk_factors": analysis_result.get("risk_factors", []),
                "amount": expense_data.get("amount"),
                "category": expense_data.get("category"),
                "created_at": datetime.utcnow(),
                "status": "active"
            }
            
            await self.db[self.fraud_detection_collection].insert_one(alert)
            logger.warning(f"Fraud alert generated: {alert['alert_id']}")
            
        except Exception as e:
            logger.error(f"Failed to generate fraud alert: {e}")
    
    async def predict_cash_flow(self, months_ahead: int = 6) -> Dict[str, Any]:
        """Predict cash flow for upcoming months."""
        try:
            # Get historical data
            historical_data = await self._get_historical_financial_data(months=12)
            
            # Analyze trends
            trends = await self._analyze_financial_trends(historical_data)
            
            # Generate predictions
            predictions = []
            current_date = datetime.utcnow().date()
            
            for month in range(1, months_ahead + 1):
                prediction_date = current_date.replace(month=current_date.month + month)
                if prediction_date.month > 12:
                    prediction_date = prediction_date.replace(year=prediction_date.year + 1, month=prediction_date.month - 12)
                
                monthly_prediction = await self._predict_monthly_cash_flow(prediction_date, trends, historical_data)
                predictions.append(monthly_prediction)
            
            prediction_result = {
                "prediction_id": f"CF_{str(uuid.uuid4())[:8].upper()}",
                "months_ahead": months_ahead,
                "predictions": predictions,
                "trends_analysis": trends,
                "confidence_score": 0.75,
                "created_at": datetime.utcnow()
            }
            
            # Store prediction
            await self.db[self.cash_flow_predictions_collection].insert_one(prediction_result)
            
            logger.info(f"Cash flow prediction completed: {prediction_result['prediction_id']}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Failed to predict cash flow: {e}")
            return {}
    
    async def _get_historical_financial_data(self, months: int = 12) -> Dict[str, Any]:
        """Get historical financial data."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=months * 30)
            
            # Get expenses
            expenses = await self.db["expenses"].find({
                "created_at": {"$gte": cutoff_date},
                "status": {"$in": ["approved", "paid"]}
            }).to_list(None)
            
            # Get invoices
            invoices = await self.db["invoices"].find({
                "created_at": {"$gte": cutoff_date}
            }).to_list(None)
            
            # Get payroll
            payroll = await self.db["payroll_records"].find({
                "created_at": {"$gte": cutoff_date}
            }).to_list(None)
            
            return {
                "expenses": expenses,
                "invoices": invoices,
                "payroll": payroll
            }
            
        except Exception as e:
            logger.error(f"Failed to get historical financial data: {e}")
            return {"expenses": [], "invoices": [], "payroll": []}
    
    async def _analyze_financial_trends(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial trends from historical data."""
        try:
            expenses = historical_data.get("expenses", [])
            invoices = historical_data.get("invoices", [])
            
            # Monthly expense trends
            monthly_expenses = {}
            for expense in expenses:
                created_at = expense.get("created_at", datetime.utcnow())
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)
                
                month_key = created_at.strftime("%Y-%m")
                monthly_expenses[month_key] = monthly_expenses.get(month_key, 0) + expense.get("amount", 0)
            
            # Monthly revenue trends
            monthly_revenue = {}
            for invoice in invoices:
                if invoice.get("status") == "paid":
                    paid_at = invoice.get("paid_at", invoice.get("created_at", datetime.utcnow()))
                    if isinstance(paid_at, str):
                        paid_at = datetime.fromisoformat(paid_at)
                    
                    month_key = paid_at.strftime("%Y-%m")
                    monthly_revenue[month_key] = monthly_revenue.get(month_key, 0) + invoice.get("amount", 0)
            
            # Calculate trends
            expense_trend = self._calculate_trend(list(monthly_expenses.values()))
            revenue_trend = self._calculate_trend(list(monthly_revenue.values()))
            
            return {
                "monthly_expenses": monthly_expenses,
                "monthly_revenue": monthly_revenue,
                "expense_trend": expense_trend,
                "revenue_trend": revenue_trend,
                "avg_monthly_expenses": sum(monthly_expenses.values()) / len(monthly_expenses) if monthly_expenses else 0,
                "avg_monthly_revenue": sum(monthly_revenue.values()) / len(monthly_revenue) if monthly_revenue else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze financial trends: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend from a series of values."""
        if len(values) < 2:
            return {"direction": "stable", "rate": 0.0}
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        y = values
        
        # Calculate slope
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
        
        # Determine direction
        if slope > 0.05:
            direction = "increasing"
        elif slope < -0.05:
            direction = "decreasing"
        else:
            direction = "stable"
        
        return {"direction": direction, "rate": round(slope, 4)}
    
    async def _predict_monthly_cash_flow(self, prediction_date: date, trends: Dict[str, Any], 
                                       historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict cash flow for a specific month."""
        try:
            # Base predictions on trends
            avg_expenses = trends.get("avg_monthly_expenses", 0)
            avg_revenue = trends.get("avg_monthly_revenue", 0)
            
            expense_trend = trends.get("expense_trend", {})
            revenue_trend = trends.get("revenue_trend", {})
            
            # Adjust for trends
            if expense_trend.get("direction") == "increasing":
                predicted_expenses = avg_expenses * (1 + expense_trend.get("rate", 0))
            elif expense_trend.get("direction") == "decreasing":
                predicted_expenses = avg_expenses * (1 - abs(expense_trend.get("rate", 0)))
            else:
                predicted_expenses = avg_expenses
            
            if revenue_trend.get("direction") == "increasing":
                predicted_revenue = avg_revenue * (1 + revenue_trend.get("rate", 0))
            elif revenue_trend.get("direction") == "decreasing":
                predicted_revenue = avg_revenue * (1 - abs(revenue_trend.get("rate", 0)))
            else:
                predicted_revenue = avg_revenue
            
            # Calculate net cash flow
            predicted_net_flow = predicted_revenue - predicted_expenses
            
            return {
                "month": prediction_date.strftime("%Y-%m"),
                "predicted_revenue": round(predicted_revenue, 2),
                "predicted_expenses": round(predicted_expenses, 2),
                "predicted_net_flow": round(predicted_net_flow, 2),
                "cash_flow_status": "positive" if predicted_net_flow > 0 else "negative"
            }
            
        except Exception as e:
            logger.error(f"Failed to predict monthly cash flow: {e}")
            return {}