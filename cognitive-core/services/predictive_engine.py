"""Predictive Engine - ML/AI powered forecasting and prediction system."""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger
import uuid
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

from shared_libs.database import get_database
from shared_libs.data_lake import get_data_lake


class PredictiveEngine:
    """ML/AI powered predictive engine for enterprise forecasting."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.predictions_collection = "predictions"
        self.models_collection = "prediction_models"
        self.forecasts_collection = "forecasts"
        self.is_running = False
        self.prediction_task: Optional[asyncio.Task] = None
        
        # ML Models
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
    
    async def initialize(self):
        """Initialize the Predictive Engine."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.predictions_collection].create_index("prediction_id", unique=True)
        await self.db[self.predictions_collection].create_index("prediction_type")
        await self.db[self.predictions_collection].create_index("created_at")
        
        await self.db[self.models_collection].create_index("model_id", unique=True)
        await self.db[self.models_collection].create_index("model_type")
        
        await self.db[self.forecasts_collection].create_index("forecast_id", unique=True)
        await self.db[self.forecasts_collection].create_index("forecast_type")
        await self.db[self.forecasts_collection].create_index("created_at")
        
        # Initialize ML models
        await self._initialize_ml_models()
        
        logger.info("Predictive Engine initialized")
    
    async def start_prediction_cycles(self):
        """Start prediction cycles."""
        if not self.is_running:
            self.is_running = True
            self.prediction_task = asyncio.create_task(self._prediction_loop())
            logger.info("Predictive Engine cycles started")
    
    async def stop_prediction_cycles(self):
        """Stop prediction cycles."""
        self.is_running = False
        if self.prediction_task:
            self.prediction_task.cancel()
            try:
                await self.prediction_task
            except asyncio.CancelledError:
                pass
        logger.info("Predictive Engine cycles stopped")
    
    async def _prediction_loop(self):
        """Main prediction loop."""
        while self.is_running:
            try:
                # Generate various predictions
                await self._generate_hr_predictions()
                await self._generate_finance_predictions()
                await self._generate_sales_predictions()
                await self._generate_operational_predictions()
                
                # Update model performance
                await self._evaluate_model_performance()
                
                # Retrain models if needed
                await self._retrain_models_if_needed()
                
                # Sleep for 1 hour before next cycle
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _initialize_ml_models(self):
        """Initialize ML models for different prediction types."""
        try:
            # HR Attrition Model
            self.models["hr_attrition"] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scalers["hr_attrition"] = StandardScaler()
            
            # Finance Budget Burn Model
            self.models["budget_burn"] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scalers["budget_burn"] = StandardScaler()
            
            # Sales Revenue Model
            self.models["sales_revenue"] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scalers["sales_revenue"] = StandardScaler()
            
            # Task Delay Model
            self.models["task_delay"] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scalers["task_delay"] = StandardScaler()
            
            # Resource Overload Model
            self.models["resource_overload"] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scalers["resource_overload"] = StandardScaler()
            
            # Anomaly Detection Model
            self.models["anomaly_detection"] = IsolationForest(contamination=0.1, random_state=42)
            
            # Train initial models with synthetic data
            await self._train_initial_models()
            
            logger.info("ML models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
    
    async def _train_initial_models(self):
        """Train initial models with available historical data."""
        try:
            data_lake = await get_data_lake()
            
            # Get historical data for training
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)  # Last year of data
            
            # Train HR attrition model
            await self._train_hr_attrition_model(start_date, end_date)
            
            # Train finance models
            await self._train_finance_models(start_date, end_date)
            
            # Train sales models
            await self._train_sales_models(start_date, end_date)
            
            # Train operational models
            await self._train_operational_models(start_date, end_date)
            
            logger.info("Initial model training completed")
            
        except Exception as e:
            logger.error(f"Failed to train initial models: {e}")
    
    async def _train_hr_attrition_model(self, start_date: datetime, end_date: datetime):
        """Train HR attrition prediction model."""
        try:
            # Get employee data
            employees = await self.db["employees"].find({
                "created_at": {"$gte": start_date, "$lte": end_date}
            }).to_list(None)
            
            if len(employees) < 10:  # Need minimum data for training
                logger.warning("Insufficient employee data for HR attrition model training")
                return
            
            # Prepare features and targets
            features = []
            targets = []
            
            for employee in employees:
                # Feature engineering
                tenure_days = (datetime.utcnow() - employee.get("created_at", datetime.utcnow())).days
                salary = employee.get("salary", 50000)
                performance_rating = employee.get("performance_rating", 3)
                department_code = hash(employee.get("department", "unknown")) % 10
                
                feature_vector = [
                    tenure_days,
                    salary,
                    performance_rating,
                    department_code,
                    employee.get("overtime_hours", 0),
                    employee.get("training_hours", 0)
                ]
                
                # Target: 1 if employee left, 0 if still active
                target = 1 if employee.get("status") == "terminated" else 0
                
                features.append(feature_vector)
                targets.append(target)
            
            # Train model
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.scalers["hr_attrition"].fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Train model
            self.models["hr_attrition"].fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.models["hr_attrition"].score(X_train, y_train)
            test_score = self.models["hr_attrition"].score(X_test, y_test)
            
            self.model_performance["hr_attrition"] = {
                "train_score": train_score,
                "test_score": test_score,
                "last_trained": datetime.utcnow(),
                "data_points": len(features)
            }
            
            logger.info(f"HR attrition model trained: train_score={train_score:.3f}, test_score={test_score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train HR attrition model: {e}")
    
    async def _train_finance_models(self, start_date: datetime, end_date: datetime):
        """Train finance prediction models."""
        try:
            # Get financial data
            expenses = await self.db["expenses"].find({
                "expense_date": {"$gte": start_date.date(), "$lte": end_date.date()}
            }).to_list(None)
            
            if len(expenses) < 20:
                logger.warning("Insufficient expense data for finance model training")
                return
            
            # Prepare budget burn rate features
            monthly_expenses = {}
            for expense in expenses:
                expense_date = expense.get("expense_date")
                if isinstance(expense_date, str):
                    expense_date = datetime.fromisoformat(expense_date).date()
                
                month_key = expense_date.strftime("%Y-%m")
                if month_key not in monthly_expenses:
                    monthly_expenses[month_key] = 0
                monthly_expenses[month_key] += expense.get("amount", 0)
            
            # Create time series features
            features = []
            targets = []
            
            months = sorted(monthly_expenses.keys())
            for i in range(3, len(months)):  # Need at least 3 months of history
                # Features: last 3 months of expenses
                feature_vector = [
                    monthly_expenses[months[i-3]],
                    monthly_expenses[months[i-2]],
                    monthly_expenses[months[i-1]],
                    i  # Time trend
                ]
                
                # Target: current month expense
                target = monthly_expenses[months[i]]
                
                features.append(feature_vector)
                targets.append(target)
            
            if len(features) >= 5:  # Minimum data for training
                X = np.array(features)
                y = np.array(targets)
                
                # Scale and train
                X_scaled = self.scalers["budget_burn"].fit_transform(X)
                self.models["budget_burn"].fit(X_scaled, y)
                
                # Evaluate
                score = self.models["budget_burn"].score(X_scaled, y)
                self.model_performance["budget_burn"] = {
                    "score": score,
                    "last_trained": datetime.utcnow(),
                    "data_points": len(features)
                }
                
                logger.info(f"Budget burn model trained: score={score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train finance models: {e}")
    
    async def _train_sales_models(self, start_date: datetime, end_date: datetime):
        """Train sales prediction models."""
        try:
            # Get sales data
            deals = await self.db["deals"].find({
                "created_at": {"$gte": start_date, "$lte": end_date}
            }).to_list(None)
            
            if len(deals) < 10:
                logger.warning("Insufficient deal data for sales model training")
                return
            
            # Prepare features
            features = []
            targets = []
            
            for deal in deals:
                # Feature engineering
                deal_value = deal.get("value", 0)
                days_in_pipeline = (datetime.utcnow() - deal.get("created_at", datetime.utcnow())).days
                lead_source_code = hash(deal.get("lead_source", "unknown")) % 5
                
                feature_vector = [
                    deal_value,
                    days_in_pipeline,
                    lead_source_code,
                    deal.get("probability", 0.5),
                    len(deal.get("activities", []))
                ]
                
                # Target: 1 if deal closed won, 0 otherwise
                target = 1 if deal.get("status") == "closed_won" else 0
                
                features.append(feature_vector)
                targets.append(target)
            
            # Train model
            X = np.array(features)
            y = np.array(targets)
            
            X_scaled = self.scalers["sales_revenue"].fit_transform(X)
            self.models["sales_revenue"].fit(X_scaled, y)
            
            score = self.models["sales_revenue"].score(X_scaled, y)
            self.model_performance["sales_revenue"] = {
                "score": score,
                "last_trained": datetime.utcnow(),
                "data_points": len(features)
            }
            
            logger.info(f"Sales revenue model trained: score={score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train sales models: {e}")
    
    async def _train_operational_models(self, start_date: datetime, end_date: datetime):
        """Train operational prediction models."""
        try:
            # Get task/project data
            tasks = await self.db["tasks"].find({
                "created_at": {"$gte": start_date, "$lte": end_date}
            }).to_list(None)
            
            if len(tasks) < 20:
                logger.warning("Insufficient task data for operational model training")
                return
            
            # Task delay prediction
            features = []
            targets = []
            
            for task in tasks:
                # Feature engineering
                estimated_hours = task.get("estimated_hours", 8)
                complexity = task.get("complexity", 3)
                assignee_experience = task.get("assignee_experience", 2)
                dependencies_count = len(task.get("dependencies", []))
                
                feature_vector = [
                    estimated_hours,
                    complexity,
                    assignee_experience,
                    dependencies_count,
                    task.get("priority", 3)
                ]
                
                # Target: actual delay in hours
                due_date = task.get("due_date")
                completed_date = task.get("completed_date")
                
                if due_date and completed_date:
                    if isinstance(due_date, str):
                        due_date = datetime.fromisoformat(due_date)
                    if isinstance(completed_date, str):
                        completed_date = datetime.fromisoformat(completed_date)
                    
                    delay_hours = max(0, (completed_date - due_date).total_seconds() / 3600)
                    
                    features.append(feature_vector)
                    targets.append(delay_hours)
            
            if len(features) >= 10:
                X = np.array(features)
                y = np.array(targets)
                
                X_scaled = self.scalers["task_delay"].fit_transform(X)
                self.models["task_delay"].fit(X_scaled, y)
                
                score = self.models["task_delay"].score(X_scaled, y)
                self.model_performance["task_delay"] = {
                    "score": score,
                    "last_trained": datetime.utcnow(),
                    "data_points": len(features)
                }
                
                logger.info(f"Task delay model trained: score={score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train operational models: {e}")
    
    async def _generate_hr_predictions(self):
        """Generate HR-related predictions."""
        try:
            # Predict employee attrition
            attrition_predictions = await self._predict_employee_attrition()
            
            # Predict hiring needs
            hiring_predictions = await self._predict_hiring_needs()
            
            # Store predictions
            if attrition_predictions:
                await self._store_prediction("hr_attrition", attrition_predictions)
            
            if hiring_predictions:
                await self._store_prediction("hr_hiring", hiring_predictions)
            
        except Exception as e:
            logger.error(f"Failed to generate HR predictions: {e}")
    
    async def _predict_employee_attrition(self) -> Dict[str, Any]:
        """Predict employee attrition risk."""
        try:
            if "hr_attrition" not in self.models:
                return {}
            
            # Get current employees
            employees = await self.db["employees"].find({
                "status": "active"
            }).to_list(None)
            
            if not employees:
                return {}
            
            predictions = []
            
            for employee in employees:
                # Prepare features
                tenure_days = (datetime.utcnow() - employee.get("created_at", datetime.utcnow())).days
                salary = employee.get("salary", 50000)
                performance_rating = employee.get("performance_rating", 3)
                department_code = hash(employee.get("department", "unknown")) % 10
                
                features = np.array([[
                    tenure_days,
                    salary,
                    performance_rating,
                    department_code,
                    employee.get("overtime_hours", 0),
                    employee.get("training_hours", 0)
                ]])
                
                # Scale features
                features_scaled = self.scalers["hr_attrition"].transform(features)
                
                # Predict
                attrition_probability = self.models["hr_attrition"].predict(features_scaled)[0]
                
                predictions.append({
                    "employee_id": employee.get("employee_id"),
                    "employee_name": f"{employee.get('first_name', '')} {employee.get('last_name', '')}",
                    "attrition_probability": float(attrition_probability),
                    "risk_level": "high" if attrition_probability > 0.7 else "medium" if attrition_probability > 0.4 else "low",
                    "department": employee.get("department"),
                    "tenure_days": tenure_days
                })
            
            # Sort by risk
            predictions.sort(key=lambda x: x["attrition_probability"], reverse=True)
            
            return {
                "total_employees": len(employees),
                "high_risk_employees": len([p for p in predictions if p["risk_level"] == "high"]),
                "predictions": predictions[:20],  # Top 20 at-risk employees
                "average_risk": sum(p["attrition_probability"] for p in predictions) / len(predictions),
                "prediction_date": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict employee attrition: {e}")
            return {}
    
    async def _predict_hiring_needs(self) -> Dict[str, Any]:
        """Predict hiring needs based on attrition and growth."""
        try:
            # Get department headcounts
            departments = await self.db["employees"].aggregate([
                {"$match": {"status": "active"}},
                {"$group": {"_id": "$department", "count": {"$sum": 1}}}
            ]).to_list(None)
            
            # Get recent attrition data
            recent_departures = await self.db["employees"].find({
                "status": "terminated",
                "termination_date": {"$gte": datetime.utcnow() - timedelta(days=90)}
            }).to_list(None)
            
            # Calculate attrition rate by department
            dept_attrition = {}
            for departure in recent_departures:
                dept = departure.get("department", "unknown")
                dept_attrition[dept] = dept_attrition.get(dept, 0) + 1
            
            # Predict hiring needs
            hiring_predictions = []
            
            for dept_data in departments:
                dept_name = dept_data["_id"]
                current_headcount = dept_data["count"]
                recent_departures_count = dept_attrition.get(dept_name, 0)
                
                # Simple prediction: quarterly attrition rate * 4 for annual
                annual_attrition_estimate = recent_departures_count * 4
                
                # Add growth factor (assume 10% growth)
                growth_hires = int(current_headcount * 0.1)
                
                total_hiring_need = annual_attrition_estimate + growth_hires
                
                if total_hiring_need > 0:
                    hiring_predictions.append({
                        "department": dept_name,
                        "current_headcount": current_headcount,
                        "predicted_attrition": annual_attrition_estimate,
                        "growth_hires": growth_hires,
                        "total_hiring_need": total_hiring_need,
                        "urgency": "high" if total_hiring_need > current_headcount * 0.2 else "medium"
                    })
            
            return {
                "hiring_predictions": hiring_predictions,
                "total_hiring_need": sum(p["total_hiring_need"] for p in hiring_predictions),
                "high_urgency_departments": [p["department"] for p in hiring_predictions if p["urgency"] == "high"],
                "prediction_date": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict hiring needs: {e}")
            return {}
    
    async def _generate_finance_predictions(self):
        """Generate finance-related predictions."""
        try:
            # Predict budget burn rate
            budget_predictions = await self._predict_budget_burn()
            
            # Predict cash flow
            cash_flow_predictions = await self._predict_cash_flow()
            
            # Store predictions
            if budget_predictions:
                await self._store_prediction("finance_budget", budget_predictions)
            
            if cash_flow_predictions:
                await self._store_prediction("finance_cash_flow", cash_flow_predictions)
            
        except Exception as e:
            logger.error(f"Failed to generate finance predictions: {e}")
    
    async def _predict_budget_burn(self) -> Dict[str, Any]:
        """Predict budget burn rate."""
        try:
            if "budget_burn" not in self.models:
                return {}
            
            # Get recent expense data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)  # Last 3 months
            
            expenses = await self.db["expenses"].find({
                "expense_date": {"$gte": start_date.date(), "$lte": end_date.date()}
            }).to_list(None)
            
            if len(expenses) < 10:
                return {}
            
            # Group by month
            monthly_expenses = {}
            for expense in expenses:
                expense_date = expense.get("expense_date")
                if isinstance(expense_date, str):
                    expense_date = datetime.fromisoformat(expense_date).date()
                
                month_key = expense_date.strftime("%Y-%m")
                if month_key not in monthly_expenses:
                    monthly_expenses[month_key] = 0
                monthly_expenses[month_key] += expense.get("amount", 0)
            
            # Prepare features for prediction
            months = sorted(monthly_expenses.keys())
            if len(months) < 3:
                return {}
            
            # Use last 3 months to predict next month
            last_3_months = months[-3:]
            features = np.array([[
                monthly_expenses[last_3_months[0]],
                monthly_expenses[last_3_months[1]],
                monthly_expenses[last_3_months[2]],
                len(months)  # Time trend
            ]])
            
            # Scale and predict
            features_scaled = self.scalers["budget_burn"].transform(features)
            predicted_burn = self.models["budget_burn"].predict(features_scaled)[0]
            
            # Calculate burn rate trend
            recent_avg = sum(monthly_expenses[month] for month in last_3_months) / 3
            burn_rate_change = (predicted_burn - recent_avg) / recent_avg if recent_avg > 0 else 0
            
            return {
                "predicted_monthly_burn": float(predicted_burn),
                "recent_average_burn": float(recent_avg),
                "burn_rate_change_percentage": float(burn_rate_change * 100),
                "trend": "increasing" if burn_rate_change > 0.1 else "decreasing" if burn_rate_change < -0.1 else "stable",
                "risk_level": "high" if burn_rate_change > 0.2 else "medium" if burn_rate_change > 0.1 else "low",
                "prediction_date": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict budget burn: {e}")
            return {}
    
    async def _predict_cash_flow(self) -> Dict[str, Any]:
        """Predict cash flow for next 3 months."""
        try:
            # Get revenue and expense data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=180)  # Last 6 months
            
            # Get invoices (revenue)
            invoices = await self.db["invoices"].find({
                "issue_date": {"$gte": start_date.date(), "$lte": end_date.date()},
                "status": "paid"
            }).to_list(None)
            
            # Get expenses
            expenses = await self.db["expenses"].find({
                "expense_date": {"$gte": start_date.date(), "$lte": end_date.date()}
            }).to_list(None)
            
            # Group by month
            monthly_revenue = {}
            monthly_expenses = {}
            
            for invoice in invoices:
                issue_date = invoice.get("issue_date")
                if isinstance(issue_date, str):
                    issue_date = datetime.fromisoformat(issue_date).date()
                
                month_key = issue_date.strftime("%Y-%m")
                if month_key not in monthly_revenue:
                    monthly_revenue[month_key] = 0
                monthly_revenue[month_key] += invoice.get("amount", 0)
            
            for expense in expenses:
                expense_date = expense.get("expense_date")
                if isinstance(expense_date, str):
                    expense_date = datetime.fromisoformat(expense_date).date()
                
                month_key = expense_date.strftime("%Y-%m")
                if month_key not in monthly_expenses:
                    monthly_expenses[month_key] = 0
                monthly_expenses[month_key] += expense.get("amount", 0)
            
            # Calculate historical cash flow
            all_months = sorted(set(list(monthly_revenue.keys()) + list(monthly_expenses.keys())))
            
            if len(all_months) < 3:
                return {}
            
            # Simple trend-based prediction for next 3 months
            predictions = []
            
            # Calculate averages for last 3 months
            recent_months = all_months[-3:]
            avg_revenue = sum(monthly_revenue.get(month, 0) for month in recent_months) / 3
            avg_expenses = sum(monthly_expenses.get(month, 0) for month in recent_months) / 3
            
            # Calculate trends
            revenue_trend = 0
            expense_trend = 0
            
            if len(all_months) >= 6:
                older_months = all_months[-6:-3]
                older_avg_revenue = sum(monthly_revenue.get(month, 0) for month in older_months) / 3
                older_avg_expenses = sum(monthly_expenses.get(month, 0) for month in older_months) / 3
                
                revenue_trend = (avg_revenue - older_avg_revenue) / 3  # Monthly trend
                expense_trend = (avg_expenses - older_avg_expenses) / 3
            
            # Predict next 3 months
            for i in range(1, 4):
                predicted_revenue = avg_revenue + (revenue_trend * i)
                predicted_expenses = avg_expenses + (expense_trend * i)
                predicted_cash_flow = predicted_revenue - predicted_expenses
                
                future_date = datetime.utcnow() + timedelta(days=30 * i)
                
                predictions.append({
                    "month": future_date.strftime("%Y-%m"),
                    "predicted_revenue": float(max(0, predicted_revenue)),
                    "predicted_expenses": float(max(0, predicted_expenses)),
                    "predicted_cash_flow": float(predicted_cash_flow),
                    "cash_flow_status": "positive" if predicted_cash_flow > 0 else "negative"
                })
            
            # Calculate overall outlook
            total_predicted_cash_flow = sum(p["predicted_cash_flow"] for p in predictions)
            negative_months = len([p for p in predictions if p["predicted_cash_flow"] < 0])
            
            return {
                "predictions": predictions,
                "total_predicted_cash_flow": float(total_predicted_cash_flow),
                "negative_cash_flow_months": negative_months,
                "outlook": "positive" if total_predicted_cash_flow > 0 and negative_months == 0 else "mixed" if negative_months <= 1 else "concerning",
                "recommendation": self._generate_cash_flow_recommendation(predictions),
                "prediction_date": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict cash flow: {e}")
            return {}
    
    def _generate_cash_flow_recommendation(self, predictions: List[Dict[str, Any]]) -> str:
        """Generate cash flow recommendation based on predictions."""
        try:
            negative_months = [p for p in predictions if p["predicted_cash_flow"] < 0]
            
            if not negative_months:
                return "Cash flow outlook is positive. Consider investing in growth opportunities."
            elif len(negative_months) == 1:
                return "One month of negative cash flow predicted. Monitor expenses and consider short-term financing options."
            else:
                return "Multiple months of negative cash flow predicted. Implement cost reduction measures and secure additional funding."
                
        except Exception as e:
            logger.error(f"Failed to generate cash flow recommendation: {e}")
            return "Monitor cash flow closely and adjust financial strategy as needed."
    
    async def _generate_sales_predictions(self):
        """Generate sales-related predictions."""
        try:
            # Predict deal closure probability
            deal_predictions = await self._predict_deal_closures()
            
            # Predict revenue forecast
            revenue_predictions = await self._predict_revenue_forecast()
            
            # Store predictions
            if deal_predictions:
                await self._store_prediction("sales_deals", deal_predictions)
            
            if revenue_predictions:
                await self._store_prediction("sales_revenue", revenue_predictions)
            
        except Exception as e:
            logger.error(f"Failed to generate sales predictions: {e}")
    
    async def _predict_deal_closures(self) -> Dict[str, Any]:
        """Predict deal closure probabilities."""
        try:
            if "sales_revenue" not in self.models:
                return {}
            
            # Get active deals
            active_deals = await self.db["deals"].find({
                "status": {"$in": ["prospecting", "qualification", "proposal", "negotiation"]}
            }).to_list(None)
            
            if not active_deals:
                return {}
            
            predictions = []
            
            for deal in active_deals:
                # Prepare features
                deal_value = deal.get("value", 0)
                days_in_pipeline = (datetime.utcnow() - deal.get("created_at", datetime.utcnow())).days
                lead_source_code = hash(deal.get("lead_source", "unknown")) % 5
                
                features = np.array([[
                    deal_value,
                    days_in_pipeline,
                    lead_source_code,
                    deal.get("probability", 0.5),
                    len(deal.get("activities", []))
                ]])
                
                # Scale and predict
                features_scaled = self.scalers["sales_revenue"].transform(features)
                closure_probability = self.models["sales_revenue"].predict(features_scaled)[0]
                
                predictions.append({
                    "deal_id": deal.get("deal_id"),
                    "deal_name": deal.get("name", "Unnamed Deal"),
                    "value": deal_value,
                    "current_probability": deal.get("probability", 0.5),
                    "predicted_probability": float(closure_probability),
                    "days_in_pipeline": days_in_pipeline,
                    "status": deal.get("status"),
                    "expected_close_date": deal.get("expected_close_date")
                })
            
            # Sort by predicted probability
            predictions.sort(key=lambda x: x["predicted_probability"], reverse=True)
            
            # Calculate summary metrics
            total_pipeline_value = sum(p["value"] for p in predictions)
            weighted_pipeline_value = sum(p["value"] * p["predicted_probability"] for p in predictions)
            high_probability_deals = [p for p in predictions if p["predicted_probability"] > 0.7]
            
            return {
                "total_deals": len(predictions),
                "total_pipeline_value": float(total_pipeline_value),
                "weighted_pipeline_value": float(weighted_pipeline_value),
                "high_probability_deals": len(high_probability_deals),
                "predictions": predictions[:20],  # Top 20 deals
                "conversion_rate_estimate": float(weighted_pipeline_value / total_pipeline_value) if total_pipeline_value > 0 else 0,
                "prediction_date": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict deal closures: {e}")
            return {}
    
    async def _predict_revenue_forecast(self) -> Dict[str, Any]:
        """Predict revenue forecast for next quarter."""
        try:
            # Get historical revenue data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)  # Last year
            
            invoices = await self.db["invoices"].find({
                "issue_date": {"$gte": start_date.date(), "$lte": end_date.date()},
                "status": "paid"
            }).to_list(None)
            
            if len(invoices) < 10:
                return {}
            
            # Group by month
            monthly_revenue = {}
            for invoice in invoices:
                issue_date = invoice.get("issue_date")
                if isinstance(issue_date, str):
                    issue_date = datetime.fromisoformat(issue_date).date()
                
                month_key = issue_date.strftime("%Y-%m")
                if month_key not in monthly_revenue:
                    monthly_revenue[month_key] = 0
                monthly_revenue[month_key] += invoice.get("amount", 0)
            
            # Calculate trend
            months = sorted(monthly_revenue.keys())
            if len(months) < 6:
                return {}
            
            # Simple trend calculation
            recent_6_months = months[-6:]
            older_6_months = months[-12:-6] if len(months) >= 12 else months[:-6]
            
            recent_avg = sum(monthly_revenue[month] for month in recent_6_months) / len(recent_6_months)
            older_avg = sum(monthly_revenue[month] for month in older_6_months) / len(older_6_months) if older_6_months else recent_avg
            
            growth_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            
            # Predict next 3 months
            predictions = []
            for i in range(1, 4):
                predicted_revenue = recent_avg * (1 + growth_rate) ** i
                future_date = datetime.utcnow() + timedelta(days=30 * i)
                
                predictions.append({
                    "month": future_date.strftime("%Y-%m"),
                    "predicted_revenue": float(max(0, predicted_revenue)),
                    "growth_rate": float(growth_rate),
                    "confidence": 0.7 if len(months) >= 12 else 0.5
                })
            
            total_predicted_revenue = sum(p["predicted_revenue"] for p in predictions)
            
            return {
                "quarterly_predictions": predictions,
                "total_predicted_revenue": float(total_predicted_revenue),
                "growth_rate": float(growth_rate),
                "trend": "growing" if growth_rate > 0.05 else "declining" if growth_rate < -0.05 else "stable",
                "confidence_level": "high" if len(months) >= 12 else "medium",
                "prediction_date": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict revenue forecast: {e}")
            return {}
    
    async def _generate_operational_predictions(self):
        """Generate operational predictions."""
        try:
            # Predict task delays
            task_predictions = await self._predict_task_delays()
            
            # Predict resource overload
            resource_predictions = await self._predict_resource_overload()
            
            # Store predictions
            if task_predictions:
                await self._store_prediction("operations_tasks", task_predictions)
            
            if resource_predictions:
                await self._store_prediction("operations_resources", resource_predictions)
            
        except Exception as e:
            logger.error(f"Failed to generate operational predictions: {e}")
    
    async def _predict_task_delays(self) -> Dict[str, Any]:
        """Predict task delay probabilities."""
        try:
            if "task_delay" not in self.models:
                return {}
            
            # Get active tasks
            active_tasks = await self.db["tasks"].find({
                "status": {"$in": ["todo", "in_progress"]},
                "due_date": {"$exists": True}
            }).to_list(None)
            
            if not active_tasks:
                return {}
            
            predictions = []
            
            for task in active_tasks:
                # Prepare features
                estimated_hours = task.get("estimated_hours", 8)
                complexity = task.get("complexity", 3)
                assignee_experience = task.get("assignee_experience", 2)
                dependencies_count = len(task.get("dependencies", []))
                
                features = np.array([[
                    estimated_hours,
                    complexity,
                    assignee_experience,
                    dependencies_count,
                    task.get("priority", 3)
                ]])
                
                # Scale and predict
                features_scaled = self.scalers["task_delay"].transform(features)
                predicted_delay_hours = self.models["task_delay"].predict(features_scaled)[0]
                
                # Calculate risk level
                due_date = task.get("due_date")
                if isinstance(due_date, str):
                    due_date = datetime.fromisoformat(due_date)
                
                days_until_due = (due_date - datetime.utcnow()).days
                
                risk_level = "high" if predicted_delay_hours > 24 else "medium" if predicted_delay_hours > 8 else "low"
                
                predictions.append({
                    "task_id": task.get("task_id"),
                    "task_name": task.get("name", "Unnamed Task"),
                    "assignee": task.get("assignee"),
                    "due_date": due_date.isoformat() if due_date else None,
                    "days_until_due": days_until_due,
                    "predicted_delay_hours": float(max(0, predicted_delay_hours)),
                    "risk_level": risk_level,
                    "estimated_hours": estimated_hours,
                    "complexity": complexity
                })
            
            # Sort by risk
            predictions.sort(key=lambda x: x["predicted_delay_hours"], reverse=True)
            
            # Calculate summary
            high_risk_tasks = [p for p in predictions if p["risk_level"] == "high"]
            total_predicted_delay = sum(p["predicted_delay_hours"] for p in predictions)
            
            return {
                "total_tasks": len(predictions),
                "high_risk_tasks": len(high_risk_tasks),
                "total_predicted_delay_hours": float(total_predicted_delay),
                "average_delay_per_task": float(total_predicted_delay / len(predictions)) if predictions else 0,
                "predictions": predictions[:15],  # Top 15 at-risk tasks
                "recommendation": self._generate_task_delay_recommendation(high_risk_tasks),
                "prediction_date": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict task delays: {e}")
            return {}
    
    def _generate_task_delay_recommendation(self, high_risk_tasks: List[Dict[str, Any]]) -> str:
        """Generate task delay recommendation."""
        try:
            if not high_risk_tasks:
                return "Task timeline looks healthy. Continue monitoring progress."
            elif len(high_risk_tasks) <= 2:
                return "Few high-risk tasks identified. Focus additional resources on these critical tasks."
            else:
                return "Multiple high-risk tasks detected. Consider redistributing workload and extending deadlines where possible."
                
        except Exception as e:
            logger.error(f"Failed to generate task delay recommendation: {e}")
            return "Monitor task progress closely and adjust resources as needed."
    
    async def _predict_resource_overload(self) -> Dict[str, Any]:
        """Predict resource overload scenarios."""
        try:
            # Get team members and their current workload
            team_members = await self.db["employees"].find({
                "status": "active",
                "role": {"$in": ["developer", "designer", "analyst", "manager"]}
            }).to_list(None)
            
            if not team_members:
                return {}
            
            # Get active tasks assigned to team members
            active_tasks = await self.db["tasks"].find({
                "status": {"$in": ["todo", "in_progress"]},
                "assignee": {"$exists": True}
            }).to_list(None)
            
            # Calculate workload by assignee
            workload_by_assignee = {}
            for task in active_tasks:
                assignee = task.get("assignee")
                if assignee:
                    if assignee not in workload_by_assignee:
                        workload_by_assignee[assignee] = {
                            "total_hours": 0,
                            "task_count": 0,
                            "high_priority_tasks": 0
                        }
                    
                    workload_by_assignee[assignee]["total_hours"] += task.get("estimated_hours", 8)
                    workload_by_assignee[assignee]["task_count"] += 1
                    
                    if task.get("priority", 3) >= 4:
                        workload_by_assignee[assignee]["high_priority_tasks"] += 1
            
            # Predict overload
            predictions = []
            
            for member in team_members:
                employee_id = member.get("employee_id")
                workload = workload_by_assignee.get(employee_id, {"total_hours": 0, "task_count": 0, "high_priority_tasks": 0})
                
                # Assume 40 hours per week capacity
                weekly_capacity = 40
                current_load_percentage = (workload["total_hours"] / weekly_capacity) * 100
                
                # Determine overload risk
                if current_load_percentage > 120:
                    risk_level = "critical"
                elif current_load_percentage > 100:
                    risk_level = "high"
                elif current_load_percentage > 80:
                    risk_level = "medium"
                else:
                    risk_level = "low"
                
                predictions.append({
                    "employee_id": employee_id,
                    "employee_name": f"{member.get('first_name', '')} {member.get('last_name', '')}",
                    "department": member.get("department"),
                    "role": member.get("role"),
                    "current_workload_hours": workload["total_hours"],
                    "task_count": workload["task_count"],
                    "high_priority_tasks": workload["high_priority_tasks"],
                    "capacity_utilization_percentage": float(current_load_percentage),
                    "risk_level": risk_level,
                    "weekly_capacity": weekly_capacity
                })
            
            # Sort by utilization
            predictions.sort(key=lambda x: x["capacity_utilization_percentage"], reverse=True)
            
            # Calculate summary
            overloaded_members = [p for p in predictions if p["risk_level"] in ["critical", "high"]]
            avg_utilization = sum(p["capacity_utilization_percentage"] for p in predictions) / len(predictions) if predictions else 0
            
            return {
                "total_team_members": len(predictions),
                "overloaded_members": len(overloaded_members),
                "average_utilization_percentage": float(avg_utilization),
                "predictions": predictions,
                "critical_cases": [p for p in predictions if p["risk_level"] == "critical"],
                "recommendation": self._generate_resource_overload_recommendation(overloaded_members, avg_utilization),
                "prediction_date": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict resource overload: {e}")
            return {}
    
    def _generate_resource_overload_recommendation(self, overloaded_members: List[Dict[str, Any]], avg_utilization: float) -> str:
        """Generate resource overload recommendation."""
        try:
            if not overloaded_members and avg_utilization < 80:
                return "Team capacity is healthy. Consider taking on additional projects."
            elif len(overloaded_members) <= 2:
                return "Few team members are overloaded. Consider redistributing tasks or extending deadlines."
            else:
                return "Multiple team members are overloaded. Consider hiring additional resources or reducing project scope."
                
        except Exception as e:
            logger.error(f"Failed to generate resource overload recommendation: {e}")
            return "Monitor team workload and adjust assignments as needed."
    
    async def _store_prediction(self, prediction_type: str, prediction_data: Dict[str, Any]):
        """Store prediction in database."""
        try:
            prediction_record = {
                "prediction_id": f"PRED_{str(uuid.uuid4())[:8].upper()}",
                "prediction_type": prediction_type,
                "prediction_data": prediction_data,
                "created_at": datetime.utcnow(),
                "model_version": "1.0",
                "confidence_level": prediction_data.get("confidence_level", "medium")
            }
            
            await self.db[self.predictions_collection].insert_one(prediction_record)
            
            # Store in data lake
            data_lake = await get_data_lake()
            await data_lake.store_event(
                agent="cognitive-core",
                event_type="prediction_generated",
                entity_type="prediction",
                entity_id=prediction_record["prediction_id"],
                data={
                    "prediction_type": prediction_type,
                    "confidence_level": prediction_record["confidence_level"]
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
    
    async def _evaluate_model_performance(self):
        """Evaluate model performance against actual outcomes."""
        try:
            # This would compare predictions with actual outcomes
            # For now, we'll implement a basic performance tracking
            
            for model_name, performance in self.model_performance.items():
                # Check if model needs retraining
                last_trained = performance.get("last_trained", datetime.min)
                days_since_training = (datetime.utcnow() - last_trained).days
                
                if days_since_training > 30:  # Retrain monthly
                    performance["needs_retraining"] = True
                else:
                    performance["needs_retraining"] = False
            
        except Exception as e:
            logger.error(f"Failed to evaluate model performance: {e}")
    
    async def _retrain_models_if_needed(self):
        """Retrain models if performance has degraded."""
        try:
            for model_name, performance in self.model_performance.items():
                if performance.get("needs_retraining", False):
                    logger.info(f"Retraining model: {model_name}")
                    
                    # Retrain based on model type
                    if model_name == "hr_attrition":
                        await self._train_hr_attrition_model(
                            datetime.utcnow() - timedelta(days=365),
                            datetime.utcnow()
                        )
                    elif model_name == "budget_burn":
                        await self._train_finance_models(
                            datetime.utcnow() - timedelta(days=365),
                            datetime.utcnow()
                        )
                    # Add other model retraining as needed
                    
                    performance["needs_retraining"] = False
                    performance["last_trained"] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to retrain models: {e}")
    
    async def get_prediction_status(self) -> Dict[str, Any]:
        """Get current prediction engine status."""
        try:
            # Get recent predictions count
            recent_predictions = await self.db[self.predictions_collection].count_documents({
                "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)}
            })
            
            # Calculate model health
            healthy_models = len([p for p in self.model_performance.values() if p.get("test_score", 0) > 0.6])
            total_models = len(self.model_performance)
            
            # Calculate prediction accuracy (simplified)
            avg_accuracy = sum(p.get("test_score", 0.5) for p in self.model_performance.values()) / total_models if total_models > 0 else 0.5
            
            return {
                "engine_status": "active" if self.is_running else "inactive",
                "total_models": total_models,
                "healthy_models": healthy_models,
                "model_health_percentage": (healthy_models / total_models) * 100 if total_models > 0 else 0,
                "recent_predictions_24h": recent_predictions,
                "prediction_accuracy": round(avg_accuracy, 3),
                "model_performance": self.model_performance,
                "last_updated": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get prediction status: {e}")
            return {"engine_status": "unknown"}
    
    async def generate_custom_prediction(self, prediction_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate custom prediction based on parameters."""
        try:
            if prediction_type == "employee_attrition":
                return await self._predict_employee_attrition()
            elif prediction_type == "budget_burn":
                return await self._predict_budget_burn()
            elif prediction_type == "deal_closure":
                return await self._predict_deal_closures()
            elif prediction_type == "task_delays":
                return await self._predict_task_delays()
            elif prediction_type == "resource_overload":
                return await self._predict_resource_overload()
            elif prediction_type == "cash_flow":
                return await self._predict_cash_flow()
            else:
                return {"error": f"Unknown prediction type: {prediction_type}"}
                
        except Exception as e:
            logger.error(f"Failed to generate custom prediction: {e}")
            return {"error": str(e)}
