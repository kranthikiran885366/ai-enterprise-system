"""Shared Intelligence Layer for all agents with NLP, ML models, and business rules."""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import openai
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY", "")

class NLPProcessor:
    """Natural Language Processing utilities for all agents."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.document_vectors = {}
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            # Classify sentiment
            if sentiment.polarity > 0.1:
                classification = "positive"
            elif sentiment.polarity < -0.1:
                classification = "negative"
            else:
                classification = "neutral"
            
            return {
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity,
                "classification": classification,
                "confidence": abs(sentiment.polarity)
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"classification": "neutral", "polarity": 0.0, "subjectivity": 0.0, "confidence": 0.0}
    
    async def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        try:
            blob = TextBlob(text)
            # Get noun phrases as keywords
            keywords = list(blob.noun_phrases)
            
            # If not enough noun phrases, add individual words
            if len(keywords) < num_keywords:
                words = [word.lower() for word in blob.words if len(word) > 3]
                keywords.extend(words)
            
            return keywords[:num_keywords]
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    async def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text into predefined categories using AI."""
        try:
            if not openai.api_key:
                return {cat: 0.0 for cat in categories}
            
            prompt = f"""
            Classify the following text into these categories: {', '.join(categories)}
            
            Text: "{text}"
            
            Return a JSON object with category names as keys and confidence scores (0-1) as values.
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            return {cat: 0.0 for cat in categories}
    
    async def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Summarize text using AI."""
        try:
            if not openai.api_key:
                # Fallback to simple truncation
                return text[:max_length] + "..." if len(text) > max_length else text
            
            prompt = f"""
            Summarize the following text in {max_length} characters or less:
            
            {text}
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        try:
            vectors = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0


class MLPredictor:
    """Machine Learning prediction utilities."""
    
    def __init__(self):
        self.models = {}
        self.training_data = {}
    
    async def predict_churn(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict customer churn probability."""
        try:
            # Simple rule-based prediction (in production, use trained ML model)
            score = 0.0
            
            # Factors that increase churn probability
            if customer_data.get("days_since_last_activity", 0) > 30:
                score += 0.3
            if customer_data.get("support_tickets", 0) > 5:
                score += 0.2
            if customer_data.get("payment_delays", 0) > 2:
                score += 0.25
            if customer_data.get("usage_decline", 0) > 0.5:
                score += 0.25
            
            # Cap at 1.0
            score = min(score, 1.0)
            
            return {
                "churn_probability": score,
                "risk_level": "high" if score > 0.7 else "medium" if score > 0.4 else "low",
                "factors": self._get_churn_factors(customer_data, score)
            }
            
        except Exception as e:
            logger.error(f"Churn prediction failed: {e}")
            return {"churn_probability": 0.0, "risk_level": "low", "factors": []}
    
    def _get_churn_factors(self, data: Dict[str, Any], score: float) -> List[str]:
        """Get factors contributing to churn risk."""
        factors = []
        if data.get("days_since_last_activity", 0) > 30:
            factors.append("Inactive for over 30 days")
        if data.get("support_tickets", 0) > 5:
            factors.append("High number of support tickets")
        if data.get("payment_delays", 0) > 2:
            factors.append("Multiple payment delays")
        if data.get("usage_decline", 0) > 0.5:
            factors.append("Significant usage decline")
        return factors
    
    async def predict_lead_score(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict lead conversion probability."""
        try:
            score = 0.0
            
            # Scoring factors
            if lead_data.get("company_size", 0) > 100:
                score += 0.2
            if lead_data.get("budget", 0) > 10000:
                score += 0.25
            if lead_data.get("decision_maker", False):
                score += 0.2
            if lead_data.get("engagement_score", 0) > 0.7:
                score += 0.25
            if lead_data.get("industry") in ["technology", "finance", "healthcare"]:
                score += 0.1
            
            score = min(score, 1.0)
            
            return {
                "lead_score": score,
                "grade": "A" if score > 0.8 else "B" if score > 0.6 else "C" if score > 0.4 else "D",
                "conversion_probability": score,
                "recommended_actions": self._get_lead_actions(score)
            }
            
        except Exception as e:
            logger.error(f"Lead scoring failed: {e}")
            return {"lead_score": 0.0, "grade": "D", "conversion_probability": 0.0, "recommended_actions": []}
    
    def _get_lead_actions(self, score: float) -> List[str]:
        """Get recommended actions based on lead score."""
        if score > 0.8:
            return ["Schedule demo immediately", "Assign to senior sales rep", "Send pricing information"]
        elif score > 0.6:
            return ["Send case studies", "Schedule follow-up call", "Provide product trial"]
        elif score > 0.4:
            return ["Send educational content", "Add to nurture campaign", "Qualify further"]
        else:
            return ["Add to long-term nurture", "Send general information", "Monitor engagement"]
    
    async def predict_employee_attrition(self, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict employee attrition risk."""
        try:
            score = 0.0
            
            # Risk factors
            if employee_data.get("tenure_months", 0) < 6:
                score += 0.3  # New employees are at higher risk
            if employee_data.get("performance_rating", 5) < 3:
                score += 0.25
            if employee_data.get("salary_vs_market", 1.0) < 0.9:
                score += 0.2
            if employee_data.get("manager_rating", 5) < 3:
                score += 0.15
            if employee_data.get("overtime_hours", 0) > 20:
                score += 0.1
            
            score = min(score, 1.0)
            
            return {
                "attrition_risk": score,
                "risk_level": "high" if score > 0.7 else "medium" if score > 0.4 else "low",
                "retention_actions": self._get_retention_actions(score, employee_data)
            }
            
        except Exception as e:
            logger.error(f"Attrition prediction failed: {e}")
            return {"attrition_risk": 0.0, "risk_level": "low", "retention_actions": []}
    
    def _get_retention_actions(self, score: float, data: Dict[str, Any]) -> List[str]:
        """Get retention actions based on risk score."""
        actions = []
        if score > 0.7:
            actions.extend(["Schedule immediate 1:1 with manager", "Review compensation", "Discuss career development"])
        if data.get("salary_vs_market", 1.0) < 0.9:
            actions.append("Consider salary adjustment")
        if data.get("overtime_hours", 0) > 20:
            actions.append("Review workload and redistribute tasks")
        if data.get("manager_rating", 5) < 3:
            actions.append("Provide manager coaching or reassignment")
        return actions


class BusinessRulesEngine:
    """Business rules engine for automated decision making."""
    
    def __init__(self):
        self.rules = {}
        self.rule_history = []
    
    async def evaluate_expense_approval(self, expense_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate expense for automatic approval."""
        try:
            amount = expense_data.get("amount", 0)
            category = expense_data.get("category", "")
            employee_level = expense_data.get("employee_level", "junior")
            
            # Auto-approval rules
            if amount <= 100 and category in ["meals", "office_supplies"]:
                return {"auto_approve": True, "reason": "Small amount in approved category"}
            
            if amount <= 500 and employee_level in ["senior", "manager", "director"]:
                return {"auto_approve": True, "reason": "Within senior employee limit"}
            
            if amount <= 1000 and category == "travel" and employee_level in ["manager", "director"]:
                return {"auto_approve": True, "reason": "Travel expense within manager limit"}
            
            # Requires approval
            approval_level = "manager" if amount <= 2000 else "director" if amount <= 5000 else "ceo"
            
            return {
                "auto_approve": False,
                "requires_approval": True,
                "approval_level": approval_level,
                "reason": f"Amount ${amount} requires {approval_level} approval"
            }
            
        except Exception as e:
            logger.error(f"Expense approval evaluation failed: {e}")
            return {"auto_approve": False, "requires_approval": True, "approval_level": "manager"}
    
    async def evaluate_lead_routing(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate lead routing rules."""
        try:
            company_size = lead_data.get("company_size", 0)
            budget = lead_data.get("budget", 0)
            industry = lead_data.get("industry", "")
            region = lead_data.get("region", "")
            
            # Enterprise routing
            if company_size > 1000 or budget > 100000:
                return {
                    "route_to": "enterprise_team",
                    "priority": "high",
                    "reason": "Enterprise prospect"
                }
            
            # Mid-market routing
            if company_size > 100 or budget > 25000:
                return {
                    "route_to": "midmarket_team",
                    "priority": "medium",
                    "reason": "Mid-market prospect"
                }
            
            # SMB routing
            return {
                "route_to": "smb_team",
                "priority": "normal",
                "reason": "Small business prospect"
            }
            
        except Exception as e:
            logger.error(f"Lead routing evaluation failed: {e}")
            return {"route_to": "smb_team", "priority": "normal", "reason": "Default routing"}
    
    async def evaluate_support_escalation(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate support ticket escalation rules."""
        try:
            severity = ticket_data.get("severity", "low")
            customer_tier = ticket_data.get("customer_tier", "standard")
            response_time = ticket_data.get("hours_since_created", 0)
            sentiment = ticket_data.get("sentiment", "neutral")
            
            # Immediate escalation conditions
            if severity == "critical" or customer_tier == "enterprise":
                return {
                    "escalate": True,
                    "escalation_level": "senior_support",
                    "reason": "Critical issue or enterprise customer"
                }
            
            # Time-based escalation
            if response_time > 24 and severity == "high":
                return {
                    "escalate": True,
                    "escalation_level": "manager",
                    "reason": "High severity ticket unresolved for 24+ hours"
                }
            
            # Sentiment-based escalation
            if sentiment == "negative" and customer_tier in ["premium", "enterprise"]:
                return {
                    "escalate": True,
                    "escalation_level": "senior_support",
                    "reason": "Negative sentiment from premium customer"
                }
            
            return {"escalate": False, "reason": "No escalation criteria met"}
            
        except Exception as e:
            logger.error(f"Support escalation evaluation failed: {e}")
            return {"escalate": False, "reason": "Evaluation error"}


class FeedbackLearningSystem:
    """Feedback and learning system for continuous improvement."""
    
    def __init__(self):
        self.feedback_data = {}
        self.learning_metrics = {}
    
    async def record_feedback(self, agent_name: str, action: str, outcome: str, 
                            feedback_score: float, context: Dict[str, Any]) -> None:
        """Record feedback for an agent action."""
        try:
            feedback_entry = {
                "timestamp": datetime.utcnow(),
                "agent": agent_name,
                "action": action,
                "outcome": outcome,
                "score": feedback_score,
                "context": context
            }
            
            if agent_name not in self.feedback_data:
                self.feedback_data[agent_name] = []
            
            self.feedback_data[agent_name].append(feedback_entry)
            
            # Update learning metrics
            await self._update_learning_metrics(agent_name, action, feedback_score)
            
            logger.info(f"Feedback recorded for {agent_name}: {action} -> {outcome} (score: {feedback_score})")
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
    
    async def _update_learning_metrics(self, agent_name: str, action: str, score: float) -> None:
        """Update learning metrics based on feedback."""
        try:
            if agent_name not in self.learning_metrics:
                self.learning_metrics[agent_name] = {}
            
            if action not in self.learning_metrics[agent_name]:
                self.learning_metrics[agent_name][action] = {
                    "total_feedback": 0,
                    "average_score": 0.0,
                    "improvement_trend": 0.0
                }
            
            metrics = self.learning_metrics[agent_name][action]
            
            # Update running average
            total = metrics["total_feedback"]
            current_avg = metrics["average_score"]
            new_avg = (current_avg * total + score) / (total + 1)
            
            metrics["total_feedback"] += 1
            metrics["average_score"] = new_avg
            
            # Calculate improvement trend (simple moving average of last 10 scores)
            recent_feedback = [f["score"] for f in self.feedback_data[agent_name][-10:] 
                             if f["action"] == action]
            if len(recent_feedback) >= 2:
                metrics["improvement_trend"] = (recent_feedback[-1] - recent_feedback[0]) / len(recent_feedback)
            
        except Exception as e:
            logger.error(f"Failed to update learning metrics: {e}")
    
    async def get_learning_insights(self, agent_name: str) -> Dict[str, Any]:
        """Get learning insights for an agent."""
        try:
            if agent_name not in self.learning_metrics:
                return {"insights": [], "recommendations": []}
            
            metrics = self.learning_metrics[agent_name]
            insights = []
            recommendations = []
            
            for action, data in metrics.items():
                avg_score = data["average_score"]
                trend = data["improvement_trend"]
                
                if avg_score < 0.6:
                    insights.append(f"Action '{action}' has low performance (avg: {avg_score:.2f})")
                    recommendations.append(f"Review and improve '{action}' process")
                
                if trend > 0.1:
                    insights.append(f"Action '{action}' is improving (trend: +{trend:.2f})")
                elif trend < -0.1:
                    insights.append(f"Action '{action}' is declining (trend: {trend:.2f})")
                    recommendations.append(f"Investigate decline in '{action}' performance")
            
            return {
                "insights": insights,
                "recommendations": recommendations,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return {"insights": [], "recommendations": []}


# Global instances
nlp_processor = NLPProcessor()
ml_predictor = MLPredictor()
business_rules = BusinessRulesEngine()
feedback_system = FeedbackLearningSystem()


async def get_nlp_processor() -> NLPProcessor:
    """Get NLP processor instance."""
    return nlp_processor


async def get_ml_predictor() -> MLPredictor:
    """Get ML predictor instance."""
    return ml_predictor


async def get_business_rules() -> BusinessRulesEngine:
    """Get business rules engine instance."""
    return business_rules


async def get_feedback_system() -> FeedbackLearningSystem:
    """Get feedback learning system instance."""
    return feedback_system
