"""AI-powered support services for Support Agent."""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import openai
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger
import uuid

from shared_libs.database import get_database
from shared_libs.intelligence import get_nlp_processor, get_business_rules
from shared_libs.data_lake import get_data_lake

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY", "")


class AISupportService:
    """AI-powered customer support automation service."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.ticket_analysis_collection = "ticket_analysis"
        self.chat_sessions_collection = "chat_sessions"
        self.auto_responses_collection = "auto_responses"
        self.escalation_rules_collection = "escalation_rules"
    
    async def initialize(self):
        """Initialize the AI support service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.ticket_analysis_collection].create_index("ticket_id")
        await self.db[self.ticket_analysis_collection].create_index("sentiment_score")
        await self.db[self.ticket_analysis_collection].create_index("urgency_score")
        
        await self.db[self.chat_sessions_collection].create_index("session_id", unique=True)
        await self.db[self.chat_sessions_collection].create_index("customer_id")
        await self.db[self.chat_sessions_collection].create_index("created_at")
        
        await self.db[self.auto_responses_collection].create_index("response_id", unique=True)
        await self.db[self.auto_responses_collection].create_index("category")
        
        await self.db[self.escalation_rules_collection].create_index("rule_id", unique=True)
        await self.db[self.escalation_rules_collection].create_index("priority")
        
        # Create default escalation rules
        await self._create_default_escalation_rules()
        
        logger.info("AI Support service initialized")
    
    async def analyze_ticket_urgency(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ticket urgency using AI and business rules."""
        try:
            nlp = await get_nlp_processor()
            business_rules = await get_business_rules()
            data_lake = await get_data_lake()
            
            # Sentiment analysis
            description = ticket_data.get("description", "")
            subject = ticket_data.get("subject", "")
            combined_text = f"{subject} {description}"
            
            sentiment_analysis = await nlp.analyze_sentiment(combined_text)
            
            # Extract keywords for categorization
            keywords = await nlp.extract_keywords(combined_text, 10)
            
            # Urgency scoring
            urgency_score = 0.0
            urgency_factors = []
            
            # Sentiment-based urgency
            if sentiment_analysis.get("classification") == "negative":
                urgency_score += 0.3
                urgency_factors.append("Negative customer sentiment detected")
            
            # Keyword-based urgency
            urgent_keywords = ["urgent", "critical", "emergency", "down", "broken", "not working", "error", "bug"]
            urgent_keyword_count = sum(1 for keyword in keywords if keyword.lower() in urgent_keywords)
            if urgent_keyword_count > 0:
                urgency_score += min(0.4, urgent_keyword_count * 0.1)
                urgency_factors.append(f"Urgent keywords detected: {urgent_keyword_count}")
            
            # Customer tier impact
            customer_tier = ticket_data.get("customer_tier", "basic")
            if customer_tier == "enterprise":
                urgency_score += 0.2
                urgency_factors.append("Enterprise customer")
            elif customer_tier == "premium":
                urgency_score += 0.1
                urgency_factors.append("Premium customer")
            
            # Time-based urgency (business hours)
            current_hour = datetime.utcnow().hour
            if 9 <= current_hour <= 17:  # Business hours
                urgency_score += 0.1
                urgency_factors.append("Submitted during business hours")
            
            # Category-based urgency
            category = ticket_data.get("category", "general")
            if category in ["technical", "bug_report"]:
                urgency_score += 0.15
                urgency_factors.append("Technical issue category")
            elif category == "billing":
                urgency_score += 0.1
                urgency_factors.append("Billing issue category")
            
            # Cap urgency score at 1.0
            urgency_score = min(urgency_score, 1.0)
            
            # Determine priority and escalation
            if urgency_score >= 0.8:
                recommended_priority = "critical"
                escalate = True
                sla_hours = 1
            elif urgency_score >= 0.6:
                recommended_priority = "high"
                escalate = True
                sla_hours = 4
            elif urgency_score >= 0.4:
                recommended_priority = "medium"
                escalate = False
                sla_hours = 24
            else:
                recommended_priority = "low"
                escalate = False
                sla_hours = 72
            
            # Generate auto-response suggestions
            auto_response = await self._generate_auto_response(ticket_data, sentiment_analysis, keywords)
            
            analysis_result = {
                "analysis_id": f"TA_{str(uuid.uuid4())[:8].upper()}",
                "ticket_id": ticket_data.get("ticket_id", "unknown"),
                "urgency_score": round(urgency_score, 3),
                "urgency_factors": urgency_factors,
                "recommended_priority": recommended_priority,
                "escalate": escalate,
                "sla_hours": sla_hours,
                "sentiment_analysis": sentiment_analysis,
                "keywords": keywords,
                "auto_response": auto_response,
                "confidence": 0.85,
                "created_at": datetime.utcnow()
            }
            
            # Store analysis
            await self.db[self.ticket_analysis_collection].insert_one(analysis_result)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="support",
                event_type="ticket_analyzed",
                entity_type="ticket",
                entity_id=ticket_data.get("ticket_id", "unknown"),
                data={
                    "urgency_score": urgency_score,
                    "recommended_priority": recommended_priority,
                    "escalate": escalate
                }
            )
            
            logger.info(f"Ticket urgency analyzed: {ticket_data.get('ticket_id', 'unknown')}, urgency={urgency_score:.3f}, priority={recommended_priority}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze ticket urgency: {e}")
            return {"urgency_score": 0.5, "recommended_priority": "medium", "escalate": False}
    
    async def _generate_auto_response(self, ticket_data: Dict[str, Any], 
                                    sentiment_analysis: Dict[str, Any], 
                                    keywords: List[str]) -> Dict[str, Any]:
        """Generate automatic response suggestions."""
        try:
            category = ticket_data.get("category", "general")
            customer_name = ticket_data.get("customer_name", "Customer")
            
            # Template responses based on category and keywords
            response_templates = {
                "technical": {
                    "greeting": f"Dear {customer_name},\n\nThank you for contacting our technical support team.",
                    "acknowledgment": "We understand you're experiencing technical difficulties, and we're here to help.",
                    "next_steps": "Our technical team will investigate this issue and provide a solution within 4 hours.",
                    "closing": "Best regards,\nTechnical Support Team"
                },
                "billing": {
                    "greeting": f"Dear {customer_name},\n\nThank you for contacting our billing department.",
                    "acknowledgment": "We've received your billing inquiry and will review your account.",
                    "next_steps": "Our billing specialist will review your account and respond within 24 hours.",
                    "closing": "Best regards,\nBilling Team"
                },
                "general": {
                    "greeting": f"Dear {customer_name},\n\nThank you for contacting our support team.",
                    "acknowledgment": "We've received your inquiry and will address it promptly.",
                    "next_steps": "A support representative will respond to your request within 24 hours.",
                    "closing": "Best regards,\nCustomer Support Team"
                }
            }
            
            template = response_templates.get(category, response_templates["general"])
            
            # Adjust tone based on sentiment
            if sentiment_analysis.get("classification") == "negative":
                template["acknowledgment"] = "We sincerely apologize for any inconvenience you've experienced. " + template["acknowledgment"]
            
            # Generate full response
            auto_response_text = "\n\n".join([
                template["greeting"],
                template["acknowledgment"],
                template["next_steps"],
                template["closing"]
            ])
            
            # Suggest knowledge base articles
            suggested_articles = await self._suggest_knowledge_articles(keywords, category)
            
            return {
                "response_text": auto_response_text,
                "suggested_articles": suggested_articles,
                "tone": "empathetic" if sentiment_analysis.get("classification") == "negative" else "professional",
                "send_immediately": ticket_data.get("customer_tier") in ["premium", "enterprise"]
            }
            
        except Exception as e:
            logger.error(f"Failed to generate auto response: {e}")
            return {"response_text": "Thank you for contacting support. We will respond shortly."}
    
    async def _suggest_knowledge_articles(self, keywords: List[str], category: str) -> List[Dict[str, Any]]:
        """Suggest relevant knowledge base articles."""
        try:
            # Search for articles matching keywords and category
            query = {
                "$or": [
                    {"tags": {"$in": keywords}},
                    {"category": category},
                    {"title": {"$regex": "|".join(keywords), "$options": "i"}},
                    {"content": {"$regex": "|".join(keywords), "$options": "i"}}
                ],
                "status": "published"
            }
            
            articles = await self.db["knowledge_articles"].find(query).limit(3).to_list(None)
            
            return [
                {
                    "article_id": article.get("article_id"),
                    "title": article.get("title"),
                    "category": article.get("category"),
                    "relevance_score": 0.8  # Simplified relevance scoring
                } for article in articles
            ]
            
        except Exception as e:
            logger.error(f"Failed to suggest knowledge articles: {e}")
            return []
    
    async def handle_chat_message(self, session_id: str, message: str, customer_id: str) -> Dict[str, Any]:
        """Handle incoming chat message with AI response."""
        try:
            nlp = await get_nlp_processor()
            
            # Get or create chat session
            session = await self._get_or_create_chat_session(session_id, customer_id)
            
            # Analyze message intent
            intent_analysis = await self._analyze_message_intent(message)
            
            # Generate AI response
            ai_response = await self._generate_ai_chat_response(message, session, intent_analysis)
            
            # Store messages
            customer_message = {
                "message_id": str(uuid.uuid4()),
                "message": message,
                "sender": "customer",
                "timestamp": datetime.utcnow(),
                "intent": intent_analysis.get("intent", "general")
            }
            
            ai_message = {
                "message_id": str(uuid.uuid4()),
                "message": ai_response.get("response", "I'm here to help. Could you provide more details?"),
                "sender": "ai",
                "timestamp": datetime.utcnow(),
                "confidence": ai_response.get("confidence", 0.7)
            }
            
            # Update session
            await self.db[self.chat_sessions_collection].update_one(
                {"session_id": session_id},
                {
                    "$push": {
                        "messages": {"$each": [customer_message, ai_message]}
                    },
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
            # Check if escalation is needed
            escalation_needed = await self._check_chat_escalation(session, intent_analysis, ai_response)
            
            response_data = {
                "session_id": session_id,
                "ai_response": ai_response.get("response"),
                "confidence": ai_response.get("confidence", 0.7),
                "escalation_needed": escalation_needed,
                "suggested_actions": ai_response.get("suggested_actions", []),
                "intent": intent_analysis.get("intent", "general")
            }
            
            # Store event in data lake
            data_lake = await get_data_lake()
            await data_lake.store_event(
                agent="support",
                event_type="chat_message_handled",
                entity_type="chat_session",
                entity_id=session_id,
                data={
                    "intent": intent_analysis.get("intent"),
                    "confidence": ai_response.get("confidence"),
                    "escalation_needed": escalation_needed
                }
            )
            
            return response_data
            
        except Exception as e:
            logger.error(f"Failed to handle chat message: {e}")
            return {"ai_response": "I apologize, but I'm experiencing technical difficulties. Please try again."}
    
    async def _get_or_create_chat_session(self, session_id: str, customer_id: str) -> Dict[str, Any]:
        """Get existing chat session or create new one."""
        try:
            session = await self.db[self.chat_sessions_collection].find_one({"session_id": session_id})
            
            if not session:
                # Create new session
                session = {
                    "session_id": session_id,
                    "customer_id": customer_id,
                    "customer_email": f"customer_{customer_id}@example.com",  # Would get from customer DB
                    "messages": [],
                    "status": "active",
                    "ai_handled": True,
                    "escalated_to_human": False,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                await self.db[self.chat_sessions_collection].insert_one(session)
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to get/create chat session: {e}")
            return {}
    
    async def _analyze_message_intent(self, message: str) -> Dict[str, Any]:
        """Analyze customer message intent."""
        try:
            message_lower = message.lower()
            
            # Intent classification based on keywords
            intent_patterns = {
                "billing_inquiry": ["bill", "billing", "charge", "payment", "invoice", "refund"],
                "technical_support": ["error", "bug", "not working", "broken", "issue", "problem"],
                "account_help": ["account", "login", "password", "access", "profile"],
                "feature_request": ["feature", "request", "suggestion", "improve", "add"],
                "complaint": ["complaint", "unhappy", "disappointed", "terrible", "awful"],
                "compliment": ["great", "excellent", "amazing", "love", "fantastic"],
                "cancellation": ["cancel", "unsubscribe", "terminate", "end", "stop"],
                "general_inquiry": ["help", "question", "how", "what", "when", "where"]
            }
            
            intent_scores = {}
            for intent, keywords in intent_patterns.items():
                score = sum(1 for keyword in keywords if keyword in message_lower)
                if score > 0:
                    intent_scores[intent] = score
            
            # Determine primary intent
            if intent_scores:
                primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
                confidence = min(intent_scores[primary_intent] / 3, 1.0)  # Normalize confidence
            else:
                primary_intent = "general_inquiry"
                confidence = 0.5
            
            return {
                "intent": primary_intent,
                "confidence": confidence,
                "all_intents": intent_scores
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze message intent: {e}")
            return {"intent": "general_inquiry", "confidence": 0.5}
    
    async def _generate_ai_chat_response(self, message: str, session: Dict[str, Any], 
                                       intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI chat response."""
        try:
            intent = intent_analysis.get("intent", "general_inquiry")
            
            # Response templates based on intent
            response_templates = {
                "billing_inquiry": {
                    "response": "I can help you with billing questions. Let me look up your account information. Could you please provide your account email or customer ID?",
                    "suggested_actions": ["lookup_account", "transfer_to_billing"],
                    "confidence": 0.8
                },
                "technical_support": {
                    "response": "I understand you're experiencing a technical issue. To better assist you, could you please describe what you were trying to do when the problem occurred?",
                    "suggested_actions": ["gather_technical_details", "check_system_status"],
                    "confidence": 0.8
                },
                "account_help": {
                    "response": "I can help you with account-related questions. Are you having trouble logging in, or do you need help with account settings?",
                    "suggested_actions": ["password_reset", "account_verification"],
                    "confidence": 0.9
                },
                "complaint": {
                    "response": "I sincerely apologize for any inconvenience you've experienced. Your feedback is important to us. Could you please tell me more about the issue so I can help resolve it?",
                    "suggested_actions": ["escalate_to_manager", "gather_complaint_details"],
                    "confidence": 0.7
                },
                "cancellation": {
                    "response": "I'm sorry to hear you're considering canceling. Before we proceed, I'd like to understand if there's anything we can do to address your concerns. What's prompting this decision?",
                    "suggested_actions": ["retention_offer", "escalate_to_retention_team"],
                    "confidence": 0.8
                }
            }
            
            template = response_templates.get(intent, {
                "response": "Thank you for your message. I'm here to help! Could you please provide more details about what you need assistance with?",
                "suggested_actions": ["gather_more_info"],
                "confidence": 0.6
            })
            
            # Personalize response if we have session history
            messages = session.get("messages", [])
            if len(messages) > 2:  # Ongoing conversation
                template["response"] = "I see we've been chatting. " + template["response"]
            
            return template
            
        except Exception as e:
            logger.error(f"Failed to generate AI chat response: {e}")
            return {
                "response": "I apologize, but I'm having trouble processing your request. A human agent will assist you shortly.",
                "suggested_actions": ["escalate_to_human"],
                "confidence": 0.3
            }
    
    async def _check_chat_escalation(self, session: Dict[str, Any], 
                                   intent_analysis: Dict[str, Any], 
                                   ai_response: Dict[str, Any]) -> bool:
        """Check if chat should be escalated to human agent."""
        try:
            # Escalation triggers
            escalate = False
            
            # Low confidence responses
            if ai_response.get("confidence", 1.0) < 0.5:
                escalate = True
            
            # Specific intents that require human intervention
            human_required_intents = ["complaint", "cancellation", "complex_technical"]
            if intent_analysis.get("intent") in human_required_intents:
                escalate = True
            
            # Long conversation without resolution
            messages = session.get("messages", [])
            if len(messages) > 10:  # More than 5 exchanges
                escalate = True
            
            # Customer explicitly asks for human
            last_customer_message = None
            for msg in reversed(messages):
                if msg.get("sender") == "customer":
                    last_customer_message = msg.get("message", "").lower()
                    break
            
            if last_customer_message:
                human_keywords = ["human", "agent", "person", "representative", "manager"]
                if any(keyword in last_customer_message for keyword in human_keywords):
                    escalate = True
            
            return escalate
            
        except Exception as e:
            logger.error(f"Failed to check chat escalation: {e}")
            return False
    
    async def _create_default_escalation_rules(self):
        """Create default escalation rules."""
        try:
            default_rules = [
                {
                    "rule_id": "ESC_001",
                    "name": "Enterprise Customer Priority",
                    "description": "Escalate all enterprise customer tickets immediately",
                    "conditions": {"customer_tier": "enterprise"},
                    "action": "immediate_escalation",
                    "priority": 10,
                    "active": True
                },
                {
                    "rule_id": "ESC_002", 
                    "name": "Critical Priority Escalation",
                    "description": "Escalate critical priority tickets within 1 hour",
                    "conditions": {"priority": "critical"},
                    "action": "escalate_within_1_hour",
                    "priority": 9,
                    "active": True
                },
                {
                    "rule_id": "ESC_003",
                    "name": "Negative Sentiment Escalation",
                    "description": "Escalate tickets with very negative sentiment",
                    "conditions": {"sentiment_score": {"$lt": -0.5}},
                    "action": "escalate_to_senior_agent",
                    "priority": 7,
                    "active": True
                }
            ]
            
            for rule in default_rules:
                existing = await self.db[self.escalation_rules_collection].find_one({"rule_id": rule["rule_id"]})
                if not existing:
                    rule["created_at"] = datetime.utcnow()
                    await self.db[self.escalation_rules_collection].insert_one(rule)
            
        except Exception as e:
            logger.error(f"Failed to create default escalation rules: {e}")
    
    async def predict_ticket_resolution_time(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict ticket resolution time using historical data."""
        try:
            category = ticket_data.get("category", "general")
            priority = ticket_data.get("priority", "medium")
            customer_tier = ticket_data.get("customer_tier", "basic")
            
            # Get historical resolution times for similar tickets
            historical_query = {
                "category": category,
                "priority": priority,
                "status": "resolved",
                "resolution_time": {"$exists": True}
            }
            
            historical_tickets = await self.db["tickets"].find(historical_query).limit(100).to_list(None)
            
            if historical_tickets:
                resolution_times = [ticket.get("resolution_time", 0) for ticket in historical_tickets]
                avg_resolution_time = sum(resolution_times) / len(resolution_times)
                
                # Adjust for customer tier
                tier_multipliers = {"basic": 1.0, "premium": 0.8, "enterprise": 0.6}
                adjusted_time = avg_resolution_time * tier_multipliers.get(customer_tier, 1.0)
            else:
                # Default estimates if no historical data
                default_times = {
                    "low": 48,
                    "medium": 24,
                    "high": 8,
                    "critical": 2
                }
                adjusted_time = default_times.get(priority, 24)
            
            # Convert to hours and determine SLA
            predicted_hours = max(1, int(adjusted_time))
            sla_met_probability = 0.85 if predicted_hours <= 24 else 0.65
            
            prediction = {
                "predicted_resolution_hours": predicted_hours,
                "sla_met_probability": sla_met_probability,
                "confidence": 0.75 if historical_tickets else 0.5,
                "based_on_tickets": len(historical_tickets),
                "factors": {
                    "category": category,
                    "priority": priority,
                    "customer_tier": customer_tier
                }
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to predict ticket resolution time: {e}")
            return {"predicted_resolution_hours": 24, "confidence": 0.5}