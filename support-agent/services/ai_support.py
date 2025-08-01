"""AI-powered customer support services."""

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
from collections import defaultdict, Counter

from shared_libs.database import get_database
from shared_libs.intelligence import get_nlp_processor, get_business_rules
from shared_libs.data_lake import get_data_lake

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY", "")


class AISupportService:
    """AI-powered customer support and automation service."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.ticket_routing_collection = "ticket_routing"
        self.chatbot_conversations_collection = "chatbot_conversations"
        self.knowledge_base_collection = "knowledge_base"
        self.escalation_rules_collection = "escalation_rules"
        self.support_analytics_collection = "support_analytics"
    
    async def initialize(self):
        """Initialize the AI support service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.ticket_routing_collection].create_index("ticket_id")
        await self.db[self.ticket_routing_collection].create_index("routing_score")
        await self.db[self.ticket_routing_collection].create_index("created_at")
        
        await self.db[self.chatbot_conversations_collection].create_index("conversation_id", unique=True)
        await self.db[self.chatbot_conversations_collection].create_index("customer_id")
        await self.db[self.chatbot_conversations_collection].create_index("created_at")
        
        await self.db[self.knowledge_base_collection].create_index("article_id", unique=True)
        await self.db[self.knowledge_base_collection].create_index("category")
        await self.db[self.knowledge_base_collection].create_index("keywords")
        
        await self.db[self.escalation_rules_collection].create_index("rule_id", unique=True)
        await self.db[self.escalation_rules_collection].create_index("priority")
        
        await self.db[self.support_analytics_collection].create_index("analysis_id", unique=True)
        await self.db[self.support_analytics_collection].create_index("created_at")
        
        logger.info("AI Support service initialized")
    
    async def smart_ticket_routing(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently route support tickets using AI analysis."""
        try:
            data_lake = await get_data_lake()
            nlp = await get_nlp_processor()
            business_rules = await get_business_rules()
            
            # Analyze ticket content
            ticket_analysis = await self._analyze_ticket_content(ticket_data)
            
            # Determine urgency and priority
            urgency_analysis = await self._analyze_ticket_urgency(ticket_data, ticket_analysis)
            
            # Find best agent match
            agent_matching = await self._find_best_agent_match(ticket_data, ticket_analysis)
            
            # Apply business rules
            routing_rules = await business_rules.evaluate_support_escalation(ticket_data)
            
            # Calculate routing confidence
            routing_confidence = self._calculate_routing_confidence(
                ticket_analysis, urgency_analysis, agent_matching
            )
            
            # Generate routing decision
            routing_decision = {
                "routing_id": f"RT_{str(uuid.uuid4())[:8].upper()}",
                "ticket_id": ticket_data.get("ticket_id"),
                "ticket_analysis": ticket_analysis,
                "urgency_analysis": urgency_analysis,
                "agent_matching": agent_matching,
                "routing_rules": routing_rules,
                "recommended_agent": agent_matching.get("best_match", {}).get("agent_id"),
                "recommended_team": agent_matching.get("best_team"),
                "priority_level": urgency_analysis.get("priority_level"),
                "estimated_resolution_time": urgency_analysis.get("estimated_resolution_hours"),
                "routing_confidence": routing_confidence,
                "auto_assign": routing_confidence > 0.8,
                "routing_reason": self._generate_routing_reason(ticket_analysis, agent_matching),
                "created_at": datetime.utcnow()
            }
            
            # Store routing decision
            await self.db[self.ticket_routing_collection].insert_one(routing_decision)
            
            # Auto-assign if confidence is high
            if routing_decision["auto_assign"]:
                await self._auto_assign_ticket(ticket_data["ticket_id"], routing_decision)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="support",
                event_type="ticket_routed",
                entity_type="ticket",
                entity_id=ticket_data.get("ticket_id"),
                data={
                    "routing_id": routing_decision["routing_id"],
                    "recommended_agent": routing_decision["recommended_agent"],
                    "priority_level": routing_decision["priority_level"],
                    "routing_confidence": routing_decision["routing_confidence"],
                    "auto_assigned": routing_decision["auto_assign"]
                }
            )
            
            logger.info(f"Ticket routed: {ticket_data.get('ticket_id')} -> {routing_decision['recommended_agent']} (confidence: {routing_confidence:.2f})")
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"Failed to route ticket: {e}")
            return {}
    
    async def _analyze_ticket_content(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ticket content using NLP."""
        try:
            nlp = await get_nlp_processor()
            
            subject = ticket_data.get("subject", "")
            description = ticket_data.get("description", "")
            combined_text = f"{subject} {description}"
            
            # Sentiment analysis
            sentiment = await nlp.analyze_sentiment(combined_text)
            
            # Extract keywords
            keywords = await nlp.extract_keywords(combined_text, 10)
            
            # Categorize ticket
            categories = [
                "technical_issue", "billing_inquiry", "account_access", "feature_request",
                "bug_report", "general_inquiry", "complaint", "compliment"
            ]
            category_scores = await nlp.classify_text(combined_text, categories)
            predicted_category = max(category_scores.items(), key=lambda x: x[1])[0]
            
            # Detect issue type
            issue_type = await self._detect_issue_type(combined_text)
            
            # Extract entities (product names, error codes, etc.)
            entities = await self._extract_entities(combined_text)
            
            return {
                "sentiment": sentiment,
                "keywords": keywords,
                "predicted_category": predicted_category,
                "category_confidence": category_scores.get(predicted_category, 0),
                "issue_type": issue_type,
                "entities": entities,
                "text_complexity": len(combined_text.split()),
                "language_detected": "en"  # Could implement language detection
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze ticket content: {e}")
            return {}
    
    async def _detect_issue_type(self, text: str) -> str:
        """Detect specific issue type from ticket text."""
        try:
            text_lower = text.lower()
            
            # Define issue patterns
            issue_patterns = {
                "login_issue": ["login", "sign in", "password", "authentication", "access denied"],
                "payment_issue": ["payment", "billing", "charge", "refund", "invoice", "credit card"],
                "performance_issue": ["slow", "loading", "timeout", "performance", "lag", "freeze"],
                "bug_report": ["error", "bug", "broken", "not working", "crash", "exception"],
                "feature_request": ["feature", "enhancement", "suggestion", "improvement", "add"],
                "integration_issue": ["api", "integration", "webhook", "sync", "connection"],
                "data_issue": ["data", "missing", "incorrect", "sync", "export", "import"]
            }
            
            # Score each issue type
            issue_scores = {}
            for issue_type, keywords in issue_patterns.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    issue_scores[issue_type] = score
            
            # Return highest scoring issue type
            if issue_scores:
                return max(issue_scores.items(), key=lambda x: x[1])[0]
            
            return "general_inquiry"
            
        except Exception as e:
            logger.error(f"Failed to detect issue type: {e}")
            return "general_inquiry"
    
    async def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from ticket text."""
        try:
            entities = {
                "error_codes": [],
                "product_names": [],
                "urls": [],
                "email_addresses": [],
                "phone_numbers": []
            }
            
            # Extract error codes (pattern: ERROR_123, ERR-456, etc.)
            error_codes = re.findall(r'\b(?:ERROR|ERR)[-_]?\d+\b', text, re.IGNORECASE)
            entities["error_codes"] = error_codes
            
            # Extract URLs
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            entities["urls"] = urls
            
            # Extract email addresses
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            entities["email_addresses"] = emails
            
            # Extract phone numbers (simple pattern)
            phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
            entities["phone_numbers"] = phones
            
            # Extract potential product names (capitalized words)
            product_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            entities["product_names"] = list(set(product_names))[:5]  # Limit to 5
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return {}
    
    async def _analyze_ticket_urgency(self, ticket_data: Dict[str, Any], 
                                    ticket_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ticket urgency and priority."""
        try:
            urgency_score = 0
            urgency_factors = []
            
            # Customer tier impact
            customer_tier = ticket_data.get("customer_tier", "standard")
            if customer_tier == "enterprise":
                urgency_score += 3
                urgency_factors.append("Enterprise customer")
            elif customer_tier == "premium":
                urgency_score += 2
                urgency_factors.append("Premium customer")
            
            # Sentiment impact
            sentiment = ticket_analysis.get("sentiment", {})
            if sentiment.get("classification") == "negative":
                urgency_score += 2
                urgency_factors.append("Negative sentiment")
            
            # Issue type impact
            issue_type = ticket_analysis.get("issue_type", "")
            high_urgency_issues = ["login_issue", "payment_issue", "performance_issue", "bug_report"]
            if issue_type in high_urgency_issues:
                urgency_score += 2
                urgency_factors.append(f"High-impact issue: {issue_type}")
            
            # Keywords impact
            keywords = ticket_analysis.get("keywords", [])
            urgent_keywords = ["urgent", "critical", "emergency", "down", "broken", "asap"]
            urgent_keyword_count = sum(1 for keyword in keywords if keyword.lower() in urgent_keywords)
            if urgent_keyword_count > 0:
                urgency_score += urgent_keyword_count
                urgency_factors.append(f"Urgent keywords detected: {urgent_keyword_count}")
            
            # Business hours impact
            current_hour = datetime.utcnow().hour
            if 9 <= current_hour <= 17:  # Business hours
                urgency_score += 1
                urgency_factors.append("Submitted during business hours")
            
            # Determine priority level
            if urgency_score >= 7:
                priority_level = "critical"
                estimated_resolution_hours = 2
            elif urgency_score >= 5:
                priority_level = "high"
                estimated_resolution_hours = 8
            elif urgency_score >= 3:
                priority_level = "medium"
                estimated_resolution_hours = 24
            else:
                priority_level = "low"
                estimated_resolution_hours = 72
            
            return {
                "urgency_score": urgency_score,
                "urgency_factors": urgency_factors,
                "priority_level": priority_level,
                "estimated_resolution_hours": estimated_resolution_hours,
                "sla_deadline": datetime.utcnow() + timedelta(hours=estimated_resolution_hours)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze ticket urgency: {e}")
            return {"priority_level": "medium", "estimated_resolution_hours": 24}
    
    async def _find_best_agent_match(self, ticket_data: Dict[str, Any], 
                                   ticket_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best agent match for the ticket."""
        try:
            # Get available agents
            agents = await self.db["support_agents"].find({"status": "available"}).to_list(None)
            
            if not agents:
                return {"best_match": None, "best_team": "general_support"}
            
            agent_scores = []
            
            for agent in agents:
                score = 0
                
                # Skill matching
                agent_skills = agent.get("skills", [])
                ticket_category = ticket_analysis.get("predicted_category", "")
                issue_type = ticket_analysis.get("issue_type", "")
                
                if ticket_category in agent_skills:
                    score += 3
                if issue_type in agent_skills:
                    score += 2
                
                # Experience matching
                agent_experience = agent.get("experience_years", 0)
                if ticket_analysis.get("text_complexity", 0) > 100:  # Complex ticket
                    score += min(agent_experience, 5)  # Cap at 5 points
                
                # Workload consideration
                current_tickets = agent.get("current_ticket_count", 0)
                max_tickets = agent.get("max_concurrent_tickets", 10)
                workload_factor = 1 - (current_tickets / max_tickets)
                score *= workload_factor
                
                # Language matching
                ticket_language = ticket_analysis.get("language_detected", "en")
                agent_languages = agent.get("languages", ["en"])
                if ticket_language in agent_languages:
                    score += 1
                
                agent_scores.append({
                    "agent_id": agent.get("agent_id"),
                    "agent_name": agent.get("name"),
                    "score": score,
                    "skills": agent_skills,
                    "current_workload": current_tickets
                })
            
            # Sort by score
            agent_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # Determine best team
            category = ticket_analysis.get("predicted_category", "")
            team_mapping = {
                "technical_issue": "technical_support",
                "billing_inquiry": "billing_support",
                "account_access": "account_support",
                "bug_report": "technical_support",
                "feature_request": "product_support"
            }
            best_team = team_mapping.get(category, "general_support")
            
            return {
                "best_match": agent_scores[0] if agent_scores else None,
                "all_matches": agent_scores[:5],  # Top 5 matches
                "best_team": best_team,
                "matching_confidence": agent_scores[0]["score"] / 10 if agent_scores else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to find best agent match: {e}")
            return {"best_match": None, "best_team": "general_support"}
    
    def _calculate_routing_confidence(self, ticket_analysis: Dict[str, Any], 
                                    urgency_analysis: Dict[str, Any], 
                                    agent_matching: Dict[str, Any]) -> float:
        """Calculate confidence score for routing decision."""
        try:
            confidence = 0.5  # Base confidence
            
            # Category confidence
            category_confidence = ticket_analysis.get("category_confidence", 0)
            confidence += category_confidence * 0.3
            
            # Agent matching confidence
            matching_confidence = agent_matching.get("matching_confidence", 0)
            confidence += matching_confidence * 0.3
            
            # Urgency clarity
            urgency_factors = urgency_analysis.get("urgency_factors", [])
            if len(urgency_factors) >= 2:
                confidence += 0.2
            elif len(urgency_factors) >= 1:
                confidence += 0.1
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate routing confidence: {e}")
            return 0.5
    
    def _generate_routing_reason(self, ticket_analysis: Dict[str, Any], 
                               agent_matching: Dict[str, Any]) -> str:
        """Generate human-readable routing reason."""
        try:
            category = ticket_analysis.get("predicted_category", "general")
            issue_type = ticket_analysis.get("issue_type", "inquiry")
            best_match = agent_matching.get("best_match", {})
            
            if best_match:
                agent_name = best_match.get("agent_name", "Agent")
                return f"Routed to {agent_name} based on {category} expertise and {issue_type} specialization"
            else:
                team = agent_matching.get("best_team", "support")
                return f"Routed to {team} team based on {category} category"
            
        except Exception as e:
            logger.error(f"Failed to generate routing reason: {e}")
            return "Routed based on ticket analysis"
    
    async def _auto_assign_ticket(self, ticket_id: str, routing_decision: Dict[str, Any]) -> None:
        """Auto-assign ticket to recommended agent."""
        try:
            agent_id = routing_decision.get("recommended_agent")
            if not agent_id:
                return
            
            # Update ticket assignment
            await self.db["tickets"].update_one(
                {"ticket_id": ticket_id},
                {
                    "$set": {
                        "assigned_agent": agent_id,
                        "assigned_at": datetime.utcnow(),
                        "status": "assigned",
                        "priority": routing_decision.get("priority_level"),
                        "sla_deadline": routing_decision.get("urgency_analysis", {}).get("sla_deadline")
                    }
                }
            )
            
            # Update agent workload
            await self.db["support_agents"].update_one(
                {"agent_id": agent_id},
                {"$inc": {"current_ticket_count": 1}}
            )
            
            logger.info(f"Ticket {ticket_id} auto-assigned to agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to auto-assign ticket: {e}")
    
    async def ai_chatbot_conversation(self, customer_id: str, message: str, 
                                    conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle AI chatbot conversation with customers."""
        try:
            data_lake = await get_data_lake()
            
            # Get or create conversation
            if not conversation_id:
                conversation_id = f"CHAT_{str(uuid.uuid4())[:8].upper()}"
                conversation = {
                    "conversation_id": conversation_id,
                    "customer_id": customer_id,
                    "messages": [],
                    "status": "active",
                    "created_at": datetime.utcnow(),
                    "last_activity": datetime.utcnow()
                }
                await self.db[self.chatbot_conversations_collection].insert_one(conversation)
            else:
                conversation = await self.db[self.chatbot_conversations_collection].find_one(
                    {"conversation_id": conversation_id}
                )
            
            # Analyze customer message
            message_analysis = await self._analyze_customer_message(message, conversation)
            
            # Generate AI response
            ai_response = await self._generate_ai_response(message, message_analysis, conversation)
            
            # Check if escalation is needed
            escalation_check = await self._check_escalation_needed(message_analysis, conversation)
            
            # Update conversation
            new_message = {
                "timestamp": datetime.utcnow(),
                "sender": "customer",
                "message": message,
                "analysis": message_analysis
            }
            
            ai_message = {
                "timestamp": datetime.utcnow(),
                "sender": "ai",
                "message": ai_response.get("response", "I'm here to help!"),
                "confidence": ai_response.get("confidence", 0.5),
                "suggested_actions": ai_response.get("suggested_actions", [])
            }
            
            await self.db[self.chatbot_conversations_collection].update_one(
                {"conversation_id": conversation_id},
                {
                    "$push": {"messages": {"$each": [new_message, ai_message]}},
                    "$set": {"last_activity": datetime.utcnow()}
                }
            )
            
            # Handle escalation if needed
            if escalation_check.get("should_escalate", False):
                await self._escalate_conversation_to_human(conversation_id, escalation_check)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="support",
                event_type="chatbot_interaction",
                entity_type="conversation",
                entity_id=conversation_id,
                data={
                    "customer_id": customer_id,
                    "message_sentiment": message_analysis.get("sentiment", {}).get("classification"),
                    "ai_confidence": ai_response.get("confidence", 0.5),
                    "escalation_needed": escalation_check.get("should_escalate", False)
                }
            )
            
            response_data = {
                "conversation_id": conversation_id,
                "ai_response": ai_response.get("response"),
                "confidence": ai_response.get("confidence"),
                "suggested_actions": ai_response.get("suggested_actions", []),
                "escalation_needed": escalation_check.get("should_escalate", False),
                "escalation_reason": escalation_check.get("reason"),
                "conversation_status": "escalated" if escalation_check.get("should_escalate") else "active"
            }
            
            logger.info(f"AI chatbot response generated for conversation {conversation_id}")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Failed to handle chatbot conversation: {e}")
            return {"ai_response": "I apologize, but I'm experiencing technical difficulties. Please try again."}
    
    async def _analyze_customer_message(self, message: str, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze customer message for intent and sentiment."""
        try:
            nlp = await get_nlp_processor()
            
            # Sentiment analysis
            sentiment = await nlp.analyze_sentiment(message)
            
            # Intent detection
            intent = await self._detect_customer_intent(message)
            
            # Extract entities
            entities = await self._extract_entities(message)
            
            # Analyze conversation context
            previous_messages = conversation.get("messages", [])
            context_analysis = await self._analyze_conversation_context(previous_messages)
            
            return {
                "sentiment": sentiment,
                "intent": intent,
                "entities": entities,
                "context": context_analysis,
                "message_length": len(message),
                "urgency_indicators": self._detect_urgency_indicators(message)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze customer message: {e}")
            return {}
    
    async def _detect_customer_intent(self, message: str) -> Dict[str, Any]:
        """Detect customer intent from message."""
        try:
            message_lower = message.lower()
            
            # Define intent patterns
            intent_patterns = {
                "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
                "question": ["what", "how", "when", "where", "why", "can you", "?"],
                "complaint": ["problem", "issue", "wrong", "error", "broken", "not working"],
                "request": ["need", "want", "please", "help", "assist", "support"],
                "compliment": ["thank", "great", "excellent", "good job", "appreciate"],
                "goodbye": ["bye", "goodbye", "thanks", "that's all", "end chat"],
                "escalation": ["manager", "supervisor", "human", "person", "speak to someone"]
            }
            
            # Score each intent
            intent_scores = {}
            for intent, keywords in intent_patterns.items():
                score = sum(1 for keyword in keywords if keyword in message_lower)
                if score > 0:
                    intent_scores[intent] = score
            
            # Determine primary intent
            if intent_scores:
                primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
                confidence = intent_scores[primary_intent] / len(message.split())
            else:
                primary_intent = "general"
                confidence = 0.5
            
            return {
                "primary_intent": primary_intent,
                "confidence": min(confidence, 1.0),
                "all_intents": intent_scores
            }
            
        except Exception as e:
            logger.error(f"Failed to detect customer intent: {e}")
            return {"primary_intent": "general", "confidence": 0.5}
    
    async def _analyze_conversation_context(self, previous_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation context from previous messages."""
        try:
            if not previous_messages:
                return {"context_type": "new_conversation", "topics": []}
            
            # Extract topics from previous messages
            topics = []
            sentiment_trend = []
            
            for msg in previous_messages[-5:]:  # Last 5 messages
                if msg.get("sender") == "customer":
                    message_text = msg.get("message", "")
                    # Simple topic extraction (could be enhanced with NLP)
                    words = message_text.lower().split()
                    topics.extend([word for word in words if len(word) > 4])
                    
                    # Sentiment trend
                    analysis = msg.get("analysis", {})
                    sentiment = analysis.get("sentiment", {})
                    if sentiment.get("polarity"):
                        sentiment_trend.append(sentiment["polarity"])
            
            # Determine context type
            if len(previous_messages) == 0:
                context_type = "new_conversation"
            elif len(previous_messages) < 4:
                context_type = "early_conversation"
            else:
                context_type = "ongoing_conversation"
            
            # Calculate sentiment trend
            if sentiment_trend:
                avg_sentiment = sum(sentiment_trend) / len(sentiment_trend)
                if avg_sentiment < -0.2:
                    sentiment_direction = "declining"
                elif avg_sentiment > 0.2:
                    sentiment_direction = "improving"
                else:
                    sentiment_direction = "stable"
            else:
                sentiment_direction = "unknown"
            
            return {
                "context_type": context_type,
                "topics": list(set(topics))[:10],  # Top 10 unique topics
                "message_count": len(previous_messages),
                "sentiment_direction": sentiment_direction,
                "conversation_duration": self._calculate_conversation_duration(previous_messages)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze conversation context: {e}")
            return {"context_type": "unknown", "topics": []}
    
    def _calculate_conversation_duration(self, messages: List[Dict[str, Any]]) -> int:
        """Calculate conversation duration in minutes."""
        try:
            if len(messages) < 2:
                return 0
            
            first_message = messages[0].get("timestamp")
            last_message = messages[-1].get("timestamp")
            
            if first_message and last_message:
                duration = (last_message - first_message).total_seconds() / 60
                return int(duration)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to calculate conversation duration: {e}")
            return 0
    
    def _detect_urgency_indicators(self, message: str) -> List[str]:
        """Detect urgency indicators in customer message."""
        urgency_keywords = [
            "urgent", "asap", "immediately", "emergency", "critical", "broken",
            "down", "not working", "can't access", "lost", "stuck", "help"
        ]
        
        message_lower = message.lower()
        detected_indicators = [keyword for keyword in urgency_keywords if keyword in message_lower]
        
        return detected_indicators
    
    async def _generate_ai_response(self, message: str, message_analysis: Dict[str, Any], 
                                  conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI response to customer message."""
        try:
            intent = message_analysis.get("intent", {}).get("primary_intent", "general")
            sentiment = message_analysis.get("sentiment", {})
            context = message_analysis.get("context", {})
            
            # Check knowledge base for relevant articles
            kb_results = await self._search_knowledge_base(message, intent)
            
            # Generate response based on intent
            if intent == "greeting":
                response = "Hello! I'm here to help you. What can I assist you with today?"
                confidence = 0.9
            elif intent == "question":
                if kb_results:
                    response = f"Based on our knowledge base: {kb_results[0].get('answer', 'Let me help you with that.')}"
                    confidence = 0.8
                else:
                    response = await self._generate_contextual_response(message, context)
                    confidence = 0.6
            elif intent == "complaint":
                response = "I understand your frustration, and I'm here to help resolve this issue. Can you provide more details about what's happening?"
                confidence = 0.7
            elif intent == "compliment":
                response = "Thank you for your kind words! I'm glad I could help. Is there anything else I can assist you with?"
                confidence = 0.9
            elif intent == "goodbye":
                response = "Thank you for contacting us! If you need any further assistance, please don't hesitate to reach out. Have a great day!"
                confidence = 0.9
            elif intent == "escalation":
                response = "I'll connect you with a human agent right away. Please hold on while I transfer your conversation."
                confidence = 0.9
            else:
                response = await self._generate_contextual_response(message, context)
                confidence = 0.5
            
            # Generate suggested actions
            suggested_actions = await self._generate_suggested_actions(intent, message_analysis)
            
            return {
                "response": response,
                "confidence": confidence,
                "suggested_actions": suggested_actions,
                "knowledge_base_used": len(kb_results) > 0,
                "response_type": intent
            }
            
        except Exception as e:
            logger.error(f"Failed to generate AI response: {e}")
            return {
                "response": "I apologize, but I'm having trouble processing your request. Could you please rephrase your question?",
                "confidence": 0.3
            }
    
    async def _search_knowledge_base(self, message: str, intent: str) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant articles."""
        try:
            nlp = await get_nlp_processor()
            
            # Extract keywords from message
            keywords = await nlp.extract_keywords(message, 5)
            
            # Search knowledge base
            search_query = {
                "$or": [
                    {"keywords": {"$in": keywords}},
                    {"title": {"$regex": "|".join(keywords), "$options": "i"}},
                    {"content": {"$regex": "|".join(keywords), "$options": "i"}}
                ]
            }
            
            articles = await self.db[self.knowledge_base_collection].find(search_query).limit(3).to_list(None)
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to search knowledge base: {e}")
            return []
    
    async def _generate_contextual_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate contextual response using AI."""
        try:
            if not openai.api_key:
                return "I understand your question. Let me help you find the right information."
            
            context_info = f"Conversation context: {context.get('context_type', 'new')}, Topics discussed: {', '.join(context.get('topics', [])[:3])}"
            
            prompt = f"""
            You are a helpful customer support AI assistant. Respond to this customer message in a friendly, professional manner.
            
            Customer message: "{message}"
            Context: {context_info}
            
            Provide a helpful, concise response (max 100 words):
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate contextual response: {e}")
            return "I'm here to help! Could you provide a bit more detail about what you need assistance with?"
    
    async def _generate_suggested_actions(self, intent: str, message_analysis: Dict[str, Any]) -> List[str]:
        """Generate suggested actions based on intent and analysis."""
        try:
            actions = []
            
            if intent == "complaint":
                actions.extend([
                    "Create support ticket",
                    "Escalate to human agent",
                    "Check system status"
                ])
            elif intent == "question":
                actions.extend([
                    "Search knowledge base",
                    "Provide tutorial links",
                    "Schedule callback"
                ])
            elif intent == "request":
                actions.extend([
                    "Process request",
                    "Check account permissions",
                    "Provide status update"
                ])
            elif intent == "escalation":
                actions.extend([
                    "Transfer to human agent",
                    "Create priority ticket",
                    "Schedule manager callback"
                ])
            
            # Add urgency-based actions
            urgency_indicators = message_analysis.get("urgency_indicators", [])
            if urgency_indicators:
                actions.extend([
                    "Mark as urgent",
                    "Immediate escalation",
                    "Priority handling"
                ])
            
            return actions[:5]  # Limit to 5 actions
            
        except Exception as e:
            logger.error(f"Failed to generate suggested actions: {e}")
            return []
    
    async def _check_escalation_needed(self, message_analysis: Dict[str, Any], 
                                     conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Check if conversation needs escalation to human agent."""
        try:
            should_escalate = False
            escalation_reasons = []
            
            # Intent-based escalation
            intent = message_analysis.get("intent", {}).get("primary_intent")
            if intent == "escalation":
                should_escalate = True
                escalation_reasons.append("Customer requested human agent")
            
            # Sentiment-based escalation
            sentiment = message_analysis.get("sentiment", {})
            if sentiment.get("classification") == "negative" and sentiment.get("polarity", 0) < -0.5:
                should_escalate = True
                escalation_reasons.append("Highly negative sentiment detected")
            
            # Conversation length escalation
            message_count = len(conversation.get("messages", []))
            if message_count > 10:  # More than 10 messages
                should_escalate = True
                escalation_reasons.append("Long conversation without resolution")
            
            # Urgency escalation
            urgency_indicators = message_analysis.get("urgency_indicators", [])
            if len(urgency_indicators) >= 2:
                should_escalate = True
                escalation_reasons.append("Multiple urgency indicators detected")
            
            # Complex issue escalation
            entities = message_analysis.get("entities", {})
            if entities.get("error_codes") or len(entities.get("product_names", [])) > 2:
                should_escalate = True
                escalation_reasons.append("Complex technical issue detected")
            
            return {
                "should_escalate": should_escalate,
                "reasons": escalation_reasons,
                "escalation_priority": "high" if len(escalation_reasons) > 2 else "medium"
            }
            
        except Exception as e:
            logger.error(f"Failed to check escalation: {e}")
            return {"should_escalate": False, "reasons": []}
    
    async def _escalate_conversation_to_human(self, conversation_id: str, 
                                           escalation_check: Dict[str, Any]) -> None:
        """Escalate conversation to human agent."""
        try:
            # Update conversation status
            await self.db[self.chatbot_conversations_collection].update_one(
                {"conversation_id": conversation_id},
                {
                    "$set": {
                        "status": "escalated",
                        "escalated_at": datetime.utcnow(),
                        "escalation_reason": "; ".join(escalation_check.get("reasons", [])),
                        "escalation_priority": escalation_check.get("escalation_priority", "medium")
                    }
                }
            )
            
            # Create support ticket
            conversation = await self.db[self.chatbot_conversations_collection].find_one(
                {"conversation_id": conversation_id}
            )
            
            if conversation:
                # Extract conversation summary
                messages = conversation.get("messages", [])
                customer_messages = [msg.get("message", "") for msg in messages if msg.get("sender") == "customer"]
                conversation_summary = " | ".join(customer_messages[-3:])  # Last 3 customer messages
                
                ticket_data = {
                    "ticket_id": f"ESCALATED_{conversation_id}",
                    "customer_id": conversation.get("customer_id"),
                    "subject": f"Escalated Chat: {conversation_id}",
                    "description": f"Escalated from chatbot conversation. Summary: {conversation_summary}",
                    "priority": escalation_check.get("escalation_priority", "medium"),
                    "source": "chatbot_escalation",
                    "conversation_id": conversation_id,
                    "created_at": datetime.utcnow()
                }
                
                # Route the escalated ticket
                await self.smart_ticket_routing(ticket_data)
            
            logger.info(f"Conversation {conversation_id} escalated to human agent")
            
        except Exception as e:
            logger.error(f"Failed to escalate conversation: {e}")
    
    async def auto_build_knowledge_base(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Automatically build and update knowledge base from resolved tickets."""
        try:
            data_lake = await get_data_lake()
            
            # Get resolved tickets from the specified period
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=time_period_days)
            
            resolved_tickets = await self.db["tickets"].find({
                "status": "resolved",
                "resolved_at": {"$gte": start_date, "$lte": end_date},
                "resolution_notes": {"$exists": True, "$ne": ""}
            }).to_list(None)
            
            if not resolved_tickets:
                return {"message": "No resolved tickets found for knowledge base building"}
            
            # Analyze tickets and extract knowledge
            knowledge_articles = []
            
            for ticket in resolved_tickets:
                article = await self._extract_knowledge_from_ticket(ticket)
                if article:
                    knowledge_articles.append(article)
            
            # Cluster similar articles
            clustered_articles = await self._cluster_knowledge_articles(knowledge_articles)
            
            # Generate FAQ entries
            faq_entries = await self._generate_faq_entries(clustered_articles)
            
            # Update knowledge base
            kb_update_result = await self._update_knowledge_base(clustered_articles, faq_entries)
            
            # Generate knowledge base analytics
            kb_analytics = await self._analyze_knowledge_base_effectiveness()
            
            build_result = {
                "build_id": f"KB_{str(uuid.uuid4())[:8].upper()}",
                "tickets_analyzed": len(resolved_tickets),
                "articles_extracted": len(knowledge_articles),
                "articles_clustered": len(clustered_articles),
                "faq_entries_generated": len(faq_entries),
                "knowledge_base_updates": kb_update_result,
                "effectiveness_analytics": kb_analytics,
                "build_date": datetime.utcnow(),
                "time_period_days": time_period_days
            }
            
            # Store build result
            await self.db["kb_builds"].insert_one(build_result)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="support",
                event_type="knowledge_base_built",
                entity_type="knowledge_base",
                entity_id=build_result["build_id"],
                data={
                    "articles_created": len(clustered_articles),
                    "faq_entries": len(faq_entries),
                    "tickets_processed": len(resolved_tickets)
                }
            )
            
            logger.info(f"Knowledge base auto-built: {build_result['build_id']}")
            
            return build_result
            
        except Exception as e:
            logger.error(f"Failed to auto-build knowledge base: {e}")
            return {}
    
    async def _extract_knowledge_from_ticket(self, ticket: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract knowledge article from resolved ticket."""
        try:
            nlp = await get_nlp_processor()
            
            subject = ticket.get("subject", "")
            description = ticket.get("description", "")
            resolution_notes = ticket.get("resolution_notes", "")
            
            # Skip if resolution is too short or generic
            if len(resolution_notes) < 50:
                return None
            
            # Extract keywords
            combined_text = f"{subject} {description}"
            keywords = await nlp.extract_keywords(combined_text, 8)
            
            # Categorize the issue
            categories = [
                "technical_issue", "billing_inquiry", "account_access", "feature_request",
                "bug_report", "general_inquiry", "integration_issue", "performance_issue"
            ]
            category_scores = await nlp.classify_text(combined_text, categories)
            predicted_category = max(category_scores.items(), key=lambda x: x[1])[0]
            
            # Generate article
            article = {
                "article_id": f"KB_{str(uuid.uuid4())[:8].upper()}",
                "title": self._generate_article_title(subject, predicted_category),
                "category": predicted_category,
                "keywords": keywords,
                "problem_description": description,
                "solution": resolution_notes,
                "source_ticket_id": ticket.get("ticket_id"),
                "confidence_score": category_scores.get(predicted_category, 0),
                "created_from": "resolved_ticket",
                "created_at": datetime.utcnow()
            }
            
            return article
            
        except Exception as e:
            logger.error(f"Failed to extract knowledge from ticket: {e}")
            return None
    
    def _generate_article_title(self, subject: str, category: str) -> str:
        """Generate a clear article title."""
        try:
            # Clean up subject
            cleaned_subject = re.sub(r'^(re:|fwd:)', '', subject, flags=re.IGNORECASE).strip()
            
            # Add category context if not already present
            category_keywords = {
                "technical_issue": ["technical", "error", "bug"],
                "billing_inquiry": ["billing", "payment", "invoice"],
                "account_access": ["login", "access", "password"],
                "feature_request": ["feature", "enhancement"],
                "integration_issue": ["integration", "api", "sync"],
                "performance_issue": ["slow", "performance", "timeout"]
            }
            
            keywords = category_keywords.get(category, [])
            has_category_context = any(keyword in cleaned_subject.lower() for keyword in keywords)
            
            if not has_category_context and len(cleaned_subject) < 60:
                category_prefix = {
                    "technical_issue": "Technical Issue:",
                    "billing_inquiry": "Billing:",
                    "account_access": "Account Access:",
                    "feature_request": "Feature Request:",
                    "integration_issue": "Integration:",
                    "performance_issue": "Performance:"
                }.get(category, "")
                
                if category_prefix:
                    return f"{category_prefix} {cleaned_subject}"
            
            return cleaned_subject
            
        except Exception as e:
            logger.error(f"Failed to generate article title: {e}")
            return subject
    
    async def _cluster_knowledge_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster similar knowledge articles to avoid duplicates."""
        try:
            if not articles:
                return []
            
            nlp = await get_nlp_processor()
            
            # Group articles by category first
            category_groups = defaultdict(list)
            for article in articles:
                category = article.get("category", "general")
                category_groups[category].append(article)
            
            clustered_articles = []
            
            # Within each category, find similar articles
            for category, category_articles in category_groups.items():
                if len(category_articles) == 1:
                    clustered_articles.extend(category_articles)
                    continue
                
                # Find similar articles within category
                processed_indices = set()
                
                for i, article1 in enumerate(category_articles):
                    if i in processed_indices:
                        continue
                    
                    similar_articles = [article1]
                    article1_text = f"{article1.get('title', '')} {article1.get('problem_description', '')}"
                    
                    for j, article2 in enumerate(category_articles[i+1:], i+1):
                        if j in processed_indices:
                            continue
                        
                        article2_text = f"{article2.get('title', '')} {article2.get('problem_description', '')}"
                        similarity = await nlp.calculate_similarity(article1_text, article2_text)
                        
                        if similarity > 0.7:  # High similarity threshold
                            similar_articles.append(article2)
                            processed_indices.add(j)
                    
                    # Merge similar articles
                    if len(similar_articles) > 1:
                        merged_article = await self._merge_similar_articles(similar_articles)
                        clustered_articles.append(merged_article)
                    else:
                        clustered_articles.append(article1)
                    
                    processed_indices.add(i)
            
            return clustered_articles
            
        except Exception as e:
            logger.error(f"Failed to cluster knowledge articles: {e}")
            return articles
    
    async def _merge_similar_articles(self, similar_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge similar articles into one comprehensive article."""
        try:
            # Use the article with highest confidence as base
            base_article = max(similar_articles, key=lambda x: x.get("confidence_score", 0))
            
            # Combine keywords
            all_keywords = []
            for article in similar_articles:
                all_keywords.extend(article.get("keywords", []))
            unique_keywords = list(set(all_keywords))
            
            # Combine solutions
            solutions = []
            for article in similar_articles:
                solution = article.get("solution", "")
                if solution and solution not in solutions:
                    solutions.append(solution)
            
            # Create merged article
            merged_article = {
                **base_article,
                "article_id": f"KB_MERGED_{str(uuid.uuid4())[:8].upper()}",
                "keywords": unique_keywords[:10],  # Limit to 10 keywords
                "solution": " | ".join(solutions),
                "source_ticket_ids": [article.get("source_ticket_id") for article in similar_articles],
                "merged_from": len(similar_articles),
                "created_from": "merged_articles"
            }
            
            return merged_article
            
        except Exception as e:
            logger.error(f"Failed to merge similar articles: {e}")
            return similar_articles[0] if similar_articles else {}
    
    async def _generate_faq_entries(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate FAQ entries from knowledge articles."""
        try:
            faq_entries = []
            
            # Group articles by category
            category_groups = defaultdict(list)
            for article in articles:
                category = article.get("category", "general")
                category_groups[category].append(article)
            
            # Generate FAQ for each category
            for category, category_articles in category_groups.items():
                # Sort by confidence score
                sorted_articles = sorted(category_articles, key=lambda x: x.get("confidence_score", 0), reverse=True)
                
                # Take top articles for FAQ
                top_articles = sorted_articles[:5]  # Top 5 per category
                
                for article in top_articles:
                    faq_entry = {
                        "faq_id": f"FAQ_{str(uuid.uuid4())[:8].upper()}",
                        "question": self._generate_faq_question(article),
                        "answer": self._generate_faq_answer(article),
                        "category": category,
                        "keywords": article.get("keywords", []),
                        "source_article_id": article.get("article_id"),
                        "popularity_score": 0,  # Will be updated based on usage
                        "created_at": datetime.utcnow()
                    }
                    
                    faq_entries.append(faq_entry)
            
            return faq_entries
            
        except Exception as e:
            logger.error(f"Failed to generate FAQ entries: {e}")
            return []
    
    def _generate_faq_question(self, article: Dict[str, Any]) -> str:
        """Generate FAQ question from article."""
        try:
            title = article.get("title", "")
            problem_description = article.get("problem_description", "")
            
            # Convert statement to question format
            if "?" in title:
                return title
            
            # Common question patterns
            if any(word in title.lower() for word in ["error", "issue", "problem"]):
                return f"How do I resolve: {title}?"
            elif any(word in title.lower() for word in ["how to", "setup", "configure"]):
                return title if title.endswith("?") else f"{title}?"
            else:
                return f"How do I handle {title.lower()}?"
            
        except Exception as e:
            logger.error(f"Failed to generate FAQ question: {e}")
            return article.get("title", "Question")
    
    def _generate_faq_answer(self, article: Dict[str, Any]) -> str:
        """Generate FAQ answer from article."""
        try:
            solution = article.get("solution", "")
            
            # Clean up and format solution
            if len(solution) > 300:
                # Truncate long solutions
                solution = solution[:300] + "..."
            
            # Add helpful prefix
            if not solution.lower().startswith(("to", "you can", "please", "first")):
                solution = f"To resolve this issue: {solution}"
            
            return solution
            
        except Exception as e:
            logger.error(f"Failed to generate FAQ answer: {e}")
            return article.get("solution", "Please contact support for assistance.")
    
    async def _update_knowledge_base(self, articles: List[Dict[str, Any]], 
                                   faq_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update knowledge base with new articles and FAQ entries."""
        try:
            # Insert new articles
            articles_inserted = 0
            if articles:
                # Check for duplicates before inserting
                for article in articles:
                    existing = await self.db[self.knowledge_base_collection].find_one({
                        "title": article.get("title"),
                        "category": article.get("category")
                    })
                    
                    if not existing:
                        await self.db[self.knowledge_base_collection].insert_one(article)
                        articles_inserted += 1
            
            # Insert new FAQ entries
            faq_inserted = 0
            if faq_entries:
                for faq in faq_entries:
                    existing = await self.db["faq_entries"].find_one({
                        "question": faq.get("question"),
                        "category": faq.get("category")
                    })
                    
                    if not existing:
                        await self.db["faq_entries"].insert_one(faq)
                        faq_inserted += 1
            
            return {
                "articles_inserted": articles_inserted,
                "faq_entries_inserted": faq_inserted,
                "total_articles_processed": len(articles),
                "total_faq_processed": len(faq_entries)
            }
            
        except Exception as e:
            logger.error(f"Failed to update knowledge base: {e}")
            return {"articles_inserted": 0, "faq_entries_inserted": 0}
    
    async def _analyze_knowledge_base_effectiveness(self) -> Dict[str, Any]:
        """Analyze knowledge base effectiveness."""
        try:
            # Get knowledge base statistics
            total_articles = await self.db[self.knowledge_base_collection].count_documents({})
            total_faq = await self.db["faq_entries"].count_documents({})
            
            # Category distribution
            category_pipeline = [
                {"$group": {"_id": "$category", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            category_distribution = []
            cursor = self.db[self.knowledge_base_collection].aggregate(category_pipeline)
            async for category in cursor:
                category_distribution.append({
                    "category": category["_id"],
                    "article_count": category["count"]
                })
            
            # Usage statistics (if available)
            # This would require tracking article views/usage
            
            return {
                "total_articles": total_articles,
                "total_faq_entries": total_faq,
                "category_distribution": category_distribution,
                "coverage_score": min(total_articles / 50, 1.0),  # Assume 50 articles = good coverage
                "last_updated": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze knowledge base effectiveness: {e}")
            return {}
    
    async def get_support_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive support analytics and insights."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get ticket routing data
            routing_data = await self.db[self.ticket_routing_collection].find({
                "created_at": {"$gte": start_date, "$lte": end_date}
            }).to_list(None)
            
            # Get chatbot conversations
            chatbot_data = await self.db[self.chatbot_conversations_collection].find({
                "created_at": {"$gte": start_date, "$lte": end_date}
            }).to_list(None)
            
            # Calculate metrics
            total_tickets_routed = len(routing_data)
            auto_assigned_tickets = len([r for r in routing_data if r.get("auto_assign", False)])
            avg_routing_confidence = sum(r.get("routing_confidence", 0) for r in routing_data) / len(routing_data) if routing_data else 0
            
            total_conversations = len(chatbot_data)
            escalated_conversations = len([c for c in chatbot_data if c.get("status") == "escalated"])
            
            # Priority distribution
            priority_distribution = {}
            for routing in routing_data:
                priority = routing.get("priority_level", "unknown")
                priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
            
            # Category distribution
            category_distribution = {}
            for routing in routing_data:
                category = routing.get("ticket_analysis", {}).get("predicted_category", "unknown")
                category_distribution[category] = category_distribution.get(category, 0) + 1
            
            # Agent workload analysis
            agent_assignments = {}
            for routing in routing_data:
                agent = routing.get("recommended_agent")
                if agent:
                    agent_assignments[agent] = agent_assignments.get(agent, 0) + 1
            
            analytics = {
                "period_days": days,
                "ticket_routing": {
                    "total_tickets_routed": total_tickets_routed,
                    "auto_assigned_tickets": auto_assigned_tickets,
                    "auto_assignment_rate": (auto_assigned_tickets / total_tickets_routed) if total_tickets_routed > 0 else 0,
                    "avg_routing_confidence": round(avg_routing_confidence, 3),
                    "priority_distribution": priority_distribution,
                    "category_distribution": category_distribution
                },
                "chatbot_performance": {
                    "total_conversations": total_conversations,
                    "escalated_conversations": escalated_conversations,
                    "escalation_rate": (escalated_conversations / total_conversations) if total_conversations > 0 else 0,
                    "avg_conversation_length": self._calculate_avg_conversation_length(chatbot_data)
                },
                "agent_workload": {
                    "agent_assignments": agent_assignments,
                    "most_assigned_agent": max(agent_assignments.items(), key=lambda x: x[1])[0] if agent_assignments else None,
                    "workload_distribution": self._calculate_workload_distribution(agent_assignments)
                },
                "efficiency_metrics": {
                    "routing_automation_rate": (auto_assigned_tickets / total_tickets_routed) if total_tickets_routed > 0 else 0,
                    "chatbot_resolution_rate": 1 - ((escalated_conversations / total_conversations) if total_conversations > 0 else 0),
                    "overall_automation_score": self._calculate_automation_score(routing_data, chatbot_data)
                },
                "generated_at": datetime.utcnow()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get support analytics: {e}")
            return {}
    
    def _calculate_avg_conversation_length(self, conversations: List[Dict[str, Any]]) -> float:
        """Calculate average conversation length."""
        try:
            if not conversations:
                return 0
            
            total_messages = sum(len(conv.get("messages", [])) for conv in conversations)
            return total_messages / len(conversations)
            
        except Exception as e:
            logger.error(f"Failed to calculate average conversation length: {e}")
            return 0
    
    def _calculate_workload_distribution(self, agent_assignments: Dict[str, int]) -> Dict[str, Any]:
        """Calculate workload distribution statistics."""
        try:
            if not agent_assignments:
                return {}
            
            assignments = list(agent_assignments.values())
            total_assignments = sum(assignments)
            
            return {
                "total_assignments": total_assignments,
                "avg_assignments_per_agent": total_assignments / len(assignments),
                "max_assignments": max(assignments),
                "min_assignments": min(assignments),
                "workload_variance": max(assignments) - min(assignments)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate workload distribution: {e}")
            return {}
    
    def _calculate_automation_score(self, routing_data: List[Dict[str, Any]], 
                                  chatbot_data: List[Dict[str, Any]]) -> float:
        """Calculate overall automation score."""
        try:
            if not routing_data and not chatbot_data:
                return 0
            
            # Routing automation score
            auto_assigned = len([r for r in routing_data if r.get("auto_assign", False)])
            routing_score = (auto_assigned / len(routing_data)) if routing_data else 0
            
            # Chatbot automation score
            escalated = len([c for c in chatbot_data if c.get("status") == "escalated"])
            chatbot_score = 1 - ((escalated / len(chatbot_data)) if chatbot_data else 0)
            
            # Combined score
            if routing_data and chatbot_data:
                return (routing_score + chatbot_score) / 2
            elif routing_data:
                return routing_score
            else:
                return chatbot_score
            
        except Exception as e:
            logger.error(f"Failed to calculate automation score: {e}")
            return 0.5
