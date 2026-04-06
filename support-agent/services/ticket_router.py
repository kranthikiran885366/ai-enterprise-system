"""Intelligent ticket routing service with sentiment analysis and escalation logic."""

from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from shared_libs.ai_providers import get_orchestrator
from shared_libs.intelligence import get_intelligence_service


class TicketPriority(str, Enum):
    """Ticket priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TicketCategory(str, Enum):
    """Ticket category enumeration."""
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"


@dataclass
class SentimentAnalysis:
    """Sentiment analysis result."""
    sentiment: str  # positive, neutral, negative
    score: float  # -1 to 1
    urgency_level: str  # low, medium, high
    emotion: str  # frustrated, happy, neutral, confused
    keywords: List[str]


@dataclass
class RoutingDecision:
    """Ticket routing decision."""
    ticket_id: str
    category: str
    priority: str
    sentiment: SentimentAnalysis
    assigned_to: str  # Agent/team ID
    reasoning: str
    sla_hours: int  # Service level agreement
    suggested_response: Optional[str] = None


class TicketRouter:
    """Intelligent ticket routing engine using AI and sentiment analysis."""
    
    # Urgency keywords indicating high priority
    URGENCY_KEYWORDS = {
        "urgent": 3,
        "asap": 3,
        "immediately": 3,
        "critical": 3,
        "emergency": 3,
        "broken": 2,
        "down": 2,
        "crashed": 2,
        "error": 1,
        "problem": 1,
        "issue": 1,
        "help": 1,
    }
    
    # Positive sentiment keywords
    POSITIVE_KEYWORDS = {
        "thank": 2,
        "great": 2,
        "excellent": 2,
        "happy": 2,
        "love": 2,
        "good": 1,
        "nice": 1,
        "awesome": 1,
    }
    
    # Team expertise mapping
    TEAM_EXPERTISE = {
        "technical_team": ["technical", "bug_report", "error"],
        "billing_team": ["billing", "payment", "invoice"],
        "general_support": ["general", "feature_request"],
        "senior_team": ["critical", "escalated"],
    }
    
    def __init__(self):
        self.orchestrator = None
        self.intelligence = None
    
    async def initialize(self):
        """Initialize router with AI services."""
        self.orchestrator = await get_orchestrator()
        self.intelligence = await get_intelligence_service()
    
    async def route_ticket(
        self,
        ticket_id: str,
        customer_message: str,
        ticket_category: Optional[str] = None,
        customer_history: Optional[List[Dict[str, Any]]] = None
    ) -> RoutingDecision:
        """
        Route a support ticket intelligently using sentiment analysis and expertise matching.
        
        Args:
            ticket_id: Unique ticket identifier
            customer_message: Customer's message/description
            ticket_category: Optional pre-categorized category
            customer_history: Optional list of previous interactions
        
        Returns:
            RoutingDecision with assignment and handling instructions
        """
        try:
            # Analyze sentiment and urgency
            sentiment_analysis = await self._analyze_sentiment(customer_message)
            
            # Determine category if not provided
            if not ticket_category:
                ticket_category = await self._classify_category(customer_message)
            
            # Determine priority
            priority = self._determine_priority(
                sentiment_analysis,
                ticket_category,
                customer_history or []
            )
            
            # Determine SLA hours based on priority
            sla_hours = self._get_sla_hours(priority)
            
            # Find best agent/team
            assigned_to, reasoning = await self._find_best_agent(
                ticket_category,
                priority,
                sentiment_analysis,
                customer_message
            )
            
            # Generate suggested initial response if sentiment is negative
            suggested_response = None
            if sentiment_analysis.sentiment == "negative":
                suggested_response = await self._generate_empathetic_response(
                    customer_message,
                    priority
                )
            
            decision = RoutingDecision(
                ticket_id=ticket_id,
                category=ticket_category,
                priority=priority,
                sentiment=sentiment_analysis,
                assigned_to=assigned_to,
                reasoning=reasoning,
                sla_hours=sla_hours,
                suggested_response=suggested_response
            )
            
            logger.info(f"Ticket {ticket_id} routed to {assigned_to} - Priority: {priority}")
            return decision
            
        except Exception as e:
            logger.error(f"Ticket routing failed: {e}")
            raise
    
    async def _analyze_sentiment(self, message: str) -> SentimentAnalysis:
        """Analyze message sentiment and urgency using AI."""
        try:
            prompt = f"""Analyze the sentiment and urgency of this customer support message:

"{message}"

Respond with:
1. SENTIMENT: [positive/neutral/negative]
2. SCORE: [decimal -1 to 1, where -1 is most negative]
3. URGENCY: [low/medium/high]
4. EMOTION: [frustrated/happy/neutral/confused/angry/worried]
5. KEYWORDS: [comma-separated list of key words indicating sentiment]

Be concise."""
            
            response = await self.orchestrator.complete(prompt, temperature=0.5)
            
            # Parse response
            sentiment = "neutral"
            score = 0.0
            urgency = "medium"
            emotion = "neutral"
            keywords = []
            
            lines = response.split('\n')
            for line in lines:
                if 'SENTIMENT:' in line:
                    sentiment = line.split(':')[1].strip().lower()
                elif 'SCORE:' in line:
                    try:
                        score = float(line.split(':')[1].strip())
                    except:
                        score = 0.0
                elif 'URGENCY:' in line:
                    urgency = line.split(':')[1].strip().lower()
                elif 'EMOTION:' in line:
                    emotion = line.split(':')[1].strip().lower()
                elif 'KEYWORDS:' in line:
                    keywords = [k.strip() for k in line.split(':')[1].strip().split(',')]
            
            # Verify sentiment from keywords
            if sentiment not in ["positive", "neutral", "negative"]:
                sentiment = "neutral"
            
            return SentimentAnalysis(
                sentiment=sentiment,
                score=score,
                urgency_level=urgency,
                emotion=emotion,
                keywords=keywords
            )
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return SentimentAnalysis("neutral", 0.0, "medium", "neutral", [])
    
    async def _classify_category(self, message: str) -> str:
        """Classify ticket into a category."""
        try:
            prompt = f"""Classify this support ticket message into one category:

Message: "{message}"

Choose ONE category:
- technical (system errors, bugs, performance)
- billing (payments, invoices, subscriptions)
- general (usage questions, account issues)
- feature_request (new functionality requests)
- bug_report (specific bug reports)

Respond with ONLY the category name."""
            
            response = await self.orchestrator.complete(prompt, temperature=0.3)
            category = response.strip().lower()
            
            valid_categories = ["technical", "billing", "general", "feature_request", "bug_report"]
            if category not in valid_categories:
                category = "general"
            
            return category
            
        except Exception as e:
            logger.warning(f"Category classification failed: {e}")
            return "general"
    
    def _determine_priority(
        self,
        sentiment: SentimentAnalysis,
        category: str,
        customer_history: List[Dict[str, Any]]
    ) -> str:
        """Determine ticket priority."""
        
        # Base priority from urgency
        if sentiment.urgency_level == "high":
            base_priority = 2  # high
        elif sentiment.urgency_level == "medium":
            base_priority = 1  # medium
        else:
            base_priority = 0  # low
        
        # Sentiment adjustment
        if sentiment.sentiment == "negative":
            base_priority += 1  # Escalate negative sentiment
        
        # Category considerations
        if category == "billing":
            base_priority += 1  # Billing issues are usually higher priority
        
        # Customer history - repeat issues should be higher priority
        if customer_history:
            if len([h for h in customer_history if h.get("resolved") == False]) > 2:
                base_priority += 1  # Multiple unresolved issues
        
        # Map to priority name
        priority_map = {
            0: TicketPriority.LOW,
            1: TicketPriority.MEDIUM,
            2: TicketPriority.HIGH,
            3: TicketPriority.CRITICAL,
        }
        
        priority_level = min(base_priority, 3)
        return priority_map[priority_level].value
    
    def _get_sla_hours(self, priority: str) -> int:
        """Get SLA response time based on priority."""
        sla_map = {
            TicketPriority.CRITICAL.value: 1,
            TicketPriority.HIGH.value: 4,
            TicketPriority.MEDIUM.value: 24,
            TicketPriority.LOW.value: 48,
        }
        return sla_map.get(priority, 24)
    
    async def _find_best_agent(
        self,
        category: str,
        priority: str,
        sentiment: SentimentAnalysis,
        message: str
    ) -> Tuple[str, str]:
        """Find the best agent/team for this ticket."""
        
        # Select team based on category
        if category in ["technical", "bug_report"]:
            base_team = "technical_team"
        elif category == "billing":
            base_team = "billing_team"
        else:
            base_team = "general_support"
        
        # Escalate to senior team if critical or negative sentiment
        if priority == TicketPriority.CRITICAL.value or sentiment.sentiment == "negative":
            if priority == TicketPriority.CRITICAL.value:
                assigned_to = "senior_team"
                reasoning = f"Critical priority {category} issue requires senior team"
            else:
                assigned_to = "senior_team"
                reasoning = f"Negative sentiment requires experienced agent"
        else:
            assigned_to = base_team
            reasoning = f"Routing to {base_team} for {category} expertise"
        
        return assigned_to, reasoning
    
    async def _generate_empathetic_response(
        self,
        customer_message: str,
        priority: str
    ) -> str:
        """Generate an empathetic initial response for negative sentiment tickets."""
        try:
            urgency_tone = "immediately" if priority == TicketPriority.CRITICAL.value else "shortly"
            
            prompt = f"""Generate a brief, empathetic response to this customer support message.
The response should acknowledge their concern and assure them you'll help {urgency_tone}.

Customer Message: "{customer_message}"

Response (2-3 sentences max, professional but warm)."""
            
            response = await self.orchestrator.complete(prompt, temperature=0.7)
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Response generation failed: {e}")
            return "Thank you for contacting us. We're here to help and will address your issue as soon as possible."
    
    async def suggest_resolution(
        self,
        ticket_id: str,
        customer_message: str,
        category: str
    ) -> str:
        """Suggest a resolution for the ticket."""
        try:
            prompt = f"""Suggest a resolution for this {category} support ticket:

Customer Message: "{customer_message}"

Provide a concise, step-by-step suggestion that an agent could propose to resolve this issue."""
            
            response = await self.orchestrator.complete(prompt, temperature=0.6)
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Resolution suggestion failed: {e}")
            return "We'll investigate this issue and provide a solution shortly."
