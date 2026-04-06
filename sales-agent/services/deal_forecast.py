"""Deal forecasting service using AI analysis and historical patterns."""

from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import statistics

from shared_libs.ai_providers import get_orchestrator


@dataclass
class DealForecast:
    """Deal forecasting result."""
    deal_id: str
    current_stage: str
    current_probability: float
    predicted_probability: float
    probability_change: float
    expected_close_date: str
    confidence_score: float  # 0-100, how confident in prediction
    risk_factors: List[str]
    success_factors: List[str]
    ai_analysis: str
    recommended_actions: List[str]


class DealForecaster:
    """AI-powered deal forecasting engine."""
    
    def __init__(self):
        self.orchestrator = None
    
    async def initialize(self):
        """Initialize forecaster with AI orchestrator."""
        self.orchestrator = await get_orchestrator()
    
    # Stage probability baseline mappings
    STAGE_BASE_PROBABILITIES = {
        "prospecting": 5,
        "qualification": 15,
        "proposal": 40,
        "negotiation": 65,
        "closed_won": 100,
        "closed_lost": 0,
    }
    
    async def forecast_deal(
        self,
        deal: Dict[str, Any],
        historical_deals: List[Dict[str, Any]] = None
    ) -> DealForecast:
        """
        Forecast deal close probability using AI and historical analysis.
        
        Args:
            deal: Current deal data
            historical_deals: Historical deals for pattern analysis
        
        Returns:
            DealForecast with predictions and recommendations
        """
        try:
            # Get baseline probability from stage
            current_probability = deal.get("probability", 
                                          self.STAGE_BASE_PROBABILITIES.get(deal.get("stage", "prospecting"), 20))
            
            # Analyze deal characteristics
            stage = deal.get("stage", "prospecting")
            days_in_stage = self._calculate_days_in_stage(deal)
            deal_value = deal.get("deal_value", 0)
            
            # Get AI-powered analysis
            ai_analysis, predicted_prob, risk_factors, success_factors, actions = await self._analyze_deal_with_ai(
                deal, 
                historical_deals or [],
                current_probability
            )
            
            # Calculate confidence score based on data completeness
            confidence_score = self._calculate_confidence_score(deal, days_in_stage)
            
            # Predict close date
            expected_close = self._predict_close_date(deal, current_probability)
            
            # Calculate probability change
            prob_change = predicted_prob - current_probability
            
            forecast = DealForecast(
                deal_id=deal.get("id", deal.get("deal_id", "unknown")),
                current_stage=stage,
                current_probability=float(current_probability),
                predicted_probability=float(predicted_prob),
                probability_change=float(prob_change),
                expected_close_date=expected_close,
                confidence_score=float(confidence_score),
                risk_factors=risk_factors,
                success_factors=success_factors,
                ai_analysis=ai_analysis,
                recommended_actions=actions
            )
            
            logger.info(f"Deal forecast generated: {deal.get('deal_id')} - Probability: {predicted_prob:.1f}%")
            return forecast
            
        except Exception as e:
            logger.error(f"Deal forecasting failed: {e}")
            raise
    
    def _calculate_days_in_stage(self, deal: Dict[str, Any]) -> int:
        """Calculate how many days deal has been in current stage."""
        try:
            if "stage_entered_date" in deal:
                stage_date = deal["stage_entered_date"]
                if isinstance(stage_date, str):
                    stage_date = datetime.fromisoformat(stage_date)
                return (datetime.utcnow() - stage_date).days
            
            if "created_at" in deal:
                created_date = deal["created_at"]
                if isinstance(created_date, str):
                    created_date = datetime.fromisoformat(created_date)
                return (datetime.utcnow() - created_date).days
            
            return 0
        except:
            return 0
    
    async def _analyze_deal_with_ai(
        self,
        deal: Dict[str, Any],
        historical_deals: List[Dict[str, Any]],
        current_probability: float
    ) -> Tuple[str, float, List[str], List[str], List[str]]:
        """Get AI-powered deal analysis."""
        try:
            # Prepare historical context
            similar_deals = self._find_similar_deals(deal, historical_deals)
            historical_win_rate = self._calculate_win_rate(similar_deals)
            
            prompt = f"""Analyze this sales deal and predict its close probability.

DEAL INFORMATION:
- Company: {deal.get('company_name', 'Unknown')}
- Industry: {deal.get('industry', 'Unknown')}
- Deal Value: ${deal.get('deal_value', 0):,.0f}
- Current Stage: {deal.get('stage', 'Unknown')}
- Days in Stage: {self._calculate_days_in_stage(deal)}
- Expected Close Date: {deal.get('expected_close_date', 'Not specified')}
- Current Probability: {current_probability:.0f}%
- Contact Title: {deal.get('contact_title', 'Unknown')}
- Decision Timeline: {deal.get('decision_timeline', 'Unknown')}
- Competition: {deal.get('has_competition', False)}
- Budget Confirmed: {deal.get('budget_confirmed', False)}

HISTORICAL CONTEXT:
- Similar Deals Win Rate: {historical_win_rate:.1f}%
- Number of Similar Historical Deals: {len(similar_deals)}

KEY DEAL CHARACTERISTICS:
- Technical fit: {deal.get('technical_fit', 'Unknown')}
- Relationship strength: {deal.get('relationship_strength', 'Unknown')}
- Stakeholder alignment: {deal.get('stakeholder_alignment', 'Unknown')}

Provide your analysis in this exact format:

PREDICTED_PROBABILITY: [number 0-100]

ANALYSIS: [2-3 sentences explaining the probability]

RISK_FACTORS: [List 2-3 specific risks separated by |]

SUCCESS_FACTORS: [List 2-3 positive indicators separated by |]

RECOMMENDED_ACTIONS: [List 2-3 specific next steps separated by |]"""
            
            response = await self.orchestrator.complete(prompt, temperature=0.7)
            
            # Parse response
            predicted_prob = current_probability
            analysis = ""
            risk_factors = []
            success_factors = []
            actions = []
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('PREDICTED_PROBABILITY:'):
                    try:
                        prob_str = line.replace('PREDICTED_PROBABILITY:', '').strip()
                        predicted_prob = float(prob_str.split()[0])  # Get first number
                        predicted_prob = min(100, max(0, predicted_prob))  # Constrain to 0-100
                    except:
                        pass
                
                elif line.startswith('ANALYSIS:'):
                    analysis = line.replace('ANALYSIS:', '').strip()
                
                elif line.startswith('RISK_FACTORS:'):
                    factors_str = line.replace('RISK_FACTORS:', '').strip()
                    risk_factors = [f.strip() for f in factors_str.split('|') if f.strip()]
                
                elif line.startswith('SUCCESS_FACTORS:'):
                    factors_str = line.replace('SUCCESS_FACTORS:', '').strip()
                    success_factors = [f.strip() for f in factors_str.split('|') if f.strip()]
                
                elif line.startswith('RECOMMENDED_ACTIONS:'):
                    actions_str = line.replace('RECOMMENDED_ACTIONS:', '').strip()
                    actions = [a.strip() for a in actions_str.split('|') if a.strip()]
            
            return analysis or "Deal analysis completed", predicted_prob, risk_factors, success_factors, actions
            
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return "Unable to provide AI analysis", current_probability, [], [], []
    
    def _find_similar_deals(
        self,
        deal: Dict[str, Any],
        historical_deals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find deals similar to the current one."""
        similar = []
        
        try:
            for hist_deal in historical_deals:
                # Check for similarity: same stage, similar value, same industry
                if (hist_deal.get("stage") == deal.get("stage") and
                    abs(hist_deal.get("deal_value", 0) - deal.get("deal_value", 0)) < 50000 and
                    hist_deal.get("industry") == deal.get("industry")):
                    similar.append(hist_deal)
            
            return similar
        except:
            return []
    
    def _calculate_win_rate(self, deals: List[Dict[str, Any]]) -> float:
        """Calculate win rate from a list of deals."""
        if not deals:
            return 50.0  # Default to 50% if no data
        
        try:
            won_deals = sum(1 for d in deals if d.get("stage") == "closed_won")
            return (won_deals / len(deals)) * 100
        except:
            return 50.0
    
    def _calculate_confidence_score(self, deal: Dict[str, Any], days_in_stage: int) -> float:
        """Calculate confidence score for the forecast."""
        confidence = 50.0  # Base score
        
        # Data completeness
        if deal.get("contact_name"):
            confidence += 5
        if deal.get("company_name"):
            confidence += 5
        if deal.get("deal_value", 0) > 0:
            confidence += 5
        if deal.get("expected_close_date"):
            confidence += 5
        if deal.get("decision_timeline"):
            confidence += 5
        
        # Deal maturity (more time in stage = more data)
        if days_in_stage > 30:
            confidence += 10
        elif days_in_stage > 14:
            confidence += 5
        
        # Activity level
        activities = deal.get("activities", [])
        if len(activities) > 5:
            confidence += 10
        elif len(activities) > 2:
            confidence += 5
        
        return min(confidence, 100.0)
    
    def _predict_close_date(self, deal: Dict[str, Any], probability: float) -> str:
        """Predict deal close date based on stage and historical patterns."""
        try:
            # Stage-based average cycle times (in days)
            stage_cycle_times = {
                "prospecting": 45,
                "qualification": 40,
                "proposal": 30,
                "negotiation": 20,
                "closed_won": 0,
                "closed_lost": 0,
            }
            
            stage = deal.get("stage", "prospecting")
            cycle_time = stage_cycle_times.get(stage, 30)
            
            # If expected close date is provided, use it
            if deal.get("expected_close_date"):
                return deal.get("expected_close_date")
            
            # Otherwise predict based on stage
            if stage in ["closed_won", "closed_lost"]:
                return datetime.utcnow().isoformat()
            
            predicted_date = datetime.utcnow() + timedelta(days=cycle_time)
            return predicted_date.isoformat()
            
        except Exception as e:
            logger.warning(f"Failed to predict close date: {e}")
            return (datetime.utcnow() + timedelta(days=30)).isoformat()
    
    async def forecast_sales_pipeline(
        self,
        deals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Forecast entire sales pipeline."""
        try:
            forecasts = []
            total_value = 0.0
            weighted_value = 0.0
            
            for deal in deals:
                forecast = await self.forecast_deal(deal)
                forecasts.append(forecast)
                
                deal_value = deal.get("deal_value", 0)
                total_value += deal_value
                weighted_value += deal_value * (forecast.predicted_probability / 100)
            
            return {
                "forecasts": forecasts,
                "total_pipeline_value": round(total_value, 2),
                "weighted_pipeline_value": round(weighted_value, 2),
                "expected_close_rate": round((weighted_value / total_value * 100), 1) if total_value > 0 else 0,
                "forecast_date": datetime.utcnow().isoformat(),
                "deal_count": len(deals)
            }
            
        except Exception as e:
            logger.error(f"Pipeline forecasting failed: {e}")
            raise
