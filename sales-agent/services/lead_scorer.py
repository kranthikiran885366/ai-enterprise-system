"""Advanced lead scoring with multi-factor analysis and AI enhancement."""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import numpy as np

from shared_libs.ai_providers import get_orchestrator


@dataclass
class LeadScore:
    """Lead scoring result."""
    lead_id: str
    overall_score: float  # 0-100
    company_score: float  # 0-100
    engagement_score: float  # 0-100
    fit_score: float  # 0-100
    budget_score: float  # 0-100
    timeline_score: float  # 0-100
    breakdown: Dict[str, any]
    ai_analysis: str
    recommendation: str


class LeadScorer:
    """Advanced lead scoring engine using multiple factors and AI."""
    
    # Scoring weights (should sum to 1.0)
    WEIGHTS = {
        'company_profile': 0.20,
        'engagement': 0.25,
        'industry_fit': 0.20,
        'budget_indication': 0.15,
        'decision_timeline': 0.20,
    }
    
    # Industry firmographic data (in real system, this would be from a database)
    INDUSTRY_FIRMOGRAPHICS = {
        'technology': {'avg_budget': 500000, 'decision_speed': 'fast', 'deal_size': 'large'},
        'finance': {'avg_budget': 250000, 'decision_speed': 'slow', 'deal_size': 'medium'},
        'healthcare': {'avg_budget': 300000, 'decision_speed': 'medium', 'deal_size': 'medium'},
        'retail': {'avg_budget': 100000, 'decision_speed': 'fast', 'deal_size': 'small'},
        'manufacturing': {'avg_budget': 150000, 'decision_speed': 'slow', 'deal_size': 'medium'},
        'other': {'avg_budget': 75000, 'decision_speed': 'medium', 'deal_size': 'small'},
    }
    
    def __init__(self):
        self.orchestrator = None
    
    async def initialize(self):
        """Initialize lead scorer with AI orchestrator."""
        self.orchestrator = await get_orchestrator()
    
    async def score_lead(self, lead: Dict) -> LeadScore:
        """
        Score a lead using multiple factors and AI analysis.
        
        Args:
            lead: Lead data dictionary with fields:
                - id, company_name, contact_name, email, phone
                - industry, company_size, estimated_budget
                - engagement_level, last_interaction
                - source, decision_timeline
        
        Returns:
            LeadScore with detailed breakdown and AI analysis
        """
        try:
            # Calculate individual scores
            company_score = self._score_company_profile(lead)
            engagement_score = await self._score_engagement(lead)
            fit_score = self._score_industry_fit(lead)
            budget_score = self._score_budget_indication(lead)
            timeline_score = self._score_decision_timeline(lead)
            
            # Calculate weighted overall score
            overall_score = (
                company_score * self.WEIGHTS['company_profile'] +
                engagement_score * self.WEIGHTS['engagement'] +
                fit_score * self.WEIGHTS['industry_fit'] +
                budget_score * self.WEIGHTS['budget_indication'] +
                timeline_score * self.WEIGHTS['decision_timeline']
            )
            
            # Get AI-powered analysis and recommendation
            ai_analysis, recommendation = await self._get_ai_analysis(lead, overall_score, {
                'company_score': company_score,
                'engagement_score': engagement_score,
                'fit_score': fit_score,
                'budget_score': budget_score,
                'timeline_score': timeline_score,
            })
            
            result = LeadScore(
                lead_id=lead.get('id', 'unknown'),
                overall_score=float(overall_score),
                company_score=float(company_score),
                engagement_score=float(engagement_score),
                fit_score=float(fit_score),
                budget_score=float(budget_score),
                timeline_score=float(timeline_score),
                breakdown={
                    'company_profile': company_score,
                    'engagement': engagement_score,
                    'industry_fit': fit_score,
                    'budget_indication': budget_score,
                    'decision_timeline': timeline_score,
                },
                ai_analysis=ai_analysis,
                recommendation=recommendation,
            )
            
            logger.info(f"Scored lead {lead.get('id')}: {overall_score:.1f}")
            return result
        
        except Exception as e:
            logger.error(f"Lead scoring failed: {e}")
            raise
    
    def _score_company_profile(self, lead: Dict) -> float:
        """Score based on company profile and firmographics."""
        score = 0.0
        
        # Company size scoring
        company_size = lead.get('company_size', 'unknown').lower()
        size_scores = {
            'enterprise': 90,
            'mid-market': 75,
            'smb': 50,
            'startup': 40,
            'unknown': 30,
        }
        score += size_scores.get(company_size, 30)
        
        # Years in business scoring
        years_in_business = lead.get('years_in_business', 0)
        if years_in_business > 20:
            score += 70
        elif years_in_business > 10:
            score += 60
        elif years_in_business > 5:
            score += 40
        elif years_in_business > 0:
            score += 20
        else:
            score += 10
        
        # Revenue scoring (if available)
        annual_revenue = lead.get('annual_revenue', 0)
        if annual_revenue > 100_000_000:
            score += 80
        elif annual_revenue > 50_000_000:
            score += 70
        elif annual_revenue > 10_000_000:
            score += 60
        elif annual_revenue > 1_000_000:
            score += 40
        elif annual_revenue > 0:
            score += 20
        else:
            score += 10
        
        # Average the components
        return min(score / 3.0, 100)
    
    async def _score_engagement(self, lead: Dict) -> float:
        """Score based on engagement level and recency."""
        score = 0.0
        
        # Recent interaction scoring
        last_interaction = lead.get('last_interaction')
        if last_interaction:
            try:
                if isinstance(last_interaction, str):
                    last_interaction = datetime.fromisoformat(last_interaction)
                days_ago = (datetime.utcnow() - last_interaction).days
                
                if days_ago == 0:
                    score += 100
                elif days_ago <= 7:
                    score += 80
                elif days_ago <= 30:
                    score += 60
                elif days_ago <= 90:
                    score += 30
                else:
                    score += 10
            except:
                score += 20
        else:
            score += 10
        
        # Engagement level scoring
        engagement_level = lead.get('engagement_level', 'low').lower()
        engagement_scores = {
            'high': 90,
            'medium': 60,
            'low': 30,
            'none': 10,
        }
        score += engagement_scores.get(engagement_level, 20)
        
        # Number of interactions
        num_interactions = lead.get('num_interactions', 0)
        if num_interactions > 10:
            score += 80
        elif num_interactions > 5:
            score += 60
        elif num_interactions > 2:
            score += 40
        elif num_interactions > 0:
            score += 20
        else:
            score += 5
        
        # Response rate (if available)
        response_rate = lead.get('email_response_rate', 0)  # 0-1
        score += response_rate * 50
        
        # Average engagement components
        return min(score / 4.0, 100)
    
    def _score_industry_fit(self, lead: Dict) -> float:
        """Score based on industry match and vertical alignment."""
        score = 50  # Base score
        
        industry = lead.get('industry', 'other').lower()
        target_industries = lead.get('target_industries', [])
        
        # Direct industry match
        if industry in target_industries:
            score += 40
        
        # Use industry firmographics for scoring
        firmographics = self.INDUSTRY_FIRMOGRAPHICS.get(industry, {})
        
        # Decision speed fit
        decision_speed = firmographics.get('decision_speed', 'medium')
        sales_cycle_pref = lead.get('preferred_sales_cycle', 'medium').lower()
        
        speed_match = {
            'fast': {'fast': 1.0, 'medium': 0.7, 'slow': 0.3},
            'medium': {'fast': 0.7, 'medium': 1.0, 'slow': 0.7},
            'slow': {'fast': 0.3, 'medium': 0.7, 'slow': 1.0},
        }
        
        match_score = speed_match.get(decision_speed, {}).get(sales_cycle_pref, 0.5)
        score += match_score * 30
        
        return min(score, 100)
    
    def _score_budget_indication(self, lead: Dict) -> float:
        """Score based on budget signals and financial indicators."""
        score = 30  # Base score (conservative)
        
        # Explicit budget scoring
        estimated_budget = lead.get('estimated_budget', 0)
        if estimated_budget > 0:
            if estimated_budget > 500_000:
                score += 60
            elif estimated_budget > 250_000:
                score += 50
            elif estimated_budget > 100_000:
                score += 40
            elif estimated_budget > 50_000:
                score += 30
            else:
                score += 15
        
        # Budget explicitly stated?
        if lead.get('budget_confirmed', False):
            score += 20
        
        # Funding signals (recent funding, growth metrics)
        recent_funding = lead.get('recent_funding', False)
        if recent_funding:
            score += 25
        
        # Revenue growth indicators
        revenue_growth = lead.get('revenue_growth_rate', 0)  # Percentage
        if revenue_growth > 50:
            score += 25
        elif revenue_growth > 20:
            score += 15
        elif revenue_growth > 0:
            score += 10
        
        return min(score, 100)
    
    def _score_decision_timeline(self, lead: Dict) -> float:
        """Score based on decision timeline and urgency."""
        score = 40  # Base score
        
        timeline = lead.get('decision_timeline', 'unknown').lower()
        timeline_scores = {
            'immediate': 100,
            'this_month': 90,
            'this_quarter': 70,
            'this_year': 50,
            'next_year': 20,
            'unknown': 40,
            'no_timeline': 10,
        }
        score = timeline_scores.get(timeline, 40)
        
        # Urgency indicators
        has_pain_point = lead.get('has_active_pain_point', False)
        if has_pain_point:
            score += 20
        
        # Competitive pressure
        has_competition = lead.get('has_competitive_threat', False)
        if has_competition:
            score += 15
        
        return min(score, 100)
    
    async def _get_ai_analysis(
        self, 
        lead: Dict, 
        overall_score: float, 
        breakdown: Dict[str, float]
    ) -> Tuple[str, str]:
        """Get AI-powered analysis and recommendation using Claude."""
        try:
            prompt = f"""Analyze this sales lead and provide a brief assessment:

Lead Information:
- Company: {lead.get('company_name', 'Unknown')}
- Industry: {lead.get('industry', 'Unknown')}
- Size: {lead.get('company_size', 'Unknown')}
- Decision Timeline: {lead.get('decision_timeline', 'Unknown')}
- Budget: ${lead.get('estimated_budget', 'Unknown'):,}

Scoring Breakdown (0-100):
- Company Profile: {breakdown['company_score']:.1f}
- Engagement: {breakdown['engagement_score']:.1f}
- Industry Fit: {breakdown['fit_score']:.1f}
- Budget Indication: {breakdown['budget_score']:.1f}
- Decision Timeline: {breakdown['timeline_score']:.1f}
- Overall Score: {overall_score:.1f}

Based on this analysis, provide:
1. A 1-2 sentence summary of why this lead is valuable or risky
2. A specific recommended next action (e.g., "Schedule discovery call", "Send case study", "Wait for budget confirmation")

Format your response as:
ANALYSIS: [summary here]
RECOMMENDATION: [action here]"""
            
            response = await self.orchestrator.complete(prompt, temperature=0.7)
            
            # Parse response
            lines = response.split('\n')
            analysis = ""
            recommendation = ""
            
            current_section = None
            for line in lines:
                if 'ANALYSIS:' in line:
                    current_section = 'analysis'
                    analysis = line.replace('ANALYSIS:', '').strip()
                elif 'RECOMMENDATION:' in line:
                    current_section = 'recommendation'
                    recommendation = line.replace('RECOMMENDATION:', '').strip()
                elif current_section == 'analysis' and line.strip():
                    analysis += " " + line.strip()
                elif current_section == 'recommendation' and line.strip():
                    recommendation += " " + line.strip()
            
            return analysis or "Lead with mixed signals - needs evaluation", recommendation or "Schedule discovery call"
        
        except Exception as e:
            logger.warning(f"AI analysis failed, using defaults: {e}")
            return "Lead scoring completed", "Evaluate for next steps"
    
    async def batch_score_leads(self, leads: List[Dict]) -> List[LeadScore]:
        """Score multiple leads in parallel."""
        tasks = [self.score_lead(lead) for lead in leads]
        return await asyncio.gather(*tasks)
    
    def get_top_leads(self, leads: List[LeadScore], top_n: int = 10) -> List[LeadScore]:
        """Get top N leads by overall score."""
        return sorted(leads, key=lambda x: x.overall_score, reverse=True)[:top_n]
    
    def get_leads_by_segment(self, leads: List[LeadScore]) -> Dict[str, List[LeadScore]]:
        """Segment leads by score quality."""
        segments = {
            'hot': [],  # 80-100
            'warm': [],  # 60-79
            'cool': [],  # 40-59
            'cold': [],  # 0-39
        }
        
        for lead in leads:
            if lead.overall_score >= 80:
                segments['hot'].append(lead)
            elif lead.overall_score >= 60:
                segments['warm'].append(lead)
            elif lead.overall_score >= 40:
                segments['cool'].append(lead)
            else:
                segments['cold'].append(lead)
        
        return segments
