"""AI-powered marketing services for Marketing Agent."""

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
from shared_libs.intelligence import get_nlp_processor
from shared_libs.data_lake import get_data_lake

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY", "")


class AIMarketingService:
    """AI-powered marketing automation and optimization service."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.content_optimization_collection = "content_optimization"
        self.audience_analysis_collection = "audience_analysis"
        self.campaign_predictions_collection = "campaign_predictions"
        self.ab_test_results_collection = "ab_test_results"
    
    async def initialize(self):
        """Initialize the AI marketing service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.content_optimization_collection].create_index("content_id")
        await self.db[self.content_optimization_collection].create_index("optimization_score")
        await self.db[self.content_optimization_collection].create_index("created_at")
        
        await self.db[self.audience_analysis_collection].create_index("analysis_id", unique=True)
        await self.db[self.audience_analysis_collection].create_index("campaign_id")
        
        await self.db[self.campaign_predictions_collection].create_index("prediction_id", unique=True)
        await self.db[self.campaign_predictions_collection].create_index("campaign_id")
        
        await self.db[self.ab_test_results_collection].create_index("test_id", unique=True)
        await self.db[self.ab_test_results_collection].create_index("campaign_id")
        
        logger.info("AI Marketing service initialized")
    
    async def optimize_content(self, content_id: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize marketing content using AI."""
        try:
            nlp = await get_nlp_processor()
            data_lake = await get_data_lake()
            
            content_body = content_data.get("content_body", "")
            content_type = content_data.get("content_type", "blog_post")
            target_audience = content_data.get("target_audience", {})
            
            # Analyze content sentiment and readability
            sentiment_analysis = await nlp.analyze_sentiment(content_body)
            keywords = await nlp.extract_keywords(content_body, 15)
            
            # Generate optimization suggestions
            optimization_suggestions = await self._generate_content_optimizations(
                content_body, content_type, sentiment_analysis, keywords
            )
            
            # AI-powered content improvements
            improved_content = await self._improve_content_with_ai(content_body, content_type, target_audience)
            
            # SEO optimization
            seo_analysis = await self._analyze_seo_potential(content_body, keywords)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(
                sentiment_analysis, seo_analysis, optimization_suggestions
            )
            
            optimization_result = {
                "optimization_id": f"OPT_{str(uuid.uuid4())[:8].upper()}",
                "content_id": content_id,
                "original_content": content_body,
                "improved_content": improved_content,
                "optimization_score": optimization_score,
                "sentiment_analysis": sentiment_analysis,
                "keywords": keywords,
                "optimization_suggestions": optimization_suggestions,
                "seo_analysis": seo_analysis,
                "target_audience": target_audience,
                "created_at": datetime.utcnow()
            }
            
            # Store optimization
            await self.db[self.content_optimization_collection].insert_one(optimization_result)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="marketing",
                event_type="content_optimized",
                entity_type="content",
                entity_id=content_id,
                data={
                    "optimization_score": optimization_score,
                    "improvements_count": len(optimization_suggestions),
                    "content_type": content_type
                }
            )
            
            logger.info(f"Content optimized: {content_id}, score={optimization_score:.3f}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Failed to optimize content: {e}")
            return {}
    
    async def _generate_content_optimizations(self, content: str, content_type: str, 
                                            sentiment: Dict[str, Any], keywords: List[str]) -> List[Dict[str, Any]]:
        """Generate content optimization suggestions."""
        try:
            suggestions = []
            
            # Sentiment-based suggestions
            if sentiment.get("classification") == "negative":
                suggestions.append({
                    "type": "tone_adjustment",
                    "suggestion": "Consider using more positive language to improve engagement",
                    "priority": "high"
                })
            
            # Length-based suggestions
            word_count = len(content.split())
            if content_type == "blog_post":
                if word_count < 300:
                    suggestions.append({
                        "type": "length_optimization",
                        "suggestion": "Blog posts perform better with 300+ words for SEO",
                        "priority": "medium"
                    })
                elif word_count > 2000:
                    suggestions.append({
                        "type": "length_optimization",
                        "suggestion": "Consider breaking this into multiple posts for better readability",
                        "priority": "low"
                    })
            elif content_type == "email_template":
                if word_count > 200:
                    suggestions.append({
                        "type": "length_optimization",
                        "suggestion": "Email content should be concise (under 200 words)",
                        "priority": "high"
                    })
            
            # Keyword density suggestions
            if keywords:
                keyword_density = len(keywords) / word_count if word_count > 0 else 0
                if keyword_density < 0.01:
                    suggestions.append({
                        "type": "keyword_optimization",
                        "suggestion": "Add more relevant keywords to improve discoverability",
                        "priority": "medium"
                    })
            
            # Call-to-action suggestions
            cta_keywords = ["click", "buy", "subscribe", "download", "register", "learn more"]
            has_cta = any(keyword in content.lower() for keyword in cta_keywords)
            if not has_cta:
                suggestions.append({
                    "type": "call_to_action",
                    "suggestion": "Add a clear call-to-action to improve conversion rates",
                    "priority": "high"
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate content optimizations: {e}")
            return []
    
    async def _improve_content_with_ai(self, content: str, content_type: str, 
                                     target_audience: Dict[str, Any]) -> str:
        """Improve content using AI."""
        try:
            if not openai.api_key:
                return content  # Return original if no AI available
            
            audience_description = target_audience.get("description", "general business audience")
            
            prompt = f"""
            Improve the following {content_type} content for {audience_description}.
            Make it more engaging, clear, and persuasive while maintaining the original message.
            
            Original content:
            {content}
            
            Improved content:
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            improved_content = response.choices[0].message.content.strip()
            return improved_content
            
        except Exception as e:
            logger.error(f"Failed to improve content with AI: {e}")
            return content
    
    async def _analyze_seo_potential(self, content: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyze SEO potential of content."""
        try:
            word_count = len(content.split())
            
            # Basic SEO analysis
            seo_score = 0.0
            seo_factors = []
            
            # Word count factor
            if 300 <= word_count <= 2000:
                seo_score += 0.3
                seo_factors.append("Good word count for SEO")
            
            # Keyword density
            if keywords:
                keyword_mentions = sum(content.lower().count(keyword.lower()) for keyword in keywords)
                keyword_density = keyword_mentions / word_count if word_count > 0 else 0
                
                if 0.01 <= keyword_density <= 0.03:  # 1-3% keyword density
                    seo_score += 0.3
                    seo_factors.append("Good keyword density")
                elif keyword_density > 0.03:
                    seo_factors.append("Keyword density too high (keyword stuffing)")
                else:
                    seo_factors.append("Keyword density too low")
            
            # Readability (simplified)
            sentences = content.split('.')
            avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0
            
            if avg_sentence_length <= 20:
                seo_score += 0.2
                seo_factors.append("Good readability (short sentences)")
            
            # Structure analysis
            if content.count('\n') > 3:  # Has paragraphs
                seo_score += 0.2
                seo_factors.append("Good content structure")
            
            return {
                "seo_score": round(seo_score, 3),
                "seo_factors": seo_factors,
                "word_count": word_count,
                "keyword_density": round(keyword_density, 4) if 'keyword_density' in locals() else 0,
                "avg_sentence_length": round(avg_sentence_length, 1)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze SEO potential: {e}")
            return {"seo_score": 0.5}
    
    def _calculate_optimization_score(self, sentiment: Dict[str, Any], seo: Dict[str, Any], 
                                    suggestions: List[Dict[str, Any]]) -> float:
        """Calculate overall optimization score."""
        try:
            score = 0.0
            
            # Sentiment contribution (30%)
            if sentiment.get("classification") == "positive":
                score += 0.3
            elif sentiment.get("classification") == "neutral":
                score += 0.2
            
            # SEO contribution (40%)
            score += seo.get("seo_score", 0) * 0.4
            
            # Optimization potential (30%)
            # Fewer suggestions = better current state
            suggestion_penalty = min(len(suggestions) * 0.05, 0.3)
            score += (0.3 - suggestion_penalty)
            
            return round(min(score, 1.0), 3)
            
        except Exception as e:
            logger.error(f"Failed to calculate optimization score: {e}")
            return 0.5
    
    async def predict_campaign_performance(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict campaign performance using historical data and AI."""
        try:
            campaign_type = campaign_data.get("campaign_type", "email")
            budget = campaign_data.get("budget", 0)
            target_audience = campaign_data.get("target_audience", {})
            
            # Get historical campaign data
            historical_campaigns = await self._get_historical_campaigns(campaign_type, limit=50)
            
            # Analyze historical performance
            historical_analysis = await self._analyze_historical_performance(historical_campaigns)
            
            # Predict metrics based on budget and audience
            predicted_metrics = await self._predict_campaign_metrics(
                campaign_type, budget, target_audience, historical_analysis
            )
            
            # Calculate confidence based on historical data availability
            confidence = min(0.9, len(historical_campaigns) / 20) if historical_campaigns else 0.3
            
            prediction_result = {
                "prediction_id": f"PRED_{str(uuid.uuid4())[:8].upper()}",
                "campaign_type": campaign_type,
                "predicted_metrics": predicted_metrics,
                "confidence": confidence,
                "historical_analysis": historical_analysis,
                "recommendations": await self._generate_campaign_recommendations(predicted_metrics, historical_analysis),
                "created_at": datetime.utcnow()
            }
            
            # Store prediction
            await self.db[self.campaign_predictions_collection].insert_one(prediction_result)
            
            logger.info(f"Campaign performance predicted: {prediction_result['prediction_id']}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Failed to predict campaign performance: {e}")
            return {}
    
    async def _get_historical_campaigns(self, campaign_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get historical campaigns of the same type."""
        try:
            campaigns = await self.db["campaigns"].find({
                "campaign_type": campaign_type,
                "status": "completed",
                "metrics": {"$exists": True}
            }).limit(limit).to_list(None)
            
            return campaigns
            
        except Exception as e:
            logger.error(f"Failed to get historical campaigns: {e}")
            return []
    
    async def _analyze_historical_performance(self, campaigns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze historical campaign performance."""
        try:
            if not campaigns:
                return {"avg_roi": 0, "avg_engagement": 0, "success_rate": 0}
            
            # Calculate averages
            total_roi = 0
            total_engagement = 0
            successful_campaigns = 0
            
            for campaign in campaigns:
                metrics = campaign.get("metrics", {})
                
                roi = metrics.get("roi", 0)
                engagement = metrics.get("engagement_rate", 0)
                
                total_roi += roi
                total_engagement += engagement
                
                if roi > 2.0:  # ROI > 200%
                    successful_campaigns += 1
            
            return {
                "avg_roi": round(total_roi / len(campaigns), 2),
                "avg_engagement": round(total_engagement / len(campaigns), 2),
                "success_rate": round((successful_campaigns / len(campaigns)) * 100, 2),
                "campaigns_analyzed": len(campaigns)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze historical performance: {e}")
            return {"avg_roi": 0, "avg_engagement": 0, "success_rate": 0}
    
    async def _predict_campaign_metrics(self, campaign_type: str, budget: float, 
                                      target_audience: Dict[str, Any], 
                                      historical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predict campaign metrics."""
        try:
            base_metrics = {
                "email": {
                    "open_rate": 0.25,
                    "click_rate": 0.03,
                    "conversion_rate": 0.02,
                    "cost_per_lead": 50
                },
                "social_media": {
                    "engagement_rate": 0.05,
                    "click_rate": 0.02,
                    "conversion_rate": 0.015,
                    "cost_per_lead": 75
                },
                "paid_ads": {
                    "click_rate": 0.04,
                    "conversion_rate": 0.025,
                    "cost_per_click": 2.5,
                    "cost_per_lead": 100
                }
            }
            
            base = base_metrics.get(campaign_type, base_metrics["email"])
            
            # Adjust based on budget
            budget_factor = min(1.5, budget / 10000) if budget > 0 else 1.0
            
            # Adjust based on historical performance
            historical_factor = historical_analysis.get("avg_roi", 2.0) / 2.0
            
            # Adjust based on audience size
            audience_size = target_audience.get("size", 1000)
            audience_factor = min(1.2, audience_size / 10000)
            
            predicted_metrics = {}
            for metric, value in base.items():
                adjusted_value = value * budget_factor * historical_factor * audience_factor
                predicted_metrics[metric] = round(adjusted_value, 4)
            
            # Calculate predicted leads and revenue
            if budget > 0:
                predicted_leads = int(budget / predicted_metrics.get("cost_per_lead", 100))
                predicted_revenue = predicted_leads * 500  # Assume $500 average customer value
                predicted_roi = (predicted_revenue / budget) if budget > 0 else 0
                
                predicted_metrics.update({
                    "predicted_leads": predicted_leads,
                    "predicted_revenue": round(predicted_revenue, 2),
                    "predicted_roi": round(predicted_roi, 2)
                })
            
            return predicted_metrics
            
        except Exception as e:
            logger.error(f"Failed to predict campaign metrics: {e}")
            return {}
    
    async def _generate_campaign_recommendations(self, predicted_metrics: Dict[str, Any], 
                                               historical_analysis: Dict[str, Any]) -> List[str]:
        """Generate campaign recommendations."""
        try:
            recommendations = []
            
            # ROI-based recommendations
            predicted_roi = predicted_metrics.get("predicted_roi", 0)
            if predicted_roi < 1.5:
                recommendations.append("Consider reducing budget or improving targeting to increase ROI")
            elif predicted_roi > 5.0:
                recommendations.append("High ROI predicted - consider increasing budget to scale")
            
            # Engagement-based recommendations
            engagement_rate = predicted_metrics.get("engagement_rate", 0)
            if engagement_rate < 0.02:
                recommendations.append("Low engagement predicted - improve content quality and targeting")
            
            # Historical comparison
            historical_roi = historical_analysis.get("avg_roi", 0)
            if predicted_roi < historical_roi * 0.8:
                recommendations.append("Performance below historical average - review campaign strategy")
            
            # General best practices
            recommendations.extend([
                "A/B test subject lines and content variations",
                "Segment audience for personalized messaging",
                "Monitor performance and optimize in real-time"
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate campaign recommendations: {e}")
            return []
    
    async def analyze_audience_segments(self, campaign_id: str) -> Dict[str, Any]:
        """Analyze and segment audience for better targeting."""
        try:
            # Get campaign data
            campaign = await self.db["campaigns"].find_one({"campaign_id": campaign_id})
            if not campaign:
                return {"error": "Campaign not found"}
            
            # Get lead data for analysis
            leads = await self.db["marketing_leads"].find({
                "source": campaign_id
            }).to_list(None)
            
            if not leads:
                return {"message": "No leads data available for segmentation"}
            
            # Analyze segments
            segments = await self._create_audience_segments(leads)
            
            # Generate targeting recommendations
            targeting_recommendations = await self._generate_targeting_recommendations(segments)
            
            analysis_result = {
                "analysis_id": f"AUD_{str(uuid.uuid4())[:8].upper()}",
                "campaign_id": campaign_id,
                "segments": segments,
                "targeting_recommendations": targeting_recommendations,
                "total_leads_analyzed": len(leads),
                "created_at": datetime.utcnow()
            }
            
            # Store analysis
            await self.db[self.audience_analysis_collection].insert_one(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze audience segments: {e}")
            return {}
    
    async def _create_audience_segments(self, leads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create audience segments from lead data."""
        try:
            segments = {
                "by_company_size": {},
                "by_industry": {},
                "by_job_title": {},
                "by_engagement": {}
            }
            
            for lead in leads:
                # Company size segmentation
                company_size = lead.get("company_size", "unknown")
                if company_size != "unknown":
                    if company_size < 50:
                        size_segment = "small"
                    elif company_size < 500:
                        size_segment = "medium"
                    else:
                        size_segment = "enterprise"
                    
                    segments["by_company_size"][size_segment] = segments["by_company_size"].get(size_segment, 0) + 1
                
                # Industry segmentation
                industry = lead.get("industry", "unknown")
                segments["by_industry"][industry] = segments["by_industry"].get(industry, 0) + 1
                
                # Job title segmentation
                job_title = lead.get("job_title", "unknown")
                if job_title != "unknown":
                    if any(title in job_title.lower() for title in ["ceo", "founder", "president"]):
                        title_segment = "executive"
                    elif any(title in job_title.lower() for title in ["manager", "director", "head"]):
                        title_segment = "management"
                    elif any(title in job_title.lower() for title in ["engineer", "developer", "analyst"]):
                        title_segment = "technical"
                    else:
                        title_segment = "other"
                    
                    segments["by_job_title"][title_segment] = segments["by_job_title"].get(title_segment, 0) + 1
                
                # Engagement segmentation
                interactions = lead.get("interactions", [])
                interaction_count = len(interactions)
                
                if interaction_count > 5:
                    engagement_segment = "high"
                elif interaction_count > 2:
                    engagement_segment = "medium"
                else:
                    engagement_segment = "low"
                
                segments["by_engagement"][engagement_segment] = segments["by_engagement"].get(engagement_segment, 0) + 1
            
            return segments
            
        except Exception as e:
            logger.error(f"Failed to create audience segments: {e}")
            return {}
    
    async def _generate_targeting_recommendations(self, segments: Dict[str, Any]) -> List[str]:
        """Generate targeting recommendations based on segments."""
        try:
            recommendations = []
            
            # Company size recommendations
            company_segments = segments.get("by_company_size", {})
            if company_segments:
                largest_segment = max(company_segments.items(), key=lambda x: x[1])
                recommendations.append(f"Focus on {largest_segment[0]} companies - they represent your largest segment")
            
            # Industry recommendations
            industry_segments = segments.get("by_industry", {})
            if industry_segments:
                top_industries = sorted(industry_segments.items(), key=lambda x: x[1], reverse=True)[:3]
                industries = [industry[0] for industry in top_industries]
                recommendations.append(f"Target these top industries: {', '.join(industries)}")
            
            # Engagement recommendations
            engagement_segments = segments.get("by_engagement", {})
            high_engagement = engagement_segments.get("high", 0)
            total_leads = sum(engagement_segments.values())
            
            if high_engagement / total_leads < 0.2:  # Less than 20% high engagement
                recommendations.append("Improve content quality to increase engagement rates")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate targeting recommendations: {e}")
            return []