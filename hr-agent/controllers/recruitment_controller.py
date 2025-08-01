"""Recruitment controller with AI-powered features."""

from typing import List, Optional, Dict, Any
from fastapi import HTTPException, status
from loguru import logger

from services.ai_recruitment import AIRecruitmentService
from utils.validators import validate_job_posting, validate_application
from utils.notifications import send_recruitment_notification


class RecruitmentController:
    """Controller for recruitment operations."""
    
    def __init__(self, ai_recruitment: AIRecruitmentService):
        self.ai_recruitment = ai_recruitment
    
    async def create_job_posting(self, job_data: dict, current_user: dict) -> dict:
        """Create job posting with AI-powered optimization."""
        try:
            # Validate job posting data
            validation_result = await validate_job_posting(job_data)
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Validation failed: {validation_result.errors}"
                )
            
            # AI-optimize job description
            optimized_description = await self._optimize_job_description(job_data)
            job_data["description"] = optimized_description
            
            # Create job posting
            job_id = f"JOB{str(hash(job_data['title']))[:8].upper()}"
            job_posting = {
                "job_id": job_id,
                "created_by": current_user["sub"],
                "status": "open",
                **job_data
            }
            
            # Store in database (simplified for demo)
            logger.info(f"Job posting created: {job_id}")
            
            return {
                "message": "Job posting created successfully",
                "job_id": job_id,
                "title": job_data["title"],
                "optimizations_applied": True
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to create job posting: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def submit_application(self, application_data: dict, current_user: dict) -> dict:
        """Submit job application with AI screening."""
        try:
            # Validate application data
            validation_result = await validate_application(application_data)
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Validation failed: {validation_result.errors}"
                )
            
            # Create candidate record
            candidate_id = f"CAND{str(hash(application_data['candidate_email']))[:8].upper()}"
            
            # AI-powered resume analysis
            if application_data.get("resume_text") and application_data.get("job_description"):
                resume_analysis = await self.ai_recruitment.analyze_resume(
                    candidate_id,
                    application_data["resume_text"],
                    application_data["job_description"]
                )
                
                # Auto-shortlist high-scoring candidates
                if resume_analysis.get("overall_score", 0) >= 0.75:
                    await self._auto_shortlist_candidate(candidate_id, resume_analysis)
            
            # Send confirmation to candidate
            await send_recruitment_notification(
                application_data["candidate_email"],
                "application_received",
                {
                    "candidate_name": application_data["candidate_name"],
                    "job_title": application_data.get("job_title", "Position"),
                    "application_id": candidate_id
                }
            )
            
            logger.info(f"Application submitted: {candidate_id}")
            
            return {
                "message": "Application submitted successfully",
                "application_id": candidate_id,
                "candidate_name": application_data["candidate_name"],
                "ai_screening_completed": True
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to submit application: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def conduct_ai_interview(self, candidate_id: str, interview_type: str, 
                                 current_user: dict) -> dict:
        """Conduct AI-powered interview."""
        try:
            # Define interview questions based on type
            questions = self._get_interview_questions(interview_type)
            
            # Conduct AI interview
            interview_result = await self.ai_recruitment.conduct_ai_interview(
                candidate_id,
                interview_type,
                questions
            )
            
            if not interview_result:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to conduct AI interview"
                )
            
            # Send interview completion notification
            await send_recruitment_notification(
                "hr@company.com",  # HR team notification
                "interview_completed",
                {
                    "candidate_id": candidate_id,
                    "interview_type": interview_type,
                    "overall_score": interview_result.get("overall_score", 0),
                    "recommendation": interview_result.get("recommendation", "review")
                }
            )
            
            return {
                "message": "AI interview completed successfully",
                "interview_id": interview_result.get("interview_id"),
                "overall_score": interview_result.get("overall_score", 0),
                "recommendation": interview_result.get("recommendation", "review"),
                "next_steps": self._get_next_steps(interview_result.get("recommendation", "review"))
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to conduct AI interview: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def get_recruitment_analytics(self, days: int, current_user: dict) -> dict:
        """Get recruitment analytics and insights."""
        try:
            analytics = await self.ai_recruitment.get_recruitment_analytics(days)
            
            if not analytics:
                return {"message": "No analytics data available"}
            
            return {
                "analytics": analytics,
                "insights": self._generate_recruitment_insights(analytics),
                "recommendations": self._generate_recruitment_recommendations(analytics)
            }
            
        except Exception as e:
            logger.error(f"Failed to get recruitment analytics: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def _optimize_job_description(self, job_data: dict) -> str:
        """AI-optimize job description for better candidate attraction."""
        try:
            original_description = job_data.get("description", "")
            
            # Simple optimization rules (in production, use AI)
            optimizations = []
            
            # Add inclusive language
            if "rockstar" in original_description.lower():
                optimizations.append("Replaced 'rockstar' with 'talented professional'")
                original_description = original_description.replace("rockstar", "talented professional")
            
            # Add benefits mention
            if "benefits" not in original_description.lower():
                original_description += "\n\nWe offer competitive benefits including health insurance, retirement plans, and professional development opportunities."
                optimizations.append("Added benefits information")
            
            # Add growth opportunities
            if "growth" not in original_description.lower():
                original_description += "\n\nJoin our team and grow your career with mentorship and advancement opportunities."
                optimizations.append("Added career growth information")
            
            logger.info(f"Job description optimized: {optimizations}")
            return original_description
            
        except Exception as e:
            logger.error(f"Failed to optimize job description: {e}")
            return job_data.get("description", "")
    
    async def _auto_shortlist_candidate(self, candidate_id: str, analysis: dict) -> None:
        """Auto-shortlist high-scoring candidates."""
        try:
            # Send notification to hiring manager
            await send_recruitment_notification(
                "hiring-manager@company.com",
                "candidate_shortlisted",
                {
                    "candidate_id": candidate_id,
                    "score": analysis.get("overall_score", 0),
                    "recommendation": analysis.get("recommendation", "review"),
                    "strengths": analysis.get("strengths", [])
                }
            )
            
            logger.info(f"Candidate auto-shortlisted: {candidate_id}")
            
        except Exception as e:
            logger.error(f"Failed to auto-shortlist candidate: {e}")
    
    def _get_interview_questions(self, interview_type: str) -> List[str]:
        """Get interview questions based on type."""
        question_sets = {
            "technical": [
                "Describe your experience with the main technologies listed in the job description.",
                "Walk me through how you would approach solving a complex technical problem.",
                "Tell me about a challenging project you worked on and how you overcame obstacles.",
                "How do you stay updated with the latest technologies in your field?",
                "Describe your experience with version control and collaborative development."
            ],
            "behavioral": [
                "Tell me about a time when you had to work with a difficult team member.",
                "Describe a situation where you had to meet a tight deadline.",
                "Give me an example of when you had to learn something new quickly.",
                "Tell me about a time when you made a mistake and how you handled it.",
                "Describe your ideal work environment and team dynamics."
            ],
            "cultural": [
                "What motivates you in your work?",
                "How do you handle feedback and criticism?",
                "Describe a time when you went above and beyond in your role.",
                "What are your long-term career goals?",
                "Why are you interested in working for our company?"
            ]
        }
        
        return question_sets.get(interview_type, question_sets["behavioral"])
    
    def _get_next_steps(self, recommendation: str) -> List[str]:
        """Get next steps based on interview recommendation."""
        next_steps = {
            "strong_hire": [
                "Schedule final interview with hiring manager",
                "Prepare job offer",
                "Conduct reference checks"
            ],
            "hire": [
                "Schedule additional technical interview",
                "Conduct team fit assessment",
                "Check references"
            ],
            "maybe": [
                "Schedule follow-up interview",
                "Assess specific skill gaps",
                "Compare with other candidates"
            ],
            "no_hire": [
                "Send polite rejection email",
                "Provide constructive feedback if requested",
                "Keep candidate in talent pool for future opportunities"
            ]
        }
        
        return next_steps.get(recommendation, next_steps["maybe"])
    
    def _generate_recruitment_insights(self, analytics: dict) -> List[str]:
        """Generate insights from recruitment analytics."""
        insights = []
        
        if analytics.get("avg_resume_score", 0) < 0.6:
            insights.append("Average resume quality is below expectations - consider improving job posting clarity")
        
        if analytics.get("interview_to_hire_rate", 0) < 0.3:
            insights.append("Low interview-to-hire conversion rate - review interview process")
        
        if analytics.get("recommended_hires", 0) > analytics.get("ai_interviews_conducted", 1) * 0.8:
            insights.append("High AI recommendation rate - consider adjusting screening criteria")
        
        return insights
    
    def _generate_recruitment_recommendations(self, analytics: dict) -> List[str]:
        """Generate recommendations from recruitment analytics."""
        recommendations = []
        
        if analytics.get("total_candidates", 0) < 50:
            recommendations.append("Increase candidate sourcing efforts")
        
        if analytics.get("resume_to_interview_rate", 0) < 0.2:
            recommendations.append("Review resume screening criteria")
        
        recommendations.extend([
            "Implement candidate feedback collection",
            "Optimize job posting keywords",
            "Enhance employer branding"
        ])
        
        return recommendations