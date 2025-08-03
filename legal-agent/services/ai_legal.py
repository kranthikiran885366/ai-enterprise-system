"""AI-powered legal services for Legal Agent."""

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


class AILegalService:
    """AI-powered legal analysis and compliance service."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.contract_analysis_collection = "contract_analysis"
        self.risk_assessments_collection = "risk_assessments"
        self.compliance_monitoring_collection = "compliance_monitoring"
    
    async def initialize(self):
        """Initialize the AI legal service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.contract_analysis_collection].create_index("contract_id")
        await self.db[self.contract_analysis_collection].create_index("risk_score")
        await self.db[self.contract_analysis_collection].create_index("created_at")
        
        await self.db[self.risk_assessments_collection].create_index("assessment_id", unique=True)
        await self.db[self.risk_assessments_collection].create_index("entity_type")
        await self.db[self.risk_assessments_collection].create_index("risk_level")
        
        await self.db[self.compliance_monitoring_collection].create_index("monitoring_id", unique=True)
        await self.db[self.compliance_monitoring_collection].create_index("regulation_type")
        
        logger.info("AI Legal service initialized")
    
    async def analyze_contract_risk(self, contract_id: str, contract_text: str) -> Dict[str, Any]:
        """Analyze contract for legal risks using AI."""
        try:
            nlp = await get_nlp_processor()
            data_lake = await get_data_lake()
            
            # Extract key clauses and terms
            key_clauses = await self._extract_contract_clauses(contract_text)
            
            # Analyze contract language
            language_analysis = await nlp.analyze_sentiment(contract_text)
            
            # Risk factor analysis
            risk_factors = await self._identify_risk_factors(contract_text, key_clauses)
            
            # Compliance check
            compliance_issues = await self._check_contract_compliance(contract_text, key_clauses)
            
            # Calculate overall risk score
            risk_score = self._calculate_contract_risk_score(risk_factors, compliance_issues, language_analysis)
            
            # Generate recommendations
            recommendations = await self._generate_contract_recommendations(risk_factors, compliance_issues)
            
            analysis_result = {
                "analysis_id": f"CA_{str(uuid.uuid4())[:8].upper()}",
                "contract_id": contract_id,
                "risk_score": risk_score,
                "risk_level": self._determine_risk_level(risk_score),
                "key_clauses": key_clauses,
                "risk_factors": risk_factors,
                "compliance_issues": compliance_issues,
                "language_analysis": language_analysis,
                "recommendations": recommendations,
                "requires_legal_review": risk_score > 0.6,
                "created_at": datetime.utcnow()
            }
            
            # Store analysis
            await self.db[self.contract_analysis_collection].insert_one(analysis_result)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="legal",
                event_type="contract_analyzed",
                entity_type="contract",
                entity_id=contract_id,
                data={
                    "risk_score": risk_score,
                    "risk_level": self._determine_risk_level(risk_score),
                    "requires_review": risk_score > 0.6
                }
            )
            
            logger.info(f"Contract risk analyzed: {contract_id}, risk_score={risk_score:.3f}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze contract risk: {e}")
            return {}
    
    async def _extract_contract_clauses(self, contract_text: str) -> List[Dict[str, Any]]:
        """Extract key clauses from contract text."""
        try:
            # Common contract clause patterns
            clause_patterns = {
                "termination": ["termination", "terminate", "end agreement"],
                "liability": ["liability", "damages", "indemnify", "hold harmless"],
                "payment": ["payment", "invoice", "billing", "fees"],
                "confidentiality": ["confidential", "non-disclosure", "proprietary"],
                "intellectual_property": ["intellectual property", "copyright", "trademark", "patent"],
                "force_majeure": ["force majeure", "act of god", "unforeseeable"],
                "governing_law": ["governing law", "jurisdiction", "disputes"],
                "warranty": ["warranty", "guarantee", "representation"]
            }
            
            identified_clauses = []
            
            for clause_type, keywords in clause_patterns.items():
                for keyword in keywords:
                    if keyword.lower() in contract_text.lower():
                        # Find the sentence containing the keyword
                        sentences = contract_text.split('.')
                        for sentence in sentences:
                            if keyword.lower() in sentence.lower():
                                identified_clauses.append({
                                    "clause_type": clause_type,
                                    "keyword": keyword,
                                    "text_snippet": sentence.strip()[:200] + "..." if len(sentence) > 200 else sentence.strip(),
                                    "risk_level": self._assess_clause_risk(clause_type, sentence)
                                })
                                break
                        break
            
            return identified_clauses
            
        except Exception as e:
            logger.error(f"Failed to extract contract clauses: {e}")
            return []
    
    def _assess_clause_risk(self, clause_type: str, clause_text: str) -> str:
        """Assess risk level of a specific clause."""
        high_risk_clauses = ["liability", "termination", "governing_law"]
        medium_risk_clauses = ["payment", "warranty", "intellectual_property"]
        
        if clause_type in high_risk_clauses:
            return "high"
        elif clause_type in medium_risk_clauses:
            return "medium"
        else:
            return "low"
    
    async def _identify_risk_factors(self, contract_text: str, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential risk factors in the contract."""
        try:
            risk_factors = []
            
            # Check for missing standard clauses
            standard_clauses = ["termination", "liability", "payment", "governing_law"]
            present_clauses = [clause["clause_type"] for clause in clauses]
            
            for standard_clause in standard_clauses:
                if standard_clause not in present_clauses:
                    risk_factors.append({
                        "type": "missing_clause",
                        "description": f"Missing {standard_clause} clause",
                        "severity": "medium",
                        "recommendation": f"Add {standard_clause} clause to protect interests"
                    })
            
            # Check for unfavorable terms
            unfavorable_terms = [
                "unlimited liability",
                "no warranty",
                "immediate termination",
                "non-refundable"
            ]
            
            for term in unfavorable_terms:
                if term.lower() in contract_text.lower():
                    risk_factors.append({
                        "type": "unfavorable_term",
                        "description": f"Potentially unfavorable term: {term}",
                        "severity": "high",
                        "recommendation": f"Review and negotiate {term} clause"
                    })
            
            # Check for vague language
            vague_terms = ["reasonable", "best efforts", "as soon as possible", "appropriate"]
            vague_count = sum(1 for term in vague_terms if term.lower() in contract_text.lower())
            
            if vague_count > 3:
                risk_factors.append({
                    "type": "vague_language",
                    "description": f"Contract contains {vague_count} vague terms",
                    "severity": "medium",
                    "recommendation": "Define vague terms more specifically"
                })
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Failed to identify risk factors: {e}")
            return []
    
    async def _check_contract_compliance(self, contract_text: str, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check contract for compliance issues."""
        try:
            compliance_issues = []
            
            # GDPR compliance check
            if "personal data" in contract_text.lower() or "data processing" in contract_text.lower():
                gdpr_clauses = ["data protection", "privacy", "consent", "data subject rights"]
                gdpr_present = any(clause.lower() in contract_text.lower() for clause in gdpr_clauses)
                
                if not gdpr_present:
                    compliance_issues.append({
                        "regulation": "GDPR",
                        "issue": "Missing data protection clauses",
                        "severity": "high",
                        "requirement": "Include GDPR compliance clauses for data processing"
                    })
            
            # Employment law compliance
            if "employment" in contract_text.lower() or "employee" in contract_text.lower():
                employment_clauses = ["equal opportunity", "discrimination", "harassment"]
                employment_compliance = any(clause.lower() in contract_text.lower() for clause in employment_clauses)
                
                if not employment_compliance:
                    compliance_issues.append({
                        "regulation": "Employment Law",
                        "issue": "Missing employment protection clauses",
                        "severity": "medium",
                        "requirement": "Include equal opportunity and anti-discrimination clauses"
                    })
            
            return compliance_issues
            
        except Exception as e:
            logger.error(f"Failed to check contract compliance: {e}")
            return []
    
    def _calculate_contract_risk_score(self, risk_factors: List[Dict[str, Any]], 
                                     compliance_issues: List[Dict[str, Any]], 
                                     language_analysis: Dict[str, Any]) -> float:
        """Calculate overall contract risk score."""
        try:
            risk_score = 0.0
            
            # Risk factors contribution
            for factor in risk_factors:
                severity = factor.get("severity", "low")
                if severity == "high":
                    risk_score += 0.3
                elif severity == "medium":
                    risk_score += 0.2
                else:
                    risk_score += 0.1
            
            # Compliance issues contribution
            for issue in compliance_issues:
                severity = issue.get("severity", "low")
                if severity == "high":
                    risk_score += 0.25
                elif severity == "medium":
                    risk_score += 0.15
                else:
                    risk_score += 0.1
            
            # Language sentiment contribution
            sentiment = language_analysis.get("classification", "neutral")
            if sentiment == "negative":
                risk_score += 0.1
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate contract risk score: {e}")
            return 0.5
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score."""
        if risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    async def _generate_contract_recommendations(self, risk_factors: List[Dict[str, Any]], 
                                               compliance_issues: List[Dict[str, Any]]) -> List[str]:
        """Generate contract recommendations."""
        try:
            recommendations = []
            
            # Add recommendations from risk factors
            for factor in risk_factors:
                recommendations.append(factor.get("recommendation", "Review this risk factor"))
            
            # Add recommendations from compliance issues
            for issue in compliance_issues:
                recommendations.append(issue.get("requirement", "Address this compliance issue"))
            
            # General recommendations
            recommendations.extend([
                "Have contract reviewed by qualified legal counsel",
                "Ensure all parties understand their obligations",
                "Set up contract monitoring for key dates and milestones"
            ])
            
            return list(set(recommendations))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to generate contract recommendations: {e}")
            return []
    
    async def monitor_regulatory_compliance(self, regulation_type: str) -> Dict[str, Any]:
        """Monitor compliance with specific regulations."""
        try:
            # Get compliance requirements for regulation
            requirements = await self._get_compliance_requirements(regulation_type)
            
            # Check current compliance status
            compliance_status = await self._assess_current_compliance(regulation_type, requirements)
            
            # Identify gaps
            compliance_gaps = await self._identify_compliance_gaps(regulation_type, compliance_status)
            
            # Generate remediation plan
            remediation_plan = await self._generate_remediation_plan(compliance_gaps)
            
            monitoring_result = {
                "monitoring_id": f"MON_{str(uuid.uuid4())[:8].upper()}",
                "regulation_type": regulation_type,
                "compliance_status": compliance_status,
                "compliance_gaps": compliance_gaps,
                "remediation_plan": remediation_plan,
                "overall_compliance_score": self._calculate_compliance_score(compliance_status),
                "next_review_date": datetime.utcnow() + timedelta(days=90),
                "created_at": datetime.utcnow()
            }
            
            # Store monitoring result
            await self.db[self.compliance_monitoring_collection].insert_one(monitoring_result)
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Failed to monitor regulatory compliance: {e}")
            return {}
    
    async def _get_compliance_requirements(self, regulation_type: str) -> List[Dict[str, Any]]:
        """Get compliance requirements for a regulation."""
        # Simplified compliance requirements
        requirements_map = {
            "gdpr": [
                {"requirement": "Data Protection Officer appointed", "mandatory": True},
                {"requirement": "Privacy policy published", "mandatory": True},
                {"requirement": "Data processing records maintained", "mandatory": True},
                {"requirement": "Breach notification procedures", "mandatory": True}
            ],
            "sox": [
                {"requirement": "Internal controls documented", "mandatory": True},
                {"requirement": "Financial reporting controls", "mandatory": True},
                {"requirement": "Management assessment", "mandatory": True}
            ],
            "hipaa": [
                {"requirement": "Privacy policies implemented", "mandatory": True},
                {"requirement": "Security safeguards in place", "mandatory": True},
                {"requirement": "Employee training completed", "mandatory": True}
            ]
        }
        
        return requirements_map.get(regulation_type.lower(), [])
    
    async def _assess_current_compliance(self, regulation_type: str, 
                                       requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess current compliance status."""
        try:
            # Get existing compliance checks
            compliance_checks = await self.db["compliance_checks"].find({
                "regulation_type": regulation_type
            }).to_list(None)
            
            status = {
                "total_requirements": len(requirements),
                "compliant_requirements": 0,
                "non_compliant_requirements": 0,
                "requirements_status": {}
            }
            
            for requirement in requirements:
                req_name = requirement["requirement"]
                
                # Check if we have a compliance check for this requirement
                check = next((c for c in compliance_checks if req_name.lower() in c.get("regulation_name", "").lower()), None)
                
                if check and check.get("status") == "compliant":
                    status["compliant_requirements"] += 1
                    status["requirements_status"][req_name] = "compliant"
                else:
                    status["non_compliant_requirements"] += 1
                    status["requirements_status"][req_name] = "non_compliant"
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to assess current compliance: {e}")
            return {}
    
    async def _identify_compliance_gaps(self, regulation_type: str, 
                                      compliance_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify compliance gaps."""
        try:
            gaps = []
            
            requirements_status = compliance_status.get("requirements_status", {})
            
            for requirement, status in requirements_status.items():
                if status == "non_compliant":
                    gaps.append({
                        "requirement": requirement,
                        "gap_type": "missing_implementation",
                        "priority": "high" if "mandatory" in requirement.lower() else "medium",
                        "estimated_effort": "medium"
                    })
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to identify compliance gaps: {e}")
            return []
    
    async def _generate_remediation_plan(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate remediation plan for compliance gaps."""
        try:
            remediation_actions = []
            
            for gap in gaps:
                requirement = gap["requirement"]
                priority = gap.get("priority", "medium")
                
                action = {
                    "action": f"Implement {requirement}",
                    "priority": priority,
                    "estimated_timeline": "30 days" if priority == "high" else "60 days",
                    "responsible_party": "compliance_team",
                    "resources_needed": ["legal_review", "policy_development"]
                }
                
                remediation_actions.append(action)
            
            return remediation_actions
            
        except Exception as e:
            logger.error(f"Failed to generate remediation plan: {e}")
            return []
    
    def _calculate_compliance_score(self, compliance_status: Dict[str, Any]) -> float:
        """Calculate overall compliance score."""
        try:
            total = compliance_status.get("total_requirements", 0)
            compliant = compliance_status.get("compliant_requirements", 0)
            
            if total == 0:
                return 0.0
            
            return round((compliant / total), 3)
            
        except Exception as e:
            logger.error(f"Failed to calculate compliance score: {e}")
            return 0.0