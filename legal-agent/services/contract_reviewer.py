"""AI-powered contract review and legal analysis service."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from shared_libs.ai_providers import get_orchestrator


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ContractClause:
    """Analyzed contract clause."""
    clause_name: str
    content: str
    risk_level: str
    issues: List[str]
    recommendations: List[str]
    is_standard: bool


@dataclass
class ContractReview:
    """Contract review result."""
    contract_id: str
    contract_type: str
    overall_risk_level: str
    risk_score: float  # 0-100
    total_issues: int
    critical_issues: List[str]
    key_clauses: List[ContractClause]
    missing_clauses: List[str]
    compliance_issues: List[str]
    recommendations: List[str]
    summary: str
    review_timestamp: datetime


class ContractReviewer:
    """AI-powered contract review engine."""
    
    # Standard contract clauses by type
    STANDARD_CLAUSES = {
        "service_agreement": [
            "Scope of Work",
            "Payment Terms",
            "Confidentiality",
            "Limitation of Liability",
            "Termination",
            "Intellectual Property Rights",
            "Warranty Disclaimer",
        ],
        "nda": [
            "Definition of Confidential Information",
            "Permitted Use",
            "Non-Disclosure Obligations",
            "Return of Information",
            "Exceptions to Confidentiality",
            "Term and Survival",
        ],
        "employee_agreement": [
            "Job Description",
            "Compensation",
            "At-Will Employment",
            "Confidentiality",
            "Non-Compete",
            "Intellectual Property Assignment",
            "Benefits",
        ],
        "vendor_agreement": [
            "Services/Products",
            "Payment Terms",
            "Insurance Requirements",
            "Indemnification",
            "Termination",
            "Data Protection",
            "Performance Standards",
        ],
    }
    
    # Red flag terms and phrases
    RED_FLAG_TERMS = {
        "unlimited liability": "high",
        "indemnify all claims": "high",
        "non-exclusive": "medium",
        "perpetual": "medium",
        "irrevocable": "medium",
        "force majeure": "low",
        "binding arbitration": "medium",
        "severability": "low",
    }
    
    def __init__(self):
        self.orchestrator = None
    
    async def initialize(self):
        """Initialize contract reviewer with AI."""
        self.orchestrator = await get_orchestrator()
    
    async def review_contract(
        self,
        contract_id: str,
        contract_text: str,
        contract_type: str,
        applicable_regulations: Optional[List[str]] = None
    ) -> ContractReview:
        """
        Review a contract for legal risks, missing clauses, and compliance issues.
        
        Args:
            contract_id: Unique contract identifier
            contract_text: Full contract text
            contract_type: Type of contract (service_agreement, nda, etc.)
            applicable_regulations: List of applicable regulations/compliance standards
        
        Returns:
            ContractReview with detailed analysis and recommendations
        """
        try:
            # Get AI-powered analysis
            ai_analysis, risk_level = await self._analyze_contract_with_ai(
                contract_text,
                contract_type,
                applicable_regulations or []
            )
            
            # Identify key clauses and issues
            key_clauses = await self._identify_key_clauses(contract_text, contract_type)
            
            # Find missing standard clauses
            missing_clauses = self._find_missing_clauses(contract_text, contract_type)
            
            # Check for compliance issues
            compliance_issues = self._check_compliance(contract_text, applicable_regulations or [])
            
            # Extract specific issues and recommendations
            critical_issues, all_recommendations = self._extract_issues_and_recommendations(
                ai_analysis
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                risk_level,
                len(critical_issues),
                len(missing_clauses),
                len(compliance_issues)
            )
            
            # Generate summary
            summary = await self._generate_summary(
                contract_type,
                risk_level,
                critical_issues,
                missing_clauses
            )
            
            review = ContractReview(
                contract_id=contract_id,
                contract_type=contract_type,
                overall_risk_level=risk_level,
                risk_score=risk_score,
                total_issues=len(critical_issues) + len(missing_clauses) + len(compliance_issues),
                critical_issues=critical_issues,
                key_clauses=key_clauses,
                missing_clauses=missing_clauses,
                compliance_issues=compliance_issues,
                recommendations=all_recommendations,
                summary=summary,
                review_timestamp=datetime.utcnow()
            )
            
            logger.info(f"Contract {contract_id} reviewed - Risk Level: {risk_level}")
            return review
            
        except Exception as e:
            logger.error(f"Contract review failed: {e}")
            raise
    
    async def _analyze_contract_with_ai(
        self,
        contract_text: str,
        contract_type: str,
        regulations: List[str]
    ) -> Tuple[str, str]:
        """Get AI-powered contract analysis."""
        try:
            regulations_text = f"\nApplicable Regulations: {', '.join(regulations)}" if regulations else ""
            
            prompt = f"""Provide a detailed legal analysis of this {contract_type} contract.{regulations_text}

CONTRACT TEXT:
---
{contract_text[:3000]}
---

Analyze for:
1. UNFAVORABLE TERMS: List any terms heavily favoring one party
2. LIABILITY ISSUES: Identify liability and indemnification concerns
3. INTELLECTUAL PROPERTY: Check IP ownership and usage rights
4. TERMINATION RISKS: Assess termination clauses and exit difficulties
5. COMPLIANCE: Flag any compliance concerns
6. MISSING PROTECTIONS: Identify missing protective clauses

Format your response as:
RISK_LEVEL: [low/medium/high/critical]
UNFAVORABLE_TERMS: [list separated by |]
LIABILITY_ISSUES: [list separated by |]
IP_ISSUES: [list separated by |]
TERMINATION_RISKS: [list separated by |]
COMPLIANCE_CONCERNS: [list separated by |]
MISSING_PROTECTIONS: [list separated by |]
SUMMARY: [brief 2-3 sentence summary]"""
            
            response = await self.orchestrator.complete(prompt, temperature=0.5)
            
            # Parse risk level
            risk_level = "medium"
            for line in response.split('\n'):
                if 'RISK_LEVEL:' in line:
                    level = line.split(':')[1].strip().lower()
                    if level in ["low", "medium", "high", "critical"]:
                        risk_level = level
                    break
            
            return response, risk_level
            
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return "Analysis unavailable", "medium"
    
    async def _identify_key_clauses(
        self,
        contract_text: str,
        contract_type: str
    ) -> List[ContractClause]:
        """Identify and analyze key clauses."""
        try:
            key_clauses = []
            standard_clauses = self.STANDARD_CLAUSES.get(contract_type, [])
            
            # Find important clauses in text
            for clause_name in standard_clauses[:5]:  # Top 5 clauses
                if clause_name.lower() in contract_text.lower():
                    # Extract clause content (simplified)
                    start_idx = contract_text.lower().find(clause_name.lower())
                    content = contract_text[start_idx:min(start_idx + 500, len(contract_text))]
                    
                    clause = ContractClause(
                        clause_name=clause_name,
                        content=content[:200],
                        risk_level="medium",
                        issues=[],
                        recommendations=[],
                        is_standard=True
                    )
                    key_clauses.append(clause)
            
            return key_clauses
            
        except Exception as e:
            logger.warning(f"Clause identification failed: {e}")
            return []
    
    def _find_missing_clauses(self, contract_text: str, contract_type: str) -> List[str]:
        """Find missing standard clauses."""
        missing = []
        standard_clauses = self.STANDARD_CLAUSES.get(contract_type, [])
        
        for clause in standard_clauses:
            if clause.lower() not in contract_text.lower():
                missing.append(clause)
        
        return missing[:5]  # Return top 5 missing
    
    def _check_compliance(self, contract_text: str, regulations: List[str]) -> List[str]:
        """Check compliance with regulations."""
        issues = []
        
        if not regulations:
            return issues
        
        # Check for GDPR compliance
        if "gdpr" in [r.lower() for r in regulations]:
            if "personal data" not in contract_text.lower() and "data protection" not in contract_text.lower():
                issues.append("Missing GDPR personal data protection clause")
            if "data processing" not in contract_text.lower():
                issues.append("No clear data processing agreement for GDPR")
        
        # Check for CCPA compliance
        if "ccpa" in [r.lower() for r in regulations]:
            if "california consumer" not in contract_text.lower():
                issues.append("Missing CCPA consumer rights acknowledgment")
        
        return issues
    
    def _extract_issues_and_recommendations(self, analysis: str) -> Tuple[List[str], List[str]]:
        """Extract critical issues and recommendations from analysis."""
        critical_issues = []
        recommendations = []
        
        try:
            lines = analysis.split('\n')
            for line in lines:
                if 'UNFAVORABLE_TERMS:' in line or 'LIABILITY_ISSUES:' in line:
                    items = line.split(':')[1].strip().split('|')
                    critical_issues.extend([i.strip() for i in items if i.strip()])
                
                if 'RECOMMENDATIONS' in line or 'SUMMARY:' in line:
                    items = line.split(':')[1].strip().split('|')
                    recommendations.extend([r.strip() for r in items if r.strip()])
            
            return critical_issues[:5], recommendations[:5]  # Top 5 each
            
        except:
            return [], []
    
    def _calculate_risk_score(
        self,
        risk_level: str,
        critical_count: int,
        missing_count: int,
        compliance_count: int
    ) -> float:
        """Calculate overall risk score 0-100."""
        base_score = {
            "low": 20,
            "medium": 50,
            "high": 75,
            "critical": 95,
        }.get(risk_level, 50)
        
        # Adjust for specific issues
        score = base_score
        score += min(critical_count * 5, 20)  # Up to 20 points for critical issues
        score += min(missing_count * 3, 15)   # Up to 15 points for missing clauses
        score += min(compliance_count * 5, 20)  # Up to 20 points for compliance
        
        return min(score, 100.0)
    
    async def _generate_summary(
        self,
        contract_type: str,
        risk_level: str,
        critical_issues: List[str],
        missing_clauses: List[str]
    ) -> str:
        """Generate contract review summary."""
        try:
            issues_text = ", ".join(critical_issues[:3]) if critical_issues else "None"
            missing_text = ", ".join(missing_clauses[:2]) if missing_clauses else "None"
            
            prompt = f"""Provide a 2-3 sentence executive summary of a {contract_type} contract review.
Risk Level: {risk_level}
Critical Issues: {issues_text}
Missing Clauses: {missing_text}

Make it concise and actionable for a business decision-maker."""
            
            summary = await self.orchestrator.complete(prompt, temperature=0.6)
            return summary.strip()
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return f"This {contract_type} has {risk_level} risk with {len(critical_issues)} critical issues."
