"""AI-powered infrastructure incident analysis and root cause analysis service."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from shared_libs.ai_providers import get_orchestrator


class IncidentSeverity(str, Enum):
    """Incident severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IncidentStatus(str, Enum):
    """Incident status."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class RootCauseAnalysis:
    """Root cause analysis result."""
    probable_causes: List[str]
    likelihood_scores: Dict[str, float]  # cause -> confidence 0-1
    contributing_factors: List[str]
    timeline: List[Dict[str, Any]]
    pattern_analysis: str
    recommendations: List[str]


@dataclass
class IncidentAnalysis:
    """Complete incident analysis."""
    incident_id: str
    service_name: str
    severity: str
    status: str
    detected_at: datetime
    impact_summary: str
    affected_systems: List[str]
    user_impact: Dict[str, Any]
    root_cause_analysis: RootCauseAnalysis
    immediate_actions: List[str]
    prevention_measures: List[str]
    estimated_resolution_time: int  # minutes
    analysis_timestamp: datetime


class IncidentAnalyzer:
    """AI-powered incident analysis engine for infrastructure."""
    
    # Service impact mappings
    SERVICE_CRITICALITY = {
        "auth_service": "critical",
        "api_gateway": "critical",
        "database": "critical",
        "cache_layer": "high",
        "notification_service": "medium",
        "logging_service": "medium",
        "analytics": "low",
    }
    
    # Known incident patterns
    INCIDENT_PATTERNS = {
        "memory_leak": ["high_memory", "memory_growth", "oom"],
        "database_issue": ["connection_timeout", "query_slow", "deadlock"],
        "deployment_failure": ["service_down", "unhealthy", "crash"],
        "network_issue": ["latency_high", "connection_refused", "packet_loss"],
        "load_issue": ["cpu_high", "response_time_up", "queue_length_up"],
    }
    
    def __init__(self):
        self.orchestrator = None
    
    async def initialize(self):
        """Initialize analyzer with AI."""
        self.orchestrator = await get_orchestrator()
    
    async def analyze_incident(
        self,
        incident_id: str,
        service_name: str,
        symptoms: List[str],
        metrics: Dict[str, Any],
        timeline: Optional[List[Dict[str, Any]]] = None,
        recent_changes: Optional[List[str]] = None
    ) -> IncidentAnalysis:
        """
        Analyze infrastructure incident to determine root cause and resolution.
        
        Args:
            incident_id: Unique incident identifier
            service_name: Name of affected service
            symptoms: List of symptoms observed
            metrics: Current system metrics
            timeline: Historical timeline of events
            recent_changes: Recent changes/deployments
        
        Returns:
            IncidentAnalysis with detailed RCA and recommendations
        """
        try:
            # Determine severity
            severity = self._determine_severity(service_name, symptoms, metrics)
            
            # Detect likely pattern
            pattern = self._detect_pattern(symptoms)
            
            # Get AI-powered RCA
            probable_causes, contributing_factors, recommendations = await self._perform_rca(
                service_name,
                symptoms,
                metrics,
                timeline or [],
                recent_changes or [],
                pattern
            )
            
            # Calculate impact
            impact_summary, affected_systems, user_impact = self._calculate_impact(
                service_name,
                severity,
                symptoms,
                metrics
            )
            
            # Generate immediate actions
            immediate_actions = await self._generate_immediate_actions(
                service_name,
                probable_causes,
                severity
            )
            
            # Estimate resolution time
            est_time = self._estimate_resolution_time(probable_causes, severity)
            
            # Root cause analysis object
            likelihood_scores = {cause: 0.8 - (i * 0.15) for i, cause in enumerate(probable_causes)}
            
            rca = RootCauseAnalysis(
                probable_causes=probable_causes,
                likelihood_scores=likelihood_scores,
                contributing_factors=contributing_factors,
                timeline=timeline or [],
                pattern_analysis=f"Detected pattern: {pattern}" if pattern else "No pattern match",
                recommendations=recommendations
            )
            
            analysis = IncidentAnalysis(
                incident_id=incident_id,
                service_name=service_name,
                severity=severity,
                status=IncidentStatus.INVESTIGATING.value,
                detected_at=datetime.utcnow(),
                impact_summary=impact_summary,
                affected_systems=affected_systems,
                user_impact=user_impact,
                root_cause_analysis=rca,
                immediate_actions=immediate_actions,
                prevention_measures=self._generate_prevention_measures(probable_causes),
                estimated_resolution_time=est_time,
                analysis_timestamp=datetime.utcnow()
            )
            
            logger.info(f"Incident {incident_id} analyzed - Severity: {severity}")
            return analysis
            
        except Exception as e:
            logger.error(f"Incident analysis failed: {e}")
            raise
    
    def _determine_severity(
        self,
        service_name: str,
        symptoms: List[str],
        metrics: Dict[str, Any]
    ) -> str:
        """Determine incident severity."""
        
        # Start with service criticality
        base_severity = self.SERVICE_CRITICALITY.get(service_name.lower(), "medium")
        severity_level = {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(base_severity, 2)
        
        # Critical symptoms
        critical_symptoms = ["service_down", "data_loss", "security_breach", "complete_outage"]
        if any(sym in symptoms for sym in critical_symptoms):
            severity_level = 4
        
        # Check metrics severity
        if metrics.get("cpu_percent", 0) > 95 or metrics.get("memory_percent", 0) > 95:
            severity_level = max(severity_level, 3)
        
        if metrics.get("error_rate", 0) > 10:  # >10% error rate
            severity_level = max(severity_level, 3)
        
        if metrics.get("response_time_ms", 0) > 5000:  # >5 seconds
            severity_level = max(severity_level, 2)
        
        severity_map = {4: "critical", 3: "high", 2: "medium", 1: "low"}
        return severity_map.get(severity_level, "medium")
    
    def _detect_pattern(self, symptoms: List[str]) -> Optional[str]:
        """Detect known incident pattern."""
        symptoms_lower = [s.lower() for s in symptoms]
        
        for pattern_name, pattern_keywords in self.INCIDENT_PATTERNS.items():
            if any(keyword in symptom for symptom in symptoms_lower for keyword in pattern_keywords):
                return pattern_name
        
        return None
    
    async def _perform_rca(
        self,
        service_name: str,
        symptoms: List[str],
        metrics: Dict[str, Any],
        timeline: List[Dict[str, Any]],
        recent_changes: List[str],
        pattern: Optional[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Perform root cause analysis using AI."""
        try:
            timeline_text = "\n".join([f"  {t['time']}: {t['event']}" for t in timeline[:5]])
            changes_text = "\n".join([f"  - {c}" for c in recent_changes[:3]])
            
            prompt = f"""Analyze this infrastructure incident and suggest root causes.

SERVICE: {service_name}
SYMPTOMS: {', '.join(symptoms)}
PATTERN: {pattern or 'Unknown'}

CURRENT METRICS:
- CPU: {metrics.get('cpu_percent', 'N/A')}%
- Memory: {metrics.get('memory_percent', 'N/A')}%
- Error Rate: {metrics.get('error_rate', 'N/A')}%
- Response Time: {metrics.get('response_time_ms', 'N/A')}ms

TIMELINE:
{timeline_text or '  No timeline events'}

RECENT CHANGES:
{changes_text or '  No recent changes'}

Provide analysis in this format:

PROBABLE_CAUSES: [List 3-4 likely root causes separated by |]

CONTRIBUTING_FACTORS: [List factors that made issue worse separated by |]

IMMEDIATE_ACTIONS: [List 2-3 immediate remediation steps separated by |]

RECOMMENDATIONS: [List 3-4 fix recommendations separated by |]"""
            
            response = await self.orchestrator.complete(prompt, temperature=0.6)
            
            probable_causes = []
            contributing_factors = []
            recommendations = []
            
            lines = response.split('\n')
            for line in lines:
                if 'PROBABLE_CAUSES:' in line:
                    causes_text = line.split(':')[1].strip()
                    probable_causes = [c.strip() for c in causes_text.split('|') if c.strip()]
                
                elif 'CONTRIBUTING_FACTORS:' in line:
                    factors_text = line.split(':')[1].strip()
                    contributing_factors = [f.strip() for f in factors_text.split('|') if f.strip()]
                
                elif 'RECOMMENDATIONS:' in line:
                    recs_text = line.split(':')[1].strip()
                    recommendations = [r.strip() for r in recs_text.split('|') if r.strip()]
            
            return probable_causes or ["Service misconfiguration"], contributing_factors, recommendations or ["Restart service", "Check logs"]
            
        except Exception as e:
            logger.warning(f"RCA failed: {e}")
            return ["Unknown cause"], [], ["Review service logs", "Check recent deployments"]
    
    def _calculate_impact(
        self,
        service_name: str,
        severity: str,
        symptoms: List[str],
        metrics: Dict[str, Any]
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """Calculate incident impact."""
        
        # Determine affected systems
        affected = {
            "auth_service": ["api_gateway", "all_services"],
            "database": ["all_services"],
            "cache_layer": ["api_gateway", "search_service"],
            "api_gateway": ["client_apps"],
        }
        affected_systems = affected.get(service_name.lower(), [service_name])
        
        # Calculate impact
        if severity == "critical":
            impact_text = "Complete service outage affecting all users"
            user_count = "All active users"
            error_rate = 100
        elif severity == "high":
            impact_text = "Severe degradation affecting majority of users"
            user_count = "80-90% of users"
            error_rate = metrics.get("error_rate", 50)
        else:
            impact_text = "Partial degradation affecting some users"
            user_count = "10-30% of users"
            error_rate = metrics.get("error_rate", 20)
        
        user_impact = {
            "affected_users": user_count,
            "error_rate": error_rate,
            "avg_response_time_ms": metrics.get("response_time_ms", 0),
            "unable_to_complete_transactions": severity in ["critical", "high"]
        }
        
        return impact_text, affected_systems, user_impact
    
    async def _generate_immediate_actions(
        self,
        service_name: str,
        probable_causes: List[str],
        severity: str
    ) -> List[str]:
        """Generate immediate actions to mitigate incident."""
        
        actions = []
        
        if severity in ["critical", "high"]:
            # Always restart affected service
            actions.append(f"1. Prepare to restart {service_name} service")
            actions.append("2. Isolate affected instances from load balancer")
        
        # Action based on probable causes
        if probable_causes:
            cause = probable_causes[0].lower()
            
            if "memory" in cause or "leak" in cause:
                actions.append("3. Check memory usage and restart if needed")
                actions.append("4. Review application logs for memory leaks")
            
            elif "database" in cause:
                actions.append("3. Verify database connectivity")
                actions.append("4. Check database locks and slow queries")
            
            elif "cpu" in cause or "load" in cause:
                actions.append("3. Identify high-CPU processes")
                actions.append("4. Scale up instances if needed")
            
            elif "deployment" in cause:
                actions.append("3. Rollback recent deployment")
                actions.append("4. Verify previous version stability")
        
        return actions[:4]
    
    def _estimate_resolution_time(self, probable_causes: List[str], severity: str) -> int:
        """Estimate time to resolution in minutes."""
        
        # Base time on severity
        base_time = {
            "critical": 30,
            "high": 60,
            "medium": 120,
            "low": 240,
        }.get(severity, 120)
        
        # Adjust based on cause clarity
        if probable_causes:
            cause = probable_causes[0].lower()
            if "restart" in cause or "simple" in cause:
                base_time = int(base_time * 0.5)
            elif "database" in cause or "complex" in cause:
                base_time = int(base_time * 1.5)
        
        return base_time
    
    def _generate_prevention_measures(self, probable_causes: List[str]) -> List[str]:
        """Generate prevention measures for future."""
        
        measures = [
            "Set up alerting for early warning",
            "Add load testing to deployment process",
            "Implement circuit breakers",
        ]
        
        if probable_causes and "memory" in probable_causes[0].lower():
            measures.append("Add memory monitoring and auto-restart thresholds")
        
        if probable_causes and "database" in probable_causes[0].lower():
            measures.append("Implement connection pooling and query optimization")
        
        return measures[:4]
