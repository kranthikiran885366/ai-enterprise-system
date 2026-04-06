"""QA and testing service with bug pattern recognition and test analytics."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from shared_libs.ai_providers import get_orchestrator


class BugSeverity(str, Enum):
    """Bug severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"


class TestStatus(str, Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class BugCluster:
    """Clustered similar bugs."""
    cluster_id: str
    bug_ids: List[str]
    root_cause: str
    affected_component: str
    pattern: str
    similar_bugs_count: int
    estimated_fix_effort: str  # low, medium, high
    recommended_action: str


@dataclass
class TestMetrics:
    """Test execution metrics."""
    total_tests: int
    passed: int
    failed: int
    skipped: int
    blocked: int
    pass_rate: float
    flaky_tests: List[str]
    critical_failures: List[str]
    code_coverage: float
    execution_time_ms: int


class QATestAnalyzer:
    """QA and testing analysis engine."""
    
    # Bug pattern keywords
    BUG_PATTERNS = {
        "ui_rendering": ["ui", "layout", "display", "render", "visual"],
        "data_validation": ["validation", "input", "format", "data", "incorrect"],
        "performance": ["slow", "timeout", "lag", "performance", "response_time"],
        "security": ["security", "authentication", "authorization", "sql_injection", "xss"],
        "database": ["database", "query", "deadlock", "connection", "transaction"],
        "integration": ["api", "integration", "external", "third_party", "webhook"],
        "concurrency": ["race_condition", "thread", "parallel", "concurrent", "deadlock"],
    }
    
    def __init__(self):
        self.orchestrator = None
    
    async def initialize(self):
        """Initialize QA analyzer with AI."""
        self.orchestrator = await get_orchestrator()
    
    async def analyze_bugs(
        self,
        bugs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze reported bugs for patterns and clustering.
        
        Args:
            bugs: List of bug reports with descriptions and metadata
        
        Returns:
            Analysis with clusters, patterns, and recommendations
        """
        try:
            clusters = await self._cluster_similar_bugs(bugs)
            pattern_analysis = self._analyze_patterns(bugs)
            recommendations = await self._generate_recommendations(clusters)
            
            return {
                "total_bugs": len(bugs),
                "clusters": clusters,
                "patterns": pattern_analysis,
                "recommendations": recommendations,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Bug analysis failed: {e}")
            raise
    
    async def _cluster_similar_bugs(self, bugs: List[Dict[str, Any]]) -> List[BugCluster]:
        """Cluster similar bugs together."""
        try:
            clusters = []
            processed = set()
            
            for i, bug in enumerate(bugs):
                if bug.get("id") in processed:
                    continue
                
                # Find similar bugs
                similar_bugs = [bug]
                similar_ids = [bug.get("id")]
                
                for j, other_bug in enumerate(bugs[i+1:], i+1):
                    if other_bug.get("id") in processed:
                        continue
                    
                    if await self._bugs_are_similar(bug, other_bug):
                        similar_bugs.append(other_bug)
                        similar_ids.append(other_bug.get("id"))
                        processed.add(other_bug.get("id"))
                
                # Create cluster if more than 1 similar bug
                if len(similar_bugs) > 1:
                    cluster = await self._create_cluster(similar_bugs)
                    clusters.append(cluster)
                    processed.add(bug.get("id"))
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Bug clustering failed: {e}")
            return []
    
    async def _bugs_are_similar(self, bug1: Dict[str, Any], bug2: Dict[str, Any]) -> bool:
        """Determine if two bugs are similar."""
        try:
            # Check component match
            if bug1.get("component") == bug2.get("component"):
                return True
            
            # Check description similarity using AI
            prompt = f"""Are these two bug reports describing the same issue?

Bug 1: {bug1.get('description', '')}
Bug 2: {bug2.get('description', '')}

Respond with only 'yes' or 'no'."""
            
            response = await self.orchestrator.complete(prompt, temperature=0.3)
            return "yes" in response.lower()
            
        except:
            return False
    
    async def _create_cluster(self, bugs: List[Dict[str, Any]]) -> BugCluster:
        """Create a bug cluster."""
        try:
            descriptions = "\n".join([f"- {b.get('description')}" for b in bugs[:3]])
            
            prompt = f"""Analyze these similar bugs and identify the root cause:

{descriptions}

Respond with:
ROOT_CAUSE: [one sentence]
PATTERN: [bug pattern type]
FIX_EFFORT: [low/medium/high]"""
            
            response = await self.orchestrator.complete(prompt, temperature=0.5)
            
            root_cause = "Unknown"
            pattern = "general"
            effort = "medium"
            
            for line in response.split('\n'):
                if 'ROOT_CAUSE:' in line:
                    root_cause = line.split(':')[1].strip()[:100]
                elif 'PATTERN:' in line:
                    pattern = line.split(':')[1].strip().lower()
                elif 'FIX_EFFORT:' in line:
                    effort = line.split(':')[1].strip().lower()
            
            cluster = BugCluster(
                cluster_id=f"CLUST{len(bugs)}{datetime.utcnow().timestamp()}",
                bug_ids=[b.get("id") for b in bugs],
                root_cause=root_cause,
                affected_component=bugs[0].get("component", "unknown"),
                pattern=pattern,
                similar_bugs_count=len(bugs),
                estimated_fix_effort=effort,
                recommended_action=f"Group and fix together - {effort} effort"
            )
            
            return cluster
            
        except Exception as e:
            logger.warning(f"Cluster creation failed: {e}")
            return None
    
    def _analyze_patterns(self, bugs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze bug patterns."""
        patterns = {}
        
        for bug in bugs:
            description = bug.get("description", "").lower()
            
            for pattern_name, keywords in self.BUG_PATTERNS.items():
                if any(keyword in description for keyword in keywords):
                    patterns[pattern_name] = patterns.get(pattern_name, 0) + 1
        
        # Sort by count
        return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))
    
    async def _generate_recommendations(self, clusters: List[BugCluster]) -> List[str]:
        """Generate recommendations."""
        recommendations = []
        
        if not clusters:
            return ["Continue normal testing process", "Review regression tests"]
        
        # High effort clusters should be prioritized
        high_effort = [c for c in clusters if c.estimated_fix_effort == "high"]
        if high_effort:
            recommendations.append(f"Prioritize {len(high_effort)} high-effort clusters for team discussion")
        
        # Suggest testing improvements
        patterns = {}
        for cluster in clusters:
            patterns[cluster.pattern] = patterns.get(cluster.pattern, 0) + 1
        
        top_pattern = max(patterns.items(), key=lambda x: x[1]) if patterns else None
        if top_pattern:
            recommendations.append(f"Add more {top_pattern[0]} test cases")
        
        recommendations.append("Set up automated regression tests for bug clusters")
        
        return recommendations[:4]
    
    async def analyze_test_execution(
        self,
        test_results: Dict[str, Any],
        previous_results: Optional[Dict[str, Any]] = None
    ) -> TestMetrics:
        """
        Analyze test execution results.
        
        Args:
            test_results: Current test execution results
            previous_results: Previous test results for comparison
        
        Returns:
            TestMetrics with analysis
        """
        try:
            total = test_results.get("total", 0)
            passed = test_results.get("passed", 0)
            failed = test_results.get("failed", 0)
            skipped = test_results.get("skipped", 0)
            blocked = test_results.get("blocked", 0)
            
            pass_rate = (passed / total * 100) if total > 0 else 0
            code_coverage = test_results.get("coverage_percent", 0)
            execution_time = test_results.get("execution_time_ms", 0)
            
            # Identify flaky tests
            flaky = self._identify_flaky_tests(test_results, previous_results)
            
            # Identify critical failures
            critical = self._identify_critical_failures(test_results)
            
            metrics = TestMetrics(
                total_tests=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                blocked=blocked,
                pass_rate=round(pass_rate, 1),
                flaky_tests=flaky,
                critical_failures=critical,
                code_coverage=code_coverage,
                execution_time_ms=execution_time
            )
            
            logger.info(f"Test execution analyzed - Pass Rate: {pass_rate:.1f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Test analysis failed: {e}")
            raise
    
    def _identify_flaky_tests(
        self,
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify flaky tests."""
        if not previous:
            return []
        
        flaky = []
        
        # Tests that failed before but passed now, or vice versa
        current_failed = set(current.get("failed_tests", []))
        previous_failed = set(previous.get("failed_tests", []))
        
        # Tests that inconsistently fail
        inconsistent = current_failed.symmetric_difference(previous_failed)
        
        return list(inconsistent)[:5]
    
    def _identify_critical_failures(self, test_results: Dict[str, Any]) -> List[str]:
        """Identify critical test failures."""
        critical = []
        
        for test in test_results.get("failed_tests", []):
            if any(keyword in test.lower() for keyword in ["auth", "payment", "critical", "regression"]):
                critical.append(test)
        
        return critical[:5]
    
    async def generate_test_cases(
        self,
        feature_description: str,
        existing_tests: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate test cases for a feature using AI."""
        try:
            prompt = f"""Generate comprehensive test cases for this feature:

{feature_description}

Existing Tests: {', '.join(existing_tests) if existing_tests else 'None'}

For each test case, provide:
1. Test Name
2. Preconditions
3. Test Steps
4. Expected Result

Format as numbered list with clear step-by-step instructions."""
            
            response = await self.orchestrator.complete(prompt, temperature=0.7)
            
            # Parse test cases (simplified)
            test_cases = []
            current_test = {}
            
            for line in response.split('\n'):
                if line.startswith(('1.', '2.', '3.', '4.')):
                    if current_test:
                        test_cases.append(current_test)
                    current_test = {"description": line.strip()}
                elif current_test and line.strip():
                    current_test["description"] += "\n" + line.strip()
            
            if current_test:
                test_cases.append(current_test)
            
            logger.info(f"Generated {len(test_cases)} test cases")
            return test_cases
            
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return []
