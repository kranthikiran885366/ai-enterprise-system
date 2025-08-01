"""AI-powered testing services for QA Agent."""

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
from collections import defaultdict
import ast

from shared_libs.database import get_database
from shared_libs.intelligence import get_nlp_processor
from shared_libs.data_lake import get_data_lake

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY", "")


class AITestingService:
    """AI-powered testing and quality assurance service."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.test_cases_collection = "ai_test_cases"
        self.test_execution_collection = "test_executions"
        self.coverage_analysis_collection = "coverage_analysis"
        self.regression_analysis_collection = "regression_analysis"
        self.test_failure_clusters_collection = "test_failure_clusters"
    
    async def initialize(self):
        """Initialize the AI testing service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.test_cases_collection].create_index("test_case_id", unique=True)
        await self.db[self.test_cases_collection].create_index("story_id")
        await self.db[self.test_cases_collection].create_index("created_at")
        
        await self.db[self.test_execution_collection].create_index("execution_id", unique=True)
        await self.db[self.test_execution_collection].create_index("test_suite_id")
        await self.db[self.test_execution_collection].create_index("executed_at")
        
        await self.db[self.coverage_analysis_collection].create_index("analysis_id", unique=True)
        await self.db[self.coverage_analysis_collection].create_index("project_id")
        
        await self.db[self.regression_analysis_collection].create_index("analysis_id", unique=True)
        await self.db[self.regression_analysis_collection].create_index("build_id")
        
        await self.db[self.test_failure_clusters_collection].create_index("cluster_id", unique=True)
        await self.db[self.test_failure_clusters_collection].create_index("created_at")
        
        logger.info("AI Testing service initialized")
    
    async def generate_ai_test_cases(self, story_id: str, story_data: Dict[str, Any], 
                                   test_types: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive AI test cases for a user story."""
        try:
            data_lake = await get_data_lake()
            
            if test_types is None:
                test_types = ["unit", "integration", "e2e", "performance", "security"]
            
            # Analyze story requirements
            requirements_analysis = await self._analyze_story_requirements(story_data)
            
            # Generate test cases for each type
            generated_test_cases = {}
            
            for test_type in test_types:
                test_cases = await self._generate_test_cases_by_type(
                    story_data, requirements_analysis, test_type
                )
                generated_test_cases[test_type] = test_cases
            
            # Generate test data requirements
            test_data_requirements = await self._generate_test_data_requirements(
                story_data, requirements_analysis
            )
            
            # Generate automation suggestions
            automation_suggestions = await self._generate_automation_suggestions(
                generated_test_cases, story_data
            )
            
            # Calculate test coverage estimation
            coverage_estimation = await self._estimate_test_coverage(
                generated_test_cases, requirements_analysis
            )
            
            # Generate test execution strategy
            execution_strategy = await self._generate_test_execution_strategy(
                generated_test_cases, story_data
            )
            
            test_generation_result = {
                "generation_id": f"TG_{str(uuid.uuid4())[:8].upper()}",
                "story_id": story_id,
                "requirements_analysis": requirements_analysis,
                "generated_test_cases": generated_test_cases,
                "test_data_requirements": test_data_requirements,
                "automation_suggestions": automation_suggestions,
                "coverage_estimation": coverage_estimation,
                "execution_strategy": execution_strategy,
                "total_test_cases": sum(len(cases) for cases in generated_test_cases.values()),
                "estimated_execution_time": self._calculate_execution_time(generated_test_cases),
                "created_at": datetime.utcnow()
            }
            
            # Store test cases
            await self.db[self.test_cases_collection].insert_one(test_generation_result)
            
            # Store individual test cases for tracking
            await self._store_individual_test_cases(story_id, generated_test_cases, test_generation_result["generation_id"])
            
            # Store event in data lake
            await data_lake.store_event(
                agent="qa",
                event_type="test_cases_generated",
                entity_type="story",
                entity_id=story_id,
                data={
                    "generation_id": test_generation_result["generation_id"],
                    "total_test_cases": test_generation_result["total_test_cases"],
                    "test_types": test_types,
                    "coverage_estimation": coverage_estimation.get("overall_coverage", 0)
                }
            )
            
            logger.info(f"AI test cases generated: {test_generation_result['generation_id']} for story {story_id}")
            
            return test_generation_result
            
        except Exception as e:
            logger.error(f"Failed to generate AI test cases: {e}")
            return {}
    
    async def _analyze_story_requirements(self, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze story requirements for test case generation."""
        try:
            nlp = await get_nlp_processor()
            
            title = story_data.get("title", "")
            description = story_data.get("description", "")
            acceptance_criteria = story_data.get("acceptance_criteria", [])
            
            # Extract functional requirements
            functional_requirements = []
            for criteria in acceptance_criteria:
                # Parse Given-When-Then format
                gherkin_match = re.search(r'given\s+(.+?)\s+when\s+(.+?)\s+then\s+(.+)', criteria.lower())
                if gherkin_match:
                    functional_requirements.append({
                        "given": gherkin_match.group(1).strip(),
                        "when": gherkin_match.group(2).strip(),
                        "then": gherkin_match.group(3).strip(),
                        "original": criteria
                    })
                else:
                    functional_requirements.append({
                        "requirement": criteria,
                        "type": "general"
                    })
            
            # Extract technical requirements
            combined_text = f"{title} {description} {' '.join(acceptance_criteria)}"
            technical_keywords = await nlp.extract_keywords(combined_text, 15)
            
            # Identify test categories
            test_categories = []
            text_lower = combined_text.lower()
            
            if any(word in text_lower for word in ["form", "input", "validation"]):
                test_categories.append("input_validation")
            if any(word in text_lower for word in ["api", "service", "endpoint"]):
                test_categories.append("api_testing")
            if any(word in text_lower for word in ["ui", "interface", "display"]):
                test_categories.append("ui_testing")
            if any(word in text_lower for word in ["database", "data", "storage"]):
                test_categories.append("data_testing")
            if any(word in text_lower for word in ["security", "auth", "permission"]):
                test_categories.append("security_testing")
            if any(word in text_lower for word in ["performance", "speed", "load"]):
                test_categories.append("performance_testing")
            
            # Identify edge cases
            edge_cases = []
            if "empty" in text_lower or "null" in text_lower:
                edge_cases.append("empty_input")
            if "maximum" in text_lower or "limit" in text_lower:
                edge_cases.append("boundary_values")
            if "invalid" in text_lower or "error" in text_lower:
                edge_cases.append("invalid_input")
            
            return {
                "functional_requirements": functional_requirements,
                "technical_keywords": technical_keywords,
                "test_categories": test_categories,
                "edge_cases": edge_cases,
                "complexity_score": len(acceptance_criteria) + len(technical_keywords) / 5
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze story requirements: {e}")
            return {}
    
    async def _generate_test_cases_by_type(self, story_data: Dict[str, Any], 
                                         requirements_analysis: Dict[str, Any], 
                                         test_type: str) -> List[Dict[str, Any]]:
        """Generate test cases for a specific test type."""
        try:
            if test_type == "unit":
                return await self._generate_unit_test_cases(story_data, requirements_analysis)
            elif test_type == "integration":
                return await self._generate_integration_test_cases(story_data, requirements_analysis)
            elif test_type == "e2e":
                return await self._generate_e2e_test_cases(story_data, requirements_analysis)
            elif test_type == "performance":
                return await self._generate_performance_test_cases(story_data, requirements_analysis)
            elif test_type == "security":
                return await self._generate_security_test_cases(story_data, requirements_analysis)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to generate {test_type} test cases: {e}")
            return []
    
    async def _generate_unit_test_cases(self, story_data: Dict[str, Any], 
                                      requirements_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate unit test cases."""
        try:
            test_cases = []
            functional_requirements = requirements_analysis.get("functional_requirements", [])
            
            # Generate test cases for each functional requirement
            for i, requirement in enumerate(functional_requirements):
                if isinstance(requirement, dict) and "given" in requirement:
                    # Gherkin-style requirement
                    test_cases.append({
                        "test_case_id": f"UNIT_{i+1:03d}",
                        "name": f"Test {requirement['when']}",
                        "description": f"Verify that {requirement['then']} when {requirement['when']} given {requirement['given']}",
                        "preconditions": requirement["given"],
                        "test_steps": [
                            f"Setup: {requirement['given']}",
                            f"Execute: {requirement['when']}",
                            f"Verify: {requirement['then']}"
                        ],
                        "expected_result": requirement["then"],
                        "test_type": "unit",
                        "priority": "high",
                        "automation_feasible": True
                    })
                else:
                    # General requirement
                    req_text = requirement.get("requirement", str(requirement))
                    test_cases.append({
                        "test_case_id": f"UNIT_{i+1:03d}",
                        "name": f"Test {req_text[:50]}...",
                        "description": f"Verify {req_text}",
                        "test_steps": [
                            "Setup test environment",
                            f"Execute functionality: {req_text}",
                            "Verify expected behavior"
                        ],
                        "expected_result": "Functionality works as specified",
                        "test_type": "unit",
                        "priority": "medium",
                        "automation_feasible": True
                    })
            
            # Add edge case tests
            edge_cases = requirements_analysis.get("edge_cases", [])
            for edge_case in edge_cases:
                test_cases.append({
                    "test_case_id": f"UNIT_EDGE_{len(test_cases)+1:03d}",
                    "name": f"Test {edge_case} handling",
                    "description": f"Verify system handles {edge_case} correctly",
                    "test_steps": [
                        f"Setup {edge_case} condition",
                        "Execute functionality",
                        "Verify appropriate handling"
                    ],
                    "expected_result": f"System handles {edge_case} gracefully",
                    "test_type": "unit",
                    "priority": "medium",
                    "automation_feasible": True
                })
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to generate unit test cases: {e}")
            return []
    
    async def _generate_integration_test_cases(self, story_data: Dict[str, Any], 
                                             requirements_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate integration test cases."""
        try:
            test_cases = []
            test_categories = requirements_analysis.get("test_categories", [])
            
            # API integration tests
            if "api_testing" in test_categories:
                test_cases.extend([
                    {
                        "test_case_id": "INT_API_001",
                        "name": "Test API endpoint integration",
                        "description": "Verify API endpoints work correctly with other services",
                        "test_steps": [
                            "Setup test data",
                            "Call API endpoint",
                            "Verify response format and data",
                            "Check downstream service integration"
                        ],
                        "expected_result": "API returns correct data and integrates properly",
                        "test_type": "integration",
                        "priority": "high",
                        "automation_feasible": True
                    },
                    {
                        "test_case_id": "INT_API_002",
                        "name": "Test API error handling",
                        "description": "Verify API handles errors correctly in integration",
                        "test_steps": [
                            "Setup error conditions",
                            "Call API endpoint",
                            "Verify error response",
                            "Check error propagation"
                        ],
                        "expected_result": "API handles errors gracefully",
                        "test_type": "integration",
                        "priority": "medium",
                        "automation_feasible": True
                    }
                ])
            
            # Database integration tests
            if "data_testing" in test_categories:
                test_cases.extend([
                    {
                        "test_case_id": "INT_DB_001",
                        "name": "Test database integration",
                        "description": "Verify data persistence and retrieval",
                        "test_steps": [
                            "Setup test database",
                            "Perform data operations",
                            "Verify data integrity",
                            "Check transaction handling"
                        ],
                        "expected_result": "Data operations work correctly",
                        "test_type": "integration",
                        "priority": "high",
                        "automation_feasible": True
                    }
                ])
            
            # UI integration tests
            if "ui_testing" in test_categories:
                test_cases.extend([
                    {
                        "test_case_id": "INT_UI_001",
                        "name": "Test UI component integration",
                        "description": "Verify UI components work together correctly",
                        "test_steps": [
                            "Load application",
                            "Interact with UI components",
                            "Verify component communication",
                            "Check data flow"
                        ],
                        "expected_result": "UI components integrate seamlessly",
                        "test_type": "integration",
                        "priority": "medium",
                        "automation_feasible": True
                    }
                ])
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to generate integration test cases: {e}")
            return []
    
    async def _generate_e2e_test_cases(self, story_data: Dict[str, Any], 
                                     requirements_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate end-to-end test cases."""
        try:
            test_cases = []
            functional_requirements = requirements_analysis.get("functional_requirements", [])
            
            # Generate E2E scenarios from acceptance criteria
            for i, requirement in enumerate(functional_requirements):
                if isinstance(requirement, dict) and "given" in requirement:
                    test_cases.append({
                        "test_case_id": f"E2E_{i+1:03d}",
                        "name": f"E2E: {requirement['when']}",
                        "description": f"Complete user journey: {requirement['original']}",
                        "test_steps": [
                            f"User setup: {requirement['given']}",
                            f"User action: {requirement['when']}",
                            f"System response: {requirement['then']}",
                            "Verify complete workflow"
                        ],
                        "expected_result": requirement["then"],
                        "test_type": "e2e",
                        "priority": "high",
                        "automation_feasible": True
                    })
            
            # Add common E2E scenarios
            title = story_data.get("title", "").lower()
            if "login" in title or "auth" in title:
                test_cases.append({
                    "test_case_id": "E2E_AUTH_001",
                    "name": "Complete authentication flow",
                    "description": "Test complete user authentication process",
                    "test_steps": [
                        "Navigate to login page",
                        "Enter valid credentials",
                        "Submit login form",
                        "Verify successful authentication",
                        "Check user dashboard access"
                    ],
                    "expected_result": "User successfully authenticates and accesses system",
                    "test_type": "e2e",
                    "priority": "critical",
                    "automation_feasible": True
                })
            
            if "form" in title or "create" in title:
                test_cases.append({
                    "test_case_id": "E2E_FORM_001",
                    "name": "Complete form submission flow",
                    "description": "Test complete form creation and submission process",
                    "test_steps": [
                        "Navigate to form page",
                        "Fill in all required fields",
                        "Submit form",
                        "Verify form processing",
                        "Check confirmation message"
                    ],
                    "expected_result": "Form is successfully submitted and processed",
                    "test_type": "e2e",
                    "priority": "high",
                    "automation_feasible": True
                })
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to generate E2E test cases: {e}")
            return []
    
    async def _generate_performance_test_cases(self, story_data: Dict[str, Any], 
                                             requirements_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance test cases."""
        try:
            test_cases = []
            test_categories = requirements_analysis.get("test_categories", [])
            
            if "performance_testing" in test_categories or "api_testing" in test_categories:
                test_cases.extend([
                    {
                        "test_case_id": "PERF_001",
                        "name": "Response time test",
                        "description": "Verify system response time under normal load",
                        "test_steps": [
                            "Setup performance monitoring",
                            "Execute functionality with normal load",
                            "Measure response times",
                            "Analyze performance metrics"
                        ],
                        "expected_result": "Response time < 2 seconds for 95% of requests",
                        "test_type": "performance",
                        "priority": "medium",
                        "automation_feasible": True
                    },
                    {
                        "test_case_id": "PERF_002",
                        "name": "Load test",
                        "description": "Verify system handles expected user load",
                        "test_steps": [
                            "Setup load testing environment",
                            "Simulate concurrent users",
                            "Monitor system resources",
                            "Analyze performance degradation"
                        ],
                        "expected_result": "System handles expected load without significant degradation",
                        "test_type": "performance",
                        "priority": "medium",
                        "automation_feasible": True
                    },
                    {
                        "test_case_id": "PERF_003",
                        "name": "Stress test",
                        "description": "Verify system behavior under extreme load",
                        "test_steps": [
                            "Setup stress testing environment",
                            "Gradually increase load beyond normal capacity",
                            "Monitor system breaking point",
                            "Verify graceful degradation"
                        ],
                        "expected_result": "System fails gracefully under extreme load",
                        "test_type": "performance",
                        "priority": "low",
                        "automation_feasible": True
                    }
                ])
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to generate performance test cases: {e}")
            return []
    
    async def _generate_security_test_cases(self, story_data: Dict[str, Any], 
                                          requirements_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate security test cases."""
        try:
            test_cases = []
            test_categories = requirements_analysis.get("test_categories", [])
            
            if "security_testing" in test_categories or "input_validation" in test_categories:
                test_cases.extend([
                    {
                        "test_case_id": "SEC_001",
                        "name": "Input validation test",
                        "description": "Verify system validates input properly",
                        "test_steps": [
                            "Identify input fields",
                            "Test with malicious input (XSS, SQL injection)",
                            "Verify input sanitization",
                            "Check error handling"
                        ],
                        "expected_result": "System properly validates and sanitizes input",
                        "test_type": "security",
                        "priority": "high",
                        "automation_feasible": True
                    },
                    {
                        "test_case_id": "SEC_002",
                        "name": "Authentication test",
                        "description": "Verify authentication mechanisms are secure",
                        "test_steps": [
                            "Test with invalid credentials",
                            "Test session management",
                            "Verify password policies",
                            "Check for authentication bypass"
                        ],
                        "expected_result": "Authentication is secure and cannot be bypassed",
                        "test_type": "security",
                        "priority": "critical",
                        "automation_feasible": True
                    },
                    {
                        "test_case_id": "SEC_003",
                        "name": "Authorization test",
                        "description": "Verify proper access control",
                        "test_steps": [
                            "Test with different user roles",
                            "Attempt unauthorized access",
                            "Verify permission checks",
                            "Check for privilege escalation"
                        ],
                        "expected_result": "Users can only access authorized resources",
                        "test_type": "security",
                        "priority": "high",
                        "automation_feasible": True
                    }
                ])
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to generate security test cases: {e}")
            return []
    
    async def _generate_test_data_requirements(self, story_data: Dict[str, Any], 
                                             requirements_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test data requirements."""
        try:
            test_data_requirements = {
                "user_data": [],
                "system_data": [],
                "test_scenarios": [],
                "data_cleanup": []
            }
            
            # Analyze story for data needs
            description = story_data.get("description", "").lower()
            acceptance_criteria = " ".join(story_data.get("acceptance_criteria", [])).lower()
            combined_text = f"{description} {acceptance_criteria}"
            
            # User data requirements
            if "user" in combined_text or "customer" in combined_text:
                test_data_requirements["user_data"].extend([
                    "Valid user accounts with different roles",
                    "Invalid user credentials for negative testing",
                    "User profiles with various data combinations"
                ])
            
            # System data requirements
            if "product" in combined_text or "item" in combined_text:
                test_data_requirements["system_data"].extend([
                    "Product catalog with various categories",
                    "Inventory data with different stock levels",
                    "Pricing information for different scenarios"
                ])
            
            if "order" in combined_text or "transaction" in combined_text:
                test_data_requirements["system_data"].extend([
                    "Order history data",
                    "Payment method configurations",
                    "Transaction records for testing"
                ])
            
            # Test scenarios
            functional_requirements = requirements_analysis.get("functional_requirements", [])
            for requirement in functional_requirements:
                if isinstance(requirement, dict) and "given" in requirement:
                    test_data_requirements["test_scenarios"].append({
                        "scenario": requirement["when"],
                        "setup_data": requirement["given"],
                        "expected_outcome": requirement["then"]
                    })
            
            # Data cleanup requirements
            test_data_requirements["data_cleanup"].extend([
                "Remove test users after execution",
                "Clean up test transactions",
                "Reset system state for next test run"
            ])
            
            return test_data_requirements
            
        except Exception as e:
            logger.error(f"Failed to generate test data requirements: {e}")
            return {}
    
    async def _generate_automation_suggestions(self, generated_test_cases: Dict[str, List[Dict[str, Any]]], 
                                             story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test automation suggestions."""
        try:
            automation_suggestions = {
                "high_priority_automation": [],
                "automation_tools": [],
                "automation_strategy": {},
                "estimated_automation_effort": {}
            }
            
            # Analyze test cases for automation priority
            total_automatable = 0
            total_tests = 0
            
            for test_type, test_cases in generated_test_cases.items():
                automatable_count = 0
                for test_case in test_cases:
                    total_tests += 1
                    if test_case.get("automation_feasible", False):
                        total_automatable += 1
                        automatable_count += 1
                        
                        # High priority automation candidates
                        if test_case.get("priority") in ["critical", "high"]:
                            automation_suggestions["high_priority_automation"].append({
                                "test_case_id": test_case["test_case_id"],
                                "name": test_case["name"],
                                "test_type": test_type,
                                "priority": test_case["priority"],
                                "automation_effort": self._estimate_automation_effort(test_case, test_type)
                            })
                
                # Estimate automation effort for each test type
                automation_suggestions["estimated_automation_effort"][test_type] = {
                    "total_tests": len(test_cases),
                    "automatable_tests": automatable_count,
                    "automation_percentage": (automatable_count / len(test_cases)) * 100 if test_cases else 0,
                    "estimated_hours": automatable_count * self._get_base_automation_hours(test_type)
                }
            
            # Suggest automation tools
            if "unit" in generated_test_cases:
                automation_suggestions["automation_tools"].append({
                    "tool": "Jest/Pytest",
                    "purpose": "Unit testing framework",
                    "test_types": ["unit"]
                })
            
            if "integration" in generated_test_cases:
                automation_suggestions["automation_tools"].append({
                    "tool": "Postman/Newman",
                    "purpose": "API testing automation",
                    "test_types": ["integration"]
                })
            
            if "e2e" in generated_test_cases:
                automation_suggestions["automation_tools"].append({
                    "tool": "Selenium/Playwright",
                    "purpose": "End-to-end testing automation",
                    "test_types": ["e2e"]
                })
            
            if "performance" in generated_test_cases:
                automation_suggestions["automation_tools"].append({
                    "tool": "JMeter/K6",
                    "purpose": "Performance testing automation",
                    "test_types": ["performance"]
                })
            
            # Automation strategy
            automation_suggestions["automation_strategy"] = {
                "overall_automation_percentage": (total_automatable / total_tests) * 100 if total_tests > 0 else 0,
                "recommended_approach": self._recommend_automation_approach(generated_test_cases),
                "implementation_phases": self._suggest_automation_phases(automation_suggestions["high_priority_automation"])
            }
            
            return automation_suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate automation suggestions: {e}")
            return {}
    
    def _estimate_automation_effort(self, test_case: Dict[str, Any], test_type: str) -> str:
        """Estimate automation effort for a test case."""
        base_hours = self._get_base_automation_hours(test_type)
        
        # Adjust based on complexity
        steps_count = len(test_case.get("test_steps", []))
        if steps_count > 5:
            base_hours *= 1.5
        elif steps_count < 3:
            base_hours *= 0.8
        
        if base_hours <= 2:
            return "low"
        elif base_hours <= 6:
            return "medium"
        else:
            return "high"
    
    def _get_base_automation_hours(self, test_type: str) -> float:
        """Get base automation hours for test type."""
        base_hours = {
            "unit": 1.0,
            "integration": 3.0,
            "e2e": 6.0,
            "performance": 8.0,
            "security": 4.0
        }
        return base_hours.get(test_type, 3.0)
    
    def _recommend_automation_approach(self, generated_test_cases: Dict[str, List[Dict[str, Any]]]) -> str:
        """Recommend automation approach based on test case distribution."""
        total_tests = sum(len(cases) for cases in generated_test_cases.values())
        
        if total_tests <= 10:
            return "selective_automation"
        elif total_tests <= 30:
            return "progressive_automation"
        else:
            return "comprehensive_automation"
    
    def _suggest_automation_phases(self, high_priority_tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest automation implementation phases."""
        phases = []
        
        # Phase 1: Critical tests
        critical_tests = [t for t in high_priority_tests if t["priority"] == "critical"]
        if critical_tests:
            phases.append({
                "phase": 1,
                "name": "Critical Test Automation",
                "tests": len(critical_tests),
                "estimated_weeks": 2,
                "description": "Automate critical path tests first"
            })
        
        # Phase 2: High priority tests
        high_tests = [t for t in high_priority_tests if t["priority"] == "high"]
        if high_tests:
            phases.append({
                "phase": 2,
                "name": "High Priority Test Automation",
                "tests": len(high_tests),
                "estimated_weeks": 3,
                "description": "Automate high priority tests"
            })
        
        # Phase 3: Remaining tests
        remaining_tests = len(high_priority_tests) - len(critical_tests) - len(high_tests)
        if remaining_tests > 0:
            phases.append({
                "phase": 3,
                "name": "Complete Test Automation",
                "tests": remaining_tests,
                "estimated_weeks": 4,
                "description": "Automate remaining test cases"
            })
        
        return phases
    
    async def _estimate_test_coverage(self, generated_test_cases: Dict[str, List[Dict[str, Any]]], 
                                    requirements_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate test coverage based on generated test cases."""
        try:
            functional_requirements = requirements_analysis.get("functional_requirements", [])
            test_categories = requirements_analysis.get("test_categories", [])
            
            coverage_analysis = {
                "functional_coverage": 0,
                "category_coverage": {},
                "overall_coverage": 0,
                "coverage_gaps": []
            }
            
            # Calculate functional coverage
            total_requirements = len(functional_requirements)
            covered_requirements = 0
            
            for test_type, test_cases in generated_test_cases.items():
                if test_type in ["unit", "integration", "e2e"]:
                    covered_requirements += len(test_cases)
            
            if total_requirements > 0:
                coverage_analysis["functional_coverage"] = min((covered_requirements / total_requirements) * 100, 100)
            
            # Calculate category coverage
            for category in test_categories:
                category_tests = 0
                for test_type, test_cases in generated_test_cases.items():
                    if self._category_matches_test_type(category, test_type):
                        category_tests += len(test_cases)
                
                coverage_analysis["category_coverage"][category] = min(category_tests * 20, 100)  # 20% per test
            
            # Calculate overall coverage
            coverage_scores = [coverage_analysis["functional_coverage"]]
            coverage_scores.extend(coverage_analysis["category_coverage"].values())
            
            if coverage_scores:
                coverage_analysis["overall_coverage"] = sum(coverage_scores) / len(coverage_scores)
            
            # Identify coverage gaps
            if coverage_analysis["functional_coverage"] < 80:
                coverage_analysis["coverage_gaps"].append("Insufficient functional test coverage")
            
            for category, coverage in coverage_analysis["category_coverage"].items():
                if coverage < 60:
                    coverage_analysis["coverage_gaps"].append(f"Low coverage for {category}")
            
            return coverage_analysis
            
        except Exception as e:
            logger.error(f"Failed to estimate test coverage: {e}")
            return {"overall_coverage": 50}
    
    def _category_matches_test_type(self, category: str, test_type: str) -> bool:
        """Check if a test category matches a test type."""
        category_mapping = {
            "input_validation": ["unit", "security"],
            "api_testing": ["integration", "performance"],
            "ui_testing": ["e2e", "integration"],
            "data_testing": ["integration", "unit"],
            "security_testing": ["security"],
            "performance_testing": ["performance"]
        }
        
        return test_type in category_mapping.get(category, [])
    
    async def _generate_test_execution_strategy(self, generated_test_cases: Dict[str, List[Dict[str, Any]]], 
                                              story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test execution strategy."""
        try:
            total_tests = sum(len(cases) for cases in generated_test_cases.values())
            
            execution_strategy = {
                "execution_order": [],
                "parallel_execution": {},
                "estimated_duration": {},
                "resource_requirements": {},
                "risk_mitigation": []
            }
            
            # Define execution order
            test_type_priority = ["unit", "integration", "security", "e2e", "performance"]
            for test_type in test_type_priority:
                if test_type in generated_test_cases:
                    execution_strategy["execution_order"].append(test_type)
            
            # Parallel execution suggestions
            for test_type, test_cases in generated_test_cases.items():
                if len(test_cases) > 3:
                    execution_strategy["parallel_execution"][test_type] = {
                        "can_parallelize": True,
                        "max_parallel": min(len(test_cases), 5),
                        "estimated_time_savings": "30-50%"
                    }
                else:
                    execution_strategy["parallel_execution"][test_type] = {
                        "can_parallelize": False,
                        "reason": "Too few tests to benefit from parallelization"
                    }
            
            # Estimated duration
            for test_type, test_cases in generated_test_cases.items():
                base_time_per_test = self._get_base_execution_time(test_type)
                total_time = len(test_cases) * base_time_per_test
                
                # Adjust for parallelization
                if execution_strategy["parallel_execution"][test_type]["can_parallelize"]:
                    parallel_factor = execution_strategy["parallel_execution"][test_type]["max_parallel"]
                    total_time = total_time / parallel_factor
                
                execution_strategy["estimated_duration"][test_type] = {
                    "sequential_minutes": len(test_cases) * base_time_per_test,
                    "parallel_minutes": total_time,
                    "test_count": len(test_cases)
                }
            
            # Resource requirements
            execution_strategy["resource_requirements"] = {
                "test_environments": self._calculate_environment_needs(generated_test_cases),
                "test_data": "Comprehensive test data set required",
                "infrastructure": self._calculate_infrastructure_needs(generated_test_cases),
                "team_size": max(1, total_tests // 20)  # 1 person per 20 tests
            }
            
            # Risk mitigation
            execution_strategy["risk_mitigation"] = [
                "Implement test data backup and restore procedures",
                "Set up monitoring for test execution",
                "Create rollback procedures for failed tests",
                "Establish clear test failure escalation process"
            ]
            
            return execution_strategy
            
        except Exception as e:
            logger.error(f"Failed to generate test execution strategy: {e}")
            return {}
    
    def _get_base_execution_time(self, test_type: str) -> float:
        """Get base execution time per test in minutes."""
        execution_times = {
            "unit": 0.5,
            "integration": 2.0,
            "e2e": 5.0,
            "performance": 10.0,
            "security": 3.0
        }
        return execution_times.get(test_type, 2.0)
    
    def _calculate_environment_needs(self, generated_test_cases: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Calculate test environment needs."""
        environments = ["development"]
        
        if "integration" in generated_test_cases or "e2e" in generated_test_cases:
            environments.append("staging")
        
        if "performance" in generated_test_cases:
            environments.append("performance_testing")
        
        if "security" in generated_test_cases:
            environments.append("security_testing")
        
        return environments
    
    def _calculate_infrastructure_needs(self, generated_test_cases: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate infrastructure needs for test execution."""
        needs = {
            "compute_resources": "standard",
            "storage_requirements": "minimal",
            "network_requirements": "standard"
        }
        
        if "performance" in generated_test_cases:
            needs["compute_resources"] = "high"
            needs["network_requirements"] = "high_bandwidth"
        
        if "e2e" in generated_test_cases:
            needs["storage_requirements"] = "moderate"
        
        return needs
    
    def _calculate_execution_time(self, generated_test_cases: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate total execution time for all test cases."""
        total_time = 0
        time_breakdown = {}
        
        for test_type, test_cases in generated_test_cases.items():
            base_time = self._get_base_execution_time(test_type)
            type_time = len(test_cases) * base_time
            time_breakdown[test_type] = type_time
            total_time += type_time
        
        time_breakdown["total_minutes"] = total_time
        time_breakdown["total_hours"] = total_time / 60
        
        return time_breakdown
    
    async def _store_individual_test_cases(self, story_id: str, generated_test_cases: Dict[str, List[Dict[str, Any]]], 
                                         generation_id: str) -> None:
        """Store individual test cases for tracking."""
        try:
            for test_type, test_cases in generated_test_cases.items():
                for test_case in test_cases:
                    test_case_record = {
                        **test_case,
                        "story_id": story_id,
                        "generation_id": generation_id,
                        "status": "generated",
                        "created_at": datetime.utcnow()
                    }
                    
                    await self.db["individual_test_cases"].insert_one(test_case_record)
            
        except Exception as e:
            logger.error(f"Failed to store individual test cases: {e}")
