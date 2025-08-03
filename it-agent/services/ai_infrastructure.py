"""AI-powered infrastructure management services."""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger
import uuid

from shared_libs.database import get_database
from shared_libs.data_lake import get_data_lake


class AIInfrastructureService:
    """AI-powered infrastructure monitoring and management."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.monitoring_collection = "infrastructure_monitoring"
        self.predictions_collection = "infrastructure_predictions"
        self.incidents_collection = "infrastructure_incidents"
    
    async def initialize(self):
        """Initialize the AI infrastructure service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.monitoring_collection].create_index("device_id")
        await self.db[self.monitoring_collection].create_index("timestamp")
        await self.db[self.monitoring_collection].create_index("metric_type")
        
        await self.db[self.predictions_collection].create_index("prediction_id", unique=True)
        await self.db[self.predictions_collection].create_index("device_id")
        await self.db[self.predictions_collection].create_index("prediction_type")
        
        await self.db[self.incidents_collection].create_index("incident_id", unique=True)
        await self.db[self.incidents_collection].create_index("severity")
        await self.db[self.incidents_collection].create_index("status")
        
        logger.info("AI Infrastructure service initialized")
    
    async def predict_hardware_failure(self, asset_id: str) -> Dict[str, Any]:
        """Predict hardware failure probability."""
        try:
            # Get asset information
            asset = await self.db["it_assets"].find_one({"asset_id": asset_id})
            if not asset:
                return {"error": "Asset not found"}
            
            # Get monitoring data
            monitoring_data = await self._get_asset_monitoring_data(asset_id, days=30)
            
            # Calculate failure probability based on various factors
            failure_probability = 0.0
            risk_factors = []
            
            # Age factor
            purchase_date = asset.get("purchase_date")
            if purchase_date:
                if isinstance(purchase_date, str):
                    purchase_date = datetime.fromisoformat(purchase_date)
                
                age_years = (datetime.utcnow() - purchase_date).days / 365
                if age_years > 3:
                    age_risk = min(0.3, (age_years - 3) * 0.1)
                    failure_probability += age_risk
                    risk_factors.append(f"Asset age: {age_years:.1f} years")
            
            # Usage pattern analysis
            if monitoring_data:
                usage_analysis = self._analyze_usage_patterns(monitoring_data)
                if usage_analysis.get("high_usage", False):
                    failure_probability += 0.2
                    risk_factors.append("High usage detected")
                
                if usage_analysis.get("temperature_issues", False):
                    failure_probability += 0.25
                    risk_factors.append("Temperature anomalies detected")
            
            # Warranty status
            warranty_expiry = asset.get("warranty_expiry")
            if warranty_expiry:
                if isinstance(warranty_expiry, str):
                    warranty_expiry = datetime.fromisoformat(warranty_expiry)
                
                if warranty_expiry < datetime.utcnow():
                    failure_probability += 0.15
                    risk_factors.append("Warranty expired")
            
            # Historical incident data
            incident_count = await self.db["it_tickets"].count_documents({
                "asset_id": asset_id,
                "category": "hardware",
                "created_at": {"$gte": datetime.utcnow() - timedelta(days=90)}
            })
            
            if incident_count > 3:
                failure_probability += min(0.2, incident_count * 0.05)
                risk_factors.append(f"Multiple recent incidents: {incident_count}")
            
            # Cap at 1.0
            failure_probability = min(failure_probability, 1.0)
            
            # Determine risk level
            if failure_probability >= 0.7:
                risk_level = "critical"
                recommended_action = "immediate_replacement"
            elif failure_probability >= 0.5:
                risk_level = "high"
                recommended_action = "schedule_maintenance"
            elif failure_probability >= 0.3:
                risk_level = "medium"
                recommended_action = "monitor_closely"
            else:
                risk_level = "low"
                recommended_action = "continue_monitoring"
            
            prediction_result = {
                "prediction_id": f"PRED_{str(uuid.uuid4())[:8].upper()}",
                "asset_id": asset_id,
                "failure_probability": round(failure_probability, 3),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommended_action": recommended_action,
                "confidence": 0.8,
                "prediction_date": datetime.utcnow(),
                "valid_until": datetime.utcnow() + timedelta(days=30)
            }
            
            # Store prediction
            await self.db[self.predictions_collection].insert_one(prediction_result)
            
            # Store event in data lake
            data_lake = await get_data_lake()
            await data_lake.store_event(
                agent="it",
                event_type="hardware_failure_predicted",
                entity_type="asset",
                entity_id=asset_id,
                data={
                    "failure_probability": failure_probability,
                    "risk_level": risk_level,
                    "recommended_action": recommended_action
                }
            )
            
            logger.info(f"Hardware failure prediction completed for {asset_id}: {failure_probability:.3f} probability")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Failed to predict hardware failure: {e}")
            return {}
    
    async def _get_asset_monitoring_data(self, asset_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get monitoring data for an asset."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            monitoring_data = await self.db[self.monitoring_collection].find({
                "device_id": asset_id,
                "timestamp": {"$gte": cutoff_date}
            }).sort("timestamp", -1).to_list(None)
            
            return monitoring_data
            
        except Exception as e:
            logger.error(f"Failed to get monitoring data for {asset_id}: {e}")
            return []
    
    def _analyze_usage_patterns(self, monitoring_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze usage patterns from monitoring data."""
        try:
            if not monitoring_data:
                return {"high_usage": False, "temperature_issues": False}
            
            # Analyze CPU usage
            cpu_readings = [data.get("cpu_usage", 0) for data in monitoring_data if "cpu_usage" in data]
            high_cpu_usage = sum(1 for reading in cpu_readings if reading > 80) / len(cpu_readings) if cpu_readings else 0
            
            # Analyze temperature
            temp_readings = [data.get("temperature", 0) for data in monitoring_data if "temperature" in data]
            high_temp_readings = sum(1 for temp in temp_readings if temp > 70) / len(temp_readings) if temp_readings else 0
            
            return {
                "high_usage": high_cpu_usage > 0.3,  # More than 30% of readings above 80%
                "temperature_issues": high_temp_readings > 0.2,  # More than 20% of readings above 70°C
                "avg_cpu_usage": sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0,
                "avg_temperature": sum(temp_readings) / len(temp_readings) if temp_readings else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze usage patterns: {e}")
            return {"high_usage": False, "temperature_issues": False}
    
    async def detect_security_threats(self) -> Dict[str, Any]:
        """Detect potential security threats."""
        try:
            # Analyze recent login patterns
            login_analysis = await self._analyze_login_patterns()
            
            # Check for unusual network activity
            network_analysis = await self._analyze_network_activity()
            
            # Check for software vulnerabilities
            vulnerability_analysis = await self._check_software_vulnerabilities()
            
            threats_detected = []
            overall_risk = 0.0
            
            # Process login analysis
            if login_analysis.get("suspicious_logins", 0) > 0:
                threats_detected.append({
                    "type": "suspicious_login",
                    "severity": "medium",
                    "count": login_analysis["suspicious_logins"],
                    "description": "Unusual login patterns detected"
                })
                overall_risk += 0.3
            
            # Process network analysis
            if network_analysis.get("unusual_traffic", False):
                threats_detected.append({
                    "type": "network_anomaly",
                    "severity": "high",
                    "description": "Unusual network traffic patterns detected"
                })
                overall_risk += 0.4
            
            # Process vulnerability analysis
            critical_vulns = vulnerability_analysis.get("critical_vulnerabilities", 0)
            if critical_vulns > 0:
                threats_detected.append({
                    "type": "software_vulnerability",
                    "severity": "critical",
                    "count": critical_vulns,
                    "description": f"{critical_vulns} critical vulnerabilities found"
                })
                overall_risk += 0.5
            
            # Cap risk at 1.0
            overall_risk = min(overall_risk, 1.0)
            
            # Determine security status
            if overall_risk >= 0.7:
                security_status = "critical"
            elif overall_risk >= 0.4:
                security_status = "elevated"
            else:
                security_status = "normal"
            
            detection_result = {
                "detection_id": f"SEC_{str(uuid.uuid4())[:8].upper()}",
                "threats_detected": threats_detected,
                "overall_risk_score": round(overall_risk, 3),
                "security_status": security_status,
                "recommendations": self._generate_security_recommendations(threats_detected),
                "scan_timestamp": datetime.utcnow()
            }
            
            # Store detection results
            await self.db["security_scans"].insert_one(detection_result)
            
            # Create incidents for high-risk threats
            for threat in threats_detected:
                if threat.get("severity") in ["high", "critical"]:
                    await self._create_security_incident(threat, detection_result)
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Failed to detect security threats: {e}")
            return {"security_status": "unknown"}
    
    async def _analyze_login_patterns(self) -> Dict[str, Any]:
        """Analyze login patterns for anomalies."""
        try:
            # Get recent login data (would come from authentication logs)
            # For demo, simulate analysis
            
            suspicious_indicators = 0
            
            # Check for logins outside business hours
            # Check for multiple failed attempts
            # Check for logins from unusual locations
            
            return {
                "suspicious_logins": suspicious_indicators,
                "total_logins_analyzed": 100,  # Simulated
                "analysis_period_hours": 24
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze login patterns: {e}")
            return {"suspicious_logins": 0}
    
    async def _analyze_network_activity(self) -> Dict[str, Any]:
        """Analyze network activity for anomalies."""
        try:
            # Simulate network traffic analysis
            # In real implementation, would analyze firewall logs, network monitoring data
            
            return {
                "unusual_traffic": False,
                "bandwidth_anomalies": False,
                "suspicious_connections": 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze network activity: {e}")
            return {"unusual_traffic": False}
    
    async def _check_software_vulnerabilities(self) -> Dict[str, Any]:
        """Check for software vulnerabilities."""
        try:
            # Get software licenses and check for known vulnerabilities
            licenses = await self.db["software_licenses"].find({}).to_list(None)
            
            critical_vulnerabilities = 0
            high_vulnerabilities = 0
            
            # Simulate vulnerability scanning
            for license in licenses:
                software_name = license.get("software_name", "").lower()
                
                # Simulate known vulnerable software
                if "adobe" in software_name or "java" in software_name:
                    critical_vulnerabilities += 1
                elif "chrome" in software_name or "firefox" in software_name:
                    high_vulnerabilities += 1
            
            return {
                "critical_vulnerabilities": critical_vulnerabilities,
                "high_vulnerabilities": high_vulnerabilities,
                "software_scanned": len(licenses)
            }
            
        except Exception as e:
            logger.error(f"Failed to check software vulnerabilities: {e}")
            return {"critical_vulnerabilities": 0, "high_vulnerabilities": 0}
    
    def _generate_security_recommendations(self, threats: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on detected threats."""
        recommendations = []
        
        for threat in threats:
            threat_type = threat.get("type")
            
            if threat_type == "suspicious_login":
                recommendations.extend([
                    "Enable multi-factor authentication",
                    "Review user access permissions",
                    "Implement login monitoring alerts"
                ])
            elif threat_type == "network_anomaly":
                recommendations.extend([
                    "Review firewall rules",
                    "Implement network segmentation",
                    "Increase network monitoring"
                ])
            elif threat_type == "software_vulnerability":
                recommendations.extend([
                    "Update vulnerable software immediately",
                    "Implement patch management process",
                    "Conduct security audit"
                ])
        
        return list(set(recommendations))  # Remove duplicates
    
    async def _create_security_incident(self, threat: Dict[str, Any], detection_result: Dict[str, Any]) -> None:
        """Create security incident for high-risk threats."""
        try:
            incident = {
                "incident_id": f"INC_{str(uuid.uuid4())[:8].upper()}",
                "incident_type": threat.get("type"),
                "severity": threat.get("severity"),
                "description": threat.get("description"),
                "detection_id": detection_result.get("detection_id"),
                "status": "open",
                "detected_at": datetime.utcnow(),
                "response_actions": []
            }
            
            await self.db[self.incidents_collection].insert_one(incident)
            logger.warning(f"Security incident created: {incident['incident_id']}")
            
        except Exception as e:
            logger.error(f"Failed to create security incident: {e}")