"""Legal Service for managing legal operations."""

import uuid
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from loguru import logger

from shared_libs.database import get_database
from models.legal import Contract, ContractCreate, LegalCase, CaseCreate, ComplianceCheck


class LegalService:
    """Legal service for managing contracts, cases, and compliance."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.contracts_collection = "contracts"
        self.cases_collection = "legal_cases"
        self.compliance_collection = "compliance_checks"
        self.documents_collection = "legal_documents"
    
    async def initialize(self):
        """Initialize the Legal service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.contracts_collection].create_index("contract_id", unique=True)
        await self.db[self.contracts_collection].create_index("contract_type")
        await self.db[self.contracts_collection].create_index("status")
        await self.db[self.contracts_collection].create_index("end_date")
        await self.db[self.contracts_collection].create_index("assigned_lawyer")
        
        await self.db[self.cases_collection].create_index("case_id", unique=True)
        await self.db[self.cases_collection].create_index("case_number")
        await self.db[self.cases_collection].create_index("status")
        await self.db[self.cases_collection].create_index("assigned_lawyer")
        await self.db[self.cases_collection].create_index("priority")
        
        await self.db[self.compliance_collection].create_index("check_id", unique=True)
        await self.db[self.compliance_collection].create_index("regulation_type")
        await self.db[self.compliance_collection].create_index("department")
        await self.db[self.compliance_collection].create_index("status")
        await self.db[self.compliance_collection].create_index("next_check_date")
        
        await self.db[self.documents_collection].create_index("document_id", unique=True)
        await self.db[self.documents_collection].create_index("document_type")
        await self.db[self.documents_collection].create_index("status")
        
        logger.info("Legal service initialized")
    
    async def create_contract(self, contract_data: ContractCreate) -> Optional[Contract]:
        """Create a new contract."""
        try:
            contract_id = f"CONT{str(uuid.uuid4())[:8].upper()}"
            
            contract_dict = contract_data.dict()
            contract_dict.update({
                "contract_id": contract_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            result = await self.db[self.contracts_collection].insert_one(contract_dict)
            
            if result.inserted_id:
                contract_dict["_id"] = result.inserted_id
                logger.info(f"Contract created: {contract_id}")
                return Contract(**contract_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create contract: {e}")
            return None
    
    async def get_contract(self, contract_id: str) -> Optional[Contract]:
        """Get a contract by ID."""
        try:
            contract_doc = await self.db[self.contracts_collection].find_one({"contract_id": contract_id})
            
            if contract_doc:
                return Contract(**contract_doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get contract {contract_id}: {e}")
            return None
    
    async def get_expiring_contracts(self, days_ahead: int = 30) -> List[Contract]:
        """Get contracts expiring within specified days."""
        try:
            cutoff_date = date.today() + timedelta(days=days_ahead)
            
            contracts = []
            cursor = self.db[self.contracts_collection].find({
                "end_date": {"$lte": cutoff_date},
                "status": "active"
            }).sort("end_date", 1)
            
            async for contract_doc in cursor:
                contracts.append(Contract(**contract_doc))
            
            return contracts
            
        except Exception as e:
            logger.error(f"Failed to get expiring contracts: {e}")
            return []
    
    async def create_legal_case(self, case_data: CaseCreate) -> Optional[LegalCase]:
        """Create a new legal case."""
        try:
            case_id = f"CASE{str(uuid.uuid4())[:8].upper()}"
            case_number = f"LC-{datetime.utcnow().year}-{str(uuid.uuid4())[:6].upper()}"
            
            case_dict = case_data.dict()
            case_dict.update({
                "case_id": case_id,
                "case_number": case_number,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "timeline": [{
                    "event": "Case created",
                    "date": datetime.utcnow(),
                    "description": "Legal case opened"
                }]
            })
            
            result = await self.db[self.cases_collection].insert_one(case_dict)
            
            if result.inserted_id:
                case_dict["_id"] = result.inserted_id
                logger.info(f"Legal case created: {case_id}")
                return LegalCase(**case_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create legal case: {e}")
            return None
    
    async def get_compliance_status(self, department: Optional[str] = None) -> Dict[str, Any]:
        """Get compliance status overview."""
        try:
            query = {}
            if department:
                query["department"] = department
            
            # Get all compliance checks
            compliance_checks = await self.db[self.compliance_collection].find(query).to_list(None)
            
            status_summary = {
                "total_checks": len(compliance_checks),
                "compliant": 0,
                "non_compliant": 0,
                "under_review": 0,
                "remediation_required": 0,
                "overdue_checks": 0,
                "upcoming_checks": 0
            }
            
            today = datetime.utcnow().date()
            
            for check in compliance_checks:
                status = check.get("status", "under_review")
                status_summary[status] = status_summary.get(status, 0) + 1
                
                # Check for overdue
                next_check = check.get("next_check_date")
                if next_check:
                    if isinstance(next_check, str):
                        next_check = datetime.fromisoformat(next_check).date()
                    
                    if next_check < today:
                        status_summary["overdue_checks"] += 1
                    elif next_check <= today + timedelta(days=30):
                        status_summary["upcoming_checks"] += 1
            
            # Calculate compliance percentage
            compliant_count = status_summary["compliant"]
            total_count = status_summary["total_checks"]
            compliance_percentage = (compliant_count / total_count * 100) if total_count > 0 else 0
            
            return {
                "compliance_summary": status_summary,
                "compliance_percentage": round(compliance_percentage, 2),
                "department": department,
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get compliance status: {e}")
            return {}
    
    async def monitor_contract_renewals(self) -> Dict[str, Any]:
        """Monitor contracts for renewal opportunities."""
        try:
            # Get contracts expiring in next 90 days
            expiring_contracts = await self.get_expiring_contracts(90)
            
            renewal_analysis = {
                "contracts_expiring": len(expiring_contracts),
                "renewal_opportunities": [],
                "urgent_renewals": [],
                "total_value_at_risk": 0.0
            }
            
            for contract in expiring_contracts:
                days_until_expiry = (contract.end_date - date.today()).days if contract.end_date else 0
                contract_value = contract.value or 0
                
                renewal_analysis["total_value_at_risk"] += contract_value
                
                renewal_info = {
                    "contract_id": contract.contract_id,
                    "title": contract.title,
                    "contract_type": contract.contract_type,
                    "end_date": contract.end_date.isoformat() if contract.end_date else None,
                    "days_until_expiry": days_until_expiry,
                    "value": contract_value,
                    "auto_renewal": contract.auto_renewal
                }
                
                if days_until_expiry <= 30:
                    renewal_analysis["urgent_renewals"].append(renewal_info)
                else:
                    renewal_analysis["renewal_opportunities"].append(renewal_info)
            
            return renewal_analysis
            
        except Exception as e:
            logger.error(f"Failed to monitor contract renewals: {e}")
            return {}