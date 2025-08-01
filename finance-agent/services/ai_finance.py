"""AI-powered finance services for Finance Agent."""

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
from PIL import Image
import pytesseract
import io
import base64

from shared_libs.database import get_database
from shared_libs.intelligence import get_nlp_processor, get_ml_predictor, get_business_rules
from shared_libs.data_lake import get_data_lake

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY", "")


class AIFinanceService:
    """AI-powered finance automation and analysis service."""
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.expense_analysis_collection = "expense_analysis"
        self.invoice_analysis_collection = "invoice_analysis"
        self.fraud_detection_collection = "fraud_detection"
        self.cash_flow_predictions_collection = "cash_flow_predictions"
        self.tax_suggestions_collection = "tax_suggestions"
    
    async def initialize(self):
        """Initialize the AI finance service."""
        self.db = await get_database()
        
        # Create indexes
        await self.db[self.expense_analysis_collection].create_index("analysis_id", unique=True)
        await self.db[self.expense_analysis_collection].create_index("expense_id")
        await self.db[self.expense_analysis_collection].create_index("created_at")
        
        await self.db[self.invoice_analysis_collection].create_index("analysis_id", unique=True)
        await self.db[self.invoice_analysis_collection].create_index("invoice_id")
        
        await self.db[self.fraud_detection_collection].create_index("detection_id", unique=True)
        await self.db[self.fraud_detection_collection].create_index("risk_score")
        
        await self.db[self.cash_flow_predictions_collection].create_index("prediction_id", unique=True)
        await self.db[self.cash_flow_predictions_collection].create_index("prediction_date")
        
        await self.db[self.tax_suggestions_collection].create_index("suggestion_id", unique=True)
        await self.db[self.tax_suggestions_collection].create_index("tax_year")
        
        logger.info("AI Finance service initialized")
    
    async def classify_expense_with_ocr(self, expense_id: str, receipt_image: str, 
                                      expense_description: str) -> Dict[str, Any]:
        """Classify expense using OCR and ML analysis."""
        try:
            nlp = await get_nlp_processor()
            data_lake = await get_data_lake()
            
            # Extract text from receipt image using OCR
            ocr_text = await self._extract_text_from_image(receipt_image)
            
            # Combine OCR text with expense description
            combined_text = f"{expense_description} {ocr_text}"
            
            # Extract key information using AI
            expense_details = await self._extract_expense_details(combined_text, ocr_text)
            
            # Classify expense category
            categories = [
                "travel", "meals", "office_supplies", "software", "equipment", 
                "marketing", "utilities", "professional_services", "training", "other"
            ]
            
            category_scores = await nlp.classify_text(combined_text, categories)
            predicted_category = max(category_scores.items(), key=lambda x: x[1])[0]
            
            # Validate expense amount
            extracted_amount = expense_details.get("amount", 0)
            declared_amount = expense_details.get("declared_amount", 0)
            
            amount_variance = abs(extracted_amount - declared_amount) / max(declared_amount, 1) if declared_amount > 0 else 0
            
            # Detect potential issues
            issues = []
            if amount_variance > 0.1:  # 10% variance threshold
                issues.append(f"Amount mismatch: Declared ${declared_amount}, Extracted ${extracted_amount}")
            
            if not expense_details.get("vendor_name"):
                issues.append("Vendor name not clearly identifiable")
            
            if not expense_details.get("date"):
                issues.append("Transaction date not found")
            
            # Calculate confidence score
            confidence_score = (
                category_scores.get(predicted_category, 0) * 0.4 +
                (1 - amount_variance) * 0.3 +
                (1 if expense_details.get("vendor_name") else 0) * 0.2 +
                (1 if expense_details.get("date") else 0) * 0.1
            )
            
            # Determine auto-approval eligibility
            business_rules = await get_business_rules()
            approval_decision = await business_rules.evaluate_expense_approval({
                "amount": extracted_amount or declared_amount,
                "category": predicted_category,
                "employee_level": "standard",  # Would get from employee data
                "has_receipt": True,
                "confidence_score": confidence_score
            })
            
            analysis = {
                "analysis_id": f"EA{str(uuid.uuid4())[:8].upper()}",
                "expense_id": expense_id,
                "ocr_text": ocr_text,
                "extracted_details": expense_details,
                "predicted_category": predicted_category,
                "category_confidence": category_scores.get(predicted_category, 0),
                "amount_variance": amount_variance,
                "confidence_score": confidence_score,
                "issues_detected": issues,
                "approval_decision": approval_decision,
                "requires_review": len(issues) > 0 or confidence_score < 0.7,
                "created_at": datetime.utcnow()
            }
            
            # Store analysis
            await self.db[self.expense_analysis_collection].insert_one(analysis)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="finance",
                event_type="expense_classified",
                entity_type="expense",
                entity_id=expense_id,
                data={
                    "category": predicted_category,
                    "confidence": confidence_score,
                    "auto_approved": approval_decision.get("auto_approve", False),
                    "issues_count": len(issues)
                }
            )
            
            logger.info(f"Expense classified: {expense_id} -> {predicted_category} (confidence: {confidence_score:.2f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to classify expense with OCR: {e}")
            return {}
    
    async def _extract_text_from_image(self, image_data: str) -> str:
        """Extract text from receipt image using OCR."""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Extract text using Tesseract OCR
            text = pytesseract.image_to_string(image)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}")
            return ""
    
    async def _extract_expense_details(self, combined_text: str, ocr_text: str) -> Dict[str, Any]:
        """Extract structured expense details using AI."""
        try:
            if not openai.api_key:
                return self._extract_details_with_regex(ocr_text)
            
            prompt = f"""
            Extract expense details from this receipt text and description:
            
            Text: {combined_text}
            
            Please extract and return a JSON object with these fields:
            {{
                "vendor_name": "Name of the vendor/merchant",
                "amount": 123.45,
                "date": "2024-01-15",
                "tax_amount": 12.34,
                "items": ["item1", "item2"],
                "payment_method": "credit_card/cash/check",
                "declared_amount": 123.45
            }}
            
            If a field cannot be determined, use null.
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            details = json.loads(response.choices[0].message.content)
            return details
            
        except Exception as e:
            logger.error(f"AI expense detail extraction failed: {e}")
            return self._extract_details_with_regex(ocr_text)
    
    def _extract_details_with_regex(self, text: str) -> Dict[str, Any]:
        """Fallback method to extract details using regex patterns."""
        try:
            details = {}
            
            # Extract amount patterns
            amount_patterns = [
                r'\$(\d+\.\d{2})',
                r'Total:?\s*\$?(\d+\.\d{2})',
                r'Amount:?\s*\$?(\d+\.\d{2})'
            ]
            
            for pattern in amount_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    details["amount"] = float(match.group(1))
                    break
            
            # Extract date patterns
            date_patterns = [
                r'(\d{1,2}/\d{1,2}/\d{4})',
                r'(\d{4}-\d{2}-\d{2})',
                r'(\d{1,2}-\d{1,2}-\d{4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    details["date"] = match.group(1)
                    break
            
            # Extract vendor name (first line usually)
            lines = text.split('\n')
            if lines:
                details["vendor_name"] = lines[0].strip()
            
            return details
            
        except Exception as e:
            logger.error(f"Regex detail extraction failed: {e}")
            return {}
    
    async def detect_invoice_fraud(self, invoice_id: str, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential fraud in invoices using ML and pattern analysis."""
        try:
            ml_predictor = await get_ml_predictor()
            data_lake = await get_data_lake()
            
            # Get historical invoice data for comparison
            historical_invoices = await self._get_historical_invoices(
                invoice_data.get("vendor_name", ""),
                days=365
            )
            
            # Analyze patterns and detect anomalies
            fraud_indicators = []
            risk_score = 0.0
            
            # Check amount anomalies
            amount_analysis = await self._analyze_invoice_amount(invoice_data, historical_invoices)
            if amount_analysis.get("is_anomaly", False):
                fraud_indicators.append("Unusual invoice amount compared to historical data")
                risk_score += 0.3
            
            # Check vendor consistency
            vendor_analysis = await self._analyze_vendor_consistency(invoice_data, historical_invoices)
            if vendor_analysis.get("inconsistencies", []):
                fraud_indicators.extend(vendor_analysis["inconsistencies"])
                risk_score += 0.2
            
            # Check duplicate detection
            duplicate_analysis = await self._check_duplicate_invoices(invoice_data)
            if duplicate_analysis.get("potential_duplicates", []):
                fraud_indicators.append("Potential duplicate invoice detected")
                risk_score += 0.4
            
            # Check timing patterns
            timing_analysis = await self._analyze_invoice_timing(invoice_data, historical_invoices)
            if timing_analysis.get("suspicious_timing", False):
                fraud_indicators.append("Suspicious invoice timing pattern")
                risk_score += 0.2
            
            # Check line item analysis
            line_item_analysis = await self._analyze_line_items(invoice_data)
            if line_item_analysis.get("suspicious_items", []):
                fraud_indicators.extend(line_item_analysis["suspicious_items"])
                risk_score += 0.1
            
            # Cap risk score at 1.0
            risk_score = min(risk_score, 1.0)
            
            # Determine risk level
            if risk_score >= 0.8:
                risk_level = "critical"
                recommended_action = "block_payment_investigate"
            elif risk_score >= 0.6:
                risk_level = "high"
                recommended_action = "manual_review_required"
            elif risk_score >= 0.4:
                risk_level = "medium"
                recommended_action = "additional_verification"
            else:
                risk_level = "low"
                recommended_action = "proceed_with_caution"
            
            detection_result = {
                "detection_id": f"FD{str(uuid.uuid4())[:8].upper()}",
                "invoice_id": invoice_id,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "fraud_indicators": fraud_indicators,
                "recommended_action": recommended_action,
                "amount_analysis": amount_analysis,
                "vendor_analysis": vendor_analysis,
                "duplicate_analysis": duplicate_analysis,
                "timing_analysis": timing_analysis,
                "line_item_analysis": line_item_analysis,
                "created_at": datetime.utcnow()
            }
            
            # Store detection result
            await self.db[self.fraud_detection_collection].insert_one(detection_result)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="finance",
                event_type="fraud_detection_completed",
                entity_type="invoice",
                entity_id=invoice_id,
                data={
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "indicators_count": len(fraud_indicators)
                }
            )
            
            # Generate alert for high-risk invoices
            if risk_score >= 0.6:
                await self._generate_fraud_alert(invoice_id, detection_result)
            
            logger.info(f"Fraud detection completed for invoice {invoice_id}: risk_score={risk_score:.2f}, level={risk_level}")
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Failed to detect invoice fraud: {e}")
            return {}
    
    async def _get_historical_invoices(self, vendor_name: str, days: int = 365) -> List[Dict[str, Any]]:
        """Get historical invoices for pattern analysis."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            invoices = await self.db["invoices"].find({
                "vendor_name": {"$regex": vendor_name, "$options": "i"},
                "created_at": {"$gte": start_date, "$lte": end_date}
            }).to_list(None)
            
            return invoices
            
        except Exception as e:
            logger.error(f"Failed to get historical invoices: {e}")
            return []
    
    async def _analyze_invoice_amount(self, invoice_data: Dict[str, Any], 
                                    historical_invoices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze invoice amount for anomalies."""
        try:
            current_amount = invoice_data.get("amount", 0)
            
            if not historical_invoices:
                return {"is_anomaly": False, "reason": "No historical data"}
            
            # Calculate statistics from historical data
            amounts = [inv.get("amount", 0) for inv in historical_invoices]
            avg_amount = sum(amounts) / len(amounts)
            
            # Calculate standard deviation
            variance = sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)
            std_dev = variance ** 0.5
            
            # Check if current amount is an outlier (more than 2 standard deviations)
            z_score = abs(current_amount - avg_amount) / std_dev if std_dev > 0 else 0
            is_anomaly = z_score > 2
            
            return {
                "is_anomaly": is_anomaly,
                "z_score": z_score,
                "historical_average": avg_amount,
                "current_amount": current_amount,
                "deviation_percentage": ((current_amount - avg_amount) / avg_amount * 100) if avg_amount > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze invoice amount: {e}")
            return {"is_anomaly": False}
    
    async def _analyze_vendor_consistency(self, invoice_data: Dict[str, Any], 
                                        historical_invoices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze vendor information consistency."""
        try:
            inconsistencies = []
            
            if not historical_invoices:
                return {"inconsistencies": []}
            
            # Check vendor name variations
            current_vendor = invoice_data.get("vendor_name", "").lower()
            historical_vendors = [inv.get("vendor_name", "").lower() for inv in historical_invoices]
            
            # Simple similarity check (in production, use more sophisticated matching)
            similar_vendors = [v for v in historical_vendors if v and current_vendor in v or v in current_vendor]
            
            if not similar_vendors and historical_vendors:
                inconsistencies.append("Vendor name significantly different from historical records")
            
            # Check payment terms consistency
            current_terms = invoice_data.get("payment_terms", "")
            historical_terms = [inv.get("payment_terms", "") for inv in historical_invoices if inv.get("payment_terms")]
            
            if current_terms and historical_terms:
                if current_terms not in historical_terms:
                    inconsistencies.append("Payment terms different from historical invoices")
            
            # Check bank account changes
            current_account = invoice_data.get("bank_account", "")
            historical_accounts = [inv.get("bank_account", "") for inv in historical_invoices if inv.get("bank_account")]
            
            if current_account and historical_accounts:
                if current_account not in historical_accounts:
                    inconsistencies.append("Bank account information changed")
            
            return {"inconsistencies": inconsistencies}
            
        except Exception as e:
            logger.error(f"Failed to analyze vendor consistency: {e}")
            return {"inconsistencies": []}
    
    async def _check_duplicate_invoices(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for potential duplicate invoices."""
        try:
            # Search for similar invoices
            search_criteria = {
                "vendor_name": invoice_data.get("vendor_name", ""),
                "amount": invoice_data.get("amount", 0),
                "created_at": {
                    "$gte": datetime.utcnow() - timedelta(days=90)  # Check last 90 days
                }
            }
            
            similar_invoices = await self.db["invoices"].find(search_criteria).to_list(None)
            
            potential_duplicates = []
            for invoice in similar_invoices:
                # Check if invoice number is similar
                current_number = invoice_data.get("invoice_number", "")
                historical_number = invoice.get("invoice_number", "")
                
                if current_number and historical_number:
                    if current_number == historical_number:
                        potential_duplicates.append({
                            "invoice_id": invoice.get("invoice_id"),
                            "reason": "Identical invoice number",
                            "similarity_score": 1.0
                        })
                    elif abs(len(current_number) - len(historical_number)) <= 1:
                        # Check for similar invoice numbers (typos, etc.)
                        similarity = self._calculate_string_similarity(current_number, historical_number)
                        if similarity > 0.8:
                            potential_duplicates.append({
                                "invoice_id": invoice.get("invoice_id"),
                                "reason": "Similar invoice number",
                                "similarity_score": similarity
                            })
            
            return {"potential_duplicates": potential_duplicates}
            
        except Exception as e:
            logger.error(f"Failed to check duplicate invoices: {e}")
            return {"potential_duplicates": []}
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        try:
            # Simple Levenshtein distance-based similarity
            if not str1 or not str2:
                return 0.0
            
            # Convert to lowercase for comparison
            str1, str2 = str1.lower(), str2.lower()
            
            if str1 == str2:
                return 1.0
            
            # Calculate Levenshtein distance
            len1, len2 = len(str1), len(str2)
            matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
            
            for i in range(len1 + 1):
                matrix[i][0] = i
            for j in range(len2 + 1):
                matrix[0][j] = j
            
            for i in range(1, len1 + 1):
                for j in range(1, len2 + 1):
                    cost = 0 if str1[i-1] == str2[j-1] else 1
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,      # deletion
                        matrix[i][j-1] + 1,      # insertion
                        matrix[i-1][j-1] + cost  # substitution
                    )
            
            distance = matrix[len1][len2]
            max_len = max(len1, len2)
            similarity = (max_len - distance) / max_len if max_len > 0 else 0
            
            return similarity
            
        except Exception as e:
            logger.error(f"Failed to calculate string similarity: {e}")
            return 0.0
    
    async def _analyze_invoice_timing(self, invoice_data: Dict[str, Any], 
                                    historical_invoices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze invoice timing patterns."""
        try:
            current_date = invoice_data.get("invoice_date", datetime.utcnow())
            if isinstance(current_date, str):
                current_date = datetime.fromisoformat(current_date)
            
            suspicious_timing = False
            timing_issues = []
            
            # Check if invoice is dated in the future
            if current_date > datetime.utcnow():
                suspicious_timing = True
                timing_issues.append("Invoice dated in the future")
            
            # Check if invoice is very old
            if current_date < datetime.utcnow() - timedelta(days=365):
                suspicious_timing = True
                timing_issues.append("Invoice is over a year old")
            
            # Check for unusual timing patterns with historical data
            if historical_invoices:
                historical_dates = []
                for inv in historical_invoices:
                    inv_date = inv.get("invoice_date")
                    if inv_date:
                        if isinstance(inv_date, str):
                            inv_date = datetime.fromisoformat(inv_date)
                        historical_dates.append(inv_date)
                
                if historical_dates:
                    # Check if current invoice timing is unusual
                    # (e.g., multiple invoices on the same day, unusual frequency)
                    same_day_invoices = [d for d in historical_dates if d.date() == current_date.date()]
                    if len(same_day_invoices) > 3:  # More than 3 invoices on same day
                        suspicious_timing = True
                        timing_issues.append("Multiple invoices on the same day")
            
            return {
                "suspicious_timing": suspicious_timing,
                "timing_issues": timing_issues
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze invoice timing: {e}")
            return {"suspicious_timing": False}
    
    async def _analyze_line_items(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze invoice line items for suspicious patterns."""
        try:
            line_items = invoice_data.get("line_items", [])
            suspicious_items = []
            
            if not line_items:
                return {"suspicious_items": []}
            
            for item in line_items:
                description = item.get("description", "").lower()
                amount = item.get("amount", 0)
                quantity = item.get("quantity", 1)
                
                # Check for suspicious descriptions
                suspicious_keywords = [
                    "cash", "gift card", "personal", "bonus", "tip", "gratuity",
                    "consulting fee", "miscellaneous", "other", "various"
                ]
                
                if any(keyword in description for keyword in suspicious_keywords):
                    suspicious_items.append(f"Suspicious item description: {description}")
                
                # Check for unusual amounts
                if amount > 10000:  # Very high amount for a line item
                    suspicious_items.append(f"Unusually high line item amount: ${amount}")
                
                # Check for unusual quantities
                if quantity > 1000:
                    suspicious_items.append(f"Unusually high quantity: {quantity}")
                
                # Check for round numbers (might indicate estimates rather than actual costs)
                if amount > 100 and amount % 100 == 0:
                    suspicious_items.append(f"Round number amount might indicate estimate: ${amount}")
            
            return {"suspicious_items": suspicious_items}
            
        except Exception as e:
            logger.error(f"Failed to analyze line items: {e}")
            return {"suspicious_items": []}
    
    async def _generate_fraud_alert(self, invoice_id: str, detection_result: Dict[str, Any]) -> None:
        """Generate fraud alert for high-risk invoices."""
        try:
            alert = {
                "alert_id": f"FRAUD_{str(uuid.uuid4())[:8].upper()}",
                "type": "invoice_fraud_risk",
                "invoice_id": invoice_id,
                "risk_score": detection_result.get("risk_score", 0),
                "risk_level": detection_result.get("risk_level", "unknown"),
                "fraud_indicators": detection_result.get("fraud_indicators", []),
                "recommended_action": detection_result.get("recommended_action", "review"),
                "created_at": datetime.utcnow(),
                "status": "active",
                "assigned_to": "finance_manager"
            }
            
            # Store alert
            await self.db["fraud_alerts"].insert_one(alert)
            
            # In a real system, this would send notifications
            logger.warning(f"Fraud alert generated for invoice {invoice_id}: risk_level={detection_result.get('risk_level')}")
            
        except Exception as e:
            logger.error(f"Failed to generate fraud alert: {e}")
    
    async def predict_cash_flow(self, months_ahead: int = 12) -> Dict[str, Any]:
        """Predict cash flow for the specified number of months ahead."""
        try:
            data_lake = await get_data_lake()
            
            # Get historical financial data
            historical_data = await self._get_historical_financial_data(months=24)
            
            # Analyze patterns and trends
            revenue_analysis = await self._analyze_revenue_patterns(historical_data)
            expense_analysis = await self._analyze_expense_patterns(historical_data)
            
            # Generate predictions
            predictions = []
            current_date = datetime.utcnow()
            
            for month in range(1, months_ahead + 1):
                prediction_date = current_date + timedelta(days=30 * month)
                
                # Predict revenue
                predicted_revenue = await self._predict_monthly_revenue(
                    prediction_date, revenue_analysis, historical_data
                )
                
                # Predict expenses
                predicted_expenses = await self._predict_monthly_expenses(
                    prediction_date, expense_analysis, historical_data
                )
                
                # Calculate net cash flow
                net_cash_flow = predicted_revenue - predicted_expenses
                
                predictions.append({
                    "month": prediction_date.strftime("%Y-%m"),
                    "predicted_revenue": predicted_revenue,
                    "predicted_expenses": predicted_expenses,
                    "net_cash_flow": net_cash_flow,
                    "confidence_score": self._calculate_prediction_confidence(month, historical_data)
                })
            
            # Calculate cumulative cash flow
            cumulative_cash_flow = 0
            for prediction in predictions:
                cumulative_cash_flow += prediction["net_cash_flow"]
                prediction["cumulative_cash_flow"] = cumulative_cash_flow
            
            # Identify potential cash flow issues
            cash_flow_alerts = []
            for prediction in predictions:
                if prediction["net_cash_flow"] < -10000:  # Negative cash flow threshold
                    cash_flow_alerts.append({
                        "month": prediction["month"],
                        "issue": "Negative cash flow predicted",
                        "amount": prediction["net_cash_flow"],
                        "severity": "high" if prediction["net_cash_flow"] < -50000 else "medium"
                    })
            
            prediction_result = {
                "prediction_id": f"CF{str(uuid.uuid4())[:8].upper()}",
                "prediction_date": datetime.utcnow(),
                "months_ahead": months_ahead,
                "predictions": predictions,
                "revenue_analysis": revenue_analysis,
                "expense_analysis": expense_analysis,
                "cash_flow_alerts": cash_flow_alerts,
                "recommendations": await self._generate_cash_flow_recommendations(predictions, cash_flow_alerts)
            }
            
            # Store prediction
            await self.db[self.cash_flow_predictions_collection].insert_one(prediction_result)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="finance",
                event_type="cash_flow_predicted",
                entity_type="prediction",
                entity_id=prediction_result["prediction_id"],
                data={
                    "months_ahead": months_ahead,
                    "alerts_count": len(cash_flow_alerts),
                    "avg_monthly_cash_flow": sum(p["net_cash_flow"] for p in predictions) / len(predictions)
                }
            )
            
            logger.info(f"Cash flow prediction completed: {months_ahead} months ahead, {len(cash_flow_alerts)} alerts")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Failed to predict cash flow: {e}")
            return {}
    
    async def _get_historical_financial_data(self, months: int = 24) -> Dict[str, Any]:
        """Get historical financial data for analysis."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30 * months)
            
            # Get invoices (revenue)
            invoices = await self.db["invoices"].find({
                "status": "paid",
                "paid_at": {"$gte": start_date, "$lte": end_date}
            }).to_list(None)
            
            # Get expenses
            expenses = await self.db["expenses"].find({
                "status": "paid",
                "expense_date": {"$gte": start_date.date(), "$lte": end_date.date()}
            }).to_list(None)
            
            # Group by month
            monthly_data = {}
            
            # Process invoices
            for invoice in invoices:
                paid_date = invoice.get("paid_at", datetime.utcnow())
                month_key = paid_date.strftime("%Y-%m")
                
                if month_key not in monthly_data:
                    monthly_data[month_key] = {"revenue": 0, "expenses": 0}
                
                monthly_data[month_key]["revenue"] += invoice.get("amount", 0)
            
            # Process expenses
            for expense in expenses:
                expense_date = expense.get("expense_date", datetime.utcnow().date())
                if isinstance(expense_date, str):
                    expense_date = datetime.fromisoformat(expense_date).date()
                
                month_key = expense_date.strftime("%Y-%m")
                
                if month_key not in monthly_data:
                    monthly_data[month_key] = {"revenue": 0, "expenses": 0}
                
                monthly_data[month_key]["expenses"] += expense.get("amount", 0)
            
            return {
                "monthly_data": monthly_data,
                "total_months": len(monthly_data),
                "date_range": {"start": start_date, "end": end_date}
            }
            
        except Exception as e:
            logger.error(f"Failed to get historical financial data: {e}")
            return {"monthly_data": {}, "total_months": 0}
    
    async def _analyze_revenue_patterns(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze revenue patterns and trends."""
        try:
            monthly_data = historical_data.get("monthly_data", {})
            
            if not monthly_data:
                return {"trend": "insufficient_data"}
            
            # Extract revenue values
            revenues = [data.get("revenue", 0) for data in monthly_data.values()]
            
            # Calculate trend
            if len(revenues) >= 2:
                # Simple linear trend calculation
                x_values = list(range(len(revenues)))
                n = len(revenues)
                
                sum_x = sum(x_values)
                sum_y = sum(revenues)
                sum_xy = sum(x * y for x, y in zip(x_values, revenues))
                sum_x2 = sum(x * x for x in x_values)
                
                # Calculate slope (trend)
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
                
                trend = "increasing" if slope > 100 else "decreasing" if slope < -100 else "stable"
            else:
                trend = "insufficient_data"
                slope = 0
            
            # Calculate statistics
            avg_revenue = sum(revenues) / len(revenues) if revenues else 0
            max_revenue = max(revenues) if revenues else 0
            min_revenue = min(revenues) if revenues else 0
            
            # Calculate seasonality (simple month-over-month comparison)
            seasonality = {}
            if len(revenues) >= 12:
                for i in range(12):
                    month_revenues = [revenues[j] for j in range(i, len(revenues), 12)]
                    if month_revenues:
                        seasonality[f"month_{i+1}"] = sum(month_revenues) / len(month_revenues)
            
            return {
                "trend": trend,
                "slope": slope,
                "avg_monthly_revenue": avg_revenue,
                "max_monthly_revenue": max_revenue,
                "min_monthly_revenue": min_revenue,
                "revenue_volatility": (max_revenue - min_revenue) / avg_revenue if avg_revenue > 0 else 0,
                "seasonality": seasonality
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze revenue patterns: {e}")
            return {"trend": "analysis_failed"}
    
    async def _analyze_expense_patterns(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze expense patterns and trends."""
        try:
            monthly_data = historical_data.get("monthly_data", {})
            
            if not monthly_data:
                return {"trend": "insufficient_data"}
            
            # Extract expense values
            expenses = [data.get("expenses", 0) for data in monthly_data.values()]
            
            # Calculate trend (similar to revenue analysis)
            if len(expenses) >= 2:
                x_values = list(range(len(expenses)))
                n = len(expenses)
                
                sum_x = sum(x_values)
                sum_y = sum(expenses)
                sum_xy = sum(x * y for x, y in zip(x_values, expenses))
                sum_x2 = sum(x * x for x in x_values)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
                trend = "increasing" if slope > 100 else "decreasing" if slope < -100 else "stable"
            else:
                trend = "insufficient_data"
                slope = 0
            
            # Calculate statistics
            avg_expenses = sum(expenses) / len(expenses) if expenses else 0
            max_expenses = max(expenses) if expenses else 0
            min_expenses = min(expenses) if expenses else 0
            
            return {
                "trend": trend,
                "slope": slope,
                "avg_monthly_expenses": avg_expenses,
                "max_monthly_expenses": max_expenses,
                "min_monthly_expenses": min_expenses,
                "expense_volatility": (max_expenses - min_expenses) / avg_expenses if avg_expenses > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze expense patterns: {e}")
            return {"trend": "analysis_failed"}
    
    async def _predict_monthly_revenue(self, prediction_date: datetime, 
                                     revenue_analysis: Dict[str, Any], 
                                     historical_data: Dict[str, Any]) -> float:
        """Predict revenue for a specific month."""
        try:
            base_revenue = revenue_analysis.get("avg_monthly_revenue", 0)
            trend_slope = revenue_analysis.get("slope", 0)
            
            # Apply trend
            months_from_now = (prediction_date.year - datetime.utcnow().year) * 12 + (prediction_date.month - datetime.utcnow().month)
            trend_adjustment = trend_slope * months_from_now
            
            # Apply seasonality if available
            seasonality = revenue_analysis.get("seasonality", {})
            seasonal_factor = seasonality.get(f"month_{prediction_date.month}", base_revenue)
            seasonal_adjustment = (seasonal_factor - base_revenue) * 0.3  # 30% weight to seasonality
            
            predicted_revenue = base_revenue + trend_adjustment + seasonal_adjustment
            
            # Ensure non-negative
            return max(0, predicted_revenue)
            
        except Exception as e:
            logger.error(f"Failed to predict monthly revenue: {e}")
            return 0
    
    async def _predict_monthly_expenses(self, prediction_date: datetime, 
                                      expense_analysis: Dict[str, Any], 
                                      historical_data: Dict[str, Any]) -> float:
        """Predict expenses for a specific month."""
        try:
            base_expenses = expense_analysis.get("avg_monthly_expenses", 0)
            trend_slope = expense_analysis.get("slope", 0)
            
            # Apply trend
            months_from_now = (prediction_date.year - datetime.utcnow().year) * 12 + (prediction_date.month - datetime.utcnow().month)
            trend_adjustment = trend_slope * months_from_now
            
            predicted_expenses = base_expenses + trend_adjustment
            
            # Ensure non-negative
            return max(0, predicted_expenses)
            
        except Exception as e:
            logger.error(f"Failed to predict monthly expenses: {e}")
            return 0
    
    def _calculate_prediction_confidence(self, months_ahead: int, historical_data: Dict[str, Any]) -> float:
        """Calculate confidence score for predictions."""
        try:
            # Confidence decreases with time and increases with more historical data
            data_months = historical_data.get("total_months", 0)
            
            # Base confidence based on historical data availability
            base_confidence = min(data_months / 24, 1.0)  # Max confidence with 24 months of data
            
            # Decrease confidence for longer predictions
            time_penalty = max(0, 1 - (months_ahead - 1) * 0.1)  # 10% decrease per month
            
            confidence = base_confidence * time_penalty
            return max(0.1, min(1.0, confidence))  # Keep between 0.1 and 1.0
            
        except Exception as e:
            logger.error(f"Failed to calculate prediction confidence: {e}")
            return 0.5
    
    async def _generate_cash_flow_recommendations(self, predictions: List[Dict[str, Any]], 
                                                alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate cash flow management recommendations."""
        try:
            recommendations = []
            
            # Analyze predictions for patterns
            negative_months = [p for p in predictions if p["net_cash_flow"] < 0]
            
            if negative_months:
                recommendations.append(f"Prepare for {len(negative_months)} months with negative cash flow")
                
                # Find the worst month
                worst_month = min(negative_months, key=lambda x: x["net_cash_flow"])
                recommendations.append(f"Worst predicted month: {worst_month['month']} with ${worst_month['net_cash_flow']:,.2f}")
                
                # Suggest actions
                recommendations.extend([
                    "Consider establishing a line of credit",
                    "Review and potentially delay non-essential expenses",
                    "Accelerate accounts receivable collection",
                    "Explore additional revenue opportunities"
                ])
            
            # Check for consistent growth
            growth_months = [p for p in predictions if p["net_cash_flow"] > 0]
            if len(growth_months) > len(predictions) * 0.8:  # 80% positive months
                recommendations.append("Strong cash flow predicted - consider investment opportunities")
            
            # Seasonal recommendations
            if len(predictions) >= 12:
                q4_predictions = [p for p in predictions if p["month"].endswith(("10", "11", "12"))]
                if q4_predictions:
                    avg_q4_flow = sum(p["net_cash_flow"] for p in q4_predictions) / len(q4_predictions)
                    if avg_q4_flow > 0:
                        recommendations.append("Q4 shows strong cash flow - good time for year-end investments")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate cash flow recommendations: {e}")
            return ["Unable to generate recommendations"]
    
    async def generate_tax_suggestions(self, tax_year: int) -> Dict[str, Any]:
        """Generate automated tax filing suggestions and optimizations."""
        try:
            data_lake = await get_data_lake()
            
            # Get financial data for the tax year
            start_date = datetime(tax_year, 1, 1)
            end_date = datetime(tax_year, 12, 31)
            
            # Get all expenses for the tax year
            expenses = await self.db["expenses"].find({
                "expense_date": {
                    "$gte": start_date.date(),
                    "$lte": end_date.date()
                },
                "status": {"$in": ["approved", "paid"]}
            }).to_list(None)
            
            # Get all revenue for the tax year
            invoices = await self.db["invoices"].find({
                "issue_date": {
                    "$gte": start_date.date(),
                    "$lte": end_date.date()
                },
                "status": "paid"
            }).to_list(None)
            
            # Categorize expenses for tax purposes
            tax_categories = await self._categorize_expenses_for_tax(expenses)
            
            # Calculate deductions
            deductions = await self._calculate_tax_deductions(tax_categories)
            
            # Calculate total revenue
            total_revenue = sum(inv.get("amount", 0) for inv in invoices)
            
            # Generate tax optimization suggestions
            optimization_suggestions = await self._generate_tax_optimizations(
                tax_categories, deductions, total_revenue, tax_year
            )
            
            # Check for missing documentation
            documentation_issues = await self._check_tax_documentation(expenses, invoices)
            
            # Calculate estimated tax liability (simplified)
            estimated_tax = await self._estimate_tax_liability(total_revenue, deductions)
            
            tax_suggestions = {
                "suggestion_id": f"TAX{str(uuid.uuid4())[:8].upper()}",
                "tax_year": tax_year,
                "total_revenue": total_revenue,
                "tax_categories": tax_categories,
                "total_deductions": sum(deductions.values()),
                "deduction_breakdown": deductions,
                "estimated_tax_liability": estimated_tax,
                "optimization_suggestions": optimization_suggestions,
                "documentation_issues": documentation_issues,
                "compliance_checklist": await self._generate_compliance_checklist(tax_year),
                "generated_at": datetime.utcnow()
            }
            
            # Store suggestions
            await self.db[self.tax_suggestions_collection].insert_one(tax_suggestions)
            
            # Store event in data lake
            await data_lake.store_event(
                agent="finance",
                event_type="tax_suggestions_generated",
                entity_type="tax_filing",
                entity_id=tax_suggestions["suggestion_id"],
                data={
                    "tax_year": tax_year,
                    "total_deductions": sum(deductions.values()),
                    "optimization_count": len(optimization_suggestions),
                    "documentation_issues": len(documentation_issues)
                }
            )
            
            logger.info(f"Tax suggestions generated for {tax_year}: ${sum(deductions.values()):,.2f} in deductions")
            
            return tax_suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate tax suggestions: {e}")
            return {}
    
    async def _categorize_expenses_for_tax(self, expenses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize expenses for tax purposes."""
        try:
            tax_categories = {
                "office_expenses": [],
                "travel_expenses": [],
                "meals_entertainment": [],
                "professional_services": [],
                "equipment_depreciation": [],
                "software_subscriptions": [],
                "marketing_advertising": [],
                "utilities": [],
                "rent_lease": [],
                "insurance": [],
                "training_education": [],
                "other_deductible": [],
                "non_deductible": []
            }
            
            for expense in expenses:
                category = expense.get("category", "other").lower()
                amount = expense.get("amount", 0)
                description = expense.get("description", "").lower()
                
                # Map expense categories to tax categories
                if category in ["office_supplies", "supplies"]:
                    tax_categories["office_expenses"].append(expense)
                elif category == "travel":
                    tax_categories["travel_expenses"].append(expense)
                elif category in ["meals", "entertainment"]:
                    # Meals are typically 50% deductible
                    tax_categories["meals_entertainment"].append(expense)
                elif category in ["software", "saas", "subscriptions"]:
                    tax_categories["software_subscriptions"].append(expense)
                elif category in ["equipment", "hardware"]:
                    if amount > 2500:  # Threshold for depreciation vs. immediate expense
                        tax_categories["equipment_depreciation"].append(expense)
                    else:
                        tax_categories["office_expenses"].append(expense)
                elif category == "marketing":
                    tax_categories["marketing_advertising"].append(expense)
                elif category == "utilities":
                    tax_categories["utilities"].append(expense)
                elif "insurance" in description:
                    tax_categories["insurance"].append(expense)
                elif "training" in description or "education" in description:
                    tax_categories["training_education"].append(expense)
                elif "legal" in description or "accounting" in description or "consulting" in description:
                    tax_categories["professional_services"].append(expense)
                elif "rent" in description or "lease" in description:
                    tax_categories["rent_lease"].append(expense)
                else:
                    # Check if expense is likely deductible
                    if self._is_likely_deductible(expense):
                        tax_categories["other_deductible"].append(expense)
                    else:
                        tax_categories["non_deductible"].append(expense)
            
            return tax_categories
            
        except Exception as e:
            logger.error(f"Failed to categorize expenses for tax: {e}")
            return {}
    
    def _is_likely_deductible(self, expense: Dict[str, Any]) -> bool:
        """Determine if an expense is likely tax deductible."""
        try:
            description = expense.get("description", "").lower()
            
            # Common business expense keywords
            deductible_keywords = [
                "business", "office", "work", "client", "meeting", "conference",
                "professional", "service", "maintenance", "repair", "supplies"
            ]
            
            # Non-deductible keywords
            non_deductible_keywords = [
                "personal", "gift", "entertainment", "fine", "penalty", "political"
            ]
            
            # Check for non-deductible keywords first
            if any(keyword in description for keyword in non_deductible_keywords):
                return False
            
            # Check for deductible keywords
            if any(keyword in description for keyword in deductible_keywords):
                return True
            
            # Default to potentially deductible for business expenses
            return True
            
        except Exception as e:
            logger.error(f"Failed to determine deductibility: {e}")
            return False
    
    async def _calculate_tax_deductions(self, tax_categories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate tax deductions by category."""
        try:
            deductions = {}
            
            for category, expenses in tax_categories.items():
                if category == "non_deductible":
                    continue
                
                total_amount = sum(exp.get("amount", 0) for exp in expenses)
                
                # Apply category-specific rules
                if category == "meals_entertainment":
                    # Meals are typically 50% deductible
                    deductions[category] = total_amount * 0.5
                elif category == "equipment_depreciation":
                    # Simplified depreciation calculation (would be more complex in reality)
                    deductions[category] = total_amount * 0.2  # 20% first year depreciation
                else:
                    deductions[category] = total_amount
            
            return deductions
            
        except Exception as e:
            logger.error(f"Failed to calculate tax deductions: {e}")
            return {}
    
    async def _generate_tax_optimizations(self, tax_categories: Dict[str, List[Dict[str, Any]]], 
                                        deductions: Dict[str, float], total_revenue: float, 
                                        tax_year: int) -> List[str]:
        """Generate tax optimization suggestions."""
        try:
            suggestions = []
            
            # Check for missed deductions
            if deductions.get("office_expenses", 0) < total_revenue * 0.02:  # Less than 2% of revenue
                suggestions.append("Consider reviewing office expenses - they seem low compared to revenue")
            
            if not deductions.get("professional_services", 0):
                suggestions.append("No professional services expenses found - consider accounting/legal fees")
            
            # Equipment depreciation opportunities
            equipment_expenses = tax_categories.get("equipment_depreciation", [])
            if equipment_expenses:
                total_equipment = sum(exp.get("amount", 0) for exp in equipment_expenses)
                if total_equipment > 10000:
                    suggestions.append("Consider Section 179 deduction for equipment purchases over $10,000")
            
            # Home office deduction
            office_rent = deductions.get("rent_lease", 0)
            if office_rent == 0:
                suggestions.append("If working from home, consider home office deduction")
            
            # Business vehicle expenses
            travel_expenses = deductions.get("travel_expenses", 0)
            if travel_expenses > 5000:
                suggestions.append("High travel expenses - ensure proper documentation for vehicle deductions")
            
            # Retirement contributions
            suggestions.append("Consider maximizing retirement plan contributions for additional deductions")
            
            # Year-end planning
            current_year = datetime.utcnow().year
            if tax_year == current_year:
                suggestions.append("Consider accelerating deductible expenses before year-end")
                suggestions.append("Review accounts receivable - consider delaying some collections to next year")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate tax optimizations: {e}")
            return []
    
    async def _check_tax_documentation(self, expenses: List[Dict[str, Any]], 
                                     invoices: List[Dict[str, Any]]) -> List[str]:
        """Check for missing tax documentation."""
        try:
            issues = []
            
            # Check expenses for missing receipts
            missing_receipts = [exp for exp in expenses if not exp.get("receipt_url") and exp.get("amount", 0) > 75]
            if missing_receipts:
                issues.append(f"{len(missing_receipts)} expenses over $75 missing receipts")
            
            # Check for missing vendor information
            missing_vendor = [exp for exp in expenses if not exp.get("vendor_name")]
            if missing_vendor:
                issues.append(f"{len(missing_vendor)} expenses missing vendor information")
            
            # Check invoices for missing client information
            missing_client_info = [inv for inv in invoices if not inv.get("client_name") or not inv.get("client_email")]
            if missing_client_info:
                issues.append(f"{len(missing_client_info)} invoices missing complete client information")
            
            # Check for large cash transactions
            large_cash = [exp for exp in expenses if exp.get("payment_method") == "cash" and exp.get("amount", 0) > 500]
            if large_cash:
                issues.append(f"{len(large_cash)} large cash transactions may need additional documentation")
            
            return issues
            
        except Exception as e:
            logger.error(f"Failed to check tax documentation: {e}")
            return []
    
    async def _estimate_tax_liability(self, total_revenue: float, deductions: Dict[str, float]) -> Dict[str, float]:
        """Estimate tax liability (simplified calculation)."""
        try:
            total_deductions = sum(deductions.values())
            taxable_income = max(0, total_revenue - total_deductions)
            
            # Simplified tax calculation (would use actual tax brackets in reality)
            federal_rate = 0.21  # Corporate tax rate
            state_rate = 0.06   # Average state tax rate
            self_employment_rate = 0.1413  # Self-employment tax rate
            
            federal_tax = taxable_income * federal_rate
            state_tax = taxable_income * state_rate
            self_employment_tax = total_revenue * self_employment_rate  # On gross income
            
            return {
                "taxable_income": taxable_income,
                "federal_tax": federal_tax,
                "state_tax": state_tax,
                "self_employment_tax": self_employment_tax,
                "total_estimated_tax": federal_tax + state_tax + self_employment_tax
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate tax liability: {e}")
            return {}
    
    async def _generate_compliance_checklist(self, tax_year: int) -> List[Dict[str, Any]]:
        """Generate tax compliance checklist."""
        try:
            checklist = [
                {
                    "item": "Gather all 1099 forms received",
                    "deadline": f"{tax_year + 1}-01-31",
                    "status": "pending",
                    "priority": "high"
                },
                {
                    "item": "Organize receipts and expense documentation",
                    "deadline": f"{tax_year + 1}-04-15",
                    "status": "pending",
                    "priority": "high"
                },
                {
                    "item": "Calculate quarterly estimated tax payments",
                    "deadline": f"{tax_year + 1}-01-15",
                    "status": "pending",
                    "priority": "medium"
                },
                {
                    "item": "Review depreciation schedules",
                    "deadline": f"{tax_year + 1}-03-15",
                    "status": "pending",
                    "priority": "medium"
                },
                {
                    "item": "File federal tax return",
                    "deadline": f"{tax_year + 1}-04-15",
                    "status": "pending",
                    "priority": "critical"
                },
                {
                    "item": "File state tax return",
                    "deadline": f"{tax_year + 1}-04-15",
                    "status": "pending",
                    "priority": "critical"
                },
                {
                    "item": "Make final tax payment if owed",
                    "deadline": f"{tax_year + 1}-04-15",
                    "status": "pending",
                    "priority": "critical"
                }
            ]
            
            return checklist
            
        except Exception as e:
            logger.error(f"Failed to generate compliance checklist: {e}")
            return []
    
    async def get_finance_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive finance analytics and insights."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get expense analyses
            expense_analyses = await self.db[self.expense_analysis_collection].find({
                "created_at": {"$gte": start_date, "$lte": end_date}
            }).to_list(None)
            
            # Get fraud detections
            fraud_detections = await self.db[self.fraud_detection_collection].find({
                "created_at": {"$gte": start_date, "$lte": end_date}
            }).to_list(None)
            
            # Get cash flow predictions
            cash_flow_predictions = await self.db[self.cash_flow_predictions_collection].find({
                "prediction_date": {"$gte": start_date, "$lte": end_date}
            }).to_list(None)
            
            # Calculate metrics
            total_expenses_analyzed = len(expense_analyses)
            auto_approved_expenses = len([a for a in expense_analyses if a.get("approval_decision", {}).get("auto_approve", False)])
            
            fraud_cases_detected = len(fraud_detections)
            high_risk_invoices = len([f for f in fraud_detections if f.get("risk_level") == "high"])
            
            avg_expense_confidence = sum(a.get("confidence_score", 0) for a in expense_analyses) / len(expense_analyses) if expense_analyses else 0
            
            # Category distribution
            category_distribution = {}
            for analysis in expense_analyses:
                category = analysis.get("predicted_category", "unknown")
                category_distribution[category] = category_distribution.get(category, 0) + 1
            
            # Fraud risk distribution
            fraud_risk_distribution = {}
            for detection in fraud_detections:
                risk_level = detection.get("risk_level", "unknown")
                fraud_risk_distribution[risk_level] = fraud_risk_distribution.get(risk_level, 0) + 1
            
            analytics = {
                "period_days": days,
                "expense_analysis": {
                    "total_analyzed": total_expenses_analyzed,
                    "auto_approved": auto_approved_expenses,
                    "auto_approval_rate": (auto_approved_expenses / total_expenses_analyzed) if total_expenses_analyzed > 0 else 0,
                    "avg_confidence_score": round(avg_confidence_confidence, 3),
                    "category_distribution": category_distribution
                },
                "fraud_detection": {
                    "total_cases": fraud_cases_detected,
                    "high_risk_cases": high_risk_invoices,
                    "risk_distribution": fraud_risk_distribution
                },
                "cash_flow": {
                    "predictions_generated": len(cash_flow_predictions),
                    "latest_prediction": cash_flow_predictions[-1] if cash_flow_predictions else None
                },
                "efficiency_metrics": {
                    "processing_automation_rate": (auto_approved_expenses / total_expenses_analyzed) if total_expenses_analyzed > 0 else 0,
                    "fraud_detection_coverage": (fraud_cases_detected / total_expenses_analyzed) if total_expenses_analyzed > 0 else 0
                },
                "generated_at": datetime.utcnow()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get finance analytics: {e}")
            return {}
