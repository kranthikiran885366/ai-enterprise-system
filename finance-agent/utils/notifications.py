"""Notification utilities for Finance Agent."""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
from loguru import logger
import httpx


async def send_expense_notification(email: str, notification_type: str, data: Dict[str, Any]) -> bool:
    """Send expense-related notification."""
    try:
        templates = {
            "expense_auto_approved": {
                "subject": "Expense Auto-Approved - {expense_id}",
                "template": """
                Dear Employee,

                Your expense has been automatically approved:

                Expense ID: {expense_id}
                Amount: ${amount}
                Reason: {reason}

                The expense will be processed in the next payroll cycle.

                Best regards,
                Finance Team
                """
            },
            "expense_approval_required": {
                "subject": "Expense Approval Required - {expense_id}",
                "template": """
                Dear Approver,

                An expense requires your approval:

                Employee: {employee_name}
                Expense ID: {expense_id}
                Amount: ${amount}
                Category: {category}
                Description: {description}
                Approval Level: {approval_level}

                Please review and approve/reject in the system.

                Best regards,
                Finance System
                """
            },
            "suspicious_expense_detected": {
                "subject": "ALERT: Suspicious Expense Detected - {expense_id}",
                "template": """
                Dear Finance Audit Team,

                A suspicious expense has been detected:

                Expense ID: {expense_id}
                Employee ID: {employee_id}
                Amount: ${amount}
                Risk Score: {risk_score}
                Risk Factors: {risk_factors}

                Please investigate immediately.

                Best regards,
                AI Finance System
                """
            },
            "expense_rejected": {
                "subject": "Expense Rejected - {expense_id}",
                "template": """
                Dear Employee,

                Your expense has been rejected:

                Expense ID: {expense_id}
                Amount: ${amount}
                Reason: {rejection_reason}

                Please contact your manager for clarification.

                Best regards,
                Finance Team
                """
            }
        }
        
        template_data = templates.get(notification_type)
        if not template_data:
            logger.warning(f"Unknown expense notification type: {notification_type}")
            return False
        
        # Format template
        subject = template_data["subject"].format(**data)
        body = template_data["template"].format(**data)
        
        # Send email
        success = await _send_email(email, subject, body)
        
        if success:
            logger.info(f"Expense notification sent to {email}: {notification_type}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to send expense notification: {e}")
        return False


async def send_invoice_notification(email: str, notification_type: str, data: Dict[str, Any]) -> bool:
    """Send invoice-related notification."""
    try:
        templates = {
            "invoice_created": {
                "subject": "Invoice Created - {invoice_id}",
                "template": """
                Dear {client_name},

                A new invoice has been created for you:

                Invoice ID: {invoice_id}
                Amount: ${amount}
                Due Date: {due_date}
                Description: {description}

                Please remit payment by the due date.

                Best regards,
                Finance Team
                """
            },
            "invoice_overdue": {
                "subject": "OVERDUE: Invoice {invoice_id}",
                "template": """
                Dear {client_name},

                Your invoice is now overdue:

                Invoice ID: {invoice_id}
                Amount: ${amount}
                Original Due Date: {due_date}
                Days Overdue: {days_overdue}

                Please remit payment immediately to avoid late fees.

                Best regards,
                Finance Team
                """
            },
            "payment_received": {
                "subject": "Payment Received - {invoice_id}",
                "template": """
                Dear {client_name},

                We have received your payment:

                Invoice ID: {invoice_id}
                Amount Paid: ${amount}
                Payment Date: {payment_date}

                Thank you for your business!

                Best regards,
                Finance Team
                """
            }
        }
        
        template_data = templates.get(notification_type)
        if not template_data:
            logger.warning(f"Unknown invoice notification type: {notification_type}")
            return False
        
        # Format template
        subject = template_data["subject"].format(**data)
        body = template_data["template"].format(**data)
        
        # Send email
        success = await _send_email(email, subject, body)
        
        if success:
            logger.info(f"Invoice notification sent to {email}: {notification_type}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to send invoice notification: {e}")
        return False


async def _send_email(to_email: str, subject: str, body: str) -> bool:
    """Send email using configured email service."""
    try:
        email_service = os.getenv("EMAIL_SERVICE", "log")
        
        if email_service == "log":
            logger.info(f"EMAIL TO: {to_email}")
            logger.info(f"SUBJECT: {subject}")
            logger.info(f"BODY: {body}")
            return True
        
        # Add SMTP implementation here for production
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False