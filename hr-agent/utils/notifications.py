"""Notification utilities for HR Agent."""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
from loguru import logger
import httpx


async def send_employee_notification(email: str, notification_type: str, data: Dict[str, Any]) -> bool:
    """Send notification to employee."""
    try:
        templates = {
            "welcome": {
                "subject": "Welcome to the Company!",
                "template": """
                Dear {name},

                Welcome to our company! We're excited to have you join our team.

                Your employee ID is: {employee_id}
                Your start date is: {start_date}

                Please check your email for onboarding instructions.

                Best regards,
                HR Team
                """
            },
            "profile_update": {
                "subject": "Profile Updated",
                "template": """
                Dear {name},

                Your employee profile has been updated with the following changes:
                {changes}

                If you have any questions, please contact HR.

                Best regards,
                HR Team
                """
            },
            "offboarding": {
                "subject": "Offboarding Process",
                "template": """
                Dear {name},

                We're sorry to see you go. Your last day with us is {last_day}.

                Please complete the offboarding checklist and return all company property.

                Best wishes for your future endeavors,
                HR Team
                """
            }
        }
        
        template_data = templates.get(notification_type)
        if not template_data:
            logger.warning(f"Unknown notification type: {notification_type}")
            return False
        
        # Format template
        subject = template_data["subject"]
        body = template_data["template"].format(**data)
        
        # Send email (simplified - in production use proper email service)
        success = await _send_email(email, subject, body)
        
        if success:
            logger.info(f"Notification sent to {email}: {notification_type}")
        else:
            logger.error(f"Failed to send notification to {email}: {notification_type}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to send employee notification: {e}")
        return False


async def send_recruitment_notification(email: str, notification_type: str, data: Dict[str, Any]) -> bool:
    """Send recruitment-related notification."""
    try:
        templates = {
            "application_received": {
                "subject": "Application Received - {job_title}",
                "template": """
                Dear {candidate_name},

                Thank you for your interest in the {job_title} position at our company.

                We have received your application (ID: {application_id}) and will review it shortly.
                
                Our AI screening system will analyze your application, and we'll contact you within 5 business days.

                Best regards,
                Recruitment Team
                """
            },
            "interview_scheduled": {
                "subject": "Interview Scheduled - {job_title}",
                "template": """
                Dear {candidate_name},

                Congratulations! We would like to invite you for an interview for the {job_title} position.

                Interview Details:
                - Date: {interview_date}
                - Time: {interview_time}
                - Type: {interview_type}
                - Duration: {duration}

                Please confirm your availability.

                Best regards,
                Recruitment Team
                """
            },
            "candidate_shortlisted": {
                "subject": "Candidate Auto-Shortlisted",
                "template": """
                Dear Hiring Manager,

                A candidate has been automatically shortlisted by our AI system:

                Candidate ID: {candidate_id}
                AI Score: {score}
                Recommendation: {recommendation}
                Key Strengths: {strengths}

                Please review and schedule an interview.

                Best regards,
                AI Recruitment System
                """
            },
            "interview_completed": {
                "subject": "AI Interview Completed",
                "template": """
                Dear HR Team,

                An AI interview has been completed:

                Candidate ID: {candidate_id}
                Interview Type: {interview_type}
                Overall Score: {overall_score}
                Recommendation: {recommendation}

                Please review the detailed results in the system.

                Best regards,
                AI Interview System
                """
            }
        }
        
        template_data = templates.get(notification_type)
        if not template_data:
            logger.warning(f"Unknown recruitment notification type: {notification_type}")
            return False
        
        # Format template
        subject = template_data["subject"].format(**data)
        body = template_data["template"].format(**data)
        
        # Send email
        success = await _send_email(email, subject, body)
        
        if success:
            logger.info(f"Recruitment notification sent to {email}: {notification_type}")
        else:
            logger.error(f"Failed to send recruitment notification to {email}: {notification_type}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to send recruitment notification: {e}")
        return False


async def send_attendance_notification(email: str, notification_type: str, data: Dict[str, Any]) -> bool:
    """Send attendance-related notification."""
    try:
        templates = {
            "leave_approved": {
                "subject": "Leave Request Approved",
                "template": """
                Dear {employee_name},

                Your leave request has been approved:

                Leave Type: {leave_type}
                Start Date: {start_date}
                End Date: {end_date}
                Duration: {duration} days

                Please ensure proper handover before your leave.

                Best regards,
                HR Team
                """
            },
            "leave_rejected": {
                "subject": "Leave Request Rejected",
                "template": """
                Dear {employee_name},

                Unfortunately, your leave request has been rejected:

                Leave Type: {leave_type}
                Requested Dates: {start_date} to {end_date}
                Reason: {rejection_reason}

                Please contact your manager for more details.

                Best regards,
                HR Team
                """
            },
            "attendance_alert": {
                "subject": "Attendance Alert",
                "template": """
                Dear {employee_name},

                This is an automated alert regarding your attendance:

                Issue: {issue}
                Date: {date}
                Details: {details}

                Please contact HR if you have any questions.

                Best regards,
                HR Team
                """
            }
        }
        
        template_data = templates.get(notification_type)
        if not template_data:
            logger.warning(f"Unknown attendance notification type: {notification_type}")
            return False
        
        # Format template
        subject = template_data["subject"]
        body = template_data["template"].format(**data)
        
        # Send email
        success = await _send_email(email, subject, body)
        
        if success:
            logger.info(f"Attendance notification sent to {email}: {notification_type}")
        else:
            logger.error(f"Failed to send attendance notification to {email}: {notification_type}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to send attendance notification: {e}")
        return False


async def _send_email(to_email: str, subject: str, body: str) -> bool:
    """Send email using configured email service."""
    try:
        # In production, use proper email service like SendGrid, AWS SES, etc.
        # For demo, we'll just log the email
        
        email_service = os.getenv("EMAIL_SERVICE", "log")
        
        if email_service == "log":
            logger.info(f"EMAIL TO: {to_email}")
            logger.info(f"SUBJECT: {subject}")
            logger.info(f"BODY: {body}")
            return True
        
        elif email_service == "smtp":
            return await _send_smtp_email(to_email, subject, body)
        
        elif email_service == "sendgrid":
            return await _send_sendgrid_email(to_email, subject, body)
        
        else:
            logger.warning(f"Unknown email service: {email_service}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


async def _send_smtp_email(to_email: str, subject: str, body: str) -> bool:
    """Send email using SMTP."""
    try:
        smtp_server = os.getenv("SMTP_SERVER", "localhost")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        from_email = os.getenv("FROM_EMAIL", "noreply@company.com")
        
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        
        if smtp_username and smtp_password:
            server.login(smtp_username, smtp_password)
        
        server.send_message(msg)
        server.quit()
        
        return True
        
    except Exception as e:
        logger.error(f"SMTP email failed: {e}")
        return False


async def _send_sendgrid_email(to_email: str, subject: str, body: str) -> bool:
    """Send email using SendGrid API."""
    try:
        api_key = os.getenv("SENDGRID_API_KEY")
        if not api_key:
            logger.error("SendGrid API key not configured")
            return False
        
        from_email = os.getenv("FROM_EMAIL", "noreply@company.com")
        
        payload = {
            "personalizations": [
                {
                    "to": [{"email": to_email}],
                    "subject": subject
                }
            ],
            "from": {"email": from_email},
            "content": [
                {
                    "type": "text/plain",
                    "value": body
                }
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                json=payload,
                headers=headers
            )
            
            return response.status_code == 202
            
    except Exception as e:
        logger.error(f"SendGrid email failed: {e}")
        return False


async def send_system_alert(alert_type: str, message: str, data: Dict[str, Any] = None) -> bool:
    """Send system alert to administrators."""
    try:
        admin_emails = os.getenv("ADMIN_EMAILS", "admin@company.com").split(",")
        
        subject = f"HR System Alert: {alert_type}"
        body = f"""
        HR System Alert

        Alert Type: {alert_type}
        Message: {message}
        Timestamp: {data.get('timestamp', 'N/A') if data else 'N/A'}
        
        Additional Data:
        {data if data else 'None'}
        
        Please investigate immediately.
        
        HR System
        """
        
        success_count = 0
        for email in admin_emails:
            if await _send_email(email.strip(), subject, body):
                success_count += 1
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Failed to send system alert: {e}")
        return False