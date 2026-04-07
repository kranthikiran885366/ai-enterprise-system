"""AI Enterprise System - Combined Backend API."""

import os
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.context import CryptContext
from loguru import logger

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "enterprise-secret-key-change-in-prod-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
DB_PATH = os.path.join(os.path.dirname(__file__), "enterprise.db")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)


# ──────────────────────────────────────────
# Database
# ──────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            roles TEXT DEFAULT '["user"]',
            department TEXT DEFAULT 'IT',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            action TEXT NOT NULL,
            user_id TEXT,
            resource TEXT,
            ip_address TEXT,
            status TEXT DEFAULT 'success',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)

    # Seed default admin
    existing = c.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()
    if not existing:
        c.execute("""
            INSERT INTO users (id, username, email, hashed_password, is_admin, roles, department, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            "admin",
            "admin@enterprise.ai",
            pwd_context.hash("admin123"),
            1,
            '["admin", "user"]',
            "IT",
            datetime.utcnow().isoformat()
        ))

    conn.commit()
    conn.close()
    logger.info("Database initialized")


# ──────────────────────────────────────────
# Auth Utilities
# ──────────────────────────────────────────
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: sqlite3.Connection = Depends(get_db)
) -> Dict[str, Any]:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ──────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    is_admin: bool = False


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    is_admin: bool = False
    department: str = "IT"


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


# ──────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("AI Enterprise Backend started on port 8000")
    yield
    logger.info("AI Enterprise Backend shutting down")


# ──────────────────────────────────────────
# App
# ──────────────────────────────────────────
app = FastAPI(
    title="AI Enterprise System API",
    version="1.0.0",
    description="Enterprise AI Platform Backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────
# Root & Health
# ──────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "AI Enterprise System", "version": "1.0.0", "status": "running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/system/status")
def system_status(current_user: dict = Depends(get_current_user)):
    return {
        "orchestrator": "healthy",
        "total_services": 9,
        "active_services": 9,
        "services": {
            "finance-agent": {"status": "healthy"},
            "hr-agent": {"status": "healthy"},
            "sales-agent": {"status": "healthy"},
            "marketing-agent": {"status": "healthy"},
            "support-agent": {"status": "healthy"},
            "legal-agent": {"status": "healthy"},
            "it-agent": {"status": "healthy"},
            "admin-agent": {"status": "healthy"},
            "ai-engine": {"status": "healthy"},
        }
    }


# ──────────────────────────────────────────
# Auth Routes
# ──────────────────────────────────────────
@app.post("/auth/login")
def login(body: LoginRequest, db: sqlite3.Connection = Depends(get_db)):
    user = db.execute("SELECT * FROM users WHERE username = ?", (body.username,)).fetchone()
    if not user or not pwd_context.verify(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account disabled")

    token = create_access_token({"sub": user["username"], "user_id": user["id"]})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/register")
def register(body: RegisterRequest, db: sqlite3.Connection = Depends(get_db)):
    existing = db.execute("SELECT id FROM users WHERE username = ? OR email = ?", (body.username, body.email)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already taken")

    user_id = str(uuid.uuid4())
    roles = '["admin", "user"]' if body.is_admin else '["user"]'
    db.execute("""
        INSERT INTO users (id, username, email, hashed_password, is_admin, roles, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, body.username, body.email, pwd_context.hash(body.password), int(body.is_admin), roles, datetime.utcnow().isoformat()))
    db.commit()
    return {"message": "User created successfully"}


@app.get("/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "is_admin": bool(current_user["is_admin"]),
        "roles": json.loads(current_user.get("roles", '["user"]')),
        "department": current_user.get("department", "IT"),
    }


# ──────────────────────────────────────────
# Dashboard Metrics
# ──────────────────────────────────────────
@app.get("/api/v1/dashboard/metrics")
def dashboard_metrics(current_user: dict = Depends(get_current_user)):
    return {
        "active_leads": 1247,
        "total_revenue": 2500000,
        "team_members": 156,
        "fraud_alerts": 8,
        "system_uptime": 99.8,
        "open_tickets": 43,
        "ai_decisions_today": 342,
        "campaigns_active": 5,
    }


# ──────────────────────────────────────────
# Sales Routes
# ──────────────────────────────────────────
@app.get("/api/v1/sales/leads")
def get_leads(current_user: dict = Depends(get_current_user)):
    return {
        "items": [
            {"id": "L001", "name": "TechCorp Inc.", "email": "contact@techcorp.com", "score": 92, "value": 480000, "stage": "Negotiation", "status": "hot"},
            {"id": "L002", "name": "DataFlow Ltd.", "email": "sales@dataflow.io", "score": 78, "value": 220000, "stage": "Proposal", "status": "warm"},
            {"id": "L003", "name": "GlobalOps Co.", "email": "info@globalops.com", "score": 65, "value": 150000, "stage": "Qualification", "status": "warm"},
            {"id": "L004", "name": "StartupXYZ", "email": "hello@startupxyz.com", "score": 41, "value": 45000, "stage": "Prospecting", "status": "cold"},
            {"id": "L005", "name": "MegaCorp", "email": "enterprise@megacorp.com", "score": 88, "value": 750000, "stage": "Proposal", "status": "hot"},
        ],
        "total": 1247,
        "page": 1,
    }


@app.get("/api/v1/sales/analytics")
def sales_analytics(current_user: dict = Depends(get_current_user)):
    return {
        "total_pipeline": 8200000,
        "won_this_month": 1200000,
        "lost_this_month": 340000,
        "conversion_rate": 23.4,
        "avg_deal_size": 185000,
        "sales_cycle_days": 47,
    }


@app.get("/api/v1/sales/pipeline")
def sales_pipeline(current_user: dict = Depends(get_current_user)):
    return {
        "stages": [
            {"name": "Prospecting", "count": 245, "value": 1200000},
            {"name": "Qualification", "count": 156, "value": 3100000},
            {"name": "Proposal", "count": 89, "value": 2400000},
            {"name": "Negotiation", "count": 42, "value": 1500000},
        ]
    }


# ──────────────────────────────────────────
# Finance Routes
# ──────────────────────────────────────────
@app.get("/api/v1/finance/expenses")
def get_expenses(current_user: dict = Depends(get_current_user)):
    return {
        "items": [
            {"id": "EXP001", "vendor": "TechSupplies Inc", "amount": 8500, "category": "Technology", "status": "flagged", "date": "2024-03-15", "risk": "high"},
            {"id": "EXP002", "vendor": "AWS Cloud", "amount": 12400, "category": "Technology", "status": "approved", "date": "2024-03-14", "risk": "low"},
            {"id": "EXP003", "vendor": "United Airlines", "amount": 3200, "category": "Travel", "status": "pending", "date": "2024-03-13", "risk": "medium"},
            {"id": "EXP004", "vendor": "Office Depot", "amount": 890, "category": "Operations", "status": "approved", "date": "2024-03-12", "risk": "low"},
        ],
        "total": 487000,
        "flagged": 8,
    }


@app.get("/api/v1/finance/analytics")
def finance_analytics(current_user: dict = Depends(get_current_user)):
    return {
        "total_expenses_month": 487000,
        "budget_utilization": 73,
        "fraud_alerts": 8,
        "cash_flow_projected": 2100000,
        "monthly_trend": [320000, 355000, 398000, 420000, 451000, 487000],
    }


@app.get("/api/v1/finance/budget")
def get_budget(current_user: dict = Depends(get_current_user)):
    return {
        "categories": [
            {"name": "Payroll", "allocated": 250000, "spent": 245000},
            {"name": "Operations", "allocated": 80000, "spent": 72500},
            {"name": "Marketing", "allocated": 50000, "spent": 58000},
            {"name": "Technology", "allocated": 40000, "spent": 35200},
            {"name": "Travel", "allocated": 30000, "spent": 34800},
        ],
        "total_allocated": 450000,
        "total_spent": 445500,
    }


# ──────────────────────────────────────────
# HR Routes
# ──────────────────────────────────────────
@app.get("/api/v1/hr/employees")
def get_employees(current_user: dict = Depends(get_current_user)):
    return {
        "items": [
            {"id": "EMP001", "name": "Alice Johnson", "department": "Engineering", "role": "Senior Engineer", "status": "active", "hire_date": "2021-03-15"},
            {"id": "EMP002", "name": "Bob Smith", "department": "Sales", "role": "Account Executive", "status": "active", "hire_date": "2022-07-01"},
            {"id": "EMP003", "name": "Carol White", "department": "HR", "role": "HR Manager", "status": "active", "hire_date": "2020-11-20"},
            {"id": "EMP004", "name": "David Lee", "department": "Finance", "role": "Financial Analyst", "status": "active", "hire_date": "2023-01-10"},
        ],
        "total": 156,
        "by_department": {"Engineering": 42, "Sales": 31, "Finance": 18, "HR": 12, "Marketing": 15, "Support": 22, "Legal": 8, "IT": 8},
    }


@app.get("/api/v1/hr/recruitment")
def get_recruitment(current_user: dict = Depends(get_current_user)):
    return {
        "open_positions": 12,
        "candidates_pipeline": 87,
        "interviews_this_week": 14,
        "offers_extended": 3,
        "positions": [
            {"id": "JOB001", "title": "Senior Engineer", "department": "Engineering", "candidates": 23, "status": "active"},
            {"id": "JOB002", "title": "Product Manager", "department": "Product", "candidates": 15, "status": "active"},
            {"id": "JOB003", "title": "Data Scientist", "department": "Engineering", "candidates": 31, "status": "active"},
        ]
    }


@app.get("/api/v1/hr/analytics")
def hr_analytics(current_user: dict = Depends(get_current_user)):
    return {
        "headcount": 156,
        "attrition_rate": 8.2,
        "avg_tenure_years": 3.4,
        "engagement_score": 4.1,
        "attrition_risk_count": 8,
    }


# ──────────────────────────────────────────
# Marketing Routes
# ──────────────────────────────────────────
@app.get("/api/v1/marketing/campaigns")
def get_campaigns(current_user: dict = Depends(get_current_user)):
    return {
        "items": [
            {"id": "CAM001", "name": "Q1 Product Launch", "status": "active", "channel": "Multi-channel", "budget": 45000, "spent": 32400, "leads": 847, "conversion": 3.2},
            {"id": "CAM002", "name": "Email Re-engagement", "status": "active", "channel": "Email", "budget": 8000, "spent": 5200, "leads": 312, "conversion": 4.8},
            {"id": "CAM003", "name": "Google Ads - SaaS", "status": "active", "channel": "PPC", "budget": 30000, "spent": 27100, "leads": 634, "conversion": 5.6},
        ],
        "total_leads": 3691,
        "total_budget": 117000,
    }


@app.get("/api/v1/marketing/analytics")
def marketing_analytics(current_user: dict = Depends(get_current_user)):
    return {
        "total_leads": 3691,
        "conversion_rate": 3.8,
        "revenue_attributed": 1200000,
        "email_open_rate": 28.4,
        "ctr": 4.7,
    }


@app.get("/api/v1/marketing/lead-sources")
def lead_sources(current_user: dict = Depends(get_current_user)):
    return {
        "sources": [
            {"name": "Organic Search", "leads": 1243, "percentage": 34},
            {"name": "Paid Ads", "leads": 892, "percentage": 24},
            {"name": "Email", "leads": 634, "percentage": 17},
            {"name": "Social Media", "leads": 521, "percentage": 14},
            {"name": "Referral", "leads": 289, "percentage": 8},
            {"name": "Direct", "leads": 112, "percentage": 3},
        ]
    }


# ──────────────────────────────────────────
# Support Routes
# ──────────────────────────────────────────
@app.get("/api/v1/support/tickets")
def get_tickets(current_user: dict = Depends(get_current_user)):
    return {
        "items": [
            {"id": "TKT-4821", "subject": "API integration failing on production", "priority": "critical", "status": "open", "customer": "TechCorp Inc.", "created": "2h ago"},
            {"id": "TKT-4820", "subject": "Cannot export reports to PDF", "priority": "high", "status": "in_progress", "customer": "DataFlow Ltd.", "created": "4h ago"},
            {"id": "TKT-4819", "subject": "Billing discrepancy for March invoice", "priority": "medium", "status": "in_progress", "customer": "GlobalOps Co.", "created": "6h ago"},
        ],
        "total": 43,
        "open": 28,
        "in_progress": 11,
        "resolved": 4,
    }


@app.get("/api/v1/support/analytics")
def support_analytics(current_user: dict = Depends(get_current_user)):
    return {
        "open_tickets": 43,
        "avg_resolution_hours": 4.2,
        "csat_score": 4.7,
        "first_response_minutes": 12,
        "ai_resolved_percentage": 31,
    }


# ──────────────────────────────────────────
# Legal Routes
# ──────────────────────────────────────────
@app.get("/api/v1/legal/contracts")
def get_contracts(current_user: dict = Depends(get_current_user)):
    return {
        "items": [
            {"id": "CTR-2841", "name": "SaaS Master Agreement - TechCorp", "type": "MSA", "status": "active", "value": 480000, "expiry": "2025-12-31", "risk": "low"},
            {"id": "CTR-2840", "name": "Data Processing Agreement - EU GDPR", "type": "DPA", "status": "review", "value": None, "expiry": "2024-06-30", "risk": "high"},
            {"id": "CTR-2839", "name": "Enterprise License - GlobalOps", "type": "License", "status": "active", "value": 220000, "expiry": "2025-03-31", "risk": "low"},
        ],
        "total": 127,
        "active": 112,
        "expiring_soon": 8,
    }


@app.get("/api/v1/legal/compliance")
def get_compliance(current_user: dict = Depends(get_current_user)):
    return {
        "frameworks": [
            {"name": "SOC 2 Type II", "status": "compliant", "score": 98, "next_audit": "Sep 2024"},
            {"name": "GDPR", "status": "action_required", "score": 84, "next_audit": "Ongoing"},
            {"name": "ISO 27001", "status": "compliant", "score": 96, "next_audit": "Jan 2025"},
            {"name": "CCPA", "status": "compliant", "score": 92, "next_audit": "Jun 2024"},
        ],
        "avg_score": 93,
    }


@app.get("/api/v1/legal/risks")
def get_risks(current_user: dict = Depends(get_current_user)):
    return {
        "risks": [
            {"type": "Regulatory", "severity": "high", "description": "GDPR sub-processor clauses missing in 3 EU DPAs"},
            {"type": "Contractual", "severity": "medium", "description": "Vendor contract expiring in 23 days, no auto-renewal"},
        ],
        "total": 5,
    }


# ──────────────────────────────────────────
# IT Routes
# ──────────────────────────────────────────
@app.get("/api/v1/it/infrastructure")
def get_infrastructure(current_user: dict = Depends(get_current_user)):
    return {
        "servers": [
            {"name": "prod-api-01", "type": "API Server", "cpu": 68, "memory": 72, "disk": 45, "status": "healthy", "region": "us-east-1"},
            {"name": "prod-db-01", "type": "Database", "cpu": 42, "memory": 85, "disk": 67, "status": "warning", "region": "us-east-1"},
            {"name": "prod-ai-01", "type": "AI Engine", "cpu": 92, "memory": 88, "disk": 51, "status": "critical", "region": "us-east-1"},
        ],
        "total_servers": 48,
        "healthy": 44,
        "warning": 3,
        "critical": 1,
    }


@app.get("/api/v1/it/incidents")
def get_incidents(current_user: dict = Depends(get_current_user)):
    return {
        "items": [
            {"id": "INC-0421", "title": "AI Engine CPU Spike - prod-ai-01", "severity": "critical", "status": "investigating", "started": "18m ago"},
            {"id": "INC-0420", "title": "Database memory usage above 85%", "severity": "warning", "status": "monitoring", "started": "2h ago"},
        ],
        "total": 2,
        "active": 2,
    }


@app.get("/api/v1/it/metrics")
def it_metrics(current_user: dict = Depends(get_current_user)):
    return {
        "uptime_percentage": 99.8,
        "avg_cpu_usage": 52,
        "storage_used_percentage": 67,
        "deployments_month": 24,
        "active_incidents": 2,
    }


@app.get("/api/v1/it/alerts")
def get_alerts(current_user: dict = Depends(get_current_user)):
    return {
        "alerts": [
            {"severity": "critical", "message": "prod-ai-01 CPU > 90%", "time": "18m ago"},
            {"severity": "warning", "message": "prod-db-01 memory > 85%", "time": "2h ago"},
            {"severity": "warning", "message": "SSL cert expiring in 14 days", "time": "1d ago"},
        ]
    }


# ──────────────────────────────────────────
# Admin Routes
# ──────────────────────────────────────────
@app.get("/api/v1/admin/users")
def get_users(current_user: dict = Depends(require_admin), db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute("SELECT id, username, email, is_admin, is_active, roles, department, created_at FROM users").fetchall()
    return {
        "items": [dict(r) for r in rows],
        "total": len(rows),
    }


@app.post("/api/v1/admin/users")
def create_user(body: UserCreate, current_user: dict = Depends(require_admin), db: sqlite3.Connection = Depends(get_db)):
    existing = db.execute("SELECT id FROM users WHERE username = ? OR email = ?", (body.username, body.email)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already taken")

    user_id = str(uuid.uuid4())
    roles = '["admin", "user"]' if body.is_admin else '["user"]'
    db.execute("""
        INSERT INTO users (id, username, email, hashed_password, is_admin, roles, department, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, body.username, body.email, pwd_context.hash(body.password), int(body.is_admin), roles, body.department, datetime.utcnow().isoformat()))
    db.commit()
    return {"message": "User created", "id": user_id}


@app.put("/api/v1/admin/users/{user_id}")
def update_user(user_id: str, body: dict, current_user: dict = Depends(require_admin), db: sqlite3.Connection = Depends(get_db)):
    db.execute("UPDATE users SET is_active = ? WHERE id = ?", (body.get("is_active", 1), user_id))
    db.commit()
    return {"message": "User updated"}


@app.delete("/api/v1/admin/users/{user_id}")
def delete_user(user_id: str, current_user: dict = Depends(require_admin), db: sqlite3.Connection = Depends(get_db)):
    db.execute("DELETE FROM users WHERE id = ? AND username != 'admin'", (user_id,))
    db.commit()
    return {"message": "User deleted"}


@app.get("/api/v1/admin/audit-logs")
def get_audit_logs(current_user: dict = Depends(require_admin), db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute("SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT 100").fetchall()
    return {"items": [dict(r) for r in rows]}


@app.get("/api/v1/admin/config")
def get_config(current_user: dict = Depends(require_admin)):
    return {
        "jwt_expiration_minutes": ACCESS_TOKEN_EXPIRE_MINUTES,
        "rate_limit_per_minute": 100,
        "mfa_enabled": True,
        "password_min_length": 12,
        "session_timeout_hours": 8,
    }


# ──────────────────────────────────────────
# AI Routes
# ──────────────────────────────────────────
AI_RESPONSES = {
    "sales": "📊 **Sales Analysis**: Pipeline value is $8.2M across 1,247 active leads. Top opportunity: TechCorp Inc. at $480K (82% close probability). 14 deals at risk — recommend immediate follow-up.",
    "finance": "💰 **Finance Audit**: 8 expenses flagged for review totaling $42,400. Highest risk: $8,500 at TechSupplies Inc. Marketing budget is 16% over allocation.",
    "hr": "👥 **HR Report**: 156 employees, 12 open positions. 8 employees flagged as attrition risk by predictive model. Top candidate for Senior Engineer role scored 94/100.",
    "support": "🎧 **Support Status**: 43 open tickets, 2 critical SLA breaches. CSAT: 4.7/5. AI auto-resolved 31% of volume. Escalation recommended for TechCorp Inc.",
    "legal": "⚖️ **Legal Analysis**: GDPR DPA missing sub-processor clauses — action required by Jun 2024. Vendor contract expiring in 23 days. Portfolio risk: LOW overall.",
    "it": "🖥️ **IT Status**: 2 active incidents. prod-ai-01 at 92% CPU — recommend scale-up. 99.8% overall uptime. 24 successful deployments this month.",
    "marketing": "📣 **Marketing**: 3,691 leads generated this quarter, 3.8% conversion rate. Email campaigns performing well (28.4% open rate). Q1 Product Launch: 847 leads at $38/lead.",
    "default": "I've processed your request through the multi-agent system (Planner → Executor → Analyzer). I have access to real-time data across all enterprise departments. Could you be more specific about which area you'd like insights on?",
}


@app.post("/api/v1/ai/chat")
def ai_chat(body: ChatRequest, current_user: dict = Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    msg_lower = body.message.lower()
    response = AI_RESPONSES["default"]
    for key in ["sales", "finance", "hr", "support", "legal", "it", "marketing"]:
        if key in msg_lower or any(w in msg_lower for w in {
            "sales": ["lead", "pipeline", "deal", "revenue"],
            "finance": ["expense", "budget", "fraud", "invoice", "money"],
            "hr": ["employee", "hiring", "recruit", "attrition", "team"],
            "support": ["ticket", "customer", "csat", "sla"],
            "legal": ["contract", "compliance", "gdpr", "risk", "legal"],
            "it": ["server", "incident", "uptime", "infrastructure", "cpu"],
            "marketing": ["campaign", "email", "lead source", "conversion", "ads"],
        }.get(key, [])):
            response = AI_RESPONSES[key]
            break

    conv_id = body.conversation_id or str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    db.execute("INSERT OR IGNORE INTO conversations (id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
               (conv_id, current_user["id"], body.message[:50], now))
    db.execute("INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
               (str(uuid.uuid4()), conv_id, "user", body.message, now))
    db.execute("INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
               (str(uuid.uuid4()), conv_id, "assistant", response, now))
    db.commit()

    return {"response": response, "conversation_id": conv_id}


@app.get("/api/v1/ai/conversations")
def get_conversations(current_user: dict = Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute("SELECT * FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT 20", (current_user["id"],)).fetchall()
    return {"items": [dict(r) for r in rows]}


@app.get("/api/v1/ai/workflows")
def get_ai_workflows(current_user: dict = Depends(get_current_user)):
    return {
        "workflows": [
            {"id": "lead-scoring", "name": "Lead Scoring", "description": "AI-powered lead analysis", "status": "ready", "last_run": "2h ago"},
            {"id": "expense-audit", "name": "Expense Audit", "description": "Automated fraud detection", "status": "ready", "last_run": "6h ago"},
            {"id": "contract-review", "name": "Contract Review", "description": "Legal document analysis", "status": "ready", "last_run": "1d ago"},
            {"id": "candidate-match", "name": "Candidate Matching", "description": "AI resume screening", "status": "ready", "last_run": "3h ago"},
        ]
    }


@app.post("/api/v1/ai/workflows/{workflow_id}/run")
def run_workflow(workflow_id: str, current_user: dict = Depends(get_current_user)):
    return {
        "workflow_id": workflow_id,
        "run_id": str(uuid.uuid4()),
        "status": "completed",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": (datetime.utcnow() + timedelta(seconds=2)).isoformat(),
        "result": f"Workflow '{workflow_id}' completed successfully. Results stored.",
    }


@app.get("/api/v1/ai/agents/status")
def agent_status(current_user: dict = Depends(get_current_user)):
    return {
        "agents": [
            {"name": "PlannerAgent", "status": "active", "tasks_today": 342},
            {"name": "ExecutorAgent", "status": "active", "tasks_today": 298},
            {"name": "AnalyzerAgent", "status": "active", "tasks_today": 276},
            {"name": "MemoryAgent", "status": "active", "embeddings_stored": 14820},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
