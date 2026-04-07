# AI Enterprise Dashboard

## Overview
A full-stack enterprise AI platform with a Next.js 14 frontend and FastAPI backend. Features 11 department dashboards, an AI multi-agent chat system, authentication, and a complete REST API.

## Architecture

```
frontend/          - Next.js 14 frontend (port 5000)
  app/
    dashboard/     - Department pages (sales, finance, hr, marketing, support, legal, it, admin, ai, settings)
    login/         - Authentication page
    page.tsx       - Root → redirects to /dashboard
    layout.tsx     - Root layout with Sidebar + Header
  components/
    dashboard/     - MetricCard, charts
    layout/        - Sidebar, Header (with auth + notifications)
  lib/
    api.ts         - Axios API client (relative URLs via Next.js rewrites proxy)
    store.ts       - Zustand auth state with persist

backend/           - FastAPI combined API server (port 8000)
  main.py          - All routes: auth, dashboard, sales, finance, hr, marketing, support, legal, it, admin, AI chat
  enterprise.db    - SQLite database (auto-created)

shared-libs/       - Python shared utilities (auth, DB, middleware, AI providers)
orchestrator/      - Full FastAPI orchestrator/API gateway (MongoDB-based, for Docker)
finance-agent/     - Finance microservice (Docker)
hr-agent/          - HR microservice (Docker)
sales-agent/       - Sales microservice (Docker)
marketing-agent/   - Marketing microservice (Docker)
support-agent/     - Support microservice (Docker)
legal-agent/       - Legal microservice (Docker)
it-agent/          - IT microservice (Docker)
admin-agent/       - Admin microservice (Docker)
ai-decision-engine/ - AI decision engine (Docker)
cognitive-core/    - Cognitive processing (Docker)
docker-compose.yml - Full Docker setup with MongoDB, PostgreSQL, Redis, RabbitMQ
k8s/               - Kubernetes deployment configs
```

## Running the App (Replit)

Two workflows run simultaneously:
1. **Start application** → `cd frontend && npm run dev` (port 5000, webview)
2. **Backend API** → `cd backend && python main.py` (port 8000, console)

Frontend proxies `/auth/*`, `/api/v1/*`, `/system/*` to the backend via Next.js rewrites.

## Default Login
- **Username:** admin
- **Password:** admin123

## Frontend Pages
| Route | Description |
|-------|-------------|
| `/login` | Authentication page |
| `/dashboard` | Main overview with metrics |
| `/dashboard/ai` | AI Chat + Workflow automation |
| `/dashboard/sales` | Sales pipeline & lead scoring |
| `/dashboard/finance` | Expense audit & budget |
| `/dashboard/hr` | Employee & recruitment management |
| `/dashboard/marketing` | Campaign analytics & lead sources |
| `/dashboard/support` | Ticket queue & CSAT |
| `/dashboard/legal` | Contract registry & compliance |
| `/dashboard/it` | Infrastructure monitoring |
| `/dashboard/admin` | User management & audit logs |
| `/dashboard/settings` | Profile, notifications, API keys |

## Backend API Endpoints
- `POST /auth/login` — JWT login
- `POST /auth/register` — User registration
- `GET /auth/me` — Current user info
- `GET /api/v1/dashboard/metrics` — Overview metrics
- `GET /api/v1/sales/*` — Leads, pipeline, analytics
- `GET /api/v1/finance/*` — Expenses, invoices, budget, analytics
- `GET /api/v1/hr/*` — Employees, recruitment, analytics
- `GET /api/v1/marketing/*` — Campaigns, analytics, lead sources
- `GET /api/v1/support/*` — Tickets, analytics
- `GET /api/v1/legal/*` — Contracts, compliance, risks
- `GET /api/v1/it/*` — Infrastructure, incidents, metrics, alerts
- `GET /api/v1/admin/*` — Users, audit logs, config (admin only)
- `POST /api/v1/ai/chat` — AI multi-agent chat
- `GET /api/v1/ai/workflows` — Available AI workflows
- `POST /api/v1/ai/workflows/{id}/run` — Run AI workflow
- `GET /api/v1/ai/agents/status` — Agent health status

## Environment Variables
```
JWT_SECRET_KEY      - JWT signing secret (defaults to built-in dev key)
BACKEND_URL         - Backend URL for Next.js proxy (defaults to http://localhost:8000)
OPENAI_API_KEY      - OpenAI API key (for production AI features)
ANTHROPIC_API_KEY   - Anthropic Claude API key
GOOGLE_API_KEY      - Google Gemini API key
```

## Tech Stack
- **Frontend:** Next.js 14, React 18, TypeScript, Tailwind CSS, Zustand, Axios, Lucide React
- **Backend:** FastAPI, Uvicorn, SQLite (Replit), JWT auth, Passlib/bcrypt
- **Docker Stack:** MongoDB, PostgreSQL, Redis, RabbitMQ + 9 microservices
- **AI:** Multi-provider abstraction (OpenAI, Anthropic, Google, DeepSeek), multi-agent system
- **DevOps:** Docker Compose, Kubernetes (k8s/), GitHub Actions ready

## Security
- JWT Bearer token authentication
- Bcrypt password hashing
- RBAC (admin vs user roles)
- Security headers on all responses
- Rate limiting via slowapi
- CORS configured
