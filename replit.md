# AI Enterprise Dashboard

## Overview
A Next.js 14 admin dashboard for an AI Enterprise System, featuring real-time insights across Sales, Finance, HR, Marketing, Support, Legal, IT, and Admin departments.

## Project Structure
```
frontend/          - Next.js 14 app (main web UI)
  app/             - App router pages
    dashboard/     - Main dashboard and sub-pages
    page.tsx       - Root redirect to /dashboard
    layout.tsx     - Root layout with Sidebar + Header
  components/      - Reusable React components
admin-agent/       - Backend agent modules
ai-decision-engine/
cognitive-core/
finance-agent/
hr-agent/
it-agent/
legal-agent/
marketing-agent/
orchestrator/
product-agent/
qa-agent/
sales-agent/
shared-libs/
support-agent/
database/
scripts/
```

## Running the App
The frontend Next.js app runs on port 5000 via the "Start application" workflow:
```
cd frontend && npm run dev
```

## Environment Variables
Copy `.env.example` to `.env` and fill in values. Key variables:
- `NEXT_PUBLIC_API_URL` - Backend API URL (defaults to http://localhost:8000)
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` - AI provider keys
- `MONGODB_URL`, `POSTGRESQL_URL`, `REDIS_URL` - Database connections
- `JWT_SECRET` - Auth token secret

## Replit Configuration
- Port: 5000 (bound to 0.0.0.0 for Replit proxy compatibility)
- Package manager: npm
- Node version: 20
- `allowedDevOrigins: ['*']` set in next.config.js for Replit iframe proxy

## Tech Stack
- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **State**: Zustand
- **Charts**: Recharts
- **Forms**: React Hook Form
- **Icons**: Lucide React
- **HTTP**: Axios
