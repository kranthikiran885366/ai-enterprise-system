# AI-Powered Autonomous Enterprise Management System

## 🚀 Overview

A scalable microservices-based backend architecture for an AI-powered enterprise management system. Each department functions as an independent microservice with secure inter-service communication, centralized orchestration, and AI-driven decision making.

## 🏗️ Architecture

### Microservices (Agents)
- **HR Agent** (Port 8001) - Employee management, recruitment, attendance
- **Finance Agent** (Port 8002) - Expenses, invoices, budget, payroll
- **Sales Agent** (Port 8003) - Leads, deals, targets
- **Marketing Agent** (Port 8004) - Campaigns, metrics, ads
- **IT Agent** (Port 8005) - Assets, tickets, infrastructure
- **Admin Agent** (Port 8006) - Notices, permissions, announcements
- **Legal Agent** (Port 8007) - Compliance, documents, cases
- **Support Agent** (Port 8008) - Customer tickets, FAQs, feedback
- **Central Orchestrator** (Port 8000) - API Gateway, service registry
- **AI Decision Engine** (Port 8009) - Rule-based decisions, recommendations

### Infrastructure
- **MongoDB** - Primary database
- **Redis** - Caching and sessions
- **RabbitMQ** - Message broker for async communication

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### Installation

1. Clone the repository:
\`\`\`bash
git clone <repository-url>
cd ai-enterprise-system
\`\`\`

2. Start all services:
\`\`\`bash
docker-compose up -d
\`\`\`

3. Verify services are running:
\`\`\`bash
docker-compose ps
\`\`\`

### API Documentation
- **Orchestrator**: http://localhost:8000/docs
- **HR Agent**: http://localhost:8001/docs
- **Finance Agent**: http://localhost:8002/docs
- **Sales Agent**: http://localhost:8003/docs
- **Marketing Agent**: http://localhost:8004/docs
- **IT Agent**: http://localhost:8005/docs
- **Admin Agent**: http://localhost:8006/docs
- **Legal Agent**: http://localhost:8007/docs
- **Support Agent**: http://localhost:8008/docs
- **AI Engine**: http://localhost:8009/docs

### Management Interfaces
- **RabbitMQ Management**: http://localhost:15672 (admin/password123)
- **MongoDB**: localhost:27017

## 🔧 Development

### Local Development Setup

1. Install dependencies:
\`\`\`bash
pip install -e ".[dev]"
\`\`\`

2. Set up environment variables:
\`\`\`bash
cp .env.example .env
# Edit .env with your configuration
\`\`\`

3. Run individual services:
\`\`\`bash
# Orchestrator
cd orchestrator && uvicorn main:app --reload --port 8000

# HR Agent
cd hr-agent && uvicorn main:app --reload --port 8001
\`\`\`

### Testing
\`\`\`bash
pytest
\`\`\`

### Code Formatting
\`\`\`bash
black .
isort .
flake8 .
\`\`\`

## 📊 Monitoring

### Health Checks
All services expose health check endpoints:
- `GET /health` - Service health status
- `GET /metrics` - Prometheus metrics

### Logging
Centralized logging using Loguru with structured JSON output.

## 🔐 Security

- JWT-based authentication between services
- Rate limiting on all endpoints
- CORS protection
- Input validation using Pydantic
- Secure headers with custom middleware

## 🚀 Phase 2 Roadiness

The architecture is designed for easy integration of:
- Machine Learning models
- Real-time analytics
- Advanced AI decision making
- Kubernetes deployment
- Auto-scaling capabilities

## 📝 API Examples

### Authentication
\`\`\`bash
# Get JWT token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
\`\`\`

### HR Operations
\`\`\`bash
# Create employee
curl -X POST "http://localhost:8001/api/hr/employees" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "email": "john@company.com", "department": "Engineering"}'
\`\`\`

### Finance Operations
\`\`\`bash
# Create expense
curl -X POST "http://localhost:8002/api/finance/expenses" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"amount": 1000, "category": "Travel", "description": "Business trip"}'
\`\`\`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
\`\`\`

Now let's create the shared libraries:

```python file="shared-libs/__init__.py"
"""Shared libraries for the AI Enterprise System."""
