# AI Enterprise System - API Documentation

## Overview

The AI Enterprise System is a microservices-based architecture where each department functions as an independent agent with its own API endpoints.

## Authentication

All API endpoints require JWT authentication. Get a token from the orchestrator:

\`\`\`bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
\`\`\`

Use the returned token in subsequent requests:

\`\`\`bash
curl -H "Authorization: Bearer <token>" "http://localhost:8000/api/hr/employees"
\`\`\`

## Service Endpoints

### Central Orchestrator (Port 8000)

#### Authentication
- `POST /auth/login` - Login and get JWT token
- `POST /auth/register` - Register new user
- `GET /auth/me` - Get current user info

#### Service Registry
- `GET /services/` - List all registered services
- `POST /services/register` - Register a new service
- `DELETE /services/{service_name}` - Unregister a service

#### System Status
- `GET /health` - Orchestrator health check
- `GET /system/status` - System-wide status

#### API Proxy
- `GET|POST|PUT|DELETE /api/{service}/{path}` - Proxy requests to microservices

### HR Agent (Port 8001)

#### Employees
- `POST /api/hr/employees/` - Create employee
- `GET /api/hr/employees/` - List employees (with filters)
- `GET /api/hr/employees/{employee_id}` - Get employee details
- `PUT /api/hr/employees/{employee_id}` - Update employee
- `DELETE /api/hr/employees/{employee_id}` - Delete employee
- `GET /api/hr/employees/search/{term}` - Search employees

#### Recruitment
- `POST /api/hr/recruitment/jobs` - Create job posting
- `GET /api/hr/recruitment/jobs` - List job postings
- `POST /api/hr/recruitment/applications` - Submit application
- `GET /api/hr/recruitment/applications` - List applications

#### Attendance
- `POST /api/hr/attendance/records` - Create attendance record
- `GET /api/hr/attendance/records` - List attendance records
- `POST /api/hr/attendance/leave-requests` - Submit leave request
- `GET /api/hr/attendance/leave-requests` - List leave requests

### Finance Agent (Port 8002)

#### Expenses
- `POST /api/finance/expenses/` - Create expense
- `GET /api/finance/expenses/` - List expenses
- `GET /api/finance/expenses/{expense_id}` - Get expense details
- `PUT /api/finance/expenses/{expense_id}/approve` - Approve expense

#### Invoices
- `POST /api/finance/invoices/` - Create invoice
- `GET /api/finance/invoices/` - List invoices
- `GET /api/finance/invoices/{invoice_id}` - Get invoice details
- `PUT /api/finance/invoices/{invoice_id}/mark-paid` - Mark invoice as paid

#### Budget
- `POST /api/finance/budget/categories` - Create budget category
- `GET /api/finance/budget/summary` - Get budget summary

#### Payroll
- `POST /api/finance/payroll/` - Create payroll record
- `GET /api/finance/payroll/` - List payroll records
- `GET /api/finance/payroll/{payroll_id}` - Get payroll details

### AI Decision Engine (Port 8009)

#### Rules
- `POST /api/ai/rules/` - Create business rule
- `GET /api/ai/rules/` - List rules
- `GET /api/ai/rules/{rule_id}` - Get rule details
- `GET /api/ai/rules/alerts/` - List alerts
- `PUT /api/ai/rules/alerts/{alert_id}/acknowledge` - Acknowledge alert

#### Recommendations
- `POST /api/ai/recommendations/` - Create recommendation
- `GET /api/ai/recommendations/` - List recommendations
- `GET /api/ai/recommendations/{recommendation_id}` - Get recommendation details
- `PUT /api/ai/recommendations/{recommendation_id}/status` - Update recommendation status

#### Analytics
- `GET /api/ai/analytics/summary` - Get analytics summary
- `GET /api/ai/analytics/dashboard` - Get dashboard data

## Example Requests

### Create Employee
\`\`\`bash
curl -X POST "http://localhost:8001/api/hr/employees/" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@company.com",
    "department": "engineering",
    "position": "Software Engineer",
    "hire_date": "2024-01-15T00:00:00Z"
  }'
\`\`\`

### Create Expense
\`\`\`bash
curl -X POST "http://localhost:8002/api/finance/expenses/" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "employee_id": "EMP001",
    "amount": 150.00,
    "category": "travel",
    "description": "Business trip to client site",
    "expense_date": "2024-01-15"
  }'
\`\`\`

### Create Business Rule
\`\`\`bash
curl -X POST "http://localhost:8009/api/ai/rules/" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "High Expense Alert",
    "description": "Alert when expense exceeds $500",
    "rule_type": "threshold",
    "conditions": {
      "field": "expense.amount",
      "operator": "greater_than",
      "value": 500
    },
    "actions": [
      {
        "type": "alert",
        "severity": "medium",
        "message": "High expense detected: ${expense.amount}"
      }
    ],
    "priority": 7,
    "department": "finance"
  }'
\`\`\`

## Error Responses

All endpoints return consistent error responses:

\`\`\`json
{
  "error": "Error type",
  "message": "Human readable error message",
  "timestamp": "2024-01-15T10:30:00Z",
  "correlation_id": "abc123"
}
\`\`\`

## Rate Limiting

All endpoints are rate limited to prevent abuse. Default limits:
- 100 requests per minute per IP
- 1000 requests per hour per authenticated user

## Pagination

List endpoints support pagination:

\`\`\`bash
curl "http://localhost:8001/api/hr/employees/?page=1&limit=10"
\`\`\`

Response includes pagination metadata:

\`\`\`json
{
  "items": [...],
  "total": 150,
  "page": 1,
  "limit": 10,
  "pages": 15
}
