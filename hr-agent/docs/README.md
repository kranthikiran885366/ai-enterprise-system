# HR Agent - Human Resources Management Microservice

## Overview

The HR Agent is a comprehensive human resources management microservice that provides AI-powered employee management, recruitment, attendance tracking, and performance management capabilities.

## Features

### Core HR Management
- **Employee Management**: Complete CRUD operations for employee records
- **Recruitment**: AI-powered job posting optimization and candidate screening
- **Attendance Tracking**: Time tracking, leave management, and attendance analytics
- **Performance Management**: Performance reviews and employee development tracking

### AI-Powered Features
- **Resume Analysis**: Automated resume screening and candidate scoring
- **AI Interviews**: Automated interview conducting and evaluation
- **Employee Mood Tracking**: Sentiment analysis and performance correlation
- **Predictive Analytics**: Employee attrition prediction and retention strategies

### Security & Compliance
- **Role-based Access Control**: Department and role-based permissions
- **Audit Logging**: Complete audit trail for all HR operations
- **Data Privacy**: Sensitive data filtering and anonymization
- **Compliance**: GDPR and employment law compliance features

## API Endpoints

### Employee Management
- `POST /api/hr/employees/` - Create new employee
- `GET /api/hr/employees/` - List employees with filtering
- `GET /api/hr/employees/{employee_id}` - Get employee details
- `PUT /api/hr/employees/{employee_id}` - Update employee
- `DELETE /api/hr/employees/{employee_id}` - Terminate employee
- `GET /api/hr/employees/search/{term}` - Search employees

### Recruitment
- `POST /api/hr/recruitment/jobs` - Create job posting
- `GET /api/hr/recruitment/jobs` - List job postings
- `POST /api/hr/recruitment/applications` - Submit application
- `GET /api/hr/recruitment/applications` - List applications
- `POST /api/hr/recruitment/ai-interview` - Conduct AI interview
- `GET /api/hr/recruitment/analytics` - Get recruitment analytics

### Attendance
- `POST /api/hr/attendance/records` - Create attendance record
- `GET /api/hr/attendance/records` - List attendance records
- `POST /api/hr/attendance/leave-requests` - Submit leave request
- `GET /api/hr/attendance/leave-requests` - List leave requests

## Architecture

```
hr-agent/
├── controllers/           # Business logic controllers
│   ├── employee_controller.py
│   └── recruitment_controller.py
├── services/             # Core business services
│   ├── hr_service.py
│   └── ai_recruitment.py
├── routes/               # FastAPI route definitions
│   ├── employees.py
│   ├── recruitment.py
│   └── attendance.py
├── models/               # Data models
│   └── employee.py
├── utils/                # Utility functions
│   ├── validators.py
│   ├── notifications.py
│   └── helpers.py
├── middleware/           # Custom middleware
│   ├── auth.py
│   ├── validation.py
│   └── logging.py
├── config/               # Configuration
│   ├── settings.py
│   └── database.py
└── docs/                 # Documentation
    └── README.md
```

## Configuration

### Environment Variables

```bash
# Service Configuration
HR_SERVICE_NAME=hr-agent
HR_SERVICE_VERSION=1.0.0
HR_DEBUG=false
HR_HOST=0.0.0.0
HR_PORT=8001

# Database Configuration
HR_MONGODB_URL=mongodb://localhost:27017/enterprise
HR_DATABASE_NAME=enterprise

# Authentication
HR_JWT_SECRET_KEY=your-secret-key
HR_JWT_ALGORITHM=HS256
HR_JWT_EXPIRATION_MINUTES=30

# Email Configuration
HR_EMAIL_SERVICE=smtp
HR_SMTP_SERVER=smtp.gmail.com
HR_SMTP_PORT=587
HR_SMTP_USERNAME=your-email@gmail.com
HR_SMTP_PASSWORD=your-password
HR_FROM_EMAIL=hr@company.com

# AI Configuration
HR_OPENAI_API_KEY=your-openai-key
HR_ENABLE_AI_FEATURES=true
HR_AI_CONFIDENCE_THRESHOLD=0.7

# HR-Specific Settings
HR_DEFAULT_LEAVE_ACCRUAL_RATE=2.0
HR_MAX_LEAVE_DAYS_PER_REQUEST=30
HR_ADVANCE_NOTICE_DAYS=7
HR_PROBATION_PERIOD_MONTHS=6
HR_PERFORMANCE_REVIEW_FREQUENCY_MONTHS=12
```

## Setup & Installation

### Local Development

1. **Install Dependencies**
   ```bash
   cd hr-agent
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the Service**
   ```bash
   uvicorn main:app --reload --port 8001
   ```

### Docker Deployment

1. **Build Image**
   ```bash
   docker build -t hr-agent .
   ```

2. **Run Container**
   ```bash
   docker run -p 8001:8000 -e MONGODB_URL=mongodb://host:27017/enterprise hr-agent
   ```

## Usage Examples

### Create Employee

```bash
curl -X POST "http://localhost:8001/api/hr/employees/" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@company.com",
    "department": "engineering",
    "position": "Software Engineer",
    "hire_date": "2024-01-15T00:00:00Z",
    "salary": 75000,
    "skills": ["Python", "JavaScript", "React"],
    "manager_id": "EMP001"
  }'
```

### Submit Job Application

```bash
curl -X POST "http://localhost:8001/api/hr/recruitment/applications" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "JOB001",
    "candidate_name": "Jane Smith",
    "candidate_email": "jane.smith@email.com",
    "candidate_phone": "+1-555-0123",
    "resume_text": "Experienced software engineer with 5 years...",
    "cover_letter": "I am excited to apply for this position...",
    "job_description": "We are looking for a senior software engineer..."
  }'
```

### Conduct AI Interview

```bash
curl -X POST "http://localhost:8001/api/hr/recruitment/ai-interview" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "CAND001",
    "interview_type": "technical"
  }'
```

## AI Features

### Resume Analysis
- Automatic keyword extraction and matching
- Skill assessment and gap analysis
- Experience level evaluation
- Cultural fit indicators
- Recommendation scoring (0-1 scale)

### AI Interviews
- Automated question generation based on role
- Real-time response analysis
- Technical skill assessment
- Behavioral evaluation
- Comprehensive scoring and recommendations

### Employee Analytics
- Mood and sentiment tracking
- Performance correlation analysis
- Attrition risk prediction
- Retention strategy recommendations
- Team dynamics insights

## Security Features

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- Department-based permissions
- Manager-subordinate access patterns

### Data Protection
- Sensitive data filtering in logs
- PII anonymization for analytics
- Secure password generation
- Audit trail for all operations

### Compliance
- GDPR compliance features
- Employment law adherence
- Data retention policies
- Privacy controls

## Monitoring & Logging

### Structured Logging
- Correlation ID tracking
- User action logging
- Performance metrics
- Error tracking
- Audit trails

### Health Checks
- Database connectivity
- Service dependencies
- AI service availability
- Email service status

## Integration

### Inter-Service Communication
- Finance Agent integration for payroll
- IT Agent integration for account provisioning
- Central orchestrator communication
- Message queue integration

### External Integrations
- Email service providers (SMTP, SendGrid)
- AI services (OpenAI)
- File storage services
- Calendar systems

## Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### API Tests
```bash
pytest tests/api/
```

## Performance

### Optimization Features
- Database indexing strategy
- Caching for frequent queries
- Async processing for heavy operations
- Batch processing for notifications

### Scalability
- Horizontal scaling support
- Database connection pooling
- Load balancing ready
- Microservice architecture

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check MongoDB URL and credentials
   - Verify network connectivity
   - Check database permissions

2. **Authentication Errors**
   - Verify JWT secret key configuration
   - Check token expiration settings
   - Validate user permissions

3. **AI Features Not Working**
   - Verify OpenAI API key
   - Check AI service configuration
   - Monitor API rate limits

4. **Email Notifications Failed**
   - Check email service configuration
   - Verify SMTP credentials
   - Check firewall settings

### Logs Location
- Application logs: `/var/log/hr-agent/`
- Error logs: `/var/log/hr-agent/error.log`
- Audit logs: Database collection `audit_logs`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.