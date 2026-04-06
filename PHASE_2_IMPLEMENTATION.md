# Phase 2: Implementing Real Backend Agents

This document outlines the comprehensive implementation strategy for upgrading all 9 microservices with production-grade logic, database operations, and AI integrations.

## Strategy Overview

Rather than rewrite every agent from scratch, we'll:

1. **Enhance existing agents** with PostgreSQL support (HR, Finance already have MongoDB)
2. **Add missing AI features** (resume parsing, forecasting, sentiment analysis)
3. **Implement missing services** (Marketing, Legal, Product/QA missing service files)
4. **Integrate with external APIs** (SendGrid, Salesforce, Stripe where applicable)
5. **Replace all mock/stub code** with real business logic

## Agent-by-Agent Implementation Plan

### 1. HR Agent - Employee & Recruitment Management
**Current State:** ✅ Database operations exist in MongoDB, needs enhancement

**PostgreSQL Migration Tasks:**
- [ ] Update `hr_service.py` to use `db_abstraction.UnifiedDatabase` for structured employee data
- [ ] Migrate employee records from MongoDB to PostgreSQL using new schema
- [ ] Add PostgreSQL support to employee repository

**New Features to Add:**
- [ ] Resume parsing with Claude Vision API (extract skills, experience)
- [ ] AI-powered candidate matching (similarity scoring against job requirements)
- [ ] Interview question generation (use Claude)
- [ ] Job description optimization with GPT-4
- [ ] SendGrid email integration for recruitment notifications
- [ ] Attendance tracking with PostgreSQL
- [ ] Salary data analysis using Pandas

**Files to Create/Modify:**
```
hr-agent/services/resume_parser.py         # NEW: AI-powered resume parsing
hr-agent/services/candidate_matcher.py     # NEW: ML-based candidate matching
hr-agent/services/interview_generator.py   # NEW: Interview question generation
hr-agent/services/hr_service.py            # MODIFY: Add PostgreSQL, new methods
hr-agent/requirements.txt                  # ADD: asyncpg, aiofiles for resume handling
hr-agent/routes/recruitment.py             # MODIFY: Add new endpoints
```

**Example Implementation Structure:**
```python
# resume_parser.py
async def parse_resume_pdf(file_path: str) -> Dict[str, Any]:
    """Extract structured data from resume using Claude Vision API."""
    # Read PDF content
    # Call Claude with vision capability
    # Extract: skills, experience, education, certifications
    # Return structured data

# candidate_matcher.py
async def match_candidates_to_role(job_description: str, candidates: List[Dict]) -> List[Tuple[str, float]]:
    """Score candidates against job requirements using embeddings."""
    # Generate embedding for job description
    # Generate embeddings for each candidate
    # Compute cosine similarity
    # Return ranked list with scores
```

---

### 2. Finance Agent - Budget & Expense Management
**Current State:** ✅ Database operations exist, needs AI features

**PostgreSQL Migration Tasks:**
- [ ] Keep existing MongoDB for transaction logs
- [ ] Move structured financial records to PostgreSQL
- [ ] Add financial analytics queries in PostgreSQL

**New Features to Add:**
- [ ] Advanced tax calculation with Claude (handles complex scenarios)
- [ ] Receipt/invoice scanning with Claude Vision API + text extraction
- [ ] Real accounting validation rules (debits = credits, etc.)
- [ ] Budget forecasting using time-series analysis (Prophet or similar)
- [ ] Integration with QuickBooks or Xero API (if available)
- [ ] Expense categorization using ML
- [ ] Financial report generation (PDF export)

**Files to Create/Modify:**
```
finance-agent/services/tax_calculator.py   # MODIFY: Use Claude for complex scenarios
finance-agent/services/receipt_scanner.py  # NEW: Vision API + OCR
finance-agent/services/budget_forecast.py  # NEW: Time-series forecasting
finance-agent/services/report_generator.py # NEW: PDF financial reports
finance-agent/services/accounting_validator.py  # NEW: Real accounting rules
finance-agent/requirements.txt              # ADD: prophet, pypdf, pillow
```

**Example Implementation:**
```python
# tax_calculator.py - Enhanced with Claude for complex scenarios
async def calculate_taxes(income_data: Dict, deductions: List[Dict]) -> Dict:
    """Calculate taxes with AI assistance for complex scenarios."""
    # For standard cases, use rule-based calculation
    # For complex scenarios, ask Claude for analysis
    # Return tax liability breakdown

# budget_forecast.py
async def forecast_budget(historical_expenses: List[float], months_ahead: int = 6) -> List[float]:
    """Forecast future expenses using time-series analysis."""
    # Use Prophet or similar library
    # Return predicted monthly expenses
```

---

### 3. Sales Agent - Lead & Deal Management
**Current State:** ✅ Database operations exist, needs AI features

**PostgreSQL Migration Tasks:**
- [ ] Move lead & deal records to PostgreSQL
- [ ] Add sales pipeline queries

**New Features to Add:**
- [ ] Real lead scoring with multi-factor analysis (company size, engagement, industry fit)
- [ ] Predictive deal close probability using Claude analysis
- [ ] Intelligent deal stage recommendations
- [ ] Email/SMS template generation for outreach
- [ ] Sales forecast AI (GPT-4 analyzes sales trends)
- [ ] CRM integration (Salesforce, HubSpot APIs)
- [ ] Activity logging from actual emails and calls

**Files to Create/Modify:**
```
sales-agent/services/lead_scorer.py        # MODIFY: Real multi-factor scoring
sales-agent/services/deal_forecast.py      # NEW: Predictive deal analysis
sales-agent/services/sales_ai.py           # MODIFY: Enhanced with GPT-4
sales-agent/services/crm_integration.py    # NEW: Salesforce/HubSpot sync
sales-agent/services/outreach_generator.py # NEW: Email template generation
sales-agent/requirements.txt                # ADD: salesforce-api, hubspot-api
```

**Example Implementation:**
```python
# lead_scorer.py
async def score_lead(lead: Dict) -> float:
    """Score lead on 0-100 scale using multiple factors."""
    factors = {
        'company_size': score_by_company_size(lead.company_size),
        'engagement': score_by_engagement_level(lead),
        'industry_fit': score_industry_match(lead.industry),
        'budget_availability': score_budget(lead.estimated_budget),
    }
    weights = {'company_size': 0.2, 'engagement': 0.3, 'industry_fit': 0.25, 'budget': 0.25}
    return sum(score * weights[factor] for factor, score in factors.items())

# deal_forecast.py - Uses Claude for analysis
async def predict_deal_close_probability(deal: Dict, historical_deals: List[Dict]) -> float:
    """Use Claude to analyze deal and predict close probability."""
    prompt = f"""Analyze this sales deal and compare to historical patterns.
    Deal: {deal}
    Historical deals: {historical_deals}
    Return a probability 0-1 of this deal closing."""
    response = await ai_providers.complete(prompt)
    # Parse probability from response
```

---

### 4. Marketing Agent - Campaign Management
**Current State:** ❌ No service file, minimal implementation

**Complete New Implementation:**
- [ ] Create `marketing_service.py` from scratch
- [ ] Campaign CRUD operations in PostgreSQL
- [ ] Email marketing integration (SendGrid)
- [ ] Social media content scheduling
- [ ] A/B testing framework
- [ ] Campaign analytics tracking

**New Files to Create:**
```
marketing-agent/services/marketing_service.py   # NEW: Campaign management
marketing-agent/services/email_manager.py       # NEW: SendGrid integration
marketing-agent/services/social_manager.py      # NEW: Social media posting
marketing-agent/services/analytics_service.py   # NEW: Campaign metrics
marketing-agent/services/content_generator.py   # NEW: AI content creation
marketing-agent/models/campaign.py              # NEW: Campaign data model
marketing-agent/routes/campaigns.py             # NEW: Campaign endpoints
```

**Example Implementation:**
```python
# marketing_service.py
class MarketingService:
    async def create_campaign(self, campaign_data: CampaignCreate) -> Campaign:
        """Create marketing campaign and store in PostgreSQL."""
        # Insert into campaigns table
        # Set status to 'draft'
        # Return created campaign
    
    async def launch_campaign(self, campaign_id: str) -> bool:
        """Launch campaign and trigger email/social sending."""
        # Get campaign from database
        # Generate personalized content using AI
        # Send emails via SendGrid
        # Schedule social posts
        # Update campaign status to 'active'

# email_manager.py
async def send_campaign_emails(campaign_id: str, recipients: List[Dict]) -> Dict:
    """Send personalized emails using SendGrid."""
    # Get campaign template
    # For each recipient:
    #   - Personalize content using AI
    #   - Send via SendGrid API
    # Track opens/clicks
```

---

### 5. Support Agent - Ticket Management
**Current State:** ✅ Database exists, needs AI features

**Enhancement Tasks:**
- [ ] Real ticket routing logic based on complexity
- [ ] AI-powered sentiment analysis of incoming messages
- [ ] Intelligent escalation (route to specialists when needed)
- [ ] Response suggestions using Claude
- [ ] Knowledge base integration
- [ ] SLA tracking and violation alerts

**Files to Create/Modify:**
```
support-agent/services/ticket_router.py    # MODIFY: Real routing logic
support-agent/services/sentiment_analyzer.py  # NEW: Real sentiment analysis
support-agent/services/response_suggester.py  # NEW: Claude-powered suggestions
support-agent/services/knowledge_base.py   # NEW: KB search and integration
support-agent/routes/tickets.py            # MODIFY: Add new endpoints
```

**Example Implementation:**
```python
# sentiment_analyzer.py
async def analyze_ticket_sentiment(message: str) -> Dict:
    """Analyze sentiment and urgency of support ticket."""
    sentiment = await intelligence.analyze_sentiment(message)
    urgency_keywords = {'urgent', 'asap', 'broken', 'down', 'critical'}
    has_urgency = any(word in message.lower() for word in urgency_keywords)
    return {
        'sentiment': sentiment,
        'urgency_level': 'high' if has_urgency else 'normal',
    }

# ticket_router.py
async def route_ticket(ticket: Dict) -> str:
    """Intelligently route ticket to appropriate agent."""
    sentiment = await analyze_ticket_sentiment(ticket['description'])
    
    # Route based on category, sentiment, and availability
    if sentiment['urgency_level'] == 'high':
        return await find_available_senior_agent()
    
    if ticket['category'] == 'billing':
        return await find_billing_specialist()
    
    return await find_available_agent()
```

---

### 6. Legal Agent - Document Management
**Current State:** ❌ No service file

**Complete New Implementation:**
- [ ] Create `legal_service.py` for document management
- [ ] Contract review AI using Claude
- [ ] Compliance checking logic
- [ ] Risk assessment framework
- [ ] Document versioning system
- [ ] Audit trail for all changes

**Files to Create:**
```
legal-agent/services/legal_service.py       # NEW: Core legal operations
legal-agent/services/contract_reviewer.py   # NEW: Contract analysis
legal-agent/services/compliance_checker.py  # NEW: Compliance rules
legal-agent/services/risk_assessor.py       # NEW: Risk scoring
legal-agent/models/document.py              # NEW: Document model
legal-agent/routes/documents.py             # NEW: Document endpoints
```

**Example Implementation:**
```python
# contract_reviewer.py
async def review_contract(contract_text: str, contract_type: str) -> Dict:
    """Review contract using Claude for red flags and risks."""
    prompt = f"""Review this {contract_type} contract and identify:
    1. Unfavorable clauses
    2. Missing standard terms
    3. Liability risks
    4. Compliance issues
    
    Contract: {contract_text}"""
    
    review = await ai_providers.complete(prompt)
    return {
        'review_summary': review,
        'risk_level': extract_risk_level(review),
        'recommended_actions': extract_recommendations(review),
    }

# compliance_checker.py
async def check_compliance(document: Dict, regulations: List[str]) -> List[str]:
    """Check document against applicable regulations."""
    violations = []
    for regulation in regulations:
        if not meets_regulation(document, regulation):
            violations.append(regulation)
    return violations
```

---

### 7. IT Agent - Infrastructure Management
**Current State:** ❌ Very minimal implementation

**Complete New Implementation:**
- [ ] Infrastructure monitoring data collection
- [ ] Security incident logging
- [ ] Performance metrics tracking
- [ ] Incident root cause analysis (Claude AI)
- [ ] Automated remediation recommendations
- [ ] Capacity planning and predictions
- [ ] Cloud provider integrations (AWS, Azure, GCP)

**Files to Create:**
```
it-agent/services/it_service.py             # NEW: Core IT operations
it-agent/services/incident_analyzer.py      # NEW: RCA with AI
it-agent/services/monitoring_service.py     # NEW: Metrics collection
it-agent/services/capacity_planner.py       # NEW: Capacity forecasting
it-agent/services/cloud_integration.py      # NEW: AWS/Azure/GCP API calls
it-agent/routes/infrastructure.py           # NEW: Infrastructure endpoints
```

**Example Implementation:**
```python
# incident_analyzer.py
async def analyze_incident(incident: Dict) -> Dict:
    """Analyze incident and suggest root causes using Claude."""
    prompt = f"""Analyze this infrastructure incident:
    Service: {incident['service']}
    Symptoms: {incident['symptoms']}
    Timeline: {incident['timeline']}
    
    Suggest likely root causes and remediation steps."""
    
    analysis = await ai_providers.complete(prompt)
    return {
        'probable_causes': extract_causes(analysis),
        'recommended_fixes': extract_fixes(analysis),
        'severity': calculate_severity(incident),
    }

# monitoring_service.py
async def collect_metrics() -> Dict:
    """Collect infrastructure metrics from cloud providers."""
    aws_metrics = await get_aws_metrics()
    azure_metrics = await get_azure_metrics()
    gcp_metrics = await get_gcp_metrics()
    
    return {
        'cpu_usage': aggregate_metric('cpu', [aws_metrics, azure_metrics, gcp_metrics]),
        'memory_usage': aggregate_metric('memory', ...),
        'disk_usage': aggregate_metric('disk', ...),
        'network_latency': aggregate_metric('latency', ...),
    }
```

---

### 8. Admin Agent - System Administration
**Current State:** ✅ Partial implementation exists

**Enhancement Tasks:**
- [ ] Real audit logging (every action tracked)
- [ ] User provisioning workflows
- [ ] Role-based access control (RBAC)
- [ ] Organization settings management
- [ ] Security policy enforcement
- [ ] User session management

**Files to Create/Modify:**
```
admin-agent/services/admin_service.py       # MODIFY: Complete RBAC
admin-agent/services/audit_service.py       # MODIFY: Real audit trails
admin-agent/services/user_management.py     # NEW: User provisioning
admin-agent/services/security_policy.py     # NEW: Policy enforcement
admin-agent/routes/users.py                 # MODIFY: Add user endpoints
admin-agent/routes/audit.py                 # MODIFY: Add audit endpoints
```

---

### 9. Product/QA Agent - Bug & Test Management
**Current State:** ❌ Minimal, needs complete implementation

**Complete New Implementation:**
- [ ] Bug clustering algorithm (group similar bugs)
- [ ] Root cause linking (trace bugs to commits/features)
- [ ] Pattern detection in bug reports
- [ ] Test execution tracking
- [ ] Coverage metrics collection
- [ ] AI-powered test case generation
- [ ] Regression detection

**Files to Create:**
```
product-agent/services/bug_analyzer.py      # NEW: Bug pattern analysis
product-agent/services/test_generator.py    # NEW: AI test generation
product-agent/services/coverage_tracker.py  # NEW: Coverage metrics
qa-agent/services/qa_service.py             # NEW: QA operations
qa-agent/services/regression_detector.py    # NEW: Regression detection
```

---

## Implementation Priority Order

**High Priority (Week 1-2):**
1. HR Agent - Resume parsing + candidate matching (revenue impact)
2. Finance Agent - Receipt scanning + forecasting (compliance + savings)
3. Sales Agent - Lead scoring + deal forecasting (direct revenue)

**Medium Priority (Week 3-4):**
4. Marketing Agent - Complete implementation (customer acquisition)
5. Support Agent - Sentiment analysis + routing (customer satisfaction)
6. Legal Agent - Contract review (risk mitigation)

**Lower Priority (Week 5-6):**
7. IT Agent - Infrastructure monitoring (operational)
8. Admin Agent - Enhanced RBAC (security)
9. Product/QA Agent - Bug analysis + test generation (quality)

---

## Common Patterns Across All Agents

### Database Pattern
```python
# Use unified database abstraction
from shared_libs.db_abstraction import get_unified_database

db = await get_unified_database()

# Create structured data (uses PostgreSQL)
await db.create_record('employees', {
    'email': 'john@company.com',
    'first_name': 'John',
    'last_name': 'Doe',
})

# Create flexible data (uses MongoDB)
await db.create_record('agent_state', {
    'agent': 'sales',
    'processing': True,
    'data': {...},
})
```

### AI Integration Pattern
```python
from shared_libs.ai_providers import get_orchestrator

orchestrator = await get_orchestrator()

# Use multi-provider with fallback
response = await orchestrator.complete(
    prompt="Generate sales email",
    model="gpt-4-mini",
    temperature=0.7,
)

# Get provider status anytime
status = orchestrator.get_provider_status()
```

### Error Handling Pattern
```python
try:
    result = await service.operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    # Log to PostgreSQL audit table
    await db.create_record('audit_logs', {
        'action': 'operation_failed',
        'error': str(e),
        'timestamp': datetime.utcnow(),
    })
    # Return meaningful error to client
    raise
```

---

## Testing Strategy

For each agent, implement:
1. **Unit tests** - Test individual functions in isolation
2. **Integration tests** - Test agent with database + AI providers
3. **E2E tests** - Test complete workflows

Example test structure:
```python
# tests/test_hr_service.py
async def test_parse_resume():
    """Test resume parsing with mock PDF."""
    parser = ResumeParser()
    result = await parser.parse_resume(mock_pdf_path)
    assert 'skills' in result
    assert 'experience' in result

async def test_match_candidates():
    """Test candidate matching against job."""
    matcher = CandidateMatcher()
    scores = await matcher.match_candidates(job_desc, candidates)
    assert len(scores) == len(candidates)
    assert all(0 <= score <= 1 for _, score in scores)
```

---

## Success Criteria for Phase 2

✅ All agents have real database operations (no more mocks)
✅ PostgreSQL and MongoDB hybrid functioning for all agents
✅ Multi-provider AI integrated in all intelligent features
✅ External API integrations working (SendGrid, Salesforce, etc.)
✅ Proper error handling and logging throughout
✅ At least 3 AI features implemented per agent
✅ All services have unit + integration tests
✅ All endpoints documented and tested

---

## Next Steps

1. Review this implementation guide
2. Prioritize which agents to implement first
3. Implement one agent at a time following the patterns above
4. Test thoroughly before moving to next agent
5. Document any deviations or improvements made
