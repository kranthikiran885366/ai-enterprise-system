# Quick Start Guide - AI Enterprise System (Phase 1 Complete)

## TL;DR

Phase 1 has transformed the system with a solid foundation. Here's how to test it:

### 1. Environment Setup (2 minutes)
```bash
cp .env.example .env

# Add your API keys to .env:
OPENAI_API_KEY=sk-...          # From platform.openai.com
ANTHROPIC_API_KEY=sk-ant-...   # From console.anthropic.com
GOOGLE_API_KEY=AIza...         # From makersuite.google.com
DEEPSEEK_API_KEY=sk-...        # From platform.deepseek.com (optional)
```

### 2. Start Services (1 minute)
```bash
docker-compose up -d
```

### 3. Verify Setup (1 minute)
```bash
# Check all 13 containers running
docker-compose ps

# Test orchestrator
curl http://localhost:8000/

# Test each agent
for port in 8001 8002 8003 8004 8005 8006 8007 8008 8009; do
  curl http://localhost:$port/ 2>/dev/null | jq '.status'
done
```

## What's New in Phase 1

### 1. Multi-Provider AI System
**Before:** Only OpenAI, no fallback
**After:** Auto-fallback chain (OpenAI → Claude → Gemini → DeepSeek)

```python
from shared_libs.ai_providers import get_orchestrator

orchestrator = await get_orchestrator()

# Automatically tries providers in order, returns on first success
response = await orchestrator.complete(
    "Generate a sales email",
    temperature=0.7
)

# Monitor provider health
status = orchestrator.get_provider_status()
print(status)
# Output: {'openai': {'available': true, 'failures': 0, ...}, ...}
```

### 2. Hybrid Database (PostgreSQL + MongoDB)
**Before:** MongoDB only
**After:** Smart routing - SQL for structured, NoSQL for flexible

```python
from shared_libs.db_abstraction import get_unified_database

db = await get_unified_database()

# Automatically uses PostgreSQL
await db.create_record('employees', {
    'email': 'john@company.com',
    'first_name': 'John',
    'salary': 100000,
})

# Automatically uses MongoDB
await db.create_record('logs', {
    'agent': 'sales',
    'message': 'Lead scored',
})
```

### 3. Clean Infrastructure
**Before:** 10 duplicate services in docker-compose
**After:** Proper configuration with all 13 services working

### 4. Advanced Lead Scoring
**Phase 2 example:** Real multi-factor lead scoring

```python
from sales_agent.services.lead_scorer import LeadScorer

scorer = LeadScorer()
await scorer.initialize()

score = await scorer.score_lead({
    'company_name': 'TechCorp',
    'company_size': 'enterprise',
    'industry': 'technology',
    'estimated_budget': 500000,
    'decision_timeline': 'this_quarter',
    'engagement_level': 'high',
    ...
})

print(f"Overall Score: {score.overall_score:.1f}/100")
print(f"AI Analysis: {score.ai_analysis}")
print(f"Recommendation: {score.recommendation}")
```

## Running Tests

### Test 1: AI Provider Fallback (Most Important!)
```bash
python3 << 'EOF'
import asyncio
from shared_libs.ai_providers import get_orchestrator

async def test_ai():
    orchestrator = await get_orchestrator()
    
    # This will try providers in order
    response = await orchestrator.complete(
        "Write a 1-line product tagline for an AI sales tool"
    )
    
    print(f"Response: {response}")
    print(f"\nProvider Status:")
    import json
    print(json.dumps(orchestrator.get_provider_status(), indent=2))

asyncio.run(test_ai())
EOF
```

### Test 2: Database Operations
```bash
python3 << 'EOF'
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import asyncpg

async def test_databases():
    # Test MongoDB
    mongo_client = AsyncIOMotorClient('mongodb://admin:password123@localhost:27017')
    db = mongo_client['enterprise']
    collection = db['test']
    
    result = await collection.insert_one({'test': 'document'})
    print(f"MongoDB: Inserted document {result.inserted_id}")
    
    # Test PostgreSQL
    pool = await asyncpg.create_pool(
        'postgresql://enterprise_user:password123@localhost:5432/enterprise',
        min_size=1, max_size=5
    )
    
    async with pool.acquire() as conn:
        result = await conn.fetchval('SELECT 1')
        print(f"PostgreSQL: Connected successfully (result: {result})")
    
    await pool.close()

asyncio.run(test_databases())
EOF
```

### Test 3: Unified Database Interface
```bash
python3 << 'EOF'
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import asyncpg
from shared_libs.db_abstraction import initialize_unified_database, get_unified_database

async def test_unified_db():
    # Initialize
    mongo_client = AsyncIOMotorClient('mongodb://admin:password123@localhost:27017')
    mongo_db = mongo_client['enterprise']
    
    pg_pool = await asyncpg.create_pool(
        'postgresql://enterprise_user:password123@localhost:5432/enterprise',
        min_size=1, max_size=5
    )
    
    initialize_unified_database(mongo_db, pg_pool)
    db = await get_unified_database()
    
    # Create in PostgreSQL (structured data)
    emp_id = await db.create_record('employees', {
        'email': 'test@example.com',
        'first_name': 'Test',
        'last_name': 'User',
    })
    print(f"PostgreSQL: Created employee {emp_id}")
    
    # Create in MongoDB (flexible data)
    log_id = await db.create_record('logs', {
        'message': 'Test log entry',
        'level': 'info',
    })
    print(f"MongoDB: Created log {log_id}")
    
    await pg_pool.close()

asyncio.run(test_unified_db())
EOF
```

## Configuration

All configuration is environment-based. Key variables:

```env
# AI Providers (add your API keys)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
DEEPSEEK_API_KEY=sk-...

# Databases
MONGODB_URL=mongodb://admin:password123@mongodb:27017/enterprise?authSource=admin
POSTGRESQL_URL=postgresql://enterprise_user:password123@postgresql:5432/enterprise
REDIS_URL=redis://redis:6379
RABBITMQ_URL=amqp://admin:password123@rabbitmq:5672/

# Other settings
LOG_LEVEL=INFO
ENVIRONMENT=development
```

## Architecture at a Glance

```
┌─────────────────────────────────────────┐
│   9 Microservices (HR, Finance, Sales, │
│    Marketing, Support, Legal, IT,       │
│    Admin, Product/QA)                   │
└──────────────┬──────────────────────────┘
               │
        ┌──────▼─────────┐
        │  Orchestrator  │
        │  (API Gateway) │
        └──────┬─────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼──────┐         ┌────▼─────┐
│  Multi-  │         │  Hybrid   │
│ Provider │         │ Database  │
│    AI    │         │  Router   │
│  (4 LLMs)│         │(SQL+NoSQL)│
└──────────┘         └───┬──────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐  ┌──────▼──┐  ┌────────▼───┐
    │PostgreSQL│  │ MongoDB │  │Redis+RabbitMQ│
    └──────────┘  └─────────┘  └──────────────┘
```

## Key Files to Know

### New Files (Phase 1)
- `shared-libs/ai_providers.py` - Multi-provider AI system (430 lines)
- `database/postgresql.py` - PostgreSQL async client (250 lines)
- `shared-libs/db_abstraction.py` - Database routing (305 lines)
- `.env.example` - Configuration template

### Modified Files
- `docker-compose.yml` - Fixed duplicates, added PostgreSQL
- `shared-libs/requirements.txt` - Added asyncpg, aiohttp

### Documentation
- `PHASE_1_COMPLETE.md` - Foundation details
- `PHASE_2_IMPLEMENTATION.md` - How to build agent features
- `UPGRADE_SUMMARY.md` - System overview

## Common Tasks

### Add a New AI Provider
1. Create new client class in `ai_providers.py`
2. Implement `complete()` method
3. Add to `provider_map` in `AIOrchestrator`
4. Add API key to `.env.example`

### Add a New Data Model
1. Add table to PostgreSQL schema in `database/postgresql.py`
2. Use via: `db.create_record('table_name', data)`
3. Routing is automatic (PostgreSQL vs MongoDB)

### Integrate External API
1. Add API key to `.env`
2. Create integration module in agent service
3. Call during business logic

## Troubleshooting

### "Docker containers not starting"
```bash
# Check logs
docker-compose logs orchestrator

# Restart all
docker-compose restart
```

### "Database connection refused"
```bash
# Check services started
docker-compose ps

# Wait for health checks to pass (30-60 seconds)
# Then try again
```

### "API key not recognized"
```bash
# Verify .env file exists
ls -la .env

# Check format
cat .env | grep API_KEY

# Make sure docker-compose restarted after .env change
docker-compose restart
```

### "AI response empty"
```bash
# Check AI provider status
curl http://localhost:8000/health

# Check orchestrator logs
docker-compose logs orchestrator | tail -20

# Verify API key is correct in .env
```

## Next Steps

### For Testing
1. ✅ Environment setup (2 min)
2. ✅ Start services (1 min)
3. ✅ Run test scripts above
4. Run test suite: `python -m pytest tests/`

### For Development
1. Review `PHASE_2_IMPLEMENTATION.md`
2. Choose an agent to enhance
3. Implement features from guide
4. Test thoroughly

### For Production
1. Complete Phase 2 (all agents enhanced)
2. Add monitoring (Phase 4)
3. Add testing suite (Phase 5)
4. Deploy to Kubernetes (Phase 6)
5. Add admin dashboard (Phase 7)

## Useful Commands

```bash
# View all services
docker-compose ps

# View logs
docker-compose logs -f orchestrator

# Stop everything
docker-compose down

# Remove volumes (clean slate)
docker-compose down -v

# Run a specific service
docker-compose up -d orchestrator

# Scale a service (e.g., 3 HR agents)
docker-compose up -d --scale hr-agent=3

# Connect to database
psql postgresql://enterprise_user:password123@localhost:5432/enterprise

# MongoDB shell
mongosh mongodb://admin:password123@localhost:27017
```

## What Works Now

✅ 4 AI providers with fallback
✅ PostgreSQL + MongoDB hybrid routing
✅ All 13 services running
✅ Health checks
✅ Environment-based config
✅ Async operations throughout
✅ Connection pooling
✅ Error handling and logging

## What's Coming (Phase 2+)

✅ Advanced features in all 9 agents
✅ Monitoring and metrics
✅ Testing suite
✅ Kubernetes deployment
✅ Admin dashboard
✅ Full production readiness

## Get Help

1. **Read:** `PHASE_2_IMPLEMENTATION.md` for agent details
2. **Check:** Code docstrings for API documentation
3. **Review:** `PHASE_1_COMPLETE.md` for architecture details
4. **Ask:** Check logs for specific error messages

---

**You're ready to test and extend the system!**

Start with running the tests above, then move to Phase 2 implementation.
