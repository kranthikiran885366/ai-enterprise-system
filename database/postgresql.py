"""PostgreSQL async connection manager for structured data."""

import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool, Connection
from loguru import logger


class PostgreSQLManager:
    """PostgreSQL connection pool manager."""
    
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv(
            "POSTGRESQL_URL",
            "postgresql://enterprise_user:password123@localhost:5432/enterprise"
        )
        self.pool: Optional[Pool] = None
    
    async def connect(self):
        """Initialize connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=5,
                max_size=20,
                command_timeout=60,
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    async def disconnect(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    async def acquire(self) -> Connection:
        """Acquire connection from pool."""
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")
        return await self.pool.acquire()
    
    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[Connection, None]:
        """Context manager for database connection."""
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.pool.release(conn)
    
    async def execute(self, query: str, *args):
        """Execute query and return result."""
        async with self.connection() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args):
        """Fetch multiple rows."""
        async with self.connection() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args):
        """Fetch single row."""
        async with self.connection() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args):
        """Fetch single value."""
        async with self.connection() as conn:
            return await conn.fetchval(query, *args)
    
    async def transaction(self):
        """Get transaction context manager."""
        async with self.connection() as conn:
            async with conn.transaction():
                yield conn


# Global PostgreSQL manager instance
pg_manager = PostgreSQLManager()


async def get_connection() -> Connection:
    """Dependency to get database connection."""
    return await pg_manager.acquire()


async def initialize_schema():
    """Initialize database schema with all required tables."""
    async with pg_manager.connection() as conn:
        # Enable UUID extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
        
        # Employees table (HR Agent)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                email VARCHAR(255) UNIQUE NOT NULL,
                first_name VARCHAR(255) NOT NULL,
                last_name VARCHAR(255) NOT NULL,
                department VARCHAR(255),
                position VARCHAR(255),
                salary DECIMAL(12, 2),
                hire_date DATE,
                status VARCHAR(50) DEFAULT 'active',
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Financial records table (Finance Agent)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_records (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                type VARCHAR(50) NOT NULL,
                amount DECIMAL(12, 2) NOT NULL,
                currency VARCHAR(3) DEFAULT 'USD',
                description TEXT,
                employee_id UUID REFERENCES employees(id),
                category VARCHAR(100),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Leads table (Sales Agent)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                company_name VARCHAR(255) NOT NULL,
                contact_name VARCHAR(255),
                email VARCHAR(255),
                phone VARCHAR(20),
                status VARCHAR(50) DEFAULT 'new',
                score DECIMAL(5, 2),
                estimated_value DECIMAL(12, 2),
                source VARCHAR(100),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Deals table (Sales Agent)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS deals (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                lead_id UUID REFERENCES leads(id),
                title VARCHAR(255) NOT NULL,
                amount DECIMAL(12, 2),
                probability DECIMAL(5, 2),
                expected_close_date DATE,
                stage VARCHAR(100),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Marketing campaigns table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS marketing_campaigns (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name VARCHAR(255) NOT NULL,
                description TEXT,
                status VARCHAR(50) DEFAULT 'draft',
                channel VARCHAR(100),
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                budget DECIMAL(12, 2),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Support tickets table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS support_tickets (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                ticket_number VARCHAR(50) UNIQUE NOT NULL,
                customer_name VARCHAR(255),
                email VARCHAR(255),
                subject VARCHAR(255) NOT NULL,
                description TEXT,
                status VARCHAR(50) DEFAULT 'open',
                priority VARCHAR(50) DEFAULT 'medium',
                assigned_to UUID REFERENCES employees(id),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Documents table (Legal Agent)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                title VARCHAR(255) NOT NULL,
                type VARCHAR(100),
                content TEXT,
                status VARCHAR(50) DEFAULT 'draft',
                owner_id UUID REFERENCES employees(id),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Audit logs table (Admin Agent)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES employees(id),
                action VARCHAR(255) NOT NULL,
                resource_type VARCHAR(100),
                resource_id UUID,
                changes JSONB,
                ip_address VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_employees_email ON employees(email)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_employees_status ON employees(status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_financial_records_employee_id ON financial_records(employee_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deals_lead_id ON deals(lead_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_deals_stage ON deals(stage)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_campaigns_status ON marketing_campaigns(status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status ON support_tickets(status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_assigned_to ON support_tickets(assigned_to)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner_id ON documents(owner_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at)")
        
        logger.info("PostgreSQL schema initialized successfully")
