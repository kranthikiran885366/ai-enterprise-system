"""Unified database abstraction layer routing to MongoDB or PostgreSQL."""

import os
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
import asyncpg
from loguru import logger


class DataCategory(str, Enum):
    """Data categories for routing decisions."""
    STRUCTURED = "structured"      # PostgreSQL: employees, financial, deals, campaigns
    UNSTRUCTURED = "unstructured"  # MongoDB: messages, logs, documents, metadata
    FLEXIBLE = "flexible"          # MongoDB: user-defined, polymorphic data


class DatabaseRouter:
    """Routes database operations to appropriate backend (PostgreSQL or MongoDB)."""
    
    # PostgreSQL collections (structured data)
    POSTGRESQL_COLLECTIONS = {
        "employees": DataCategory.STRUCTURED,
        "financial_records": DataCategory.STRUCTURED,
        "leads": DataCategory.STRUCTURED,
        "deals": DataCategory.STRUCTURED,
        "marketing_campaigns": DataCategory.STRUCTURED,
        "support_tickets": DataCategory.STRUCTURED,
        "documents": DataCategory.STRUCTURED,
        "audit_logs": DataCategory.STRUCTURED,
    }
    
    # MongoDB collections (unstructured/flexible data)
    MONGODB_COLLECTIONS = {
        "conversations": DataCategory.UNSTRUCTURED,
        "logs": DataCategory.UNSTRUCTURED,
        "messages": DataCategory.UNSTRUCTURED,
        "notifications": DataCategory.UNSTRUCTURED,
        "cache": DataCategory.FLEXIBLE,
        "metadata": DataCategory.FLEXIBLE,
        "agent_state": DataCategory.FLEXIBLE,
    }
    
    def __init__(self, mongo_db: AsyncIOMotorDatabase, pg_pool: asyncpg.Pool):
        self.mongo_db = mongo_db
        self.pg_pool = pg_pool
    
    def should_use_postgresql(self, collection: str) -> bool:
        """Determine if collection should use PostgreSQL."""
        return collection in self.POSTGRESQL_COLLECTIONS
    
    def should_use_mongodb(self, collection: str) -> bool:
        """Determine if collection should use MongoDB."""
        return collection in self.MONGODB_COLLECTIONS
    
    async def create_record(self, collection: str, data: Dict[str, Any]) -> str:
        """Create a new record in appropriate database."""
        if self.should_use_postgresql(collection):
            return await self._create_postgresql(collection, data)
        else:
            return await self._create_mongodb(collection, data)
    
    async def read_record(self, collection: str, id: str) -> Optional[Dict[str, Any]]:
        """Read record by ID from appropriate database."""
        if self.should_use_postgresql(collection):
            return await self._read_postgresql(collection, id)
        else:
            return await self._read_mongodb(collection, id)
    
    async def update_record(self, collection: str, id: str, data: Dict[str, Any]) -> bool:
        """Update record in appropriate database."""
        if self.should_use_postgresql(collection):
            return await self._update_postgresql(collection, id, data)
        else:
            return await self._update_mongodb(collection, id, data)
    
    async def delete_record(self, collection: str, id: str) -> bool:
        """Delete record from appropriate database."""
        if self.should_use_postgresql(collection):
            return await self._delete_postgresql(collection, id)
        else:
            return await self._delete_mongodb(collection, id)
    
    async def query(self, collection: str, filters: Dict[str, Any], limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
        """Query records from appropriate database."""
        if self.should_use_postgresql(collection):
            return await self._query_postgresql(collection, filters, limit, skip)
        else:
            return await self._query_mongodb(collection, filters, limit, skip)
    
    # PostgreSQL implementations
    async def _create_postgresql(self, table: str, data: Dict[str, Any]) -> str:
        """Insert record into PostgreSQL."""
        try:
            columns = ", ".join(data.keys())
            placeholders = ", ".join(f"${i+1}" for i in range(len(data)))
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING id"
            
            async with self.pg_pool.acquire() as conn:
                result = await conn.fetchval(query, *data.values())
            
            logger.info(f"Created record in {table}: {result}")
            return str(result)
        except Exception as e:
            logger.error(f"PostgreSQL create failed: {e}")
            raise
    
    async def _read_postgresql(self, table: str, id: str) -> Optional[Dict[str, Any]]:
        """Fetch record from PostgreSQL."""
        try:
            query = f"SELECT * FROM {table} WHERE id = $1"
            
            async with self.pg_pool.acquire() as conn:
                row = await conn.fetchrow(query, id)
            
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"PostgreSQL read failed: {e}")
            raise
    
    async def _update_postgresql(self, table: str, id: str, data: Dict[str, Any]) -> bool:
        """Update record in PostgreSQL."""
        try:
            # Build SET clause dynamically
            set_clause = ", ".join(f"{k} = ${i+1}" for i, k in enumerate(data.keys()))
            query = f"UPDATE {table} SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ${len(data)+1}"
            
            async with self.pg_pool.acquire() as conn:
                result = await conn.execute(query, *data.values(), id)
            
            success = result.endswith("1")
            logger.info(f"Updated record in {table}: {id}")
            return success
        except Exception as e:
            logger.error(f"PostgreSQL update failed: {e}")
            raise
    
    async def _delete_postgresql(self, table: str, id: str) -> bool:
        """Delete record from PostgreSQL."""
        try:
            query = f"DELETE FROM {table} WHERE id = $1"
            
            async with self.pg_pool.acquire() as conn:
                result = await conn.execute(query, id)
            
            success = result.endswith("1")
            logger.info(f"Deleted record from {table}: {id}")
            return success
        except Exception as e:
            logger.error(f"PostgreSQL delete failed: {e}")
            raise
    
    async def _query_postgresql(self, table: str, filters: Dict[str, Any], limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
        """Query records from PostgreSQL."""
        try:
            where_clause = " AND ".join(f"{k} = ${i+1}" for i, k in enumerate(filters.keys()))
            query = f"SELECT * FROM {table}"
            
            if where_clause:
                query += f" WHERE {where_clause}"
            
            query += f" LIMIT {limit} OFFSET {skip}"
            
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(query, *filters.values())
            
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"PostgreSQL query failed: {e}")
            raise
    
    # MongoDB implementations
    async def _create_mongodb(self, collection: str, data: Dict[str, Any]) -> str:
        """Insert document into MongoDB."""
        try:
            result = await self.mongo_db[collection].insert_one(data)
            logger.info(f"Created document in {collection}: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB create failed: {e}")
            raise
    
    async def _read_mongodb(self, collection: str, id: str) -> Optional[Dict[str, Any]]:
        """Fetch document from MongoDB."""
        try:
            from bson.objectid import ObjectId
            
            doc = await self.mongo_db[collection].find_one({"_id": ObjectId(id)})
            
            if doc:
                doc["id"] = str(doc.pop("_id"))
                return doc
            return None
        except Exception as e:
            logger.error(f"MongoDB read failed: {e}")
            raise
    
    async def _update_mongodb(self, collection: str, id: str, data: Dict[str, Any]) -> bool:
        """Update document in MongoDB."""
        try:
            from bson.objectid import ObjectId
            
            result = await self.mongo_db[collection].update_one(
                {"_id": ObjectId(id)},
                {"$set": data}
            )
            
            success = result.modified_count > 0
            logger.info(f"Updated document in {collection}: {id}")
            return success
        except Exception as e:
            logger.error(f"MongoDB update failed: {e}")
            raise
    
    async def _delete_mongodb(self, collection: str, id: str) -> bool:
        """Delete document from MongoDB."""
        try:
            from bson.objectid import ObjectId
            
            result = await self.mongo_db[collection].delete_one({"_id": ObjectId(id)})
            
            success = result.deleted_count > 0
            logger.info(f"Deleted document from {collection}: {id}")
            return success
        except Exception as e:
            logger.error(f"MongoDB delete failed: {e}")
            raise
    
    async def _query_mongodb(self, collection: str, filters: Dict[str, Any], limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
        """Query documents from MongoDB."""
        try:
            cursor = self.mongo_db[collection].find(filters).limit(limit).skip(skip)
            docs = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string
            for doc in docs:
                if "_id" in doc:
                    doc["id"] = str(doc.pop("_id"))
            
            return docs
        except Exception as e:
            logger.error(f"MongoDB query failed: {e}")
            raise


class UnifiedDatabase:
    """Unified database interface combining MongoDB and PostgreSQL."""
    
    def __init__(self, mongo_db: AsyncIOMotorDatabase, pg_pool: asyncpg.Pool):
        self.router = DatabaseRouter(mongo_db, pg_pool)
        self.mongo_db = mongo_db
        self.pg_pool = pg_pool
    
    def get_mongo_collection(self, name: str) -> AsyncIOMotorCollection:
        """Get MongoDB collection directly."""
        return self.mongo_db[name]
    
    async def get_postgres_connection(self):
        """Get PostgreSQL connection from pool."""
        return await self.pg_pool.acquire()
    
    # Forward router methods
    async def create_record(self, collection: str, data: Dict[str, Any]) -> str:
        """Create record using router."""
        return await self.router.create_record(collection, data)
    
    async def read_record(self, collection: str, id: str) -> Optional[Dict[str, Any]]:
        """Read record using router."""
        return await self.router.read_record(collection, id)
    
    async def update_record(self, collection: str, id: str, data: Dict[str, Any]) -> bool:
        """Update record using router."""
        return await self.router.update_record(collection, id, data)
    
    async def delete_record(self, collection: str, id: str) -> bool:
        """Delete record using router."""
        return await self.router.delete_record(collection, id)
    
    async def query(self, collection: str, filters: Dict[str, Any], limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
        """Query records using router."""
        return await self.router.query(collection, filters, limit, skip)


# Global unified database instance
_unified_db: Optional[UnifiedDatabase] = None


async def get_unified_database() -> UnifiedDatabase:
    """Get or create global unified database instance."""
    # Assumes pg_pool and mongo_db are set up globally elsewhere
    # This will be initialized in each agent's main.py
    global _unified_db
    if _unified_db is None:
        raise RuntimeError("Unified database not initialized")
    return _unified_db


def initialize_unified_database(mongo_db: AsyncIOMotorDatabase, pg_pool: asyncpg.Pool):
    """Initialize global unified database instance."""
    global _unified_db
    _unified_db = UnifiedDatabase(mongo_db, pg_pool)
    logger.info("Unified database initialized")
