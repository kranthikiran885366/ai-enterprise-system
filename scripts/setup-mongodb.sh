#!/bin/bash

echo "🗄️ Setting up MongoDB for AI Enterprise System..."

# Wait for MongoDB to be ready
echo "Waiting for MongoDB to start..."
sleep 10

# Initialize MongoDB with collections and indexes
echo "Initializing MongoDB collections and indexes..."
docker exec enterprise_mongodb mongosh --eval "load('/docker-entrypoint-initdb.d/init-mongodb.js')" enterprise

# Seed initial data
echo "Seeding initial data..."
docker exec enterprise_mongodb mongosh --eval "load('/docker-entrypoint-initdb.d/seed-data.js')" enterprise

echo "✅ MongoDB setup completed!"
echo ""
echo "📊 Database Information:"
echo "• Database: enterprise"
echo "• Collections: employees, expenses, leads, tickets, it_assets, contracts, campaigns, rules"
echo "• Indexes: Optimized for query performance"
echo "• Sample Data: Seeded with realistic test data"
echo ""
echo "🔗 Connection Details:"
echo "• Host: localhost:27017"
echo "• Username: admin"
echo "• Password: password123"
echo "• Database: enterprise"