#!/bin/bash

echo "🚀 Starting All AI Enterprise Agents..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start all services
echo "📦 Building and starting all agents..."
docker-compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 45

# Initialize MongoDB
echo "🗄️ Initializing MongoDB..."
docker cp scripts/init-mongodb.js enterprise_mongodb:/docker-entrypoint-initdb.d/
docker cp scripts/seed-data.js enterprise_mongodb:/docker-entrypoint-initdb.d/
docker exec enterprise_mongodb mongosh enterprise /docker-entrypoint-initdb.d/init-mongodb.js
docker exec enterprise_mongodb mongosh enterprise /docker-entrypoint-initdb.d/seed-data.js

# Check service health
echo "🔍 Checking service health..."

services=(
    "orchestrator:8000"
    "hr-agent:8001" 
    "finance-agent:8002"
    "sales-agent:8003"
    "marketing-agent:8004"
    "it-agent:8005"
    "admin-agent:8006"
    "legal-agent:8007"
    "support-agent:8008"
    "ai-decision-engine:8009"
)

healthy_services=0
total_services=${#services[@]}

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -s "http://localhost:$port/health" > /dev/null; then
        echo "✅ $name is healthy"
        ((healthy_services++))
    else
        echo "❌ $name is not responding"
    fi
done

echo ""
echo "🎉 AI Enterprise System Status: $healthy_services/$total_services services healthy"
echo ""
echo "📊 Service Access Points:"
echo "• Central Orchestrator: http://localhost:8000/docs"
echo "• HR Agent: http://localhost:8001/docs"
echo "• Finance Agent: http://localhost:8002/docs"
echo "• Sales Agent: http://localhost:8003/docs"
echo "• Marketing Agent: http://localhost:8004/docs"
echo "• IT Agent: http://localhost:8005/docs"
echo "• Admin Agent: http://localhost:8006/docs"
echo "• Legal Agent: http://localhost:8007/docs"
echo "• Support Agent: http://localhost:8008/docs"
echo "• AI Decision Engine: http://localhost:8009/docs"
echo ""
echo "🔐 Default Login:"
echo "• Username: admin"
echo "• Password: admin123"
echo ""
echo "🛠️ Management Interfaces:"
echo "• RabbitMQ: http://localhost:15672 (admin/password123)"
echo "• MongoDB: localhost:27017 (admin/password123)"
echo ""
echo "📝 To stop the system: docker-compose down"
echo "📝 To view logs: docker-compose logs -f [service-name]"