#!/bin/bash

echo "🚀 Starting AI Enterprise System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start all services
echo "📦 Building and starting all services..."
docker-compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

services=(
    "orchestrator:8000"
    "hr-agent:8001" 
    "finance-agent:8002"
    "ai-decision-engine:8009"
)

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -s "http://localhost:$port/health" > /dev/null; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is not responding"
    fi
done

echo ""
echo "🎉 AI Enterprise System is running!"
echo ""
echo "📊 Access Points:"
echo "• Central Orchestrator: http://localhost:8000/docs"
echo "• HR Agent: http://localhost:8001/docs"
echo "• Finance Agent: http://localhost:8002/docs"
echo "• AI Decision Engine: http://localhost:8009/docs"
echo ""
echo "🔐 Default Login:"
echo "• Username: admin"
echo "• Password: admin123"
echo ""
echo "🛠️ Management Interfaces:"
echo "• RabbitMQ: http://localhost:15672 (admin/password123)"
echo "• MongoDB: localhost:27017"
echo ""
echo "📝 To stop the system: docker-compose down"
