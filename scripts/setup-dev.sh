#!/bin/bash

echo "🔧 Setting up development environment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Create .env files for each service
services=("orchestrator" "hr-agent" "finance-agent" "ai-decision-engine")

for service in "${services[@]}"; do
    if [ ! -f "$service/.env" ]; then
        cp "$service/.env.example" "$service/.env" 2>/dev/null || echo "# Environment variables for $service" > "$service/.env"
        echo "📝 Created .env file for $service"
    fi
done

echo "✅ Development environment setup complete!"
echo ""
echo "🚀 To start development:"
echo "1. Start infrastructure: docker-compose up -d mongodb redis rabbitmq"
echo "2. Run individual services: cd <service> && uvicorn main:app --reload"
