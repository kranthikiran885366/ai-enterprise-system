# AI Enterprise System - Deployment Guide

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM
- 10GB+ disk space

## Quick Start

1. **Clone and Start**
   \`\`\`bash
   git clone <repository-url>
   cd ai-enterprise-system
   chmod +x scripts/start-system.sh
   ./scripts/start-system.sh
   \`\`\`

2. **Verify Installation**
   - Visit http://localhost:8000/docs
   - Login with admin/admin123
   - Check system status at http://localhost:8000/system/status

## Production Deployment

### Environment Configuration

1. **Update Environment Variables**
   \`\`\`bash
   # Update .env files in each service directory
   # Change default passwords and secrets
   JWT_SECRET_KEY=your-production-secret-key
   MONGODB_URL=mongodb://user:pass@prod-mongo:27017/enterprise
   \`\`\`

2. **Configure TLS/SSL**
   \`\`\`yaml
   # Add to docker-compose.yml
   nginx:
     image: nginx:alpine
     ports:
       - "443:443"
       - "80:80"
     volumes:
       - ./nginx.conf:/etc/nginx/nginx.conf
       - ./ssl:/etc/ssl/certs
   \`\`\`

### Scaling Services

\`\`\`yaml
# docker-compose.override.yml
version: '3.8'
services:
  hr-agent:
    deploy:
      replicas: 3
  finance-agent:
    deploy:
      replicas: 2
\`\`\`

### Monitoring Setup

1. **Add Prometheus & Grafana**
   \`\`\`yaml
   prometheus:
     image: prom/prometheus
     ports:
       - "9090:9090"
     volumes:
       - ./prometheus.yml:/etc/prometheus/prometheus.yml
   
   grafana:
     image: grafana/grafana
     ports:
       - "3000:3000"
     environment:
       - GF_SECURITY_ADMIN_PASSWORD=admin
   \`\`\`

2. **Configure Alerts**
   \`\`\`yaml
   # prometheus.yml
   rule_files:
     - "alert_rules.yml"
   
   alerting:
     alertmanagers:
       - static_configs:
           - targets:
             - alertmanager:9093
   \`\`\`

### Database Backup

\`\`\`bash
# MongoDB backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker exec enterprise_mongodb mongodump --out /backup/mongodb_$DATE
\`\`\`

### Health Checks

\`\`\`yaml
# Add to each service in docker-compose.yml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
\`\`\`

## Kubernetes Deployment

### Prerequisites
- Kubernetes 1.20+
- kubectl configured
- Helm 3.0+

### Deploy with Helm

1. **Create Namespace**
   \`\`\`bash
   kubectl create namespace ai-enterprise
   \`\`\`

2. **Install Chart**
   \`\`\`bash
   helm install ai-enterprise ./helm-chart \
     --namespace ai-enterprise \
     --set mongodb.auth.rootPassword=your-password
   \`\`\`

3. **Verify Deployment**
   \`\`\`bash
   kubectl get pods -n ai-enterprise
   kubectl get services -n ai-enterprise
   \`\`\`

### Ingress Configuration

\`\`\`yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-enterprise-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - api.yourcompany.com
      secretName: ai-enterprise-tls
  rules:
    - host: api.yourcompany.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: orchestrator
                port:
                  number: 8000
\`\`\`

## Security Considerations

### Network Security
- Use private networks for inter-service communication
- Implement API rate limiting
- Enable CORS only for trusted domains
- Use HTTPS in production

### Authentication & Authorization
- Change default passwords
- Implement role-based access control (RBAC)
- Use strong JWT secrets
- Enable audit logging

### Data Protection
- Encrypt data at rest
- Use encrypted connections (TLS)
- Implement data backup and recovery
- Regular security updates

## Performance Optimization

### Database Optimization
\`\`\`javascript
// MongoDB indexes
db.employees.createIndex({ "department": 1, "status": 1 })
db.expenses.createIndex({ "employee_id": 1, "expense_date": -1 })
db.rules.createIndex({ "status": 1, "priority": -1 })
\`\`\`

### Caching Strategy
```python
# Redis caching configuration
CACHE_CONFIG = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://redis:6379/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    }
}
