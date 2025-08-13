# README_DEPLOYMENT.md - Comprehensive Deployment Guide
# ============================================================================
# Master Generators ODE System - Deployment Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployments](#cloud-deployments)
6. [Production Configuration](#production-configuration)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- Node.js (for Railway deployment)
- kubectl (for Kubernetes)

### Installation
```bash
# Clone repository
git clone https://github.com/master-generators/master-generators-ode.git
cd master-generators-ode

# Run setup
make setup

# Start services
make run-all
```

Access:
- Streamlit App: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Local Development

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Create .env file
cp .env.example .env
# Edit .env with your configuration
```

### Running Services

#### Option 1: Using Make
```bash
make run-app    # Streamlit only
make run-api    # API only
make run-all    # Both services
```

#### Option 2: Manual
```bash
# Terminal 1: Streamlit
streamlit run master_generators_app.py

# Terminal 2: API
uvicorn api_server:app --reload --port 8000

# Terminal 3: Celery (optional)
celery -A src.tasks worker --loglevel=info

# Terminal 4: Flower (optional)
celery -A src.tasks flower
```

### Testing
```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration
make test-performance

# Generate coverage report
make coverage
```

## Docker Deployment

### Build and Run
```bash
# Build images
make docker-build

# Start services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

### Docker Compose Services
- `app`: Main application (Streamlit + API)
- `redis`: Cache and message broker
- `postgres`: Database
- `nginx`: Reverse proxy
- `celery`: Background task worker
- `flower`: Celery monitoring
- `prometheus`: Metrics collection
- `grafana`: Metrics visualization

### Custom Configuration
```yaml
# docker-compose.override.yml
version: '3.8'

services:
  app:
    environment:
      - DEBUG=True
      - ML_BATCH_SIZE=64
    volumes:
      - ./custom_models:/app/models
```

## Kubernetes Deployment

### Prerequisites
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Configure cluster access
kubectl config set-context my-cluster
```

### Deploy to Kubernetes
```bash
# Apply configurations
kubectl apply -f kubernetes/deployment.yaml

# Check status
kubectl get pods -n master-generators
kubectl get services -n master-generators

# Scale deployment
kubectl scale deployment master-generators-app -n master-generators --replicas=5

# Update image
kubectl set image deployment/master-generators-app app=master-generators:v2.0.0 -n master-generators
```

### Accessing Services
```bash
# Port forwarding (development)
kubectl port-forward service/master-generators-service 8000:8000 -n master-generators

# Get external IP (production)
kubectl get service master-generators-service -n master-generators
```

## Cloud Deployments

### Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
railway up

# View logs
railway logs
```

### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI
docker build -t master-generators .
docker tag master-generators:latest $ECR_URI:latest
docker push $ECR_URI:latest

# Update service
aws ecs update-service --cluster master-generators --service app --force-new-deployment
```

### Google Cloud Run
```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/master-generators

# Deploy
gcloud run deploy master-generators \
  --image gcr.io/PROJECT_ID/master-generators \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances
```bash
# Push to ACR
az acr build --registry myregistry --image master-generators .

# Deploy
az container create \
  --resource-group myResourceGroup \
  --name master-generators \
  --image myregistry.azurecr.io/master-generators:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 8501
```

## Production Configuration

### Environment Variables
```bash
# .env.production
APP_ENV=production
DEBUG=False
SECRET_KEY=your-production-secret-key
DATABASE_URL=postgresql://user:pass@db:5432/master_generators
REDIS_URL=redis://redis:6379
ALLOWED_ORIGINS=https://yourdomain.com
RATE_LIMIT_ENABLED=true
```

### SSL/TLS Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    location / {
        proxy_pass http://app:8501;
        proxy_set_header X-Forwarded-Proto https;
    }
}
```

### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Backup Strategy
```bash
# Automated backups
0 2 * * * /usr/local/bin/backup.sh

# backup.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump $DATABASE_URL | gzip > backups/db_$DATE.sql.gz
tar -czf backups/models_$DATE.tar.gz models/
aws s3 sync backups/ s3://my-backup-bucket/
```

## Monitoring and Maintenance

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Comprehensive check
./scripts/health_check.sh
```

### Monitoring Stack
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Flower**: http://localhost:5555

### Log Management
```bash
# View logs
docker-compose logs -f app

# Log rotation
/var/log/master-generators/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

### Performance Tuning
```python
# Optimize workers
WORKERS = (2 * CPU_COUNT) + 1

# Cache configuration
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 10000

# Database pool
SQLALCHEMY_POOL_SIZE = 20
SQLALCHEMY_MAX_OVERFLOW = 40
```

### Troubleshooting

#### Common Issues

1. **Port already in use**
```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>
```

2. **Docker memory issues**
```json
{
  "default-runtime": "nvidia",
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

3. **Database connection issues**
```bash
# Check connectivity
pg_isready -h localhost -p 5432

# Reset connections
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE datname = 'master_generators';
```

### Support

For issues and questions:
- GitHub Issues: https://github.com/master-generators/master-generators-ode/issues
- Documentation: https://master-generators.readthedocs.io
- Email: support@master-generators.com
