# scripts/deploy.sh - Enhanced Deployment Script
# ============================================================================
#!/bin/bash

# Master Generators Enhanced Deployment Script
# Supports multiple deployment targets: Docker, Kubernetes, Railway, AWS

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Functions
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
    exit 1
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_status "Docker is installed"
    else
        print_warning "Docker is not installed"
    fi
    
    # Check Kubernetes
    if command -v kubectl &> /dev/null; then
        print_status "kubectl is installed"
    else
        print_warning "kubectl is not installed"
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        print_status "Python is installed"
    else
        print_error "Python 3 is required"
    fi
}

# Run tests
run_tests() {
    print_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run unit tests
    python -m pytest tests/ -v --cov=src --cov-report=term || print_warning "Some tests failed"
    
    # Run startup check
    python startup_check.py || print_error "Startup check failed"
    
    print_status "Tests completed"
}

# Build Docker image
build_docker() {
    print_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build with buildkit for better caching
    DOCKER_BUILDKIT=1 docker build \
        --tag master-generators:latest \
        --tag master-generators:$TIMESTAMP \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VERSION=$(python -c "from src import __version__; print(__version__)") \
        .
    
    print_status "Docker image built successfully"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    print_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Check if .env exists
    if [ ! -f .env ]; then
        print_warning ".env file not found. Copying from .env.example"
        cp .env.example .env
    fi
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    print_info "Waiting for services to start..."
    sleep 10
    
    # Check health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "API is healthy"
    else
        print_warning "API health check failed"
    fi
    
    print_status "Docker Compose deployment completed"
    print_info "Services available at:"
    print_info "  - Streamlit: http://localhost:8501"
    print_info "  - API: http://localhost:8000"
    print_info "  - API Docs: http://localhost:8000/docs"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    print_info "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
    fi
    
    # Apply configurations
    kubectl apply -f kubernetes/deployment.yaml
    
    # Wait for pods to be ready
    print_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod \
        -l app=master-generators \
        -n master-generators \
        --timeout=300s
    
    # Get service endpoints
    print_status "Kubernetes deployment completed"
    print_info "Service endpoints:"
    kubectl get services -n master-generators
}

# Deploy to Railway
deploy_railway() {
    print_info "Deploying to Railway..."
    
    cd "$PROJECT_ROOT"
    
    # Check Railway CLI
    if ! command -v railway &> /dev/null; then
        print_error "Railway CLI is not installed. Install with: npm install -g @railway/cli"
    fi
    
    # Deploy
    railway up
    
    print_status "Railway deployment initiated"
}

# Deploy to AWS ECS
deploy_aws_ecs() {
    print_info "Deploying to AWS ECS..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed"
    fi
    
    # Build and push to ECR
    REGION=${AWS_REGION:-us-east-1}
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_URI=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/master-generators
    
    # Login to ECR
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI
    
    # Tag and push
    docker tag master-generators:latest $ECR_URI:latest
    docker push $ECR_URI:latest
    
    # Update ECS service
    aws ecs update-service \
        --cluster master-generators-cluster \
        --service master-generators-service \
        --force-new-deployment
    
    print_status "AWS ECS deployment initiated"
}

# Create backup
create_backup() {
    print_info "Creating backup..."
    
    BACKUP_DIR="$PROJECT_ROOT/backups/$TIMESTAMP"
    mkdir -p "$BACKUP_DIR"
    
    # Backup models
    if [ -d "$PROJECT_ROOT/models" ]; then
        tar -czf "$BACKUP_DIR/models.tar.gz" -C "$PROJECT_ROOT" models/
        print_status "Models backed up"
    fi
    
    # Backup data
    if [ -d "$PROJECT_ROOT/data" ]; then
        tar -czf "$BACKUP_DIR/data.tar.gz" -C "$PROJECT_ROOT" data/
        print_status "Data backed up"
    fi
    
    # Backup database (if using docker-compose)
    if docker-compose ps | grep -q postgres; then
        docker-compose exec -T postgres pg_dump -U master_gen master_generators | gzip > "$BACKUP_DIR/database.sql.gz"
        print_status "Database backed up"
    fi
    
    print_status "Backup created at $BACKUP_DIR"
}

# Main menu
show_menu() {
    echo ""
    echo "================================================"
    echo "Master Generators Deployment Script"
    echo "================================================"
    echo "1. Run Tests"
    echo "2. Build Docker Image"
    echo "3. Deploy with Docker Compose"
    echo "4. Deploy to Kubernetes"
    echo "5. Deploy to Railway"
    echo "6. Deploy to AWS ECS"
    echo "7. Create Backup"
    echo "8. Full Deployment (Tests + Docker + Deploy)"
    echo "9. Exit"
    echo "================================================"
    echo -n "Select option: "
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    # Check if running with arguments
    if [ $# -gt 0 ]; then
        case "$1" in
            test)
                run_tests
                ;;
            build)
                build_docker
                ;;
            deploy-docker)
                deploy_docker_compose
                ;;
            deploy-k8s)
                deploy_kubernetes
                ;;
            deploy-railway)
                deploy_railway
                ;;
            deploy-aws)
                deploy_aws_ecs
                ;;
            backup)
                create_backup
                ;;
            full)
                run_tests
                build_docker
                deploy_docker_compose
                ;;
            *)
                print_error "Unknown command: $1"
                ;;
        esac
    else
        # Interactive mode
        check_prerequisites
        
        while true; do
            show_menu
            read -r option
            
            case $option in
                1) run_tests ;;
                2) build_docker ;;
                3) deploy_docker_compose ;;
                4) deploy_kubernetes ;;
                5) deploy_railway ;;
                6) deploy_aws_ecs ;;
                7) create_backup ;;
                8) 
                    run_tests
                    build_docker
                    deploy_docker_compose
                    ;;
                9) 
                    print_info "Exiting..."
                    exit 0
                    ;;
                *)
                    print_warning "Invalid option"
                    ;;
            esac
        done
    fi
}

# Run main
main "$@"
