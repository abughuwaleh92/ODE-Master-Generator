# Makefile - Complete Build Automation
# ============================================================================
# Master Generators Makefile - Enhanced Version

.PHONY: help install test run clean docker deploy docs

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
COMPOSE := docker-compose
PROJECT_NAME := master-generators-ode
VERSION := $(shell python -c "from src import __version__; print(__version__)")

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Default target
help:
	@echo "$(GREEN)Master Generators ODE System - v$(VERSION)$(NC)"
	@echo "============================================="
	@echo "Available commands:"
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make install          - Install all dependencies"
	@echo "  make install-dev      - Install with dev dependencies"
	@echo "  make setup            - Complete project setup"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-performance - Run performance tests"
	@echo "  make coverage         - Generate test coverage report"
	@echo "  make lint             - Run code linting"
	@echo "  make format           - Format code with black"
	@echo "  make type-check       - Run type checking with mypy"
	@echo ""
	@echo "$(YELLOW)Running:$(NC)"
	@echo "  make run-app          - Run Streamlit application"
	@echo "  make run-api          - Run FastAPI server"
	@echo "  make run-all          - Run all services"
	@echo "  make run-celery       - Run Celery worker"
	@echo "  make run-flower       - Run Flower monitoring"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  make docker-build     - Build Docker images"
	@echo "  make docker-up        - Start with Docker Compose"
	@echo "  make docker-down      - Stop Docker services"
	@echo "  make docker-logs      - View Docker logs"
	@echo "  make docker-clean     - Clean Docker resources"
	@echo ""
	@echo "$(YELLOW)Deployment:$(NC)"
	@echo "  make deploy-dev       - Deploy to development"
	@echo "  make deploy-staging   - Deploy to staging"
	@echo "  make deploy-prod      - Deploy to production"
	@echo "  make deploy-k8s       - Deploy to Kubernetes"
	@echo ""
	@echo "$(YELLOW)Documentation:$(NC)"
	@echo "  make docs             - Generate documentation"
	@echo "  make docs-serve       - Serve documentation locally"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(NC)"
	@echo "  make clean            - Clean temporary files"
	@echo "  make clean-all        - Clean everything including models"
	@echo "  make backup           - Create backup"
	@echo "  make version          - Show version"

# Installation targets
install:
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: install
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

setup: install-dev
	@echo "$(GREEN)Setting up project...$(NC)"
	@mkdir -p models data/generated data/training data/examples logs static/css static/js static/images
	@touch models/.gitkeep data/.gitkeep logs/.gitkeep
	@cp -n .env.example .env || true
	$(PYTHON) startup_check.py
	@echo "$(GREEN)✓ Project setup complete$(NC)"

# Testing targets
test:
	@echo "$(GREEN)Running all tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v

test-unit:
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTHON) -m pytest tests/test_generators_improved.py tests/test_functions.py -v

test-integration:
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTHON) -m pytest tests/test_integration_improved.py -v

test-performance:
	@echo "$(GREEN)Running performance tests...$(NC)"
	$(PYTHON) -m pytest tests/test_performance.py -v --benchmark-only

coverage:
	@echo "$(GREEN)Generating coverage report...$(NC)"
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

# Code quality targets
lint:
	@echo "$(GREEN)Running linters...$(NC)"
	flake8 src/ tests/ --max-line-length=120 --exclude=__pycache__
	pylint src/ --max-line-length=120 --disable=C0111

format:
	@echo "$(GREEN)Formatting code...$(NC)"
	black src/ tests/ --line-length=120
	isort src/ tests/ --profile black

type-check:
	@echo "$(GREEN)Running type checks...$(NC)"
	mypy src/ --ignore-missing-imports

# Running targets
run-app:
	@echo "$(GREEN)Starting Streamlit app...$(NC)"
	streamlit run master_generators_app.py --server.port=8501

run-api:
	@echo "$(GREEN)Starting API server...$(NC)"
	uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

run-all:
	@echo "$(GREEN)Starting all services...$(NC)"
	@bash -c "trap 'kill %1; kill %2' SIGINT; \
		streamlit run master_generators_app.py --server.port=8501 & \
		uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload & \
		wait"

run-celery:
	@echo "$(GREEN)Starting Celery worker...$(NC)"
	celery -A src.tasks worker --loglevel=info

run-flower:
	@echo "$(GREEN)Starting Flower monitoring...$(NC)"
	celery -A src.tasks flower --port=5555

# Docker targets
docker-build:
	@echo "$(GREEN)Building Docker images...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) .
	$(DOCKER) tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest
	@echo "$(GREEN)✓ Docker images built$(NC)"

docker-up:
	@echo "$(GREEN)Starting Docker services...$(NC)"
	$(COMPOSE) up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "  - Streamlit: http://localhost:8501"
	@echo "  - API: http://localhost:8000"
	@echo "  - Flower: http://localhost:5555"
	@echo "  - Grafana: http://localhost:3000"

docker-down:
	@echo "$(YELLOW)Stopping Docker services...$(NC)"
	$(COMPOSE) down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-logs:
	$(COMPOSE) logs -f

docker-clean:
	@echo "$(YELLOW)Cleaning Docker resources...$(NC)"
	$(COMPOSE) down -v
	$(DOCKER) system prune -f
	@echo "$(GREEN)✓ Docker resources cleaned$(NC)"

# Deployment targets
deploy-dev:
	@echo "$(GREEN)Deploying to development...$(NC)"
	./scripts/deploy.sh deploy-docker

deploy-staging:
	@echo "$(GREEN)Deploying to staging...$(NC)"
	./scripts/deploy.sh deploy-staging

deploy-prod:
	@echo "$(YELLOW)Deploying to production...$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	./scripts/deploy.sh deploy-prod

deploy-k8s:
	@echo "$(GREEN)Deploying to Kubernetes...$(NC)"
	kubectl apply -f kubernetes/deployment.yaml
	@echo "$(GREEN)✓ Deployed to Kubernetes$(NC)"

# Documentation targets
docs:
	@echo "$(GREEN)Generating documentation...$(NC)"
	cd docs && $(MAKE) html
	@echo "$(GREEN)✓ Documentation generated in docs/_build/html/$(NC)"

docs-serve:
	@echo "$(GREEN)Serving documentation...$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8080

# Maintenance targets
clean:
	@echo "$(YELLOW)Cleaning temporary files...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -f .coverage
	rm -f logs/*.log
	@echo "$(GREEN)✓ Cleaned temporary files$(NC)"

clean-all: clean
	@echo "$(YELLOW)Cleaning all generated files...$(NC)"
	rm -rf models/*.pth
	rm -rf data/generated/*
	rm -rf data/training/*
	@echo "$(GREEN)✓ All generated files cleaned$(NC)"

backup:
	@echo "$(GREEN)Creating backup...$(NC)"
	@mkdir -p backups
	tar -czf backups/backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		--exclude=__pycache__ \
		--exclude=.pytest_cache \
		--exclude=htmlcov \
		--exclude=dist \
		--exclude=build \
		models/ data/ logs/ src/
	@echo "$(GREEN)✓ Backup created in backups/$(NC)"

version:
	@echo "$(GREEN)Master Generators ODE System$(NC)"
	@echo "Version: $(VERSION)"
	@echo "Python: $(shell python --version)"
	@echo "Git: $(shell git describe --tags --always 2>/dev/null || echo 'not a git repo')"

# Phony targets
.DEFAULT_GOAL := help
