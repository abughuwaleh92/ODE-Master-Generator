#!/bin/bash
# deploy_to_railway.sh - Complete deployment script for Railway

echo "🚀 Preparing Master Generators for Railway Deployment"
echo "===================================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models data logs static

# Clean up unnecessary files for deployment
echo "🧹 Cleaning up..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
rm -rf .pytest_cache 2>/dev/null || true
rm -rf htmlcov 2>/dev/null || true

# Create a lightweight requirements file for Railway
echo "📦 Creating optimized requirements..."
cat > requirements_railway.txt << 'EOF'
# Core Dependencies Only
streamlit==1.28.2
numpy==1.24.3
sympy==1.12
pandas==2.0.3
plotly==5.18.0
matplotlib==3.7.2
scipy==1.11.4

# Web Framework (optional - comment out if not using API)
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-multipart==0.0.6

# Minimal utilities
python-dotenv==1.0.0
python-json-logger==2.0.7
EOF

# Create Railway-specific environment file
echo "🔧 Creating Railway configuration..."
cat > .env.railway << 'EOF'
# Railway Environment
APP_ENV=production
DEBUG=False
DISABLE_API=false
ENABLE_ML_FEATURES=false
MAX_BATCH_SIZE=10
LOG_LEVEL=INFO
EOF

# Update railway.json for optimized deployment
cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install --no-cache-dir -r requirements_railway.txt"
  },
  "deploy": {
    "startCommand": "bash start.sh",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3,
    "healthcheckPath": "/",
    "healthcheckTimeout": 30
  },
  "variables": {
    "PYTHONUNBUFFERED": "1",
    "STREAMLIT_SERVER_HEADLESS": "true",
    "STREAMLIT_SERVER_PORT": "$PORT",
    "DISABLE_API": "false",
    "TF_CPP_MIN_LOG_LEVEL": "3"
  }
}
EOF

echo "✅ Preparation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run: railway login"
echo "2. Run: railway init"
echo "3. Run: railway up"
echo "4. Set environment variables in Railway dashboard if needed"
echo ""
echo "💡 Tips for Railway:"
echo "- If you encounter memory issues, set DISABLE_API=true"
echo "- Start with basic features, then enable ML/DL gradually"
echo "- Monitor logs with: railway logs"
