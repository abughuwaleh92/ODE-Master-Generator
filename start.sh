#!/bin/bash
# start.sh - Railway deployment startup script with error handling

set -e

echo "Starting Master Generators Application..."
echo "========================================="

# Create necessary directories
mkdir -p models data logs static/css static/js static/images

# Run startup check (non-blocking)
if python startup_check.py; then
    echo "✅ Startup check passed"
else
    echo "⚠️ Startup check failed, continuing anyway..."
fi

# Export environment variables
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2  # Suppress TensorFlow warnings

# Check if we should disable API
if [ "${DISABLE_API}" = "true" ]; then
    echo "ℹ️ API server disabled by environment variable"
    
    # Start only Streamlit
    echo "Starting Streamlit on port ${PORT:-8501}..."
    exec streamlit run master_generators_app.py \
        --server.port=${PORT:-8501} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false
else
    # Start both services
    echo "Starting Streamlit on port ${PORT:-8501}..."
    streamlit run master_generators_app.py \
        --server.port=${PORT:-8501} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false &
    STREAMLIT_PID=$!
    
    echo "Starting API server on port ${API_PORT:-8000}..."
    python api_server.py &
    API_PID=$!
    
    # Function to handle shutdown
    cleanup() {
        echo "Shutting down services..."
        kill $STREAMLIT_PID $API_PID 2>/dev/null || true
        exit 0
    }
    
    # Set up signal handlers
    trap cleanup SIGTERM SIGINT
    
    # Wait for any process to exit
    wait -n $STREAMLIT_PID $API_PID
    
    # If one process exits, kill the other
    cleanup
fi
