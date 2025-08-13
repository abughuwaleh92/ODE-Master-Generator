"""
scripts/launch.py - Complete launch script
"""

import subprocess
import sys
import os
import time
import threading
import signal
import webbrowser

processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nShutting down services...")
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=5)
        except:
            p.kill()
    sys.exit(0)

def run_streamlit():
    """Run Streamlit app"""
    print("Starting Streamlit app...")
    port = os.environ.get('STREAMLIT_SERVER_PORT', '8501')
    cmd = [sys.executable, '-m', 'streamlit', 'run', 'master_generators_app.py', 
           '--server.port', port, '--server.address', '0.0.0.0']
    p = subprocess.Popen(cmd)
    processes.append(p)
    return p

def run_api():
    """Run FastAPI server"""
    print("Starting API server...")
    port = os.environ.get('API_PORT', '8000')
    cmd = [sys.executable, 'api_server.py']
    p = subprocess.Popen(cmd)
    processes.append(p)
    return p

def wait_for_services():
    """Wait for services to start"""
    import requests
    
    print("\nWaiting for services to start...")
    
    # Wait for API
    api_url = "http://localhost:8000/health"
    for i in range(30):
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                print("✅ API server is ready")
                break
        except:
            time.sleep(1)
    
    # Wait for Streamlit
    streamlit_url = "http://localhost:8501"
    for i in range(30):
        try:
            response = requests.get(streamlit_url)
            if response.status_code == 200:
                print("✅ Streamlit app is ready")
                break
        except:
            time.sleep(1)

def main():
    """Main launcher function"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 50)
    print("Master Generators - Service Launcher")
    print("=" * 50)
    
    # Run startup check first
    print("\nRunning startup check...")
    result = subprocess.run([sys.executable, 'startup_check.py'])
    if result.returncode != 0:
        print("Startup check failed. Please fix the issues and try again.")
        return 1
    
    print("\nStarting services...")
    
    # Start services in threads
    streamlit_thread = threading.Thread(target=run_streamlit)
    api_thread = threading.Thread(target=run_api)
    
    streamlit_thread.start()
    time.sleep(2)
    api_thread.start()
    
    # Wait for services to be ready
    wait_for_services()
    
    print("\n" + "=" * 50)
    print("✅ All services are running!")
    print("=" * 50)
    print("\n📊 Streamlit App: http://localhost:8501")
    print("🔌 API Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all services")
    print("=" * 50)
    
    # Open browser
    time.sleep(2)
    webbrowser.open('http://localhost:8501')
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    sys.exit(main())

# ==============================================================================
# src/api/__init__.py
# ==============================================================================
"""API module initialization"""

from .routes import router

__all__ = ['router']

# ==============================================================================
# src/ui/__init__.py
# ==============================================================================
"""UI module initialization"""

from .components import UIComponents

__all__ = ['UIComponents']

# ==============================================================================
# src/dl/__init__.py
# ==============================================================================
"""Deep Learning module initialization"""

from .novelty_detector import ODENoveltyDetector, NoveltyAnalysis

__all__ = ['ODENoveltyDetector', 'NoveltyAnalysis']

# ==============================================================================
# src/ml/__init__.py
# ==============================================================================
"""Machine Learning module initialization"""

from .pattern_learner import GeneratorPatternLearner, GeneratorVAE, GeneratorTransformer
from .trainer import MLTrainer

__all__ = [
    'GeneratorPatternLearner',
    'GeneratorVAE', 
    'GeneratorTransformer',
    'MLTrainer'
]

# ==============================================================================
# src/generators/__init__.py
# ==============================================================================
"""Generators module initialization"""

from .master_generator import MasterGenerator
from .linear_generators import LinearGeneratorFactory
from .nonlinear_generators import NonlinearGeneratorFactory

__all__ = [
    'MasterGenerator',
    'LinearGeneratorFactory',
    'NonlinearGeneratorFactory'
]

# ==============================================================================
# src/functions/__init__.py
# ==============================================================================
"""Functions module initialization"""

from .basic_functions import BasicFunctions
from .special_functions import SpecialFunctions

__all__ = ['BasicFunctions', 'SpecialFunctions']

# ==============================================================================
# models/.gitkeep
# ==============================================================================
# This file ensures the models directory is tracked by git

# ==============================================================================
# data/.gitkeep
# ==============================================================================
# This file ensures the data directory is tracked by git

# ==============================================================================
# logs/.gitkeep
# ==============================================================================
# This file ensures the logs directory is tracked by git

# ==============================================================================
# config.json - Default configuration
# ==============================================================================
{
  "app_name": "Master Generators ODE System",
  "version": "1.0.0",
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": true,
    "workers": 1
  },
  "streamlit": {
    "port": 8501,
    "server.maxUploadSize": 200,
    "server.enableCORS": true,
    "server.enableXsrfProtection": true
  },
  "ml": {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "device": "auto",
    "model_save_path": "models",
    "checkpoint_interval": 10
  },
  "dl": {
    "transformer_layers": 6,
    "attention_heads": 8,
    "hidden_dim": 256,
    "dropout": 0.1
  },
  "paths": {
    "models": "models",
    "data": "data",
    "logs": "logs",
    "static": "static",
    "temp": "temp"
  },
  "features": {
    "enable_ml": true,
    "enable_novelty_detection": true,
    "enable_special_functions": true,
    "enable_batch_generation": true,
    "max_batch_size": 1000,
    "enable_export": true,
    "enable_visualization": true
  },
  "limits": {
    "max_alpha": 100,
    "max_beta": 100,
    "max_n": 10,
    "max_order": 10,
    "max_power": 10,
    "max_batch": 1000,
    "max_epochs": 1000
  },
  "logging": {
    "level": "INFO",
    "file": "logs/app.log",
    "max_size": "10MB",
    "backup_count": 5
  },
  "security": {
    "enable_auth": false,
    "api_key": null,
    "allowed_origins": ["*"],
    "rate_limit": 100
  }
}

# ==============================================================================
# docker-compose.yml - Complete Docker Compose configuration
# ==============================================================================
version: '3.8'

services:
  app:
    build: .
    container_name: master-generators-app
    ports:
      - "8501:8501"
      - "8000:8000"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - API_PORT=8000
      - PYTHONUNBUFFERED=1
      - TF_CPP_MIN_LOG_LEVEL=2
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    networks:
      - master-generators-network

  nginx:
    image: nginx:alpine
    container_name: master-generators-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./static:/usr/share/nginx/html
    depends_on:
      - app
    restart: unless-stopped
    networks:
      - master-generators-network

  redis:
    image: redis:alpine
    container_name: master-generators-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - master-generators-network

networks:
  master-generators-network:
    driver: bridge

volumes:
  redis-data:
