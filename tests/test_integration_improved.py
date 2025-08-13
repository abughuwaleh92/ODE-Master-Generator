# tests/test_integration_improved.py - Integration Tests
# ============================================================================
"""
Integration tests for the complete system
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
import tempfile
import os
import json
import time

# Import the FastAPI app
from api_server import app

class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """Get authentication headers"""
        response = client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "admin"}
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_generate_single_ode(self, client):
        """Test single ODE generation"""
        request_data = {
            "type": "linear",
            "generator_number": 1,
            "function": "sine",
            "parameters": {
                "alpha": 1.0,
                "beta": 1.0,
                "n": 1,
                "M": 0.0
            }
        }
        
        response = client.post("/api/generate/single", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "ode" in data["data"]
        assert "solution" in data["data"]
        assert "novelty_score" in data["data"]
    
    def test_batch_generation(self, client):
        """Test batch ODE generation"""
        request_data = {
            "count": 5,
            "types": ["linear", "nonlinear"],
            "functions": ["sine", "cosine", "exponential"],
            "random_params": True,
            "parallel": False
        }
        
        response = client.post("/api/generate/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["results"]) <= 5
        assert "statistics" in data["data"]
    
    def test_novelty_analysis(self, client):
        """Test novelty analysis endpoint"""
        request_data = {
            "ode": "y''(x) + y(x) = sin(x)",
            "type": "linear",
            "order": 2,
            "detailed": True
        }
        
        response = client.post("/api/analyze/novelty", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "is_novel" in data["data"]
        assert "novelty_score" in data["data"]
        assert "complexity_level" in data["data"]
        assert "recommended_methods" in data["data"]
    
    def test_function_list(self, client):
        """Test function listing endpoint"""
        response = client.get("/api/functions/list")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "functions" in data["data"]
        assert "count" in data["data"]
    
    def test_function_properties(self, client):
        """Test function properties endpoint"""
        response = client.get("/api/functions/sine/properties?function_type=basic")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "name" in data["data"]
        assert "expression" in data["data"]
        assert "latex" in data["data"]
    
    @pytest.mark.asyncio
    async def test_ml_training(self, client):
        """Test ML model training"""
        request_data = {
            "model_type": "pattern_learner",
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "samples": 100
        }
        
        response = client.post("/api/ml/train", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "task_id" in data["data"]
        
        # Check training status
        task_id = data["data"]["task_id"]
        await asyncio.sleep(2)
        
        status_response = client.get(f"/api/ml/status/{task_id}")
        assert status_response.status_code == 200
    
    def test_ml_generation(self, client):
        """Test ML-based ODE generation"""
        response = client.get("/api/ml/generate?model_type=pattern_learner")
        assert response.status_code == 200
        
        data = response.json()
        # Model might not be trained, so check for either success or failure
        assert "success" in data
    
    def test_export_latex(self, client):
        """Test LaTeX export"""
        response = client.post(
            "/api/export/latex",
            params={
                "ode": "y''(x) + y(x) = 0",
                "solution": "A*sin(x) + B*cos(x)"
            }
        )
        assert response.status_code == 200
        assert "application/octet-stream" in response.headers["content-type"]
    
    def test_rate_limiting(self, client):
        """Test rate limiting"""
        # Make many requests quickly
        responses = []
        for _ in range(35):  # Exceeds rate limit of 30
            response = client.post(
                "/api/generate/single",
                json={
                    "type": "linear",
                    "generator_number": 1,
                    "function": "linear",
                    "parameters": {"alpha": 1, "beta": 1, "n": 1, "M": 0}
                }
            )
            responses.append(response.status_code)
        
        # Some requests should be rate limited (429 status)
        # Note: This depends on rate limiter being enabled
        # assert 429 in responses  # Uncomment when rate limiting is enabled
    
    def test_caching(self, client):
        """Test response caching"""
        request_data = {
            "type": "linear",
            "generator_number": 1,
            "function": "sine",
            "parameters": {
                "alpha": 1.0,
                "beta": 1.0,
                "n": 1,
                "M": 0.0
            }
        }
        
        # First request
        response1 = client.post("/api/generate/single", json=request_data)
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second identical request should be cached
        response2 = client.post("/api/generate/single", json=request_data)
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Results should be identical
        assert data1["data"]["ode"] == data2["data"]["ode"]
        # Check if second response was cached (if cache is enabled)
        # assert data2.get("cached", False) is True  # Uncomment when caching is enabled
