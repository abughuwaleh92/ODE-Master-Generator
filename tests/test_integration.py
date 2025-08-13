import pytest
from fastapi.testclient import TestClient
from api_server import app

client = TestClient(app)

def test_generate_single_ode():
    response = client.post("/api/generate/single", json={
        "type": "linear",
        "generator_number": 1,
        "function": "sine",
        "parameters": {
            "alpha": 1.0,
            "beta": 1.0,
            "n": 1,
            "M": 0.0
        }
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "ode" in data
