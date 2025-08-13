# tests/conftest.py - Pytest Configuration
# ============================================================================
"""
Pytest configuration and fixtures
"""

import pytest
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture(scope="session")
def test_data_dir(temp_dir):
    """Create test data directory"""
    data_dir = os.path.join(temp_dir, "test_data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

@pytest.fixture(scope="session")
def test_models_dir(temp_dir):
    """Create test models directory"""
    models_dir = os.path.join(temp_dir, "test_models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # Clear cached functions
    from src.functions import basic_functions, special_functions
    basic_functions.functions.clear()
    special_functions.functions.clear()
    
    # Reinitialize
    basic_functions.__init__()
    special_functions.__init__()

@pytest.fixture
def mock_redis(monkeypatch):
    """Mock Redis for tests"""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        def get(self, key):
            return self.data.get(key)
        
        def set(self, key, value, ex=None):
            self.data[key] = value
        
        def delete(self, key):
            if key in self.data:
                del self.data[key]
        
        def ping(self):
            return True
    
    mock = MockRedis()
    monkeypatch.setattr("redis.from_url", lambda url: mock)
    return mock

# Pytest markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )

# Test coverage configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    if config.getoption("--no-slow"):
        skip_slow = pytest.mark.skip(reason="--no-slow option provided")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

def pytest_addoption(parser):
    """Add custom options"""
    parser.addoption(
        "--no-slow",
        action="store_true",
        default=False,
        help="Skip slow tests"
    )
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="Run benchmark tests"
    )
