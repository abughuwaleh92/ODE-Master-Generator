"""
Configuration utilities with proper Pydantic v2 support and security improvements
"""

import os
import secrets
from typing import Optional, List
import warnings

# Try importing pydantic_settings, fall back if not available
try:
    from pydantic_settings import BaseSettings
    from pydantic import field_validator, Field
    
    class Settings(BaseSettings):
        """Application settings with validation and security"""
        
        app_name: str = Field(default="Master Generators", min_length=1, max_length=100)
        api_port: int = Field(default=8000, ge=1, le=65535)
        streamlit_port: int = Field(default=8501, ge=1, le=65535)
        debug: bool = Field(default=False)
        
        # ML Settings
        ml_batch_size: int = Field(default=32, ge=1, le=1024)
        ml_epochs: int = Field(default=100, ge=1, le=10000)
        ml_learning_rate: float = Field(default=0.001, gt=0, le=1)
        
        # Database
        database_url: Optional[str] = Field(default=None)
        redis_url: Optional[str] = Field(default=None)
        
        # Security
        secret_key: str = Field(default=None)
        api_key: Optional[str] = Field(default=None)
        jwt_algorithm: str = Field(default="HS256")
        access_token_expire_minutes: int = Field(default=30, ge=1, le=10080)
        
        # CORS
        allowed_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:8501"])
        
        # Rate Limiting
        rate_limit_enabled: bool = Field(default=True)
        rate_limit_requests: int = Field(default=100, ge=1, le=10000)
        rate_limit_window: int = Field(default=60, ge=1, le=3600)
        
        # Cache
        cache_ttl: int = Field(default=3600, ge=0, le=86400)
        max_cache_size: int = Field(default=1000, ge=1, le=100000)
        
        # Limits
        max_batch_size: int = Field(default=1000, ge=1, le=10000)
        max_alpha: float = Field(default=100.0, gt=0, le=1000)
        max_beta: float = Field(default=100.0, gt=0, le=1000)
        max_n: int = Field(default=10, ge=1, le=100)
        max_complexity: int = Field(default=1000, ge=1, le=10000)
        
        @field_validator('api_port', 'streamlit_port')
        @classmethod
        def validate_port(cls, v):
            if not 1 <= v <= 65535:
                raise ValueError('Port must be between 1 and 65535')
            return v
        
        @field_validator('secret_key')
        @classmethod
        def validate_secret_key(cls, v):
            """Validate and generate secret key if needed"""
            if v is None or v == "your-secret-key-change-in-production":
                if os.getenv("APP_ENV") == "production":
                    raise ValueError(
                        "SECRET_KEY must be set in production! "
                        "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                    )
                else:
                    # Generate a random secret key for development
                    v = secrets.token_urlsafe(32)
                    warnings.warn(
                        f"Generated development SECRET_KEY. "
                        f"Set SECRET_KEY environment variable in production!",
                        UserWarning
                    )
            return v
        
        @field_validator('allowed_origins')
        @classmethod
        def validate_allowed_origins(cls, v):
            """Validate CORS origins"""
            if "*" in v and os.getenv("APP_ENV") == "production":
                warnings.warn(
                    "Using wildcard (*) in ALLOWED_ORIGINS is not recommended in production!",
                    UserWarning
                )
            return v
        
        class Config:
            env_file = '.env'
            env_file_encoding = 'utf-8'
            case_sensitive = False
            
            # Allow environment variables to override settings
            # Format: APP_NAME -> app_name
            env_prefix = ''
    
except ImportError:
    # Fallback for when pydantic-settings is not available
    class Settings:
        """Fallback settings class without Pydantic"""
        
        def __init__(self):
            self.app_name = os.getenv("APP_NAME", "Master Generators")
            self.api_port = int(os.getenv("API_PORT", "8000"))
            self.streamlit_port = int(os.getenv("STREAMLIT_PORT", "8501"))
            self.debug = os.getenv("DEBUG", "False").lower() == "true"
            
            # ML Settings
            self.ml_batch_size = int(os.getenv("ML_BATCH_SIZE", "32"))
            self.ml_epochs = int(os.getenv("ML_EPOCHS", "100"))
            self.ml_learning_rate = float(os.getenv("ML_LEARNING_RATE", "0.001"))
            
            # Database
            self.database_url = os.getenv("DATABASE_URL")
            self.redis_url = os.getenv("REDIS_URL")
            
            # Security - Generate secret key if not provided
            self.secret_key = os.getenv("SECRET_KEY")
            if not self.secret_key or self.secret_key == "your-secret-key-change-in-production":
                if os.getenv("APP_ENV") == "production":
                    raise ValueError("SECRET_KEY must be set in production!")
                else:
                    self.secret_key = secrets.token_urlsafe(32)
                    warnings.warn("Generated development SECRET_KEY", UserWarning)
            
            self.api_key = os.getenv("API_KEY")
            self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
            self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
            
            # CORS
            origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501")
            self.allowed_origins = origins.split(",")
            
            # Rate Limiting
            self.rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
            self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
            self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
            
            # Cache
            self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
            self.max_cache_size = int(os.getenv("MAX_CACHE_SIZE", "1000"))
            
            # Limits
            self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "1000"))
            self.max_alpha = float(os.getenv("MAX_ALPHA", "100"))
            self.max_beta = float(os.getenv("MAX_BETA", "100"))
            self.max_n = int(os.getenv("MAX_N", "10"))
            self.max_complexity = int(os.getenv("MAX_COMPLEXITY", "1000"))

# Application configuration class with enhanced security
class AppConfig:
    """Application configuration with enhanced security settings"""
    
    def __init__(self):
        # Load settings
        self.settings = Settings()
        
        # Environment
        self.ENV = os.getenv("APP_ENV", "development")
        self.IS_PRODUCTION = self.ENV == "production"
        self.IS_DEVELOPMENT = self.ENV == "development"
        self.IS_TESTING = self.ENV == "testing"
        
        # API Settings
        self.APP_NAME = self.settings.app_name
        self.VERSION = "2.0.0"
        self.DEBUG = self.settings.debug
        
        # Validate debug mode in production
        if self.IS_PRODUCTION and self.DEBUG:
            warnings.warn("DEBUG mode should be False in production!", UserWarning)
        
        # Security
        self.SECRET_KEY = self.settings.secret_key
        self.API_KEY = self.settings.api_key
        self.JWT_ALGORITHM = self.settings.jwt_algorithm
        self.ACCESS_TOKEN_EXPIRE_MINUTES = self.settings.access_token_expire_minutes
        
        # CORS
        self.ALLOWED_ORIGINS = self.settings.allowed_origins
        
        # Rate Limiting
        self.RATE_LIMIT_ENABLED = self.settings.rate_limit_enabled
        self.RATE_LIMIT_REQUESTS = self.settings.rate_limit_requests
        self.RATE_LIMIT_WINDOW = self.settings.rate_limit_window
        
        # Database
        self.DATABASE_URL = self.settings.database_url
        self.REDIS_URL = self.settings.redis_url
        
        # Cache
        self.CACHE_TTL = self.settings.cache_ttl
        self.MAX_CACHE_SIZE = self.settings.max_cache_size
        
        # Limits
        self.MAX_BATCH_SIZE = self.settings.max_batch_size
        self.MAX_ALPHA = self.settings.max_alpha
        self.MAX_BETA = self.settings.max_beta
        self.MAX_N = self.settings.max_n
        self.MAX_COMPLEXITY = self.settings.max_complexity
        
        # ML Configuration
        self.ML_BATCH_SIZE = self.settings.ml_batch_size
        self.ML_EPOCHS = self.settings.ml_epochs
        self.ML_LEARNING_RATE = self.settings.ml_learning_rate
    
    def validate_production_config(self):
        """Validate configuration for production deployment"""
        errors = []
        
        if self.IS_PRODUCTION:
            # Check required settings
            if not self.SECRET_KEY or len(self.SECRET_KEY) < 32:
                errors.append("SECRET_KEY must be at least 32 characters in production")
            
            if self.DEBUG:
                errors.append("DEBUG must be False in production")
            
            if "*" in self.ALLOWED_ORIGINS:
                errors.append("ALLOWED_ORIGINS should not use wildcard (*) in production")
            
            if not self.DATABASE_URL:
                warnings.warn("DATABASE_URL not set - using in-memory storage", UserWarning)
            
            if not self.REDIS_URL:
                warnings.warn("REDIS_URL not set - using in-memory cache", UserWarning)
            
            if not self.RATE_LIMIT_ENABLED:
                warnings.warn("Rate limiting is disabled in production", UserWarning)
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
        
        return True
    
    def get_database_settings(self):
        """Get database configuration"""
        if self.DATABASE_URL:
            # Parse database URL for SQLAlchemy
            return {
                "url": self.DATABASE_URL,
                "pool_size": 20,
                "max_overflow": 40,
                "pool_pre_ping": True,
                "pool_recycle": 3600,
            }
        else:
            # Use SQLite for development
            return {
                "url": "sqlite:///./master_generators.db",
                "connect_args": {"check_same_thread": False},
            }
    
    def get_redis_settings(self):
        """Get Redis configuration"""
        if self.REDIS_URL:
            return {
                "url": self.REDIS_URL,
                "decode_responses": True,
                "max_connections": 50,
            }
        else:
            return None

# Create singleton instances
try:
    settings = Settings()
    app_config = AppConfig()
    
    # Validate production config if in production
    if app_config.IS_PRODUCTION:
        app_config.validate_production_config()
        
except Exception as e:
    print(f"Configuration error: {e}")
    # Provide minimal fallback configuration
    class MinimalConfig:
        APP_NAME = "Master Generators"
        DEBUG = False
        SECRET_KEY = secrets.token_urlsafe(32)
        IS_PRODUCTION = False
        IS_DEVELOPMENT = True
    
    app_config = MinimalConfig()
    warnings.warn("Using minimal fallback configuration due to error", UserWarning)

# Export main config object
__all__ = ['settings', 'app_config', 'Settings', 'AppConfig']
