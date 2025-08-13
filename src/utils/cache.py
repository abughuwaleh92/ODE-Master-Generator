# src/utils/cache.py - Enhanced caching utilities
# ============================================================================
"""
Enhanced caching utilities with Redis and in-memory fallback
"""

import hashlib
import json
import time
from typing import Any, Optional, Dict
from functools import wraps
import redis
import pickle
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Enhanced cache manager with Redis and in-memory fallback"""
    
    def __init__(self, redis_url: Optional[str] = None, max_memory_size: int = 1000):
        self.redis_url = redis_url
        self.max_memory_size = max_memory_size
        self.memory_cache = {}
        self.access_times = {}
        self.redis_client = None
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
                self.redis_client = None
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = f"{prefix}:{args}:{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Try Redis first
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
            except Exception as e:
                logger.debug(f"Redis get error: {e}")
        
        # Fallback to memory cache
        if key in self.memory_cache:
            self.access_times[key] = time.time()
            return self.memory_cache[key]
        
        return None
    
    def set(self, key: str, value: Any, expire: int = 3600):
        """Set value in cache"""
        # Save to Redis
        if self.redis_client:
            try:
                self.redis_client.set(key, pickle.dumps(value), ex=expire)
            except Exception as e:
                logger.debug(f"Redis set error: {e}")
        
        # Save to memory cache
        if len(self.memory_cache) >= self.max_memory_size:
            # Remove least recently used
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.memory_cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.memory_cache[key] = value
        self.access_times[key] = time.time()
    
    def delete(self, key: str):
        """Delete value from cache"""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except:
                pass
        
        if key in self.memory_cache:
            del self.memory_cache[key]
            del self.access_times[key]
    
    def clear(self):
        """Clear all cache"""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except:
                pass
        
        self.memory_cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'memory_size': len(self.memory_cache),
            'max_memory_size': self.max_memory_size,
            'redis_connected': self.redis_client is not None,
            'oldest_access': min(self.access_times.values()) if self.access_times else None,
            'newest_access': max(self.access_times.values()) if self.access_times else None
        }

def cached(expire: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager from global or create new
            cache = getattr(wrapper, '_cache', None)
            if not cache:
                cache = CacheManager()
                wrapper._cache = cache
            
            # Generate cache key
            cache_key = cache._generate_key(key_prefix or func.__name__, *args, **kwargs)
            
            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, expire)
            
            return result
        
        return wrapper
    return decorator
