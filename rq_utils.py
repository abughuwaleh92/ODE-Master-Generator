# rq_utils.py
"""
Enhanced RQ utilities with robust Redis connection management,
proper error handling, and comprehensive job monitoring.
"""

import os
import json
import pickle
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from contextlib import contextmanager

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    from rq import Queue, Worker
    from rq.job import Job, JobStatus
    from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class RedisConnectionManager:
    """Manages Redis connections with retry logic and connection pooling"""
    
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._pool is None:
            self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize Redis connection pool"""
        redis_url = self._get_redis_url()
        if redis_url and REDIS_AVAILABLE:
            try:
                # Create connection pool with proper settings
                pool_kwargs = {
                    'max_connections': 50,
                    'socket_connect_timeout': 5,
                    'socket_timeout': 5,
                    'retry_on_timeout': True,
                    'decode_responses': False,  # Important for RQ compatibility
                    'health_check_interval': 30,
                }
                
                self._pool = redis.ConnectionPool.from_url(redis_url, **pool_kwargs)
                
                # Test connection
                conn = redis.Redis(connection_pool=self._pool)
                conn.ping()
                logger.info("Redis connection pool initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Redis pool: {e}")
                self._pool = None
    
    @staticmethod
    def _get_redis_url() -> Optional[str]:
        """Get Redis URL from environment variables"""
        return (
            os.getenv("REDIS_URL") or 
            os.getenv("REDIS_TLS_URL") or 
            os.getenv("UPSTASH_REDIS_URL") or
            os.getenv("REDISCLOUD_URL")
        )
    
    def get_connection(self, decode_responses: bool = False) -> Optional[redis.Redis]:
        """
        Get Redis connection from pool
        
        Args:
            decode_responses: Whether to decode responses (use False for RQ, True for logs)
        """
        if not self._pool:
            return None
            
        try:
            if decode_responses:
                # Create a separate connection for decoded responses
                redis_url = self._get_redis_url()
                return redis.from_url(redis_url, decode_responses=True)
            else:
                return redis.Redis(connection_pool=self._pool)
        except Exception as e:
            logger.error(f"Failed to get Redis connection: {e}")
            return None
    
    @contextmanager
    def get_connection_context(self, decode_responses: bool = False):
        """Context manager for Redis connections"""
        conn = self.get_connection(decode_responses)
        try:
            yield conn
        finally:
            if conn:
                conn.close()

# Singleton instance
_connection_manager = RedisConnectionManager()

def has_redis() -> bool:
    """Check if Redis is available and connected"""
    if not REDIS_AVAILABLE:
        return False
    
    conn = _connection_manager.get_connection()
    if not conn:
        return False
        
    try:
        conn.ping()
        return True
    except Exception:
        return False

def get_queue(name: Optional[str] = None, is_async: bool = True) -> Optional[Queue]:
    """
    Get RQ Queue instance
    
    Args:
        name: Queue name (defaults to RQ_QUEUE env var or 'ode_jobs')
        is_async: Whether queue should be async (False for testing)
    """
    conn = _connection_manager.get_connection()
    if not conn:
        return None
    
    queue_name = name or os.getenv("RQ_QUEUE", "ode_jobs")
    
    try:
        return Queue(
            name=queue_name,
            connection=conn,
            is_async=is_async,
            default_timeout=os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "3600s")
        )
    except Exception as e:
        logger.error(f"Failed to create queue '{queue_name}': {e}")
        return None

def enqueue_job(
    func_path: str,
    payload: dict,
    queue: Optional[str] = None,
    job_id: Optional[str] = None,
    timeout: Optional[Union[int, str]] = None,
    result_ttl: Optional[int] = None,
    ttl: Optional[int] = None,
    failure_ttl: Optional[int] = None,
    depends_on: Optional[Union[Job, str]] = None,
    job_kwargs: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Enqueue a job on the RQ queue with comprehensive options
    
    Args:
        func_path: Dotted path to the function (e.g., 'worker.compute_job')
        payload: Dictionary payload to pass to the function
        queue: Queue name (optional)
        job_id: Custom job ID (optional)
        timeout: Job timeout in seconds or string format (e.g., '1h')
        result_ttl: How long to keep successful results (seconds)
        ttl: Job expiration time (seconds)
        failure_ttl: How long to keep failed job info (seconds)
        depends_on: Job or job ID this depends on
        job_kwargs: Additional kwargs for job creation
        
    Returns:
        Job ID if successful, None otherwise
    """
    q = get_queue(queue)
    if not q:
        logger.error("Failed to get queue for job enqueue")
        return None
    
    try:
        # Prepare job arguments
        job_args = {
            'timeout': timeout or int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT", 3600)),
            'result_ttl': result_ttl or int(os.getenv("RQ_RESULT_TTL", 604800)),
            'ttl': ttl,
            'failure_ttl': failure_ttl or int(os.getenv("RQ_FAILURE_TTL", 86400)),
            'job_id': job_id,
            'depends_on': depends_on,
            'meta': {
                'enqueued_at': datetime.utcnow().isoformat(),
                'payload_keys': list(payload.keys())
            }
        }
        
        # Add any additional kwargs
        if job_kwargs:
            job_args.update(job_kwargs)
        
        # Enqueue the job
        job = q.enqueue(func_path, payload, **job_args)
        
        logger.info(f"Job enqueued: {job.id} -> {func_path}")
        return job.id
        
    except Exception as e:
        logger.error(f"Failed to enqueue job: {e}")
        return None

def fetch_job(job_id: str, include_meta: bool = True) -> Optional[Dict[str, Any]]:
    """
    Fetch comprehensive job information
    
    Args:
        job_id: Job ID to fetch
        include_meta: Whether to include metadata
        
    Returns:
        Dictionary with job information or None if not found
    """
    conn = _connection_manager.get_connection()
    if not conn:
        return None
    
    try:
        job = Job.fetch(job_id, connection=conn)
        
        info = {
            'id': job.id,
            'status': job.get_status(),
            'origin': job.origin,
            'func_name': job.func_name,
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'enqueued_at': job.enqueued_at.isoformat() if job.enqueued_at else None,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'ended_at': job.ended_at.isoformat() if job.ended_at else None,
            'timeout': job.timeout,
            'result_ttl': job.result_ttl,
            'worker_name': job.worker_name,
        }
        
        # Add metadata if requested
        if include_meta:
            info['meta'] = job.meta or {}
        
        # Add result for finished jobs
        if job.is_finished:
            try:
                info['result'] = job.result
            except Exception as e:
                info['result'] = f"Error retrieving result: {e}"
        
        # Add failure info for failed jobs
        if job.is_failed:
            info['exc_info'] = job.exc_info
            info['failed_at'] = job.ended_at.isoformat() if job.ended_at else None
        
        # Calculate runtime if applicable
        if job.started_at and job.ended_at:
            runtime = (job.ended_at - job.started_at).total_seconds()
            info['runtime_seconds'] = runtime
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to fetch job {job_id}: {e}")
        return None

def get_job_status(job_id: str) -> Optional[str]:
    """Get simple job status"""
    conn = _connection_manager.get_connection()
    if not conn:
        return None
    
    try:
        job = Job.fetch(job_id, connection=conn)
        return job.get_status()
    except Exception:
        return None

def cancel_job(job_id: str, remove: bool = False) -> bool:
    """
    Cancel a job
    
    Args:
        job_id: Job ID to cancel
        remove: Whether to remove the job completely
        
    Returns:
        True if successful, False otherwise
    """
    conn = _connection_manager.get_connection()
    if not conn:
        return False
    
    try:
        job = Job.fetch(job_id, connection=conn)
        
        if remove:
            job.delete()
        else:
            job.cancel()
        
        logger.info(f"Job {job_id} {'removed' if remove else 'cancelled'}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        return False

def get_progress(job_id: str) -> Dict[str, Any]:
    """Get job progress information"""
    conn = _connection_manager.get_connection()
    if not conn:
        return {}
    
    try:
        job = Job.fetch(job_id, connection=conn)
        meta = job.meta or {}
        
        # Extract progress info
        progress = meta.get('progress', {})
        
        # Add status info
        progress['status'] = job.get_status()
        
        # Add timing info if available
        if job.started_at:
            elapsed = (datetime.utcnow() - job.started_at).total_seconds()
            progress['elapsed_seconds'] = elapsed
        
        return progress
        
    except Exception as e:
        logger.debug(f"Failed to get progress for {job_id}: {e}")
        return {}

def get_artifacts(job_id: str) -> Dict[str, Any]:
    """Get job artifacts (files, models, etc.)"""
    conn = _connection_manager.get_connection()
    if not conn:
        return {}
    
    try:
        job = Job.fetch(job_id, connection=conn)
        meta = job.meta or {}
        return meta.get('artifacts', {})
        
    except Exception as e:
        logger.debug(f"Failed to get artifacts for {job_id}: {e}")
        return {}

def get_logs(
    job_id: str,
    start: int = 0,
    end: int = -1,
    max_lines: int = 5000
) -> List[str]:
    """
    Get job logs from Redis list
    
    Args:
        job_id: Job ID
        start: Start index (0-based)
        end: End index (-1 for all)
        max_lines: Maximum number of lines to return
        
    Returns:
        List of log lines
    """
    # Use decoded connection for logs
    conn = _connection_manager.get_connection(decode_responses=True)
    if not conn:
        return []
    
    log_key = f"job:{job_id}:logs"
    
    try:
        # Get total log count
        total = conn.llen(log_key)
        
        # Adjust end index
        if end == -1:
            end = min(start + max_lines - 1, total - 1)
        else:
            end = min(end, start + max_lines - 1)
        
        # Fetch logs
        logs = conn.lrange(log_key, start, end)
        
        return logs if logs else []
        
    except Exception as e:
        logger.debug(f"Failed to get logs for {job_id}: {e}")
        return []

def append_log(job_id: str, message: str, max_logs: int = 10000) -> bool:
    """
    Append a log message for a job
    
    Args:
        job_id: Job ID
        message: Log message to append
        max_logs: Maximum number of logs to keep
        
    Returns:
        True if successful
    """
    conn = _connection_manager.get_connection(decode_responses=True)
    if not conn:
        return False
    
    log_key = f"job:{job_id}:logs"
    
    try:
        # Format message with timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        formatted_message = f"[{timestamp}] {message}"
        
        # Append to list
        conn.rpush(log_key, formatted_message)
        
        # Trim to max size
        conn.ltrim(log_key, -max_logs, -1)
        
        # Set expiration (7 days)
        conn.expire(log_key, 604800)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to append log for {job_id}: {e}")
        return False

def update_progress(
    job_id: str,
    progress_data: Dict[str, Any],
    merge: bool = True
) -> bool:
    """
    Update job progress
    
    Args:
        job_id: Job ID
        progress_data: Progress data to set
        merge: Whether to merge with existing progress
        
    Returns:
        True if successful
    """
    conn = _connection_manager.get_connection()
    if not conn:
        return False
    
    try:
        job = Job.fetch(job_id, connection=conn)
        
        if merge:
            current_progress = job.meta.get('progress', {})
            current_progress.update(progress_data)
            job.meta['progress'] = current_progress
        else:
            job.meta['progress'] = progress_data
        
        job.meta['last_updated'] = datetime.utcnow().isoformat()
        job.save_meta()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update progress for {job_id}: {e}")
        return False

def update_artifacts(
    job_id: str,
    artifacts: Dict[str, Any],
    merge: bool = True
) -> bool:
    """
    Update job artifacts
    
    Args:
        job_id: Job ID
        artifacts: Artifacts data to set
        merge: Whether to merge with existing artifacts
        
    Returns:
        True if successful
    """
    conn = _connection_manager.get_connection()
    if not conn:
        return False
    
    try:
        job = Job.fetch(job_id, connection=conn)
        
        if merge:
            current_artifacts = job.meta.get('artifacts', {})
            current_artifacts.update(artifacts)
            job.meta['artifacts'] = current_artifacts
        else:
            job.meta['artifacts'] = artifacts
        
        job.save_meta()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update artifacts for {job_id}: {e}")
        return False

def get_queue_stats(queue_name: Optional[str] = None) -> Dict[str, Any]:
    """Get queue statistics"""
    q = get_queue(queue_name)
    if not q:
        return {}
    
    try:
        conn = _connection_manager.get_connection()
        
        # Get registries
        started = StartedJobRegistry(queue=q)
        finished = FinishedJobRegistry(queue=q)
        failed = FailedJobRegistry(queue=q)
        
        stats = {
            'name': q.name,
            'queued': q.count,
            'started': len(started),
            'finished': len(finished),
            'failed': len(failed),
            'workers': Worker.count(queue=q),
            'is_empty': q.is_empty(),
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return {}

def cleanup_old_jobs(
    queue_name: Optional[str] = None,
    older_than_days: int = 7,
    include_failed: bool = False
) -> int:
    """
    Clean up old finished/failed jobs
    
    Args:
        queue_name: Queue name
        older_than_days: Remove jobs older than this many days
        include_failed: Whether to also remove failed jobs
        
    Returns:
        Number of jobs removed
    """
    q = get_queue(queue_name)
    if not q:
        return 0
    
    try:
        conn = _connection_manager.get_connection()
        count = 0
        
        # Calculate cutoff time
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        
        # Clean finished jobs
        finished = FinishedJobRegistry(queue=q)
        for job_id in finished.get_job_ids():
            try:
                job = Job.fetch(job_id, connection=conn)
                if job.ended_at and job.ended_at < cutoff:
                    job.delete()
                    count += 1
            except Exception:
                pass
        
        # Clean failed jobs if requested
        if include_failed:
            failed = FailedJobRegistry(queue=q)
            for job_id in failed.get_job_ids():
                try:
                    job = Job.fetch(job_id, connection=conn)
                    if job.ended_at and job.ended_at < cutoff:
                        job.delete()
                        count += 1
                except Exception:
                    pass
        
        logger.info(f"Cleaned up {count} old jobs from queue '{q.name}'")
        return count
        
    except Exception as e:
        logger.error(f"Failed to cleanup old jobs: {e}")
        return 0

# Convenience function for checking Redis health
def check_redis_health() -> Dict[str, Any]:
    """Check Redis connection health and return diagnostics"""
    health = {
        'connected': False,
        'redis_url_configured': bool(_connection_manager._get_redis_url()),
        'redis_library_available': REDIS_AVAILABLE,
        'error': None,
        'info': {}
    }
    
    if not REDIS_AVAILABLE:
        health['error'] = "Redis library not installed"
        return health
    
    conn = _connection_manager.get_connection()
    if not conn:
        health['error'] = "Failed to create connection"
        return health
    
    try:
        # Ping test
        conn.ping()
        health['connected'] = True
        
        # Get server info
        info = conn.info()
        health['info'] = {
            'version': info.get('redis_version'),
            'used_memory_human': info.get('used_memory_human'),
            'connected_clients': info.get('connected_clients'),
            'uptime_in_days': info.get('uptime_in_days'),
        }
        
    except Exception as e:
        health['error'] = str(e)
    
    return health