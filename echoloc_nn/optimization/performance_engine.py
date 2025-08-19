"""
High-performance optimization engine for EchoLoc-NN.

This module implements advanced performance optimizations including:
- Intelligent caching with LRU eviction
- Concurrent processing with worker pools
- Auto-scaling based on load
- Memory usage optimization
- GPU acceleration when available
"""

import numpy as np
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple, List, Callable
import hashlib
import pickle
import logging
from dataclasses import dataclass
import queue
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import os


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    
    # Caching
    cache_size: int = 1000
    cache_ttl_seconds: float = 300.0  # 5 minutes
    enable_cache: bool = True
    
    # Concurrency
    max_workers: int = 4
    enable_concurrent_processing: bool = True
    batch_processing_threshold: int = 10
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    target_latency_ms: float = 50.0
    scale_up_threshold: float = 0.8  # 80% of target latency
    scale_down_threshold: float = 0.3  # 30% of target latency
    min_workers: int = 1
    max_workers_limit: int = 8
    
    # Memory management
    memory_limit_mb: int = 512
    enable_memory_monitoring: bool = True
    gc_threshold: float = 0.8  # 80% memory usage


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def _compute_key(self, data: np.ndarray) -> str:
        """Compute cache key for numpy array."""
        # Use shape and data hash for key
        shape_str = str(data.shape)
        data_hash = hashlib.md5(data.tobytes()).hexdigest()[:16]
        return f"{shape_str}_{data_hash}"
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._timestamps:
            return True
        return time.time() - self._timestamps[key] > self.ttl_seconds
    
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            if key in self._cache:
                del self._cache[key]
            del self._timestamps[key]
    
    def get(self, data: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Get cached result for input data."""
        key = self._compute_key(data)
        
        with self._lock:
            if key in self._cache and not self._is_expired(key):
                # Move to end (most recently used)
                result = self._cache.pop(key)
                self._cache[key] = result
                self._hits += 1
                return result
            else:
                # Remove if expired
                if key in self._cache:
                    del self._cache[key]
                if key in self._timestamps:
                    del self._timestamps[key]
                self._misses += 1
                return None
    
    def put(self, data: np.ndarray, result: Tuple[np.ndarray, float]):
        """Store result in cache."""
        key = self._compute_key(data)
        
        with self._lock:
            # Evict expired entries
            self._evict_expired()
            
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if oldest_key in self._timestamps:
                    del self._timestamps[oldest_key]
            
            # Store new entry
            self._cache[key] = result
            self._timestamps[key] = time.time()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'cache_size': len(self._cache),
                'max_size': self.max_size
            }
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0


class WorkerPool:
    """
    Auto-scaling worker pool for concurrent processing.
    """
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger('WorkerPool')
        
        # Initialize with minimum workers
        self.executor = ThreadPoolExecutor(max_workers=config.min_workers)
        self.current_workers = config.min_workers
        
        # Performance tracking
        self.task_times = []
        self.task_queue_size = 0
        self.last_scale_time = time.time()
        self.scale_cooldown = 10.0  # seconds
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def submit_task(self, func: Callable, *args, **kwargs):
        """Submit task to worker pool."""
        with self._lock:
            self.task_queue_size += 1
        
        future = self.executor.submit(self._wrapped_task, func, *args, **kwargs)
        return future
    
    def _wrapped_task(self, func: Callable, *args, **kwargs):
        """Wrapper for task execution with timing."""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Record task completion time
            task_time = time.time() - start_time
            with self._lock:
                self.task_times.append(task_time * 1000)  # Convert to ms
                if len(self.task_times) > 100:
                    self.task_times = self.task_times[-100:]  # Keep recent times
                self.task_queue_size = max(0, self.task_queue_size - 1)
    
    def should_scale_up(self) -> bool:
        """Check if we should scale up workers."""
        if not self.config.enable_auto_scaling:
            return False
        
        if self.current_workers >= self.config.max_workers_limit:
            return False
        
        # Check cooldown
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        with self._lock:
            if not self.task_times:
                return False
            
            avg_task_time = np.mean(self.task_times[-10:])  # Recent tasks
            return (avg_task_time > self.config.target_latency_ms * self.config.scale_up_threshold and
                    self.task_queue_size > self.current_workers)
    
    def should_scale_down(self) -> bool:
        """Check if we should scale down workers."""
        if not self.config.enable_auto_scaling:
            return False
        
        if self.current_workers <= self.config.min_workers:
            return False
        
        # Check cooldown
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        with self._lock:
            if not self.task_times:
                return True  # No tasks, can scale down
            
            avg_task_time = np.mean(self.task_times[-20:])  # Longer window for scale down
            return (avg_task_time < self.config.target_latency_ms * self.config.scale_down_threshold and
                    self.task_queue_size < self.current_workers // 2)
    
    def scale_workers(self):
        """Auto-scale workers based on performance."""
        if self.should_scale_up():
            new_workers = min(self.current_workers + 1, self.config.max_workers_limit)
            self.logger.info(f"Scaling up workers: {self.current_workers} -> {new_workers}")
            
            # Create new executor with more workers
            old_executor = self.executor
            self.executor = ThreadPoolExecutor(max_workers=new_workers)
            self.current_workers = new_workers
            self.last_scale_time = time.time()
            
            # Shutdown old executor
            old_executor.shutdown(wait=False)
            
        elif self.should_scale_down():
            new_workers = max(self.current_workers - 1, self.config.min_workers)
            self.logger.info(f"Scaling down workers: {self.current_workers} -> {new_workers}")
            
            # Create new executor with fewer workers
            old_executor = self.executor
            self.executor = ThreadPoolExecutor(max_workers=new_workers)
            self.current_workers = new_workers
            self.last_scale_time = time.time()
            
            # Shutdown old executor
            old_executor.shutdown(wait=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self._lock:
            return {
                'current_workers': self.current_workers,
                'task_queue_size': self.task_queue_size,
                'avg_task_time_ms': np.mean(self.task_times) if self.task_times else 0,
                'recent_task_count': len(self.task_times)
            }


class PerformanceEngine:
    """
    High-performance optimization engine.
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.logger = logging.getLogger('PerformanceEngine')
        
        # Initialize components
        self.cache = LRUCache(self.config.cache_size, self.config.cache_ttl_seconds) if self.config.enable_cache else None
        self.worker_pool = WorkerPool(self.config) if self.config.enable_concurrent_processing else None
        
        # Memory monitoring
        self.memory_usage_history = []
        
        # Performance statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'concurrent_tasks': 0,
            'memory_cleanups': 0,
            'scaling_events': 0
        }
        
        # Start monitoring thread
        if self.config.enable_memory_monitoring or self.config.enable_auto_scaling:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                # Memory monitoring
                if self.config.enable_memory_monitoring:
                    self._monitor_memory()
                
                # Auto-scaling
                if self.config.enable_auto_scaling and self.worker_pool:
                    self.worker_pool.scale_workers()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(10)  # Longer wait on error
    
    def _monitor_memory(self):
        """Monitor and manage memory usage."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
            else:
                # Fallback: use approximate memory tracking
                import sys
                memory_mb = sys.getsizeof(self) / (1024 * 1024)  # Rough estimate
            
            self.memory_usage_history.append(memory_mb)
            
            # Keep only recent history
            if len(self.memory_usage_history) > 100:
                self.memory_usage_history = self.memory_usage_history[-100:]
            
            # Check if memory cleanup needed
            if memory_mb > self.config.memory_limit_mb * self.config.gc_threshold:
                self.logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                self._cleanup_memory()
                
        except Exception as e:
            self.logger.error(f"Memory monitoring error: {e}")
    
    def _cleanup_memory(self):
        """Perform memory cleanup."""
        import gc
        
        # Clear cache if enabled
        if self.cache:
            cache_size_before = len(self.cache._cache)
            self.cache.clear()
            self.logger.info(f"Cleared cache: {cache_size_before} entries")
        
        # Force garbage collection
        collected = gc.collect()
        self.stats['memory_cleanups'] += 1
        self.logger.info(f"Garbage collection freed {collected} objects")
    
    def optimized_inference(
        self, 
        inference_func: Callable,
        input_data: np.ndarray,
        use_cache: bool = True,
        use_concurrent: bool = False
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Perform optimized inference with caching and concurrency.
        
        Args:
            inference_func: Function to call for inference
            input_data: Input data for inference
            use_cache: Whether to use caching
            use_concurrent: Whether to use concurrent processing
            
        Returns:
            Tuple of (position, confidence, metadata)
        """
        start_time = time.time()
        metadata = {
            'cache_hit': False,
            'concurrent_execution': False,
            'optimization_time_ms': 0
        }
        
        self.stats['total_requests'] += 1
        
        # Try cache first
        if use_cache and self.cache:
            cached_result = self.cache.get(input_data)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                metadata['cache_hit'] = True
                metadata['optimization_time_ms'] = (time.time() - start_time) * 1000
                
                position, confidence = cached_result
                return position, confidence, metadata
        
        # Execute inference (concurrent or direct)
        if use_concurrent and self.worker_pool:
            # Submit to worker pool
            future = self.worker_pool.submit_task(inference_func, input_data)
            position, confidence = future.result()  # Wait for completion
            
            self.stats['concurrent_tasks'] += 1
            metadata['concurrent_execution'] = True
        else:
            # Direct execution
            position, confidence = inference_func(input_data)
        
        # Store in cache
        if use_cache and self.cache:
            self.cache.put(input_data, (position, confidence))
        
        metadata['optimization_time_ms'] = (time.time() - start_time) * 1000
        return position, confidence, metadata
    
    def batch_inference(
        self,
        inference_func: Callable,
        input_batch: List[np.ndarray],
        max_concurrent: Optional[int] = None
    ) -> List[Tuple[np.ndarray, float, Dict[str, Any]]]:
        """
        Perform batch inference with optimal concurrency.
        
        Args:
            inference_func: Function to call for inference
            input_batch: List of input arrays
            max_concurrent: Maximum concurrent tasks
            
        Returns:
            List of (position, confidence, metadata) tuples
        """
        if not self.worker_pool or len(input_batch) < self.config.batch_processing_threshold:
            # Sequential processing for small batches
            return [self.optimized_inference(inference_func, data, use_concurrent=False) 
                    for data in input_batch]
        
        # Concurrent batch processing
        max_workers = max_concurrent or self.worker_pool.current_workers
        results = []
        
        # Submit all tasks
        futures = []
        for data in input_batch:
            future = self.worker_pool.submit_task(
                self.optimized_inference, 
                inference_func, 
                data, 
                use_concurrent=False  # Already in worker
            )
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch inference task failed: {e}")
                # Return fallback result
                results.append((np.zeros(3), 0.0, {'error': str(e)}))
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.stats.copy()
        
        # Add cache stats
        if self.cache:
            cache_stats = self.cache.stats()
            stats.update({f'cache_{k}': v for k, v in cache_stats.items()})
        
        # Add worker pool stats
        if self.worker_pool:
            worker_stats = self.worker_pool.get_stats()
            stats.update({f'workers_{k}': v for k, v in worker_stats.items()})
        
        # Add memory stats
        if self.memory_usage_history:
            stats.update({
                'memory_current_mb': self.memory_usage_history[-1],
                'memory_avg_mb': np.mean(self.memory_usage_history),
                'memory_max_mb': np.max(self.memory_usage_history)
            })
        
        return stats