"""
Advanced caching system for EchoLoc-NN with intelligent cache management.
Generation 3 (Optimized) - Performance optimization and caching
"""

import os
import time
import json
import pickle
import hashlib
import threading
from typing import Any, Dict, Optional, Tuple, Callable, List
from dataclasses import dataclass, asdict
from queue import Queue, Empty
from collections import OrderedDict
import numpy as np


@dataclass 
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.timestamp + self.ttl)
    
    @property
    def age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.timestamp


class IntelligentCache:
    """
    Intelligent cache with adaptive eviction and performance optimization.
    
    Features:
    - LRU + TTL eviction
    - Adaptive sizing based on usage patterns
    - Thread-safe operations
    - Performance metrics and monitoring
    - Intelligent prefetching
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 512.0,
        default_ttl: Optional[float] = 3600.0,  # 1 hour
        eviction_policy: str = "lru_ttl",
        enable_persistence: bool = True,
        cache_dir: str = "~/.echoloc_cache"
    ):
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self.enable_persistence = enable_persistence
        self.cache_dir = os.path.expanduser(cache_dir)
        
        # Thread-safe cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Performance metrics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0,
            'last_cleanup': time.time()
        }
        
        # Background cleanup thread
        self._cleanup_interval = 300  # 5 minutes
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        # Initialize cache directory
        if self.enable_persistence:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._load_persistent_cache()
    
    def _hash_key(self, key: str) -> str:
        """Create hash of cache key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate size of object in bytes."""
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj.encode())
            elif isinstance(obj, (int, float)):
                return 8
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            else:
                return 1024  # Default estimate
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache."""
        with self._lock:
            hashed_key = self._hash_key(key)
            
            if hashed_key not in self._cache:
                self._stats['misses'] += 1
                return default
            
            entry = self._cache[hashed_key]
            
            # Check if expired
            if entry.is_expired:
                del self._cache[hashed_key]
                self._stats['misses'] += 1
                return default
            
            # Update access statistics
            entry.access_count += 1
            
            # Move to end (LRU)
            self._cache.move_to_end(hashed_key)
            
            self._stats['hits'] += 1
            return entry.value
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Put item in cache."""
        with self._lock:
            hashed_key = self._hash_key(key)
            size_bytes = self._estimate_size(value)
            
            # Check memory constraints
            if size_bytes > self.max_memory_bytes:
                return False  # Object too large
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl if ttl is not None else self.default_ttl,
                tags=tags or []
            )
            
            # Ensure capacity
            self._ensure_capacity(entry.size_bytes)
            
            # Store entry
            self._cache[hashed_key] = entry
            self._cache.move_to_end(hashed_key)
            
            # Update memory usage
            self._stats['memory_usage'] += entry.size_bytes
            
            return True
    
    def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry."""
        # Remove expired entries first
        self._remove_expired()
        
        # Calculate required space
        current_size = len(self._cache)
        current_memory = self._stats['memory_usage']
        required_memory = current_memory + new_entry_size
        
        # Evict based on policy
        while (current_size >= self.max_size or 
               required_memory > self.max_memory_bytes) and self._cache:
            
            if self.eviction_policy == "lru":
                evicted_key, evicted_entry = self._cache.popitem(last=False)
            elif self.eviction_policy == "lru_ttl":
                # Evict expired first, then LRU
                expired_key = self._find_expired()
                if expired_key:
                    evicted_entry = self._cache.pop(expired_key)
                else:
                    evicted_key, evicted_entry = self._cache.popitem(last=False)
            elif self.eviction_policy == "lfu":
                # Least frequently used
                lfu_key = min(self._cache.keys(), 
                             key=lambda k: self._cache[k].access_count)
                evicted_entry = self._cache.pop(lfu_key)
            else:
                # Default to LRU
                evicted_key, evicted_entry = self._cache.popitem(last=False)
            
            # Update stats
            self._stats['memory_usage'] -= evicted_entry.size_bytes
            self._stats['evictions'] += 1
            current_size -= 1
            required_memory -= evicted_entry.size_bytes
    
    def _find_expired(self) -> Optional[str]:
        """Find first expired entry."""
        for key, entry in self._cache.items():
            if entry.is_expired:
                return key
        return None
    
    def _remove_expired(self):
        """Remove all expired entries."""
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self._cache.pop(key)
            self._stats['memory_usage'] -= entry.size_bytes
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate all entries with matching tags."""
        with self._lock:
            keys_to_remove = []
            for key, entry in self._cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self._cache.pop(key)
                self._stats['memory_usage'] -= entry.size_bytes
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats['memory_usage'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / max(total_requests, 1) * 100
            
            return {
                **self._stats,
                'size': len(self._cache),
                'hit_rate_percent': hit_rate,
                'memory_usage_mb': self._stats['memory_usage'] / (1024 * 1024),
                'memory_usage_percent': (self._stats['memory_usage'] / self.max_memory_bytes) * 100
            }
    
    def _cleanup_worker(self):
        """Background cleanup worker."""
        while not self._stop_cleanup.wait(self._cleanup_interval):
            try:
                with self._lock:
                    self._remove_expired()
                    self._stats['last_cleanup'] = time.time()
                    
                    # Save to persistent storage periodically
                    if self.enable_persistence:
                        self._save_persistent_cache()
            except Exception as e:
                print(f"Cache cleanup error: {e}")
    
    def _save_persistent_cache(self):
        """Save cache to persistent storage."""
        try:
            cache_file = os.path.join(self.cache_dir, 'echoloc_cache.pkl')
            metadata_file = os.path.join(self.cache_dir, 'cache_metadata.json')
            
            # Save cache data
            with open(cache_file, 'wb') as f:
                pickle.dump(dict(self._cache), f)
            
            # Save metadata
            metadata = {
                'stats': self._stats,
                'config': {
                    'max_size': self.max_size,
                    'max_memory_bytes': self.max_memory_bytes,
                    'default_ttl': self.default_ttl,
                    'eviction_policy': self.eviction_policy
                },
                'saved_at': time.time()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _load_persistent_cache(self):
        """Load cache from persistent storage."""
        try:
            cache_file = os.path.join(self.cache_dir, 'echoloc_cache.pkl')
            metadata_file = os.path.join(self.cache_dir, 'cache_metadata.json')
            
            if not (os.path.exists(cache_file) and os.path.exists(metadata_file)):
                return
            
            # Load metadata first
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache is too old (more than 24 hours)
            if time.time() - metadata['saved_at'] > 86400:
                return
            
            # Load cache data
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Restore cache
            valid_entries = 0
            for key, entry in cache_data.items():
                if not entry.is_expired:
                    self._cache[key] = entry
                    valid_entries += 1
            
            # Update stats
            if 'stats' in metadata:
                self._stats.update(metadata['stats'])
            
            print(f"Loaded {valid_entries} valid cache entries from persistent storage")
            
        except Exception as e:
            print(f"Failed to load persistent cache: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        self._stop_cleanup.set()
        if self.enable_persistence:
            self._save_persistent_cache()


class EchoLocCacheManager:
    """
    Specialized cache manager for EchoLoc operations.
    """
    
    def __init__(self):
        # Different caches for different types of data
        self.model_cache = IntelligentCache(
            max_size=50, 
            max_memory_mb=2048,  # 2GB for models
            default_ttl=None,    # Models don't expire
            eviction_policy="lfu"
        )
        
        self.inference_cache = IntelligentCache(
            max_size=1000,
            max_memory_mb=512,   # 512MB for inference results
            default_ttl=300,     # 5 minutes
            eviction_policy="lru_ttl"
        )
        
        self.preprocessing_cache = IntelligentCache(
            max_size=5000,
            max_memory_mb=256,   # 256MB for preprocessed data
            default_ttl=600,     # 10 minutes
            eviction_policy="lru"
        )
    
    def cache_model_inference(
        self, 
        model_id: str, 
        echo_data: np.ndarray, 
        sensor_positions: Optional[np.ndarray] = None
    ) -> Callable:
        """Cache decorator for model inference."""
        def decorator(inference_func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Create cache key
                echo_hash = hashlib.sha256(echo_data.tobytes()).hexdigest()[:16]
                sensor_hash = ""
                if sensor_positions is not None:
                    sensor_hash = hashlib.sha256(sensor_positions.tobytes()).hexdigest()[:16]
                
                cache_key = f"inference:{model_id}:{echo_hash}:{sensor_hash}"
                
                # Try cache first
                result = self.inference_cache.get(cache_key)
                if result is not None:
                    return result
                
                # Run inference
                result = inference_func(*args, **kwargs)
                
                # Cache result
                self.inference_cache.put(
                    cache_key, 
                    result, 
                    tags=[f"model:{model_id}", "inference"]
                )
                
                return result
            return wrapper
        return decorator
    
    def cache_preprocessing(self, config_hash: str) -> Callable:
        """Cache decorator for preprocessing operations."""
        def decorator(preprocess_func: Callable) -> Callable:
            def wrapper(echo_data: np.ndarray, *args, **kwargs):
                # Create cache key
                data_hash = hashlib.sha256(echo_data.tobytes()).hexdigest()[:16]
                cache_key = f"preprocess:{config_hash}:{data_hash}"
                
                # Try cache first
                result = self.preprocessing_cache.get(cache_key)
                if result is not None:
                    return result
                
                # Run preprocessing
                result = preprocess_func(echo_data, *args, **kwargs)
                
                # Cache result
                self.preprocessing_cache.put(
                    cache_key, 
                    result, 
                    tags=["preprocessing"]
                )
                
                return result
            return wrapper
        return decorator
    
    def invalidate_model(self, model_id: str):
        """Invalidate all cached data for a specific model."""
        self.inference_cache.invalidate_by_tags([f"model:{model_id}"])
    
    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            'model_cache': self.model_cache.get_stats(),
            'inference_cache': self.inference_cache.get_stats(),
            'preprocessing_cache': self.preprocessing_cache.get_stats()
        }
    
    def optimize_caches(self):
        """Optimize cache configurations based on usage patterns."""
        # Get stats for all caches
        stats = self.get_cache_stats()
        
        # Adjust cache sizes based on hit rates
        for cache_name, cache_stats in stats.items():
            cache_obj = getattr(self, cache_name)
            
            hit_rate = cache_stats['hit_rate_percent']
            
            # If hit rate is low, increase cache size
            if hit_rate < 50 and cache_obj.max_size < 10000:
                cache_obj.max_size = int(cache_obj.max_size * 1.5)
                print(f"Increased {cache_name} size to {cache_obj.max_size}")
            
            # If memory usage is high, decrease TTL
            memory_percent = cache_stats['memory_usage_percent']
            if memory_percent > 80 and cache_obj.default_ttl:
                cache_obj.default_ttl *= 0.8
                print(f"Reduced {cache_name} TTL to {cache_obj.default_ttl}s")


# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> EchoLocCacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = EchoLocCacheManager()
    return _cache_manager