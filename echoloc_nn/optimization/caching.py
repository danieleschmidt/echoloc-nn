"""
Intelligent caching system for EchoLoc-NN.
"""

import time
import threading
import hashlib
import pickle
from typing import Any, Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
import torch
from ..utils.logging_config import get_logger


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    
    max_size_mb: float = 100.0  # Maximum cache size in MB
    ttl_seconds: float = 3600.0  # Time to live for cache entries
    cleanup_interval: float = 300.0  # Cleanup interval in seconds
    enable_persistence: bool = False  # Persist cache to disk
    cache_directory: Optional[str] = None  # Directory for persistent cache
    hash_algorithm: str = "sha256"  # Hashing algorithm for keys


@dataclass
class CacheStats:
    """Cache performance statistics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_mb: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


class LRUCache:
    """
    Thread-safe LRU cache with TTL and size limits.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.sizes: Dict[str, int] = {}
        self.total_size = 0
        self.lock = threading.RLock()
        self.stats = CacheStats()
        
        self.logger = get_logger('lru_cache')
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self.stop_cleanup = threading.Event()
        self.cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.timestamps[key] > self.config.ttl_seconds:
                    self._remove_key(key)
                    self.stats.misses += 1
                    self.stats.update_hit_rate()
                    return None
                
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                
                self.stats.hits += 1
                self.stats.update_hit_rate()
                return value
            else:
                self.stats.misses += 1
                self.stats.update_hit_rate()
                return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache."""
        with self.lock:
            # Calculate size
            try:
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    size = self._get_tensor_size(value)
                else:
                    size = len(pickle.dumps(value))
            except Exception:
                size = 1024  # Default size if can't calculate
            
            # Check if single item is too large
            max_size_bytes = self.config.max_size_mb * 1024 * 1024
            if size > max_size_bytes:
                self.logger.warning(f"Item too large for cache: {size} bytes")
                return False
            
            # Remove existing key if present
            if key in self.cache:
                self._remove_key(key)
            
            # Make space if needed
            while self.total_size + size > max_size_bytes and self.cache:
                oldest_key = next(iter(self.cache))
                self._remove_key(oldest_key)
                self.stats.evictions += 1
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.sizes[key] = size
            self.total_size += size
            
            # Update stats
            self.stats.size_mb = self.total_size / 1024 / 1024
            
            return True
    
    def _remove_key(self, key: str):
        """Remove key from cache (internal use)."""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
            self.total_size -= self.sizes[key]
            del self.sizes[key]
    
    def _get_tensor_size(self, tensor: Any) -> int:
        """Get size of tensor in bytes."""
        if isinstance(tensor, np.ndarray):
            return tensor.nbytes
        elif isinstance(tensor, torch.Tensor):
            return tensor.element_size() * tensor.numel()
        else:
            return len(pickle.dumps(tensor))
    
    def _cleanup_loop(self):
        """Background cleanup of expired entries."""
        while not self.stop_cleanup.wait(self.config.cleanup_interval):
            self._cleanup_expired()
    
    def _cleanup_expired(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, timestamp in self.timestamps.items():
                if current_time - timestamp > self.config.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_key(key)
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                self.stats.size_mb = self.total_size / 1024 / 1024
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.sizes.clear()
            self.total_size = 0
            self.stats.size_mb = 0.0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                size_mb=self.stats.size_mb,
                hit_rate=self.stats.hit_rate
            )


class EchoCache:
    """
    Specialized cache for echo data and processing results.
    
    Provides intelligent caching of preprocessed echo data,
    intermediate features, and localization results.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = LRUCache(config)
        self.logger = get_logger('echo_cache')
        
        # Cache categories
        self.echo_data_cache = LRUCache(config)
        self.feature_cache = LRUCache(config)
        self.result_cache = LRUCache(config)
    
    def _hash_array(self, array: np.ndarray) -> str:
        """Create hash of numpy array."""
        if self.config.hash_algorithm == "sha256":
            hash_obj = hashlib.sha256()
        elif self.config.hash_algorithm == "md5":
            hash_obj = hashlib.md5()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.config.hash_algorithm}")
        
        # Hash array data and metadata
        hash_obj.update(array.tobytes())
        hash_obj.update(str(array.shape).encode())
        hash_obj.update(str(array.dtype).encode())
        
        return hash_obj.hexdigest()
    
    def cache_echo_data(
        self,
        raw_echo: np.ndarray,
        processed_echo: np.ndarray
    ) -> str:
        """
        Cache processed echo data.
        
        Args:
            raw_echo: Raw input echo data
            processed_echo: Processed echo data
            
        Returns:
            Cache key for the processed data
        """
        cache_key = f"echo_{self._hash_array(raw_echo)}"
        
        if self.echo_data_cache.put(cache_key, processed_echo):
            self.logger.debug(f"Cached processed echo data: {cache_key}")
        
        return cache_key
    
    def get_cached_echo_data(self, raw_echo: np.ndarray) -> Optional[np.ndarray]:
        """Get cached processed echo data."""
        cache_key = f"echo_{self._hash_array(raw_echo)}"
        cached_data = self.echo_data_cache.get(cache_key)
        
        if cached_data is not None:
            self.logger.debug(f"Echo data cache hit: {cache_key}")
        
        return cached_data
    
    def cache_features(
        self,
        echo_data: np.ndarray,
        features: torch.Tensor,
        feature_type: str = "cnn"
    ) -> str:
        """
        Cache extracted features.
        
        Args:
            echo_data: Input echo data
            features: Extracted features
            feature_type: Type of features (cnn, transformer, etc.)
            
        Returns:
            Cache key for the features
        """
        cache_key = f"features_{feature_type}_{self._hash_array(echo_data)}"
        
        if self.feature_cache.put(cache_key, features):
            self.logger.debug(f"Cached features: {cache_key}")
        
        return cache_key
    
    def get_cached_features(
        self,
        echo_data: np.ndarray,
        feature_type: str = "cnn"
    ) -> Optional[torch.Tensor]:
        """Get cached features."""
        cache_key = f"features_{feature_type}_{self._hash_array(echo_data)}"
        cached_features = self.feature_cache.get(cache_key)
        
        if cached_features is not None:
            self.logger.debug(f"Feature cache hit: {cache_key}")
        
        return cached_features
    
    def cache_result(
        self,
        echo_data: np.ndarray,
        position: np.ndarray,
        confidence: float,
        sensor_positions: Optional[np.ndarray] = None
    ) -> str:
        """
        Cache localization result.
        
        Args:
            echo_data: Input echo data
            position: Estimated position
            confidence: Confidence score
            sensor_positions: Sensor array positions
            
        Returns:
            Cache key for the result
        """
        # Create composite key
        key_components = [self._hash_array(echo_data)]
        if sensor_positions is not None:
            key_components.append(self._hash_array(sensor_positions))
        
        cache_key = f"result_{'_'.join(key_components)}"
        
        result = {
            'position': position,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        if self.result_cache.put(cache_key, result):
            self.logger.debug(f"Cached localization result: {cache_key}")
        
        return cache_key
    
    def get_cached_result(
        self,
        echo_data: np.ndarray,
        sensor_positions: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached localization result."""
        key_components = [self._hash_array(echo_data)]
        if sensor_positions is not None:
            key_components.append(self._hash_array(sensor_positions))
        
        cache_key = f"result_{'_'.join(key_components)}"
        cached_result = self.result_cache.get(cache_key)
        
        if cached_result is not None:
            self.logger.debug(f"Result cache hit: {cache_key}")
        
        return cached_result
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        echo_stats = self.echo_data_cache.get_stats()
        feature_stats = self.feature_cache.get_stats()
        result_stats = self.result_cache.get_stats()
        
        total_hits = echo_stats.hits + feature_stats.hits + result_stats.hits
        total_misses = echo_stats.misses + feature_stats.misses + result_stats.misses
        overall_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
        
        return {
            'overall': {
                'hit_rate': overall_hit_rate,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'total_size_mb': echo_stats.size_mb + feature_stats.size_mb + result_stats.size_mb
            },
            'echo_data': echo_stats.__dict__,
            'features': feature_stats.__dict__,
            'results': result_stats.__dict__
        }


class ModelCache:
    """
    Cache for model states and intermediate computations.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = LRUCache(config)
        self.logger = get_logger('model_cache')
        
        # Model-specific caches
        self.weight_cache = {}  # Cache for model weights
        self.computation_cache = LRUCache(config)  # Cache for intermediate computations
    
    def cache_model_weights(
        self,
        model_id: str,
        state_dict: Dict[str, torch.Tensor]
    ):
        """Cache model weights."""
        # Store in persistent cache (not LRU due to importance)
        self.weight_cache[model_id] = {
            'state_dict': state_dict,
            'timestamp': time.time()
        }
        
        self.logger.info(f"Cached model weights: {model_id}")
    
    def get_cached_model_weights(self, model_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached model weights."""
        if model_id in self.weight_cache:
            cached_data = self.weight_cache[model_id]
            
            # Check if weights are still valid
            age = time.time() - cached_data['timestamp']
            if age < self.config.ttl_seconds:
                self.logger.debug(f"Model weight cache hit: {model_id}")
                return cached_data['state_dict']
            else:
                # Expired
                del self.weight_cache[model_id]
        
        return None
    
    def cache_computation(
        self,
        computation_id: str,
        input_hash: str,
        result: torch.Tensor
    ) -> str:
        """Cache intermediate computation result."""
        cache_key = f"{computation_id}_{input_hash}"
        
        if self.computation_cache.put(cache_key, result):
            self.logger.debug(f"Cached computation: {cache_key}")
        
        return cache_key
    
    def get_cached_computation(
        self,
        computation_id: str,
        input_hash: str
    ) -> Optional[torch.Tensor]:
        """Get cached computation result."""
        cache_key = f"{computation_id}_{input_hash}"
        return self.computation_cache.get(cache_key)


class AdaptiveCache:
    """
    Adaptive cache that learns usage patterns and optimizes accordingly.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.base_cache = LRUCache(config)
        self.access_patterns: Dict[str, List[float]] = {}
        self.prediction_cache = LRUCache(config)
        
        self.logger = get_logger('adaptive_cache')
        
        # Learning parameters
        self.learning_window = 100  # Track last N accesses
        self.prediction_threshold = 0.7  # Confidence threshold for predictions
    
    def get(self, key: str) -> Optional[Any]:
        """Get item with access pattern learning."""
        self._record_access(key)
        return self.base_cache.get(key)
    
    def put(self, key: str, value: Any) -> bool:
        """Put item with adaptive priority."""
        # Predict future access probability
        access_prob = self._predict_access_probability(key)
        
        if access_prob > self.prediction_threshold:
            # High probability of future access - prioritize
            self.logger.debug(f"High priority cache item: {key} (prob={access_prob:.2f})")
            return self.base_cache.put(key, value)
        else:
            # Lower priority - might not cache or use shorter TTL
            return self.base_cache.put(key, value)
    
    def _record_access(self, key: str):
        """Record access pattern for learning."""
        current_time = time.time()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(current_time)
        
        # Keep only recent accesses
        if len(self.access_patterns[key]) > self.learning_window:
            self.access_patterns[key] = self.access_patterns[key][-self.learning_window:]
    
    def _predict_access_probability(self, key: str) -> float:
        """Predict probability of future access."""
        if key not in self.access_patterns or len(self.access_patterns[key]) < 3:
            return 0.5  # Default probability
        
        access_times = self.access_patterns[key]
        current_time = time.time()
        
        # Calculate access frequency
        time_span = current_time - access_times[0]
        frequency = len(access_times) / max(time_span, 1.0)  # accesses per second
        
        # Calculate time since last access  
        time_since_last = current_time - access_times[-1]
        
        # Simple probability model
        # High frequency and recent access -> high probability
        recency_factor = max(0, 1 - time_since_last / 3600)  # Decay over 1 hour
        frequency_factor = min(1, frequency * 10)  # Scale frequency
        
        probability = (recency_factor + frequency_factor) / 2
        return probability
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learned access patterns."""
        total_keys = len(self.access_patterns)
        avg_accesses = np.mean([len(pattern) for pattern in self.access_patterns.values()]) if total_keys > 0 else 0
        
        # Find most frequently accessed items
        frequent_items = sorted(
            self.access_patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:10]
        
        return {
            'total_tracked_keys': total_keys,
            'average_accesses_per_key': avg_accesses,
            'most_frequent_items': [(key, len(pattern)) for key, pattern in frequent_items]
        }