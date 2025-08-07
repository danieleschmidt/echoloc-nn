"""
Performance optimization components for sentiment analysis.

Provides model quantization, caching, batch processing, and GPU acceleration
for high-throughput spatial-aware sentiment analysis.
"""

import torch
import torch.nn as nn
import torch.jit
from torch.quantization import quantize_dynamic
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import asyncio
import time
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from collections import OrderedDict
import threading
import queue
import psutil
import gc

from .models import SpatialSentimentAnalyzer, MultiModalSentimentModel
from ..utils.error_handling import handle_errors, EchoLocError
from ..utils.monitoring import PerformanceMonitor
from ..utils.validation import validate_tensor

@dataclass
class OptimizationConfig:
    \"\"\"Configuration for sentiment analysis optimizations.\"\"\"
    # Model optimization
    enable_quantization: bool = True
    quantization_backend: str = \"x86\"  # \"x86\", \"arm\", \"qnnpack\"
    enable_jit_compilation: bool = True
    enable_tensor_cores: bool = True  # For A100/V100 GPUs
    
    # Caching
    enable_caching: bool = True
    cache_size_mb: int = 512
    cache_ttl_seconds: int = 3600
    enable_persistent_cache: bool = True
    cache_dir: str = \"/tmp/echoloc_sentiment_cache\"
    
    # Concurrency
    max_workers: int = 4
    enable_async_processing: bool = True
    batch_size: int = 32
    enable_gpu_batching: bool = True
    
    # Memory management
    enable_memory_optimization: bool = True
    memory_limit_mb: int = 2048
    enable_garbage_collection: bool = True
    gc_frequency: int = 100  # Process N requests before GC
    
    # Performance targets
    target_latency_ms: float = 50.0
    target_throughput_rps: float = 100.0
    enable_adaptive_batching: bool = True

class ModelOptimizer:
    \"\"\"Optimizes sentiment analysis models for production deployment.\"\"\"
    
    def __init__(self, config: OptimizationConfig = OptimizationConfig()):
        self.config = config
        self.performance_monitor = PerformanceMonitor(\"ModelOptimizer\")
        
    @handle_errors
    def optimize_spatial_model(self, model: SpatialSentimentAnalyzer) -> SpatialSentimentAnalyzer:
        \"\"\"Optimize spatial sentiment model for inference.\"\"\"
        
        optimized_model = model
        
        # Switch to evaluation mode
        optimized_model.eval()
        
        # Apply quantization
        if self.config.enable_quantization:
            optimized_model = self._apply_quantization(optimized_model)
            
        # Apply JIT compilation
        if self.config.enable_jit_compilation:
            optimized_model = self._apply_jit_compilation(optimized_model)
            
        # Optimize for tensor cores
        if self.config.enable_tensor_cores and torch.cuda.is_available():
            optimized_model = self._optimize_for_tensor_cores(optimized_model)
            
        return optimized_model
        
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        \"\"\"Apply dynamic quantization to model.\"\"\"
        
        with self.performance_monitor.measure(\"quantization\"):
            # Configure quantization backend
            torch.backends.quantized.engine = self.config.quantization_backend
            
            # Apply dynamic quantization to linear layers
            quantized_model = quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},  # Target layer types
                dtype=torch.qint8
            )
            
            return quantized_model
            
    def _apply_jit_compilation(self, model: nn.Module) -> torch.jit.ScriptModule:
        \"\"\"Apply JIT compilation for faster inference.\"\"\"
        
        with self.performance_monitor.measure(\"jit_compilation\"):
            # Create example inputs for tracing
            example_text = [\"This is a test sentence for JIT compilation\"]
            example_spatial = torch.randn(1, 9)  # Batch size 1, 9 spatial features
            
            # Trace the model
            try:
                traced_model = torch.jit.trace(
                    model,
                    (example_text, example_spatial),
                    strict=False
                )
                
                # Optimize traced model
                traced_model = torch.jit.optimize_for_inference(traced_model)
                
                return traced_model
                
            except Exception as e:
                print(f\"JIT compilation failed: {e}\")
                return model
                
    def _optimize_for_tensor_cores(self, model: nn.Module) -> nn.Module:
        \"\"\"Optimize model for tensor core utilization.\"\"\"
        
        # Enable mixed precision training
        model = model.half()  # Convert to FP16
        
        # Ensure layer dimensions are multiples of 8 for tensor cores
        # This would require architectural changes in practice
        
        return model
        
    def benchmark_model(self, model: nn.Module, num_samples: int = 1000) -> Dict[str, float]:
        \"\"\"Benchmark model performance.\"\"\"
        
        model.eval()
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                dummy_input = [\"warmup text\"]
                dummy_spatial = torch.randn(1, 9)
                _ = model(dummy_input, dummy_spatial)
                
        # Benchmark
        times = []
        
        for _ in range(num_samples):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                dummy_input = [\"benchmark text for performance measurement\"]
                dummy_spatial = torch.randn(1, 9)
                _ = model(dummy_input, dummy_spatial)
                
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
            
        return {
            \"avg_latency_ms\": np.mean(times),
            \"p50_latency_ms\": np.percentile(times, 50),
            \"p95_latency_ms\": np.percentile(times, 95),
            \"p99_latency_ms\": np.percentile(times, 99),
            \"min_latency_ms\": np.min(times),
            \"max_latency_ms\": np.max(times),
            \"throughput_rps\": 1000.0 / np.mean(times)
        }

class InferenceCache:
    \"\"\"High-performance caching system for sentiment analysis results.\"\"\"
    
    def __init__(self, config: OptimizationConfig = OptimizationConfig()):
        self.config = config
        self.memory_cache = OrderedDict()
        self.cache_stats = {
            \"hits\": 0,
            \"misses\": 0,
            \"evictions\": 0
        }
        self.lock = threading.RLock()
        
        # Calculate max entries based on memory limit
        estimated_entry_size_bytes = 1024  # Rough estimate
        max_memory_bytes = self.config.cache_size_mb * 1024 * 1024
        self.max_entries = max_memory_bytes // estimated_entry_size_bytes
        
    def _generate_cache_key(self, text: str, spatial_context: Optional[np.ndarray] = None) -> str:
        \"\"\"Generate cache key for text and spatial context.\"\"\"
        
        key_components = [text]
        
        if spatial_context is not None:
            # Round spatial values for better cache hit rate
            rounded_spatial = np.round(spatial_context, decimals=3)
            key_components.append(str(rounded_spatial.tolist()))
            
        key_string = \"|\".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def get(self, text: str, spatial_context: Optional[np.ndarray] = None) -> Optional[Dict]:
        \"\"\"Get cached result if available.\"\"\"
        
        if not self.config.enable_caching:
            return None
            
        cache_key = self._generate_cache_key(text, spatial_context)
        
        with self.lock:
            if cache_key in self.memory_cache:
                # Check TTL
                entry = self.memory_cache[cache_key]
                current_time = time.time()
                
                if (current_time - entry[\"timestamp\"]) < self.config.cache_ttl_seconds:
                    # Move to end (LRU)
                    self.memory_cache.move_to_end(cache_key)
                    self.cache_stats[\"hits\"] += 1
                    return entry[\"result\"]
                else:
                    # Expired entry
                    del self.memory_cache[cache_key]
                    
            self.cache_stats[\"misses\"] += 1
            return None
            
    def put(self, text: str, result: Dict, spatial_context: Optional[np.ndarray] = None):
        \"\"\"Store result in cache.\"\"\"
        
        if not self.config.enable_caching:
            return
            
        cache_key = self._generate_cache_key(text, spatial_context)
        
        with self.lock:
            # Check if we need to evict entries
            while len(self.memory_cache) >= self.max_entries:
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]
                self.cache_stats[\"evictions\"] += 1
                
            # Store new entry
            self.memory_cache[cache_key] = {
                \"result\": result,
                \"timestamp\": time.time()
            }
            
    def get_stats(self) -> Dict[str, Union[int, float]]:
        \"\"\"Get cache performance statistics.\"\"\"
        
        with self.lock:
            total_requests = self.cache_stats[\"hits\"] + self.cache_stats[\"misses\"]
            hit_rate = self.cache_stats[\"hits\"] / total_requests if total_requests > 0 else 0.0
            
            return {
                \"hit_rate\": hit_rate,
                \"total_entries\": len(self.memory_cache),
                \"max_entries\": self.max_entries,
                **self.cache_stats
            }
            
    def clear(self):
        \"\"\"Clear all cached entries.\"\"\"
        with self.lock:
            self.memory_cache.clear()
            self.cache_stats = {\"hits\": 0, \"misses\": 0, \"evictions\": 0}

class BatchProcessor:
    \"\"\"High-throughput batch processing for sentiment analysis.\"\"\"
    
    def __init__(
        self,
        model: Union[SpatialSentimentAnalyzer, MultiModalSentimentModel],
        config: OptimizationConfig = OptimizationConfig()
    ):
        self.model = model
        self.config = config
        self.cache = InferenceCache(config)
        self.performance_monitor = PerformanceMonitor(\"BatchProcessor\")
        
        # Threading setup
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Request queue for batching
        self.request_queue = queue.Queue()
        self.result_futures = {}
        
        # Adaptive batching
        self.batch_stats = {
            \"avg_batch_size\": config.batch_size,
            \"avg_latency_ms\": 0.0,
            \"throughput_rps\": 0.0
        }
        
        # Start batch processing thread
        self.batch_thread = threading.Thread(target=self._batch_processing_loop, daemon=True)
        self.batch_thread.start()
        
        # Memory management
        self.request_count = 0
        
    @handle_errors
    def process_single(self, text: str, spatial_context: Optional[np.ndarray] = None) -> Dict:
        \"\"\"Process single text with caching and optimization.\"\"\"
        
        # Check cache first
        cached_result = self.cache.get(text, spatial_context)
        if cached_result is not None:
            return cached_result
            
        # Process with model
        with self.performance_monitor.measure(\"single_inference\"):
            result = self.model.predict_sentiment(
                text=text,
                spatial_context=spatial_context
            )
            
        # Cache result
        self.cache.put(text, result, spatial_context)
        
        # Memory management
        self._manage_memory()
        
        return result
        
    def process_batch(self, batch_data: List[Dict]) -> List[Dict]:
        \"\"\"Process batch of requests efficiently.\"\"\"
        
        with self.performance_monitor.measure(\"batch_inference\"):
            # Separate cached and uncached requests
            cached_results = {}
            uncached_batch = []
            
            for i, item in enumerate(batch_data):
                text = item[\"text\"]
                spatial_context = item.get(\"spatial_context\")
                
                cached_result = self.cache.get(text, spatial_context)
                if cached_result is not None:
                    cached_results[i] = cached_result
                else:
                    uncached_batch.append((i, item))
                    
            # Process uncached requests in batch
            batch_results = {}
            
            if uncached_batch:
                # Prepare batch inputs
                texts = [item[1][\"text\"] for item in uncached_batch]
                spatial_contexts = []
                
                for _, item in uncached_batch:
                    spatial_context = item.get(\"spatial_context\")
                    if spatial_context is not None:
                        spatial_contexts.append(spatial_context)
                    else:
                        spatial_contexts.append(np.zeros(9))  # Default spatial context
                        
                # Convert to tensor if needed
                if spatial_contexts:
                    spatial_tensor = torch.tensor(
                        np.array(spatial_contexts), 
                        dtype=torch.float32
                    )
                else:
                    spatial_tensor = None
                    
                # Run batch inference
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(texts, spatial_tensor)
                    
                # Process outputs
                sentiment_probs = outputs[\"sentiment_probs\"].cpu().numpy()
                confidences = outputs[\"confidence\"].cpu().numpy()
                spatial_influences = outputs.get(\"spatial_influence\", torch.zeros(len(texts))).cpu().numpy()
                
                # Convert to individual results
                class_names = [\"negative\", \"neutral\", \"positive\"]
                
                for j, (original_idx, item) in enumerate(uncached_batch):
                    probs = sentiment_probs[j]
                    predicted_class = np.argmax(probs)
                    
                    result = {
                        \"sentiment\": class_names[predicted_class],
                        \"probabilities\": probs,
                        \"confidence\": float(confidences[j]),
                        \"spatial_influence\": float(spatial_influences[j])
                    }
                    
                    batch_results[original_idx] = result
                    
                    # Cache individual results
                    self.cache.put(
                        item[\"text\"], 
                        result, 
                        item.get(\"spatial_context\")
                    )
                    
            # Combine cached and computed results
            final_results = []
            for i in range(len(batch_data)):
                if i in cached_results:
                    final_results.append(cached_results[i])
                else:
                    final_results.append(batch_results[i])
                    
            return final_results
            
    def _batch_processing_loop(self):
        \"\"\"Background loop for batch processing requests.\"\"\"
        
        while True:
            try:
                # Collect requests for batching
                batch_requests = []
                batch_futures = []
                
                # Wait for first request
                try:
                    first_request = self.request_queue.get(timeout=1.0)
                    batch_requests.append(first_request[\"data\"])
                    batch_futures.append(first_request[\"future\"])
                except queue.Empty:
                    continue
                    
                # Collect additional requests up to batch size
                batch_size = self._get_adaptive_batch_size()
                
                while len(batch_requests) < batch_size:
                    try:
                        request = self.request_queue.get(timeout=0.01)
                        batch_requests.append(request[\"data\"])
                        batch_futures.append(request[\"future\"])
                    except queue.Empty:
                        break
                        
                # Process batch
                try:
                    results = self.process_batch(batch_requests)
                    
                    # Set results for futures
                    for future, result in zip(batch_futures, results):
                        if not future.cancelled():
                            future.set_result(result)
                            
                except Exception as e:
                    # Set exception for all futures
                    for future in batch_futures:
                        if not future.cancelled():
                            future.set_exception(e)
                            
                # Update batch statistics
                self._update_batch_stats(len(batch_requests))
                
            except Exception as e:
                print(f\"Batch processing error: {e}\")
                time.sleep(0.1)
                
    def _get_adaptive_batch_size(self) -> int:
        \"\"\"Get adaptive batch size based on performance.\"\"\"
        
        if not self.config.enable_adaptive_batching:
            return self.config.batch_size
            
        # Simple adaptive logic - increase batch size if latency is good
        current_batch_size = int(self.batch_stats[\"avg_batch_size\"])
        current_latency = self.batch_stats[\"avg_latency_ms\"]
        
        if current_latency < self.config.target_latency_ms * 0.8:
            # Latency is good, try increasing batch size
            return min(current_batch_size + 2, self.config.batch_size * 2)
        elif current_latency > self.config.target_latency_ms * 1.2:
            # Latency is too high, decrease batch size
            return max(current_batch_size - 2, 1)
        else:
            return current_batch_size
            
    def _update_batch_stats(self, batch_size: int):
        \"\"\"Update batch processing statistics.\"\"\"
        
        # Exponential moving average
        alpha = 0.1
        self.batch_stats[\"avg_batch_size\"] = (
            (1 - alpha) * self.batch_stats[\"avg_batch_size\"] + 
            alpha * batch_size
        )
        
    def _manage_memory(self):
        \"\"\"Manage memory usage and garbage collection.\"\"\"
        
        self.request_count += 1
        
        if (self.config.enable_garbage_collection and 
            self.request_count % self.config.gc_frequency == 0):
            
            # Check memory usage
            memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            if memory_usage_mb > self.config.memory_limit_mb:
                # Clear some cache entries
                cache_size_before = len(self.cache.memory_cache)
                entries_to_remove = cache_size_before // 4  # Remove 25%
                
                with self.cache.lock:
                    for _ in range(entries_to_remove):
                        if self.cache.memory_cache:
                            oldest_key = next(iter(self.cache.memory_cache))
                            del self.cache.memory_cache[oldest_key]
                            
                # Force garbage collection
                gc.collect()
                
                print(f\"Memory management: {memory_usage_mb:.1f} MB -> {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB\")
                
    def get_performance_stats(self) -> Dict[str, Union[float, int, Dict]]:
        \"\"\"Get comprehensive performance statistics.\"\"\"
        
        return {
            \"batch_stats\": self.batch_stats,
            \"cache_stats\": self.cache.get_stats(),
            \"performance_metrics\": self.performance_monitor.get_metrics(),
            \"memory_usage_mb\": psutil.Process().memory_info().rss / 1024 / 1024,
            \"request_count\": self.request_count
        }

class ConcurrentSentimentAnalyzer:
    \"\"\"High-performance concurrent sentiment analyzer.\"\"\"
    
    def __init__(
        self,
        model: Union[SpatialSentimentAnalyzer, MultiModalSentimentModel],
        config: OptimizationConfig = OptimizationConfig()
    ):
        self.config = config
        
        # Optimize model
        optimizer = ModelOptimizer(config)
        self.model = optimizer.optimize_spatial_model(model)
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(self.model, config)
        
        # Async processing setup
        if config.enable_async_processing:
            self.process_executor = ProcessPoolExecutor(
                max_workers=min(config.max_workers, psutil.cpu_count())
            )
        else:
            self.process_executor = None
            
    async def analyze_async(
        self, 
        text: str, 
        spatial_context: Optional[np.ndarray] = None
    ) -> Dict:
        \"\"\"Asynchronous sentiment analysis.\"\"\"
        
        if self.config.enable_async_processing and self.process_executor:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.process_executor,
                self.batch_processor.process_single,
                text,
                spatial_context
            )
            return result
        else:
            return self.batch_processor.process_single(text, spatial_context)
            
    async def analyze_batch_async(self, batch_data: List[Dict]) -> List[Dict]:
        \"\"\"Asynchronous batch sentiment analysis.\"\"\"
        
        if self.config.enable_async_processing and self.process_executor:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.process_executor,
                self.batch_processor.process_batch,
                batch_data
            )
            return result
        else:
            return self.batch_processor.process_batch(batch_data)
            
    def get_optimization_report(self) -> Dict[str, Any]:
        \"\"\"Get comprehensive optimization report.\"\"\"
        
        # Model benchmarks
        benchmark_results = ModelOptimizer(self.config).benchmark_model(self.model)
        
        # Performance statistics
        performance_stats = self.batch_processor.get_performance_stats()
        
        return {
            \"configuration\": {
                \"quantization_enabled\": self.config.enable_quantization,
                \"jit_compilation_enabled\": self.config.enable_jit_compilation,
                \"caching_enabled\": self.config.enable_caching,
                \"async_processing_enabled\": self.config.enable_async_processing,
                \"max_workers\": self.config.max_workers,
                \"batch_size\": self.config.batch_size
            },
            \"model_benchmarks\": benchmark_results,
            \"runtime_performance\": performance_stats,
            \"system_info\": {
                \"cpu_count\": psutil.cpu_count(),
                \"memory_total_gb\": psutil.virtual_memory().total / (1024**3),
                \"gpu_available\": torch.cuda.is_available(),
                \"gpu_name\": torch.cuda.get_device_name() if torch.cuda.is_available() else None
            }
        }"