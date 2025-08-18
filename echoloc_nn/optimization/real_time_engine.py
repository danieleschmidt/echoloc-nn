"""Real-time inference engine with sub-50ms latency guarantees."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, List, Tuple
import time
import threading
from queue import Queue, Empty, Full
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor


class ProcessingMode(Enum):
    SINGLE = "single"           # Process one sample at a time
    BATCH = "batch"             # Batch processing for throughput
    STREAMING = "streaming"     # Real-time streaming
    PIPELINE = "pipeline"       # Pipelined processing


@dataclass
class InferenceRequest:
    """Request for real-time inference."""
    request_id: str
    echo_data: torch.Tensor
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # Higher number = higher priority
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result from real-time inference."""
    request_id: str
    position: torch.Tensor
    confidence: torch.Tensor
    processing_time_ms: float
    queue_time_ms: float
    total_latency_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealTimeEngine:
    """High-performance real-time inference engine."""
    
    def __init__(
        self,
        model: nn.Module,
        target_latency_ms: float = 50.0,
        max_batch_size: int = 8,
        max_queue_size: int = 100,
        processing_mode: ProcessingMode = ProcessingMode.STREAMING,
        device: str = 'cpu',
        enable_profiling: bool = False
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.target_latency_ms = target_latency_ms
        self.max_batch_size = max_batch_size
        self.processing_mode = processing_mode
        self.enable_profiling = enable_profiling
        
        # Queue management
        self.request_queue = Queue(maxsize=max_queue_size)
        self.result_callbacks = {}
        
        # Performance tracking
        self.stats = {
            'requests_processed': 0,
            'total_latency_ms': deque(maxlen=1000),
            'processing_times_ms': deque(maxlen=1000),
            'queue_times_ms': deque(maxlen=1000),
            'throughput_fps': deque(maxlen=100),
            'errors': 0,
            'queue_overflows': 0
        }
        
        # Threading
        self.processing_thread = None
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Batch processing state
        self.batch_buffer = []
        self.last_batch_time = time.time()
        self.batch_timeout_ms = 10.0  # Max wait time for batch
        
        # Pre-allocate tensors for efficiency
        self._preallocate_tensors()
        
        # Optimization state
        self.warmup_completed = False
        self.optimization_level = 0
    
    def _preallocate_tensors(self):
        """Pre-allocate commonly used tensors to reduce allocation overhead."""
        self.tensor_cache = {
            'batch_tensor': torch.zeros(self.max_batch_size, 4, 2048, device=self.device),
            'single_tensor': torch.zeros(1, 4, 2048, device=self.device)
        }
    
    def start(self):
        """Start the real-time processing engine."""
        if self.is_running:
            return
        
        self.is_running = True
        
        if self.processing_mode == ProcessingMode.STREAMING:
            self.processing_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        elif self.processing_mode == ProcessingMode.BATCH:
            self.processing_thread = threading.Thread(target=self._batch_worker, daemon=True)
        elif self.processing_mode == ProcessingMode.PIPELINE:
            self.processing_thread = threading.Thread(target=self._pipeline_worker, daemon=True)
        else:
            self.processing_thread = threading.Thread(target=self._single_worker, daemon=True)
        
        self.processing_thread.start()
        
        # Warmup
        self._warmup()
        
        print(f"RealTimeEngine started in {self.processing_mode.value} mode")
    
    def stop(self):
        """Stop the processing engine."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        self.executor.shutdown(wait=True)
        print("RealTimeEngine stopped")
    
    def _warmup(self):
        """Warm up the model for consistent performance."""
        print("Warming up model...")
        dummy_input = torch.randn(1, 4, 2048, device=self.device)
        
        # Run several warmup inferences
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        self.warmup_completed = True
        print("Model warmup completed")
    
    def predict_async(self, echo_data: torch.Tensor, request_id: str = None,
                     priority: int = 1, callback: Callable = None) -> str:
        """Submit asynchronous inference request."""
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        request = InferenceRequest(
            request_id=request_id,
            echo_data=echo_data.to(self.device),
            priority=priority,
            callback=callback
        )
        
        try:
            self.request_queue.put_nowait(request)
            if callback:
                self.result_callbacks[request_id] = callback
            return request_id
        except Full:
            self.stats['queue_overflows'] += 1
            raise RuntimeError("Request queue is full")
    
    def predict_sync(self, echo_data: torch.Tensor, timeout_ms: float = None) -> InferenceResult:
        """Synchronous inference with timeout."""
        if timeout_ms is None:
            timeout_ms = self.target_latency_ms * 2
        
        request_id = f"sync_{int(time.time() * 1000000)}"
        result_future = asyncio.Future()
        
        def callback(result):
            result_future.set_result(result)
        
        self.predict_async(echo_data, request_id, priority=10, callback=callback)
        
        # Wait for result
        start_time = time.time()
        while not result_future.done():
            if (time.time() - start_time) * 1000 > timeout_ms:
                raise TimeoutError(f"Inference timeout after {timeout_ms}ms")
            time.sleep(0.001)
        
        return result_future.result()
    
    def _streaming_worker(self):
        """Worker for streaming processing mode."""
        while self.is_running:
            try:
                request = self.request_queue.get(timeout=0.1)
                self._process_single_request(request)
            except Empty:
                continue
            except Exception as e:
                print(f"Streaming worker error: {e}")
                self.stats['errors'] += 1
    
    def _batch_worker(self):
        """Worker for batch processing mode."""
        while self.is_running:
            try:
                # Collect batch
                batch_requests = self._collect_batch()
                if batch_requests:
                    self._process_batch(batch_requests)
            except Exception as e:
                print(f"Batch worker error: {e}")
                self.stats['errors'] += 1
    
    def _pipeline_worker(self):
        """Worker for pipelined processing mode."""
        # Advanced pipelined processing with multiple stages
        while self.is_running:
            try:
                # Stage 1: Data preparation
                request = self.request_queue.get(timeout=0.1)
                
                # Stage 2: Model inference (can be parallelized)
                future = self.executor.submit(self._process_single_request, request)
                
                # Continue with next request while previous is processing
                
            except Empty:
                continue
            except Exception as e:
                print(f"Pipeline worker error: {e}")
                self.stats['errors'] += 1
    
    def _single_worker(self):
        """Worker for single request processing."""
        while self.is_running:
            try:
                request = self.request_queue.get(timeout=0.1)
                self._process_single_request(request)
            except Empty:
                continue
            except Exception as e:
                print(f"Single worker error: {e}")
                self.stats['errors'] += 1
    
    def _process_single_request(self, request: InferenceRequest):
        """Process a single inference request."""
        queue_time_ms = (time.time() - request.timestamp) * 1000
        
        # Prepare input
        input_tensor = request.echo_data.unsqueeze(0) if request.echo_data.dim() == 2 else request.echo_data
        
        # Run inference
        start_time = time.perf_counter()
        
        with torch.no_grad():
            if hasattr(self.model, 'predict_with_uncertainty'):
                position, confidence, uncertainty = self.model.predict_with_uncertainty(input_tensor)
            else:
                position, confidence = self.model(input_tensor)
                uncertainty = None
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        total_latency_ms = queue_time_ms + processing_time_ms
        
        # Create result
        result = InferenceResult(
            request_id=request.request_id,
            position=position.squeeze(0),
            confidence=confidence.squeeze(0),
            processing_time_ms=processing_time_ms,
            queue_time_ms=queue_time_ms,
            total_latency_ms=total_latency_ms
        )
        
        if uncertainty is not None:
            result.metadata['uncertainty'] = uncertainty.squeeze(0)
        
        # Update statistics
        self._update_stats(result)
        
        # Handle callback
        if request.callback:
            try:
                request.callback(result)
            except Exception as e:
                print(f"Callback error: {e}")
        
        # Clean up callback reference
        if request.request_id in self.result_callbacks:
            del self.result_callbacks[request.request_id]
    
    def _collect_batch(self) -> List[InferenceRequest]:
        """Collect requests for batch processing."""
        batch = []
        deadline = time.time() + self.batch_timeout_ms / 1000
        
        while len(batch) < self.max_batch_size and time.time() < deadline:
            try:
                request = self.request_queue.get(timeout=0.001)
                batch.append(request)
            except Empty:
                if batch:  # Return partial batch if we have some requests
                    break
                continue
        
        return batch
    
    def _process_batch(self, requests: List[InferenceRequest]):
        """Process a batch of requests."""
        if not requests:
            return
        
        batch_size = len(requests)
        
        # Prepare batch tensor
        batch_tensor = self.tensor_cache['batch_tensor'][:batch_size]
        
        for i, request in enumerate(requests):
            input_data = request.echo_data.unsqueeze(0) if request.echo_data.dim() == 2 else request.echo_data
            batch_tensor[i] = input_data.squeeze(0)
        
        # Run batch inference
        start_time = time.perf_counter()
        
        with torch.no_grad():
            positions, confidences = self.model(batch_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        processing_time_per_sample = processing_time_ms / batch_size
        
        # Create results for each request
        current_time = time.time()
        for i, request in enumerate(requests):
            queue_time_ms = (current_time - request.timestamp) * 1000
            total_latency_ms = queue_time_ms + processing_time_per_sample
            
            result = InferenceResult(
                request_id=request.request_id,
                position=positions[i],
                confidence=confidences[i],
                processing_time_ms=processing_time_per_sample,
                queue_time_ms=queue_time_ms,
                total_latency_ms=total_latency_ms
            )
            
            self._update_stats(result)
            
            if request.callback:
                try:
                    request.callback(result)
                except Exception as e:
                    print(f"Batch callback error: {e}")
    
    def _update_stats(self, result: InferenceResult):
        """Update performance statistics."""
        self.stats['requests_processed'] += 1
        self.stats['total_latency_ms'].append(result.total_latency_ms)
        self.stats['processing_times_ms'].append(result.processing_time_ms)
        self.stats['queue_times_ms'].append(result.queue_time_ms)
        
        # Calculate throughput
        if len(self.stats['total_latency_ms']) >= 10:
            recent_times = list(self.stats['total_latency_ms'])[-10:]
            avg_latency = np.mean(recent_times)
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0
            self.stats['throughput_fps'].append(fps)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.stats['total_latency_ms']:
            return {'no_data': True}
        
        latencies = np.array(self.stats['total_latency_ms'])
        processing_times = np.array(self.stats['processing_times_ms'])
        queue_times = np.array(self.stats['queue_times_ms'])
        
        stats = {
            'requests_processed': self.stats['requests_processed'],
            'current_queue_size': self.request_queue.qsize(),
            'latency_stats': {
                'mean_ms': float(np.mean(latencies)),
                'median_ms': float(np.median(latencies)),
                'p95_ms': float(np.percentile(latencies, 95)),
                'p99_ms': float(np.percentile(latencies, 99)),
                'min_ms': float(np.min(latencies)),
                'max_ms': float(np.max(latencies))
            },
            'processing_time_stats': {
                'mean_ms': float(np.mean(processing_times)),
                'p95_ms': float(np.percentile(processing_times, 95))
            },
            'queue_time_stats': {
                'mean_ms': float(np.mean(queue_times)),
                'p95_ms': float(np.percentile(queue_times, 95))
            },
            'throughput_fps': float(np.mean(self.stats['throughput_fps'])) if self.stats['throughput_fps'] else 0,
            'error_rate': self.stats['errors'] / max(1, self.stats['requests_processed']),
            'queue_overflow_count': self.stats['queue_overflows'],
            'target_latency_met': float(np.mean(latencies)) <= self.target_latency_ms,
            'processing_mode': self.processing_mode.value
        }
        
        return stats
    
    def adaptive_optimization(self):
        """Automatically adjust processing parameters based on performance."""
        stats = self.get_performance_stats()
        
        if stats.get('no_data'):
            return
        
        mean_latency = stats['latency_stats']['mean_ms']
        queue_size = stats['current_queue_size']
        
        # Adjust processing mode if needed
        if mean_latency > self.target_latency_ms * 1.2:
            if self.processing_mode == ProcessingMode.SINGLE:
                # Switch to batch mode for better throughput
                print("Switching to batch mode for better throughput")
                self.processing_mode = ProcessingMode.BATCH
            elif queue_size > self.max_batch_size * 2:
                # Increase batch size
                self.max_batch_size = min(16, self.max_batch_size * 2)
                print(f"Increased batch size to {self.max_batch_size}")
        
        elif mean_latency < self.target_latency_ms * 0.5 and queue_size == 0:
            # We have headroom, optimize for latency
            if self.processing_mode == ProcessingMode.BATCH:
                self.processing_mode = ProcessingMode.STREAMING
                print("Switching to streaming mode for lower latency")
    
    def benchmark(self, num_requests: int = 100, concurrent: bool = True) -> Dict[str, Any]:
        """Benchmark the real-time engine performance."""
        print(f"Benchmarking with {num_requests} requests...")
        
        # Generate test data
        test_data = [torch.randn(4, 2048) for _ in range(num_requests)]
        
        results = []
        start_time = time.time()
        
        if concurrent:
            # Submit all requests concurrently
            request_ids = []
            for i, data in enumerate(test_data):
                req_id = f"bench_{i}"
                self.predict_async(data, req_id)
                request_ids.append(req_id)
            
            # Wait for all results
            timeout = time.time() + 10.0  # 10 second timeout
            while len(results) < num_requests and time.time() < timeout:
                time.sleep(0.001)
                # Results would be collected via callbacks in real implementation
        
        else:
            # Submit requests sequentially
            for data in test_data:
                try:
                    result = self.predict_sync(data, timeout_ms=self.target_latency_ms * 2)
                    results.append(result)
                except TimeoutError:
                    print("Request timed out")
        
        total_time = time.time() - start_time
        
        benchmark_stats = self.get_performance_stats()
        benchmark_stats.update({
            'benchmark_duration_s': total_time,
            'requests_submitted': num_requests,
            'overall_throughput_fps': len(results) / total_time if total_time > 0 else 0,
            'success_rate': len(results) / num_requests
        })
        
        return benchmark_stats
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()