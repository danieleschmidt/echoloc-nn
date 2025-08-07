"""
Concurrent inference engine for high-throughput ultrasonic localization.
Generation 3 (Optimized) - Concurrent processing and scaling
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass
from collections import deque
import logging


@dataclass
class InferenceRequest:
    """Request for inference processing."""
    request_id: str
    echo_data: np.ndarray
    sensor_positions: Optional[np.ndarray] = None
    timestamp: float = None
    priority: int = 0  # Higher numbers = higher priority
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class InferenceResult:
    """Result from inference processing."""
    request_id: str
    position: np.ndarray
    confidence: float
    processing_time_ms: float
    timestamp: float
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConcurrentInferenceEngine:
    """
    High-performance concurrent inference engine.
    
    Features:
    - Multi-threaded and multi-process processing
    - Priority queue for request scheduling
    - Load balancing across workers
    - Real-time performance monitoring
    - Adaptive resource scaling
    """
    
    def __init__(
        self,
        model,
        max_workers: int = None,
        use_processes: bool = False,
        queue_size: int = 1000,
        batch_size: int = 1,
        target_latency_ms: float = 50.0,
        enable_batching: bool = True,
        worker_timeout: float = 30.0
    ):
        self.model = model
        self.use_processes = use_processes
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.target_latency_ms = target_latency_ms
        self.enable_batching = enable_batching
        self.worker_timeout = worker_timeout
        
        # Determine optimal number of workers
        if max_workers is None:
            if use_processes:
                max_workers = min(mp.cpu_count(), 4)  # Limit processes
            else:
                max_workers = min(threading.active_count() * 2, 16)  # More threads
        
        self.max_workers = max_workers
        
        # Request queues (separate for different priorities)
        self.high_priority_queue = Queue(maxsize=queue_size // 2)
        self.normal_priority_queue = Queue(maxsize=queue_size // 2)
        
        # Result storage
        self.results: Dict[str, InferenceResult] = {}
        self.result_callbacks: Dict[str, Callable] = {}
        
        # Worker management
        self.workers_active = False
        self.executor = None
        self.worker_futures = []
        
        # Performance monitoring
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'average_latency_ms': 0.0,
            'current_queue_size': 0,
            'active_workers': 0,
            'throughput_per_second': 0.0,
            'last_throughput_check': time.time(),
            'requests_in_last_second': deque(maxlen=1000)
        }
        
        # Adaptive scaling
        self.scaling_enabled = True
        self.last_scale_check = time.time()
        self.scale_check_interval = 10.0  # seconds
        
        # Logging
        self.logger = logging.getLogger('ConcurrentInference')
    
    def start(self):
        """Start the concurrent inference engine."""
        if self.workers_active:
            return
        
        self.workers_active = True
        
        # Create executor
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Start worker threads/processes
        for i in range(self.max_workers):
            future = self.executor.submit(self._worker_loop, i)
            self.worker_futures.append(future)
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info(f"Started {self.max_workers} {'processes' if self.use_processes else 'threads'}")
    
    def stop(self):
        """Stop the concurrent inference engine."""
        if not self.workers_active:
            return
        
        self.workers_active = False
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        # Clear futures
        self.worker_futures = []
        
        self.logger.info("Concurrent inference engine stopped")
    
    def submit(
        self,
        request_id: str,
        echo_data: np.ndarray,
        sensor_positions: Optional[np.ndarray] = None,
        priority: int = 0,
        callback: Optional[Callable[[InferenceResult], None]] = None,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Submit inference request.
        
        Args:
            request_id: Unique request identifier
            echo_data: Ultrasonic echo data
            sensor_positions: Optional sensor positions
            priority: Request priority (higher = more urgent)
            callback: Optional callback for result notification
            timeout: Optional timeout for request processing
            
        Returns:
            True if request was queued successfully
        """
        if not self.workers_active:
            self.start()
        
        # Create request
        request = InferenceRequest(
            request_id=request_id,
            echo_data=echo_data,
            sensor_positions=sensor_positions,
            priority=priority,
            metadata={'timeout': timeout}
        )
        
        # Store callback if provided
        if callback:
            self.result_callbacks[request_id] = callback
        
        # Queue request based on priority
        try:
            if priority > 0:
                self.high_priority_queue.put_nowait(request)
            else:
                self.normal_priority_queue.put_nowait(request)
            
            self.stats['total_requests'] += 1
            self.stats['current_queue_size'] += 1
            return True
            
        except:
            return False
    
    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[InferenceResult]:
        """
        Get result for a specific request.
        
        Args:
            request_id: Request identifier
            timeout: Maximum wait time in seconds
            
        Returns:
            InferenceResult if available, None otherwise
        """
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            if request_id in self.results:
                result = self.results.pop(request_id)
                # Remove callback if exists
                self.result_callbacks.pop(request_id, None)
                return result
            
            time.sleep(0.001)  # Small delay to prevent busy waiting
        
        return None
    
    def _worker_loop(self, worker_id: int):
        """Main worker processing loop."""
        self.logger.debug(f"Worker {worker_id} started")
        
        while self.workers_active:
            try:
                # Get next request (prioritize high priority queue)
                request = self._get_next_request()
                
                if request is None:
                    time.sleep(0.01)  # Brief pause if no work
                    continue
                
                # Process request
                result = self._process_request(request, worker_id)
                
                # Store result
                self.results[request.request_id] = result
                
                # Call callback if provided
                if request.request_id in self.result_callbacks:
                    try:
                        callback = self.result_callbacks[request.request_id]
                        callback(result)
                    except Exception as e:
                        self.logger.error(f"Callback error for {request.request_id}: {e}")
                
                # Update stats
                self.stats['completed_requests'] += 1
                self.stats['current_queue_size'] -= 1
                self.stats['requests_in_last_second'].append(time.time())
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                if 'request' in locals():
                    self._handle_failed_request(request, str(e))
    
    def _get_next_request(self) -> Optional[InferenceRequest]:
        """Get next request from priority queues."""
        # Try high priority queue first
        try:
            return self.high_priority_queue.get_nowait()
        except Empty:
            pass
        
        # Try normal priority queue
        try:
            return self.normal_priority_queue.get_nowait()
        except Empty:
            pass
        
        return None
    
    def _process_request(self, request: InferenceRequest, worker_id: int) -> InferenceResult:
        """Process individual inference request."""
        start_time = time.time()
        
        try:
            # Check timeout
            timeout = request.metadata.get('timeout')
            if timeout and (time.time() - request.timestamp) > timeout:
                raise TimeoutError(f"Request {request.request_id} timed out")
            
            # Perform inference
            position, confidence = self.model.predict_position(
                request.echo_data, 
                request.sensor_positions
            )
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            return InferenceResult(
                request_id=request.request_id,
                position=position,
                confidence=confidence,
                processing_time_ms=processing_time,
                timestamp=time.time(),
                metadata={
                    'worker_id': worker_id,
                    'queue_wait_time_ms': (start_time - request.timestamp) * 1000
                }
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return InferenceResult(
                request_id=request.request_id,
                position=np.array([0.0, 0.0, 0.0]),
                confidence=0.0,
                processing_time_ms=processing_time,
                timestamp=time.time(),
                error=str(e),
                metadata={'worker_id': worker_id}
            )
    
    def _handle_failed_request(self, request: InferenceRequest, error: str):
        """Handle failed request."""
        result = InferenceResult(
            request_id=request.request_id,
            position=np.array([0.0, 0.0, 0.0]),
            confidence=0.0,
            processing_time_ms=0.0,
            timestamp=time.time(),
            error=error
        )
        
        self.results[request.request_id] = result
        self.stats['failed_requests'] += 1
        self.stats['current_queue_size'] -= 1
    
    def _monitor_loop(self):
        """Background monitoring and optimization loop."""
        while self.workers_active:
            try:
                # Update throughput statistics
                self._update_throughput_stats()
                
                # Check for adaptive scaling
                if self.scaling_enabled:
                    self._check_adaptive_scaling()
                
                # Update average latency
                self._update_latency_stats()
                
                # Clean old results
                self._cleanup_old_results()
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
    
    def _update_throughput_stats(self):
        """Update throughput statistics."""
        current_time = time.time()
        
        # Count requests in last second
        cutoff_time = current_time - 1.0
        recent_requests = [t for t in self.stats['requests_in_last_second'] if t > cutoff_time]
        
        self.stats['throughput_per_second'] = len(recent_requests)
        self.stats['last_throughput_check'] = current_time
    
    def _check_adaptive_scaling(self):
        """Check if we need to scale workers up or down."""
        current_time = time.time()
        
        if current_time - self.last_scale_check < self.scale_check_interval:
            return
        
        self.last_scale_check = current_time
        
        # Calculate metrics
        queue_size = self.stats['current_queue_size']
        throughput = self.stats['throughput_per_second']
        avg_latency = self.stats['average_latency_ms']
        
        # Scale up if queue is building up and latency is high
        if queue_size > self.queue_size * 0.7 and avg_latency > self.target_latency_ms * 1.5:
            if len(self.worker_futures) < self.max_workers * 2:  # Don't exceed 2x original
                self._add_worker()
        
        # Scale down if queue is empty and we have excess workers
        elif queue_size < self.queue_size * 0.1 and len(self.worker_futures) > self.max_workers:
            if throughput < self.max_workers * 0.5:  # Low throughput
                self._remove_worker()
    
    def _add_worker(self):
        """Add additional worker."""
        try:
            worker_id = len(self.worker_futures)
            future = self.executor.submit(self._worker_loop, worker_id)
            self.worker_futures.append(future)
            self.logger.info(f"Added worker {worker_id} (total: {len(self.worker_futures)})")
        except Exception as e:
            self.logger.error(f"Failed to add worker: {e}")
    
    def _remove_worker(self):
        """Remove worker (graceful degradation)."""
        if len(self.worker_futures) > self.max_workers // 2:  # Keep at least half
            # Note: In practice, we'd need more sophisticated worker management
            # For now, just reduce the target in monitoring
            pass
    
    def _update_latency_stats(self):
        """Update average latency statistics."""
        if self.stats['completed_requests'] == 0:
            return
        
        # This is a simplified calculation - in practice you'd track individual latencies
        total_requests = self.stats['completed_requests']
        if total_requests > 0:
            # Use recent throughput as proxy for latency trends
            throughput = max(self.stats['throughput_per_second'], 1)
            estimated_latency = (1000.0 / throughput) * self.max_workers
            
            # Exponential moving average
            alpha = 0.1
            if self.stats['average_latency_ms'] == 0:
                self.stats['average_latency_ms'] = estimated_latency
            else:
                self.stats['average_latency_ms'] = (
                    alpha * estimated_latency + 
                    (1 - alpha) * self.stats['average_latency_ms']
                )
    
    def _cleanup_old_results(self):
        """Clean up old results to prevent memory leaks."""
        current_time = time.time()
        cleanup_age = 300  # 5 minutes
        
        old_results = []
        for request_id, result in self.results.items():
            if current_time - result.timestamp > cleanup_age:
                old_results.append(request_id)
        
        for request_id in old_results:
            self.results.pop(request_id, None)
            self.result_callbacks.pop(request_id, None)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        total_requests = self.stats['total_requests']
        completed_requests = self.stats['completed_requests']
        
        success_rate = 0.0
        if total_requests > 0:
            success_rate = (completed_requests - self.stats['failed_requests']) / total_requests * 100
        
        return {
            **self.stats,
            'success_rate_percent': success_rate,
            'queue_utilization_percent': (self.stats['current_queue_size'] / self.queue_size) * 100,
            'target_latency_ms': self.target_latency_ms,
            'latency_target_met': self.stats['average_latency_ms'] <= self.target_latency_ms,
            'active_workers': len(self.worker_futures),
            'max_workers': self.max_workers,
            'pending_results': len(self.results)
        }
    
    def batch_submit(
        self, 
        requests: List[Tuple[str, np.ndarray, Optional[np.ndarray]]]
    ) -> List[bool]:
        """
        Submit multiple requests at once.
        
        Args:
            requests: List of (request_id, echo_data, sensor_positions) tuples
            
        Returns:
            List of success flags for each request
        """
        results = []
        for request_id, echo_data, sensor_positions in requests:
            success = self.submit(request_id, echo_data, sensor_positions)
            results.append(success)
        
        return results
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all pending requests to complete.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if all requests completed, False if timeout
        """
        start_time = time.time()
        
        while self.stats['current_queue_size'] > 0:
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)
        
        return True


class BatchInferenceProcessor:
    """
    Optimized batch processing for high-throughput scenarios.
    """
    
    def __init__(self, model, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self.logger = logging.getLogger('BatchProcessor')
    
    def process_batch(
        self, 
        echo_data_batch: List[np.ndarray],
        sensor_positions_batch: Optional[List[np.ndarray]] = None
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Process a batch of echo data efficiently.
        
        Args:
            echo_data_batch: List of echo data arrays
            sensor_positions_batch: Optional list of sensor position arrays
            
        Returns:
            List of (position, confidence) tuples
        """
        results = []
        
        # Process in chunks if batch is too large
        for i in range(0, len(echo_data_batch), self.batch_size):
            chunk_echo = echo_data_batch[i:i+self.batch_size]
            chunk_sensors = None
            if sensor_positions_batch:
                chunk_sensors = sensor_positions_batch[i:i+self.batch_size]
            
            chunk_results = self._process_chunk(chunk_echo, chunk_sensors)
            results.extend(chunk_results)
        
        return results
    
    def _process_chunk(
        self, 
        echo_chunk: List[np.ndarray],
        sensor_chunk: Optional[List[np.ndarray]] = None
    ) -> List[Tuple[np.ndarray, float]]:
        """Process a single chunk of data."""
        results = []
        
        for i, echo_data in enumerate(echo_chunk):
            sensor_positions = sensor_chunk[i] if sensor_chunk else None
            
            try:
                position, confidence = self.model.predict_position(echo_data, sensor_positions)
                results.append((position, confidence))
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                results.append((np.array([0.0, 0.0, 0.0]), 0.0))
        
        return results