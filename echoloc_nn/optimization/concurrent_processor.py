"""
Concurrent processing and resource pooling for scalable inference.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple, Iterator
from queue import Queue, Empty, Full
from dataclasses import dataclass
import numpy as np
import torch
from ..utils.logging_config import get_logger
from ..utils.exceptions import ResourceError, TimeoutError


@dataclass
class ProcessingJob:
    """Job for concurrent processing."""
    
    job_id: str
    echo_data: np.ndarray
    sensor_positions: Optional[np.ndarray]
    priority: int = 1  # Higher number = higher priority
    timeout: float = 10.0  # Job timeout in seconds
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority > other.priority


@dataclass
class ProcessingResult:
    """Result from concurrent processing."""
    
    job_id: str
    position: np.ndarray
    confidence: float
    processing_time_ms: float
    worker_id: str
    success: bool = True
    error: Optional[str] = None


class WorkerProcess:
    """Individual worker process for model inference."""
    
    def __init__(
        self,
        worker_id: str,
        model_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.worker_id = worker_id
        self.model_path = model_path
        self.device = device
        self.model = None
        self.logger = get_logger(f'worker_{worker_id}')
        
    def initialize(self):
        """Initialize worker (load model, etc.)."""
        try:
            if self.model_path:
                # Load model in worker process
                from ..models.hybrid_architecture import EchoLocModel
                self.model = EchoLocModel.load_model(self.model_path, self.device)
            else:
                # Create default model
                from ..models.hybrid_architecture import EchoLocModel
                self.model = EchoLocModel(n_sensors=4, model_size="base")
                
            self.model.eval()
            self.logger.info(f"Worker {self.worker_id} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} initialization failed: {e}")
            return False
    
    def process_job(self, job: ProcessingJob) -> ProcessingResult:
        """Process a single job."""
        start_time = time.time()
        
        try:
            if self.model is None:
                raise RuntimeError("Worker not initialized")
            
            # Perform inference
            position, confidence = self.model.predict_position(
                job.echo_data,
                job.sensor_positions
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            result = ProcessingResult(
                job_id=job.job_id,
                position=position,
                confidence=confidence,
                processing_time_ms=processing_time,
                worker_id=self.worker_id,
                success=True
            )
            
            # Call callback if provided
            if job.callback:
                job.callback(result)
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.error(f"Job {job.job_id} failed: {e}")
            
            return ProcessingResult(
                job_id=job.job_id,
                position=np.array([0.0, 0.0, 0.0]),
                confidence=0.0,
                processing_time_ms=processing_time,
                worker_id=self.worker_id,
                success=False,
                error=str(e)
            )


class ProcessorPool:
    """
    Pool of worker processes for concurrent model inference.
    
    Manages multiple worker processes with job queuing, load balancing,
    and automatic scaling capabilities.
    """
    
    def __init__(
        self,
        num_workers: int = None,
        model_path: Optional[str] = None,
        device: str = "cpu",
        max_queue_size: int = 1000,
        worker_timeout: float = 30.0
    ):
        self.num_workers = num_workers or min(4, mp.cpu_count())
        self.model_path = model_path
        self.device = device
        self.max_queue_size = max_queue_size
        self.worker_timeout = worker_timeout
        
        # Job management
        self.job_queue: Queue = Queue(maxsize=max_queue_size)
        self.result_queue: Queue = Queue()
        
        # Worker management
        self.workers: List[mp.Process] = []
        self.worker_stats: Dict[str, Dict] = {}
        
        # Control
        self.is_running = False
        self.stop_event = mp.Event()
        
        self.logger = get_logger('processor_pool')
        
    def start(self):
        """Start the processor pool."""
        if self.is_running:
            return
        
        self.logger.info(f"Starting processor pool with {self.num_workers} workers")
        
        # Start worker processes
        for i in range(self.num_workers):
            worker_id = f"worker_{i}"
            worker_process = mp.Process(
                target=self._worker_loop,
                args=(worker_id,),
                daemon=True
            )
            worker_process.start()
            self.workers.append(worker_process)
            
            # Initialize worker stats
            self.worker_stats[worker_id] = {
                'jobs_processed': 0,
                'total_processing_time': 0.0,
                'errors': 0,
                'last_job_time': 0.0
            }
        
        self.is_running = True
        self.logger.info("Processor pool started")
    
    def stop(self, timeout: float = 10.0):
        """Stop the processor pool."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping processor pool")
        
        # Signal workers to stop
        self.stop_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
            if worker.is_alive():
                self.logger.warning(f"Force terminating worker {worker.pid}")
                worker.terminate()
                worker.join()
        
        self.workers.clear()
        self.is_running = False
        self.logger.info("Processor pool stopped")
    
    def submit_job(
        self,
        job_id: str,
        echo_data: np.ndarray,
        sensor_positions: Optional[np.ndarray] = None,
        priority: int = 1,
        timeout: float = 10.0,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Submit job for processing.
        
        Args:
            job_id: Unique job identifier
            echo_data: Echo data to process
            sensor_positions: Sensor positions
            priority: Job priority (higher = more important)
            timeout: Job timeout
            callback: Optional callback for result
            
        Returns:
            True if job was queued successfully
        """
        if not self.is_running:
            raise RuntimeError("Processor pool not running")
        
        job = ProcessingJob(
            job_id=job_id,
            echo_data=echo_data,
            sensor_positions=sensor_positions,
            priority=priority,
            timeout=timeout,
            callback=callback
        )
        
        try:
            self.job_queue.put(job, timeout=1.0)
            self.logger.debug(f"Job {job_id} queued")
            return True
        except Full:
            self.logger.warning(f"Job queue full, rejecting job {job_id}")
            return False
    
    def get_result(self, timeout: float = None) -> Optional[ProcessingResult]:
        """Get next available result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_results(self, max_results: int = 10, timeout: float = 1.0) -> List[ProcessingResult]:
        """Get multiple results."""
        results = []
        end_time = time.time() + timeout
        
        while len(results) < max_results and time.time() < end_time:
            remaining_time = end_time - time.time()
            result = self.get_result(timeout=remaining_time)
            if result:
                results.append(result)
            else:
                break
        
        return results
    
    def _worker_loop(self, worker_id: str):
        """Main loop for worker process."""
        # Initialize worker
        worker = WorkerProcess(worker_id, self.model_path, self.device)
        if not worker.initialize():
            return
        
        logger = get_logger(f'worker_{worker_id}')
        logger.info(f"Worker {worker_id} started")
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Get job from queue
                    job = self.job_queue.get(timeout=1.0)
                    
                    # Check job timeout
                    if time.time() - job.timestamp > job.timeout:
                        logger.warning(f"Job {job.job_id} timed out before processing")
                        continue
                    
                    # Process job
                    result = worker.process_job(job)
                    
                    # Put result in result queue
                    try:
                        self.result_queue.put(result, timeout=1.0)
                    except Full:
                        logger.warning(f"Result queue full, dropping result {result.job_id}")
                    
                    # Update stats
                    self._update_worker_stats(worker_id, result)
                    
                except Empty:
                    continue  # No job available, keep looping
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    
        except KeyboardInterrupt:
            pass
        finally:
            logger.info(f"Worker {worker_id} stopped")
    
    def _update_worker_stats(self, worker_id: str, result: ProcessingResult):
        """Update worker statistics."""
        if worker_id in self.worker_stats:
            stats = self.worker_stats[worker_id]
            stats['jobs_processed'] += 1
            stats['total_processing_time'] += result.processing_time_ms
            stats['last_job_time'] = time.time()
            
            if not result.success:
                stats['errors'] += 1
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get processor pool statistics."""
        if not self.is_running:
            return {'status': 'stopped'}
        
        total_jobs = sum(stats['jobs_processed'] for stats in self.worker_stats.values())
        total_errors = sum(stats['errors'] for stats in self.worker_stats.values())
        
        avg_processing_time = 0.0
        if total_jobs > 0:
            total_time = sum(stats['total_processing_time'] for stats in self.worker_stats.values())
            avg_processing_time = total_time / total_jobs
        
        return {
            'status': 'running',
            'num_workers': len(self.workers),
            'queue_size': self.job_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'total_jobs_processed': total_jobs,
            'total_errors': total_errors,
            'error_rate': total_errors / total_jobs if total_jobs > 0 else 0.0,
            'avg_processing_time_ms': avg_processing_time,
            'worker_stats': self.worker_stats.copy()
        }


class BatchProcessor:
    """
    Batch processor for efficient processing of multiple echo samples.
    
    Groups individual requests into batches for improved throughput
    and GPU utilization.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 32,
        batch_timeout: float = 0.1  # Max time to wait for batch to fill
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        if model_path:
            from ..models.hybrid_architecture import EchoLocModel
            self.model = EchoLocModel.load_model(model_path, str(self.device))
        else:
            from ..models.hybrid_architecture import EchoLocModel
            self.model = EchoLocModel(n_sensors=4, model_size="base")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Batch management
        self.pending_jobs: List[Tuple[ProcessingJob, float]] = []  # (job, timestamp)
        self.batch_lock = threading.Lock()
        
        # Background batch processor
        self.batch_thread = threading.Thread(target=self._batch_processor_loop, daemon=True)
        self.stop_batching = threading.Event()
        self.batch_thread.start()
        
        self.logger = get_logger('batch_processor')
        self.logger.info(f"Batch processor initialized on {self.device}")
    
    def submit_job(
        self,
        job_id: str,
        echo_data: np.ndarray,
        sensor_positions: Optional[np.ndarray] = None,
        callback: Optional[Callable] = None
    ) -> bool:
        """Submit job for batch processing."""
        job = ProcessingJob(
            job_id=job_id,
            echo_data=echo_data,
            sensor_positions=sensor_positions,
            callback=callback
        )
        
        with self.batch_lock:
            self.pending_jobs.append((job, time.time()))
            
            # If batch is full, process immediately
            if len(self.pending_jobs) >= self.batch_size:
                self._process_batch()
                
        return True
    
    def _batch_processor_loop(self):
        """Background loop for processing batches."""
        while not self.stop_batching.wait(self.batch_timeout):
            with self.batch_lock:
                if self.pending_jobs:
                    # Check if oldest job has timed out
                    oldest_timestamp = self.pending_jobs[0][1]
                    if time.time() - oldest_timestamp >= self.batch_timeout:
                        self._process_batch()
    
    def _process_batch(self):
        """Process current batch of jobs."""
        if not self.pending_jobs:
            return
        
        batch_jobs = self.pending_jobs.copy()
        self.pending_jobs.clear()
        
        try:
            self._process_batch_internal(batch_jobs)
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            
            # Send error results
            for job, _ in batch_jobs:
                result = ProcessingResult(
                    job_id=job.job_id,
                    position=np.array([0.0, 0.0, 0.0]),
                    confidence=0.0,
                    processing_time_ms=0.0,
                    worker_id="batch_processor",
                    success=False,
                    error=str(e)
                )
                
                if job.callback:
                    job.callback(result)
    
    def _process_batch_internal(self, batch_jobs: List[Tuple[ProcessingJob, float]]):
        """Internal batch processing implementation."""
        start_time = time.time()
        
        # Prepare batch data
        echo_batch = []
        sensor_batch = []
        
        for job, _ in batch_jobs:
            echo_batch.append(job.echo_data)
            if job.sensor_positions is not None:
                sensor_batch.append(job.sensor_positions)
        
        # Convert to tensors
        echo_tensor = torch.from_numpy(np.array(echo_batch)).float().to(self.device)
        
        sensor_tensor = None
        if sensor_batch:
            sensor_tensor = torch.from_numpy(np.array(sensor_batch)).float().to(self.device)
        
        # Batch inference
        with torch.no_grad():
            positions, confidences = self.model(echo_tensor, sensor_tensor)
            
            # Convert back to numpy
            positions_np = positions.cpu().numpy()
            confidences_np = confidences.cpu().numpy()
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create results and call callbacks
        for i, (job, _) in enumerate(batch_jobs):
            result = ProcessingResult(
                job_id=job.job_id,
                position=positions_np[i],
                confidence=float(confidences_np[i]),
                processing_time_ms=processing_time / len(batch_jobs),  # Amortized time
                worker_id="batch_processor",
                success=True
            )
            
            if job.callback:
                job.callback(result)
        
        self.logger.debug(f"Processed batch of {len(batch_jobs)} jobs in {processing_time:.1f}ms")
    
    def stop(self):
        """Stop batch processor."""
        self.stop_batching.set()
        self.batch_thread.join()
        
        # Process any remaining jobs
        with self.batch_lock:
            if self.pending_jobs:
                self._process_batch()


class ConcurrentProcessor:
    """
    High-level concurrent processing coordinator.
    
    Manages multiple processing strategies (pool, batch) and
    provides intelligent job routing based on load and requirements.
    """
    
    def __init__(
        self,
        pool_workers: int = 4,
        enable_batching: bool = True,
        batch_size: int = 32,
        model_path: Optional[str] = None,
        device: str = "auto"
    ):
        self.pool_workers = pool_workers
        self.enable_batching = enable_batching
        
        # Initialize processor pool
        self.processor_pool = ProcessorPool(
            num_workers=pool_workers,
            model_path=model_path,
            device=device
        )
        
        # Initialize batch processor if enabled
        self.batch_processor = None
        if enable_batching:
            self.batch_processor = BatchProcessor(
                model_path=model_path,
                device=device,
                batch_size=batch_size
            )
        
        # Job routing
        self.job_counter = 0
        self.routing_lock = threading.Lock()
        
        # Result management
        self.active_jobs: Dict[str, Dict] = {}
        self.result_handlers: Dict[str, Callable] = {}
        
        self.logger = get_logger('concurrent_processor')
    
    def start(self):
        """Start concurrent processor."""
        self.processor_pool.start()
        self.logger.info("Concurrent processor started")
    
    def stop(self):
        """Stop concurrent processor."""
        self.processor_pool.stop()
        if self.batch_processor:
            self.batch_processor.stop()
        self.logger.info("Concurrent processor stopped")
    
    def submit_job(
        self,
        echo_data: np.ndarray,
        sensor_positions: Optional[np.ndarray] = None,
        priority: int = 1,
        prefer_batching: bool = None,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Submit job with intelligent routing.
        
        Args:
            echo_data: Echo data to process
            sensor_positions: Sensor positions
            priority: Job priority
            prefer_batching: Force batching preference
            callback: Result callback
            
        Returns:
            Job ID
        """
        with self.routing_lock:
            self.job_counter += 1
            job_id = f"job_{self.job_counter}"
        
        # Decide routing strategy
        use_batching = self._should_use_batching(priority, prefer_batching)
        
        # Store job info
        self.active_jobs[job_id] = {
            'timestamp': time.time(),
            'strategy': 'batch' if use_batching else 'pool',
            'priority': priority
        }
        
        if callback:
            self.result_handlers[job_id] = callback
        
        # Route job
        if use_batching and self.batch_processor:
            success = self.batch_processor.submit_job(
                job_id, echo_data, sensor_positions,
                callback=self._handle_result
            )
        else:
            success = self.processor_pool.submit_job(
                job_id, echo_data, sensor_positions, priority,
                callback=self._handle_result
            )
        
        if not success:
            # Clean up if submission failed
            self.active_jobs.pop(job_id, None)
            self.result_handlers.pop(job_id, None)
            raise ResourceError(f"Failed to submit job {job_id}")
        
        self.logger.debug(f"Job {job_id} submitted via {'batch' if use_batching else 'pool'}")
        return job_id
    
    def _should_use_batching(self, priority: int, prefer_batching: Optional[bool]) -> bool:
        """Decide whether to use batching for a job."""
        if prefer_batching is not None:
            return prefer_batching and self.enable_batching
        
        if not self.enable_batching:
            return False
        
        # High priority jobs bypass batching for lower latency
        if priority > 5:
            return False
        
        # Use batching for normal priority jobs
        return True
    
    def _handle_result(self, result: ProcessingResult):
        """Handle processing result."""
        job_id = result.job_id
        
        # Remove from active jobs
        job_info = self.active_jobs.pop(job_id, None)
        
        # Call user callback if registered
        callback = self.result_handlers.pop(job_id, None)
        if callback:
            callback(result)
        
        # Log result
        if job_info:
            total_time = (time.time() - job_info['timestamp']) * 1000
            self.logger.debug(
                f"Job {job_id} completed: {result.processing_time_ms:.1f}ms processing, "
                f"{total_time:.1f}ms total"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        pool_stats = self.processor_pool.get_pool_stats()
        
        stats = {
            'active_jobs': len(self.active_jobs),
            'pool_stats': pool_stats,
            'batching_enabled': self.enable_batching
        }
        
        if self.batch_processor:
            batch_stats = {
                'pending_jobs': len(self.batch_processor.pending_jobs),
                'device': str(self.batch_processor.device)
            }
            stats['batch_stats'] = batch_stats
        
        return stats
    
    def wait_for_job(self, job_id: str, timeout: float = 10.0) -> Optional[ProcessingResult]:
        """Wait for specific job to complete (blocking)."""
        if job_id not in self.active_jobs:
            return None
        
        # This is a simplified implementation
        # In production, you'd want proper event-based waiting
        start_time = time.time()
        while time.time() - start_time < timeout:
            if job_id not in self.active_jobs:
                # Job completed, but we don't have the result here
                # In a full implementation, you'd store results temporarily
                break
            time.sleep(0.01)
        
        return None  # Simplified - would return actual result