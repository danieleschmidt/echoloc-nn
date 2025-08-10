"""
Generation 3 Performance Monitoring and Real-time Profiling System.

Provides comprehensive performance monitoring, profiling, and optimization
recommendations for the EchoLoc-NN system.
"""

import time
import threading
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from ..utils.logging_config import get_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    
    timestamp: float
    
    # Inference metrics
    inference_time_ms: float
    throughput_hz: float
    queue_length: int
    batch_size: int
    
    # System metrics
    cpu_percent: float
    memory_percent: float
    gpu_memory_mb: Optional[float]
    gpu_utilization_percent: Optional[float]
    
    # Model metrics
    model_size_mb: float
    parameters_count: int
    
    # Custom metrics
    custom_metrics: Dict[str, float]


@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str  # 'info', 'warning', 'critical'
    message: str
    callback: Optional[Callable] = None


class RealTimeProfiler:
    """
    Real-time performance profiler with hotspot detection.
    
    Tracks function-level performance and identifies bottlenecks
    in real-time during inference.
    """
    
    def __init__(self, enabled: bool = True, max_samples: int = 1000):
        self.enabled = enabled
        self.max_samples = max_samples
        self.logger = get_logger('profiler')
        
        # Profiling data
        self.function_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.active_calls: Dict[str, float] = {}  # function -> start_time
        
        # Threading
        self.profile_lock = threading.RLock()
        
    def start_call(self, function_name: str) -> str:
        """Start timing a function call."""
        if not self.enabled:
            return function_name
        
        call_id = f"{function_name}_{id(threading.current_thread())}"
        
        with self.profile_lock:
            self.active_calls[call_id] = time.perf_counter()
            
        return call_id
    
    def end_call(self, call_id: str) -> float:
        """End timing a function call and return duration."""
        if not self.enabled or call_id not in self.active_calls:
            return 0.0
        
        end_time = time.perf_counter()
        
        with self.profile_lock:
            start_time = self.active_calls.pop(call_id, end_time)
            duration = (end_time - start_time) * 1000  # Convert to ms
            
            function_name = call_id.split('_')[0]
            self.function_times[function_name].append(duration)
            self.call_counts[function_name] += 1
            
        return duration
    
    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """Get statistics for a specific function."""
        with self.profile_lock:
            if function_name not in self.function_times:
                return {'no_data': True}
            
            times = list(self.function_times[function_name])
            
            return {
                'function_name': function_name,
                'call_count': self.call_counts[function_name],
                'avg_time_ms': np.mean(times),
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times),
                'p95_time_ms': np.percentile(times, 95),
                'p99_time_ms': np.percentile(times, 99),
                'total_time_ms': np.sum(times),
                'samples': len(times)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all profiled functions."""
        with self.profile_lock:
            return {func: self.get_function_stats(func) 
                   for func in self.function_times.keys()}
    
    def get_hotspots(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top performance hotspots by total time."""
        all_stats = self.get_all_stats()
        
        # Sort by total time
        hotspots = []
        for func_name, stats in all_stats.items():
            if not stats.get('no_data', False):
                hotspots.append(stats)
        
        hotspots.sort(key=lambda x: x['total_time_ms'], reverse=True)
        return hotspots[:top_k]
    
    def clear_stats(self):
        """Clear all profiling statistics."""
        with self.profile_lock:
            self.function_times.clear()
            self.call_counts.clear()
            self.active_calls.clear()


class PerformanceMonitor:
    """
    Generation 3 comprehensive performance monitoring system.
    
    Features:
    - Real-time metrics collection and analysis
    - Performance trend analysis and prediction
    - Automated alert system with customizable thresholds
    - Resource usage optimization recommendations
    - Integration with auto-scaling system
    - Hardware-aware performance profiling
    """
    
    def __init__(
        self,
        collection_interval: float = 1.0,
        history_size: int = 3600,  # 1 hour at 1s intervals
        enable_profiling: bool = True,
        enable_gpu_monitoring: bool = True
    ):
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_profiling = enable_profiling
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        
        self.logger = get_logger('performance_monitor')
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_metrics: Optional[PerformanceMetrics] = None
        
        # Real-time profiler
        self.profiler = RealTimeProfiler(enabled=enable_profiling)
        
        # Alert system
        self.alerts: List[PerformanceAlert] = []
        self.triggered_alerts: Dict[str, float] = {}  # alert_id -> last_trigger_time
        
        # Control
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="perf_monitor")
        
        # Performance analysis
        self.analysis_window = 300  # 5 minutes for trend analysis
        
        # Initialize GPU monitoring if available
        if self.enable_gpu_monitoring:
            self._init_gpu_monitoring()
        
        self.logger.info("Performance monitor initialized")
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring capabilities."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.logger.info("GPU monitoring enabled")
        except ImportError:
            self.logger.warning("pynvml not available, limited GPU monitoring")
            self.gpu_handle = None
        except Exception as e:
            self.logger.warning(f"GPU monitoring initialization failed: {e}")
            self.enable_gpu_monitoring = False
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        
        self.logger.info("Starting performance monitoring")
        self.is_monitoring = True
        self.stop_event.clear()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="performance_monitor"
        )
        self.monitor_thread.start()
        
        self.logger.info(f"Performance monitoring started (interval: {self.collection_interval}s)")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping performance monitoring")
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=False)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.wait(self.collection_interval):
            try:
                self._collect_metrics()
                self._check_alerts()
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self):
        """Collect current performance metrics."""
        try:
            timestamp = time.time()
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU metrics
            gpu_memory_mb = None
            gpu_utilization = None
            
            if self.enable_gpu_monitoring and torch.cuda.is_available():
                try:
                    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    
                    if hasattr(self, 'gpu_handle') and self.gpu_handle:
                        import pynvml
                        gpu_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        gpu_utilization = gpu_info.gpu
                except Exception as e:
                    self.logger.debug(f"GPU metrics collection failed: {e}")
            
            # Create metrics snapshot
            metrics = PerformanceMetrics(
                timestamp=timestamp,
                inference_time_ms=0.0,  # Will be updated by inference calls
                throughput_hz=0.0,      # Will be calculated
                queue_length=0,         # Will be updated by processor
                batch_size=1,           # Will be updated by inference
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_memory_mb=gpu_memory_mb,
                gpu_utilization_percent=gpu_utilization,
                model_size_mb=0.0,      # Will be updated when model is loaded
                parameters_count=0,     # Will be updated when model is loaded
                custom_metrics={}
            )
            
            # Store metrics
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    def update_inference_metrics(
        self,
        inference_time_ms: float,
        batch_size: int = 1,
        queue_length: int = 0,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """Update inference-specific metrics."""
        if not self.current_metrics:
            return
        
        # Calculate throughput
        throughput_hz = (batch_size * 1000) / inference_time_ms if inference_time_ms > 0 else 0.0
        
        # Update current metrics
        self.current_metrics.inference_time_ms = inference_time_ms
        self.current_metrics.throughput_hz = throughput_hz
        self.current_metrics.batch_size = batch_size
        self.current_metrics.queue_length = queue_length
        
        if custom_metrics:
            self.current_metrics.custom_metrics.update(custom_metrics)
    
    def update_model_metrics(self, model_size_mb: float, parameters_count: int):
        """Update model-specific metrics."""
        if not self.current_metrics:
            return
        
        self.current_metrics.model_size_mb = model_size_mb
        self.current_metrics.parameters_count = parameters_count
    
    def add_alert(self, alert: PerformanceAlert):
        """Add a performance alert."""
        self.alerts.append(alert)
        self.logger.info(f"Added performance alert: {alert.metric_name} {alert.comparison} {alert.threshold}")
    
    def remove_alert(self, metric_name: str):
        """Remove alerts for a specific metric."""
        self.alerts = [alert for alert in self.alerts if alert.metric_name != metric_name]
    
    def _check_alerts(self):
        """Check all alerts against current metrics."""
        if not self.current_metrics:
            return
        
        current_time = time.time()
        
        for alert in self.alerts:
            alert_id = f"{alert.metric_name}_{alert.comparison}_{alert.threshold}"
            
            # Get metric value
            metric_value = self._get_metric_value(alert.metric_name)
            if metric_value is None:
                continue
            
            # Check condition
            triggered = False
            if alert.comparison == 'gt' and metric_value > alert.threshold:
                triggered = True
            elif alert.comparison == 'lt' and metric_value < alert.threshold:
                triggered = True
            elif alert.comparison == 'eq' and abs(metric_value - alert.threshold) < 1e-6:
                triggered = True
            
            if triggered:
                # Check if alert was recently triggered (avoid spam)
                last_trigger = self.triggered_alerts.get(alert_id, 0)
                if current_time - last_trigger > 60:  # 1 minute cooldown
                    self._trigger_alert(alert, metric_value)
                    self.triggered_alerts[alert_id] = current_time
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric."""
        if not self.current_metrics:
            return None
        
        # Map metric names to values
        metric_map = {
            'inference_time_ms': self.current_metrics.inference_time_ms,
            'throughput_hz': self.current_metrics.throughput_hz,
            'cpu_percent': self.current_metrics.cpu_percent,
            'memory_percent': self.current_metrics.memory_percent,
            'gpu_memory_mb': self.current_metrics.gpu_memory_mb,
            'gpu_utilization_percent': self.current_metrics.gpu_utilization_percent,
            'queue_length': float(self.current_metrics.queue_length),
            'batch_size': float(self.current_metrics.batch_size)
        }
        
        # Check custom metrics
        if metric_name in self.current_metrics.custom_metrics:
            return self.current_metrics.custom_metrics[metric_name]
        
        return metric_map.get(metric_name)
    
    def _trigger_alert(self, alert: PerformanceAlert, metric_value: float):
        """Trigger a performance alert."""
        message = alert.message.format(metric_value=metric_value, threshold=alert.threshold)
        
        if alert.severity == 'critical':
            self.logger.critical(message)
        elif alert.severity == 'warning':
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        # Call callback if provided
        if alert.callback:
            try:
                alert.callback(alert, metric_value)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics."""
        return self.current_metrics
    
    def get_metrics_history(self, duration_seconds: Optional[int] = None) -> List[PerformanceMetrics]:
        """Get metrics history."""
        if duration_seconds is None:
            return list(self.metrics_history)
        
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_performance_summary(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Get performance summary over specified duration."""
        history = self.get_metrics_history(duration_seconds)
        
        if not history:
            return {'no_data': True}
        
        # Extract metrics
        inference_times = [m.inference_time_ms for m in history if m.inference_time_ms > 0]
        throughputs = [m.throughput_hz for m in history if m.throughput_hz > 0]
        cpu_usage = [m.cpu_percent for m in history]
        memory_usage = [m.memory_percent for m in history]
        
        summary = {
            'duration_seconds': duration_seconds,
            'samples': len(history),
            'timestamp_range': {
                'start': history[0].timestamp if history else 0,
                'end': history[-1].timestamp if history else 0
            }
        }
        
        # Inference metrics
        if inference_times:
            summary['inference'] = {
                'avg_time_ms': np.mean(inference_times),
                'min_time_ms': np.min(inference_times),
                'max_time_ms': np.max(inference_times),
                'p95_time_ms': np.percentile(inference_times, 95),
                'p99_time_ms': np.percentile(inference_times, 99),
                'std_time_ms': np.std(inference_times),
                'meets_50ms_target': np.mean(np.array(inference_times) < 50.0) * 100
            }
        
        # Throughput metrics
        if throughputs:
            summary['throughput'] = {
                'avg_hz': np.mean(throughputs),
                'max_hz': np.max(throughputs),
                'total_inferences': len(inference_times)
            }
        
        # System metrics
        summary['system'] = {
            'avg_cpu_percent': np.mean(cpu_usage),
            'max_cpu_percent': np.max(cpu_usage),
            'avg_memory_percent': np.mean(memory_usage),
            'max_memory_percent': np.max(memory_usage)
        }
        
        # GPU metrics if available
        gpu_memory = [m.gpu_memory_mb for m in history if m.gpu_memory_mb is not None]
        gpu_util = [m.gpu_utilization_percent for m in history if m.gpu_utilization_percent is not None]
        
        if gpu_memory:
            summary['gpu'] = {
                'avg_memory_mb': np.mean(gpu_memory),
                'max_memory_mb': np.max(gpu_memory),
                'avg_utilization_percent': np.mean(gpu_util) if gpu_util else None
            }
        
        return summary
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends."""
        if len(self.metrics_history) < self.analysis_window:
            return {'insufficient_data': True}
        
        recent_history = list(self.metrics_history)[-self.analysis_window:]
        older_history = list(self.metrics_history)[-(2*self.analysis_window):-self.analysis_window]
        
        if len(older_history) < self.analysis_window // 2:
            return {'insufficient_data': True}
        
        # Compare recent vs older performance
        recent_inference = [m.inference_time_ms for m in recent_history if m.inference_time_ms > 0]
        older_inference = [m.inference_time_ms for m in older_history if m.inference_time_ms > 0]
        
        trends = {}
        
        if recent_inference and older_inference:
            recent_avg = np.mean(recent_inference)
            older_avg = np.mean(older_inference)
            
            trends['inference_time'] = {
                'recent_avg_ms': recent_avg,
                'older_avg_ms': older_avg,
                'change_percent': ((recent_avg - older_avg) / older_avg) * 100,
                'trend': 'improving' if recent_avg < older_avg else 'degrading'
            }
        
        # CPU trend
        recent_cpu = [m.cpu_percent for m in recent_history]
        older_cpu = [m.cpu_percent for m in older_history]
        
        if recent_cpu and older_cpu:
            recent_cpu_avg = np.mean(recent_cpu)
            older_cpu_avg = np.mean(older_cpu)
            
            trends['cpu_usage'] = {
                'recent_avg_percent': recent_cpu_avg,
                'older_avg_percent': older_cpu_avg,
                'change_percent': ((recent_cpu_avg - older_cpu_avg) / older_cpu_avg) * 100,
                'trend': 'increasing' if recent_cpu_avg > older_cpu_avg else 'decreasing'
            }
        
        return trends
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if not self.current_metrics or len(self.metrics_history) < 10:
            return recommendations
        
        recent_history = list(self.metrics_history)[-60:]  # Last minute
        
        # High inference time recommendation
        inference_times = [m.inference_time_ms for m in recent_history if m.inference_time_ms > 0]
        if inference_times:
            avg_inference = np.mean(inference_times)
            
            if avg_inference > 100:  # > 100ms
                recommendations.append({
                    'category': 'inference_optimization',
                    'priority': 'high',
                    'issue': f'High inference time: {avg_inference:.1f}ms',
                    'recommendation': 'Consider model quantization, pruning, or TensorRT optimization',
                    'expected_improvement': '2-10x speedup'
                })
            elif avg_inference > 50:  # > 50ms
                recommendations.append({
                    'category': 'inference_optimization',
                    'priority': 'medium',
                    'issue': f'Inference time above 50ms target: {avg_inference:.1f}ms',
                    'recommendation': 'Apply batch processing and GPU acceleration',
                    'expected_improvement': '1.5-3x speedup'
                })
        
        # High CPU usage recommendation
        cpu_usage = [m.cpu_percent for m in recent_history]
        if cpu_usage:
            avg_cpu = np.mean(cpu_usage)
            
            if avg_cpu > 80:
                recommendations.append({
                    'category': 'resource_optimization',
                    'priority': 'high',
                    'issue': f'High CPU usage: {avg_cpu:.1f}%',
                    'recommendation': 'Enable auto-scaling or add more workers',
                    'expected_improvement': 'Reduced latency and better throughput'
                })
        
        # Memory usage recommendation
        memory_usage = [m.memory_percent for m in recent_history]
        if memory_usage:
            avg_memory = np.mean(memory_usage)
            
            if avg_memory > 85:
                recommendations.append({
                    'category': 'memory_optimization',
                    'priority': 'high',
                    'issue': f'High memory usage: {avg_memory:.1f}%',
                    'recommendation': 'Enable caching compression or reduce batch sizes',
                    'expected_improvement': 'Reduced memory pressure and OOM risk'
                })
        
        # GPU underutilization
        if self.enable_gpu_monitoring:
            gpu_util = [m.gpu_utilization_percent for m in recent_history if m.gpu_utilization_percent is not None]
            if gpu_util and np.mean(gpu_util) < 30:
                recommendations.append({
                    'category': 'gpu_optimization',
                    'priority': 'medium',
                    'issue': f'Low GPU utilization: {np.mean(gpu_util):.1f}%',
                    'recommendation': 'Increase batch sizes or enable more parallel processing',
                    'expected_improvement': 'Better hardware utilization'
                })
        
        return recommendations
    
    def profile_function(self, function_name: str):
        """Decorator for profiling function calls."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.enable_profiling:
                    return func(*args, **kwargs)
                
                call_id = self.profiler.start_call(function_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.profiler.end_call(call_id)
            return wrapper
        return decorator
    
    def get_profiling_report(self) -> Dict[str, Any]:
        """Get comprehensive profiling report."""
        if not self.enable_profiling:
            return {'profiling_disabled': True}
        
        hotspots = self.profiler.get_hotspots(10)
        all_stats = self.profiler.get_all_stats()
        
        return {
            'profiling_enabled': True,
            'total_functions_profiled': len(all_stats),
            'hotspots': hotspots,
            'all_function_stats': all_stats,
            'total_profiled_calls': sum(stats.get('call_count', 0) for stats in all_stats.values()),
            'total_profiled_time_ms': sum(stats.get('total_time_ms', 0) for stats in all_stats.values())
        }
    
    def export_metrics(self, format: str = 'json', duration_seconds: Optional[int] = None) -> str:
        """Export metrics in specified format."""
        history = self.get_metrics_history(duration_seconds)
        
        if format == 'json':
            import json
            data = {
                'export_timestamp': time.time(),
                'duration_seconds': duration_seconds,
                'metrics': [asdict(m) for m in history],
                'summary': self.get_performance_summary(duration_seconds or 3600)
            }
            return json.dumps(data, indent=2)
        
        elif format == 'csv':
            if not history:
                return "No data available"
            
            # Create CSV header
            fieldnames = list(asdict(history[0]).keys())
            csv_lines = [','.join(fieldnames)]
            
            # Add data rows
            for metric in history:
                row_data = []
                for field in fieldnames:
                    value = getattr(metric, field)
                    if isinstance(value, dict):
                        value = str(value)
                    row_data.append(str(value))
                csv_lines.append(','.join(row_data))
            
            return '\n'.join(csv_lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if self.is_monitoring:
                self.stop_monitoring()
        except:
            pass


# Global performance monitor instance
_global_performance_monitor = None

def get_global_performance_monitor(**kwargs) -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor(**kwargs)
    return _global_performance_monitor