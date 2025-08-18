"""
Enhanced Performance Monitoring and Health Checking for EchoLoc-NN.

Provides comprehensive monitoring, alerting, and health checking
for quantum planning and ultrasonic localization systems.
"""

import time
import threading
# import psutil  # Optional dependency - handle gracefully\ntry:\n    import psutil\n    PSUTIL_AVAILABLE = True\nexcept ImportError:\n    PSUTIL_AVAILABLE = False
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
from datetime import datetime, timedelta
import torch
from .logging_config import get_logger

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """Enhanced system resource metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    gpu_utilization: float = 0.0
    gpu_memory_percent: float = 0.0

@dataclass
class QuantumPlanningMetrics:
    """Quantum planning specific metrics."""
    timestamp: float
    planning_time_ms: float
    optimization_energy: float
    convergence_iterations: int
    quantum_coherence: float
    position_confidence: float
    success_rate: float
    throughput_tasks_per_sec: float = 0.0

@dataclass
class Alert:
    """System alert notification."""
    id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None
    throughput_hz: Optional[float] = None
    queue_size: Optional[int] = None
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'gpu_memory_mb': self.gpu_memory_mb,
            'inference_time_ms': self.inference_time_ms,
            'throughput_hz': self.throughput_hz,
            'queue_size': self.queue_size,
            'error_count': self.error_count
        }


class PerformanceMonitor:
    """
    Real-time performance monitoring system.
    
    Tracks system resources, inference performance,
    and provides alerts for performance degradation.
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        sampling_interval: float = 1.0,
        enable_gpu_monitoring: bool = True
    ):
        self.max_history = max_history
        self.sampling_interval = sampling_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        
        # Metric storage
        self.metrics_history: deque = deque(maxlen=max_history)
        self.inference_times: deque = deque(maxlen=100)
        self.error_count = 0
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.is_monitoring = False
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_mb': 1024.0,  # 1GB
            'inference_time_ms': 100.0,
            'gpu_memory_mb': 1024.0
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        self.logger = get_logger('performance_monitor')
        
    def start_monitoring(self):
        """Start background performance monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.stop_monitoring.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.is_monitoring:
            return
            
        self.stop_monitoring.set()
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(self.sampling_interval):
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check thresholds and trigger alerts
                self._check_thresholds(metrics)
                
                # Log performance data
                self.logger.performance(
                    "Performance metrics collected",
                    extra=metrics.to_dict()
                )
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.error_count += 1
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.used / 1024 / 1024
        
        # GPU metrics
        gpu_memory_mb = None
        if self.enable_gpu_monitoring:
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            except:
                pass
        
        # Inference performance
        inference_time_ms = None
        throughput_hz = None
        
        if self.inference_times:
            recent_times = list(self.inference_times)
            inference_time_ms = np.mean(recent_times)
            throughput_hz = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            inference_time_ms=inference_time_ms,
            throughput_hz=throughput_hz,
            error_count=self.error_count
        )
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and trigger alerts."""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_mb > self.thresholds['memory_mb']:
            alerts.append(f"High memory usage: {metrics.memory_mb:.1f} MB")
        
        if (metrics.inference_time_ms is not None and 
            metrics.inference_time_ms > self.thresholds['inference_time_ms']):
            alerts.append(f"Slow inference: {metrics.inference_time_ms:.1f} ms")
        
        if (metrics.gpu_memory_mb is not None and 
            metrics.gpu_memory_mb > self.thresholds['gpu_memory_mb']):
            alerts.append(f"High GPU memory: {metrics.gpu_memory_mb:.1f} MB")
        
        # Trigger alerts
        for alert_msg in alerts:
            self.logger.warning(f"Performance alert: {alert_msg}")
            for callback in self.alert_callbacks:
                try:
                    callback(alert_msg, metrics)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def record_inference_time(self, duration_ms: float):
        """Record an inference time measurement."""
        self.inference_times.append(duration_ms)
    
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get performance summary over time window."""
        if not self.metrics_history:
            return {'no_data': True}
        
        # Filter recent metrics
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'no_data': True}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_mb for m in recent_metrics]
        
        inference_values = [
            m.inference_time_ms for m in recent_metrics 
            if m.inference_time_ms is not None
        ]
        
        summary = {
            'window_minutes': window_minutes,
            'sample_count': len(recent_metrics),
            'cpu_percent': {
                'mean': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values)
            },
            'memory_mb': {
                'mean': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values)
            },
            'total_errors': self.error_count
        }
        
        if inference_values:
            summary['inference_time_ms'] = {
                'mean': np.mean(inference_values),
                'p95': np.percentile(inference_values, 95),
                'max': np.max(inference_values)
            }
            summary['throughput_hz'] = 1000.0 / np.mean(inference_values)
        
        return summary
    
    def add_alert_callback(self, callback: Callable[[str, PerformanceMetrics], None]):
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, metric: str, value: float):
        """Set performance threshold."""
        if metric in self.thresholds:
            self.thresholds[metric] = value
            self.logger.info(f"Threshold updated: {metric} = {value}")
        else:
            raise ValueError(f"Unknown threshold metric: {metric}")


class HealthChecker:
    """
    System health checking and diagnostics.
    
    Performs comprehensive health checks on the EchoLoc-NN system
    including hardware, models, and data pipelines.
    """
    
    def __init__(self):
        self.logger = get_logger('health_checker')
        self.check_history: List[Dict[str, Any]] = []
        
    def run_health_check(self) -> Dict[str, Any]:
        """
        Run comprehensive health check.
        
        Returns:
            Health check results
        """
        self.logger.info("Starting health check")
        start_time = time.time()
        
        results = {
            'timestamp': start_time,
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # System health
        results['checks']['system'] = self._check_system_health()
        
        # GPU health (if available)
        if torch.cuda.is_available():
            results['checks']['gpu'] = self._check_gpu_health()
        
        # Memory health
        results['checks']['memory'] = self._check_memory_health()
        
        # Dependencies
        results['checks']['dependencies'] = self._check_dependencies()
        
        # Determine overall status
        failed_checks = [
            name for name, check in results['checks'].items() 
            if not check['passed']
        ]
        
        if failed_checks:
            results['overall_status'] = 'unhealthy'
            results['failed_checks'] = failed_checks
        
        duration = time.time() - start_time
        results['duration_ms'] = duration * 1000
        
        # Store in history
        self.check_history.append(results)
        if len(self.check_history) > 100:  # Keep last 100 checks
            self.check_history = self.check_history[-100:]
        
        self.logger.info(
            f"Health check completed in {duration:.2f}s - Status: {results['overall_status']}"
        )
        
        return results
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check basic system health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Health criteria
            cpu_ok = cpu_percent < 90
            memory_ok = memory.percent < 90
            disk_ok = disk.percent < 90
            
            return {
                'passed': cpu_ok and memory_ok and disk_ok,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'issues': [
                    issue for issue, ok in [
                        ('High CPU usage', not cpu_ok),
                        ('High memory usage', not memory_ok),
                        ('Low disk space', not disk_ok)
                    ] if not ok
                ]
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _check_gpu_health(self) -> Dict[str, Any]:
        """Check GPU health and availability."""
        try:
            if not torch.cuda.is_available():
                return {
                    'passed': False,
                    'error': 'CUDA not available'
                }
            
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            
            # Get GPU memory info
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            
            # Simple GPU test
            try:
                test_tensor = torch.randn(100, 100, device='cuda')
                _ = torch.mm(test_tensor, test_tensor)
                gpu_functional = True
            except Exception:
                gpu_functional = False
            
            return {
                'passed': gpu_functional,
                'device_count': device_count,
                'current_device': current_device,
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved,
                'functional': gpu_functional
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory health and fragmentation."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            
            # Check for memory leaks (simplified)
            current_usage = memory.used
            
            # PyTorch memory if GPU available
            torch_memory = {}
            if torch.cuda.is_available():
                torch_memory = {
                    'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                    'cached_gb': torch.cuda.memory_reserved() / 1024**3
                }
            
            memory_ok = memory.percent < 85
            
            return {
                'passed': memory_ok,
                'system_memory_percent': memory.percent,
                'system_memory_gb': memory.used / 1024**3,
                'torch_memory': torch_memory
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies."""
        try:
            dependencies = {}
            
            # Check PyTorch
            try:
                import torch
                dependencies['torch'] = {
                    'available': True,
                    'version': torch.__version__,
                    'cuda_available': torch.cuda.is_available()
                }
            except ImportError:
                dependencies['torch'] = {'available': False}
            
            # Check NumPy
            try:
                import numpy
                dependencies['numpy'] = {
                    'available': True,
                    'version': numpy.__version__
                }
            except ImportError:
                dependencies['numpy'] = {'available': False}
            
            # Check SciPy
            try:
                import scipy
                dependencies['scipy'] = {
                    'available': True,
                    'version': scipy.__version__
                }
            except ImportError:
                dependencies['scipy'] = {'available': False}
            
            # Check serial (for hardware)
            try:
                import serial
                dependencies['serial'] = {
                    'available': True,
                    'version': getattr(serial, '__version__', 'unknown')
                }
            except ImportError:
                dependencies['serial'] = {'available': False}
            
            # All critical dependencies available
            critical_deps = ['torch', 'numpy', 'scipy']
            all_available = all(
                dependencies[dep]['available'] 
                for dep in critical_deps
            )
            
            return {
                'passed': all_available,
                'dependencies': dependencies,
                'missing': [
                    dep for dep in critical_deps 
                    if not dependencies[dep]['available']
                ]
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def check_model_health(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Check model-specific health."""
        try:
            # Check model parameters
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Check for NaN parameters
            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            
            # Check parameter gradients (if available)
            has_gradients = any(p.grad is not None for p in model.parameters())
            
            # Simple forward pass test
            try:
                with torch.no_grad():
                    test_input = torch.randn(1, 4, 2048)  # Batch, sensors, samples
                    if next(model.parameters()).is_cuda:
                        test_input = test_input.cuda()
                    
                    output = model(test_input)
                    forward_pass_ok = True
                    output_shape = output[0].shape if isinstance(output, tuple) else output.shape
            except Exception as e:
                forward_pass_ok = False
                output_shape = None
            
            return {
                'passed': not has_nan and forward_pass_ok,
                'parameter_count': param_count,
                'trainable_parameters': trainable_params,
                'has_nan_parameters': has_nan,
                'has_gradients': has_gradients,
                'forward_pass_ok': forward_pass_ok,
                'output_shape': output_shape,
                'device': str(next(model.parameters()).device)
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent health check history."""
        return self.check_history[-limit:] if self.check_history else []

class EnhancedHealthChecker:
    """
    Enhanced health checker with quantum planning monitoring.
    
    Provides comprehensive health checking including:
    - System resources and performance
    - Quantum planning metrics
    - Ultrasonic localization accuracy
    - Hardware status and connectivity
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.logger = get_logger('enhanced_health_checker')
        
        # Component registry
        self.registered_components = {}
        
        # Metrics storage
        self.system_metrics_history = deque(maxlen=1000)
        self.quantum_metrics_history = deque(maxlen=1000)
        self.health_status = {}
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.alert_callbacks = []
        
        # Monitoring
        self.is_monitoring = False
        self.monitor_thread = None
        
    def register_quantum_component(self, name: str, health_check_func: Callable[[], Tuple[HealthStatus, str]]):
        """Register quantum planning component for health monitoring."""
        self.registered_components[name] = health_check_func
        self.logger.info(f"Registered quantum component: {name}")
        
    def record_quantum_metrics(self, 
                              planning_time_ms: float,
                              optimization_energy: float = 0.0,
                              convergence_iterations: int = 0,
                              quantum_coherence: float = 1.0,
                              position_confidence: float = 1.0,
                              success_rate: float = 1.0):
        """Record quantum planning performance metrics."""
        
        metrics = QuantumPlanningMetrics(
            timestamp=time.time(),
            planning_time_ms=planning_time_ms,
            optimization_energy=optimization_energy,
            convergence_iterations=convergence_iterations,
            quantum_coherence=quantum_coherence,
            position_confidence=position_confidence,
            success_rate=success_rate
        )
        
        self.quantum_metrics_history.append(metrics)
        
        # Check quantum-specific thresholds
        self._check_quantum_thresholds(metrics)
        
    def get_quantum_health_summary(self) -> Dict[str, Any]:
        """Get quantum planning health summary."""
        if not self.quantum_metrics_history:
            return {'no_data': True}
            
        recent_metrics = list(self.quantum_metrics_history)[-20:]  # Last 20 measurements
        
        return {
            'avg_planning_time_ms': np.mean([m.planning_time_ms for m in recent_metrics]),
            'avg_quantum_coherence': np.mean([m.quantum_coherence for m in recent_metrics]),
            'avg_position_confidence': np.mean([m.position_confidence for m in recent_metrics]),
            'avg_success_rate': np.mean([m.success_rate for m in recent_metrics]),
            'samples': len(recent_metrics)
        }
        
    def _check_quantum_thresholds(self, metrics: QuantumPlanningMetrics):
        """Check quantum planning metrics against thresholds."""
        
        # Planning time thresholds
        if metrics.planning_time_ms > 10000:  # 10 seconds
            self._create_alert(
                AlertLevel.CRITICAL, 'quantum_planning',
                f"Planning time critical: {metrics.planning_time_ms:.0f}ms"
            )
        elif metrics.planning_time_ms > 5000:  # 5 seconds
            self._create_alert(
                AlertLevel.WARNING, 'quantum_planning',
                f"Planning time high: {metrics.planning_time_ms:.0f}ms"
            )
            
        # Quantum coherence thresholds
        if metrics.quantum_coherence < 0.1:
            self._create_alert(
                AlertLevel.CRITICAL, 'quantum_coherence',
                f"Quantum coherence critical: {metrics.quantum_coherence:.3f}"
            )
        elif metrics.quantum_coherence < 0.3:
            self._create_alert(
                AlertLevel.WARNING, 'quantum_coherence',
                f"Quantum coherence low: {metrics.quantum_coherence:.3f}"
            )
            
        # Position confidence thresholds
        if metrics.position_confidence < 0.5:
            self._create_alert(
                AlertLevel.CRITICAL, 'position_confidence',
                f"Position confidence critical: {metrics.position_confidence:.3f}"
            )
        elif metrics.position_confidence < 0.7:
            self._create_alert(
                AlertLevel.WARNING, 'position_confidence',
                f"Position confidence low: {metrics.position_confidence:.3f}"
            )
            
    def _create_alert(self, level: AlertLevel, component: str, message: str, metadata: Optional[Dict] = None):
        """Create and manage alert."""
        
        alert_id = f"{component}_{level.value}_{int(time.time())}"
        
        # Check for duplicate alert
        existing_alerts = [a for a in self.active_alerts.values() 
                         if a.component == component and a.level == level]
        
        if existing_alerts:
            existing_alerts[0].timestamp = time.time()
            return
            
        alert = Alert(
            id=alert_id,
            level=level,
            component=component,
            message=message,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
                
        # Log alert
        log_level = getattr(self.logger, level.value.lower(), self.logger.info)
        log_level(f"[{component.upper()}] {message}")

# Global enhanced health checker
_global_enhanced_health_checker = EnhancedHealthChecker()

def get_global_enhanced_health_checker() -> EnhancedHealthChecker:
    """Get the global enhanced health checker instance."""
    return _global_enhanced_health_checker