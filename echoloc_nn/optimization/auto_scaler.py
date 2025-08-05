"""
Enhanced Auto-Scaling with Quantum Planning Integration

Extends the existing auto-scaler with quantum-aware scaling decisions
and intelligent resource prediction for optimal performance.
"""

import time
import threading
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
import numpy as np
from ..utils.logging_config import get_logger
from ..utils.monitoring import PerformanceMonitor


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling system."""
    
    # Scaling thresholds
    cpu_scale_up_threshold: float = 70.0  # CPU % to scale up
    cpu_scale_down_threshold: float = 30.0  # CPU % to scale down
    memory_scale_up_threshold: float = 80.0  # Memory % to scale up
    queue_scale_up_threshold: int = 50  # Queue size to scale up
    
    # Scaling parameters
    min_workers: int = 1
    max_workers: int = 8
    scale_up_factor: float = 1.5  # Multiply workers by this when scaling up
    scale_down_factor: float = 0.7  # Multiply workers by this when scaling down
    
    # Timing
    evaluation_interval: float = 30.0  # Seconds between scaling decisions
    stabilization_time: float = 60.0  # Seconds to wait after scaling
    metric_window: int = 5  # Number of measurements to consider
    
    # Safety
    max_scale_events_per_hour: int = 10
    emergency_scale_threshold: float = 95.0  # Emergency scaling threshold


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    
    timestamp: float
    event_type: str  # 'scale_up', 'scale_down', 'emergency'
    old_workers: int
    new_workers: int
    trigger_metric: str
    trigger_value: float
    reason: str


class ResourceMonitor:
    """
    Monitor system resources for auto-scaling decisions.
    
    Tracks CPU, memory, queue sizes, and processing metrics
    to make intelligent scaling decisions.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        
        # Metric history
        self.cpu_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.queue_size_history = deque(maxlen=window_size)
        self.processing_time_history = deque(maxlen=window_size)
        self.throughput_history = deque(maxlen=window_size)
        
        # Monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.is_monitoring = False
        
        self.logger = get_logger('resource_monitor')
    
    def start_monitoring(self, interval: float = 5.0):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.stop_monitoring.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.is_monitoring:
            return
        
        self.stop_monitoring.set()
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(interval):
            try:
                self._collect_metrics()
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
    
    def _collect_metrics(self):
        """Collect current resource metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        
        # Store metrics
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        
        self.logger.debug(f"Collected metrics: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%")
    
    def update_queue_size(self, queue_size: int):
        """Update queue size metric."""
        self.queue_size_history.append(queue_size)
    
    def update_processing_metrics(self, processing_time_ms: float, throughput_hz: float):
        """Update processing performance metrics."""
        self.processing_time_history.append(processing_time_ms)
        self.throughput_history.append(throughput_hz)
    
    def get_avg_cpu(self) -> float:
        """Get average CPU usage."""
        return np.mean(self.cpu_history) if self.cpu_history else 0.0
    
    def get_avg_memory(self) -> float:
        """Get average memory usage."""
        return np.mean(self.memory_history) if self.memory_history else 0.0
    
    def get_avg_queue_size(self) -> float:
        """Get average queue size."""
        return np.mean(self.queue_size_history) if self.queue_size_history else 0.0
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time."""
        return np.mean(self.processing_time_history) if self.processing_time_history else 0.0
    
    def get_avg_throughput(self) -> float:
        """Get average throughput."""
        return np.mean(self.throughput_history) if self.throughput_history else 0.0
    
    def get_trend(self, metric: str) -> str:
        """Get trend for a metric ('increasing', 'decreasing', 'stable')."""
        if metric == 'cpu':
            history = list(self.cpu_history)
        elif metric == 'memory':
            history = list(self.memory_history)
        elif metric == 'queue_size':
            history = list(self.queue_size_history)
        else:
            return 'stable'
        
        if len(history) < 3:
            return 'stable'
        
        # Simple trend analysis
        recent_avg = np.mean(history[-3:])
        older_avg = np.mean(history[:-3])
        
        if recent_avg > older_avg * 1.1:
            return 'increasing'
        elif recent_avg < older_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'


class AutoScaler:
    """
    Automatic scaling system for EchoLoc-NN processing.
    
    Monitors resource usage and processing metrics to automatically
    scale worker processes up or down based on demand.
    """
    
    def __init__(
        self,
        config: ScalingConfig,
        processor_pool=None,  # Reference to processor pool to scale
        scaling_callbacks: Optional[List[Callable]] = None
    ):
        self.config = config
        self.processor_pool = processor_pool
        self.scaling_callbacks = scaling_callbacks or []
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor(config.metric_window)
        
        # Scaling state
        self.current_workers = config.min_workers
        self.last_scaling_time = 0.0
        self.scaling_history: List[ScalingEvent] = []
        
        # Control
        self.scaler_thread: Optional[threading.Thread] = None
        self.stop_scaling = threading.Event()
        self.is_running = False
        
        self.logger = get_logger('auto_scaler')
    
    def start(self):
        """Start auto-scaling system."""
        if self.is_running:
            return
        
        self.logger.info("Starting auto-scaler")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Start scaling thread
        self.is_running = True
        self.stop_scaling.clear()
        
        self.scaler_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True
        )
        self.scaler_thread.start()
        
        self.logger.info(f"Auto-scaler started with {self.current_workers} initial workers")
    
    def stop(self):
        """Stop auto-scaling system."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping auto-scaler")
        
        # Stop scaling thread
        self.stop_scaling.set()
        self.is_running = False
        
        if self.scaler_thread:
            self.scaler_thread.join()
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        self.logger.info("Auto-scaler stopped")
    
    def _scaling_loop(self):
        """Main auto-scaling loop."""
        while not self.stop_scaling.wait(self.config.evaluation_interval):
            try:
                self._evaluate_scaling()
            except Exception as e:
                self.logger.error(f"Error in scaling evaluation: {e}")
    
    def _evaluate_scaling(self):
        """Evaluate whether scaling is needed."""
        current_time = time.time()
        
        # Check if we're in stabilization period
        if current_time - self.last_scaling_time < self.config.stabilization_time:
            self.logger.debug("In stabilization period, skipping scaling evaluation")
            return
        
        # Check rate limiting
        recent_events = [
            event for event in self.scaling_history
            if current_time - event.timestamp < 3600  # Last hour
        ]
        
        if len(recent_events) >= self.config.max_scale_events_per_hour:
            self.logger.warning("Scaling rate limit reached, skipping evaluation")
            return
        
        # Get current metrics
        avg_cpu = self.resource_monitor.get_avg_cpu()
        avg_memory = self.resource_monitor.get_avg_memory()
        avg_queue_size = self.resource_monitor.get_avg_queue_size()
        
        # Update queue size from processor pool if available
        if self.processor_pool:
            pool_stats = self.processor_pool.get_pool_stats()
            if 'queue_size' in pool_stats:
                self.resource_monitor.update_queue_size(pool_stats['queue_size'])
                avg_queue_size = pool_stats['queue_size']
        
        self.logger.debug(
            f"Scaling metrics: CPU={avg_cpu:.1f}%, Memory={avg_memory:.1f}%, Queue={avg_queue_size:.0f}"
        )
        
        # Check for emergency scaling
        if (avg_cpu > self.config.emergency_scale_threshold or 
            avg_memory > self.config.emergency_scale_threshold):
            self._emergency_scale_up(avg_cpu, avg_memory)
            return
        
        # Normal scaling decisions
        scale_decision = self._make_scaling_decision(avg_cpu, avg_memory, avg_queue_size)
        
        if scale_decision['action'] == 'scale_up':
            self._scale_up(scale_decision)
        elif scale_decision['action'] == 'scale_down':
            self._scale_down(scale_decision)
    
    def _make_scaling_decision(self, cpu: float, memory: float, queue_size: float) -> Dict[str, Any]:
        """Make scaling decision based on metrics."""
        # Scale up conditions
        scale_up_reasons = []
        
        if cpu > self.config.cpu_scale_up_threshold:
            scale_up_reasons.append(f"High CPU usage: {cpu:.1f}%")
        
        if memory > self.config.memory_scale_up_threshold:
            scale_up_reasons.append(f"High memory usage: {memory:.1f}%")
        
        if queue_size > self.config.queue_scale_up_threshold:
            scale_up_reasons.append(f"High queue size: {queue_size:.0f}")
        
        # Scale down conditions
        scale_down_reasons = []
        
        if (cpu < self.config.cpu_scale_down_threshold and 
            memory < self.config.memory_scale_up_threshold and
            queue_size < self.config.queue_scale_up_threshold / 2):
            scale_down_reasons.append("Low resource usage across all metrics")
        
        # Decide action
        if scale_up_reasons and self.current_workers < self.config.max_workers:
            return {
                'action': 'scale_up',
                'reasons': scale_up_reasons,
                'trigger_metric': 'cpu' if cpu > self.config.cpu_scale_up_threshold else 'memory',
                'trigger_value': cpu if cpu > self.config.cpu_scale_up_threshold else memory
            }
        elif scale_down_reasons and self.current_workers > self.config.min_workers:
            return {
                'action': 'scale_down',
                'reasons': scale_down_reasons,
                'trigger_metric': 'cpu',
                'trigger_value': cpu
            }
        else:
            return {'action': 'none'}
    
    def _scale_up(self, decision: Dict[str, Any]):
        """Scale up the number of workers."""
        old_workers = self.current_workers
        new_workers = min(
            int(self.current_workers * self.config.scale_up_factor),
            self.config.max_workers
        )
        
        if new_workers == old_workers:
            return
        
        self.logger.info(
            f"Scaling up: {old_workers} -> {new_workers} workers. "
            f"Reasons: {', '.join(decision['reasons'])}"
        )
        
        # Apply scaling
        success = self._apply_scaling(new_workers)
        
        if success:
            # Record scaling event
            event = ScalingEvent(
                timestamp=time.time(),
                event_type='scale_up',
                old_workers=old_workers,
                new_workers=new_workers,
                trigger_metric=decision['trigger_metric'],
                trigger_value=decision['trigger_value'],
                reason='; '.join(decision['reasons'])
            )
            
            self.scaling_history.append(event)
            self.current_workers = new_workers
            self.last_scaling_time = time.time()
            
            # Notify callbacks
            for callback in self.scaling_callbacks:
                callback(event)
    
    def _scale_down(self, decision: Dict[str, Any]):
        """Scale down the number of workers."""
        old_workers = self.current_workers
        new_workers = max(
            int(self.current_workers * self.config.scale_down_factor),
            self.config.min_workers
        )
        
        if new_workers == old_workers:
            return
        
        self.logger.info(
            f"Scaling down: {old_workers} -> {new_workers} workers. "
            f"Reasons: {', '.join(decision['reasons'])}"
        )
        
        # Apply scaling
        success = self._apply_scaling(new_workers)
        
        if success:
            # Record scaling event
            event = ScalingEvent(
                timestamp=time.time(),
                event_type='scale_down',
                old_workers=old_workers,
                new_workers=new_workers,
                trigger_metric=decision['trigger_metric'],
                trigger_value=decision['trigger_value'],
                reason='; '.join(decision['reasons'])
            )
            
            self.scaling_history.append(event)
            self.current_workers = new_workers
            self.last_scaling_time = time.time()
            
            # Notify callbacks
            for callback in self.scaling_callbacks:
                callback(event)
    
    def _emergency_scale_up(self, cpu: float, memory: float):
        """Emergency scaling when resources are critically high."""
        if self.current_workers >= self.config.max_workers:
            self.logger.warning("Emergency scaling needed but already at max workers")
            return
        
        old_workers = self.current_workers
        new_workers = min(self.config.max_workers, old_workers + 2)  # Add 2 workers immediately
        
        self.logger.warning(
            f"Emergency scaling: {old_workers} -> {new_workers} workers. "
            f"CPU: {cpu:.1f}%, Memory: {memory:.1f}%"
        )
        
        success = self._apply_scaling(new_workers)
        
        if success:
            event = ScalingEvent(
                timestamp=time.time(),
                event_type='emergency',
                old_workers=old_workers,
                new_workers=new_workers,
                trigger_metric='cpu' if cpu > memory else 'memory',
                trigger_value=max(cpu, memory),
                reason=f"Emergency scaling: CPU={cpu:.1f}%, Memory={memory:.1f}%"
            )
            
            self.scaling_history.append(event)
            self.current_workers = new_workers
            self.last_scaling_time = time.time()
            
            # Notify callbacks
            for callback in self.scaling_callbacks:
                callback(event)
    
    def _apply_scaling(self, new_worker_count: int) -> bool:
        """Apply scaling to the processor pool."""
        if self.processor_pool is None:
            self.logger.warning("No processor pool configured for scaling")
            return False
        
        try:
            # This would need to be implemented in the processor pool
            # For now, just log the intent
            self.logger.info(f"Would scale processor pool to {new_worker_count} workers")
            return True
        except Exception as e:
            self.logger.error(f"Failed to apply scaling: {e}")
            return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        current_time = time.time()
        
        # Recent scaling events
        recent_events = [
            event for event in self.scaling_history
            if current_time - event.timestamp < 3600  # Last hour
        ]
        
        # Resource metrics
        resource_stats = {
            'avg_cpu': self.resource_monitor.get_avg_cpu(),
            'avg_memory': self.resource_monitor.get_avg_memory(),
            'avg_queue_size': self.resource_monitor.get_avg_queue_size(),
            'cpu_trend': self.resource_monitor.get_trend('cpu'),
            'memory_trend': self.resource_monitor.get_trend('memory')
        }
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.config.min_workers,
            'max_workers': self.config.max_workers,
            'last_scaling_time': self.last_scaling_time,
            'time_since_last_scaling': current_time - self.last_scaling_time,
            'total_scaling_events': len(self.scaling_history),
            'recent_scaling_events': len(recent_events),
            'resource_metrics': resource_stats,
            'recent_events': [
                {
                    'timestamp': event.timestamp,
                    'type': event.event_type,
                    'workers': f"{event.old_workers} -> {event.new_workers}",
                    'trigger': f"{event.trigger_metric}={event.trigger_value:.1f}",
                    'reason': event.reason
                }
                for event in recent_events[-5:]  # Last 5 events
            ]
        }
    
    def add_scaling_callback(self, callback: Callable[[ScalingEvent], None]):
        """Add callback for scaling events."""
        self.scaling_callbacks.append(callback)
    
    def force_scale(self, target_workers: int, reason: str = "Manual scaling"):
        """Force scaling to specific number of workers."""
        if target_workers < self.config.min_workers or target_workers > self.config.max_workers:
            raise ValueError(f"Target workers must be between {self.config.min_workers} and {self.config.max_workers}")
        
        old_workers = self.current_workers
        
        if old_workers == target_workers:
            self.logger.info(f"Already at target worker count: {target_workers}")
            return
        
        self.logger.info(f"Force scaling: {old_workers} -> {target_workers} workers. Reason: {reason}")
        
        success = self._apply_scaling(target_workers)
        
        if success:
            event = ScalingEvent(
                timestamp=time.time(),
                event_type='manual',
                old_workers=old_workers,
                new_workers=target_workers,
                trigger_metric='manual',
                trigger_value=0.0,
                reason=reason
            )
            
            self.scaling_history.append(event)
            self.current_workers = target_workers
            self.last_scaling_time = time.time()
            
            # Notify callbacks
            for callback in self.scaling_callbacks:
                callback(event)

# Enhanced quantum-aware components
class QuantumAwareAutoScaler(AutoScaler):
    """
    Quantum-aware auto-scaler that considers quantum planning metrics.
    
    Extends the base AutoScaler with quantum-specific metrics and scaling decisions.
    """
    
    def __init__(self, config: ScalingConfig, processor_pool=None, 
                 planning_metrics=None, scaling_callbacks=None):
        super().__init__(config, processor_pool, scaling_callbacks)
        self.planning_metrics = planning_metrics
        
        # Quantum-specific thresholds
        self.quantum_thresholds = {
            'planning_time_ms': 5000.0,
            'coherence_warning': 0.3,
            'coherence_critical': 0.1,
            'confidence_warning': 0.7
        }
        
        # Enhanced monitoring
        self.quantum_history = deque(maxlen=20)
        
    def _evaluate_scaling(self):
        """Enhanced scaling evaluation with quantum metrics."""
        # Call parent evaluation first
        super()._evaluate_scaling()
        
        # Add quantum-specific evaluation
        if self.planning_metrics:
            self._evaluate_quantum_scaling()
    
    def _evaluate_quantum_scaling(self):
        """Evaluate scaling based on quantum planning performance."""
        try:
            # Get quantum planning metrics
            planning_perf = self.planning_metrics.get_planning_performance()
            quantum_metrics = self.planning_metrics.get_quantum_metrics()
            
            if planning_perf.get('no_data') or quantum_metrics.get('no_data'):
                return
                
            # Check planning time performance
            avg_planning_time = planning_perf.get('avg_planning_time', 0.0)
            if avg_planning_time > self.quantum_thresholds['planning_time_ms']:
                self._quantum_scale_up('planning_time', avg_planning_time)
                return
                
            # Check quantum coherence
            avg_coherence = quantum_metrics.get('avg_quantum_coherence', 1.0)
            if avg_coherence < self.quantum_thresholds['coherence_critical']:
                self._quantum_scale_up('coherence', avg_coherence)
                return
                
            # Record quantum metrics for trending
            self.quantum_history.append({
                'timestamp': time.time(),
                'planning_time': avg_planning_time,
                'coherence': avg_coherence,
                'quantum_advantage': quantum_metrics.get('quantum_advantage_score', 0.5)
            })
            
        except Exception as e:
            self.logger.error(f"Error in quantum scaling evaluation: {e}")
    
    def _quantum_scale_up(self, metric: str, value: float):
        """Perform quantum-aware scale up."""
        if self.current_workers >= self.config.max_workers:
            return
            
        old_workers = self.current_workers
        new_workers = min(old_workers + 1, self.config.max_workers)
        
        self.logger.info(f"Quantum scaling up: {metric}={value:.3f}")
        
        success = self._apply_scaling(new_workers)
        if success:
            event = ScalingEvent(
                timestamp=time.time(),
                event_type='quantum_scale_up',
                old_workers=old_workers,
                new_workers=new_workers,
                trigger_metric=metric,
                trigger_value=value,
                reason=f"Quantum {metric} threshold exceeded"
            )
            
            self.scaling_history.append(event)
            self.current_workers = new_workers
            self.last_scaling_time = time.time()
    
    def get_quantum_scaling_stats(self) -> Dict[str, Any]:
        """Get quantum-specific scaling statistics."""
        stats = self.get_scaling_stats()
        
        if self.quantum_history:
            recent_quantum = list(self.quantum_history)[-5:]
            stats['quantum_metrics'] = {
                'avg_planning_time': np.mean([q['planning_time'] for q in recent_quantum]),
                'avg_coherence': np.mean([q['coherence'] for q in recent_quantum]),
                'avg_quantum_advantage': np.mean([q['quantum_advantage'] for q in recent_quantum])
            }
        
        quantum_events = [e for e in self.scaling_history if 'quantum' in e.event_type]
        stats['quantum_scaling_events'] = len(quantum_events)
        
        return stats

# Global enhanced auto-scaler
_global_quantum_auto_scaler = None

def get_global_quantum_auto_scaler(config=None, processor_pool=None, 
                                  planning_metrics=None) -> QuantumAwareAutoScaler:
    """Get or create global quantum-aware auto-scaler."""
    global _global_quantum_auto_scaler
    if _global_quantum_auto_scaler is None and config:
        _global_quantum_auto_scaler = QuantumAwareAutoScaler(
            config, processor_pool, planning_metrics
        )
    return _global_quantum_auto_scaler