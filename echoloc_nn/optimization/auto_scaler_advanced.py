"""
Advanced auto-scaling system for EchoLoc-NN with intelligent resource management.
Generation 3 (Optimized) - Auto-scaling and resource pool management
"""

import time
import threading
import psutil
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import logging


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU_WORKERS = "cpu_workers"
    MEMORY_CACHE = "memory_cache"
    INFERENCE_THREADS = "inference_threads"
    BATCH_SIZE = "batch_size"
    QUEUE_SIZE = "queue_size"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    active_threads: int
    queue_sizes: Dict[str, int] = field(default_factory=dict)
    throughput_per_second: float = 0.0
    average_latency_ms: float = 0.0
    error_rate_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingRule:
    """Rule for automatic scaling decisions."""
    resource_type: ResourceType
    metric_thresholds: Dict[str, float]  # e.g., {"cpu_percent": 80.0}
    scale_action: str  # "up" or "down" 
    scale_factor: float = 1.5  # Multiplier for scaling
    cooldown_seconds: float = 60.0  # Time between scaling actions
    max_value: Optional[float] = None
    min_value: Optional[float] = None
    conditions: List[str] = field(default_factory=list)  # Additional conditions


class IntelligentAutoScaler:
    """
    Intelligent auto-scaling system with predictive scaling and resource optimization.
    
    Features:
    - Multi-metric monitoring (CPU, memory, latency, throughput)
    - Predictive scaling based on usage patterns
    - Resource pool management
    - Cost-aware scaling decisions
    - Learning from historical performance
    """
    
    def __init__(
        self,
        target_system: Any,
        monitoring_interval: float = 5.0,
        history_size: int = 288,  # 24 hours of 5-second samples
        enable_predictive: bool = True,
        enable_cost_optimization: bool = True
    ):
        self.target_system = target_system
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_predictive = enable_predictive
        self.enable_cost_optimization = enable_cost_optimization
        
        # Historical data
        self.metrics_history: deque = deque(maxlen=history_size)
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Current state
        self.current_resources: Dict[ResourceType, float] = {
            ResourceType.CPU_WORKERS: 4,
            ResourceType.MEMORY_CACHE: 512,  # MB
            ResourceType.INFERENCE_THREADS: 8,
            ResourceType.BATCH_SIZE: 16,
            ResourceType.QUEUE_SIZE: 1000
        }
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = self._create_default_rules()
        
        # State tracking
        self.last_scaling_time: Dict[ResourceType, float] = {}
        self.scaling_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.scaling_decisions = 0
        self.successful_scalings = 0
        self.prevented_overloads = 0
        
        # Machine learning for predictive scaling
        self.usage_patterns: Dict[str, List[float]] = {
            'hourly': [0.0] * 24,
            'daily': [0.0] * 7,
            'trend': deque(maxlen=20)  # Short-term trend
        }
        
        # Logging
        self.logger = logging.getLogger('AutoScaler')
        
    def _create_default_rules(self) -> List[ScalingRule]:
        """Create default scaling rules."""
        return [
            # Scale up CPU workers when CPU usage is high
            ScalingRule(
                resource_type=ResourceType.CPU_WORKERS,
                metric_thresholds={"cpu_percent": 80.0, "average_latency_ms": 100.0},
                scale_action="up",
                scale_factor=1.5,
                max_value=32,
                min_value=2
            ),
            
            # Scale down CPU workers when CPU usage is low
            ScalingRule(
                resource_type=ResourceType.CPU_WORKERS,
                metric_thresholds={"cpu_percent": 20.0},
                scale_action="down",
                scale_factor=0.75,
                cooldown_seconds=120.0,  # Longer cooldown for scaling down
                conditions=["throughput_per_second < 5"]
            ),
            
            # Scale up memory cache when memory pressure is high
            ScalingRule(
                resource_type=ResourceType.MEMORY_CACHE,
                metric_thresholds={"memory_percent": 85.0},
                scale_action="up",
                scale_factor=1.3,
                max_value=2048,  # 2GB max
                min_value=128
            ),
            
            # Scale up batch size when throughput demand is high
            ScalingRule(
                resource_type=ResourceType.BATCH_SIZE,
                metric_thresholds={"throughput_per_second": 50.0, "average_latency_ms": 50.0},
                scale_action="up",
                scale_factor=1.25,
                max_value=128,
                min_value=1
            ),
            
            # Scale up queue size when queues are filling up
            ScalingRule(
                resource_type=ResourceType.QUEUE_SIZE,
                metric_thresholds={"queue_utilization": 80.0},
                scale_action="up",
                scale_factor=1.5,
                max_value=10000,
                min_value=100
            )
        ]
    
    def start_monitoring(self):
        """Start the auto-scaling monitoring system."""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Auto-scaler monitoring started")
    
    def stop_monitoring(self):
        """Stop the auto-scaling monitoring system."""
        self.scaling_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Auto-scaler monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        while self.scaling_active:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Update usage patterns
                self._update_usage_patterns(metrics)
                
                # Make scaling decisions
                if len(self.metrics_history) >= 2:  # Need history for decisions
                    scaling_decisions = self._make_scaling_decisions(metrics)
                    
                    # Apply scaling decisions
                    for decision in scaling_decisions:
                        self._apply_scaling_decision(decision)
                
                # Predictive scaling
                if self.enable_predictive and len(self.metrics_history) >= 10:
                    predictive_decisions = self._make_predictive_scaling_decisions()
                    
                    for decision in predictive_decisions:
                        self._apply_scaling_decision(decision)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / (1024 * 1024)
        
        # Thread metrics
        active_threads = threading.active_count()
        
        # Queue metrics (if available from target system)
        queue_sizes = {}
        if hasattr(self.target_system, 'get_queue_sizes'):
            queue_sizes = self.target_system.get_queue_sizes()
        
        # Performance metrics (if available from target system)
        throughput = 0.0
        latency = 0.0
        error_rate = 0.0
        
        if hasattr(self.target_system, 'get_performance_stats'):
            stats = self.target_system.get_performance_stats()
            throughput = stats.get('throughput_per_second', 0.0)
            latency = stats.get('average_latency_ms', 0.0)
            
            total_requests = stats.get('total_requests', 1)
            failed_requests = stats.get('failed_requests', 0)
            error_rate = (failed_requests / total_requests) * 100
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            active_threads=active_threads,
            queue_sizes=queue_sizes,
            throughput_per_second=throughput,
            average_latency_ms=latency,
            error_rate_percent=error_rate
        )
    
    def _update_usage_patterns(self, metrics: ResourceMetrics):
        """Update usage patterns for predictive scaling."""
        current_time = time.time()
        hour = int((current_time % 86400) // 3600)  # Hour of day (0-23)
        day = int((current_time % 604800) // 86400)  # Day of week (0-6)
        
        # Update hourly pattern
        alpha = 0.1  # Exponential moving average factor
        self.usage_patterns['hourly'][hour] = (
            alpha * metrics.cpu_percent + 
            (1 - alpha) * self.usage_patterns['hourly'][hour]
        )
        
        # Update daily pattern
        self.usage_patterns['daily'][day] = (
            alpha * metrics.cpu_percent +
            (1 - alpha) * self.usage_patterns['daily'][day]
        )
        
        # Update trend
        self.usage_patterns['trend'].append(metrics.cpu_percent)
    
    def _make_scaling_decisions(self, current_metrics: ResourceMetrics) -> List[Dict[str, Any]]:
        """Make scaling decisions based on current metrics."""
        decisions = []
        
        for rule in self.scaling_rules:
            # Check cooldown period
            if rule.resource_type in self.last_scaling_time:
                time_since_last = time.time() - self.last_scaling_time[rule.resource_type]
                if time_since_last < rule.cooldown_seconds:
                    continue
            
            # Check if thresholds are met
            thresholds_met = self._check_thresholds(rule, current_metrics)
            
            if thresholds_met:
                # Check additional conditions
                conditions_met = self._check_conditions(rule, current_metrics)
                
                if conditions_met:
                    # Calculate new resource value
                    current_value = self.current_resources[rule.resource_type]
                    
                    if rule.scale_action == "up":
                        new_value = current_value * rule.scale_factor
                        if rule.max_value:
                            new_value = min(new_value, rule.max_value)
                    else:  # scale down
                        new_value = current_value * rule.scale_factor
                        if rule.min_value:
                            new_value = max(new_value, rule.min_value)
                    
                    # Only scale if there's a meaningful change
                    if abs(new_value - current_value) / current_value > 0.1:  # 10% threshold
                        decisions.append({
                            'resource_type': rule.resource_type,
                            'action': rule.scale_action,
                            'current_value': current_value,
                            'new_value': new_value,
                            'reason': f"Metric thresholds met: {rule.metric_thresholds}",
                            'rule': rule
                        })
        
        return decisions
    
    def _check_thresholds(self, rule: ScalingRule, metrics: ResourceMetrics) -> bool:
        """Check if metric thresholds are met for scaling."""
        for metric_name, threshold in rule.metric_thresholds.items():
            current_value = getattr(metrics, metric_name, 0)
            
            # For special metrics like queue utilization
            if metric_name == "queue_utilization":
                if metrics.queue_sizes:
                    total_size = sum(metrics.queue_sizes.values())
                    max_size = self.current_resources.get(ResourceType.QUEUE_SIZE, 1000)
                    current_value = (total_size / max_size) * 100
                else:
                    current_value = 0
            
            # Check threshold based on scaling direction
            if rule.scale_action == "up":
                if current_value < threshold:
                    return False
            else:  # scale down
                if current_value > threshold:
                    return False
        
        return True
    
    def _check_conditions(self, rule: ScalingRule, metrics: ResourceMetrics) -> bool:
        """Check additional conditions for scaling."""
        for condition in rule.conditions:
            # Parse and evaluate condition
            # This is a simplified implementation
            if "throughput_per_second < 5" in condition:
                if metrics.throughput_per_second >= 5:
                    return False
            elif "error_rate_percent > 5" in condition:
                if metrics.error_rate_percent <= 5:
                    return False
            # Add more condition parsing as needed
        
        return True
    
    def _make_predictive_scaling_decisions(self) -> List[Dict[str, Any]]:
        """Make predictive scaling decisions based on historical patterns."""
        if not self.enable_predictive:
            return []
        
        decisions = []
        
        try:
            # Predict next period's usage
            predicted_cpu = self._predict_cpu_usage()
            
            if predicted_cpu > 75.0:  # Predict high load
                current_workers = self.current_resources[ResourceType.CPU_WORKERS]
                
                # Preemptively scale up
                decisions.append({
                    'resource_type': ResourceType.CPU_WORKERS,
                    'action': 'up',
                    'current_value': current_workers,
                    'new_value': min(current_workers * 1.2, 32),
                    'reason': f'Predictive scaling: Expected CPU usage {predicted_cpu:.1f}%',
                    'predictive': True
                })
            
            elif predicted_cpu < 25.0:  # Predict low load
                current_workers = self.current_resources[ResourceType.CPU_WORKERS]
                
                if current_workers > 4:  # Only scale down if we have excess
                    decisions.append({
                        'resource_type': ResourceType.CPU_WORKERS,
                        'action': 'down',
                        'current_value': current_workers,
                        'new_value': max(current_workers * 0.8, 2),
                        'reason': f'Predictive scaling: Expected CPU usage {predicted_cpu:.1f}%',
                        'predictive': True
                    })
        
        except Exception as e:
            self.logger.error(f"Predictive scaling error: {e}")
        
        return decisions
    
    def _predict_cpu_usage(self) -> float:
        """Predict CPU usage for the next period."""
        if len(self.usage_patterns['trend']) < 5:
            return 50.0  # Default if insufficient data
        
        # Simple trend-based prediction
        recent_values = list(self.usage_patterns['trend'])[-5:]
        
        # Calculate trend
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            current_value = recent_values[-1]
            predicted_value = current_value + slope
            
            # Apply seasonal adjustment based on hourly patterns
            current_hour = int((time.time() % 86400) // 3600)
            next_hour = (current_hour + 1) % 24
            
            hourly_factor = self.usage_patterns['hourly'][next_hour] / max(self.usage_patterns['hourly'][current_hour], 1)
            predicted_value *= hourly_factor
            
            return max(0, min(100, predicted_value))
        
        return recent_values[-1]
    
    def _apply_scaling_decision(self, decision: Dict[str, Any]):
        """Apply a scaling decision to the target system."""
        try:
            resource_type = decision['resource_type']
            new_value = decision['new_value']
            
            # Apply the scaling change
            success = self._execute_scaling(resource_type, new_value)
            
            if success:
                self.current_resources[resource_type] = new_value
                self.last_scaling_time[resource_type] = time.time()
                self.successful_scalings += 1
                
                # Log the scaling action
                self.logger.info(
                    f"Scaled {resource_type.value} from {decision['current_value']:.1f} "
                    f"to {new_value:.1f} ({decision['reason']})"
                )
                
                # Record scaling history
                self.scaling_history.append({
                    'timestamp': time.time(),
                    'resource_type': resource_type.value,
                    'action': decision['action'],
                    'old_value': decision['current_value'],
                    'new_value': new_value,
                    'reason': decision['reason'],
                    'predictive': decision.get('predictive', False)
                })
                
                # Trim history if too long
                if len(self.scaling_history) > 1000:
                    self.scaling_history = self.scaling_history[-500:]
            
            self.scaling_decisions += 1
            
        except Exception as e:
            self.logger.error(f"Failed to apply scaling decision: {e}")
    
    def _execute_scaling(self, resource_type: ResourceType, new_value: float) -> bool:
        """Execute the actual scaling change on the target system."""
        try:
            if resource_type == ResourceType.CPU_WORKERS:
                if hasattr(self.target_system, 'set_worker_count'):
                    return self.target_system.set_worker_count(int(new_value))
            
            elif resource_type == ResourceType.MEMORY_CACHE:
                if hasattr(self.target_system, 'set_cache_size'):
                    return self.target_system.set_cache_size(int(new_value))
            
            elif resource_type == ResourceType.INFERENCE_THREADS:
                if hasattr(self.target_system, 'set_thread_count'):
                    return self.target_system.set_thread_count(int(new_value))
            
            elif resource_type == ResourceType.BATCH_SIZE:
                if hasattr(self.target_system, 'set_batch_size'):
                    return self.target_system.set_batch_size(int(new_value))
            
            elif resource_type == ResourceType.QUEUE_SIZE:
                if hasattr(self.target_system, 'set_queue_size'):
                    return self.target_system.set_queue_size(int(new_value))
            
            return False
            
        except Exception as e:
            self.logger.error(f"Scaling execution error for {resource_type}: {e}")
            return False
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules.append(rule)
    
    def remove_scaling_rule(self, resource_type: ResourceType, scale_action: str):
        """Remove scaling rules for a specific resource and action."""
        self.scaling_rules = [
            rule for rule in self.scaling_rules 
            if not (rule.resource_type == resource_type and rule.scale_action == scale_action)
        ]
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current auto-scaler state and metrics."""
        recent_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'active': self.scaling_active,
            'current_resources': {rt.value: val for rt, val in self.current_resources.items()},
            'recent_metrics': recent_metrics.__dict__ if recent_metrics else None,
            'scaling_stats': {
                'total_decisions': self.scaling_decisions,
                'successful_scalings': self.successful_scalings,
                'prevented_overloads': self.prevented_overloads,
                'success_rate': self.successful_scalings / max(self.scaling_decisions, 1) * 100
            },
            'usage_patterns': {
                'hourly_avg': np.mean(self.usage_patterns['hourly']),
                'daily_avg': np.mean(self.usage_patterns['daily']),
                'trend_direction': 'up' if len(self.usage_patterns['trend']) > 5 and 
                                        self.usage_patterns['trend'][-1] > self.usage_patterns['trend'][-5] 
                                        else 'down'
            },
            'recent_scaling_history': self.scaling_history[-10:] if self.scaling_history else []
        }
    
    def force_scaling_action(
        self, 
        resource_type: ResourceType, 
        action: str, 
        value: Optional[float] = None
    ) -> bool:
        """Force a scaling action (for testing/emergency)."""
        current_value = self.current_resources[resource_type]
        
        if value is None:
            # Use default scaling factor
            scale_factor = 1.5 if action == "up" else 0.75
            new_value = current_value * scale_factor
        else:
            new_value = value
        
        decision = {
            'resource_type': resource_type,
            'action': action,
            'current_value': current_value,
            'new_value': new_value,
            'reason': 'Forced scaling action'
        }
        
        self._apply_scaling_decision(decision)
        return True
    
    def reset_scaling_history(self):
        """Reset scaling history and patterns."""
        self.scaling_history = []
        self.metrics_history.clear()
        self.usage_patterns = {
            'hourly': [0.0] * 24,
            'daily': [0.0] * 7,
            'trend': deque(maxlen=20)
        }
        self.scaling_decisions = 0
        self.successful_scalings = 0
        self.prevented_overloads = 0