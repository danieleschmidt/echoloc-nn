"""
Planning Performance Metrics and Monitoring

Provides comprehensive metrics collection, analysis, and monitoring
for quantum-inspired task planning systems.
"""

import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class PlanningCycleMetrics:
    """Metrics for a single planning cycle."""
    timestamp: datetime
    planning_time: float  # seconds
    final_energy: float
    convergence_iterations: int
    n_tasks: int
    n_resources: int
    strategy_used: str
    quantum_coherence: float = 1.0
    measurement_count: int = 0
    superposition_states: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'planning_time': self.planning_time,
            'final_energy': self.final_energy,
            'convergence_iterations': self.convergence_iterations,
            'n_tasks': self.n_tasks,
            'n_resources': self.n_resources,
            'strategy_used': self.strategy_used,
            'quantum_coherence': self.quantum_coherence,
            'measurement_count': self.measurement_count,
            'superposition_states': self.superposition_states
        }

@dataclass 
class ExecutionMetrics:
    """Metrics for task execution performance."""
    task_id: str
    planned_start: float
    actual_start: float
    planned_duration: float
    actual_duration: float
    resource_used: str
    success: bool
    error_message: Optional[str] = None
    
    @property
    def start_deviation(self) -> float:
        """Deviation from planned start time."""
        return abs(self.actual_start - self.planned_start)
        
    @property
    def duration_deviation(self) -> float:
        """Deviation from planned duration."""
        return abs(self.actual_duration - self.planned_duration)
        
    @property
    def schedule_accuracy(self) -> float:
        """Overall schedule accuracy (0.0 to 1.0)."""
        max_start_dev = max(1.0, self.planned_start * 0.1)  # 10% tolerance
        max_duration_dev = max(1.0, self.planned_duration * 0.1)
        
        start_accuracy = max(0.0, 1.0 - self.start_deviation / max_start_dev)
        duration_accuracy = max(0.0, 1.0 - self.duration_deviation / max_duration_dev)
        
        return (start_accuracy + duration_accuracy) / 2.0

class PlanningMetrics:
    """
    Comprehensive metrics collection and analysis for quantum task planning.
    
    Tracks planning performance, execution accuracy, resource utilization,
    and quantum-specific metrics like coherence and superposition effectiveness.
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        
        # Planning metrics
        self.planning_cycles: deque = deque(maxlen=history_size)
        self.execution_metrics: deque = deque(maxlen=history_size)
        
        # Performance tracking
        self.total_plans_generated = 0
        self.total_tasks_executed = 0
        self.total_planning_time = 0.0
        
        # Resource utilization tracking
        self.resource_usage_history = defaultdict(list)
        self.resource_efficiency_scores = defaultdict(list)
        
        # Quantum metrics
        self.quantum_coherence_history = []
        self.superposition_effectiveness = []
        self.measurement_efficiency = []
        
        # Real-time statistics
        self.current_stats = {
            'avg_planning_time': 0.0,
            'avg_energy': 0.0,
            'avg_convergence': 0.0,
            'success_rate': 1.0,
            'schedule_accuracy': 1.0,
            'resource_utilization': 0.0
        }
        
        self.start_time = datetime.now()
        
    def record_planning_cycle(self, 
                            planning_time: float,
                            final_energy: float,
                            convergence_iterations: int,
                            n_tasks: int = 0,
                            n_resources: int = 0,
                            strategy_used: str = "unknown",
                            quantum_coherence: float = 1.0,
                            measurement_count: int = 0,
                            superposition_states: int = 1):
        """Record metrics from a planning cycle."""
        
        cycle_metrics = PlanningCycleMetrics(
            timestamp=datetime.now(),
            planning_time=planning_time,
            final_energy=final_energy,
            convergence_iterations=convergence_iterations,
            n_tasks=n_tasks,
            n_resources=n_resources,
            strategy_used=strategy_used,
            quantum_coherence=quantum_coherence,
            measurement_count=measurement_count,
            superposition_states=superposition_states
        )
        
        self.planning_cycles.append(cycle_metrics)
        self.total_plans_generated += 1
        self.total_planning_time += planning_time
        
        # Update quantum metrics
        self.quantum_coherence_history.append(quantum_coherence)
        if measurement_count > 0:
            self.measurement_efficiency.append(convergence_iterations / measurement_count)
        if superposition_states > 1:
            # Effectiveness based on convergence vs. superposition complexity
            effectiveness = max(0.0, 1.0 - convergence_iterations / (superposition_states * 100))
            self.superposition_effectiveness.append(effectiveness)
            
        # Update real-time statistics
        self._update_current_stats()
        
        logger.debug(f"Recorded planning cycle: {planning_time:.3f}s, energy={final_energy:.2f}")
        
    def record_task_execution(self,
                            task_id: str,
                            planned_start: float,
                            actual_start: float,
                            planned_duration: float,
                            actual_duration: float,
                            resource_used: str,
                            success: bool,
                            error_message: Optional[str] = None):
        """Record metrics from task execution."""
        
        execution_metrics = ExecutionMetrics(
            task_id=task_id,
            planned_start=planned_start,
            actual_start=actual_start,
            planned_duration=planned_duration,
            actual_duration=actual_duration,
            resource_used=resource_used,
            success=success,
            error_message=error_message
        )
        
        self.execution_metrics.append(execution_metrics)
        self.total_tasks_executed += 1
        
        # Update resource utilization tracking
        efficiency = min(1.0, planned_duration / max(0.1, actual_duration))
        self.resource_efficiency_scores[resource_used].append(efficiency)
        
        utilization = actual_duration / max(1.0, planned_duration + execution_metrics.start_deviation)
        self.resource_usage_history[resource_used].append(utilization)
        
        # Update real-time statistics
        self._update_current_stats()
        
        logger.debug(f"Recorded task execution: {task_id}, success={success}")
        
    def get_planning_performance(self) -> Dict[str, Any]:
        """Get overall planning performance metrics."""
        if not self.planning_cycles:
            return {'no_data': True}
            
        cycles = list(self.planning_cycles)
        
        return {
            'total_cycles': len(cycles),
            'avg_planning_time': statistics.mean([c.planning_time for c in cycles]),
            'median_planning_time': statistics.median([c.planning_time for c in cycles]),
            'avg_final_energy': statistics.mean([c.final_energy for c in cycles]),
            'avg_convergence_iterations': statistics.mean([c.convergence_iterations for c in cycles]),
            'strategy_distribution': self._get_strategy_distribution(cycles),
            'planning_time_trend': self._calculate_trend([c.planning_time for c in cycles[-20:]]),
            'energy_trend': self._calculate_trend([c.final_energy for c in cycles[-20:]]),
            'total_planning_time': self.total_planning_time
        }
        
    def get_execution_performance(self) -> Dict[str, Any]:
        """Get task execution performance metrics."""
        if not self.execution_metrics:
            return {'no_data': True}
            
        executions = list(self.execution_metrics)
        successful_executions = [e for e in executions if e.success]
        
        return {
            'total_executions': len(executions),
            'success_rate': len(successful_executions) / len(executions),
            'avg_schedule_accuracy': statistics.mean([e.schedule_accuracy for e in successful_executions]) if successful_executions else 0.0,
            'avg_start_deviation': statistics.mean([e.start_deviation for e in executions]),
            'avg_duration_deviation': statistics.mean([e.duration_deviation for e in executions]),
            'schedule_accuracy_trend': self._calculate_trend([e.schedule_accuracy for e in executions[-20:]]),
            'failure_rate': (len(executions) - len(successful_executions)) / len(executions),
            'common_errors': self._get_error_distribution(executions)
        }
        
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get resource utilization metrics."""
        if not self.resource_usage_history:
            return {'no_data': True}
            
        utilization_stats = {}
        efficiency_stats = {}
        
        for resource, usage_list in self.resource_usage_history.items():
            if usage_list:
                utilization_stats[resource] = {
                    'avg_utilization': statistics.mean(usage_list),
                    'max_utilization': max(usage_list),
                    'utilization_variance': statistics.variance(usage_list) if len(usage_list) > 1 else 0.0
                }
                
        for resource, efficiency_list in self.resource_efficiency_scores.items():
            if efficiency_list:
                efficiency_stats[resource] = {
                    'avg_efficiency': statistics.mean(efficiency_list),
                    'efficiency_trend': self._calculate_trend(efficiency_list[-10:])
                }
                
        return {
            'utilization_by_resource': utilization_stats,
            'efficiency_by_resource': efficiency_stats,
            'overall_utilization': statistics.mean([stats['avg_utilization'] 
                                                   for stats in utilization_stats.values()]) if utilization_stats else 0.0
        }
        
    def get_quantum_metrics(self) -> Dict[str, Any]:
        """Get quantum-specific performance metrics."""
        return {
            'avg_quantum_coherence': statistics.mean(self.quantum_coherence_history) if self.quantum_coherence_history else 1.0,
            'coherence_stability': 1.0 - statistics.stdev(self.quantum_coherence_history) if len(self.quantum_coherence_history) > 1 else 1.0,
            'avg_superposition_effectiveness': statistics.mean(self.superposition_effectiveness) if self.superposition_effectiveness else 0.0,
            'avg_measurement_efficiency': statistics.mean(self.measurement_efficiency) if self.measurement_efficiency else 0.0,
            'coherence_trend': self._calculate_trend(self.quantum_coherence_history[-20:]),
            'quantum_advantage_score': self._calculate_quantum_advantage()
        }
        
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'system_info': {
                'start_time': self.start_time.isoformat(),
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'total_plans_generated': self.total_plans_generated,
                'total_tasks_executed': self.total_tasks_executed
            },
            'planning_performance': self.get_planning_performance(),
            'execution_performance': self.get_execution_performance(),
            'resource_utilization': self.get_resource_utilization(),
            'quantum_metrics': self.get_quantum_metrics(),
            'current_stats': self.current_stats,
            'performance_score': self._calculate_overall_performance_score()
        }
        
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file."""
        report = self.get_comprehensive_report()
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        logger.info(f"Metrics exported to {filepath}")
        
    def reset(self):
        """Reset all metrics."""
        self.planning_cycles.clear()
        self.execution_metrics.clear()
        self.resource_usage_history.clear()
        self.resource_efficiency_scores.clear()
        self.quantum_coherence_history.clear()
        self.superposition_effectiveness.clear()
        self.measurement_efficiency.clear()
        
        self.total_plans_generated = 0
        self.total_tasks_executed = 0
        self.total_planning_time = 0.0
        
        self.start_time = datetime.now()
        
        logger.info("Metrics reset completed")
        
    def _update_current_stats(self):
        """Update real-time statistics."""
        if self.planning_cycles:
            recent_cycles = list(self.planning_cycles)[-10:]  # Last 10 cycles
            self.current_stats['avg_planning_time'] = statistics.mean([c.planning_time for c in recent_cycles])
            self.current_stats['avg_energy'] = statistics.mean([c.final_energy for c in recent_cycles])
            self.current_stats['avg_convergence'] = statistics.mean([c.convergence_iterations for c in recent_cycles])
            
        if self.execution_metrics:
            recent_executions = list(self.execution_metrics)[-20:]  # Last 20 executions
            successful = [e for e in recent_executions if e.success]
            
            self.current_stats['success_rate'] = len(successful) / len(recent_executions)
            if successful:
                self.current_stats['schedule_accuracy'] = statistics.mean([e.schedule_accuracy for e in successful])
                
        if self.resource_usage_history:
            all_recent_usage = []
            for usage_list in self.resource_usage_history.values():
                if usage_list:
                    all_recent_usage.extend(usage_list[-5:])  # Last 5 uses per resource
            if all_recent_usage:
                self.current_stats['resource_utilization'] = statistics.mean(all_recent_usage)
                
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 2:
            return "stable"
            
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        
        slope = (n * sum(x[i] * values[i] for i in range(n)) - sum(x) * sum(values)) / \
                (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
                
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "improving" if values[0] > values[-1] else "deteriorating"
        else:
            return "deteriorating" if values[0] < values[-1] else "improving"
            
    def _get_strategy_distribution(self, cycles: List[PlanningCycleMetrics]) -> Dict[str, int]:
        """Get distribution of strategies used."""
        distribution = defaultdict(int)
        for cycle in cycles:
            distribution[cycle.strategy_used] += 1
        return dict(distribution)
        
    def _get_error_distribution(self, executions: List[ExecutionMetrics]) -> Dict[str, int]:
        """Get distribution of error types."""
        error_dist = defaultdict(int)
        for execution in executions:
            if not execution.success and execution.error_message:
                # Simplified error categorization
                error_type = execution.error_message.split(':')[0] if ':' in execution.error_message else execution.error_message
                error_dist[error_type] += 1
        return dict(error_dist)
        
    def _calculate_quantum_advantage(self) -> float:
        """Calculate quantum advantage score based on performance metrics."""
        if not self.planning_cycles:
            return 0.0
            
        # Quantum advantage based on convergence speed and coherence
        recent_cycles = list(self.planning_cycles)[-20:]
        
        avg_convergence = statistics.mean([c.convergence_iterations for c in recent_cycles])
        avg_coherence = statistics.mean([c.quantum_coherence for c in recent_cycles])
        
        # Normalize convergence (lower is better)
        convergence_score = max(0.0, 1.0 - avg_convergence / 1000.0)
        
        # Quantum advantage score
        quantum_advantage = (convergence_score * avg_coherence) ** 0.5
        
        return min(1.0, quantum_advantage)
        
    def _calculate_overall_performance_score(self) -> float:
        """Calculate overall system performance score (0.0 to 1.0)."""
        scores = []
        
        # Planning efficiency (based on planning time)
        if self.current_stats['avg_planning_time'] > 0:
            planning_score = max(0.0, 1.0 - self.current_stats['avg_planning_time'] / 10.0)  # 10s baseline
            scores.append(planning_score)
            
        # Execution success rate
        scores.append(self.current_stats['success_rate'])
        
        # Schedule accuracy
        scores.append(self.current_stats['schedule_accuracy'])
        
        # Resource utilization (optimal around 0.8)
        util = self.current_stats['resource_utilization']
        util_score = 1.0 - abs(util - 0.8) / 0.8 if util > 0 else 0.0
        scores.append(max(0.0, util_score))
        
        # Quantum advantage
        scores.append(self._calculate_quantum_advantage())
        
        return statistics.mean(scores) if scores else 0.0
        
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts and recommendations."""
        alerts = []
        
        # Check planning time
        if self.current_stats['avg_planning_time'] > 5.0:
            alerts.append({
                'type': 'warning',
                'metric': 'planning_time',
                'message': f"Average planning time is high: {self.current_stats['avg_planning_time']:.2f}s",
                'recommendation': "Consider optimizing algorithm parameters or using hybrid approach"
            })
            
        # Check success rate
        if self.current_stats['success_rate'] < 0.9:
            alerts.append({
                'type': 'critical',
                'metric': 'success_rate',
                'message': f"Low success rate: {self.current_stats['success_rate']:.1%}",
                'recommendation': "Review task dependencies and resource constraints"
            })
            
        # Check resource utilization
        if self.current_stats['resource_utilization'] < 0.5:
            alerts.append({
                'type': 'info',
                'metric': 'resource_utilization',
                'message': f"Low resource utilization: {self.current_stats['resource_utilization']:.1%}",
                'recommendation': "Consider load balancing or resource pooling"
            })
        elif self.current_stats['resource_utilization'] > 0.95:
            alerts.append({
                'type': 'warning',
                'metric': 'resource_utilization',
                'message': f"High resource utilization: {self.current_stats['resource_utilization']:.1%}",
                'recommendation': "Consider adding resources or optimizing task scheduling"
            })
            
        # Check quantum coherence
        if self.quantum_coherence_history:
            recent_coherence = statistics.mean(self.quantum_coherence_history[-10:])
            if recent_coherence < 0.5:
                alerts.append({
                    'type': 'warning',
                    'metric': 'quantum_coherence',
                    'message': f"Low quantum coherence: {recent_coherence:.2f}",
                    'recommendation': "Reduce decoherence rate or increase coherence time"
                })
                
        return alerts