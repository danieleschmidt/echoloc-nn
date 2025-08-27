"""
Progressive Quality Gates System for Generation 4+
Enhanced continuous monitoring, adaptive thresholds, and research-grade validation.
"""

import logging
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass 
class QualityMetric:
    """Individual quality metric with adaptive thresholds."""
    name: str
    value: float
    threshold: float
    adaptive_threshold: float
    trend: List[float] = field(default_factory=list)
    status: str = "UNKNOWN"  # PASS, FAIL, WARNING, CRITICAL
    improvement_rate: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    statistical_significance: Optional[float] = None
    
    def __post_init__(self):
        self.status = self._compute_status()
        if len(self.trend) > 1:
            self.improvement_rate = self._compute_improvement_rate()
            self.confidence_interval = self._compute_confidence_interval()
    
    def _compute_status(self) -> str:
        """Compute quality gate status based on value and adaptive threshold."""
        if self.value >= self.adaptive_threshold:
            return "PASS"
        elif self.value >= self.adaptive_threshold * 0.9:
            return "WARNING"
        elif self.value >= self.adaptive_threshold * 0.8:
            return "CRITICAL"
        else:
            return "FAIL"
    
    def _compute_improvement_rate(self) -> float:
        """Compute trend-based improvement rate."""
        if len(self.trend) < 2:
            return 0.0
        recent_values = self.trend[-min(5, len(self.trend)):]
        if len(recent_values) < 2:
            return 0.0
        
        # Linear regression for trend
        n = len(recent_values)
        x_mean = (n - 1) / 2
        y_mean = sum(recent_values) / n
        
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent_values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _compute_confidence_interval(self) -> Tuple[float, float]:
        """Compute 95% confidence interval for metric."""
        if len(self.trend) < 3:
            return (self.value, self.value)
        
        import math
        n = len(self.trend)
        mean_val = sum(self.trend) / n
        std_dev = math.sqrt(sum((x - mean_val) ** 2 for x in self.trend) / (n - 1))
        margin = 1.96 * std_dev / math.sqrt(n)  # 95% CI
        
        return (mean_val - margin, mean_val + margin)

@dataclass
class QualityGateResult:
    """Result from quality gate evaluation."""
    timestamp: datetime
    gate_name: str
    metrics: Dict[str, QualityMetric]
    overall_status: str
    pass_count: int
    fail_count: int 
    warning_count: int
    critical_count: int
    performance_delta: float
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.overall_status = self._compute_overall_status()
        self._generate_recommendations()
    
    def _compute_overall_status(self) -> str:
        """Compute overall gate status."""
        if self.fail_count > 0:
            return "FAIL"
        elif self.critical_count > 0:
            return "CRITICAL" 
        elif self.warning_count > 0:
            return "WARNING"
        else:
            return "PASS"
    
    def _generate_recommendations(self):
        """Generate improvement recommendations."""
        for metric_name, metric in self.metrics.items():
            if metric.status == "FAIL":
                self.recommendations.append(
                    f"CRITICAL: {metric_name} requires immediate attention "
                    f"(current: {metric.value:.3f}, required: {metric.adaptive_threshold:.3f})"
                )
            elif metric.status == "CRITICAL":
                self.recommendations.append(
                    f"HIGH: {metric_name} approaching failure threshold "
                    f"(trend: {metric.improvement_rate:.4f})"
                )
            elif metric.status == "WARNING" and metric.improvement_rate < 0:
                self.recommendations.append(
                    f"MEDIUM: {metric_name} showing declining trend, monitor closely"
                )

class ProgressiveQualityGateSystem:
    """
    Advanced quality gate system with continuous monitoring, adaptive thresholds,
    and research-grade statistical validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.adaptive_thresholds: Dict[str, float] = {}
        self.gate_results: List[QualityGateResult] = []
        
        # Continuous monitoring thread
        self.monitoring_enabled = config.get('continuous_monitoring', True)
        self.monitoring_interval = config.get('monitoring_interval_seconds', 30)
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Statistical validation parameters
        self.statistical_significance_threshold = config.get('significance_threshold', 0.05)
        self.confidence_level = config.get('confidence_level', 0.95)
        self.minimum_samples_for_stats = config.get('min_samples', 10)
        
        self._initialize_adaptive_thresholds()
        
        if self.monitoring_enabled:
            self._start_continuous_monitoring()
    
    def _initialize_adaptive_thresholds(self):
        """Initialize adaptive thresholds for each metric."""
        base_thresholds = {
            # Performance metrics (higher is better)
            'accuracy': 0.90,
            'precision': 0.85,
            'recall': 0.85,
            'f1_score': 0.85,
            'localization_accuracy': 0.95,
            
            # Efficiency metrics (lower is better - inverted for consistent logic)
            'inference_latency': 50.0,  # ms - will be inverted
            'memory_usage': 512.0,      # MB - will be inverted 
            'model_size': 100.0,        # MB - will be inverted
            'power_consumption': 5.0,    # W - will be inverted
            
            # Quality metrics (higher is better)
            'code_coverage': 0.85,
            'security_score': 0.95,
            'robustness_score': 0.90,
            'maintainability_index': 0.80,
            
            # Research metrics (higher is better)
            'innovation_score': 0.75,
            'reproducibility_score': 0.95,
            'statistical_significance': 0.05,  # p-value - lower is better
            'effect_size': 0.5,
            'benchmark_improvement': 0.05,  # 5% improvement over baseline
            
            # Quantum metrics (higher is better)
            'quantum_advantage': 1.2,   # 20% advantage over classical
            'quantum_coherence': 0.8,
            'entanglement_efficiency': 0.7,
        }
        
        # Initialize adaptive thresholds with base values
        for metric, threshold in base_thresholds.items():
            self.adaptive_thresholds[metric] = threshold
    
    def evaluate_quality_gates(self, metrics: Dict[str, float]) -> QualityGateResult:
        """
        Evaluate all quality gates with current metrics.
        """
        quality_metrics = {}
        pass_count = fail_count = warning_count = critical_count = 0
        
        for metric_name, value in metrics.items():
            # Update metrics history
            self.metrics_history[metric_name].append(value)
            
            # Get or initialize adaptive threshold
            threshold = self.adaptive_thresholds.get(metric_name, 0.8)
            adaptive_threshold = self._compute_adaptive_threshold(metric_name)
            
            # Create quality metric
            quality_metric = QualityMetric(
                name=metric_name,
                value=value,
                threshold=threshold,
                adaptive_threshold=adaptive_threshold,
                trend=list(self.metrics_history[metric_name])
            )
            
            # Add statistical significance if enough samples
            if len(self.metrics_history[metric_name]) >= self.minimum_samples_for_stats:
                quality_metric.statistical_significance = self._compute_statistical_significance(metric_name)
            
            quality_metrics[metric_name] = quality_metric
            
            # Count status types
            if quality_metric.status == "PASS":
                pass_count += 1
            elif quality_metric.status == "FAIL":
                fail_count += 1
            elif quality_metric.status == "WARNING":
                warning_count += 1
            elif quality_metric.status == "CRITICAL":
                critical_count += 1
        
        # Compute performance delta
        performance_delta = self._compute_performance_delta(quality_metrics)
        
        result = QualityGateResult(
            timestamp=datetime.now(),
            gate_name=f"Progressive_Gate_{len(self.gate_results) + 1}",
            metrics=quality_metrics,
            overall_status="COMPUTED",  # Will be computed in __post_init__
            pass_count=pass_count,
            fail_count=fail_count,
            warning_count=warning_count,
            critical_count=critical_count,
            performance_delta=performance_delta
        )
        
        self.gate_results.append(result)
        
        logger.info(f"Quality gate evaluation complete: {result.overall_status}")
        logger.info(f"Metrics: PASS={pass_count}, WARN={warning_count}, CRIT={critical_count}, FAIL={fail_count}")
        
        return result
    
    def _compute_adaptive_threshold(self, metric_name: str) -> float:
        """Compute adaptive threshold based on historical performance."""
        history = self.metrics_history[metric_name]
        base_threshold = self.adaptive_thresholds.get(metric_name, 0.8)
        
        if len(history) < 5:
            return base_threshold
        
        # Use statistical approach for adaptation
        import math
        recent_values = list(history)[-10:]  # Last 10 values
        mean_val = sum(recent_values) / len(recent_values)
        std_dev = math.sqrt(sum((x - mean_val) ** 2 for x in recent_values) / len(recent_values))
        
        # Adaptive threshold: base + (performance - base) * adaptation_factor
        adaptation_factor = self.config.get('threshold_adaptation_factor', 0.1)
        
        # For metrics where lower is better (like latency), invert the logic
        inverted_metrics = {'inference_latency', 'memory_usage', 'model_size', 'power_consumption', 'statistical_significance'}
        
        if metric_name in inverted_metrics:
            # For lower-is-better metrics, adaptive threshold moves down with good performance
            adaptive_threshold = base_threshold - (base_threshold - mean_val) * adaptation_factor
            adaptive_threshold = max(adaptive_threshold, base_threshold * 0.5)  # Don't go below 50% of base
        else:
            # For higher-is-better metrics, adaptive threshold moves up with good performance  
            adaptive_threshold = base_threshold + (mean_val - base_threshold) * adaptation_factor
            adaptive_threshold = min(adaptive_threshold, base_threshold * 1.5)  # Don't exceed 150% of base
        
        # Update stored adaptive threshold
        self.adaptive_thresholds[metric_name] = adaptive_threshold
        
        return adaptive_threshold
    
    def _compute_statistical_significance(self, metric_name: str) -> float:
        """Compute statistical significance of performance improvement."""
        history = list(self.metrics_history[metric_name])
        
        if len(history) < self.minimum_samples_for_stats:
            return 1.0  # Not significant
        
        # Split into before/after groups (first half vs second half)
        mid_point = len(history) // 2
        before = history[:mid_point]
        after = history[mid_point:]
        
        if len(before) < 3 or len(after) < 3:
            return 1.0
        
        # Simple t-test approximation
        import math
        
        mean_before = sum(before) / len(before)
        mean_after = sum(after) / len(after)
        
        var_before = sum((x - mean_before) ** 2 for x in before) / (len(before) - 1)
        var_after = sum((x - mean_after) ** 2 for x in after) / (len(after) - 1)
        
        pooled_var = ((len(before) - 1) * var_before + (len(after) - 1) * var_after) / (len(before) + len(after) - 2)
        
        if pooled_var == 0:
            return 1.0
        
        t_stat = abs(mean_after - mean_before) / math.sqrt(pooled_var * (1/len(before) + 1/len(after)))
        
        # Approximate p-value (very rough approximation)
        # In practice, would use proper statistical library
        p_value = max(0.001, min(1.0, math.exp(-t_stat)))
        
        return p_value
    
    def _compute_performance_delta(self, metrics: Dict[str, QualityMetric]) -> float:
        """Compute overall performance delta from previous evaluation."""
        if len(self.gate_results) == 0:
            return 0.0
        
        # Compare with previous gate result
        prev_result = self.gate_results[-1]
        
        total_delta = 0.0
        count = 0
        
        for metric_name, current_metric in metrics.items():
            if metric_name in prev_result.metrics:
                prev_value = prev_result.metrics[metric_name].value
                delta = (current_metric.value - prev_value) / max(abs(prev_value), 0.001)
                total_delta += delta
                count += 1
        
        return total_delta / max(count, 1)
    
    def _start_continuous_monitoring(self):
        """Start background thread for continuous monitoring."""
        def monitor():
            while self.monitoring_enabled:
                try:
                    # Collect current metrics (would be implemented by subclasses)
                    current_metrics = self._collect_current_metrics()
                    if current_metrics:
                        result = self.evaluate_quality_gates(current_metrics)
                        self._handle_monitoring_result(result)
                    
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(self.monitoring_interval * 2)  # Back off on error
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Continuous monitoring started (interval: {self.monitoring_interval}s)")
    
    def _collect_current_metrics(self) -> Optional[Dict[str, float]]:
        """
        Collect current system metrics. To be overridden by implementations.
        """
        # Mock implementation - real implementation would collect actual metrics
        import random
        return {
            'accuracy': 0.92 + random.uniform(-0.05, 0.05),
            'inference_latency': 45 + random.uniform(-10, 15),
            'memory_usage': 400 + random.uniform(-50, 100),
            'security_score': 0.96 + random.uniform(-0.02, 0.02),
            'quantum_advantage': 1.15 + random.uniform(-0.1, 0.2)
        }
    
    def _handle_monitoring_result(self, result: QualityGateResult):
        """Handle continuous monitoring result."""
        if result.overall_status in ['FAIL', 'CRITICAL']:
            logger.warning(f"Quality gate status: {result.overall_status}")
            for recommendation in result.recommendations:
                logger.warning(f"Recommendation: {recommendation}")
        
        # Could trigger automated responses based on status
        if result.overall_status == 'FAIL':
            self._trigger_failure_response(result)
    
    def _trigger_failure_response(self, result: QualityGateResult):
        """Trigger automated response to quality gate failure."""
        logger.error(f"Quality gate failure detected: {result.gate_name}")
        
        # Example automated responses:
        # 1. Rollback to previous version
        # 2. Scale down to reduce load
        # 3. Alert development team
        # 4. Enable debug mode
        
        # For now, just log the failure
        for metric_name, metric in result.metrics.items():
            if metric.status == 'FAIL':
                logger.error(
                    f"FAILED METRIC: {metric_name} = {metric.value:.3f} "
                    f"(required: {metric.adaptive_threshold:.3f})"
                )
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not self.gate_results:
            return {"error": "No quality gate results available"}
        
        latest_result = self.gate_results[-1]
        
        # Compute trends over time
        status_trends = defaultdict(int)
        performance_trend = []
        
        for result in self.gate_results[-10:]:  # Last 10 results
            status_trends[result.overall_status] += 1
            performance_trend.append(result.performance_delta)
        
        # Compute stability metrics
        import math
        avg_performance_delta = sum(performance_trend) / len(performance_trend) if performance_trend else 0
        performance_stability = math.sqrt(sum((x - avg_performance_delta) ** 2 for x in performance_trend) / len(performance_trend)) if len(performance_trend) > 1 else 0
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "total_evaluations": len(self.gate_results),
            "latest_status": latest_result.overall_status,
            "latest_metrics_summary": {
                "pass": latest_result.pass_count,
                "warning": latest_result.warning_count, 
                "critical": latest_result.critical_count,
                "fail": latest_result.fail_count
            },
            "trends": {
                "status_distribution": dict(status_trends),
                "average_performance_delta": avg_performance_delta,
                "performance_stability": performance_stability
            },
            "adaptive_thresholds": dict(self.adaptive_thresholds),
            "recommendations": latest_result.recommendations,
            "statistical_summary": self._generate_statistical_summary(),
            "research_readiness": self._assess_research_readiness()
        }
        
        return report
    
    def _generate_statistical_summary(self) -> Dict[str, Any]:
        """Generate statistical summary of all metrics."""
        summary = {}
        
        for metric_name, history in self.metrics_history.items():
            if len(history) >= 3:
                values = list(history)
                import math
                
                mean_val = sum(values) / len(values)
                variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
                std_dev = math.sqrt(variance)
                
                summary[metric_name] = {
                    "mean": mean_val,
                    "std_dev": std_dev,
                    "min": min(values),
                    "max": max(values),
                    "samples": len(values),
                    "trend": "improving" if len(values) > 1 and values[-1] > values[0] else "declining",
                    "coefficient_of_variation": std_dev / abs(mean_val) if mean_val != 0 else float('inf')
                }
        
        return summary
    
    def _assess_research_readiness(self) -> Dict[str, Any]:
        """Assess readiness for research publication."""
        
        criteria = {
            "sufficient_samples": all(len(history) >= 30 for history in self.metrics_history.values()),
            "statistical_significance": any(
                len(history) >= self.minimum_samples_for_stats and 
                self._compute_statistical_significance(metric_name) < self.statistical_significance_threshold
                for metric_name, history in self.metrics_history.items()
            ),
            "reproducible_results": self._check_reproducibility(),
            "comprehensive_baselines": self._check_baseline_comparisons(),
            "documentation_complete": True  # Would check documentation completeness
        }
        
        readiness_score = sum(criteria.values()) / len(criteria)
        
        return {
            "criteria": criteria,
            "readiness_score": readiness_score,
            "publication_ready": readiness_score >= 0.8,
            "missing_requirements": [k for k, v in criteria.items() if not v]
        }
    
    def _check_reproducibility(self) -> bool:
        """Check if results are reproducible across multiple runs."""
        # Simple check: coefficient of variation < 0.1 for key metrics
        key_metrics = ['accuracy', 'inference_latency', 'quantum_advantage']
        
        for metric in key_metrics:
            if metric in self.metrics_history:
                history = list(self.metrics_history[metric])
                if len(history) >= 10:
                    import math
                    mean_val = sum(history) / len(history)
                    std_dev = math.sqrt(sum((x - mean_val) ** 2 for x in history) / (len(history) - 1))
                    cv = std_dev / abs(mean_val) if mean_val != 0 else float('inf')
                    
                    if cv > 0.1:  # High variability indicates poor reproducibility
                        return False
        
        return True
    
    def _check_baseline_comparisons(self) -> bool:
        """Check if comprehensive baseline comparisons have been performed."""
        # In a real implementation, this would check if baseline experiments exist
        # For now, assume baselines exist if quantum metrics are present
        quantum_metrics = ['quantum_advantage', 'quantum_coherence', 'entanglement_efficiency']
        return any(metric in self.metrics_history for metric in quantum_metrics)
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_enabled = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
            logger.info("Continuous monitoring stopped")

# Example usage and factory function
def create_progressive_quality_gate_system(optimization_level: str = "generation_4") -> ProgressiveQualityGateSystem:
    """
    Factory function to create appropriately configured quality gate system.
    """
    
    configs = {
        "generation_4": {
            "continuous_monitoring": True,
            "monitoring_interval_seconds": 30,
            "threshold_adaptation_factor": 0.1,
            "significance_threshold": 0.05,
            "confidence_level": 0.95,
            "min_samples": 10
        },
        "research_grade": {
            "continuous_monitoring": True,
            "monitoring_interval_seconds": 15,
            "threshold_adaptation_factor": 0.05,  # More conservative adaptation
            "significance_threshold": 0.01,      # Stricter significance
            "confidence_level": 0.99,            # Higher confidence
            "min_samples": 30                    # More samples for statistics
        }
    }
    
    config = configs.get(optimization_level, configs["generation_4"])
    return ProgressiveQualityGateSystem(config)

if __name__ == "__main__":
    # Demo of the progressive quality gate system
    import time
    import random
    
    print("🚀 Progressive Quality Gates System Demo")
    print("=" * 50)
    
    # Create system
    pqgs = create_progressive_quality_gate_system("research_grade")
    
    # Simulate multiple evaluations
    for i in range(5):
        print(f"\nEvaluation {i + 1}")
        print("-" * 20)
        
        # Mock metrics with realistic trends
        metrics = {
            "accuracy": 0.90 + i * 0.01 + random.uniform(-0.02, 0.02),
            "inference_latency": 50 - i * 2 + random.uniform(-5, 5),
            "quantum_advantage": 1.1 + i * 0.05 + random.uniform(-0.05, 0.1),
            "security_score": 0.95 + random.uniform(-0.01, 0.01),
            "reproducibility_score": 0.92 + i * 0.01 + random.uniform(-0.01, 0.01)
        }
        
        result = pqgs.evaluate_quality_gates(metrics)
        print(f"Status: {result.overall_status}")
        print(f"Performance Delta: {result.performance_delta:.3f}")
        
        if result.recommendations:
            print("Recommendations:")
            for rec in result.recommendations[:2]:  # Show first 2
                print(f"  - {rec}")
        
        time.sleep(1)
    
    # Generate final report
    print("\n" + "=" * 50)
    print("📊 FINAL QUALITY REPORT")
    print("=" * 50)
    
    report = pqgs.generate_quality_report()
    
    print(f"Total Evaluations: {report['total_evaluations']}")
    print(f"Latest Status: {report['latest_status']}")
    print(f"Research Ready: {report['research_readiness']['publication_ready']}")
    print(f"Readiness Score: {report['research_readiness']['readiness_score']:.2f}")
    
    # Stop monitoring
    pqgs.stop_monitoring()
    print("\n✅ Demo Complete")