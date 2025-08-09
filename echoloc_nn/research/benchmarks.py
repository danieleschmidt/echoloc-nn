"""
Comprehensive Research Benchmarking Suite

Provides rigorous benchmarking, statistical validation, and performance
profiling tools for academic research in quantum-spatial localization.
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections import defaultdict
import statistics


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result with statistical metadata"""
    algorithm_name: str
    dataset_name: str
    accuracy_mean: float
    accuracy_std: float
    accuracy_p95: float
    accuracy_p99: float
    latency_mean: float
    latency_std: float
    throughput: float
    memory_usage: float
    energy_consumption: float
    convergence_rate: float
    statistical_significance: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    metadata: Dict[str, Any]
    

@dataclass 
class StatisticalTest:
    """Statistical test result"""
    test_name: str
    p_value: float
    test_statistic: float
    critical_value: float
    effect_size: float
    power: float
    is_significant: bool
    interpretation: str


class ResearchBenchmarkSuite:
    """
    Comprehensive benchmarking suite for research algorithms.
    
    Provides standardized evaluation across multiple metrics with
    rigorous statistical analysis and reproducibility controls.
    """
    
    def __init__(self, random_seed: int = 42, n_bootstrap_samples: int = 1000):
        self.random_seed = random_seed
        self.n_bootstrap = n_bootstrap_samples
        np.random.seed(random_seed)
        
        self.benchmark_results = {}
        self.comparison_results = {}
        self.statistical_tests = {}
        
    def run_algorithm_benchmark(self, 
                               algorithm: Callable,
                               algorithm_name: str,
                               test_datasets: List[Tuple[str, np.ndarray]],
                               n_trials: int = 10,
                               **algorithm_kwargs) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive benchmark of an algorithm across multiple datasets.
        
        Args:
            algorithm: Algorithm function to test
            algorithm_name: Name for identification
            test_datasets: List of (dataset_name, data) pairs
            n_trials: Number of trials per dataset
            **algorithm_kwargs: Arguments passed to algorithm
            
        Returns:
            Dictionary mapping dataset names to benchmark results
        """
        results = {}
        
        for dataset_name, dataset in test_datasets:
            print(f"Benchmarking {algorithm_name} on {dataset_name}...")
            
            # Run multiple trials
            trial_results = []
            for trial in range(n_trials):
                trial_start = time.time()
                
                # Run algorithm
                try:
                    algorithm_result = algorithm(dataset, **algorithm_kwargs)
                    trial_time = time.time() - trial_start
                    
                    # Extract metrics
                    accuracy = getattr(algorithm_result, 'accuracy', 0.0)
                    energy = getattr(algorithm_result, 'energy', 0.0)
                    convergence_time = getattr(algorithm_result, 'convergence_time', trial_time)
                    
                    trial_results.append({
                        'accuracy': accuracy,
                        'latency': trial_time,
                        'energy': energy,
                        'convergence_time': convergence_time,
                        'trial': trial
                    })
                    
                except Exception as e:
                    print(f"Trial {trial} failed: {e}")
                    trial_results.append({
                        'accuracy': 0.0,
                        'latency': float('inf'),
                        'energy': float('inf'),
                        'convergence_time': float('inf'),
                        'trial': trial,
                        'error': str(e)
                    })
            
            # Calculate statistics
            accuracies = [r['accuracy'] for r in trial_results if np.isfinite(r['accuracy'])]
            latencies = [r['latency'] for r in trial_results if np.isfinite(r['latency'])]
            
            if accuracies and latencies:
                benchmark_result = BenchmarkResult(
                    algorithm_name=algorithm_name,
                    dataset_name=dataset_name,
                    accuracy_mean=np.mean(accuracies),
                    accuracy_std=np.std(accuracies),
                    accuracy_p95=np.percentile(accuracies, 95),
                    accuracy_p99=np.percentile(accuracies, 99),
                    latency_mean=np.mean(latencies),
                    latency_std=np.std(latencies),
                    throughput=1.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0.0,
                    memory_usage=self._estimate_memory_usage(dataset),
                    energy_consumption=np.mean([r['energy'] for r in trial_results]),
                    convergence_rate=len([r for r in trial_results if r.get('error') is None]) / n_trials,
                    statistical_significance=self._calculate_significance(accuracies),
                    effect_size=self._calculate_effect_size(accuracies),
                    confidence_interval=self._bootstrap_confidence_interval(accuracies),
                    metadata={
                        'n_trials': n_trials,
                        'dataset_size': dataset.shape if hasattr(dataset, 'shape') else len(dataset),
                        'algorithm_kwargs': algorithm_kwargs,
                        'random_seed': self.random_seed,
                        'trial_results': trial_results
                    }
                )
                
                results[dataset_name] = benchmark_result
                
                # Store for later comparison
                if algorithm_name not in self.benchmark_results:
                    self.benchmark_results[algorithm_name] = {}
                self.benchmark_results[algorithm_name][dataset_name] = benchmark_result
        
        return results
    
    def compare_algorithms(self, 
                          algorithm_results: Dict[str, Dict[str, BenchmarkResult]],
                          metric: str = 'accuracy_mean',
                          alpha: float = 0.05) -> Dict[str, StatisticalTest]:
        """
        Compare multiple algorithms statistically.
        
        Args:
            algorithm_results: Nested dict {algorithm: {dataset: result}}
            metric: Metric to compare ('accuracy_mean', 'latency_mean', etc.)
            alpha: Significance level
            
        Returns:
            Statistical test results for pairwise comparisons
        """
        comparisons = {}
        
        # Get all algorithm pairs
        algorithms = list(algorithm_results.keys())
        datasets = set()
        for alg_results in algorithm_results.values():
            datasets.update(alg_results.keys())
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms[i+1:], i+1):
                
                # Collect data across datasets
                alg1_values = []
                alg2_values = []
                
                for dataset in datasets:
                    if dataset in algorithm_results[alg1] and dataset in algorithm_results[alg2]:
                        val1 = getattr(algorithm_results[alg1][dataset], metric)
                        val2 = getattr(algorithm_results[alg2][dataset], metric)
                        alg1_values.append(val1)
                        alg2_values.append(val2)
                
                if len(alg1_values) >= 2:  # Need at least 2 points for statistical test
                    # Perform Welch's t-test (unequal variance)
                    stat_test = self._welch_t_test(alg1_values, alg2_values, alpha)
                    
                    comparison_name = f"{alg1}_vs_{alg2}_{metric}"
                    comparisons[comparison_name] = stat_test
        
        self.comparison_results = comparisons
        return comparisons
    
    def generate_performance_profile(self, 
                                   algorithm_results: Dict[str, BenchmarkResult],
                                   baseline_algorithm: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance profile.
        
        Args:
            algorithm_results: Results for a single algorithm across datasets
            baseline_algorithm: Optional baseline for relative metrics
            
        Returns:
            Performance profile dictionary
        """
        profile = {
            'algorithm_summary': {
                'mean_accuracy': np.mean([r.accuracy_mean for r in algorithm_results.values()]),
                'mean_latency': np.mean([r.latency_mean for r in algorithm_results.values()]),
                'mean_throughput': np.mean([r.throughput for r in algorithm_results.values()]),
                'reliability': np.mean([r.convergence_rate for r in algorithm_results.values()]),
                'consistency': 1.0 - np.mean([r.accuracy_std for r in algorithm_results.values()])
            },
            'dataset_breakdown': {},
            'scaling_analysis': self._analyze_scaling_behavior(algorithm_results),
            'robustness_metrics': self._calculate_robustness_metrics(algorithm_results)
        }
        
        # Per-dataset analysis
        for dataset_name, result in algorithm_results.items():
            profile['dataset_breakdown'][dataset_name] = {
                'accuracy_stats': {
                    'mean': result.accuracy_mean,
                    'std': result.accuracy_std,
                    'p95': result.accuracy_p95,
                    'p99': result.accuracy_p99
                },
                'performance_stats': {
                    'latency_mean': result.latency_mean,
                    'latency_std': result.latency_std,
                    'throughput': result.throughput
                },
                'efficiency_ratio': result.accuracy_mean / max(result.latency_mean, 1e-8),
                'quality_score': self._calculate_quality_score(result)
            }
        
        # Relative performance if baseline provided
        if baseline_algorithm and baseline_algorithm in self.benchmark_results:
            profile['relative_performance'] = self._calculate_relative_performance(
                algorithm_results, self.benchmark_results[baseline_algorithm]
            )
        
        return profile
    
    def _estimate_memory_usage(self, dataset: np.ndarray) -> float:
        """Estimate memory usage in MB"""
        if hasattr(dataset, 'nbytes'):
            return dataset.nbytes / (1024**2)  # Convert to MB
        else:
            return 0.0
    
    def _calculate_significance(self, values: List[float]) -> float:
        """Calculate statistical significance (p-value from one-sample t-test against 0)"""
        if len(values) < 2:
            return 1.0
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        n = len(values)
        
        if std == 0:
            return 0.0 if mean != 0 else 1.0
        
        t_stat = mean / (std / np.sqrt(n))
        # Approximate p-value (simplified)
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), n-1))
        
        return p_value
    
    def _calculate_effect_size(self, values: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        if len(values) < 2:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if std == 0:
            return float('inf') if mean != 0 else 0.0
        
        # Effect size relative to population (assuming population mean = 0.5 for accuracy)
        population_mean = 0.5
        cohens_d = (mean - population_mean) / std
        
        return abs(cohens_d)
    
    def _bootstrap_confidence_interval(self, values: List[float], 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        if len(values) < 2:
            return (0.0, 0.0)
        
        bootstrap_means = []
        n = len(values)
        
        for _ in range(self.n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, (alpha/2) * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return (lower, upper)
    
    def _welch_t_test(self, group1: List[float], group2: List[float], 
                      alpha: float = 0.05) -> StatisticalTest:
        """Perform Welch's t-test for unequal variances"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        if var1 == 0 and var2 == 0:
            # Both groups have no variance
            t_stat = 0.0 if mean1 == mean2 else float('inf')
            p_value = 1.0 if mean1 == mean2 else 0.0
        else:
            # Standard Welch's t-test
            pooled_se = np.sqrt(var1/n1 + var2/n2)
            t_stat = (mean1 - mean2) / pooled_se if pooled_se > 0 else 0.0
            
            # Welch-Satterthwaite degrees of freedom
            if var1/n1 + var2/n2 > 0:
                df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
            else:
                df = n1 + n2 - 2
            
            # Approximate p-value
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        # Critical value (approximate)
        critical_value = self._t_critical(alpha/2, n1+n2-2)
        
        # Statistical power (simplified estimate)
        power = 1 - self._beta_error_estimate(cohens_d, n1, n2, alpha)
        
        return StatisticalTest(
            test_name="Welch's t-test",
            p_value=p_value,
            test_statistic=t_stat,
            critical_value=critical_value,
            effect_size=cohens_d,
            power=power,
            is_significant=p_value < alpha,
            interpretation=self._interpret_t_test(p_value, cohens_d, alpha)
        )
    
    def _t_cdf(self, t: float, df: float) -> float:
        """Approximate t-distribution CDF (simplified)"""
        # Very rough approximation for demonstration
        if df > 30:
            # Use normal approximation for large df
            return 0.5 * (1 + np.tanh(t / np.sqrt(2)))
        else:
            # Rough approximation for small df
            return 0.5 + 0.5 * np.tanh(t / np.sqrt(df + t**2))
    
    def _t_critical(self, alpha: float, df: float) -> float:
        """Approximate critical t-value"""
        # Simplified approximation
        if df > 30:
            return 1.96  # Normal approximation
        else:
            return 2.0 + 0.5 / np.sqrt(df)  # Rough approximation
    
    def _beta_error_estimate(self, effect_size: float, n1: int, n2: int, alpha: float) -> float:
        """Estimate Type II error probability (simplified)"""
        # Very rough power analysis estimate
        total_n = n1 + n2
        ncp = effect_size * np.sqrt(total_n / 4)  # Non-centrality parameter
        power_estimate = 1 / (1 + np.exp(-ncp + 2))  # Sigmoid approximation
        return 1 - power_estimate
    
    def _interpret_t_test(self, p_value: float, effect_size: float, alpha: float) -> str:
        """Interpret t-test results"""
        if p_value < alpha:
            significance = "statistically significant"
        else:
            significance = "not statistically significant"
        
        if effect_size < 0.2:
            effect = "negligible"
        elif effect_size < 0.5:
            effect = "small"
        elif effect_size < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        return f"Difference is {significance} (p={p_value:.4f}) with {effect} effect size (d={effect_size:.3f})"
    
    def _analyze_scaling_behavior(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Analyze algorithm scaling behavior across datasets"""
        dataset_sizes = []
        latencies = []
        accuracies = []
        
        for result in results.values():
            size = result.metadata.get('dataset_size', 0)
            if hasattr(size, '__len__'):  # If size is a shape tuple
                size = np.prod(size)
            dataset_sizes.append(size)
            latencies.append(result.latency_mean)
            accuracies.append(result.accuracy_mean)
        
        scaling_analysis = {
            'time_complexity_estimate': self._estimate_time_complexity(dataset_sizes, latencies),
            'accuracy_scaling': self._estimate_accuracy_scaling(dataset_sizes, accuracies),
            'efficiency_trend': self._calculate_efficiency_trend(dataset_sizes, latencies, accuracies)
        }
        
        return scaling_analysis
    
    def _estimate_time_complexity(self, sizes: List[int], times: List[float]) -> str:
        """Estimate algorithmic time complexity"""
        if len(sizes) < 3:
            return "insufficient data"
        
        # Test different complexity models
        log_sizes = [np.log(s) if s > 0 else 0 for s in sizes]
        
        # Linear correlation with different complexity functions
        linear_corr = np.corrcoef(sizes, times)[0, 1] if len(sizes) > 1 else 0
        loglinear_corr = np.corrcoef(log_sizes, times)[0, 1] if len(sizes) > 1 else 0
        quadratic_corr = np.corrcoef([s**2 for s in sizes], times)[0, 1] if len(sizes) > 1 else 0
        
        correlations = {
            'O(n)': abs(linear_corr),
            'O(n log n)': abs(loglinear_corr), 
            'O(n²)': abs(quadratic_corr)
        }
        
        best_fit = max(correlations, key=correlations.get)
        confidence = correlations[best_fit]
        
        return f"{best_fit} (confidence: {confidence:.2f})"
    
    def _estimate_accuracy_scaling(self, sizes: List[int], accuracies: List[float]) -> Dict[str, float]:
        """Analyze how accuracy scales with dataset size"""
        if len(sizes) < 2:
            return {'trend': 0.0, 'correlation': 0.0}
        
        correlation = np.corrcoef(sizes, accuracies)[0, 1]
        
        # Linear trend
        if len(sizes) > 1:
            slope = (accuracies[-1] - accuracies[0]) / (sizes[-1] - sizes[0]) if sizes[-1] != sizes[0] else 0
        else:
            slope = 0
        
        return {
            'trend': slope,  # Positive = accuracy improves with more data
            'correlation': correlation,
            'interpretation': 'improving' if slope > 0.01 else 'stable' if abs(slope) < 0.01 else 'degrading'
        }
    
    def _calculate_efficiency_trend(self, sizes: List[int], times: List[float], 
                                  accuracies: List[float]) -> Dict[str, float]:
        """Calculate efficiency (accuracy/time) trend"""
        if len(sizes) < 2:
            return {'trend': 0.0, 'mean_efficiency': 0.0}
        
        efficiencies = [acc/time if time > 0 else 0 for acc, time in zip(accuracies, times)]
        mean_efficiency = np.mean(efficiencies)
        
        if len(sizes) > 1:
            efficiency_trend = (efficiencies[-1] - efficiencies[0]) / (sizes[-1] - sizes[0]) if sizes[-1] != sizes[0] else 0
        else:
            efficiency_trend = 0
        
        return {
            'trend': efficiency_trend,
            'mean_efficiency': mean_efficiency,
            'best_efficiency': max(efficiencies),
            'worst_efficiency': min(efficiencies)
        }
    
    def _calculate_robustness_metrics(self, results: Dict[str, BenchmarkResult]) -> Dict[str, float]:
        """Calculate algorithm robustness across different conditions"""
        accuracies = [r.accuracy_mean for r in results.values()]
        latencies = [r.latency_mean for r in results.values()]
        convergence_rates = [r.convergence_rate for r in results.values()]
        
        return {
            'accuracy_stability': 1.0 - (np.std(accuracies) / max(np.mean(accuracies), 1e-8)),
            'latency_consistency': 1.0 - (np.std(latencies) / max(np.mean(latencies), 1e-8)),
            'reliability': np.mean(convergence_rates),
            'worst_case_accuracy': min(accuracies) if accuracies else 0.0,
            'best_case_accuracy': max(accuracies) if accuracies else 0.0,
            'robustness_score': np.mean(convergence_rates) * (1.0 - np.std(accuracies))
        }
    
    def _calculate_quality_score(self, result: BenchmarkResult) -> float:
        """Calculate composite quality score"""
        # Weighted combination of metrics
        accuracy_weight = 0.4
        speed_weight = 0.3
        reliability_weight = 0.2
        consistency_weight = 0.1
        
        # Normalize metrics (0-1 scale)
        accuracy_score = min(result.accuracy_mean, 1.0)
        speed_score = min(1.0 / max(result.latency_mean, 0.001), 1.0)  # Invert for speed
        reliability_score = result.convergence_rate
        consistency_score = max(0, 1.0 - result.accuracy_std)
        
        quality_score = (
            accuracy_weight * accuracy_score +
            speed_weight * speed_score +
            reliability_weight * reliability_score +
            consistency_weight * consistency_score
        )
        
        return quality_score
    
    def _calculate_relative_performance(self, current_results: Dict[str, BenchmarkResult],
                                      baseline_results: Dict[str, BenchmarkResult]) -> Dict[str, Dict[str, float]]:
        """Calculate performance relative to baseline algorithm"""
        relative_performance = {}
        
        for dataset_name in current_results:
            if dataset_name in baseline_results:
                current = current_results[dataset_name]
                baseline = baseline_results[dataset_name]
                
                relative_performance[dataset_name] = {
                    'accuracy_improvement': (current.accuracy_mean - baseline.accuracy_mean) / max(baseline.accuracy_mean, 1e-8),
                    'speed_improvement': (baseline.latency_mean - current.latency_mean) / max(baseline.latency_mean, 1e-8),
                    'reliability_improvement': current.convergence_rate - baseline.convergence_rate,
                    'overall_improvement': self._calculate_quality_score(current) - self._calculate_quality_score(baseline)
                }
        
        return relative_performance


class AlgorithmComparator:
    """Specialized tool for comparing research algorithms"""
    
    def __init__(self):
        self.comparison_history = []
        
    def head_to_head_comparison(self, 
                               algorithm1: Callable, algorithm1_name: str,
                               algorithm2: Callable, algorithm2_name: str,
                               test_data: np.ndarray,
                               n_trials: int = 20) -> Dict[str, Any]:
        """Direct head-to-head algorithm comparison"""
        
        results1 = []
        results2 = []
        
        for trial in range(n_trials):
            # Run both algorithms on the same data
            start_time = time.time()
            result1 = algorithm1(test_data)
            time1 = time.time() - start_time
            
            start_time = time.time()
            result2 = algorithm2(test_data)
            time2 = time.time() - start_time
            
            results1.append({'accuracy': getattr(result1, 'accuracy', 0), 'time': time1})
            results2.append({'accuracy': getattr(result2, 'accuracy', 0), 'time': time2})
        
        # Statistical comparison
        acc1 = [r['accuracy'] for r in results1]
        acc2 = [r['accuracy'] for r in results2]
        time1 = [r['time'] for r in results1]
        time2 = [r['time'] for r in results2]
        
        comparison = {
            'winner': {
                'accuracy': algorithm1_name if np.mean(acc1) > np.mean(acc2) else algorithm2_name,
                'speed': algorithm1_name if np.mean(time1) < np.mean(time2) else algorithm2_name
            },
            'accuracy_difference': np.mean(acc1) - np.mean(acc2),
            'speed_ratio': np.mean(time2) / max(np.mean(time1), 1e-8),
            'statistical_significance': self._paired_t_test(acc1, acc2),
            'effect_size': abs(np.mean(acc1) - np.mean(acc2)) / np.sqrt((np.var(acc1) + np.var(acc2)) / 2)
        }
        
        self.comparison_history.append(comparison)
        return comparison
    
    def _paired_t_test(self, group1: List[float], group2: List[float]) -> float:
        """Paired t-test for dependent samples"""
        differences = [a - b for a, b in zip(group1, group2)]
        if len(differences) < 2:
            return 1.0
        
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        if std_diff == 0:
            return 0.0 if mean_diff != 0 else 1.0
        
        t_stat = mean_diff / (std_diff / np.sqrt(len(differences)))
        # Simplified p-value calculation
        p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + len(differences) - 1))
        
        return p_value


class StatisticalValidator:
    """Statistical validation tools for research claims"""
    
    @staticmethod
    def validate_improvement_claim(baseline_results: List[float],
                                 improved_results: List[float],
                                 alpha: float = 0.05,
                                 min_effect_size: float = 0.5) -> Dict[str, Any]:
        """Validate claims of algorithmic improvement"""
        
        # Statistical significance test
        if len(baseline_results) != len(improved_results):
            return {'valid': False, 'reason': 'Unequal sample sizes'}
        
        differences = [imp - base for imp, base in zip(improved_results, baseline_results)]
        mean_diff = np.mean(differences)
        
        if len(differences) < 2:
            return {'valid': False, 'reason': 'Insufficient data'}
        
        std_diff = np.std(differences, ddof=1)
        t_stat = mean_diff / (std_diff / np.sqrt(len(differences))) if std_diff > 0 else 0
        
        # Approximate p-value
        p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + len(differences) - 1))
        
        # Effect size
        pooled_std = np.sqrt((np.var(baseline_results, ddof=1) + np.var(improved_results, ddof=1)) / 2)
        effect_size = abs(mean_diff) / pooled_std if pooled_std > 0 else 0
        
        is_significant = p_value < alpha
        has_practical_significance = effect_size >= min_effect_size
        
        validation = {
            'valid': is_significant and has_practical_significance and mean_diff > 0,
            'statistical_significance': is_significant,
            'practical_significance': has_practical_significance,
            'p_value': p_value,
            'effect_size': effect_size,
            'mean_improvement': mean_diff,
            'confidence_interval': StatisticalValidator._confidence_interval(differences),
            'interpretation': StatisticalValidator._interpret_validation(
                is_significant, has_practical_significance, mean_diff, effect_size
            )
        }
        
        return validation
    
    @staticmethod
    def _confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean"""
        if len(data) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        margin = 1.96 * std / np.sqrt(len(data))  # Approximate for 95% CI
        
        return (mean - margin, mean + margin)
    
    @staticmethod
    def _interpret_validation(is_significant: bool, has_practical_significance: bool, 
                            mean_improvement: float, effect_size: float) -> str:
        """Interpret validation results"""
        if is_significant and has_practical_significance and mean_improvement > 0:
            return f"Valid improvement claim: statistically significant (effect size: {effect_size:.2f})"
        elif not is_significant:
            return "Invalid: not statistically significant"
        elif not has_practical_significance:
            return f"Invalid: effect size too small ({effect_size:.2f})"
        elif mean_improvement <= 0:
            return "Invalid: no improvement shown"
        else:
            return "Invalid: multiple criteria failed"


class PerformanceProfiler:
    """Detailed performance profiling for algorithms"""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_algorithm(self, algorithm: Callable, algorithm_name: str,
                         test_inputs: List[np.ndarray]) -> Dict[str, Any]:
        """Create detailed performance profile"""
        
        profile = {
            'memory_usage': [],
            'execution_times': [],
            'accuracy_scores': [],
            'input_characteristics': [],
            'bottlenecks': []
        }
        
        for i, test_input in enumerate(test_inputs):
            # Profile memory usage (simplified)
            memory_before = 0  # Would use memory profiling in practice
            
            start_time = time.time()
            result = algorithm(test_input)
            execution_time = time.time() - start_time
            
            memory_after = 0  # Would measure actual memory usage
            
            profile['execution_times'].append(execution_time)
            profile['memory_usage'].append(memory_after - memory_before)
            profile['accuracy_scores'].append(getattr(result, 'accuracy', 0))
            
            # Input characteristics
            if hasattr(test_input, 'shape'):
                profile['input_characteristics'].append({
                    'size': np.prod(test_input.shape),
                    'shape': test_input.shape,
                    'dtype': str(test_input.dtype),
                    'sparsity': np.count_nonzero(test_input) / test_input.size
                })
        
        # Analyze bottlenecks
        profile['bottlenecks'] = self._identify_bottlenecks(profile)
        
        # Summary statistics
        profile['summary'] = {
            'mean_execution_time': np.mean(profile['execution_times']),
            'mean_memory_usage': np.mean(profile['memory_usage']),
            'mean_accuracy': np.mean(profile['accuracy_scores']),
            'time_complexity_estimate': self._estimate_complexity(profile),
            'scalability_rating': self._rate_scalability(profile)
        }
        
        self.profiles[algorithm_name] = profile
        return profile
    
    def _identify_bottlenecks(self, profile: Dict[str, List]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Check for time bottlenecks
        times = profile['execution_times']
        if times and np.std(times) > np.mean(times):
            bottlenecks.append("High execution time variance - inconsistent performance")
        
        # Check for memory bottlenecks
        memory = profile['memory_usage']
        if memory and max(memory) > 2 * np.mean(memory):
            bottlenecks.append("Memory usage spikes detected")
        
        # Check for scaling issues
        input_sizes = [char['size'] for char in profile['input_characteristics']]
        if len(input_sizes) > 1:
            size_time_corr = np.corrcoef(input_sizes, times)[0, 1] if len(times) > 1 else 0
            if size_time_corr > 0.8:
                bottlenecks.append("Strong correlation between input size and execution time")
        
        return bottlenecks
    
    def _estimate_complexity(self, profile: Dict[str, List]) -> str:
        """Estimate algorithmic complexity from profiling data"""
        input_sizes = [char['size'] for char in profile['input_characteristics']]
        times = profile['execution_times']
        
        if len(input_sizes) < 3 or len(times) < 3:
            return "insufficient data"
        
        # Test different complexity relationships
        linear_fit = np.corrcoef(input_sizes, times)[0, 1]
        quadratic_fit = np.corrcoef([s**2 for s in input_sizes], times)[0, 1]
        log_fit = np.corrcoef([np.log(s) if s > 0 else 0 for s in input_sizes], times)[0, 1]
        
        fits = {'O(n)': abs(linear_fit), 'O(n²)': abs(quadratic_fit), 'O(log n)': abs(log_fit)}
        best_fit = max(fits, key=fits.get)
        
        return f"{best_fit} (correlation: {fits[best_fit]:.2f})"
    
    def _rate_scalability(self, profile: Dict[str, List]) -> str:
        """Rate algorithm scalability"""
        times = profile['execution_times']
        
        if not times:
            return "unknown"
        
        # Simple heuristics for scalability rating
        mean_time = np.mean(times)
        time_variance = np.var(times)
        
        if mean_time < 0.1 and time_variance < 0.01:
            return "excellent"
        elif mean_time < 1.0 and time_variance < 0.1:
            return "good"
        elif mean_time < 10.0:
            return "moderate"
        else:
            return "poor"