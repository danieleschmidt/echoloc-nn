"""
Comparative Studies Framework for Academic Research

Provides A/B testing, experimental design, and statistical analysis
tools for comparing quantum-spatial localization algorithms.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod


@dataclass
class StudyDesign:
    """Experimental study design specification"""
    study_name: str
    hypothesis: str
    null_hypothesis: str
    independent_variables: List[str]
    dependent_variables: List[str]
    control_conditions: Dict[str, Any]
    experimental_conditions: List[Dict[str, Any]]
    sample_size: int
    randomization_method: str
    significance_level: float = 0.05
    power: float = 0.8


@dataclass
class StudyResult:
    """Result from a comparative study"""
    study_name: str
    algorithm_name: str
    condition_name: str
    measurements: Dict[str, List[float]]
    summary_statistics: Dict[str, Dict[str, float]]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    metadata: Dict[str, Any]


class BaselineComparator:
    """
    Systematic comparison against established baselines
    for novel quantum-spatial algorithms.
    """
    
    def __init__(self):
        self.baseline_algorithms = {}
        self.comparison_history = []
        
    def register_baseline(self, name: str, algorithm: Callable, description: str):
        """Register a baseline algorithm for comparison"""
        self.baseline_algorithms[name] = {
            'algorithm': algorithm,
            'description': description,
            'registration_time': time.time()
        }
    
    def compare_against_baselines(self, 
                                novel_algorithm: Callable,
                                novel_algorithm_name: str,
                                test_datasets: List[Tuple[str, np.ndarray]],
                                n_trials: int = 10) -> Dict[str, Any]:
        """
        Compare novel algorithm against all registered baselines.
        
        Args:
            novel_algorithm: The new algorithm to evaluate
            novel_algorithm_name: Name for the new algorithm
            test_datasets: List of (dataset_name, data) tuples
            n_trials: Number of trials per algorithm-dataset combination
            
        Returns:
            Comprehensive comparison results
        """
        
        comparison_results = {
            'novel_algorithm': novel_algorithm_name,
            'baseline_comparisons': {},
            'overall_performance': {},
            'statistical_significance': {},
            'practical_significance': {}
        }
        
        # Test novel algorithm
        novel_results = self._evaluate_algorithm(
            novel_algorithm, novel_algorithm_name, test_datasets, n_trials
        )
        
        # Compare against each baseline
        for baseline_name, baseline_info in self.baseline_algorithms.items():
            baseline_algorithm = baseline_info['algorithm']
            
            # Test baseline algorithm
            baseline_results = self._evaluate_algorithm(
                baseline_algorithm, baseline_name, test_datasets, n_trials
            )
            
            # Perform statistical comparison
            statistical_comparison = self._statistical_comparison(
                novel_results, baseline_results, baseline_name
            )
            
            comparison_results['baseline_comparisons'][baseline_name] = {
                'novel_performance': novel_results,
                'baseline_performance': baseline_results,
                'statistical_tests': statistical_comparison,
                'improvement_metrics': self._calculate_improvement_metrics(
                    novel_results, baseline_results
                )
            }
        
        # Calculate overall performance summary
        comparison_results['overall_performance'] = self._summarize_overall_performance(
            novel_results, comparison_results['baseline_comparisons']
        )
        
        self.comparison_history.append(comparison_results)
        return comparison_results
    
    def _evaluate_algorithm(self, algorithm: Callable, algorithm_name: str,
                          test_datasets: List[Tuple[str, np.ndarray]], 
                          n_trials: int) -> Dict[str, Any]:
        """Evaluate algorithm performance across datasets"""
        
        results = {
            'algorithm_name': algorithm_name,
            'dataset_results': {},
            'aggregate_metrics': {}
        }
        
        all_accuracies = []
        all_times = []
        all_energies = []
        
        for dataset_name, dataset in test_datasets:
            dataset_trials = []
            
            for trial in range(n_trials):
                start_time = time.time()
                
                try:
                    result = algorithm(dataset)
                    execution_time = time.time() - start_time
                    
                    trial_data = {
                        'accuracy': getattr(result, 'accuracy', 0.0),
                        'execution_time': execution_time,
                        'energy': getattr(result, 'energy', 0.0),
                        'convergence_time': getattr(result, 'convergence_time', execution_time),
                        'quantum_advantage': getattr(result, 'quantum_advantage', 0.0),
                        'trial_index': trial
                    }
                    
                    dataset_trials.append(trial_data)
                    all_accuracies.append(trial_data['accuracy'])
                    all_times.append(trial_data['execution_time'])
                    all_energies.append(trial_data['energy'])
                    
                except Exception as e:
                    dataset_trials.append({
                        'accuracy': 0.0,
                        'execution_time': float('inf'),
                        'energy': float('inf'),
                        'error': str(e),
                        'trial_index': trial
                    })
            
            # Calculate dataset-level statistics
            valid_trials = [t for t in dataset_trials if 'error' not in t]
            if valid_trials:
                dataset_stats = {
                    'mean_accuracy': np.mean([t['accuracy'] for t in valid_trials]),
                    'std_accuracy': np.std([t['accuracy'] for t in valid_trials]),
                    'mean_time': np.mean([t['execution_time'] for t in valid_trials]),
                    'std_time': np.std([t['execution_time'] for t in valid_trials]),
                    'success_rate': len(valid_trials) / len(dataset_trials),
                    'trials': dataset_trials
                }
            else:
                dataset_stats = {
                    'mean_accuracy': 0.0,
                    'std_accuracy': 0.0,
                    'mean_time': float('inf'),
                    'std_time': 0.0,
                    'success_rate': 0.0,
                    'trials': dataset_trials
                }
            
            results['dataset_results'][dataset_name] = dataset_stats
        
        # Calculate aggregate metrics across all datasets
        if all_accuracies:
            results['aggregate_metrics'] = {
                'overall_mean_accuracy': np.mean(all_accuracies),
                'overall_std_accuracy': np.std(all_accuracies),
                'overall_mean_time': np.mean([t for t in all_times if np.isfinite(t)]),
                'overall_std_time': np.std([t for t in all_times if np.isfinite(t)]),
                'overall_success_rate': len([a for a in all_accuracies if a > 0]) / len(all_accuracies)
            }
        
        return results
    
    def _statistical_comparison(self, novel_results: Dict, baseline_results: Dict, 
                              baseline_name: str) -> Dict[str, Any]:
        """Perform statistical comparison between algorithms"""
        
        tests = {}
        
        # Compare accuracies across datasets
        novel_accuracies = []
        baseline_accuracies = []
        
        for dataset_name in novel_results['dataset_results']:
            if dataset_name in baseline_results['dataset_results']:
                novel_acc = novel_results['dataset_results'][dataset_name]['mean_accuracy']
                baseline_acc = baseline_results['dataset_results'][dataset_name]['mean_accuracy']
                
                novel_accuracies.append(novel_acc)
                baseline_accuracies.append(baseline_acc)
        
        if len(novel_accuracies) >= 2:
            # Paired t-test for accuracy comparison
            accuracy_test = self._paired_t_test(novel_accuracies, baseline_accuracies)
            tests['accuracy_comparison'] = accuracy_test
            
            # Effect size calculation
            pooled_std = np.sqrt(
                (np.var(novel_accuracies, ddof=1) + np.var(baseline_accuracies, ddof=1)) / 2
            )
            if pooled_std > 0:
                cohens_d = (np.mean(novel_accuracies) - np.mean(baseline_accuracies)) / pooled_std
            else:
                cohens_d = 0.0
            
            tests['effect_size'] = {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_effect_size(cohens_d)
            }
        
        # Compare execution times
        novel_times = []
        baseline_times = []
        
        for dataset_name in novel_results['dataset_results']:
            if dataset_name in baseline_results['dataset_results']:
                novel_time = novel_results['dataset_results'][dataset_name]['mean_time']
                baseline_time = baseline_results['dataset_results'][dataset_name]['mean_time']
                
                if np.isfinite(novel_time) and np.isfinite(baseline_time):
                    novel_times.append(novel_time)
                    baseline_times.append(baseline_time)
        
        if len(novel_times) >= 2:
            time_test = self._paired_t_test(novel_times, baseline_times)
            tests['execution_time_comparison'] = time_test
        
        return tests
    
    def _paired_t_test(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Perform paired t-test"""
        if len(group1) != len(group2) or len(group1) < 2:
            return {'valid': False, 'reason': 'Insufficient or unmatched data'}
        
        differences = [a - b for a, b in zip(group1, group2)]
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        n = len(differences)
        
        if std_diff == 0:
            p_value = 0.0 if mean_diff != 0 else 1.0
            t_statistic = float('inf') if mean_diff != 0 else 0.0
        else:
            t_statistic = mean_diff / (std_diff / np.sqrt(n))
            # Simplified p-value calculation
            df = n - 1
            p_value = 2 * (1 - abs(t_statistic) / (abs(t_statistic) + df))
        
        return {
            'valid': True,
            'test_name': 'Paired t-test',
            't_statistic': t_statistic,
            'p_value': p_value,
            'degrees_of_freedom': n - 1,
            'mean_difference': mean_diff,
            'is_significant': p_value < 0.05,
            'interpretation': self._interpret_t_test(p_value, mean_diff)
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        direction = "positive" if cohens_d > 0 else "negative"
        return f"{magnitude} {direction} effect (d = {cohens_d:.3f})"
    
    def _interpret_t_test(self, p_value: float, mean_difference: float) -> str:
        """Interpret t-test results"""
        significance = "significant" if p_value < 0.05 else "not significant"
        direction = "improvement" if mean_difference > 0 else "degradation" if mean_difference < 0 else "no change"
        
        return f"Difference is {significance} (p = {p_value:.4f}), showing {direction}"
    
    def _calculate_improvement_metrics(self, novel_results: Dict, baseline_results: Dict) -> Dict[str, Any]:
        """Calculate various improvement metrics"""
        
        improvements = {
            'relative_accuracy_improvement': {},
            'relative_speed_improvement': {},
            'consistency_improvement': {},
            'overall_improvement_score': 0.0
        }
        
        for dataset_name in novel_results['dataset_results']:
            if dataset_name in baseline_results['dataset_results']:
                novel_stats = novel_results['dataset_results'][dataset_name]
                baseline_stats = baseline_results['dataset_results'][dataset_name]
                
                # Relative accuracy improvement
                if baseline_stats['mean_accuracy'] > 0:
                    acc_improvement = (
                        (novel_stats['mean_accuracy'] - baseline_stats['mean_accuracy']) / 
                        baseline_stats['mean_accuracy']
                    )
                else:
                    acc_improvement = 0.0
                
                improvements['relative_accuracy_improvement'][dataset_name] = acc_improvement
                
                # Relative speed improvement (lower time is better)
                if baseline_stats['mean_time'] > 0 and np.isfinite(baseline_stats['mean_time']):
                    speed_improvement = (
                        (baseline_stats['mean_time'] - novel_stats['mean_time']) / 
                        baseline_stats['mean_time']
                    )
                else:
                    speed_improvement = 0.0
                
                improvements['relative_speed_improvement'][dataset_name] = speed_improvement
                
                # Consistency improvement (lower std is better)
                if baseline_stats['std_accuracy'] > 0:
                    consistency_improvement = (
                        (baseline_stats['std_accuracy'] - novel_stats['std_accuracy']) / 
                        baseline_stats['std_accuracy']
                    )
                else:
                    consistency_improvement = 0.0
                
                improvements['consistency_improvement'][dataset_name] = consistency_improvement
        
        # Calculate overall improvement score
        acc_improvements = list(improvements['relative_accuracy_improvement'].values())
        speed_improvements = list(improvements['relative_speed_improvement'].values())
        consistency_improvements = list(improvements['consistency_improvement'].values())
        
        if acc_improvements:
            overall_score = (
                0.5 * np.mean(acc_improvements) +
                0.3 * np.mean(speed_improvements) +
                0.2 * np.mean(consistency_improvements)
            )
            improvements['overall_improvement_score'] = overall_score
        
        return improvements
    
    def _summarize_overall_performance(self, novel_results: Dict, 
                                     baseline_comparisons: Dict) -> Dict[str, Any]:
        """Summarize overall performance across all comparisons"""
        
        summary = {
            'wins_against_baselines': 0,
            'total_comparisons': len(baseline_comparisons),
            'average_improvement_score': 0.0,
            'best_performing_datasets': [],
            'worst_performing_datasets': [],
            'overall_recommendation': ""
        }
        
        improvement_scores = []
        dataset_performance = {}
        
        for baseline_name, comparison in baseline_comparisons.items():
            improvement_score = comparison['improvement_metrics']['overall_improvement_score']
            improvement_scores.append(improvement_score)
            
            if improvement_score > 0:
                summary['wins_against_baselines'] += 1
            
            # Track per-dataset performance
            for dataset_name, acc_improvement in comparison['improvement_metrics']['relative_accuracy_improvement'].items():
                if dataset_name not in dataset_performance:
                    dataset_performance[dataset_name] = []
                dataset_performance[dataset_name].append(acc_improvement)
        
        # Calculate averages
        if improvement_scores:
            summary['average_improvement_score'] = np.mean(improvement_scores)
        
        # Identify best and worst performing datasets
        if dataset_performance:
            dataset_avg_performance = {
                dataset: np.mean(improvements) 
                for dataset, improvements in dataset_performance.items()
            }
            
            sorted_datasets = sorted(dataset_avg_performance.items(), key=lambda x: x[1], reverse=True)
            summary['best_performing_datasets'] = sorted_datasets[:3]  # Top 3
            summary['worst_performing_datasets'] = sorted_datasets[-3:]  # Bottom 3
        
        # Overall recommendation
        win_rate = summary['wins_against_baselines'] / max(summary['total_comparisons'], 1)
        avg_score = summary['average_improvement_score']
        
        if win_rate >= 0.8 and avg_score > 0.2:
            summary['overall_recommendation'] = "Strong candidate for adoption - consistently outperforms baselines"
        elif win_rate >= 0.6 and avg_score > 0.1:
            summary['overall_recommendation'] = "Promising approach - outperforms most baselines with meaningful improvements"
        elif win_rate >= 0.4:
            summary['overall_recommendation'] = "Mixed results - some advantages but not consistently better"
        else:
            summary['overall_recommendation'] = "Not recommended - fails to improve over existing baselines"
        
        return summary


class ExperimentalDesign:
    """
    Systematic experimental design for comparative studies
    following scientific methodology principles.
    """
    
    def __init__(self):
        self.designs = {}
        self.executed_studies = {}
    
    def create_factorial_design(self, 
                               study_name: str,
                               factors: Dict[str, List[Any]],
                               dependent_variables: List[str],
                               hypothesis: str) -> StudyDesign:
        """
        Create a factorial experimental design.
        
        Args:
            study_name: Name of the study
            factors: Dictionary of factor names to their levels
            dependent_variables: List of variables to measure
            hypothesis: Research hypothesis
            
        Returns:
            Configured StudyDesign object
        """
        
        # Generate all combinations of factor levels (full factorial)
        factor_names = list(factors.keys())
        factor_levels = list(factors.values())
        
        import itertools
        all_combinations = list(itertools.product(*factor_levels))
        
        experimental_conditions = []
        for i, combination in enumerate(all_combinations):
            condition = {
                'condition_id': i,
                'condition_name': f"condition_{i}"
            }
            for factor_name, level in zip(factor_names, combination):
                condition[factor_name] = level
            experimental_conditions.append(condition)
        
        # Control condition (first combination or specified baseline)
        control_condition = experimental_conditions[0] if experimental_conditions else {}
        
        # Calculate required sample size (simplified power analysis)
        effect_size = 0.5  # Medium effect size
        alpha = 0.05
        power = 0.8
        n_groups = len(experimental_conditions)
        
        # Simplified sample size calculation
        sample_size_per_group = max(10, int(20 + 5 * np.sqrt(n_groups)))
        total_sample_size = sample_size_per_group * n_groups
        
        design = StudyDesign(
            study_name=study_name,
            hypothesis=hypothesis,
            null_hypothesis="No significant difference between conditions",
            independent_variables=factor_names,
            dependent_variables=dependent_variables,
            control_conditions=control_condition,
            experimental_conditions=experimental_conditions,
            sample_size=total_sample_size,
            randomization_method="complete_randomization",
            significance_level=alpha,
            power=power
        )
        
        self.designs[study_name] = design
        return design
    
    def execute_study(self, 
                     study_design: StudyDesign,
                     algorithm: Callable,
                     test_data_generator: Callable) -> List[StudyResult]:
        """
        Execute a designed study.
        
        Args:
            study_design: The experimental design to execute
            algorithm: Algorithm to test
            test_data_generator: Function that generates test data
            
        Returns:
            List of StudyResult objects
        """
        
        results = []
        
        # Calculate trials per condition
        trials_per_condition = max(1, study_design.sample_size // len(study_design.experimental_conditions))
        
        for condition in study_design.experimental_conditions:
            condition_results = {
                'measurements': {var: [] for var in study_design.dependent_variables},
                'raw_results': []
            }
            
            for trial in range(trials_per_condition):
                # Generate test data for this trial
                test_data = test_data_generator(**condition)
                
                # Run algorithm
                start_time = time.time()
                algorithm_result = algorithm(test_data, **condition)
                execution_time = time.time() - start_time
                
                # Extract measurements
                measurements = {
                    'accuracy': getattr(algorithm_result, 'accuracy', 0.0),
                    'execution_time': execution_time,
                    'energy': getattr(algorithm_result, 'energy', 0.0),
                    'convergence_time': getattr(algorithm_result, 'convergence_time', execution_time),
                    'quantum_advantage': getattr(algorithm_result, 'quantum_advantage', 0.0)
                }
                
                # Store measurements for dependent variables
                for var in study_design.dependent_variables:
                    if var in measurements:
                        condition_results['measurements'][var].append(measurements[var])
                
                condition_results['raw_results'].append({
                    'trial': trial,
                    'measurements': measurements,
                    'condition': condition.copy()
                })
            
            # Calculate summary statistics
            summary_stats = {}
            for var in study_design.dependent_variables:
                values = condition_results['measurements'][var]
                if values:
                    summary_stats[var] = {
                        'mean': np.mean(values),
                        'std': np.std(values, ddof=1),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'n': len(values)
                    }
            
            # Calculate effect sizes (compared to control)
            effect_sizes = {}
            if condition != study_design.control_conditions:
                # This would be calculated relative to control group in practice
                for var in study_design.dependent_variables:
                    effect_sizes[var] = 0.0  # Placeholder
            
            # Calculate confidence intervals
            confidence_intervals = {}
            for var in study_design.dependent_variables:
                values = condition_results['measurements'][var]
                if len(values) > 1:
                    mean = np.mean(values)
                    std = np.std(values, ddof=1)
                    margin = 1.96 * std / np.sqrt(len(values))  # 95% CI
                    confidence_intervals[var] = (mean - margin, mean + margin)
            
            # Create study result
            study_result = StudyResult(
                study_name=study_design.study_name,
                algorithm_name="test_algorithm",  # Would be parameterized
                condition_name=condition['condition_name'],
                measurements=condition_results['measurements'],
                summary_statistics=summary_stats,
                effect_sizes=effect_sizes,
                confidence_intervals=confidence_intervals,
                metadata={
                    'condition_parameters': condition,
                    'n_trials': trials_per_condition,
                    'study_design': asdict(study_design),
                    'raw_results': condition_results['raw_results']
                }
            )
            
            results.append(study_result)
        
        self.executed_studies[study_design.study_name] = results
        return results
    
    def analyze_factorial_effects(self, study_results: List[StudyResult], 
                                 dependent_variable: str = 'accuracy') -> Dict[str, Any]:
        """
        Analyze main effects and interactions in factorial design.
        
        Args:
            study_results: Results from executed factorial study
            dependent_variable: Which dependent variable to analyze
            
        Returns:
            Analysis of main effects and interactions
        """
        
        if not study_results:
            return {'error': 'No study results provided'}
        
        # Extract factor information from first result
        study_design = study_results[0].metadata['study_design']
        factors = study_design['independent_variables']
        
        # Collect data for analysis
        data_points = []
        for result in study_results:
            condition = result.metadata['condition_parameters']
            if dependent_variable in result.summary_statistics:
                mean_value = result.summary_statistics[dependent_variable]['mean']
                
                data_point = {'value': mean_value}
                for factor in factors:
                    data_point[factor] = condition.get(factor, None)
                data_points.append(data_point)
        
        if len(data_points) < 2:
            return {'error': 'Insufficient data for analysis'}
        
        # Calculate main effects
        main_effects = {}
        for factor in factors:
            factor_levels = list(set(dp[factor] for dp in data_points if dp[factor] is not None))
            if len(factor_levels) > 1:
                level_means = {}
                for level in factor_levels:
                    level_values = [dp['value'] for dp in data_points if dp[factor] == level]
                    level_means[level] = np.mean(level_values) if level_values else 0
                
                main_effects[factor] = {
                    'level_means': level_means,
                    'effect_size': max(level_means.values()) - min(level_means.values()),
                    'strongest_level': max(level_means, key=level_means.get)
                }
        
        # Simple interaction analysis (for 2-factor case)
        interactions = {}
        if len(factors) == 2:
            factor1, factor2 = factors[0], factors[1]
            
            # Create interaction matrix
            f1_levels = list(set(dp[factor1] for dp in data_points))
            f2_levels = list(set(dp[factor2] for dp in data_points))
            
            interaction_matrix = {}
            for f1_level in f1_levels:
                interaction_matrix[f1_level] = {}
                for f2_level in f2_levels:
                    cell_values = [
                        dp['value'] for dp in data_points 
                        if dp[factor1] == f1_level and dp[factor2] == f2_level
                    ]
                    interaction_matrix[f1_level][f2_level] = np.mean(cell_values) if cell_values else 0
            
            # Calculate interaction effect (simplified)
            interaction_effect = 0
            for f1_level in f1_levels:
                for f2_level in f2_levels:
                    # This is a simplified interaction calculation
                    pass
            
            interactions[f"{factor1}_x_{factor2}"] = {
                'interaction_matrix': interaction_matrix,
                'interaction_effect': interaction_effect
            }
        
        analysis = {
            'dependent_variable': dependent_variable,
            'main_effects': main_effects,
            'interactions': interactions,
            'recommendations': self._generate_factorial_recommendations(main_effects, interactions)
        }
        
        return analysis
    
    def _generate_factorial_recommendations(self, main_effects: Dict, interactions: Dict) -> List[str]:
        """Generate recommendations based on factorial analysis"""
        recommendations = []
        
        # Main effects recommendations
        for factor, effect_info in main_effects.items():
            strongest_level = effect_info['strongest_level']
            effect_size = effect_info['effect_size']
            
            if effect_size > 0.1:  # Meaningful effect
                recommendations.append(
                    f"Use {factor} = {strongest_level} for best performance (effect size: {effect_size:.3f})"
                )
            else:
                recommendations.append(f"Factor {factor} has minimal impact on performance")
        
        # Interaction recommendations
        for interaction_name, interaction_info in interactions.items():
            recommendations.append(f"Consider {interaction_name} interaction effects in optimization")
        
        if not recommendations:
            recommendations.append("No clear factor preferences identified - further investigation needed")
        
        return recommendations


class ResultsAnalyzer:
    """Advanced analysis tools for comparative study results"""
    
    @staticmethod
    def meta_analysis(study_results_list: List[List[StudyResult]]) -> Dict[str, Any]:
        """
        Perform meta-analysis across multiple studies.
        
        Args:
            study_results_list: List of study result lists from different studies
            
        Returns:
            Meta-analysis results
        """
        
        if not study_results_list:
            return {'error': 'No studies provided'}
        
        # Combine effect sizes across studies
        all_effect_sizes = []
        study_weights = []
        
        for study_results in study_results_list:
            if not study_results:
                continue
            
            # Calculate study-level effect size
            accuracies = []
            for result in study_results:
                if 'accuracy' in result.summary_statistics:
                    accuracies.append(result.summary_statistics['accuracy']['mean'])
            
            if len(accuracies) > 1:
                study_effect_size = (max(accuracies) - min(accuracies)) / np.std(accuracies)
                all_effect_sizes.append(study_effect_size)
                
                # Weight by sample size
                total_n = sum(result.summary_statistics.get('accuracy', {}).get('n', 0) for result in study_results)
                study_weights.append(total_n)
        
        if not all_effect_sizes:
            return {'error': 'Could not calculate effect sizes'}
        
        # Weighted meta-analysis
        if study_weights and sum(study_weights) > 0:
            meta_effect_size = np.average(all_effect_sizes, weights=study_weights)
        else:
            meta_effect_size = np.mean(all_effect_sizes)
        
        # Heterogeneity analysis
        heterogeneity = np.var(all_effect_sizes) if len(all_effect_sizes) > 1 else 0
        
        meta_analysis = {
            'combined_effect_size': meta_effect_size,
            'heterogeneity': heterogeneity,
            'n_studies': len(study_results_list),
            'total_n': sum(study_weights),
            'individual_effect_sizes': all_effect_sizes,
            'conclusion': ResultsAnalyzer._interpret_meta_analysis(meta_effect_size, heterogeneity)
        }
        
        return meta_analysis
    
    @staticmethod
    def _interpret_meta_analysis(effect_size: float, heterogeneity: float) -> str:
        """Interpret meta-analysis results"""
        
        # Effect size interpretation
        if abs(effect_size) < 0.2:
            effect_magnitude = "negligible"
        elif abs(effect_size) < 0.5:
            effect_magnitude = "small"
        elif abs(effect_size) < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"
        
        # Heterogeneity interpretation
        if heterogeneity < 0.1:
            consistency = "highly consistent"
        elif heterogeneity < 0.5:
            consistency = "moderately consistent"
        else:
            consistency = "highly variable"
        
        direction = "positive" if effect_size > 0 else "negative"
        
        return f"Meta-analysis shows {effect_magnitude} {direction} effect with {consistency} results across studies"
    
    @staticmethod
    def power_analysis(observed_effect_size: float, sample_size: int, 
                      alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform post-hoc power analysis.
        
        Args:
            observed_effect_size: Effect size found in study
            sample_size: Total sample size used
            alpha: Significance level used
            
        Returns:
            Power analysis results
        """
        
        # Simplified power calculation
        # In practice, would use more sophisticated statistical methods
        
        # Calculate observed power (simplified approximation)
        if sample_size < 10:
            observed_power = 0.1
        else:
            power_factor = abs(observed_effect_size) * np.sqrt(sample_size) / 2.8  # Simplified
            observed_power = min(0.99, max(0.05, 1 / (1 + np.exp(-power_factor))))
        
        # Calculate required sample size for 80% power
        if abs(observed_effect_size) > 0.01:
            required_n = int(64 / (observed_effect_size ** 2))  # Simplified formula
        else:
            required_n = 10000  # Very large for tiny effects
        
        analysis = {
            'observed_power': observed_power,
            'required_sample_size_80_power': required_n,
            'sample_size_adequate': observed_power >= 0.8,
            'effect_detectable': abs(observed_effect_size) > 0.2,
            'recommendations': ResultsAnalyzer._power_recommendations(observed_power, sample_size, required_n)
        }
        
        return analysis
    
    @staticmethod
    def _power_recommendations(observed_power: float, current_n: int, required_n: int) -> List[str]:
        """Generate recommendations based on power analysis"""
        recommendations = []
        
        if observed_power < 0.8:
            recommendations.append(f"Study is underpowered ({observed_power:.2f}). Consider increasing sample size.")
            if required_n > current_n:
                recommendations.append(f"Recommended sample size: {required_n} (current: {current_n})")
        else:
            recommendations.append(f"Study has adequate power ({observed_power:.2f})")
        
        if required_n > current_n * 10:
            recommendations.append("Effect size may be too small to detect reliably with practical sample sizes")
        
        return recommendations


class SignificanceTester:
    """Comprehensive significance testing with multiple correction methods"""
    
    @staticmethod
    def multiple_comparisons_correction(p_values: List[float], 
                                      method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Apply multiple comparisons correction.
        
        Args:
            p_values: List of uncorrected p-values
            method: Correction method ('bonferroni', 'holm', 'fdr')
            
        Returns:
            Corrected p-values and significance decisions
        """
        
        if not p_values:
            return {'error': 'No p-values provided'}
        
        n_comparisons = len(p_values)
        corrected_p_values = []
        
        if method == 'bonferroni':
            corrected_p_values = [min(1.0, p * n_comparisons) for p in p_values]
            
        elif method == 'holm':
            # Holm-Bonferroni correction
            sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
            corrected_p_values = [0] * len(p_values)
            
            for rank, idx in enumerate(sorted_indices):
                correction_factor = n_comparisons - rank
                corrected_p_values[idx] = min(1.0, p_values[idx] * correction_factor)
                
        elif method == 'fdr':
            # Benjamini-Hochberg FDR correction (simplified)
            sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
            corrected_p_values = [0] * len(p_values)
            
            for rank, idx in enumerate(sorted_indices, 1):
                correction_factor = n_comparisons / rank
                corrected_p_values[idx] = min(1.0, p_values[idx] * correction_factor)
        
        else:
            corrected_p_values = p_values  # No correction
        
        # Determine significance at α = 0.05
        significant_results = [p < 0.05 for p in corrected_p_values]
        
        return {
            'method': method,
            'original_p_values': p_values,
            'corrected_p_values': corrected_p_values,
            'significant_results': significant_results,
            'n_significant': sum(significant_results),
            'family_wise_error_controlled': method in ['bonferroni', 'holm'],
            'false_discovery_rate_controlled': method == 'fdr'
        }
    
    @staticmethod
    def equivalence_test(group1_mean: float, group2_mean: float,
                        group1_std: float, group2_std: float,
                        group1_n: int, group2_n: int,
                        equivalence_margin: float = 0.1) -> Dict[str, Any]:
        """
        Test for statistical equivalence between two groups.
        
        Args:
            group1_mean, group1_std, group1_n: Statistics for group 1
            group2_mean, group2_std, group2_n: Statistics for group 2
            equivalence_margin: Margin for practical equivalence
            
        Returns:
            Equivalence test results
        """
        
        # Calculate difference and its standard error
        mean_difference = group1_mean - group2_mean
        
        # Pooled standard error
        pooled_se = np.sqrt(
            (group1_std**2 / group1_n) + (group2_std**2 / group2_n)
        ) if group1_n > 0 and group2_n > 0 else 1.0
        
        # Two one-sided tests (TOST)
        if pooled_se > 0:
            t1 = (mean_difference - equivalence_margin) / pooled_se  # Lower bound test
            t2 = (mean_difference + equivalence_margin) / pooled_se  # Upper bound test
            
            # Simplified p-value calculation (would use proper t-distribution)
            df = group1_n + group2_n - 2
            p1 = 1 - abs(t1) / (abs(t1) + df)  # Approximate
            p2 = 1 - abs(t2) / (abs(t2) + df)  # Approximate
            
            # Equivalence p-value is the larger of the two
            equivalence_p_value = max(p1, p2)
        else:
            equivalence_p_value = 1.0
        
        # Confidence interval for difference
        margin_of_error = 1.96 * pooled_se  # 95% CI
        confidence_interval = (
            mean_difference - margin_of_error,
            mean_difference + margin_of_error
        )
        
        # Determine equivalence
        is_equivalent = (
            abs(mean_difference) < equivalence_margin and
            equivalence_p_value < 0.05
        )
        
        return {
            'mean_difference': mean_difference,
            'equivalence_margin': equivalence_margin,
            'equivalence_p_value': equivalence_p_value,
            'confidence_interval': confidence_interval,
            'is_equivalent': is_equivalent,
            'practical_equivalence': abs(mean_difference) < equivalence_margin,
            'statistical_equivalence': equivalence_p_value < 0.05,
            'interpretation': SignificanceTester._interpret_equivalence(
                is_equivalent, mean_difference, equivalence_margin
            )
        }
    
    @staticmethod
    def _interpret_equivalence(is_equivalent: bool, mean_difference: float, 
                             margin: float) -> str:
        """Interpret equivalence test results"""
        
        if is_equivalent:
            return f"Groups are statistically and practically equivalent (difference: {mean_difference:.3f}, margin: ±{margin:.3f})"
        elif abs(mean_difference) < margin:
            return f"Groups are practically equivalent but statistical equivalence not established (difference: {mean_difference:.3f})"
        else:
            return f"Groups are not equivalent (difference: {mean_difference:.3f} exceeds margin: ±{margin:.3f})"