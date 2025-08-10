"""
Comprehensive Model Validation for EchoLoc-NN Generation 2.

Provides extensive validation, health checking, and performance monitoring
for neural network models in production environments.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import threading
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from .enhanced_logging import get_enhanced_logger
from .circuit_breaker import circuit_breaker, retry, RetryConfig

logger = get_enhanced_logger(__name__)

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for model health."""
    
    # Basic metrics
    total_parameters: int = 0
    trainable_parameters: int = 0
    model_size_mb: float = 0.0
    
    # Health indicators
    has_nan_parameters: bool = False
    has_inf_parameters: bool = False
    has_zero_gradients: bool = False
    gradient_norm: float = 0.0
    weight_norm: float = 0.0
    
    # Performance metrics
    forward_pass_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_samples_per_second: float = 0.0
    
    # Accuracy metrics
    validation_accuracy: float = 0.0
    test_accuracy: float = 0.0
    confidence_score: float = 0.0
    
    # Robustness metrics
    numerical_stability_score: float = 0.0
    adversarial_robustness_score: float = 0.0
    input_sensitivity: float = 0.0
    
    # Timestamp
    validation_timestamp: float = field(default_factory=time.time)

@dataclass
class ModelConstraints:
    """Constraints for model validation."""
    
    # Size constraints
    max_parameters: int = 100_000_000  # 100M parameters
    max_model_size_mb: float = 1000.0  # 1GB
    
    # Performance constraints
    max_forward_time_ms: float = 1000.0  # 1 second
    max_memory_usage_mb: float = 2000.0  # 2GB
    min_throughput: float = 1.0  # samples per second
    
    # Accuracy constraints
    min_validation_accuracy: float = 0.7
    min_confidence_score: float = 0.5
    
    # Stability constraints
    min_numerical_stability: float = 0.8
    max_gradient_norm: float = 10.0
    max_weight_norm: float = 100.0
    
    # Input constraints
    expected_input_shape: Tuple[int, ...] = (1, 4, 2048)  # (batch, sensors, samples)
    expected_output_shape: Tuple[int, ...] = (1, 3)      # (batch, coordinates)
    input_value_range: Tuple[float, float] = (-10.0, 10.0)

class ModelValidator:
    """
    Comprehensive model validator with extensive health checks.
    
    Performs deep validation of neural network models including architecture,
    parameters, performance, and robustness testing.
    """
    
    def __init__(self, constraints: Optional[ModelConstraints] = None):
        self.constraints = constraints or ModelConstraints()
        self.validation_history: List[ValidationMetrics] = []
        self._lock = threading.Lock()
        
        logger.info("Model validator initialized",
                   component='model_validator',
                   constraints=self.constraints.__dict__)
    
    @circuit_breaker(failure_threshold=3, recovery_timeout=30.0)
    @retry(max_attempts=2, base_delay=1.0)
    def validate_model(self, 
                      model: nn.Module,
                      test_input: Optional[torch.Tensor] = None,
                      validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> ValidationMetrics:
        """
        Perform comprehensive model validation.
        
        Args:
            model: PyTorch model to validate
            test_input: Optional test input tensor
            validation_data: Optional (input, target) tuple for accuracy testing
            
        Returns:
            ValidationMetrics with complete health assessment
        """
        logger.info("Starting comprehensive model validation",
                   component='model_validator',
                   model_type=type(model).__name__)
        
        start_time = time.time()
        metrics = ValidationMetrics()
        
        try:
            # Basic model structure validation
            self._validate_model_structure(model, metrics)
            
            # Parameter health checks
            self._validate_parameters(model, metrics)
            
            # Architecture validation
            self._validate_architecture(model, metrics, test_input)
            
            # Performance validation
            self._validate_performance(model, metrics, test_input)
            
            # Numerical stability tests
            self._validate_numerical_stability(model, metrics, test_input)
            
            # Accuracy validation if data provided
            if validation_data is not None:
                self._validate_accuracy(model, metrics, validation_data)
            
            # Robustness tests
            self._validate_robustness(model, metrics, test_input)
            
            # Final health assessment
            health_score = self._calculate_health_score(metrics)
            
            # Store validation results
            with self._lock:
                self.validation_history.append(metrics)
                if len(self.validation_history) > 100:  # Keep last 100 validations
                    self.validation_history = self.validation_history[-100:]
            
            validation_time = time.time() - start_time
            
            logger.info(f"Model validation completed in {validation_time:.2f}s",
                       component='model_validator',
                       validation_time_s=validation_time,
                       health_score=health_score,
                       total_issues=self._count_issues(metrics))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}",
                        component='model_validator',
                        error_type=type(e).__name__,
                        exc_info=True)
            raise
    
    def _validate_model_structure(self, model: nn.Module, metrics: ValidationMetrics):
        """Validate basic model structure and size."""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            metrics.total_parameters = total_params
            metrics.trainable_parameters = trainable_params
            
            # Estimate model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            metrics.model_size_mb = (param_size + buffer_size) / 1024 / 1024
            
            # Check constraints
            if total_params > self.constraints.max_parameters:
                logger.warning(f"Model has too many parameters: {total_params} > {self.constraints.max_parameters}",
                              component='model_validator',
                              parameter_count=total_params,
                              max_allowed=self.constraints.max_parameters)
            
            if metrics.model_size_mb > self.constraints.max_model_size_mb:
                logger.warning(f"Model size too large: {metrics.model_size_mb:.2f}MB > {self.constraints.max_model_size_mb}MB",
                              component='model_validator',
                              model_size_mb=metrics.model_size_mb,
                              max_allowed_mb=self.constraints.max_model_size_mb)
            
            logger.info(f"Model structure validation passed",
                       component='model_validator',
                       total_parameters=total_params,
                       trainable_parameters=trainable_params,
                       model_size_mb=metrics.model_size_mb)
                       
        except Exception as e:
            logger.error(f"Model structure validation failed: {e}",
                        component='model_validator',
                        exc_info=True)
            raise
    
    def _validate_parameters(self, model: nn.Module, metrics: ValidationMetrics):
        """Validate model parameters for health issues."""
        try:
            has_nan = False
            has_inf = False
            weight_norms = []
            gradient_norms = []
            
            for name, param in model.named_parameters():
                # Check for NaN values
                if torch.isnan(param).any():
                    has_nan = True
                    logger.error(f"NaN values found in parameter: {name}",
                                component='model_validator',
                                parameter_name=name)
                
                # Check for infinite values
                if torch.isinf(param).any():
                    has_inf = True
                    logger.error(f"Infinite values found in parameter: {name}",
                                component='model_validator',
                                parameter_name=name)
                
                # Calculate norms
                weight_norm = torch.norm(param).item()
                weight_norms.append(weight_norm)
                
                # Check gradients if available
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        logger.error(f"NaN gradients found in parameter: {name}",
                                    component='model_validator',
                                    parameter_name=name)
                    
                    grad_norm = torch.norm(param.grad).item()
                    gradient_norms.append(grad_norm)
                    
                    if grad_norm == 0.0:
                        logger.warning(f"Zero gradients found in parameter: {name}",
                                      component='model_validator',
                                      parameter_name=name)
                
                # Check for unusual weight magnitudes
                if weight_norm > self.constraints.max_weight_norm:
                    logger.warning(f"Large weight norm in parameter {name}: {weight_norm:.4f}",
                                  component='model_validator',
                                  parameter_name=name,
                                  weight_norm=weight_norm)
            
            metrics.has_nan_parameters = has_nan
            metrics.has_inf_parameters = has_inf
            metrics.weight_norm = np.mean(weight_norms) if weight_norms else 0.0
            metrics.gradient_norm = np.mean(gradient_norms) if gradient_norms else 0.0
            metrics.has_zero_gradients = any(norm == 0.0 for norm in gradient_norms)
            
            logger.info("Parameter validation completed",
                       component='model_validator',
                       has_nan_parameters=has_nan,
                       has_inf_parameters=has_inf,
                       avg_weight_norm=metrics.weight_norm,
                       avg_gradient_norm=metrics.gradient_norm)
                       
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}",
                        component='model_validator',
                        exc_info=True)
            raise
    
    def _validate_architecture(self, model: nn.Module, metrics: ValidationMetrics, test_input: Optional[torch.Tensor]):
        """Validate model architecture and forward pass."""
        try:
            if test_input is None:
                # Create default test input based on constraints
                test_input = torch.randn(*self.constraints.expected_input_shape)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            
            # Validate output shape
            if isinstance(output, (list, tuple)):
                output_shapes = [o.shape for o in output]
                logger.info(f"Model output shapes: {output_shapes}",
                           component='model_validator',
                           output_shapes=output_shapes)
            else:
                logger.info(f"Model output shape: {output.shape}",
                           component='model_validator',
                           output_shape=list(output.shape))
            
            # Check for valid outputs
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    logger.error("Model output contains NaN values",
                                component='model_validator')
                if torch.isinf(output).any():
                    logger.error("Model output contains infinite values",
                                component='model_validator')
            
            logger.info("Architecture validation passed",
                       component='model_validator',
                       forward_pass_successful=True)
                       
        except Exception as e:
            logger.error(f"Architecture validation failed: {e}",
                        component='model_validator',
                        exc_info=True)
            raise
    
    def _validate_performance(self, model: nn.Module, metrics: ValidationMetrics, test_input: Optional[torch.Tensor]):
        """Validate model performance characteristics."""
        try:
            if test_input is None:
                test_input = torch.randn(*self.constraints.expected_input_shape)
            
            model.eval()
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = model(test_input)
            
            # Memory usage before
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                memory_before = 0
            
            # Time multiple forward passes
            num_iterations = 10
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(test_input)
            
            total_time = time.time() - start_time
            avg_time_per_forward = (total_time / num_iterations) * 1000  # ms
            
            # Memory usage after
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                memory_usage = memory_after - memory_before
            else:
                memory_usage = 0
            
            # Calculate throughput
            batch_size = test_input.shape[0] if len(test_input.shape) > 0 else 1
            throughput = (batch_size * num_iterations) / total_time
            
            metrics.forward_pass_time_ms = avg_time_per_forward
            metrics.memory_usage_mb = memory_usage
            metrics.throughput_samples_per_second = throughput
            
            # Check performance constraints
            if avg_time_per_forward > self.constraints.max_forward_time_ms:
                logger.warning(f"Forward pass too slow: {avg_time_per_forward:.2f}ms > {self.constraints.max_forward_time_ms}ms",
                              component='model_validator',
                              forward_time_ms=avg_time_per_forward,
                              max_allowed_ms=self.constraints.max_forward_time_ms)
            
            if memory_usage > self.constraints.max_memory_usage_mb:
                logger.warning(f"Memory usage too high: {memory_usage:.2f}MB > {self.constraints.max_memory_usage_mb}MB",
                              component='model_validator',
                              memory_usage_mb=memory_usage,
                              max_allowed_mb=self.constraints.max_memory_usage_mb)
            
            if throughput < self.constraints.min_throughput:
                logger.warning(f"Throughput too low: {throughput:.2f} < {self.constraints.min_throughput} samples/sec",
                              component='model_validator',
                              throughput=throughput,
                              min_required=self.constraints.min_throughput)
            
            logger.info("Performance validation completed",
                       component='model_validator',
                       forward_time_ms=avg_time_per_forward,
                       memory_usage_mb=memory_usage,
                       throughput=throughput)
                       
        except Exception as e:
            logger.error(f"Performance validation failed: {e}",
                        component='model_validator',
                        exc_info=True)
            raise
    
    def _validate_numerical_stability(self, model: nn.Module, metrics: ValidationMetrics, test_input: Optional[torch.Tensor]):
        """Test numerical stability of the model."""
        try:
            if test_input is None:
                test_input = torch.randn(*self.constraints.expected_input_shape)
            
            model.eval()
            stability_scores = []
            
            # Test with different input magnitudes
            magnitudes = [0.1, 0.5, 1.0, 2.0, 5.0]
            baseline_output = None
            
            for magnitude in magnitudes:
                scaled_input = test_input * magnitude
                
                with torch.no_grad():
                    try:
                        output = model(scaled_input)
                        if isinstance(output, (list, tuple)):
                            output = output[0]  # Take first output for analysis
                        
                        # Check for valid output
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            stability_scores.append(0.0)
                            logger.warning(f"Numerical instability at magnitude {magnitude}",
                                          component='model_validator',
                                          input_magnitude=magnitude)
                        else:
                            if baseline_output is None:
                                baseline_output = output
                                stability_scores.append(1.0)
                            else:
                                # Calculate relative change
                                rel_change = torch.norm(output - baseline_output) / (torch.norm(baseline_output) + 1e-8)
                                stability_score = max(0.0, 1.0 - rel_change.item())
                                stability_scores.append(stability_score)
                        
                    except Exception:
                        stability_scores.append(0.0)
                        logger.warning(f"Model failed at input magnitude {magnitude}",
                                      component='model_validator',
                                      input_magnitude=magnitude)
            
            metrics.numerical_stability_score = np.mean(stability_scores)
            
            if metrics.numerical_stability_score < self.constraints.min_numerical_stability:
                logger.warning(f"Low numerical stability: {metrics.numerical_stability_score:.3f} < {self.constraints.min_numerical_stability}",
                              component='model_validator',
                              stability_score=metrics.numerical_stability_score,
                              min_required=self.constraints.min_numerical_stability)
            
            logger.info("Numerical stability validation completed",
                       component='model_validator',
                       stability_score=metrics.numerical_stability_score)
                       
        except Exception as e:
            logger.error(f"Numerical stability validation failed: {e}",
                        component='model_validator',
                        exc_info=True)
            raise
    
    def _validate_accuracy(self, model: nn.Module, metrics: ValidationMetrics, validation_data: Tuple[torch.Tensor, torch.Tensor]):
        """Validate model accuracy on provided data."""
        try:
            inputs, targets = validation_data
            model.eval()
            
            correct_predictions = 0
            total_predictions = 0
            confidence_scores = []
            
            with torch.no_grad():
                # Handle batch processing if needed
                if inputs.dim() == 3:  # Single sample
                    inputs = inputs.unsqueeze(0)
                    targets = targets.unsqueeze(0) if targets.dim() < 2 else targets
                
                outputs = model(inputs)
                if isinstance(outputs, (list, tuple)):
                    predictions, confidences = outputs[0], outputs[1]
                else:
                    predictions = outputs
                    confidences = torch.ones(predictions.shape[0])
                
                # Calculate accuracy (simplified for regression task)
                for pred, target, conf in zip(predictions, targets, confidences):
                    error = torch.norm(pred - target).item()
                    # Consider prediction correct if within reasonable range
                    if error < 1.0:  # 1 meter accuracy threshold
                        correct_predictions += 1
                    total_predictions += 1
                    confidence_scores.append(conf.item() if hasattr(conf, 'item') else float(conf))
            
            accuracy = correct_predictions / max(total_predictions, 1)
            avg_confidence = np.mean(confidence_scores)
            
            metrics.validation_accuracy = accuracy
            metrics.confidence_score = avg_confidence
            
            # Check constraints
            if accuracy < self.constraints.min_validation_accuracy:
                logger.warning(f"Low validation accuracy: {accuracy:.3f} < {self.constraints.min_validation_accuracy}",
                              component='model_validator',
                              accuracy=accuracy,
                              min_required=self.constraints.min_validation_accuracy)
            
            if avg_confidence < self.constraints.min_confidence_score:
                logger.warning(f"Low confidence score: {avg_confidence:.3f} < {self.constraints.min_confidence_score}",
                              component='model_validator',
                              confidence=avg_confidence,
                              min_required=self.constraints.min_confidence_score)
            
            logger.info("Accuracy validation completed",
                       component='model_validator',
                       validation_accuracy=accuracy,
                       avg_confidence=avg_confidence,
                       samples_tested=total_predictions)
                       
        except Exception as e:
            logger.error(f"Accuracy validation failed: {e}",
                        component='model_validator',
                        exc_info=True)
            raise
    
    def _validate_robustness(self, model: nn.Module, metrics: ValidationMetrics, test_input: Optional[torch.Tensor]):
        """Test model robustness to input variations."""
        try:
            if test_input is None:
                test_input = torch.randn(*self.constraints.expected_input_shape)
            
            model.eval()
            robustness_scores = []
            
            # Get baseline output
            with torch.no_grad():
                baseline_output = model(test_input)
                if isinstance(baseline_output, (list, tuple)):
                    baseline_output = baseline_output[0]
            
            # Test robustness to noise
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            
            for noise_level in noise_levels:
                noise = torch.randn_like(test_input) * noise_level
                noisy_input = test_input + noise
                
                with torch.no_grad():
                    try:
                        noisy_output = model(noisy_input)
                        if isinstance(noisy_output, (list, tuple)):
                            noisy_output = noisy_output[0]
                        
                        # Calculate sensitivity
                        output_change = torch.norm(noisy_output - baseline_output)
                        input_change = torch.norm(noise)
                        sensitivity = (output_change / (input_change + 1e-8)).item()
                        
                        # Lower sensitivity = higher robustness
                        robustness_score = max(0.0, 1.0 - sensitivity / 10.0)  # Normalize
                        robustness_scores.append(robustness_score)
                        
                    except Exception:
                        robustness_scores.append(0.0)
                        logger.warning(f"Model failed with noise level {noise_level}",
                                      component='model_validator',
                                      noise_level=noise_level)
            
            metrics.adversarial_robustness_score = np.mean(robustness_scores)
            
            # Calculate input sensitivity
            if robustness_scores:
                metrics.input_sensitivity = 1.0 - metrics.adversarial_robustness_score
            
            logger.info("Robustness validation completed",
                       component='model_validator',
                       robustness_score=metrics.adversarial_robustness_score,
                       input_sensitivity=metrics.input_sensitivity)
                       
        except Exception as e:
            logger.error(f"Robustness validation failed: {e}",
                        component='model_validator',
                        exc_info=True)
            raise
    
    def _calculate_health_score(self, metrics: ValidationMetrics) -> float:
        """Calculate overall model health score."""
        score = 1.0
        
        # Penalize parameter issues
        if metrics.has_nan_parameters:
            score -= 0.5
        if metrics.has_inf_parameters:
            score -= 0.5
        
        # Performance penalties
        if metrics.forward_pass_time_ms > self.constraints.max_forward_time_ms:
            score -= 0.2
        if metrics.memory_usage_mb > self.constraints.max_memory_usage_mb:
            score -= 0.2
        
        # Stability bonuses
        score = score * metrics.numerical_stability_score
        
        # Accuracy bonuses
        if metrics.validation_accuracy > 0:
            score = score * metrics.validation_accuracy
        
        return max(0.0, min(1.0, score))
    
    def _count_issues(self, metrics: ValidationMetrics) -> int:
        """Count validation issues."""
        issues = 0
        
        if metrics.has_nan_parameters:
            issues += 1
        if metrics.has_inf_parameters:
            issues += 1
        if metrics.forward_pass_time_ms > self.constraints.max_forward_time_ms:
            issues += 1
        if metrics.memory_usage_mb > self.constraints.max_memory_usage_mb:
            issues += 1
        if metrics.numerical_stability_score < self.constraints.min_numerical_stability:
            issues += 1
        if metrics.validation_accuracy < self.constraints.min_validation_accuracy:
            issues += 1
        
        return issues
    
    def get_validation_history(self) -> List[ValidationMetrics]:
        """Get validation history."""
        with self._lock:
            return self.validation_history.copy()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get model health summary."""
        with self._lock:
            if not self.validation_history:
                return {'no_data': True}
            
            latest = self.validation_history[-1]
            health_score = self._calculate_health_score(latest)
            issues = self._count_issues(latest)
            
            return {
                'health_score': health_score,
                'total_issues': issues,
                'last_validation': latest.validation_timestamp,
                'parameter_health': not (latest.has_nan_parameters or latest.has_inf_parameters),
                'performance_ok': (latest.forward_pass_time_ms <= self.constraints.max_forward_time_ms and
                                 latest.memory_usage_mb <= self.constraints.max_memory_usage_mb),
                'stability_ok': latest.numerical_stability_score >= self.constraints.min_numerical_stability,
                'accuracy_ok': latest.validation_accuracy >= self.constraints.min_validation_accuracy,
                'latest_metrics': {
                    'total_parameters': latest.total_parameters,
                    'model_size_mb': latest.model_size_mb,
                    'forward_pass_time_ms': latest.forward_pass_time_ms,
                    'numerical_stability_score': latest.numerical_stability_score,
                    'validation_accuracy': latest.validation_accuracy
                }
            }

# Global model validator
_global_model_validator = ModelValidator()

def get_model_validator() -> ModelValidator:
    """Get the global model validator instance."""
    return _global_model_validator

def validate_model(model: nn.Module, **kwargs) -> ValidationMetrics:
    """Convenience function to validate a model."""
    validator = get_model_validator()
    return validator.validate_model(model, **kwargs)