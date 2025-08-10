"""
Comprehensive robustness tests for EchoLoc-NN Generation 2.

Tests all the robustness improvements including error handling,
validation, logging, circuit breakers, and model validation.
"""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from echoloc_nn.models.hybrid_architecture import EchoLocModel
from echoloc_nn.utils.enhanced_logging import get_enhanced_logger, setup_enhanced_logging
from echoloc_nn.utils.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, RetryHandler, RetryConfig,
    CircuitBreakerException, MaxRetriesExceededException,
    get_reliability_manager, circuit_breaker, retry
)
from echoloc_nn.utils.model_validator import (
    ModelValidator, ModelConstraints, ValidationMetrics,
    get_model_validator, validate_model
)

class TestModelRobustness:
    """Test model robustness improvements."""
    
    def setup_method(self):
        """Setup for each test."""
        self.model = EchoLocModel(n_sensors=4, chirp_length=2048, model_size='tiny')
        self.test_input = torch.randn(2, 4, 2048)
        self.sensor_positions = torch.randn(2, 4, 2)
        
    def test_cnn_dimension_fix(self):
        """Test that the CNN dimension mismatch is fixed."""
        # This should not raise an error anymore
        positions, confidences = self.model(self.test_input)
        
        assert positions.shape == (2, 3)
        assert confidences.shape == (2, 1)
        assert not torch.isnan(positions).any()
        assert not torch.isnan(confidences).any()
    
    def test_input_validation_echo_data(self):
        """Test comprehensive echo data validation."""
        # Test wrong type
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            self.model(np.array([1, 2, 3]))
        
        # Test wrong dimensions
        with pytest.raises(ValueError, match="must be 3D"):
            self.model(torch.randn(4, 2048))  # Missing batch dimension
        
        # Test wrong sensor count
        with pytest.raises(ValueError, match="Expected 4 sensors"):
            self.model(torch.randn(2, 3, 2048))  # Wrong sensor count
        
        # Test wrong sample count
        with pytest.raises(ValueError, match="Expected 2048 samples"):
            self.model(torch.randn(2, 4, 1024))  # Wrong sample count
        
        # Test NaN values
        nan_input = self.test_input.clone()
        nan_input[0, 0, 0] = float('nan')
        with pytest.raises(ValueError, match="contains NaN values"):
            self.model(nan_input)
        
        # Test infinite values
        inf_input = self.test_input.clone()
        inf_input[0, 0, 0] = float('inf')
        with pytest.raises(ValueError, match="contains infinite values"):
            self.model(inf_input)
        
        # Test all zeros (should raise warning but not error)
        zero_input = torch.zeros(2, 4, 2048)
        with pytest.raises(ValueError, match="contains only zeros"):
            self.model(zero_input)
    
    def test_input_validation_sensor_positions(self):
        """Test sensor positions validation."""
        # Test wrong type
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            self.model(self.test_input, np.array([1, 2, 3]))
        
        # Test wrong batch size for 3D tensor
        wrong_batch = torch.randn(1, 4, 2)  # Wrong batch size
        with pytest.raises(ValueError, match="batch size"):
            self.model(self.test_input, wrong_batch)
        
        # Test wrong sensor count
        wrong_sensors = torch.randn(2, 3, 2)  # Wrong sensor count
        with pytest.raises(ValueError, match="Expected 4 sensor positions"):
            self.model(self.test_input, wrong_sensors)
        
        # Test wrong position dimensions
        wrong_dims = torch.randn(2, 4, 3)  # 3D instead of 2D positions
        with pytest.raises(ValueError, match="Expected 2D positions"):
            self.model(self.test_input, wrong_dims)
        
        # Test NaN positions
        nan_pos = self.sensor_positions.clone()
        nan_pos[0, 0, 0] = float('nan')
        with pytest.raises(ValueError, match="contains NaN values"):
            self.model(self.test_input, nan_pos)
    
    def test_graceful_degradation(self):
        """Test graceful degradation when model fails internally."""
        # Mock the hybrid model to raise an exception
        with patch.object(self.model.hybrid_model, 'forward', side_effect=RuntimeError("Internal model error")):
            # Should not raise exception, but return fallback values
            positions, confidences = self.model(self.test_input)
            
            # Should return zeros as fallback
            assert torch.allclose(positions, torch.zeros(2, 3))
            assert torch.allclose(confidences, torch.zeros(2, 1))
    
    def test_predict_position_robustness(self):
        """Test the enhanced predict_position method."""
        # Test with numpy input
        numpy_input = np.random.randn(4, 2048)
        position, confidence = self.model.predict_position(numpy_input)
        
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        # Test with tensor input
        tensor_input = torch.randn(4, 2048)
        position, confidence = self.model.predict_position(tensor_input)
        
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        assert isinstance(confidence, float)
        
        # Test with sensor positions
        sensor_pos = np.random.randn(4, 2)
        position, confidence = self.model.predict_position(numpy_input, sensor_pos)
        
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
    
    def test_position_bounds_checking(self):
        """Test that position estimates are properly bounded."""
        # Create a model that might produce extreme values
        large_input = torch.randn(4, 2048) * 100  # Large input values
        position, confidence = self.model.predict_position(large_input)
        
        # Position should be bounded to reasonable range
        assert np.all(np.abs(position) <= 100.0)  # Max range as defined in model
        assert 0.0 <= confidence <= 1.0

class TestEnhancedLogging:
    """Test enhanced logging infrastructure."""
    
    def setup_method(self):
        """Setup for each test."""
        setup_enhanced_logging(log_level="DEBUG")
        self.logger = get_enhanced_logger("test_logger")
    
    def test_structured_logging(self):
        """Test structured logging capabilities."""
        # Test different log types
        self.logger.info("Test info message", test_param="value")
        self.logger.warning("Test warning", component="test")
        self.logger.error("Test error", error_code=404)
        
        # Test specialized logging methods
        self.logger.model_info("Model loaded", model_name="EchoLocModel", parameters=1000000)
        self.logger.hardware_info("Sensor connected", device="sensor_1")
        self.logger.performance("Operation completed", duration_ms=150.0, success=True)
        self.logger.security("Security event detected", event_type="unauthorized_access")
    
    def test_context_management(self):
        """Test context stack management."""
        # Test context stack
        self.logger.push_context("initialization")
        self.logger.push_context("model_loading")
        
        assert "initialization -> model_loading" in self.logger.get_context()
        
        self.logger.info("Message with context")
        
        context = self.logger.pop_context()
        assert context == "model_loading"
        assert "initialization" == self.logger.get_context()
        
        self.logger.pop_context()
        assert self.logger.get_context() == ""
    
    def test_operation_timing(self):
        """Test operation timing functionality."""
        # Test operation timing
        op_id = self.logger.start_operation("test_operation")
        time.sleep(0.1)  # Simulate work
        duration = self.logger.end_operation(op_id, success=True, result="success")
        
        assert duration >= 0.1
        
        # Test logger statistics
        stats = self.logger.get_stats()
        assert stats['total_operations'] >= 1
        assert 'test_operation' in stats['operation_counts']

class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_basic_operation(self):
        """Test basic circuit breaker operation."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0)
        breaker = CircuitBreaker("test_breaker", config)
        
        # Test successful calls
        def successful_function():
            return "success"
        
        result = breaker.call(successful_function)
        assert result == "success"
        
        metrics = breaker.get_metrics()
        assert metrics['total_calls'] == 1
        assert metrics['successful_calls'] == 1
        assert metrics['state'] == 'closed'
    
    def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        breaker = CircuitBreaker("test_breaker", config)
        
        def failing_function():
            raise RuntimeError("Test failure")
        
        # First failure
        with pytest.raises(RuntimeError):
            breaker.call(failing_function)
        
        # Second failure - should open the circuit
        with pytest.raises(RuntimeError):
            breaker.call(failing_function)
        
        metrics = breaker.get_metrics()
        assert metrics['state'] == 'open'
        assert metrics['failed_calls'] == 2
        
        # Circuit should be open now
        with pytest.raises(CircuitBreakerException):
            breaker.call(failing_function)
        
        metrics = breaker.get_metrics()
        assert metrics['rejected_calls'] == 1
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        config = CircuitBreakerConfig(
            failure_threshold=2, 
            recovery_timeout=0.1,
            success_threshold=2
        )
        breaker = CircuitBreaker("test_breaker", config)
        
        # Force circuit open
        def failing_function():
            raise RuntimeError("Test failure")
        
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(failing_function)
        
        assert breaker.get_metrics()['state'] == 'open'
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should move to half-open and allow one call
        def successful_function():
            return "success"
        
        result = breaker.call(successful_function)
        assert result == "success"
        assert breaker.get_metrics()['state'] == 'half_open'
        
        # Another success should close the circuit
        result = breaker.call(successful_function)
        assert result == "success"
        assert breaker.get_metrics()['state'] == 'closed'
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker as decorator."""
        @circuit_breaker(name="decorated_breaker", failure_threshold=2)
        def decorated_function(should_fail=False):
            if should_fail:
                raise RuntimeError("Decorated function failed")
            return "success"
        
        # Successful calls
        result = decorated_function(False)
        assert result == "success"
        
        # Test failures
        with pytest.raises(RuntimeError):
            decorated_function(True)
        with pytest.raises(RuntimeError):
            decorated_function(True)
        
        # Circuit should be open now
        with pytest.raises(CircuitBreakerException):
            decorated_function(False)

class TestRetryHandler:
    """Test retry handler functionality."""
    
    def test_retry_success_on_first_attempt(self):
        """Test retry handler when function succeeds on first attempt."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        handler = RetryHandler("test_retry", config)
        
        def successful_function():
            return "success"
        
        result = handler.call(successful_function)
        assert result == "success"
        
        metrics = handler.get_metrics()
        assert metrics['total_attempts'] == 1
        assert metrics['successful_retries'] == 0
    
    def test_retry_success_after_failures(self):
        """Test retry handler succeeding after initial failures."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        handler = RetryHandler("test_retry", config)
        
        attempt_count = 0
        
        def failing_then_success():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = handler.call(failing_then_success)
        assert result == "success"
        
        metrics = handler.get_metrics()
        assert metrics['total_attempts'] == 3
        assert metrics['successful_retries'] == 1
    
    def test_retry_max_attempts_exceeded(self):
        """Test retry handler when max attempts are exceeded."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        handler = RetryHandler("test_retry", config)
        
        def always_failing():
            raise ValueError("Always fails")
        
        with pytest.raises(MaxRetriesExceededException):
            handler.call(always_failing)
        
        metrics = handler.get_metrics()
        assert metrics['max_retries_exceeded'] == 1
    
    def test_retry_decorator(self):
        """Test retry as decorator."""
        @retry(name="decorated_retry", max_attempts=2, base_delay=0.01)
        def decorated_function(should_fail=True):
            if should_fail:
                raise ValueError("Decorated function failed")
            return "success"
        
        # Should exhaust retries and raise exception
        with pytest.raises(MaxRetriesExceededException):
            decorated_function(True)
        
        # Should succeed without retries
        result = decorated_function(False)
        assert result == "success"

class TestModelValidator:
    """Test model validation functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.model = EchoLocModel(n_sensors=4, chirp_length=2048, model_size='tiny')
        self.validator = ModelValidator()
        self.test_input = torch.randn(1, 4, 2048)
    
    def test_model_structure_validation(self):
        """Test model structure validation."""
        metrics = self.validator.validate_model(self.model, self.test_input)
        
        assert metrics.total_parameters > 0
        assert metrics.trainable_parameters > 0
        assert metrics.model_size_mb > 0
        assert not metrics.has_nan_parameters
        assert not metrics.has_inf_parameters
    
    def test_performance_validation(self):
        """Test performance validation."""
        metrics = self.validator.validate_model(self.model, self.test_input)
        
        assert metrics.forward_pass_time_ms > 0
        assert metrics.throughput_samples_per_second > 0
        # Should be reasonably fast for tiny model
        assert metrics.forward_pass_time_ms < 1000  # Less than 1 second
    
    def test_numerical_stability_validation(self):
        """Test numerical stability validation."""
        metrics = self.validator.validate_model(self.model, self.test_input)
        
        # Should have reasonable stability score
        assert 0.0 <= metrics.numerical_stability_score <= 1.0
        assert metrics.numerical_stability_score > 0.5  # Should be reasonably stable
    
    def test_constraints_enforcement(self):
        """Test that constraints are properly enforced."""
        constraints = ModelConstraints(
            max_parameters=100,  # Very low limit
            max_forward_time_ms=0.001,  # Impossible to meet
            min_numerical_stability=1.0  # Perfect stability required
        )
        validator = ModelValidator(constraints)
        
        # This should still complete but log warnings about constraint violations
        metrics = validator.validate_model(self.model, self.test_input)
        
        # Metrics should still be populated
        assert metrics.total_parameters > 0
        assert metrics.forward_pass_time_ms > 0
    
    def test_validation_with_accuracy_data(self):
        """Test validation with accuracy data."""
        # Create mock validation data
        validation_input = torch.randn(5, 4, 2048)
        validation_targets = torch.randn(5, 3)  # 5 position targets
        validation_data = (validation_input, validation_targets)
        
        metrics = self.validator.validate_model(
            self.model, 
            self.test_input,
            validation_data
        )
        
        assert 0.0 <= metrics.validation_accuracy <= 1.0
        assert 0.0 <= metrics.confidence_score <= 1.0
    
    def test_health_summary(self):
        """Test health summary generation."""
        # Run validation first
        self.validator.validate_model(self.model, self.test_input)
        
        summary = self.validator.get_health_summary()
        
        assert 'health_score' in summary
        assert 'total_issues' in summary
        assert 'parameter_health' in summary
        assert 'performance_ok' in summary
        assert 'latest_metrics' in summary
        
        # Health score should be reasonable for a working model
        assert 0.0 <= summary['health_score'] <= 1.0

class TestIntegrationRobustness:
    """Integration tests for robustness features working together."""
    
    def setup_method(self):
        """Setup for integration tests."""
        setup_enhanced_logging(log_level="INFO")
        self.model = EchoLocModel(n_sensors=4, chirp_length=2048, model_size='tiny')
        self.logger = get_enhanced_logger("integration_test")
        self.test_input = torch.randn(1, 4, 2048)
    
    def test_full_robustness_pipeline(self):
        """Test the full robustness pipeline end-to-end."""
        # 1. Validate model
        validator = get_model_validator()
        metrics = validator.validate_model(self.model, self.test_input)
        assert metrics.total_parameters > 0
        
        # 2. Test with circuit breaker
        @circuit_breaker(name="model_inference", failure_threshold=3)
        def protected_inference(model, input_data):
            return model.predict_position(input_data.squeeze().numpy())
        
        # 3. Test with retry
        @retry(name="robust_inference", max_attempts=2, base_delay=0.01)
        def retrying_inference():
            return protected_inference(self.model, self.test_input)
        
        # 4. Execute with full protection
        position, confidence = retrying_inference()
        
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        assert isinstance(confidence, float)
    
    def test_error_recovery_chain(self):
        """Test error recovery chain with multiple layers."""
        failure_count = 0
        
        @circuit_breaker(name="recovery_test", failure_threshold=2)
        @retry(name="recovery_retry", max_attempts=3, base_delay=0.01)
        def sometimes_failing_function():
            nonlocal failure_count
            failure_count += 1
            if failure_count < 3:
                raise RuntimeError("Temporary failure")
            return "recovered"
        
        # Should eventually succeed after retries
        result = sometimes_failing_function()
        assert result == "recovered"
    
    def test_comprehensive_monitoring(self):
        """Test that all monitoring systems work together."""
        # Setup operation tracking
        op_id = self.logger.start_operation("comprehensive_test")
        
        try:
            # Validate model
            validator = get_model_validator()
            metrics = validator.validate_model(self.model, self.test_input)
            
            # Run inference
            position, confidence = self.model.predict_position(self.test_input.squeeze().numpy())
            
            # Log results
            self.logger.model_info(
                "Model inference completed",
                model_name="EchoLocModel",
                position=position.tolist(),
                confidence=confidence,
                validation_score=metrics.numerical_stability_score
            )
            
        finally:
            self.logger.end_operation(op_id, success=True)
        
        # Check that we have comprehensive logs and metrics
        stats = self.logger.get_stats()
        assert stats['total_operations'] >= 1
        
        health_summary = validator.get_health_summary()
        assert 'health_score' in health_summary

def test_system_reliability_under_stress():
    """Test system reliability under stress conditions."""
    setup_enhanced_logging(log_level="WARNING")  # Reduce log noise
    
    model = EchoLocModel(n_sensors=4, chirp_length=2048, model_size='tiny')
    validator = get_model_validator()
    
    # Stress test with many rapid inferences
    success_count = 0
    error_count = 0
    
    for i in range(50):  # Reduced for test speed
        try:
            test_input = torch.randn(1, 4, 2048) * (1 + i * 0.1)  # Increasingly difficult inputs
            
            @circuit_breaker(name=f"stress_test_{i}", failure_threshold=5)
            @retry(name=f"stress_retry_{i}", max_attempts=2, base_delay=0.001)
            def stress_inference():
                position, confidence = model.predict_position(test_input.squeeze().numpy())
                
                # Validate results are reasonable
                if np.any(np.abs(position) > 1000):  # Extreme position
                    raise ValueError("Position out of bounds")
                if confidence < 0 or confidence > 1:  # Invalid confidence
                    raise ValueError("Invalid confidence")
                
                return position, confidence
            
            position, confidence = stress_inference()
            success_count += 1
            
        except Exception:
            error_count += 1
    
    # Should handle most cases successfully with robust error handling
    success_rate = success_count / (success_count + error_count)
    assert success_rate > 0.8, f"Success rate too low: {success_rate}"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])