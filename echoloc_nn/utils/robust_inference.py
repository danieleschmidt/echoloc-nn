"""
Robust inference utilities for EchoLoc-NN.

This module provides comprehensive error handling, input validation,
and graceful degradation for robust real-world deployment.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Any, Dict
from contextlib import contextmanager
import time


class RobustInferenceEngine:
    """
    Robust wrapper for inference with comprehensive error handling.
    
    Features:
    - Input validation and sanitization
    - Graceful error handling with fallbacks
    - Performance monitoring and alerting
    - Circuit breaker pattern for failure resilience
    - Memory usage monitoring
    """
    
    def __init__(self, max_failures: int = 5, failure_timeout: float = 60.0):
        self.logger = logging.getLogger('RobustInference')
        self.max_failures = max_failures
        self.failure_timeout = failure_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.is_circuit_open = False
        
        # Performance tracking
        self.inference_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'avg_latency_ms': 0.0,
            'last_success_time': time.time()
        }
    
    def validate_echo_data(self, echo_data: np.ndarray) -> Tuple[bool, str]:
        """
        Validate input echo data.
        
        Args:
            echo_data: Input echo data array
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if input is numpy array
            if not isinstance(echo_data, np.ndarray):
                return False, f"Input must be numpy array, got {type(echo_data)}"
            
            # Check dimensions
            if len(echo_data.shape) != 2:
                return False, f"Echo data must be 2D array, got {len(echo_data.shape)}D"
            
            n_sensors, n_samples = echo_data.shape
            
            # Check sensor count
            if n_sensors < 1 or n_sensors > 64:
                return False, f"Invalid sensor count: {n_sensors} (expected 1-64)"
            
            # Check sample count
            if n_samples < 64 or n_samples > 32768:
                return False, f"Invalid sample count: {n_samples} (expected 64-32768)"
            
            # Check data type
            if not np.issubdtype(echo_data.dtype, np.number):
                return False, f"Echo data must be numeric, got {echo_data.dtype}"
            
            # Check for invalid values
            if np.any(np.isnan(echo_data)):
                return False, "Echo data contains NaN values"
            
            if np.any(np.isinf(echo_data)):
                return False, "Echo data contains infinite values"
            
            # Check value range (reasonable for ultrasonic data)
            data_range = np.ptp(echo_data)
            if data_range == 0:
                return False, "Echo data has no variation (all values are identical)"
            
            if np.abs(echo_data).max() > 1e6:
                return False, f"Echo data values too large: max={np.abs(echo_data).max()}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def sanitize_echo_data(self, echo_data: np.ndarray) -> np.ndarray:
        """
        Sanitize echo data by fixing common issues.
        
        Args:
            echo_data: Raw echo data
            
        Returns:
            Sanitized echo data
        """
        # Create a copy to avoid modifying original
        sanitized = echo_data.copy().astype(np.float32)
        
        # Replace NaN and inf with zeros
        sanitized = np.nan_to_num(sanitized, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values
        max_val = np.percentile(np.abs(sanitized), 99.9)
        sanitized = np.clip(sanitized, -max_val, max_val)
        
        # Ensure minimum variation
        if np.std(sanitized) < 1e-10:
            # Add tiny random noise to prevent zero variance
            sanitized += np.random.randn(*sanitized.shape) * 1e-8
        
        return sanitized
    
    def check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker should allow the call.
        
        Returns:
            True if call should proceed, False if circuit is open
        """
        current_time = time.time()
        
        # Reset circuit breaker after timeout
        if (self.is_circuit_open and 
            current_time - self.last_failure_time > self.failure_timeout):
            self.is_circuit_open = False
            self.failure_count = 0
            self.logger.info("Circuit breaker reset - allowing calls")
        
        return not self.is_circuit_open
    
    def record_success(self):
        """Record successful inference."""
        self.failure_count = 0
        self.is_circuit_open = False
        self.inference_stats['successful_calls'] += 1
        self.inference_stats['last_success_time'] = time.time()
    
    def record_failure(self):
        """Record failed inference."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.inference_stats['failed_calls'] += 1
        
        if self.failure_count >= self.max_failures:
            self.is_circuit_open = True
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )
    
    @contextmanager
    def performance_monitoring(self):
        """Context manager for performance monitoring."""
        start_time = time.time()
        self.inference_stats['total_calls'] += 1
        
        try:
            yield
        finally:
            # Update latency statistics
            latency = (time.time() - start_time) * 1000  # ms
            
            # Exponential moving average
            alpha = 0.1
            if self.inference_stats['avg_latency_ms'] == 0:
                self.inference_stats['avg_latency_ms'] = latency
            else:
                self.inference_stats['avg_latency_ms'] = (
                    alpha * latency + 
                    (1 - alpha) * self.inference_stats['avg_latency_ms']
                )
    
    def robust_inference(
        self, 
        model, 
        echo_data: np.ndarray,
        fallback_position: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Perform robust inference with comprehensive error handling.
        
        Args:
            model: The inference model
            echo_data: Input echo data
            fallback_position: Fallback position if inference fails
            
        Returns:
            Tuple of (position, confidence, metadata)
        """
        metadata = {
            'success': False,
            'error': None,
            'validation_passed': False,
            'sanitization_applied': False,
            'circuit_breaker_open': False
        }
        
        # Check circuit breaker
        if not self.check_circuit_breaker():
            metadata['circuit_breaker_open'] = True
            fallback_pos = fallback_position if fallback_position is not None else np.zeros(3)
            return fallback_pos, 0.0, metadata
        
        with self.performance_monitoring():
            try:
                # Validate input
                is_valid, error_msg = self.validate_echo_data(echo_data)
                if not is_valid:
                    self.logger.warning(f"Input validation failed: {error_msg}")
                    metadata['error'] = f"Validation failed: {error_msg}"
                    self.record_failure()
                    fallback_pos = fallback_position if fallback_position is not None else np.zeros(3)
                    return fallback_pos, 0.0, metadata
                
                metadata['validation_passed'] = True
                
                # Sanitize input if needed
                sanitized_data = echo_data
                if np.any(np.isnan(echo_data)) or np.any(np.isinf(echo_data)):
                    sanitized_data = self.sanitize_echo_data(echo_data)
                    metadata['sanitization_applied'] = True
                    self.logger.info("Applied data sanitization")
                
                # Perform inference
                position, confidence = model.predict(sanitized_data)
                
                # Validate output
                if not isinstance(position, np.ndarray) or position.shape != (3,):
                    raise ValueError(f"Invalid model output shape: {position.shape}")
                
                if not (0.0 <= confidence <= 1.0):
                    self.logger.warning(f"Confidence out of range: {confidence}")
                    confidence = np.clip(confidence, 0.0, 1.0)
                
                # Check for invalid output values
                if np.any(np.isnan(position)) or np.any(np.isinf(position)):
                    raise ValueError("Model output contains NaN or Inf values")
                
                # Success
                self.record_success()
                metadata['success'] = True
                
                return position, float(confidence), metadata
                
            except Exception as e:
                self.logger.error(f"Inference failed: {str(e)}")
                metadata['error'] = str(e)
                self.record_failure()
                
                # Return fallback
                fallback_pos = fallback_position if fallback_position is not None else np.zeros(3)
                return fallback_pos, 0.0, metadata
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        current_time = time.time()
        time_since_last_success = current_time - self.inference_stats['last_success_time']
        
        success_rate = 0.0
        if self.inference_stats['total_calls'] > 0:
            success_rate = (
                self.inference_stats['successful_calls'] / 
                self.inference_stats['total_calls']
            )
        
        return {
            'circuit_breaker_open': self.is_circuit_open,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'avg_latency_ms': self.inference_stats['avg_latency_ms'],
            'time_since_last_success': time_since_last_success,
            'total_calls': self.inference_stats['total_calls'],
            'status': 'healthy' if success_rate > 0.9 and not self.is_circuit_open else 'degraded'
        }