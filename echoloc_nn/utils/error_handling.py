"""
Robust Error Handling and Recovery System

Provides comprehensive error handling, validation, and recovery mechanisms
for quantum-inspired task planning and ultrasonic localization systems.
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Union, Type
from dataclasses import dataclass
from enum import Enum
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    VALIDATION = "validation"
    COMPUTATION = "computation"
    HARDWARE = "hardware"
    NETWORK = "network"
    QUANTUM = "quantum"
    PLANNING = "planning"
    POSITIONING = "positioning"
    RESOURCE = "resource"

@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    timestamp: float
    context: Dict[str, Any]
    traceback_str: str
    recovery_suggestions: List[str]
    auto_recoverable: bool = False
    retry_count: int = 0
    max_retries: int = 3
    
class QuantumPlanningError(Exception):
    """Base exception for quantum planning errors."""
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.PLANNING,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()
        
class ValidationError(QuantumPlanningError):
    """Input validation errors."""
    def __init__(self, message: str, field_name: str = None, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, **kwargs)
        self.field_name = field_name
        
class QuantumCoherenceError(QuantumPlanningError):
    """Quantum coherence and decoherence errors."""
    def __init__(self, message: str, coherence_level: float = 0.0, **kwargs):
        super().__init__(message, ErrorCategory.QUANTUM, ErrorSeverity.HIGH, **kwargs)
        self.coherence_level = coherence_level
        
class ResourceAllocationError(QuantumPlanningError):
    """Resource allocation and availability errors."""
    def __init__(self, message: str, resource_id: str = None, **kwargs):
        super().__init__(message, ErrorCategory.RESOURCE, **kwargs)
        self.resource_id = resource_id
        
class PositioningError(QuantumPlanningError):
    """Ultrasonic positioning and localization errors."""
    def __init__(self, message: str, position: Optional[np.ndarray] = None, 
                 confidence: float = 0.0, **kwargs):
        super().__init__(message, ErrorCategory.POSITIONING, **kwargs)
        self.position = position
        self.confidence = confidence
        
class ErrorHandler:
    """
    Centralized error handling and recovery system.
    
    Provides:
    - Error classification and severity assessment
    - Automatic recovery strategies
    - Error logging and monitoring
    - Graceful degradation mechanisms
    """
    
    def __init__(self, enable_auto_recovery: bool = True):
        self.enable_auto_recovery = enable_auto_recovery
        self.error_history: List[ErrorInfo] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.VALIDATION: [self._recovery_validation],
            ErrorCategory.COMPUTATION: [self._recovery_computation],
            ErrorCategory.HARDWARE: [self._recovery_hardware],
            ErrorCategory.QUANTUM: [self._recovery_quantum],
            ErrorCategory.PLANNING: [self._recovery_planning],
            ErrorCategory.POSITIONING: [self._recovery_positioning],
            ErrorCategory.RESOURCE: [self._recovery_resource]
        }
        
        # Error rate tracking
        self.error_rates: Dict[ErrorCategory, List[float]] = {}
        self.last_error_check = time.time()
        
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Handle error with classification, logging, and recovery."""
        
        # Create error info
        error_info = self._create_error_info(error, context)
        self.error_history.append(error_info)
        
        # Log error
        self._log_error(error_info)
        
        # Update error rates
        self._update_error_rates(error_info)
        
        # Attempt recovery if enabled
        if self.enable_auto_recovery and error_info.auto_recoverable:
            return self._attempt_recovery(error_info)
            
        # Re-raise if not recoverable
        raise error
        
    def _create_error_info(self, error: Exception, context: Optional[Dict[str, Any]]) -> ErrorInfo:
        """Create comprehensive error information."""
        
        # Determine category and severity
        if isinstance(error, QuantumPlanningError):
            category = error.category
            severity = error.severity
            context = {**(context or {}), **error.context}
        else:
            category = self._classify_error(error)
            severity = self._assess_severity(error)
            
        # Generate suggestions
        suggestions = self._generate_recovery_suggestions(error, category)
        
        # Check if auto-recoverable
        auto_recoverable = self._is_auto_recoverable(error, category, severity)
        
        return ErrorInfo(
            error_id=f"{category.value}_{int(time.time())}_{id(error)}",
            category=category,
            severity=severity,
            message=str(error),
            timestamp=time.time(),
            context=context or {},
            traceback_str=traceback.format_exc(),
            recovery_suggestions=suggestions,
            auto_recoverable=auto_recoverable
        )
        
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error by type and content."""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        if 'validation' in error_message or 'invalid' in error_message:
            return ErrorCategory.VALIDATION
        elif 'numpy' in error_type or 'tensor' in error_type or 'math' in error_message:
            return ErrorCategory.COMPUTATION
        elif 'hardware' in error_message or 'device' in error_message:
            return ErrorCategory.HARDWARE
        elif 'network' in error_message or 'connection' in error_message:
            return ErrorCategory.NETWORK
        elif 'quantum' in error_message or 'coherence' in error_message:
            return ErrorCategory.QUANTUM
        elif 'position' in error_message or 'localization' in error_message:
            return ErrorCategory.POSITIONING
        elif 'resource' in error_message or 'allocation' in error_message:
            return ErrorCategory.RESOURCE
        else:
            return ErrorCategory.PLANNING
            
    def _assess_severity(self, error: Exception) -> ErrorSeverity:
        """Assess error severity based on type and impact."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if any(word in error_message for word in ['critical', 'fatal', 'system', 'crash']):
            return ErrorSeverity.CRITICAL
            
        # High severity errors
        if any(word in error_message for word in ['hardware', 'device', 'connection']):
            return ErrorSeverity.HIGH
            
        # Medium severity (default)
        if error_type in ['RuntimeError', 'ValueError', 'TypeError']:
            return ErrorSeverity.MEDIUM
            
        # Low severity
        return ErrorSeverity.LOW
        
    def _generate_recovery_suggestions(self, error: Exception, category: ErrorCategory) -> List[str]:
        """Generate recovery suggestions based on error type."""
        suggestions = []
        
        if category == ErrorCategory.VALIDATION:
            suggestions.extend([
                "Check input parameters and data types",
                "Verify value ranges and constraints",
                "Ensure required fields are provided"
            ])
        elif category == ErrorCategory.COMPUTATION:
            suggestions.extend([
                "Check for numerical stability issues",
                "Verify matrix dimensions and operations",
                "Consider using different numerical methods"
            ])
        elif category == ErrorCategory.HARDWARE:
            suggestions.extend([
                "Check hardware connections",
                "Verify device availability",
                "Restart hardware interfaces"
            ])
        elif category == ErrorCategory.QUANTUM:
            suggestions.extend([
                "Reduce decoherence rate",
                "Increase coherence time parameters",
                "Switch to classical optimization method"
            ])
        elif category == ErrorCategory.POSITIONING:
            suggestions.extend([
                "Check ultrasonic array calibration",
                "Verify sensor positions and orientations",
                "Increase position update frequency"
            ])
        elif category == ErrorCategory.RESOURCE:
            suggestions.extend([
                "Check resource availability",
                "Verify resource allocation constraints",
                "Consider alternative resource assignments"
            ])
            
        return suggestions
        
    def _is_auto_recoverable(self, error: Exception, category: ErrorCategory, 
                           severity: ErrorSeverity) -> bool:
        """Determine if error can be automatically recovered."""
        
        # Never auto-recover critical errors
        if severity == ErrorSeverity.CRITICAL:
            return False
            
        # Auto-recoverable categories for low/medium severity
        auto_recoverable_categories = {
            ErrorCategory.COMPUTATION,
            ErrorCategory.QUANTUM,
            ErrorCategory.PLANNING
        }
        
        return category in auto_recoverable_categories and severity != ErrorSeverity.HIGH
        
    def _attempt_recovery(self, error_info: ErrorInfo) -> Optional[Any]:
        """Attempt automatic error recovery."""
        
        strategies = self.recovery_strategies.get(error_info.category, [])
        
        for strategy in strategies:
            try:
                logger.info(f"Attempting recovery strategy for {error_info.error_id}")
                result = strategy(error_info)
                
                if result is not None:
                    logger.info(f"Recovery successful for {error_info.error_id}")
                    return result
                    
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy failed: {recovery_error}")
                continue
                
        logger.error(f"All recovery strategies failed for {error_info.error_id}")
        return None
        
    def _recovery_validation(self, error_info: ErrorInfo) -> Optional[Any]:
        """Recovery strategy for validation errors."""
        # Return default valid values or sanitized inputs
        context = error_info.context
        
        if 'default_value' in context:
            return context['default_value']
            
        return None
        
    def _recovery_computation(self, error_info: ErrorInfo) -> Optional[Any]:
        """Recovery strategy for computation errors."""
        # Try alternative numerical methods or approximations
        return None
        
    def _recovery_hardware(self, error_info: ErrorInfo) -> Optional[Any]:
        """Recovery strategy for hardware errors."""
        # Attempt hardware reset or fallback devices
        return None
        
    def _recovery_quantum(self, error_info: ErrorInfo) -> Optional[Any]:
        """Recovery strategy for quantum coherence errors."""
        # Switch to classical optimization or reset quantum state
        return None
        
    def _recovery_planning(self, error_info: ErrorInfo) -> Optional[Any]:
        """Recovery strategy for planning errors."""
        # Use fallback planning algorithm or simplified approach
        return None
        
    def _recovery_positioning(self, error_info: ErrorInfo) -> Optional[Any]:
        """Recovery strategy for positioning errors."""
        # Use last known position or estimated position
        context = error_info.context
        
        if 'last_known_position' in context:
            return context['last_known_position']
            
        return None
        
    def _recovery_resource(self, error_info: ErrorInfo) -> Optional[Any]:
        """Recovery strategy for resource allocation errors."""
        # Find alternative resources or adjust allocation
        return None
        
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level."""
        
        log_message = f"[{error_info.category.value.upper()}] {error_info.message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
            
        # Log context if available
        if error_info.context:
            logger.debug(f"Error context: {error_info.context}")
            
    def _update_error_rates(self, error_info: ErrorInfo):
        """Update error rate tracking."""
        current_time = time.time()
        
        if error_info.category not in self.error_rates:
            self.error_rates[error_info.category] = []
            
        self.error_rates[error_info.category].append(current_time)
        
        # Keep only last hour of errors
        hour_ago = current_time - 3600
        self.error_rates[error_info.category] = [
            t for t in self.error_rates[error_info.category] if t > hour_ago
        ]
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        current_time = time.time()
        
        # Calculate error rates (errors per hour)
        error_rates = {}
        for category, timestamps in self.error_rates.items():
            error_rates[category.value] = len(timestamps)
            
        # Calculate severity distribution
        severity_dist = {severity.value: 0 for severity in ErrorSeverity}
        for error in self.error_history[-100:]:  # Last 100 errors
            severity_dist[error.severity.value] += 1
            
        # Calculate recovery success rate
        recent_errors = [e for e in self.error_history[-50:] if e.auto_recoverable]
        recovery_attempts = len(recent_errors)
        # Note: Would need to track actual recovery success in real implementation
        
        return {
            'total_errors': len(self.error_history),
            'error_rates_per_hour': error_rates,
            'severity_distribution': severity_dist,
            'recovery_attempts': recovery_attempts,
            'most_common_categories': self._get_most_common_categories(),
            'error_trends': self._calculate_error_trends()
        }
        
    def _get_most_common_categories(self) -> List[str]:
        """Get most common error categories."""
        category_counts = {}
        
        for error in self.error_history[-100:]:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
        return sorted(category_counts.keys(), key=lambda k: category_counts[k], reverse=True)
        
    def _calculate_error_trends(self) -> Dict[str, str]:
        """Calculate error trend directions."""
        trends = {}
        current_time = time.time()
        
        for category, timestamps in self.error_rates.items():
            if len(timestamps) < 2:
                trends[category.value] = "stable"
                continue
                
            # Split into two halves and compare rates
            half_hour_ago = current_time - 1800
            recent_errors = len([t for t in timestamps if t > half_hour_ago])
            older_errors = len([t for t in timestamps if t <= half_hour_ago])
            
            if recent_errors > older_errors * 1.5:
                trends[category.value] = "increasing"
            elif recent_errors < older_errors * 0.5:
                trends[category.value] = "decreasing"
            else:
                trends[category.value] = "stable"
                
        return trends
        
# Decorator for automatic error handling
def handle_errors(error_handler: Optional[ErrorHandler] = None, 
                 default_return: Any = None,
                 reraise: bool = True):
    """Decorator for automatic error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # Limit context size
                    'kwargs': str(kwargs)[:200],
                    'default_return': default_return
                }
                
                try:
                    result = handler.handle_error(e, context)
                    return result if result is not None else default_return
                except Exception:
                    if reraise:
                        raise
                    return default_return
                    
        return wrapper
    return decorator
    
# Context manager for error handling
@contextmanager
def error_context(handler: ErrorHandler, operation_name: str, 
                 context: Optional[Dict[str, Any]] = None):
    """Context manager for scoped error handling."""
    
    enhanced_context = {
        'operation': operation_name,
        **(context or {})
    }
    
    try:
        yield
    except Exception as e:
        handler.handle_error(e, enhanced_context)
        
# Validation decorators
def validate_input(validation_func: Callable[[Any], bool], 
                  error_message: str = "Input validation failed"):
    """Decorator for input validation."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate first argument (typically 'self' is skipped)
            if args and not validation_func(args[0] if len(args) == 1 else args[1:]):
                raise ValidationError(error_message)
            return func(*args, **kwargs)
        return wrapper
    return decorator
    
def validate_quantum_state(coherence_threshold: float = 0.1):
    """Decorator for quantum state validation."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, 'quantum_coherence') and self.quantum_coherence < coherence_threshold:
                raise QuantumCoherenceError(
                    f"Quantum coherence too low: {self.quantum_coherence:.3f} < {coherence_threshold}",
                    coherence_level=self.quantum_coherence
                )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator
    
# Global error handler instance
_global_error_handler = ErrorHandler()

def get_global_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler
    
def set_global_error_handler(handler: ErrorHandler):
    """Set the global error handler instance."""
    global _global_error_handler
    _global_error_handler = handler