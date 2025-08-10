"""
Circuit Breaker and Retry Mechanisms for EchoLoc-NN Generation 2.

Provides robust reliability patterns including circuit breakers, retry logic,
exponential backoff, and graceful degradation for production environments.
"""

import time
import random
import threading
import functools
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
from .enhanced_logging import get_enhanced_logger

logger = get_enhanced_logger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open" # Testing if service has recovered

class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    RANDOMIZED_EXPONENTIAL = "randomized_exponential"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5           # Failures before opening
    recovery_timeout: float = 60.0       # Seconds before attempting recovery
    success_threshold: int = 3           # Successes needed to close from half-open
    timeout: Optional[float] = None      # Call timeout in seconds
    expected_exception: type = Exception # Exception type that triggers breaker
    
@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0             # Base delay in seconds
    max_delay: float = 60.0             # Maximum delay in seconds
    backoff_multiplier: float = 2.0     # Multiplier for exponential backoff
    jitter: bool = True                 # Add randomization
    exceptions: tuple = (Exception,)     # Exceptions that trigger retry

@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0        # Calls rejected when circuit is open
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    average_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)

class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class MaxRetriesExceededException(Exception):
    """Exception raised when maximum retry attempts are exceeded."""
    pass

class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Prevents cascading failures by temporarily stopping calls to a failing service
    and allowing it time to recover.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.last_attempt_time = 0.0
        self._lock = threading.RLock()
        
        logger.info(f"Circuit breaker '{name}' initialized", 
                   breaker_name=name, 
                   config=config.__dict__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function call through the circuit breaker."""
        with self._lock:
            self.metrics.total_calls += 1
            
            # Check if circuit should remain open
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._move_to_half_open()
                else:
                    self.metrics.rejected_calls += 1
                    logger.circuit_breaker(
                        f"Circuit breaker '{self.name}' rejected call - circuit is open",
                        breaker_name=self.name,
                        state=self.state.value
                    )
                    raise CircuitBreakerException(f"Circuit breaker '{self.name}' is open")
            
        # Execute the function call
        start_time = time.time()
        try:
            # Apply timeout if configured
            if self.config.timeout:
                result = self._call_with_timeout(func, self.config.timeout, *args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record successful call
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except self.config.expected_exception as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
    
    def _call_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Execute function with timeout (simplified implementation)."""
        # Note: In a production environment, you might want to use threading.Timer
        # or asyncio for proper timeout handling
        return func(*args, **kwargs)
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return time.time() - self.last_attempt_time >= self.config.recovery_timeout
    
    def _move_to_half_open(self):
        """Move circuit breaker to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.metrics.consecutive_successes = 0
        self.metrics.state_changes += 1
        
        logger.circuit_breaker(
            f"Circuit breaker '{self.name}' moved to half-open state",
            breaker_name=self.name,
            state=self.state.value
        )
    
    def _record_success(self, execution_time: float):
        """Record a successful call."""
        with self._lock:
            self.metrics.successful_calls += 1
            self.metrics.consecutive_failures = 0
            self.metrics.consecutive_successes += 1
            self.metrics.last_success_time = time.time()
            
            # Update response time metrics
            self.metrics.response_times.append(execution_time)
            if len(self.metrics.response_times) > 100:  # Keep last 100 response times
                self.metrics.response_times = self.metrics.response_times[-100:]
            
            self.metrics.average_response_time = sum(self.metrics.response_times) / len(self.metrics.response_times)
            
            # State transitions based on successes
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    self._move_to_closed()
            
            logger.performance(
                f"Circuit breaker '{self.name}' recorded success",
                breaker_name=self.name,
                execution_time_ms=execution_time * 1000,
                consecutive_successes=self.metrics.consecutive_successes
            )
    
    def _record_failure(self, exception: Exception, execution_time: float):
        """Record a failed call."""
        with self._lock:
            self.metrics.failed_calls += 1
            self.metrics.consecutive_successes = 0
            self.metrics.consecutive_failures += 1
            self.metrics.last_failure_time = time.time()
            self.last_attempt_time = time.time()
            
            # State transitions based on failures
            if self.state == CircuitBreakerState.CLOSED:
                if self.metrics.consecutive_failures >= self.config.failure_threshold:
                    self._move_to_open()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self._move_to_open()
            
            logger.error(
                f"Circuit breaker '{self.name}' recorded failure: {exception}",
                breaker_name=self.name,
                error_type=type(exception).__name__,
                consecutive_failures=self.metrics.consecutive_failures,
                execution_time_ms=execution_time * 1000,
                exc_info=True
            )
    
    def _move_to_open(self):
        """Move circuit breaker to open state."""
        self.state = CircuitBreakerState.OPEN
        self.metrics.state_changes += 1
        
        logger.circuit_breaker(
            f"Circuit breaker '{self.name}' opened due to failures",
            breaker_name=self.name,
            state=self.state.value,
            consecutive_failures=self.metrics.consecutive_failures
        )
    
    def _move_to_closed(self):
        """Move circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.metrics.consecutive_failures = 0
        self.metrics.state_changes += 1
        
        logger.circuit_breaker(
            f"Circuit breaker '{self.name}' closed after recovery",
            breaker_name=self.name,
            state=self.state.value,
            consecutive_successes=self.metrics.consecutive_successes
        )
    
    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.metrics.consecutive_failures = 0
            self.metrics.consecutive_successes = 0
            self.metrics.state_changes += 1
            
        logger.circuit_breaker(
            f"Circuit breaker '{self.name}' manually reset",
            breaker_name=self.name,
            state=self.state.value
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        with self._lock:
            success_rate = (self.metrics.successful_calls / max(self.metrics.total_calls, 1)) * 100
            failure_rate = (self.metrics.failed_calls / max(self.metrics.total_calls, 1)) * 100
            
            return {
                'name': self.name,
                'state': self.state.value,
                'total_calls': self.metrics.total_calls,
                'successful_calls': self.metrics.successful_calls,
                'failed_calls': self.metrics.failed_calls,
                'rejected_calls': self.metrics.rejected_calls,
                'success_rate': success_rate,
                'failure_rate': failure_rate,
                'consecutive_failures': self.metrics.consecutive_failures,
                'consecutive_successes': self.metrics.consecutive_successes,
                'state_changes': self.metrics.state_changes,
                'average_response_time_ms': self.metrics.average_response_time * 1000,
                'last_failure_time': self.metrics.last_failure_time,
                'last_success_time': self.metrics.last_success_time
            }

class RetryHandler:
    """
    Retry handler with multiple strategies and exponential backoff.
    
    Provides configurable retry logic with different backoff strategies
    and comprehensive error handling.
    """
    
    def __init__(self, name: str, config: RetryConfig):
        self.name = name
        self.config = config
        self.metrics = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'max_retries_exceeded': 0
        }
        self._lock = threading.Lock()
        
        logger.info(f"Retry handler '{name}' initialized",
                   handler_name=name,
                   config=config.__dict__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with retry logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function call with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            with self._lock:
                self.metrics['total_attempts'] += 1
            
            try:
                logger.debug(f"Retry handler '{self.name}' attempt {attempt}/{self.config.max_attempts}",
                           handler_name=self.name,
                           attempt=attempt,
                           max_attempts=self.config.max_attempts)
                
                result = func(*args, **kwargs)
                
                if attempt > 1:  # Was a retry
                    with self._lock:
                        self.metrics['successful_retries'] += 1
                    
                    logger.info(f"Retry handler '{self.name}' succeeded on attempt {attempt}",
                               handler_name=self.name,
                               attempt=attempt,
                               success_on_retry=True)
                
                return result
                
            except self.config.exceptions as e:
                last_exception = e
                
                logger.warning(f"Retry handler '{self.name}' attempt {attempt} failed: {e}",
                              handler_name=self.name,
                              attempt=attempt,
                              error_type=type(e).__name__,
                              exc_info=True)
                
                # If this was the last attempt, don't sleep
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.debug(f"Retry handler '{self.name}' waiting {delay:.2f}s before retry",
                                handler_name=self.name,
                                delay_seconds=delay,
                                next_attempt=attempt + 1)
                    time.sleep(delay)
            
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Retry handler '{self.name}' encountered non-retryable exception: {e}",
                            handler_name=self.name,
                            error_type=type(e).__name__,
                            exc_info=True)
                raise
        
        # All attempts exhausted
        with self._lock:
            self.metrics['failed_retries'] += 1
            self.metrics['max_retries_exceeded'] += 1
        
        logger.error(f"Retry handler '{self.name}' exceeded max attempts ({self.config.max_attempts})",
                    handler_name=self.name,
                    max_attempts=self.config.max_attempts,
                    final_exception=str(last_exception))
        
        raise MaxRetriesExceededException(
            f"Max retries ({self.config.max_attempts}) exceeded for '{self.name}'. "
            f"Last exception: {last_exception}"
        ) from last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
            
        elif self.config.strategy == RetryStrategy.RANDOMIZED_EXPONENTIAL:
            base_delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
            # Add randomization to prevent thundering herd
            jitter_range = base_delay * 0.1  # 10% jitter
            delay = base_delay + random.uniform(-jitter_range, jitter_range)
        
        else:
            delay = self.config.base_delay
        
        # Apply jitter if enabled
        if self.config.jitter and self.config.strategy != RetryStrategy.RANDOMIZED_EXPONENTIAL:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        # Ensure delay doesn't exceed maximum
        return min(delay, self.config.max_delay)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry handler metrics."""
        with self._lock:
            total_attempts = self.metrics['total_attempts']
            success_rate = (self.metrics['successful_retries'] / max(total_attempts, 1)) * 100
            
            return {
                'name': self.name,
                'total_attempts': total_attempts,
                'successful_retries': self.metrics['successful_retries'],
                'failed_retries': self.metrics['failed_retries'],
                'max_retries_exceeded': self.metrics['max_retries_exceeded'],
                'success_rate': success_rate,
                'config': {
                    'max_attempts': self.config.max_attempts,
                    'strategy': self.config.strategy.value,
                    'base_delay': self.config.base_delay,
                    'max_delay': self.config.max_delay
                }
            }

class ReliabilityManager:
    """
    Central manager for reliability patterns (circuit breakers + retries).
    
    Provides a unified interface for managing multiple reliability mechanisms
    and comprehensive monitoring.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self._lock = threading.Lock()
        
        logger.info("Reliability manager initialized")
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        with self._lock:
            if name in self.circuit_breakers:
                logger.warning(f"Circuit breaker '{name}' already exists, returning existing instance")
                return self.circuit_breakers[name]
            
            breaker = CircuitBreaker(name, config)
            self.circuit_breakers[name] = breaker
            
            logger.info(f"Created circuit breaker '{name}'", 
                       breaker_name=name)
            return breaker
    
    def create_retry_handler(self, name: str, config: RetryConfig) -> RetryHandler:
        """Create and register a retry handler."""
        with self._lock:
            if name in self.retry_handlers:
                logger.warning(f"Retry handler '{name}' already exists, returning existing instance")
                return self.retry_handlers[name]
            
            handler = RetryHandler(name, config)
            self.retry_handlers[name] = handler
            
            logger.info(f"Created retry handler '{name}'",
                       handler_name=name)
            return handler
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def get_retry_handler(self, name: str) -> Optional[RetryHandler]:
        """Get retry handler by name."""
        return self.retry_handlers.get(name)
    
    def reliable_call(self, 
                     func: Callable,
                     circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                     retry_config: Optional[RetryConfig] = None,
                     name: Optional[str] = None,
                     *args, **kwargs) -> Any:
        """
        Execute a function call with comprehensive reliability patterns.
        
        Combines circuit breaker and retry patterns for maximum reliability.
        """
        if name is None:
            name = f"{func.__module__}.{func.__name__}"
        
        # Create or get circuit breaker
        breaker = None
        if circuit_breaker_config:
            breaker_name = f"{name}_breaker"
            if breaker_name not in self.circuit_breakers:
                breaker = self.create_circuit_breaker(breaker_name, circuit_breaker_config)
            else:
                breaker = self.circuit_breakers[breaker_name]
        
        # Create or get retry handler
        retry_handler = None
        if retry_config:
            handler_name = f"{name}_retry"
            if handler_name not in self.retry_handlers:
                retry_handler = self.create_retry_handler(handler_name, retry_config)
            else:
                retry_handler = self.retry_handlers[handler_name]
        
        # Execute with reliability patterns
        def execute():
            if breaker:
                return breaker.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        if retry_handler:
            return retry_handler.call(execute)
        else:
            return execute()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health from reliability perspective."""
        with self._lock:
            circuit_breaker_stats = {}
            total_cb_calls = 0
            open_breakers = 0
            
            for name, breaker in self.circuit_breakers.items():
                metrics = breaker.get_metrics()
                circuit_breaker_stats[name] = metrics
                total_cb_calls += metrics['total_calls']
                if metrics['state'] == 'open':
                    open_breakers += 1
            
            retry_handler_stats = {}
            total_retry_attempts = 0
            
            for name, handler in self.retry_handlers.items():
                metrics = handler.get_metrics()
                retry_handler_stats[name] = metrics
                total_retry_attempts += metrics['total_attempts']
            
            # Calculate overall health score
            health_score = 1.0
            if len(self.circuit_breakers) > 0:
                health_score -= (open_breakers / len(self.circuit_breakers)) * 0.5
            
            system_status = "healthy"
            if open_breakers > 0:
                system_status = "degraded" if open_breakers < len(self.circuit_breakers) else "critical"
            
            return {
                'system_status': system_status,
                'health_score': health_score,
                'circuit_breakers': {
                    'total_count': len(self.circuit_breakers),
                    'open_count': open_breakers,
                    'total_calls': total_cb_calls,
                    'details': circuit_breaker_stats
                },
                'retry_handlers': {
                    'total_count': len(self.retry_handlers),
                    'total_attempts': total_retry_attempts,
                    'details': retry_handler_stats
                }
            }
    
    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers to closed state."""
        with self._lock:
            for breaker in self.circuit_breakers.values():
                breaker.reset()
        
        logger.info("All circuit breakers reset",
                   count=len(self.circuit_breakers))

# Global reliability manager
_global_reliability_manager = ReliabilityManager()

def get_reliability_manager() -> ReliabilityManager:
    """Get the global reliability manager instance."""
    return _global_reliability_manager

def circuit_breaker(name: str = None, **config_kwargs):
    """
    Decorator to apply circuit breaker pattern to a function.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        **config_kwargs: Circuit breaker configuration parameters
    """
    def decorator(func: Callable) -> Callable:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        config = CircuitBreakerConfig(**config_kwargs)
        manager = get_reliability_manager()
        breaker = manager.create_circuit_breaker(breaker_name, config)
        return breaker(func)
    return decorator

def retry(name: str = None, **config_kwargs):
    """
    Decorator to apply retry pattern to a function.
    
    Args:
        name: Retry handler name (defaults to function name)
        **config_kwargs: Retry configuration parameters
    """
    def decorator(func: Callable) -> Callable:
        handler_name = name or f"{func.__module__}.{func.__name__}"
        config = RetryConfig(**config_kwargs)
        manager = get_reliability_manager()
        handler = manager.create_retry_handler(handler_name, config)
        return handler(func)
    return decorator

def reliable(circuit_breaker_config: Dict[str, Any] = None,
            retry_config: Dict[str, Any] = None,
            name: str = None):
    """
    Decorator to apply both circuit breaker and retry patterns.
    
    Args:
        circuit_breaker_config: Circuit breaker configuration dict
        retry_config: Retry configuration dict
        name: Base name for reliability patterns
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cb_config = CircuitBreakerConfig(**circuit_breaker_config) if circuit_breaker_config else None
            r_config = RetryConfig(**retry_config) if retry_config else None
            
            manager = get_reliability_manager()
            return manager.reliable_call(func, cb_config, r_config, name, *args, **kwargs)
        return wrapper
    return decorator