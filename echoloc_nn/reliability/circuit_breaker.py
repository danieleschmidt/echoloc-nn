"""
Circuit breaker implementation for fault tolerance.
"""

from typing import Callable, Any, Optional
import time
import threading
from enum import Enum
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Timeout in seconds before transitioning to half-open
            expected_exception: Exception type that counts as failure
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
        
        logger.info(f"CircuitBreaker initialized (threshold: {failure_threshold}, timeout: {timeout}s)")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count
                if self.state == CircuitState.HALF_OPEN:
                    self._reset()
                    logger.info("Circuit breaker reset to CLOSED")
                
                return result
                
            except self.expected_exception as e:
                self._record_failure()
                logger.warning(f"Circuit breaker recorded failure: {e}")
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.timeout
        )
    
    def _record_failure(self):
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_failure_count(self) -> int:
        """Get current failure count."""
        return self.failure_count
    
    def force_open(self):
        """Manually open the circuit breaker."""
        with self._lock:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            logger.warning("Circuit breaker manually forced OPEN")
    
    def force_close(self):
        """Manually close the circuit breaker."""
        with self._lock:
            self._reset()
            logger.info("Circuit breaker manually forced CLOSED")


def circuit_breaker(
    failure_threshold: int = 5,
    timeout: float = 60.0,
    expected_exception: type = Exception
):
    """
    Decorator for adding circuit breaker to functions.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        timeout: Timeout before attempting reset
        expected_exception: Exception type that counts as failure
    """
    def decorator(func: Callable) -> Callable:
        breaker = CircuitBreaker(failure_threshold, timeout, expected_exception)
        
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        wrapper._circuit_breaker = breaker  # Allow access to breaker
        return wrapper
    
    return decorator