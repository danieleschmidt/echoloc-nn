"""
Utility functions and classes for EchoLoc-NN.

This module provides validation, error handling, logging,
and other utility functions used throughout the system.
"""

try:
    from .validation import (
        EchoDataValidator,
        PositionValidator,
        SensorConfigValidator,
        ValidationError
    )
except ImportError:
    # Validation not available without PyTorch
    EchoDataValidator = None
    PositionValidator = None
    SensorConfigValidator = None
    ValidationError = Exception
from .exceptions import (
    EchoLocError,
    ModelError,
    HardwareError,
    CalibrationError,
    InferenceError
)
from .logging_config import setup_logging, get_logger
try:
    from .monitoring import PerformanceMonitor, HealthChecker
    from .security import InputSanitizer, SecurityValidator
except ImportError:
    # Monitoring and security not available without dependencies
    PerformanceMonitor = None
    HealthChecker = None
    InputSanitizer = None
    SecurityValidator = None

__all__ = [
    "EchoDataValidator",
    "PositionValidator", 
    "SensorConfigValidator",
    "ValidationError",
    "EchoLocError",
    "ModelError",
    "HardwareError",
    "CalibrationError",
    "InferenceError",
    "setup_logging",
    "get_logger",
    "PerformanceMonitor",
    "HealthChecker",
    "InputSanitizer",
    "SecurityValidator"
]