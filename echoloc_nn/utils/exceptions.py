"""
Custom exceptions for EchoLoc-NN system.
"""

class EchoLocError(Exception):
    """Base exception for EchoLoc-NN system."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        
    def __str__(self):
        error_str = super().__str__()
        if self.error_code:
            error_str = f"[{self.error_code}] {error_str}"
        return error_str


class ModelError(EchoLocError):
    """Errors related to model loading, inference, or configuration."""
    pass


class HardwareError(EchoLocError):
    """Errors related to hardware communication and control."""
    pass


class CalibrationError(EchoLocError):
    """Errors during sensor calibration process."""
    pass


class InferenceError(EchoLocError):
    """Errors during inference/localization process."""
    pass


class ValidationError(EchoLocError):
    """Data validation errors."""
    pass


class ConfigurationError(EchoLocError):
    """Configuration and setup errors."""
    pass


class TimeoutError(EchoLocError):
    """Timeout errors for operations with time constraints."""
    pass


class ResourceError(EchoLocError):
    """Resource availability errors (memory, compute, etc.)."""
    pass


class SecurityError(EchoLocError):
    """Security-related errors."""
    pass