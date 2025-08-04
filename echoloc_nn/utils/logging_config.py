"""
Centralized logging configuration for EchoLoc-NN.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry)


class EchoLocAdapter(logging.LoggerAdapter):
    """Custom logger adapter for EchoLoc-NN with extra context."""
    
    def process(self, msg, kwargs):
        # Add extra context to log records
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = {'extra_fields': extra}
        return msg, kwargs
    
    def performance(self, msg, duration_ms=None, **kwargs):
        """Log performance metrics."""
        extra = kwargs.get('extra', {})
        extra['category'] = 'performance'
        if duration_ms is not None:
            extra['duration_ms'] = duration_ms
        kwargs['extra'] = extra
        self.info(msg, **kwargs)
    
    def hardware(self, msg, device=None, **kwargs):
        """Log hardware-related events."""
        extra = kwargs.get('extra', {})
        extra['category'] = 'hardware'
        if device is not None:
            extra['device'] = device
        kwargs['extra'] = extra
        self.info(msg, **kwargs)
    
    def inference(self, msg, position=None, confidence=None, **kwargs):
        """Log inference results."""
        extra = kwargs.get('extra', {})
        extra['category'] = 'inference'
        if position is not None:
            extra['position'] = position.tolist() if hasattr(position, 'tolist') else position
        if confidence is not None:
            extra['confidence'] = float(confidence)
        kwargs['extra'] = extra
        self.info(msg, **kwargs)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    json_format: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    component: Optional[str] = None
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_dir: Directory for log files (None to disable file logging)
        console_output: Enable console output
        json_format: Use JSON formatting
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        component: Component name for log file naming
        
    Returns:
        Configured logger instance
    """
    # Convert level string to constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    logger = logging.getLogger('echoloc_nn')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate filename
        if component:
            log_filename = f"echoloc_{component}.log"
        else:
            log_filename = "echoloc.log"
        
        log_file_path = os.path.join(log_dir, log_filename)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add error file handler for errors and above
    if log_dir is not None:
        error_log_path = os.path.join(log_dir, "echoloc_errors.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    logger.info(f"Logging initialized - Level: {level}, JSON: {json_format}")
    if log_dir:
        logger.info(f"Log directory: {log_dir}")
    
    return logger


def get_logger(name: str, extra_context: Optional[Dict[str, Any]] = None) -> EchoLocAdapter:
    """
    Get a logger adapter with extra context.
    
    Args:
        name: Logger name (usually module name)
        extra_context: Extra context to include in all log messages
        
    Returns:
        Logger adapter instance
    """
    base_logger = logging.getLogger(f'echoloc_nn.{name}')
    extra = extra_context or {}
    return EchoLocAdapter(base_logger, extra)


def setup_performance_logging(log_dir: str) -> logging.Logger:
    """
    Setup dedicated performance logging.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Performance logger
    """
    perf_logger = logging.getLogger('echoloc_nn.performance')
    perf_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    perf_logger.handlers.clear()
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Performance log file
    perf_log_path = os.path.join(log_dir, "performance.log")
    
    # Use JSON formatter for structured performance data
    formatter = JSONFormatter()
    
    handler = logging.handlers.RotatingFileHandler(
        perf_log_path,
        maxBytes=50 * 1024 * 1024,  # 50MB for performance logs
        backupCount=10
    )
    handler.setFormatter(formatter)
    perf_logger.addHandler(handler)
    
    # Prevent propagation to root logger
    perf_logger.propagate = False
    
    return perf_logger


def configure_third_party_logging():
    """Configure logging for third-party libraries."""
    # Reduce verbosity of common libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Set PyTorch logging
    if 'torch' in sys.modules:
        logging.getLogger('torch').setLevel(logging.WARNING)


class LoggingContext:
    """Context manager for adding extra context to logs."""
    
    def __init__(self, logger: EchoLocAdapter, **extra_context):
        self.logger = logger
        self.original_extra = logger.extra.copy()
        self.extra_context = extra_context
    
    def __enter__(self):
        self.logger.extra.update(self.extra_context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.extra = self.original_extra


# Global logger instance
_global_logger = None


def get_global_logger() -> EchoLocAdapter:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        setup_logging()
        _global_logger = get_logger('global')
    return _global_logger