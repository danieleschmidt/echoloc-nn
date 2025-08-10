"""
Enhanced structured logging configuration for EchoLoc-NN Generation 2.

Provides comprehensive logging with multiple levels, structured output,
performance metrics integration, and advanced error tracking.
"""

import logging
import sys
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import traceback
import threading
from pathlib import Path
from collections import deque, defaultdict

class PerformanceFilter(logging.Filter):
    """Custom filter for performance-related log records."""
    
    def filter(self, record):
        return hasattr(record, 'performance') and record.performance

class SecurityFilter(logging.Filter):
    """Custom filter for security-related log records."""
    
    def filter(self, record):
        return hasattr(record, 'security') and record.security

class ErrorFilter(logging.Filter):
    """Custom filter for error analysis and tracking."""
    
    def filter(self, record):
        return record.levelno >= logging.ERROR

class StructuredFormatter(logging.Formatter):
    """
    Enhanced structured JSON formatter for machine-readable logs.
    
    Adds structured fields including timing, component, context information,
    and error tracking capabilities.
    """
    
    def __init__(self, include_extra: bool = True, include_stack: bool = False):
        super().__init__()
        self.include_extra = include_extra
        self.include_stack = include_stack
        
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': getattr(record, 'threadName', 'MainThread'),
            'process': record.process
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add stack trace for errors if enabled
        if self.include_stack and record.levelno >= logging.ERROR:
            log_entry['stack_trace'] = ''.join(traceback.format_stack())
        
        # Add extra fields if enabled
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                              'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process', 'getMessage',
                              'extra']:
                    # Handle complex objects by converting to string
                    try:
                        json.dumps(value)  # Test if JSON serializable
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)
        
        return json.dumps(log_entry, default=str)

class LogAnalyzer:
    """
    Real-time log analysis and metrics collection.
    
    Tracks error patterns, performance trends, and system health indicators.
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.error_history = deque(maxlen=max_history)
        self.performance_history = deque(maxlen=max_history)
        self.component_stats = defaultdict(lambda: {'count': 0, 'errors': 0, 'warnings': 0})
        self.error_patterns = defaultdict(int)
        self._lock = threading.Lock()
        
    def analyze_record(self, record):
        """Analyze a log record and update metrics."""
        with self._lock:
            # Update component statistics
            component = getattr(record, 'component', 'unknown')
            self.component_stats[component]['count'] += 1
            
            if record.levelno >= logging.ERROR:
                self.component_stats[component]['errors'] += 1
                self.error_history.append({
                    'timestamp': record.created,
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'component': component,
                    'module': record.module,
                    'function': record.funcName
                })
                
                # Track error patterns
                error_key = f"{record.module}.{record.funcName}"
                self.error_patterns[error_key] += 1
                
            elif record.levelno >= logging.WARNING:
                self.component_stats[component]['warnings'] += 1
            
            # Track performance metrics
            if hasattr(record, 'performance') and record.performance:
                perf_data = {
                    'timestamp': record.created,
                    'operation': getattr(record, 'operation', 'unknown'),
                    'duration_ms': getattr(record, 'duration_ms', 0),
                    'success': getattr(record, 'success', True),
                    'component': component
                }
                self.performance_history.append(perf_data)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error analysis summary."""
        with self._lock:
            recent_errors = [e for e in self.error_history if time.time() - e['timestamp'] < 3600]  # Last hour
            
            return {
                'total_errors': len(self.error_history),
                'recent_errors': len(recent_errors),
                'error_rate_per_hour': len(recent_errors),
                'top_error_patterns': dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
                'component_error_stats': {k: {'errors': v['errors'], 'error_rate': v['errors'] / max(v['count'], 1)} 
                                        for k, v in self.component_stats.items() if v['errors'] > 0}
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance analysis summary."""
        with self._lock:
            if not self.performance_history:
                return {'no_data': True}
            
            recent_perf = [p for p in self.performance_history if time.time() - p['timestamp'] < 3600]
            
            if not recent_perf:
                return {'no_recent_data': True}
            
            # Calculate performance statistics
            durations = [p['duration_ms'] for p in recent_perf]
            success_rate = sum(1 for p in recent_perf if p['success']) / len(recent_perf)
            
            # Group by operation
            by_operation = defaultdict(list)
            for p in recent_perf:
                by_operation[p['operation']].append(p['duration_ms'])
            
            operation_stats = {
                op: {
                    'count': len(durations),
                    'avg_duration_ms': sum(durations) / len(durations),
                    'max_duration_ms': max(durations),
                    'min_duration_ms': min(durations)
                }
                for op, durations in by_operation.items()
            }
            
            return {
                'total_operations': len(self.performance_history),
                'recent_operations': len(recent_perf),
                'overall_success_rate': success_rate,
                'avg_duration_ms': sum(durations) / len(durations),
                'operation_stats': operation_stats
            }

class EchoLocLogger:
    """
    Enhanced logger for EchoLoc-NN Generation 2 with advanced capabilities.
    
    Provides specialized logging methods for different components,
    automatic performance metrics collection, and error analysis.
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.name = name
        
        # Performance tracking
        self._operation_times: Dict[str, float] = {}
        self._call_counts: Dict[str, int] = {}
        
        # Error tracking
        self._error_count = 0
        self._warning_count = 0
        
        # Context tracking
        self._context_stack: List[str] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
    def push_context(self, context: str) -> None:
        """Push a context onto the context stack."""
        with self._lock:
            self._context_stack.append(context)
    
    def pop_context(self) -> Optional[str]:
        """Pop a context from the context stack."""
        with self._lock:
            return self._context_stack.pop() if self._context_stack else None
    
    def get_context(self) -> str:
        """Get current context as string."""
        with self._lock:
            return " -> ".join(self._context_stack) if self._context_stack else ""
    
    def start_operation(self, operation_name: str) -> str:
        """Start timing an operation."""
        with self._lock:
            operation_id = f"{operation_name}_{int(time.time() * 1000000)}"  # Microsecond precision
            self._operation_times[operation_id] = time.time()
            
        self.debug(f"Starting operation: {operation_name}", 
                  operation=operation_name, 
                  operation_id=operation_id,
                  context=self.get_context())
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, **kwargs) -> float:
        """End timing an operation and log the result."""
        with self._lock:
            if operation_id not in self._operation_times:
                self.warning(f"Unknown operation ID: {operation_id}")
                return 0.0
            
            start_time = self._operation_times.pop(operation_id)
            duration = time.time() - start_time
            
            # Extract operation name
            operation_name = operation_id.rsplit('_', 1)[0]
            
            # Update call counts
            self._call_counts[operation_name] = self._call_counts.get(operation_name, 0) + 1
        
        # Log performance metrics
        self.performance(
            f"Operation {operation_name} {'completed' if success else 'failed'} in {duration*1000:.2f}ms",
            operation=operation_name,
            operation_id=operation_id,
            duration_ms=duration * 1000,
            duration_seconds=duration,
            success=success,
            call_count=self._call_counts[operation_name],
            context=self.get_context(),
            **kwargs
        )
        
        return duration
    
    def performance(self, message: str, **kwargs):
        """Log performance metrics."""
        extra = {'performance': True, 'component': 'performance', **kwargs}
        self.logger.info(message, extra=extra)
    
    def security(self, message: str, event_type: str = 'security_event', **kwargs):
        """Log security-related events."""
        extra = {
            'security': True, 
            'component': 'security',
            'event_type': event_type,
            'context': self.get_context(),
            **kwargs
        }
        self.logger.warning(message, extra=extra)
    
    def model_info(self, message: str, model_name: str = None, **kwargs):
        """Log model-specific information."""
        extra = {
            'component': 'model', 
            'model_name': model_name,
            'context': self.get_context(),
            **kwargs
        }
        self.logger.info(message, extra=extra)
    
    def model_error(self, message: str, model_name: str = None, error_type: str = 'model_error', **kwargs):
        """Log model-specific errors."""
        with self._lock:
            self._error_count += 1
        
        extra = {
            'component': 'model',
            'model_name': model_name,
            'error_type': error_type,
            'error_count': self._error_count,
            'context': self.get_context(),
            **kwargs
        }
        self.logger.error(message, extra=extra)
    
    def hardware_info(self, message: str, device: str = None, **kwargs):
        """Log hardware-related information."""
        extra = {
            'component': 'hardware', 
            'device': device,
            'context': self.get_context(),
            **kwargs
        }
        self.logger.info(message, extra=extra)
    
    def hardware_error(self, message: str, device: str = None, error_type: str = 'hardware_error', **kwargs):
        """Log hardware-related errors."""
        with self._lock:
            self._error_count += 1
        
        extra = {
            'component': 'hardware',
            'device': device,
            'error_type': error_type,
            'error_count': self._error_count,
            'context': self.get_context(),
            **kwargs
        }
        self.logger.error(message, extra=extra)
    
    def signal_processing(self, message: str, signal_type: str = None, **kwargs):
        """Log signal processing information."""
        extra = {
            'component': 'signal_processing',
            'signal_type': signal_type,
            'context': self.get_context(),
            **kwargs
        }
        self.logger.debug(message, extra=extra)
    
    def validation_error(self, message: str, field_name: str = None, value: Any = None, **kwargs):
        """Log validation errors."""
        with self._lock:
            self._error_count += 1
        
        extra = {
            'component': 'validation',
            'error_type': 'validation_error',
            'field_name': field_name,
            'invalid_value': str(value) if value is not None else None,
            'error_count': self._error_count,
            'context': self.get_context(),
            **kwargs
        }
        self.logger.error(message, extra=extra)
    
    def circuit_breaker(self, message: str, breaker_name: str, state: str, **kwargs):
        """Log circuit breaker events."""
        extra = {
            'component': 'circuit_breaker',
            'breaker_name': breaker_name,
            'breaker_state': state,
            'context': self.get_context(),
            **kwargs
        }
        level_method = self.logger.warning if state in ['open', 'half_open'] else self.logger.info
        level_method(message, extra=extra)
    
    def health_check(self, message: str, component: str, status: str, **kwargs):
        """Log health check results."""
        extra = {
            'component': 'health_check',
            'health_component': component,
            'health_status': status,
            'context': self.get_context(),
            **kwargs
        }
        level_method = self.logger.error if status == 'unhealthy' else self.logger.info
        level_method(message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug information."""
        extra = {'context': self.get_context(), **kwargs}
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """Log info information."""
        extra = {'context': self.get_context(), **kwargs}
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        """Log warning information."""
        with self._lock:
            self._warning_count += 1
        
        extra = {
            'warning_count': self._warning_count,
            'context': self.get_context(),
            **kwargs
        }
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs):
        """Log error information."""
        with self._lock:
            self._error_count += 1
        
        extra = {
            'error_count': self._error_count,
            'context': self.get_context(),
            **kwargs
        }
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, **kwargs):
        """Log critical information."""
        with self._lock:
            self._error_count += 1
        
        extra = {
            'error_count': self._error_count,
            'context': self.get_context(),
            **kwargs
        }
        self.logger.critical(message, extra=extra)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self._lock:
            return {
                'logger_name': self.name,
                'active_operations': len(self._operation_times),
                'operation_counts': self._call_counts.copy(),
                'total_operations': sum(self._call_counts.values()),
                'error_count': self._error_count,
                'warning_count': self._warning_count,
                'current_context': self.get_context(),
                'context_depth': len(self._context_stack)
            }

# Global log analyzer instance
_global_log_analyzer = LogAnalyzer()

class AnalyzingHandler(logging.Handler):
    """Handler that feeds records to the log analyzer."""
    
    def emit(self, record):
        _global_log_analyzer.analyze_record(record)

# Global logging configuration
def setup_enhanced_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
    enable_performance: bool = True,
    enable_security: bool = True,
    enable_analysis: bool = True,
    include_stack_traces: bool = False
) -> None:
    """
    Configure enhanced structured logging for EchoLoc-NN Generation 2.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        structured: Use structured JSON formatting
        enable_performance: Enable performance logging
        enable_security: Enable security logging  
        enable_analysis: Enable real-time log analysis
        include_stack_traces: Include stack traces in error logs
    """
    
    # Convert string level to logging level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    if structured:
        formatter = StructuredFormatter(include_extra=True, include_stack=include_stack_traces)
        console_formatter = StructuredFormatter(include_extra=False, include_stack=False)  # Less verbose for console
    else:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - [%(component)s] %(message)s'
        formatter = logging.Formatter(format_string, defaults={'component': 'unknown'})
        console_formatter = formatter
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Ensure log directory exists
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Separate error log file
        error_log_file = log_dir / f"error_{Path(log_file).name}"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.addFilter(ErrorFilter())
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
    
    # Analysis handler
    if enable_analysis:
        analysis_handler = AnalyzingHandler()
        analysis_handler.setLevel(logging.DEBUG)  # Analyze all logs
        root_logger.addHandler(analysis_handler)
    
    # Performance logging handler
    if enable_performance:
        perf_log_file = None
        if log_file:
            perf_log_file = Path(log_file).parent / f"performance_{Path(log_file).name}"
            perf_handler = logging.FileHandler(perf_log_file)
        else:
            perf_handler = logging.StreamHandler(sys.stdout)
        
        perf_handler.setLevel(logging.INFO)
        perf_handler.addFilter(PerformanceFilter())
        perf_handler.setFormatter(formatter)
        root_logger.addHandler(perf_handler)
    
    # Security logging handler  
    if enable_security:
        sec_log_file = None
        if log_file:
            sec_log_file = Path(log_file).parent / f"security_{Path(log_file).name}"
            sec_handler = logging.FileHandler(sec_log_file)
        else:
            sec_handler = logging.StreamHandler(sys.stderr)
        
        sec_handler.setLevel(logging.WARNING)
        sec_handler.addFilter(SecurityFilter())
        sec_handler.setFormatter(formatter)
        root_logger.addHandler(sec_handler)

def get_enhanced_logger(name: str, level: int = logging.INFO) -> EchoLocLogger:
    """
    Get an enhanced logger instance for EchoLoc-NN Generation 2.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Enhanced logger instance
    """
    return EchoLocLogger(name, level)

def get_log_analyzer() -> LogAnalyzer:
    """Get the global log analyzer instance."""
    return _global_log_analyzer

def get_logging_summary() -> Dict[str, Any]:
    """Get comprehensive logging system summary."""
    analyzer = get_log_analyzer()
    return {
        'error_summary': analyzer.get_error_summary(),
        'performance_summary': analyzer.get_performance_summary(),
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }