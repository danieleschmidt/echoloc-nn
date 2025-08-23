"""
Health monitoring and system diagnostics for EchoLoc-NN.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import time
import threading
from collections import deque
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """System health status."""
    
    component: str
    status: str  # 'healthy', 'degraded', 'critical', 'down'
    timestamp: float
    message: str
    metrics: Dict[str, Any]


class HealthMonitor:
    """Monitor system health and performance."""
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize health monitor."""
        self.check_interval = check_interval
        self.checks = {}
        self.status_history = deque(maxlen=1000)
        self.running = False
        self.monitor_thread = None
        logger.info(f"HealthMonitor initialized (interval: {check_interval}s)")
    
    def register_check(self, name: str, check_func: Callable[[], HealthStatus]):
        """Register a health check."""
        self.checks[name] = check_func
        logger.info(f"Health check registered: {name}")
    
    def start_monitoring(self):
        """Start monitoring in background thread."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                for name, check_func in self.checks.items():
                    try:
                        status = check_func()
                        self.status_history.append(status)
                        
                        if status.status in ['critical', 'down']:
                            logger.error(f"Health check failed: {name} - {status.message}")
                        elif status.status == 'degraded':
                            logger.warning(f"Health check degraded: {name} - {status.message}")
                        
                    except Exception as e:
                        error_status = HealthStatus(
                            component=name,
                            status='down',
                            timestamp=time.time(),
                            message=f"Health check failed with exception: {e}",
                            metrics={}
                        )
                        self.status_history.append(error_status)
                        logger.error(f"Health check exception: {name} - {e}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(self.check_interval)
    
    def get_current_status(self) -> Dict[str, HealthStatus]:
        """Get current health status for all components."""
        current_status = {}
        
        # Get latest status for each component
        for status in reversed(self.status_history):
            if status.component not in current_status:
                current_status[status.component] = status
        
        return current_status
    
    def get_overall_health(self) -> str:
        """Get overall system health status."""
        current = self.get_current_status()
        
        if not current:
            return 'unknown'
        
        statuses = [s.status for s in current.values()]
        
        if 'down' in statuses:
            return 'down'
        elif 'critical' in statuses:
            return 'critical'
        elif 'degraded' in statuses:
            return 'degraded'
        else:
            return 'healthy'


class SystemDiagnostics:
    """System diagnostics and troubleshooting."""
    
    def __init__(self):
        """Initialize system diagnostics."""
        self.diagnostic_data = {}
        logger.info("SystemDiagnostics initialized")
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics."""
        diagnostics = {
            'timestamp': time.time(),
            'system_info': self._get_system_info(),
            'memory_usage': self._get_memory_usage(),
            'performance_metrics': self._get_performance_metrics(),
            'dependency_status': self._check_dependencies()
        }
        
        self.diagnostic_data.update(diagnostics)
        logger.info("System diagnostics completed")
        return diagnostics
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0]
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'percent_used': memory.percent,
                'free': memory.free
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'cpu_count': self._get_cpu_count(),
            'load_average': self._get_load_average()
        }
    
    def _get_cpu_count(self) -> int:
        """Get CPU count."""
        import os
        return os.cpu_count() or 1
    
    def _get_load_average(self) -> Optional[List[float]]:
        """Get system load average."""
        try:
            import os
            return list(os.getloadavg())
        except (OSError, AttributeError):
            return None
    
    def _check_dependencies(self) -> Dict[str, str]:
        """Check status of key dependencies."""
        dependencies = {
            'numpy': 'unknown',
            'scipy': 'unknown',
            'torch': 'unknown',
            'yaml': 'unknown'
        }
        
        for dep in dependencies:
            try:
                if dep == 'numpy':
                    import numpy
                    dependencies[dep] = f"available (v{numpy.__version__})"
                elif dep == 'scipy':
                    import scipy
                    dependencies[dep] = f"available (v{scipy.__version__})"
                elif dep == 'torch':
                    import torch
                    dependencies[dep] = f"available (v{torch.__version__})"
                elif dep == 'yaml':
                    import yaml
                    dependencies[dep] = "available"
                    
            except ImportError:
                dependencies[dep] = 'missing'
        
        return dependencies


class PerformanceProfiler:
    """Profile system performance."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.profiles = {}
        logger.info("PerformanceProfiler initialized")
    
    def start_profile(self, name: str):
        """Start profiling a component."""
        self.profiles[name] = {
            'start_time': time.time(),
            'end_time': None,
            'duration': None
        }
        logger.debug(f"Started profiling: {name}")
    
    def end_profile(self, name: str) -> Optional[float]:
        """End profiling and return duration."""
        if name not in self.profiles:
            logger.warning(f"No profile started for: {name}")
            return None
        
        end_time = time.time()
        duration = end_time - self.profiles[name]['start_time']
        
        self.profiles[name].update({
            'end_time': end_time,
            'duration': duration
        })
        
        logger.debug(f"Profile completed: {name} - {duration:.3f}s")
        return duration
    
    def get_profile_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all profiles."""
        return self.profiles.copy()


class AlertManager:
    """Manage system alerts and notifications."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts = deque(maxlen=1000)
        self.alert_handlers = []
        logger.info("AlertManager initialized")
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
        logger.info("Alert handler added")
    
    def send_alert(self, level: str, message: str, component: str = None, metadata: Dict[str, Any] = None):
        """Send an alert."""
        alert = {
            'level': level,  # 'info', 'warning', 'error', 'critical'
            'message': message,
            'component': component or 'system',
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.alerts.append(alert)
        
        # Send to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        # Log alert
        log_func = getattr(logger, level, logger.info)
        log_func(f"Alert: {message} (component: {component})")
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return list(self.alerts)[-limit:]
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()
        logger.info("All alerts cleared")