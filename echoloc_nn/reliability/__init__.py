"""
EchoLoc-NN Reliability Framework

Comprehensive reliability, fault tolerance, and robustness components
for production-grade quantum-spatial localization systems.

Components:
- fault_tolerance: System fault handling and recovery
- health_monitoring: Real-time health checks and diagnostics  
- circuit_breaker: Circuit breaker patterns for system protection
- graceful_degradation: Fallback mechanisms when components fail
- self_healing: Autonomous recovery and adaptation systems
"""

from .fault_tolerance import (
    FaultTolerantEchoLocator,
    RedundantSensorArray,
    FaultDetector,
    RecoveryManager
)

from .health_monitoring import (
    HealthMonitor,
    SystemDiagnostics,
    PerformanceProfiler,
    AlertManager
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState
)

# from .graceful_degradation import (
#     DegradationManager,
#     FallbackLocator,
#     QualityAssurance
# )

# from .self_healing import (
#     SelfHealingSystem,
#     AdaptiveCalibration,
#     AutoRecovery
# )

__all__ = [
    # Fault tolerance
    "FaultTolerantEchoLocator",
    "RedundantSensorArray", 
    "FaultDetector",
    "RecoveryManager",
    
    # Health monitoring
    "HealthMonitor",
    "SystemDiagnostics",
    "PerformanceProfiler", 
    "AlertManager",
    
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerManager",
    "BreakerPolicy",
    
    # Graceful degradation
    "DegradationManager",
    "FallbackLocator",
    "QualityAssurance",
    
    # Self-healing
    "SelfHealingSystem",
    "AdaptiveCalibration",
    "AutoRecovery"
]