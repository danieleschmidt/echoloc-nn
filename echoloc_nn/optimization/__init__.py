"""
Performance optimization and scaling utilities for EchoLoc-NN.

This module provides model optimization, caching, concurrent processing,
and auto-scaling capabilities for production deployment.
"""

try:
    from .model_optimizer import (
        ModelOptimizer,
        QuantizationConfig,
        PruningConfig,
        OptimizationResult
    )
    from .caching import (
        EchoCache,
        ModelCache,
        CacheConfig,
        CacheStats
    )
    from .concurrent_processor import (
        ConcurrentProcessor,
        ProcessorPool,
        BatchProcessor
    )
    from .auto_scaler import (
        AutoScaler,
        ScalingConfig,
        ResourceMonitor
    )
except ImportError:
    # Optimization modules not available without PyTorch
    ModelOptimizer = None
    QuantizationConfig = None
    PruningConfig = None
    OptimizationResult = None
    EchoCache = None
    ModelCache = None
    CacheConfig = None
    CacheStats = None
    ConcurrentProcessor = None
    ProcessorPool = None
    BatchProcessor = None
    AutoScaler = None
    ScalingConfig = None
    ResourceMonitor = None

# Always available - pure Python implementations
from .performance_engine import PerformanceEngine, PerformanceConfig

__all__ = [
    "ModelOptimizer",
    "QuantizationConfig",
    "PruningConfig", 
    "OptimizationResult",
    "EchoCache",
    "ModelCache",
    "CacheConfig",
    "CacheStats",
    "ConcurrentProcessor",
    "ProcessorPool",
    "BatchProcessor",
    "AutoScaler",
    "ScalingConfig",
    "ResourceMonitor",
    "PerformanceEngine",
    "PerformanceConfig"
]