"""
Performance optimization and scaling utilities for EchoLoc-NN.

This module provides model optimization, caching, concurrent processing,
and auto-scaling capabilities for production deployment.
"""

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
    "ResourceMonitor"
]