"""
Real-time inference engine for EchoLoc-NN.

This module provides optimized inference pipelines for real-time
ultrasonic localization with support for edge deployment.
"""

from .locator import EchoLocator, InferenceConfig
# from .batch_processor import BatchProcessor, StreamProcessor  # TODO: Implement batch processing
# from .optimization import ModelOptimizer, QuantizationConfig  # TODO: Implement optimization

__all__ = [
    "EchoLocator",
    "InferenceConfig",
    # "BatchProcessor",
    # "StreamProcessor", 
    # "ModelOptimizer",
    # "QuantizationConfig"
]