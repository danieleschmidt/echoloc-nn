"""
Signal processing utilities for ultrasonic localization.

This module provides tools for chirp generation, echo processing,
filtering, and feature extraction from ultrasonic sensor data.
"""

try:
    from .chirp_generator import ChirpGenerator, ChirpDesigner
    from .echo_processing import EchoProcessor, EchoEnhancer  
    from .preprocessing import PreProcessor, SignalNormalizer
    from .beamforming import BeamFormer, DelayAndSum
    from .feature_extraction import FeatureExtractor, SpectralFeatures
    
    __all__ = [
        "ChirpGenerator",
        "ChirpDesigner", 
        "EchoProcessor",
        "EchoEnhancer",
        "PreProcessor",
        "SignalNormalizer",
        "BeamFormer",
        "DelayAndSum",
        "FeatureExtractor", 
        "SpectralFeatures"
    ]
except ImportError:
    # Fallback to simple processing when dependencies not available
    from .simple_processing import SimpleChirpGenerator as ChirpGenerator
    from .simple_processing import SimpleEchoProcessor as EchoProcessor
    from .simple_processing import SimplePreProcessor as PreProcessor
    
    # Aliases for compatibility
    ChirpDesigner = ChirpGenerator
    EchoEnhancer = EchoProcessor
    SignalNormalizer = PreProcessor
    BeamFormer = EchoProcessor
    DelayAndSum = EchoProcessor
    FeatureExtractor = EchoProcessor
    SpectralFeatures = EchoProcessor
    
    __all__ = [
        "ChirpGenerator",
        "ChirpDesigner",
        "EchoProcessor",
        "EchoEnhancer", 
        "PreProcessor",
        "SignalNormalizer",
        "BeamFormer",
        "DelayAndSum",
        "FeatureExtractor",
        "SpectralFeatures"
    ]