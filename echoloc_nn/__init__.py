"""
EchoLoc-NN: End-to-end ultrasonic localization toolkit

This package provides CNN-Transformer hybrid models for processing ultrasonic
chirp echoes to enable GPS-denied indoor positioning with commodity hardware.

Main Components:
- models: Neural network architectures for echo processing
- signal_processing: Chirp generation and echo enhancement
- hardware: Ultrasonic array interfaces and Arduino/RPi integration
- training: Model training pipelines and physics simulation
- inference: Real-time localization engine
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core imports for user convenience
from .models import EchoLocModel, CNNTransformerHybrid
from .inference import EchoLocator, InferenceConfig
from .hardware import UltrasonicArray, SensorConfig
from .signal_processing import ChirpGenerator, EchoProcessor
from .training import EchoLocTrainer, TrainingConfig, EchoSimulator

# Quick start functions
def create_locator(model_path=None, device="auto", **kwargs):
    """
    Create a ready-to-use EchoLocator instance.
    
    Args:
        model_path: Path to trained model (optional)
        device: Computing device ("auto", "cpu", "cuda")
        **kwargs: Additional configuration options
        
    Returns:
        Configured EchoLocator instance
    """
    config = InferenceConfig(model_path=model_path, device=device, **kwargs)
    return EchoLocator(config=config)

def create_square_array(spacing=0.1, port=None):
    """
    Create a 4-sensor square array configuration.
    
    Args:
        spacing: Sensor spacing in meters
        port: Serial port for hardware connection
        
    Returns:
        Configured UltrasonicArray instance
    """
    array = UltrasonicArray.create_square_array(spacing=spacing)
    if port:
        array.port = port
    return array

def quick_demo():
    """
    Run a quick demonstration of the system.
    
    Shows basic usage with synthetic data.
    """
    import numpy as np
    from .signal_processing import ChirpGenerator
    
    print("EchoLoc-NN Quick Demo")
    print("===================")
    
    # Create components
    print("1. Creating localization system...")
    locator = create_locator()
    array = create_square_array()
    chirp_gen = ChirpGenerator()
    
    # Generate synthetic echo data
    print("2. Generating synthetic echo data...")
    t, chirp = chirp_gen.generate_lfm_chirp(35000, 45000, 0.005)
    
    # Simulate echo (simplified)
    echo_data = np.random.randn(4, 2048) * 0.1  # 4 sensors, 2048 samples
    
    # Add delayed chirp echo
    delay_samples = 500
    for i in range(4):
        echo_data[i, delay_samples:delay_samples+len(chirp)] += chirp * (0.5 + i*0.1)
    
    # Localize
    print("3. Performing localization...")
    position, confidence = locator.locate(echo_data)
    
    print(f"4. Results:")
    print(f"   Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) meters")
    print(f"   Confidence: {confidence:.2%}")
    
    # Performance stats
    stats = locator.get_performance_stats()
    if not stats.get('no_data', False):
        print(f"   Inference time: {stats['avg_inference_time_ms']:.1f} ms")
    
    print("\n✓ Demo completed successfully!")
    print("  See examples/ directory for more detailed usage")

# Make key classes available at package level
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "EchoLocModel",
    "CNNTransformerHybrid",
    "EchoLocator",
    "InferenceConfig",
    "UltrasonicArray",
    "SensorConfig",
    "ChirpGenerator",
    "EchoProcessor",
    "EchoLocTrainer",
    "TrainingConfig",
    "EchoSimulator",
    "create_locator",
    "create_square_array",
    "quick_demo"
]