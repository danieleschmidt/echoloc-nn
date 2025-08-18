"""
Hardware interfaces for ultrasonic sensor arrays.

This module provides abstraction layers for different ultrasonic
sensor configurations and microcontroller interfaces.
"""

from .ultrasonic_array import UltrasonicArray, SensorConfig

# Optional hardware interfaces - available when dependencies installed
try:
    from .arduino_interface import ArduinoInterface, ArduinoCommands
    from .raspberry_pi import RaspberryPiDeployment, EdgeOptimizer
    HARDWARE_AVAILABLE = True
except ImportError:
    # Mock classes for software-only testing
    ArduinoInterface = None
    ArduinoCommands = None
    RaspberryPiDeployment = None
    EdgeOptimizer = None
    HARDWARE_AVAILABLE = False

__all__ = [
    "UltrasonicArray",
    "SensorConfig",
    "ArduinoInterface", 
    "ArduinoCommands",
    "RaspberryPiDeployment",
    "EdgeOptimizer",
    "HARDWARE_AVAILABLE"
]