"""
Hardware interfaces for ultrasonic sensor arrays.

This module provides abstraction layers for different ultrasonic
sensor configurations and microcontroller interfaces.
"""

from .ultrasonic_array import UltrasonicArray, SensorConfig
from .arduino_interface import ArduinoInterface, ArduinoCommands
from .raspberry_pi import RaspberryPiDeployment, EdgeOptimizer

__all__ = [
    "UltrasonicArray",
    "SensorConfig",
    "ArduinoInterface", 
    "ArduinoCommands",
    "RaspberryPiDeployment",
    "EdgeOptimizer"
]