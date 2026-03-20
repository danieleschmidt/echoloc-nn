"""
echoloc-nn: Echolocation Neural Network
Pure numpy/scipy implementation of bat-inspired echolocation using
chirp signals, echo simulation, and transformer-based localization.
"""

from .chirp import ChirpSignal
from .simulator import EchoSimulator
from .encoder import ChirpEncoder
from .locator import TransformerLocator

__version__ = "0.1.0"
__all__ = ["ChirpSignal", "EchoSimulator", "ChirpEncoder", "TransformerLocator"]
