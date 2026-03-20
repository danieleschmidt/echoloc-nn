"""
echoloc — Neural echolocation in pure Python/NumPy.

Modules:
    signals   : ChirpSignal — frequency-swept pulse generation
    simulator : EchoSimulator — physics-based echo simulation
    encoder   : ChirpEncoder — cosine-modulated Gaussian filter bank
    locator   : TransformerLocator — self-attention-based localization
    demo      : demo() — end-to-end demonstration
"""

from .signals import ChirpSignal
from .simulator import EchoSimulator
from .encoder import ChirpEncoder
from .locator import TransformerLocator

__all__ = ["ChirpSignal", "EchoSimulator", "ChirpEncoder", "TransformerLocator"]
__version__ = "0.1.0"
