"""
Data Augmentation for EchoLoc-NN
"""

import numpy as np
from typing import Dict, Any

class EchoAugmentation:
    """Echo data augmentation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def augment(self, echo_data: np.ndarray) -> np.ndarray:
        """Apply augmentation to echo data."""
        return echo_data

class NoiseAugmentation:
    """Noise augmentation for echo data."""
    
    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
        
    def augment(self, echo_data: np.ndarray) -> np.ndarray:
        """Add noise to echo data."""
        noise = np.random.randn(*echo_data.shape) * self.noise_level
        return echo_data + noise