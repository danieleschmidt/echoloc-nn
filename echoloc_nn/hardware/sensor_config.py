"""
Sensor Configuration for EchoLoc-NN
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class SensorConfig:
    """Configuration for ultrasonic sensor."""
    
    id: int
    position: Tuple[float, float, float]
    frequency: float
    beam_width: float = 30.0
    max_range: float = 5.0
    min_range: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'position': self.position,
            'frequency': self.frequency,
            'beam_width': self.beam_width,
            'max_range': self.max_range,
            'min_range': self.min_range
        }