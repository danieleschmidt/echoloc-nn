"""
Configuration for EchoLoc-NN Inference Engine
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    
    model_path: Optional[str] = None
    device: str = "auto"
    batch_size: int = 1
    optimization_level: int = 1
    use_quantization: bool = False
    max_latency_ms: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'batch_size': self.batch_size,
            'optimization_level': self.optimization_level,
            'use_quantization': self.use_quantization,
            'max_latency_ms': self.max_latency_ms
        }