"""
Training Configuration for EchoLoc-NN
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    device: str = "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'validation_split': self.validation_split,
            'device': self.device
        }