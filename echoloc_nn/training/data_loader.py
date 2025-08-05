"""
Data Loading for EchoLoc-NN Training
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any
import numpy as np

class EchoDataset(Dataset):
    """Dataset for echo data."""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        # Placeholder data
        self.echo_data = [np.random.randn(4, 2048) for _ in range(1000)]
        self.positions = [np.random.randn(3) * 5 for _ in range(1000)]
        
    def __len__(self) -> int:
        return len(self.echo_data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        echo = torch.FloatTensor(self.echo_data[idx])
        position = torch.FloatTensor(self.positions[idx])
        return echo, position

class EchoDataModule:
    """Data module for echo data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def setup(self):
        """Setup data loaders."""
        pass
        
    def train_dataloader(self):
        """Get training data loader."""
        dataset = EchoDataset()
        return torch.utils.data.DataLoader(dataset, batch_size=32)
        
    def val_dataloader(self):
        """Get validation data loader."""
        dataset = EchoDataset()
        return torch.utils.data.DataLoader(dataset, batch_size=32)