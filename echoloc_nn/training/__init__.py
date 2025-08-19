"""
Training utilities for EchoLoc-NN models.

This module provides training loops, data augmentation, loss functions,
and optimization strategies for ultrasonic localization models.
"""

try:
    from .trainer import EchoLocTrainer, TrainingConfig
    from .data_loader import EchoDataset, EchoDataModule
    from .simulation import EchoSimulator, RoomAcoustics
    from .losses import PositionLoss, ConfidenceLoss, CombinedLoss
    from .augmentation import EchoAugmentation, NoiseAugmentation
except ImportError:
    # Training modules not available without PyTorch
    EchoLocTrainer = None
    TrainingConfig = None
    EchoDataset = None
    EchoDataModule = None
    EchoSimulator = None
    RoomAcoustics = None
    PositionLoss = None
    ConfidenceLoss = None
    CombinedLoss = None
    EchoAugmentation = None
    NoiseAugmentation = None

__all__ = [
    "EchoLocTrainer",
    "TrainingConfig",
    "EchoDataset",
    "EchoDataModule",
    "EchoSimulator",
    "RoomAcoustics",
    "PositionLoss",
    "ConfidenceLoss", 
    "CombinedLoss",
    "EchoAugmentation",
    "NoiseAugmentation"
]