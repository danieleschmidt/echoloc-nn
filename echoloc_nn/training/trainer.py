"""
Training pipeline for EchoLoc-NN models.
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None
import logging

from ..models.base import EchoLocBaseModel
from .losses import CombinedLoss
from .data_loader import EchoDataModule


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Model parameters
    model_class: str = "EchoLocModel"
    model_kwargs: Dict[str, Any] = None
    
    # Training parameters
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    gradient_clip: float = 1.0
    
    # Loss function
    position_weight: float = 1.0
    confidence_weight: float = 0.1
    
    # Validation
    val_check_interval: int = 1
    early_stopping_patience: int = 10
    
    # Logging and checkpoints
    log_every_n_steps: int = 100
    save_top_k: int = 3
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}


class EchoLocTrainer:
    """
    Training pipeline for EchoLoc-NN models.
    
    Handles model training, validation, checkpointing, and logging
    with support for various optimization strategies and data augmentation.
    """
    
    def __init__(
        self,
        model: EchoLocBaseModel,
        config: TrainingConfig,
        device: str = "auto"
    ):
        self.model = model
        self.config = config
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Initialize components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_loss_function()
        
        # Logging
        self.logger = self._setup_logging()
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.max_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif self.config.scheduler.lower() == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function."""
        return CombinedLoss(
            position_weight=self.config.position_weight,
            confidence_weight=self.config.confidence_weight
        )
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('EchoLocTrainer')
        logger.setLevel(logging.INFO)
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def fit(
        self,
        data_module: EchoDataModule,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Train the model.
        
        Args:
            data_module: Data module with train/val dataloaders
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
            
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        self.logger.info(f"Starting training for {self.config.max_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            if epoch % self.config.val_check_interval == 0:
                val_metrics = self._validate_epoch(val_loader)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                
                # Early stopping check
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.early_stopping_counter = 0
                    self._save_checkpoint(f"best_model.pth", is_best=True)
                else:
                    self.early_stopping_counter += 1
                    
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Log validation metrics
                self._log_metrics(val_metrics, epoch)
                
            # Log training metrics
            self._log_metrics(train_metrics, epoch)
            
            # Save checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(f"epoch_{epoch}.pth")
                
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model
        self._save_checkpoint("final_model.pth")
        self.writer.close()
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_position_loss = 0.0
        total_confidence_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            echo_data = batch['echo_data'].to(self.device)
            positions = batch['position'].to(self.device)
            sensor_positions = batch.get('sensor_positions')
            if sensor_positions is not None:
                sensor_positions = sensor_positions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_positions, pred_confidence = self.model(echo_data, sensor_positions)
            
            # Compute loss
            loss_dict = self.criterion(
                pred_positions, pred_confidence, positions
            )
            
            total_loss_val = loss_dict['total_loss']
            
            # Backward pass
            total_loss_val.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += total_loss_val.item()
            total_position_loss += loss_dict['position_loss'].item()
            total_confidence_loss += loss_dict['confidence_loss'].item()
            
            self.global_step += 1
            
            # Log batch metrics
            if batch_idx % self.config.log_every_n_steps == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {total_loss_val.item():.6f}"
                )
                
                self.writer.add_scalar(
                    'train/batch_loss', total_loss_val.item(), self.global_step
                )
        
        return {
            'train_loss': total_loss / num_batches,
            'train_position_loss': total_position_loss / num_batches,
            'train_confidence_loss': total_confidence_loss / num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_position_loss = 0.0
        total_confidence_loss = 0.0
        position_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                echo_data = batch['echo_data'].to(self.device)
                positions = batch['position'].to(self.device)
                sensor_positions = batch.get('sensor_positions')
                if sensor_positions is not None:
                    sensor_positions = sensor_positions.to(self.device)
                
                # Forward pass
                pred_positions, pred_confidence = self.model(echo_data, sensor_positions)
                
                # Compute loss
                loss_dict = self.criterion(
                    pred_positions, pred_confidence, positions
                )
                
                total_loss += loss_dict['total_loss'].item()
                total_position_loss += loss_dict['position_loss'].item()
                total_confidence_loss += loss_dict['confidence_loss'].item()
                
                # Compute position errors
                pos_errors = torch.norm(pred_positions - positions, dim=1)
                position_errors.extend(pos_errors.cpu().numpy())
        
        num_batches = len(val_loader)
        position_errors = np.array(position_errors)
        
        return {
            'val_loss': total_loss / num_batches,
            'val_position_loss': total_position_loss / num_batches,
            'val_confidence_loss': total_confidence_loss / num_batches,
            'val_mean_error': np.mean(position_errors),
            'val_median_error': np.median(position_errors),
            'val_p95_error': np.percentile(position_errors, 95)
        }
    
    def _log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log metrics to tensorboard and console."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)
            
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', current_lr, epoch)
        
        # Console logging
        metric_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metric_str}")
    
    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            self.logger.info(f"Saved best model checkpoint: {filepath}")
        else:
            self.logger.info(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        
        position_errors = []
        confidence_scores = []
        
        with torch.no_grad():
            for batch in test_loader:
                echo_data = batch['echo_data'].to(self.device)
                positions = batch['position'].to(self.device)
                sensor_positions = batch.get('sensor_positions')
                if sensor_positions is not None:
                    sensor_positions = sensor_positions.to(self.device)
                
                # Forward pass
                pred_positions, pred_confidence = self.model(echo_data, sensor_positions)
                
                # Compute errors
                pos_errors = torch.norm(pred_positions - positions, dim=1)
                position_errors.extend(pos_errors.cpu().numpy())
                confidence_scores.extend(pred_confidence.cpu().numpy().flatten())
        
        position_errors = np.array(position_errors)
        confidence_scores = np.array(confidence_scores)
        
        # Convert errors to centimeters
        position_errors_cm = position_errors * 100
        
        test_metrics = {
            'test_mean_error_cm': np.mean(position_errors_cm),
            'test_median_error_cm': np.median(position_errors_cm),
            'test_p95_error_cm': np.percentile(position_errors_cm, 95),
            'test_max_error_cm': np.max(position_errors_cm),
            'test_accuracy_5cm': np.mean(position_errors_cm < 5.0) * 100,
            'test_accuracy_10cm': np.mean(position_errors_cm < 10.0) * 100,
            'test_mean_confidence': np.mean(confidence_scores)
        }
        
        # Log test results
        self.logger.info("Test Results:")
        for key, value in test_metrics.items():
            if 'accuracy' in key or 'confidence' in key:
                self.logger.info(f"  {key}: {value:.2f}%")
            else:
                self.logger.info(f"  {key}: {value:.2f}")
        
        return test_metrics
    
    def predict(
        self,
        echo_data: np.ndarray,
        sensor_positions: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Make prediction on single echo sample.
        
        Args:
            echo_data: Echo data (n_sensors, n_samples)
            sensor_positions: Sensor positions (n_sensors, 2)
            
        Returns:
            Tuple of (position, confidence)
        """
        return self.model.predict_position(echo_data, sensor_positions)