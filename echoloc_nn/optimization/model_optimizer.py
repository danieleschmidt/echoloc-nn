"""
Model optimization for edge deployment and performance.
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import time
import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
from ..utils.logging_config import get_logger
from ..utils.exceptions import ModelError


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    method: str = "dynamic"  # dynamic, static, qat
    dtype: torch.dtype = torch.qint8
    calibration_dataset_size: int = 100
    per_channel: bool = True
    reduce_range: bool = False
    
    def __post_init__(self):
        if self.method not in ["dynamic", "static", "qat"]:
            raise ValueError(f"Unknown quantization method: {self.method}")


@dataclass 
class PruningConfig:
    """Configuration for model pruning."""
    
    sparsity: float = 0.5  # Fraction of weights to prune
    structured: bool = False  # Structured vs unstructured pruning
    global_pruning: bool = True  # Global vs layer-wise pruning
    pruning_schedule: str = "magnitude"  # magnitude, random, gradual
    
    def __post_init__(self):
        if not 0 < self.sparsity < 1:
            raise ValueError("Sparsity must be between 0 and 1")


@dataclass
class OptimizationResult:
    """Results from model optimization."""
    
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    original_inference_time_ms: float
    optimized_inference_time_ms: float
    speedup_ratio: float
    accuracy_degradation_percent: float
    memory_reduction_mb: float


class ModelOptimizer:
    """
    Comprehensive model optimization for deployment.
    
    Provides quantization, pruning, distillation, and
    hardware-specific optimizations.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.logger = get_logger('model_optimizer')
        
    def quantize_model(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration_data: Optional[torch.Tensor] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Quantize model for reduced size and faster inference.
        
        Args:
            model: PyTorch model to quantize
            config: Quantization configuration
            calibration_data: Calibration data for static quantization
            
        Returns:
            Tuple of (quantized_model, quantization_stats)
        """
        self.logger.info(f"Starting {config.method} quantization")
        start_time = time.time()
        
        # Prepare model
        model.eval()
        original_size = self._get_model_size(model)
        
        try:
            if config.method == "dynamic":
                quantized_model = self._dynamic_quantization(model, config)
            elif config.method == "static":
                if calibration_data is None:
                    raise ModelError("Calibration data required for static quantization")
                quantized_model = self._static_quantization(model, config, calibration_data)
            elif config.method == "qat":
                if calibration_data is None:
                    raise ModelError("Training data required for QAT")
                quantized_model = self._quantization_aware_training(model, config, calibration_data)
            else:
                raise ModelError(f"Unknown quantization method: {config.method}")
            
            # Measure results
            quantized_size = self._get_model_size(quantized_model)
            duration = time.time() - start_time
            
            stats = {
                'method': config.method,
                'dtype': str(config.dtype),
                'original_size_mb': original_size,
                'quantized_size_mb': quantized_size,
                'compression_ratio': original_size / quantized_size,
                'quantization_time_s': duration
            }
            
            self.logger.info(
                f"Quantization completed: {original_size:.1f}MB -> {quantized_size:.1f}MB "
                f"({stats['compression_ratio']:.2f}x compression)"
            )
            
            return quantized_model, stats
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            raise ModelError(f"Quantization failed: {e}")
    
    def _dynamic_quantization(self, model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Apply dynamic quantization."""
        # Define layers to quantize
        qconfig_dict = {
            nn.Linear: torch.quantization.default_dynamic_qconfig,
            nn.Conv1d: torch.quantization.default_dynamic_qconfig,
        }
        
        # Apply quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_dict,
            dtype=config.dtype,
            inplace=False
        )
        
        return quantized_model
    
    def _static_quantization(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration_data: torch.Tensor
    ) -> nn.Module:
        """Apply static quantization with calibration."""
        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibration
        self.logger.info("Running calibration for static quantization")
        model.eval()
        
        with torch.no_grad():
            # Use subset of calibration data
            n_samples = min(config.calibration_dataset_size, len(calibration_data))
            for i in range(n_samples):
                sample = calibration_data[i:i+1]
                _ = model(sample)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        return quantized_model
    
    def _quantization_aware_training(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        training_data: torch.Tensor
    ) -> nn.Module:
        """Apply quantization-aware training (simplified)."""
        # This is a simplified implementation
        # Full QAT would require training loop integration
        
        # Prepare model for QAT
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        
        # Simulate fine-tuning (normally would be full training)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
        for _ in range(10):  # Limited fine-tuning
            optimizer.zero_grad()
            outputs = model(training_data[:32])  # Small batch
            # Simplified loss (normally would be actual loss)
            loss = torch.mean(outputs[0])  # Dummy loss
            loss.backward()
            optimizer.step()
        
        # Convert to quantized model
        model.eval()
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        return quantized_model
    
    def prune_model(
        self,
        model: nn.Module,
        config: PruningConfig,
        validation_data: Optional[torch.Tensor] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Prune model to reduce size and complexity.
        
        Args:
            model: PyTorch model to prune
            config: Pruning configuration
            validation_data: Data for accuracy validation
            
        Returns:
            Tuple of (pruned_model, pruning_stats)
        """
        try:
            import torch.nn.utils.prune as prune
        except ImportError:
            raise ModelError("PyTorch pruning not available in this version")
        
        self.logger.info(f"Starting model pruning (sparsity={config.sparsity})")
        start_time = time.time()
        
        original_size = self._get_model_size(model)
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply pruning
        if config.structured:
            pruned_model = self._structured_pruning(model, config)
        else:
            pruned_model = self._unstructured_pruning(model, config)
        
        # Measure results
        pruned_size = self._get_model_size(pruned_model)
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        actual_sparsity = 1 - (pruned_params / original_params)
        duration = time.time() - start_time
        
        # Validate accuracy if data provided
        accuracy_drop = 0.0
        if validation_data is not None:
            accuracy_drop = self._measure_accuracy_drop(model, pruned_model, validation_data)
        
        stats = {
            'sparsity_target': config.sparsity,
            'sparsity_actual': actual_sparsity,
            'structured': config.structured,
            'original_size_mb': original_size,
            'pruned_size_mb': pruned_size,
            'compression_ratio': original_size / pruned_size,
            'accuracy_drop_percent': accuracy_drop,
            'pruning_time_s': duration
        }
        
        self.logger.info(
            f"Pruning completed: {original_size:.1f}MB -> {pruned_size:.1f}MB "
            f"({actual_sparsity:.1%} sparsity)"
        )
        
        return pruned_model, stats
    
    def _unstructured_pruning(self, model: nn.Module, config: PruningConfig) -> nn.Module:
        """Apply unstructured (magnitude-based) pruning."""
        import torch.nn.utils.prune as prune
        
        # Find all linear and conv layers
        modules_to_prune = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                modules_to_prune.append((module, 'weight'))
        
        if config.global_pruning:
            # Global magnitude pruning
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=config.sparsity
            )
        else:
            # Layer-wise pruning
            for module, param_name in modules_to_prune:
                prune.l1_unstructured(module, param_name, amount=config.sparsity)
        
        # Make pruning permanent
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def _structured_pruning(self, model: nn.Module, config: PruningConfig) -> nn.Module:
        """Apply structured pruning (remove entire channels/filters)."""
        import torch.nn.utils.prune as prune
        
        # Structured pruning is more complex and model-specific
        # This is a simplified implementation
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                # Prune output channels
                n_channels = module.out_channels
                n_prune = int(n_channels * config.sparsity)
                
                prune.random_structured(
                    module,
                    name='weight',
                    amount=n_prune,
                    dim=0  # Output channels
                )
                prune.remove(module, 'weight')
        
        return model
    
    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        optimization_level: str = "default"
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Optimize model specifically for inference.
        
        Args:
            model: PyTorch model
            sample_input: Sample input for optimization
            optimization_level: "conservative", "default", "aggressive"
            
        Returns:
            Tuple of (optimized_model, optimization_stats)
        """
        self.logger.info(f"Optimizing model for inference ({optimization_level})")
        start_time = time.time()
        
        original_time = self._benchmark_inference(model, sample_input)
        
        # Apply optimizations based on level
        optimized_model = model
        
        if optimization_level in ["default", "aggressive"]:
            # Fuse operations
            optimized_model = self._fuse_operations(optimized_model)
            
        if optimization_level == "aggressive":
            # TorchScript compilation
            optimized_model = self._torchscript_optimize(optimized_model, sample_input)
        
        # Benchmark optimized model
        optimized_time = self._benchmark_inference(optimized_model, sample_input)
        
        stats = {
            'optimization_level': optimization_level,
            'original_inference_ms': original_time,
            'optimized_inference_ms': optimized_time,
            'speedup_ratio': original_time / optimized_time,
            'optimization_time_s': time.time() - start_time
        }
        
        self.logger.info(
            f"Inference optimization completed: {original_time:.1f}ms -> {optimized_time:.1f}ms "
            f"({stats['speedup_ratio']:.2f}x speedup)"
        )
        
        return optimized_model, stats
    
    def _fuse_operations(self, model: nn.Module) -> nn.Module:
        """Fuse conv-bn, linear-relu, etc. operations."""
        # Conv + BatchNorm fusion
        fused_model = torch.quantization.fuse_modules(
            model,
            [['conv', 'bn'], ['conv', 'bn', 'relu']],
            inplace=False
        )
        
        return fused_model
    
    def _torchscript_optimize(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Optimize model using TorchScript."""
        model.eval()
        
        # Trace model
        try:
            traced_model = torch.jit.trace(model, sample_input)
            
            # Optimize traced model
            optimized_model = torch.jit.optimize_for_inference(traced_model)
            
            return optimized_model
        except Exception as e:
            self.logger.warning(f"TorchScript optimization failed: {e}")
            return model
    
    def comprehensive_optimization(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        calibration_data: Optional[torch.Tensor] = None,
        target_device: str = "cpu",
        accuracy_threshold: float = 0.05  # 5% max accuracy drop
    ) -> OptimizationResult:
        """
        Apply comprehensive optimization pipeline.
        
        Args:
            model: PyTorch model to optimize
            sample_input: Sample input for benchmarking
            calibration_data: Data for quantization calibration
            target_device: Target deployment device
            accuracy_threshold: Maximum allowed accuracy degradation
            
        Returns:
            Comprehensive optimization results
        """
        self.logger.info("Starting comprehensive model optimization")
        
        # Baseline measurements
        original_size = self._get_model_size(model)
        original_time = self._benchmark_inference(model, sample_input)
        
        optimized_model = model
        total_accuracy_drop = 0.0
        
        # Step 1: Inference optimizations (fusing, etc.)
        inference_optimized, inference_stats = self.optimize_for_inference(
            optimized_model, sample_input, "default"
        )
        optimized_model = inference_optimized
        
        # Step 2: Quantization (if beneficial for target device)
        if target_device == "cpu":
            quant_config = QuantizationConfig(method="dynamic")
            quantized_model, quant_stats = self.quantize_model(
                optimized_model, quant_config, calibration_data
            )
            
            # Check if quantization improves performance
            quantized_time = self._benchmark_inference(quantized_model, sample_input)
            if quantized_time < self._benchmark_inference(optimized_model, sample_input):
                optimized_model = quantized_model
                self.logger.info("Applied quantization")
            else:
                self.logger.info("Skipped quantization (no performance benefit)")
        
        # Step 3: Pruning (if accuracy allows)
        if calibration_data is not None:
            prune_config = PruningConfig(sparsity=0.3, structured=False)
            pruned_model, prune_stats = self.prune_model(
                optimized_model, prune_config, calibration_data
            )
            
            if prune_stats['accuracy_drop_percent'] < accuracy_threshold * 100:
                optimized_model = pruned_model
                total_accuracy_drop += prune_stats['accuracy_drop_percent']
                self.logger.info("Applied pruning")
            else:
                self.logger.info("Skipped pruning (accuracy threshold exceeded)")
        
        # Final measurements
        final_size = self._get_model_size(optimized_model)
        final_time = self._benchmark_inference(optimized_model, sample_input)
        
        result = OptimizationResult(
            original_size_mb=original_size,
            optimized_size_mb=final_size,
            compression_ratio=original_size / final_size,
            original_inference_time_ms=original_time,
            optimized_inference_time_ms=final_time,
            speedup_ratio=original_time / final_time,
            accuracy_degradation_percent=total_accuracy_drop,
            memory_reduction_mb=original_size - final_size
        )
        
        self.logger.info(
            f"Comprehensive optimization completed:\n"
            f"  Size: {original_size:.1f}MB -> {final_size:.1f}MB ({result.compression_ratio:.2f}x)\n"
            f"  Speed: {original_time:.1f}ms -> {final_time:.1f}ms ({result.speedup_ratio:.2f}x)\n"
            f"  Accuracy drop: {total_accuracy_drop:.2f}%"
        )
        
        return result
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _benchmark_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        n_runs: int = 100
    ) -> float:
        """Benchmark model inference time."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(sample_input)
        
        avg_time_ms = (time.time() - start_time) * 1000 / n_runs
        return avg_time_ms
    
    def _measure_accuracy_drop(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        validation_data: torch.Tensor
    ) -> float:
        """Measure accuracy degradation (simplified)."""
        # This is a simplified accuracy measurement
        # Real implementation would use proper validation metrics
        
        original_model.eval()
        optimized_model.eval()
        
        with torch.no_grad():
            orig_outputs = original_model(validation_data)
            opt_outputs = optimized_model(validation_data)
            
            # Simple MSE-based accuracy metric
            if isinstance(orig_outputs, tuple):
                orig_outputs = orig_outputs[0]
            if isinstance(opt_outputs, tuple):
                opt_outputs = opt_outputs[0]
            
            mse = torch.mean((orig_outputs - opt_outputs) ** 2)
            accuracy_drop = float(mse.item()) * 100  # Convert to percentage
            
        return accuracy_drop