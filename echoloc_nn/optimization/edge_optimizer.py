"""Edge deployment optimization for EchoLoc models."""

import torch
import torch.nn as nn
from torch.jit import script
from typing import Dict, Any, Optional, List, Tuple
import time
import numpy as np
from pathlib import Path


class EdgeOptimizer:
    """Optimize models for edge deployment with minimal latency."""
    
    def __init__(self, target_device: str = 'cpu', memory_limit_mb: int = 512):
        self.target_device = target_device
        self.memory_limit_mb = memory_limit_mb
        self.optimization_history = []
    
    def quantize_model(self, model: nn.Module, method: str = 'dynamic') -> nn.Module:
        """Apply quantization for model compression."""
        model.eval()
        
        if method == 'dynamic':
            # Dynamic quantization - good for RNN/Transformer layers
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv1d}, 
                dtype=torch.qint8
            )
        elif method == 'static':
            # Static quantization - requires calibration data
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            prepared_model = torch.quantization.prepare(model)
            # Would need calibration data here
            quantized_model = torch.quantization.convert(prepared_model)
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        return quantized_model
    
    def prune_model(self, model: nn.Module, sparsity: float = 0.3) -> nn.Module:
        """Apply structured pruning to reduce model size."""
        import torch.nn.utils.prune as prune
        
        # Identify layers to prune
        layers_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                layers_to_prune.append((module, 'weight'))
        
        # Apply magnitude-based pruning
        prune.global_unstructured(
            layers_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity
        )
        
        # Make pruning permanent
        for module, param_name in layers_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Apply inference-specific optimizations."""
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Fuse batch norm and conv layers where possible
        try:
            model = torch.jit.optimize_for_inference(model)
        except:
            # Fallback if optimization fails
            pass
        
        return model
    
    def compile_to_torchscript(self, model: nn.Module, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """Compile model to TorchScript for deployment."""
        model.eval()
        
        # Trace the model
        try:
            traced_model = torch.jit.trace(model, example_input)
            # Optimize the traced model
            traced_model = torch.jit.optimize_for_inference(traced_model)
            return traced_model
        except Exception as e:
            print(f"Warning: TorchScript compilation failed: {e}")
            return model
    
    def benchmark_model(self, model: nn.Module, input_shape: Tuple[int, ...], num_runs: int = 100) -> Dict[str, float]:
        """Benchmark model performance."""
        device = torch.device(self.target_device)
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            'mean_latency_ms': float(np.mean(times)),
            'median_latency_ms': float(np.median(times)),
            'p95_latency_ms': float(np.percentile(times, 95)),
            'min_latency_ms': float(np.min(times)),
            'max_latency_ms': float(np.max(times)),
            'std_latency_ms': float(np.std(times))
        }
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Calculate model memory footprint."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'param_size_mb': param_size / (1024 * 1024),
            'buffer_size_mb': buffer_size / (1024 * 1024),
            'total_size_mb': total_size / (1024 * 1024),
            'param_count': sum(p.numel() for p in model.parameters())
        }
    
    def optimize_pipeline(self, model: nn.Module, input_shape: Tuple[int, ...], 
                         target_latency_ms: float = 50.0) -> Dict[str, Any]:
        """Complete optimization pipeline for edge deployment."""
        original_size = self.get_model_size(model)
        original_benchmark = self.benchmark_model(model, input_shape)
        
        optimizations_applied = []
        current_model = model
        
        # Step 1: Quantization
        if original_size['total_size_mb'] > self.memory_limit_mb / 2:
            current_model = self.quantize_model(current_model, 'dynamic')
            optimizations_applied.append('quantization')
        
        # Step 2: Pruning if still too large or slow
        current_benchmark = self.benchmark_model(current_model, input_shape)
        if (current_benchmark['mean_latency_ms'] > target_latency_ms or 
            self.get_model_size(current_model)['total_size_mb'] > self.memory_limit_mb):
            current_model = self.prune_model(current_model, sparsity=0.3)
            optimizations_applied.append('pruning')
        
        # Step 3: Inference optimization
        current_model = self.optimize_for_inference(current_model)
        optimizations_applied.append('inference_optimization')
        
        # Step 4: TorchScript compilation
        dummy_input = torch.randn(input_shape)
        try:
            current_model = self.compile_to_torchscript(current_model, dummy_input)
            optimizations_applied.append('torchscript')
        except:
            pass
        
        # Final benchmarks
        final_size = self.get_model_size(current_model)
        final_benchmark = self.benchmark_model(current_model, input_shape)
        
        optimization_result = {
            'optimized_model': current_model,
            'optimizations_applied': optimizations_applied,
            'size_reduction': {
                'original_mb': original_size['total_size_mb'],
                'final_mb': final_size['total_size_mb'],
                'reduction_ratio': final_size['total_size_mb'] / original_size['total_size_mb'],
                'saved_mb': original_size['total_size_mb'] - final_size['total_size_mb']
            },
            'performance_improvement': {
                'original_latency_ms': original_benchmark['mean_latency_ms'],
                'final_latency_ms': final_benchmark['mean_latency_ms'],
                'speedup_ratio': original_benchmark['mean_latency_ms'] / final_benchmark['mean_latency_ms'],
                'latency_reduction_ms': original_benchmark['mean_latency_ms'] - final_benchmark['mean_latency_ms']
            },
            'meets_target': final_benchmark['mean_latency_ms'] <= target_latency_ms,
            'meets_memory_limit': final_size['total_size_mb'] <= self.memory_limit_mb
        }
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def export_for_deployment(self, model: nn.Module, export_path: Path, 
                            input_shape: Tuple[int, ...]) -> Dict[str, str]:
        """Export optimized model for production deployment."""
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        exports = {}
        
        # Export TorchScript
        try:
            dummy_input = torch.randn(input_shape)
            traced_model = self.compile_to_torchscript(model, dummy_input)
            torchscript_path = export_path / 'model.pt'
            traced_model.save(str(torchscript_path))
            exports['torchscript'] = str(torchscript_path)
        except Exception as e:
            print(f"TorchScript export failed: {e}")
        
        # Export ONNX if available
        try:
            import torch.onnx
            onnx_path = export_path / 'model.onnx'
            dummy_input = torch.randn(input_shape)
            torch.onnx.export(
                model, dummy_input, str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['echo_input'],
                output_names=['position', 'confidence']
            )
            exports['onnx'] = str(onnx_path)
        except Exception as e:
            print(f"ONNX export failed: {e}")
        
        # Save model state dict as fallback
        state_dict_path = export_path / 'model_state_dict.pth'
        torch.save(model.state_dict(), state_dict_path)
        exports['state_dict'] = str(state_dict_path)
        
        # Save model info
        info_path = export_path / 'model_info.json'
        model_info = {
            'input_shape': list(input_shape),
            'model_size': self.get_model_size(model),
            'optimization_history': self.optimization_history,
            'target_device': self.target_device
        }
        
        import json
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        exports['info'] = str(info_path)
        
        return exports