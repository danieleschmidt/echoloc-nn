"""
Lightweight PyTorch fallback implementation for environments without PyTorch.

This module provides minimal tensor operations to make the system functional
when PyTorch is not available, primarily for testing and development.
"""

import numpy as np
from typing import Optional, Union, Tuple, Any


class TensorFallback:
    """Lightweight tensor class that mimics PyTorch tensor interface"""
    
    def __init__(self, data: np.ndarray, device: str = 'cpu'):
        self.data = np.array(data)
        self.device = device
        self.requires_grad = False
    
    def __getitem__(self, key):
        return TensorFallback(self.data[key], self.device)
    
    def __len__(self):
        return len(self.data)
    
    def numpy(self):
        return self.data
    
    def cpu(self):
        return TensorFallback(self.data, 'cpu')
    
    def cuda(self):
        return TensorFallback(self.data, 'cuda')
    
    def to(self, device: str):
        return TensorFallback(self.data, device)
    
    @property
    def shape(self):
        return self.data.shape
    
    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.data.shape
        return self.data.shape[dim]
    
    def unsqueeze(self, dim: int):
        return TensorFallback(np.expand_dims(self.data, dim), self.device)
    
    def squeeze(self, dim: Optional[int] = None):
        if dim is None:
            return TensorFallback(np.squeeze(self.data), self.device)
        return TensorFallback(np.squeeze(self.data, axis=dim), self.device)
    
    def float(self):
        return TensorFallback(self.data.astype(np.float32), self.device)
    
    def transpose(self, dim0, dim1):
        return TensorFallback(np.transpose(self.data, (dim0, dim1)), self.device)
    
    # Arithmetic operations
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return TensorFallback(self.data * other, self.device)
        elif isinstance(other, TensorFallback):
            return TensorFallback(self.data * other.data, self.device)
        return NotImplemented
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return TensorFallback(self.data + other, self.device)
        elif isinstance(other, TensorFallback):
            return TensorFallback(self.data + other.data, self.device)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return TensorFallback(self.data - other, self.device)
        elif isinstance(other, TensorFallback):
            return TensorFallback(self.data - other.data, self.device)
        return NotImplemented


class MockTorch:
    """Mock PyTorch module that provides basic tensor operations"""
    
    # Type alias for compatibility
    Tensor = TensorFallback
    
    # Dtype mock
    float = 'float32'
    long = 'int64'
    
    @staticmethod
    def tensor(data, device='cpu', dtype=None):
        return TensorFallback(np.array(data), device)
    
    @staticmethod
    def zeros(shape, device='cpu', dtype=None):
        return TensorFallback(np.zeros(shape), device)
    
    @staticmethod
    def ones(shape, device='cpu', dtype=None):
        return TensorFallback(np.ones(shape), device)
    
    @staticmethod
    def randn(*shape, device='cpu', dtype=None):
        return TensorFallback(np.random.randn(*shape), device)
    
    @staticmethod
    def rand(*shape, device='cpu', dtype=None):
        return TensorFallback(np.random.rand(*shape), device)
    
    @staticmethod
    def cat(tensors, dim=0):
        arrays = [t.data for t in tensors]
        return TensorFallback(np.concatenate(arrays, axis=dim))
    
    @staticmethod
    def stack(tensors, dim=0):
        arrays = [t.data for t in tensors]
        return TensorFallback(np.stack(arrays, axis=dim))
    
    @staticmethod
    def device(device_str):
        """Mock device function"""
        return device_str
    
    class cuda:
        """Mock cuda namespace"""
        @staticmethod
        def is_available():
            return False
    
    @staticmethod
    def arange(start, end=None, dtype=None):
        """Mock arange function"""
        if end is None:
            end = start
            start = 0
        return TensorFallback(np.arange(start, end, dtype=float if dtype == 'float' else int))
    
    @staticmethod
    def exp(x):
        """Mock exp function"""
        return TensorFallback(np.exp(x.data))
    
    @staticmethod
    def sin(x):
        """Mock sin function"""
        return TensorFallback(np.sin(x.data))
    
    @staticmethod
    def cos(x):
        """Mock cos function"""
        return TensorFallback(np.cos(x.data))
    
    @staticmethod
    def log(x):
        """Mock log function"""
        return TensorFallback(np.log(x.data))
    
    @staticmethod
    def get_num_threads():
        """Mock get_num_threads"""
        return 4
    
    @staticmethod
    def no_grad():
        """Mock no_grad context manager"""
        class NoGradContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoGradContext()
    
    class nn:
        class Module:
            def __init__(self):
                self.training = False
                self._parameters = {}
                self._modules = {}
                
            def train(self, mode=True):
                self.training = mode
                return self
                
            def eval(self):
                self.training = False
                return self
                
            def parameters(self):
                return []
                
            def to(self, device):
                return self
                
            def cuda(self):
                return self
                
            def cpu(self):
                return self
                
            def forward(self, x):
                return x
                
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)
        
        class Linear(Module):
            def __init__(self, in_features, out_features, bias=True):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = TensorFallback(np.random.randn(out_features, in_features) * 0.1)
                self.bias = TensorFallback(np.random.randn(out_features) * 0.1) if bias else None
            
            def forward(self, x):
                # Simple linear transformation
                output = np.dot(x.data, self.weight.data.T)
                if self.bias is not None:
                    output += self.bias.data
                return TensorFallback(output)
        
        class Conv1d(Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                
            def forward(self, x):
                # Simplified convolution - just return reshaped input
                return TensorFallback(x.data * 0.9)  # Simple mock operation
        
        class ReLU(Module):
            def forward(self, x):
                return TensorFallback(np.maximum(0, x.data))
        
        class Sigmoid(Module):
            def forward(self, x):
                return TensorFallback(1 / (1 + np.exp(-np.clip(x.data, -500, 500))))
        
        class Dropout(Module):
            def __init__(self, p=0.5):
                super().__init__()
                self.p = p
                
            def forward(self, x):
                if self.training:
                    mask = np.random.rand(*x.shape) > self.p
                    return TensorFallback(x.data * mask / (1 - self.p))
                return x
        
        class Sequential(Module):
            def __init__(self, *modules):
                super().__init__()
                self.modules = list(modules)
                
            def forward(self, x):
                for module in self.modules:
                    x = module(x)
                return x
        
        class Identity(Module):
            def forward(self, x):
                return x
        
        class BatchNorm1d(Module):
            def __init__(self, num_features):
                super().__init__()
                self.num_features = num_features
                
            def forward(self, x):
                # Mock batch norm - just return input
                return x
        
        class AdaptiveAvgPool1d(Module):
            def __init__(self, output_size):
                super().__init__()
                self.output_size = output_size
                
            def forward(self, x):
                # Mock adaptive pooling
                return TensorFallback(x.data)
        
        class TransformerEncoderLayer(Module):
            def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
                super().__init__()
                self.d_model = d_model
                self.nhead = nhead
                
            def forward(self, x):
                # Simplified transformer operation
                return TensorFallback(x.data * 0.95 + np.random.randn(*x.shape) * 0.01)
        
        class TransformerEncoder(Module):
            def __init__(self, encoder_layer, num_layers):
                super().__init__()
                self.layers = [encoder_layer for _ in range(num_layers)]
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        class MultiheadAttention(Module):
            def __init__(self, embed_dim, num_heads):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                
            def forward(self, query, key, value, attn_mask=None):
                # Simplified attention - return weighted input
                output = TensorFallback(query.data * 0.9)
                weights = TensorFallback(np.ones((query.shape[0], key.shape[0])) / key.shape[0])
                return output, weights


def get_torch():
    """Get PyTorch if available, otherwise return fallback implementation"""
    try:
        import torch
        return torch
    except ImportError:
        print("⚠️  PyTorch not available, using fallback implementation")
        return MockTorch()