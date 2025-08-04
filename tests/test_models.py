"""
Comprehensive tests for EchoLoc-NN models.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from echoloc_nn.models import (
    EchoLocModel,
    CNNTransformerHybrid,
    CNNEncoder,
    TransformerDecoder
)
from echoloc_nn.models.base import EchoLocBaseModel
from echoloc_nn.utils.exceptions import ModelError


class TestEchoLocBaseModel:
    """Test base model functionality."""
    
    def test_predict_position_interface(self):
        """Test predict_position interface."""
        # Create simple test model
        class TestModel(EchoLocBaseModel):
            def forward(self, x, sensor_positions=None):
                batch_size = x.shape[0]
                position = torch.zeros(batch_size, 3)
                confidence = torch.ones(batch_size, 1) * 0.8
                return position, confidence
        
        model = TestModel()
        echo_data = np.random.randn(4, 2048)
        
        position, confidence = model.predict_position(echo_data)
        
        assert position.shape == (3,)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_get_model_info(self):
        """Test model info extraction."""
        class TestModel(EchoLocBaseModel):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x, sensor_positions=None):
                return torch.zeros(1, 3), torch.ones(1, 1)
        
        model = TestModel()
        info = model.get_model_info()
        
        assert 'model_class' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['total_parameters'] > 0
    
    def test_set_physics_params(self):
        """Test physics parameter setting."""
        class TestModel(EchoLocBaseModel):
            def forward(self, x, sensor_positions=None):
                return torch.zeros(1, 3), torch.ones(1, 1)
        
        model = TestModel()
        model.set_physics_params(speed_of_sound=350.0, sample_rate=192000)
        
        assert model.speed_of_sound == 350.0
        assert model.sample_rate == 192000


class TestCNNEncoder:
    """Test CNN encoder component."""
    
    def test_initialization(self):
        """Test CNN encoder initialization."""
        encoder = CNNEncoder(
            n_sensors=4,
            input_length=2048,
            channels=[32, 64, 128],
            use_multipath=False
        )
        
        assert encoder.n_sensors == 4
        assert encoder.input_length == 2048
        assert len(encoder.encoder_blocks) == 2  # len(channels) - 1
    
    def test_forward_pass(self):
        """Test CNN encoder forward pass."""
        encoder = CNNEncoder(n_sensors=4, input_length=2048)
        
        # Test input
        x = torch.randn(2, 4, 2048)  # batch=2, sensors=4, samples=2048
        
        output = encoder(x)
        
        assert output.dim() == 3  # (batch, channels, seq_len)
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == encoder.channels[-1]  # output channels
    
    def test_multipath_convolution(self):
        """Test multipath convolution."""
        from echoloc_nn.models.cnn_encoder import MultiPathConv
        
        conv = MultiPathConv(in_channels=32, out_channels=64, max_paths=3)
        
        x = torch.randn(2, 32, 1000)
        output = conv(x)
        
        assert output.shape[0] == 2  # batch
        assert output.shape[1] == 64  # output channels
        assert output.shape[2] <= x.shape[2]  # sequence length (may be reduced)
    
    def test_conv_block(self):
        """Test individual convolution block."""
        from echoloc_nn.models.cnn_encoder import ConvBlock
        
        block = ConvBlock(in_channels=16, out_channels=32, kernel_size=5)
        
        x = torch.randn(1, 16, 1000)
        output = block(x)
        
        assert output.shape[1] == 32  # output channels
        assert torch.all(output >= 0)  # ReLU activation


class TestTransformerDecoder:
    """Test Transformer decoder component."""
    
    def test_initialization(self):
        """Test transformer decoder initialization."""
        decoder = TransformerDecoder(
            d_model=256,
            n_heads=8,
            n_layers=4
        )
        
        assert decoder.d_model == 256
        assert decoder.n_layers == 4
        assert len(decoder.layers) == 4
    
    def test_forward_pass(self):
        """Test transformer decoder forward pass."""
        decoder = TransformerDecoder(d_model=256, n_heads=8, n_layers=2)
        
        x = torch.randn(2, 100, 256)  # batch=2, seq_len=100, d_model=256
        output = decoder(x)
        
        assert output.shape == x.shape  # Same shape as input
    
    def test_positional_encoding(self):
        """Test echo-specific positional encoding."""
        from echoloc_nn.models.transformer_decoder import EchoPositionalEncoding
        
        pos_enc = EchoPositionalEncoding(d_model=256, max_len=1000)
        
        x = torch.randn(2, 100, 256)
        encoded = pos_enc(x)
        
        assert encoded.shape == x.shape
        assert not torch.equal(encoded, x)  # Should be different after encoding
    
    def test_echo_attention(self):
        """Test echo-aware attention mechanism."""
        from echoloc_nn.models.transformer_decoder import EchoAttention
        
        attention = EchoAttention(dim=256, n_heads=8)
        
        x = torch.randn(2, 100, 256)
        output, weights = attention(x)
        
        assert output.shape == x.shape
        assert weights.shape == (2, 100, 100)  # (batch, seq_len, seq_len)


class TestCNNTransformerHybrid:
    """Test CNN-Transformer hybrid model."""
    
    def test_initialization(self):
        """Test hybrid model initialization."""
        model = CNNTransformerHybrid(
            n_sensors=4,
            chirp_length=2048,
            cnn_channels=[32, 64, 128],
            transformer_dim=256
        )
        
        assert model.n_sensors == 4
        assert model.chirp_length == 2048
        assert model.transformer_dim == 256
    
    def test_forward_pass(self):
        """Test full forward pass."""
        model = CNNTransformerHybrid(
            n_sensors=4,
            chirp_length=2048,
            transformer_dim=256
        )
        
        x = torch.randn(2, 4, 2048)  # batch=2, sensors=4, samples=2048
        sensor_positions = torch.randn(2, 4, 2)  # sensor positions
        
        position, confidence = model(x, sensor_positions)
        
        assert position.shape == (2, 3)  # (batch, xyz)
        assert confidence.shape == (2, 1)  # (batch, confidence)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)
    
    def test_forward_without_sensor_positions(self):
        """Test forward pass without sensor positions."""
        model = CNNTransformerHybrid(n_sensors=4, chirp_length=2048)
        
        x = torch.randn(1, 4, 2048)
        position, confidence = model(x)
        
        assert position.shape == (1, 3)
        assert confidence.shape == (1, 1)
    
    def test_weight_initialization(self):
        """Test proper weight initialization."""
        model = CNNTransformerHybrid(n_sensors=4, chirp_length=2048)
        
        # Check that weights are not all zeros
        has_nonzero_weights = False
        for param in model.parameters():
            if torch.any(param != 0):
                has_nonzero_weights = True
                break
        
        assert has_nonzero_weights


class TestEchoLocModel:
    """Test main EchoLoc model class."""
    
    def test_model_sizes(self):
        """Test different model size configurations."""
        sizes = ["tiny", "base", "large"]
        
        for size in sizes:
            model = EchoLocModel(n_sensors=4, model_size=size)
            assert model.model_size == size
            
            # Test forward pass
            x = torch.randn(1, 4, 2048)
            position, confidence = model(x)
            
            assert position.shape == (1, 3)
            assert confidence.shape == (1, 1)
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        # Create and save model
        original_model = EchoLocModel(n_sensors=4, model_size="tiny")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pt")
            original_model.save_model(model_path)
            
            # Load model
            loaded_model = EchoLocModel.load_model(model_path)
            
            assert loaded_model.n_sensors == original_model.n_sensors
            assert loaded_model.model_size == original_model.model_size
            
            # Test that loaded model produces same output
            x = torch.randn(1, 4, 2048)
            
            original_model.eval()
            loaded_model.eval()
            
            with torch.no_grad():
                orig_pos, orig_conf = original_model(x)
                load_pos, load_conf = loaded_model(x)
                
                assert torch.allclose(orig_pos, load_pos, atol=1e-6)
                assert torch.allclose(orig_conf, load_conf, atol=1e-6)
    
    def test_compute_complexity(self):
        """Test computational complexity estimation."""
        model = EchoLocModel(n_sensors=4, model_size="base")
        complexity = model.estimate_compute_complexity()
        
        assert 'total_parameters' in complexity
        assert 'estimated_flops' in complexity
        assert complexity['total_parameters'] > 0
        assert complexity['estimated_flops'] > 0
    
    def test_predict_position_method(self):
        """Test predict_position convenience method."""
        model = EchoLocModel(n_sensors=4, model_size="tiny")
        
        echo_data = np.random.randn(4, 2048)
        sensor_positions = np.random.randn(4, 2)
        
        position, confidence = model.predict_position(echo_data, sensor_positions)
        
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


class TestModelEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_input_dimensions(self):
        """Test handling of invalid input dimensions."""
        model = EchoLocModel(n_sensors=4)
        
        # Wrong number of sensors
        with pytest.raises((RuntimeError, ValueError)):
            x = torch.randn(1, 3, 2048)  # 3 sensors instead of 4
            model(x)
        
        # Wrong input dimensions
        with pytest.raises((RuntimeError, ValueError)):
            x = torch.randn(4, 2048)  # Missing batch dimension
            model(x)
    
    def test_zero_input(self):
        """Test handling of zero input."""
        model = EchoLocModel(n_sensors=4)
        
        x = torch.zeros(1, 4, 2048)
        position, confidence = model(x)
        
        # Should still produce valid output
        assert position.shape == (1, 3)
        assert confidence.shape == (1, 1)
        assert not torch.any(torch.isnan(position))
        assert not torch.any(torch.isnan(confidence))
    
    def test_large_input(self):
        """Test handling of unusually large input values."""
        model = EchoLocModel(n_sensors=4)
        
        x = torch.ones(1, 4, 2048) * 1000  # Very large values
        position, confidence = model(x)
        
        # Should still produce valid output
        assert not torch.any(torch.isnan(position))
        assert not torch.any(torch.isnan(confidence))
    
    def test_nan_input_handling(self):
        """Test handling of NaN input values."""
        model = EchoLocModel(n_sensors=4)
        
        x = torch.randn(1, 4, 2048)
        x[0, 0, :100] = float('nan')  # Insert NaN values
        
        # Model should handle NaN gracefully (implementation dependent)
        try:
            position, confidence = model(x)
            # If it doesn't raise an error, check outputs are valid
            assert not torch.any(torch.isnan(position))
            assert not torch.any(torch.isnan(confidence))
        except (RuntimeError, ValueError):
            # It's acceptable to raise an error for NaN input
            pass


class TestModelPerformance:
    """Test model performance characteristics."""
    
    def test_inference_speed(self):
        """Test inference speed benchmarking."""
        model = EchoLocModel(n_sensors=4, model_size="tiny")
        model.eval()
        
        x = torch.randn(10, 4, 2048)
        
        import time
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        avg_time = (time.time() - start_time) / 10
        
        # Should be reasonably fast (adjust threshold as needed)
        assert avg_time < 1.0  # Less than 1 second for 10 samples
    
    def test_memory_usage(self):
        """Test memory usage patterns."""
        import gc
        import torch
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = EchoLocModel(n_sensors=4, model_size="base").to(device)
            
            # Clear cache
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Run inference
            x = torch.randn(1, 4, 2048, device=device)
            with torch.no_grad():
                _ = model(x)
            
            peak_memory = torch.cuda.memory_allocated()
            memory_increase = peak_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 1024 * 1024 * 1024  # Less than 1GB
    
    def test_batch_processing(self):
        """Test batch processing efficiency."""
        model = EchoLocModel(n_sensors=4, model_size="base")
        model.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 4, 2048)
            
            with torch.no_grad():
                positions, confidences = model(x)
            
            assert positions.shape == (batch_size, 3)
            assert confidences.shape == (batch_size, 1)


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for complete model pipeline."""
    
    def test_training_mode_switch(self):
        """Test switching between training and evaluation modes."""
        model = EchoLocModel(n_sensors=4)
        
        # Should start in training mode
        assert model.training
        
        # Switch to eval mode
        model.eval()
        assert not model.training
        
        # Switch back to training mode
        model.train()
        assert model.training
    
    def test_parameter_updates(self):
        """Test that parameters can be updated."""
        model = EchoLocModel(n_sensors=4, model_size="tiny")
        
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Simulate training step
        x = torch.randn(2, 4, 2048)
        target_pos = torch.randn(2, 3)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        positions, confidences = model(x)
        loss = torch.nn.functional.mse_loss(positions, target_pos)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters have changed
        changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.equal(initial, current):
                changed = True
                break
        
        assert changed, "Parameters should change after training step"
    
    def test_device_transfer(self):
        """Test model device transfer."""
        model = EchoLocModel(n_sensors=4, model_size="tiny")
        
        # Model should start on CPU
        assert next(model.parameters()).device.type == "cpu"
        
        if torch.cuda.is_available():
            # Transfer to GPU
            model = model.cuda()
            assert next(model.parameters()).device.type == "cuda"
            
            # Test inference on GPU
            x = torch.randn(1, 4, 2048, device="cuda")
            positions, confidences = model(x)
            
            assert positions.device.type == "cuda"
            assert confidences.device.type == "cuda"
            
            # Transfer back to CPU
            model = model.cpu()
            assert next(model.parameters()).device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__])