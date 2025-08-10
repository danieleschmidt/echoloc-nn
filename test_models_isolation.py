#!/usr/bin/env python3
"""
Test the models in isolation to verify they work correctly.
"""

import torch
import torch.nn as nn

# Add models to path
import sys
import os
sys.path.insert(0, 'echoloc_nn/models')

# Import our models
from base import EchoLocBaseModel
from hybrid_architecture import EchoLocModel, CNNTransformerHybrid, CNNEncoder, TransformerDecoder

def test_base_model():
    """Test base model functionality."""
    print("Testing EchoLocBaseModel...")
    
    # Create simple test model
    class TestModel(EchoLocBaseModel):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2048, 4)  # Simple linear layer
            
        def forward(self, echo_data, sensor_positions=None):
            batch_size = echo_data.shape[0]
            # Simple dummy computation
            x = echo_data.mean(dim=1)  # Average over sensors
            x = self.linear(x)
            position = x[:, :3]  # First 3 outputs as position
            confidence = torch.sigmoid(x[:, 3:4])  # Last output as confidence
            return position, confidence
    
    model = TestModel()
    echo_data = torch.randn(2, 4, 2048)
    
    # Test forward pass
    positions, confidence = model(echo_data)
    assert positions.shape == (2, 3)
    assert confidence.shape == (2, 1)
    
    # Test predict_position
    echo_np = torch.randn(4, 2048).numpy()
    pos, conf = model.predict_position(echo_np)
    assert pos.shape == (3,)
    assert isinstance(conf, float)
    
    print("✓ Base model tests passed!")

def test_cnn_encoder():
    """Test CNN encoder."""
    print("Testing CNNEncoder...")
    
    encoder = CNNEncoder(n_sensors=4, input_length=2048, channels=[32, 64, 128])
    x = torch.randn(2, 4, 2048)
    
    output = encoder(x)
    assert output.dim() == 3
    assert output.shape[0] == 2  # batch size
    assert output.shape[1] == 128  # output channels
    
    print(f"✓ CNN encoder output shape: {output.shape}")

def test_transformer_decoder():
    """Test Transformer decoder."""
    print("Testing TransformerDecoder...")
    
    decoder = TransformerDecoder(d_model=256, n_heads=8, n_layers=2)
    x = torch.randn(2, 100, 256)
    
    output = decoder(x)
    assert output.shape == x.shape
    
    print(f"✓ Transformer decoder output shape: {output.shape}")

def test_hybrid_model():
    """Test CNN-Transformer hybrid."""
    print("Testing CNNTransformerHybrid...")
    
    model = CNNTransformerHybrid(
        n_sensors=4,
        chirp_length=2048,
        transformer_dim=256
    )
    
    x = torch.randn(2, 4, 2048)
    positions, confidence = model(x)
    
    assert positions.shape == (2, 3)
    assert confidence.shape == (2, 1)
    assert torch.all(confidence >= 0) and torch.all(confidence <= 1)
    
    print(f"✓ Hybrid model output shapes: positions={positions.shape}, confidence={confidence.shape}")

def test_echoloc_model():
    """Test main EchoLoc model."""
    print("Testing EchoLocModel...")
    
    # Test all model sizes
    sizes = ["tiny", "base", "large"]
    
    for size in sizes:
        model = EchoLocModel(n_sensors=4, model_size=size)
        x = torch.randn(1, 4, 2048)
        
        positions, confidence = model(x)
        assert positions.shape == (1, 3)
        assert confidence.shape == (1, 1)
        
        # Test predict_position method
        echo_np = torch.randn(4, 2048).numpy()
        pos, conf = model.predict_position(echo_np)
        assert pos.shape == (3,)
        assert isinstance(conf, float)
        assert 0 <= conf <= 1
        
        print(f"✓ EchoLocModel {size} works correctly")

def test_model_info():
    """Test model information methods."""
    print("Testing model information methods...")
    
    model = EchoLocModel(n_sensors=4, model_size="tiny")
    
    # Test get_model_info
    info = model.get_model_info()
    assert 'model_class' in info
    assert 'total_parameters' in info
    assert info['total_parameters'] > 0
    
    # Test estimate_compute_complexity
    complexity = model.estimate_compute_complexity()
    assert 'total_parameters' in complexity
    assert 'estimated_flops' in complexity
    
    # Test summary
    summary = model.summary()
    assert 'EchoLoc Model Summary' in summary
    
    print("✓ Model information methods work correctly")

def test_device_handling():
    """Test device handling."""
    print("Testing device handling...")
    
    model = EchoLocModel(n_sensors=4, model_size="tiny")
    device = model.get_device()
    assert device.type == 'cpu'
    
    print(f"✓ Model on device: {device}")

if __name__ == "__main__":
    print("Testing EchoLoc-NN Models")
    print("=" * 40)
    
    test_base_model()
    test_cnn_encoder()
    test_transformer_decoder()
    test_hybrid_model()
    test_echoloc_model()
    test_model_info()
    test_device_handling()
    
    print("\n" + "=" * 40)
    print("All tests passed! ✓")
    
    # Show model sizes
    print("\nModel Size Comparison:")
    for size in ["tiny", "base", "large"]:
        model = EchoLocModel(n_sensors=4, model_size=size)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {size:5}: {params:,} parameters")