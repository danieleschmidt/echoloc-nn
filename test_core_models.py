#!/usr/bin/env python3
"""
Core model system test without external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_model_imports():
    """Test model imports."""
    print("🧪 Testing model imports...")
    
    try:
        from echoloc_nn.models import EchoLocModel, create_model, MODELS_AVAILABLE
        print(f"✅ Models imported (PyTorch available: {MODELS_AVAILABLE})")
        return True
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("🧪 Testing model creation...")
    
    try:
        from echoloc_nn.models import create_model
        
        # Test different model configurations
        models = ['echoloc-base', 'echoloc-large', 'echoloc-micro']
        
        for model_name in models:
            model = create_model(model_name)
            info = model.get_model_info()
            print(f"✅ Created {model_name}: {info['model_name']}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_functionality():
    """Test basic model functionality."""
    print("🧪 Testing model functionality...")
    
    try:
        from echoloc_nn.models import create_model
        from echoloc_nn.models.mock_models import MockTensor
        
        model = create_model('echoloc-base')
        
        # Test forward pass with mock data
        mock_input = MockTensor((1, 4, 2048))  # batch_size=1, sensors=4, samples=2048
        position, confidence = model.forward(mock_input)
        
        print(f"✅ Forward pass successful")
        print(f"   Position shape: {position.shape}")
        print(f"   Confidence shape: {confidence.shape}")
        
        # Test prediction interface
        result = model.predict(mock_input)
        print(f"✅ Prediction interface working")
        print(f"   Keys: {list(result.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Model functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_checkpoint_operations():
    """Test model save/load operations."""
    print("🧪 Testing checkpoint operations...")
    
    try:
        from echoloc_nn.models import create_model
        import tempfile
        import os
        
        # Create model
        model = create_model('echoloc-micro')
        
        # Test save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        model.save_checkpoint(checkpoint_path)
        print("✅ Model save successful")
        
        # Test load
        loaded_model, optimizer_state = model.__class__.load_checkpoint(checkpoint_path)
        print("✅ Model load successful")
        
        # Cleanup
        os.unlink(checkpoint_path)
        
        return True
    except Exception as e:
        print(f"❌ Checkpoint operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 EchoLoc-NN Core Model System Test")
    print("=" * 40)
    
    tests = [
        test_model_imports,
        test_model_creation,
        test_model_functionality,
        test_checkpoint_operations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Model system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return 1

if __name__ == '__main__':
    exit(main())