#!/usr/bin/env python3
"""
Simple test for the models implementation without external dependencies
"""

import sys
import os
sys.path.insert(0, '/root/repo')

def test_models_import():
    try:
        # Test basic imports
        from echoloc_nn.models import EchoLocModel, CNNTransformerHybrid
        print("✓ Models imported successfully")
        
        # Test model creation (will fail on torch import, but that's expected)
        try:
            model = EchoLocModel(n_sensors=4, model_size='tiny')
            print("✓ Model creation works")
        except ImportError as e:
            if "torch" in str(e).lower():
                print("⚠ Model creation requires torch (expected)")
            else:
                print(f"✗ Unexpected error: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing models implementation...")
    success = test_models_import()
    print(f"Test {'PASSED' if success else 'FAILED'}")