#!/usr/bin/env python3
"""
Simple test for Generation 3 optimization system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test that all Generation 3 components can be imported."""
    print("Testing Generation 3 imports...")
    
    try:
        from echoloc_nn.models.hybrid_architecture import EchoLocModel
        print("✅ EchoLocModel imported successfully")
    except ImportError as e:
        print(f"❌ EchoLocModel import failed: {e}")
        return False
    
    try:
        from echoloc_nn.optimization.generation_3_optimizer import Generation3Optimizer
        print("✅ Generation3Optimizer imported successfully")
    except ImportError as e:
        print(f"❌ Generation3Optimizer import failed: {e}")
        return False
    
    try:
        from echoloc_nn.optimization.model_optimizer import ModelOptimizer
        print("✅ ModelOptimizer imported successfully")
    except ImportError as e:
        print(f"❌ ModelOptimizer import failed: {e}")
        return False
    
    try:
        from echoloc_nn.optimization.caching import EchoCache
        print("✅ EchoCache imported successfully")
    except ImportError as e:
        print(f"❌ EchoCache import failed: {e}")
        return False
    
    try:
        from echoloc_nn.optimization.performance_monitor import PerformanceMonitor
        print("✅ PerformanceMonitor imported successfully")
    except ImportError as e:
        print(f"❌ PerformanceMonitor import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation without dependencies."""
    print("\nTesting model creation...")
    
    try:
        from echoloc_nn.models.hybrid_architecture import EchoLocModel
        
        # Create a small model for testing
        model = EchoLocModel(n_sensors=4, model_size="tiny")
        print(f"✅ Model created with {model.get_parameter_count()} parameters")
        
        # Test model info
        info = model.get_model_info()
        print(f"✅ Model info: {info['model_class']}, {info['total_parameters']} params")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_optimizer_creation():
    """Test optimizer creation without external dependencies."""
    print("\nTesting optimizer creation...")
    
    try:
        from echoloc_nn.optimization.generation_3_optimizer import create_generation_3_optimizer
        
        # Create optimizer without model (testing configuration)
        optimizer = create_generation_3_optimizer(
            model=None,
            optimization_level="conservative",
            target_latency_ms=50.0,
            device="cpu"
        )
        
        print(f"✅ Generation 3 optimizer created")
        print(f"   Target latency: {optimizer.target_latency_ms}ms")
        print(f"   Optimization level: {optimizer.optimization_level}")
        print(f"   Device: {optimizer.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimizer creation failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("=" * 60)
    print("GENERATION 3 OPTIMIZATION SYSTEM - BASIC VALIDATION")
    print("=" * 60)
    
    success = True
    
    # Test imports
    if not test_basic_imports():
        success = False
    
    # Test model creation (if imports work)
    if success:
        if not test_model_creation():
            success = False
    
    # Test optimizer creation (if imports work)
    if success:
        if not test_optimizer_creation():
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 GENERATION 3 BASIC VALIDATION PASSED!")
        print("Core components are properly implemented and can be imported.")
        print("\nNext steps:")
        print("- Install torch/numpy for full validation")  
        print("- Run comprehensive performance tests")
        print("- Validate optimization targets")
    else:
        print("❌ GENERATION 3 VALIDATION FAILED")
        print("Some components have import or creation issues.")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())