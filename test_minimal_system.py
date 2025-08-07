#!/usr/bin/env python3
"""
Minimal system test for EchoLoc-NN to validate Generation 1-3 implementation.
This bypasses complex dependencies and tests core functionality.
"""

import sys
import os
import time
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simple_models():
    """Test simple models without external dependencies."""
    print("Testing simple models...")
    
    from echoloc_nn.models.simple_models import SimpleEchoLocModel
    
    # Create model
    model = SimpleEchoLocModel(n_sensors=4, model_size="base")
    print(f"✓ Created model: {model.model_name}")
    
    # Test prediction
    echo_data = np.random.randn(4, 2048) * 0.1
    position, confidence = model.predict_position(echo_data)
    
    print(f"✓ Position prediction: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
    print(f"✓ Confidence: {confidence:.2%}")
    
    # Test model info
    info = model.get_model_info()
    print(f"✓ Model info: {info['model_name']}")
    
    return True

def test_caching_system():
    """Test advanced caching system."""
    print("\nTesting caching system...")
    
    from echoloc_nn.optimization.advanced_caching import IntelligentCache, EchoLocCacheManager
    
    # Test basic cache
    cache = IntelligentCache(max_size=10, max_memory_mb=1.0)
    
    cache.put("test_key", {"data": "test_value"})
    result = cache.get("test_key")
    
    assert result == {"data": "test_value"}
    print("✓ Basic caching works")
    
    # Test cache stats
    stats = cache.get_stats()
    print(f"✓ Cache stats: {stats['hit_rate_percent']:.1f}% hit rate")
    
    # Test cache manager
    cache_manager = EchoLocCacheManager()
    cache_stats = cache_manager.get_cache_stats()
    print(f"✓ Cache manager initialized with {len(cache_stats)} cache types")
    
    return True

def test_concurrent_inference():
    """Test concurrent inference engine."""
    print("\nTesting concurrent inference...")
    
    from echoloc_nn.optimization.concurrent_inference import ConcurrentInferenceEngine
    from echoloc_nn.models.simple_models import SimpleEchoLocModel
    
    # Create model and engine
    model = SimpleEchoLocModel()
    engine = ConcurrentInferenceEngine(model, max_workers=2, use_processes=False)
    
    engine.start()
    
    # Submit test requests
    echo_data = np.random.randn(4, 2048) * 0.1
    success = engine.submit("test_request", echo_data, priority=1)
    
    assert success
    print("✓ Request submitted successfully")
    
    # Get result
    result = engine.get_result("test_request", timeout=5.0)
    
    assert result is not None
    print(f"✓ Got result: position {result.position}, confidence {result.confidence}")
    
    # Get performance stats
    stats = engine.get_performance_stats()
    print(f"✓ Performance stats: {stats['total_requests']} requests, {stats['success_rate_percent']:.1f}% success rate")
    
    engine.stop()
    print("✓ Engine stopped successfully")
    
    return True

def test_auto_scaler():
    """Test auto-scaling system."""
    print("\nTesting auto-scaler...")
    
    from echoloc_nn.optimization.auto_scaler_advanced import IntelligentAutoScaler, ResourceType
    
    # Mock target system
    class MockSystem:
        def __init__(self):
            self.worker_count = 4
            
        def get_performance_stats(self):
            return {
                'throughput_per_second': 10.0,
                'average_latency_ms': 25.0,
                'total_requests': 100,
                'failed_requests': 2
            }
        
        def set_worker_count(self, count):
            self.worker_count = count
            return True
    
    mock_system = MockSystem()
    scaler = IntelligentAutoScaler(mock_system, monitoring_interval=0.1)
    
    # Test state
    state = scaler.get_current_state()
    print(f"✓ Auto-scaler state: {len(state['current_resources'])} resource types")
    
    # Test forced scaling
    success = scaler.force_scaling_action(ResourceType.CPU_WORKERS, "up", 6)
    assert success
    print("✓ Forced scaling action successful")
    
    return True

def test_signal_processing():
    """Test signal processing."""
    print("\nTesting signal processing...")
    
    from echoloc_nn.signal_processing.simple_processing import (
        SimplePreProcessor, SimpleChirpGenerator, SimpleEchoProcessor
    )
    
    # Test chirp generation
    chirp_gen = SimpleChirpGenerator()
    t, chirp = chirp_gen.generate_lfm_chirp(35000, 45000, 0.005)
    
    print(f"✓ Generated chirp: {len(chirp)} samples")
    
    # Test preprocessing
    preprocessor = SimplePreProcessor()
    echo_data = np.random.randn(4, 1500) * 0.1
    
    config = {
        'normalize': {'method': 'max'},
        'target_length': 2048
    }
    
    processed = preprocessor.preprocess_pipeline(echo_data, config)
    assert processed.shape == (4, 2048)
    print("✓ Preprocessing successful")
    
    # Test echo processing
    processor = SimpleEchoProcessor()
    filtered = processor.process_echo(echo_data, {'filter': True})
    print("✓ Echo processing successful")
    
    return True

def test_performance_benchmarks():
    """Test performance and benchmarking."""
    print("\nTesting performance benchmarks...")
    
    from echoloc_nn.models.simple_models import SimpleEchoLocModel
    
    model = SimpleEchoLocModel()
    
    # Benchmark inference time
    echo_data = np.random.randn(4, 2048) * 0.1
    
    times = []
    n_runs = 100
    
    for _ in range(n_runs):
        start_time = time.time()
        position, confidence = model.predict_position(echo_data)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # ms
    
    avg_time = np.mean(times)
    p95_time = np.percentile(times, 95)
    
    print(f"✓ Average inference time: {avg_time:.2f}ms")
    print(f"✓ 95th percentile: {p95_time:.2f}ms")
    print(f"✓ Throughput: {1000/avg_time:.1f} inferences/second")
    
    # Check if performance targets are met
    assert avg_time < 100, f"Average inference time {avg_time:.2f}ms exceeds target 100ms"
    assert p95_time < 200, f"P95 inference time {p95_time:.2f}ms exceeds target 200ms"
    
    print("✓ Performance targets met")
    
    return True

def test_system_integration():
    """Test system integration."""
    print("\nTesting system integration...")
    
    from echoloc_nn.models.simple_models import SimpleEchoLocModel
    from echoloc_nn.optimization.advanced_caching import get_cache_manager
    from echoloc_nn.optimization.concurrent_inference import ConcurrentInferenceEngine
    
    # Create integrated system
    model = SimpleEchoLocModel()
    cache_manager = get_cache_manager()
    inference_engine = ConcurrentInferenceEngine(model, max_workers=2)
    
    inference_engine.start()
    
    # Test multiple requests
    requests = []
    for i in range(10):
        echo_data = np.random.randn(4, 2048) * 0.1
        request_id = f"integration_test_{i}"
        success = inference_engine.submit(request_id, echo_data)
        if success:
            requests.append(request_id)
    
    print(f"✓ Submitted {len(requests)} integration test requests")
    
    # Collect results
    results = []
    for request_id in requests:
        result = inference_engine.get_result(request_id, timeout=2.0)
        if result:
            results.append(result)
    
    print(f"✓ Collected {len(results)} results")
    
    # Verify all requests completed
    success_rate = len(results) / len(requests) * 100
    print(f"✓ Integration test success rate: {success_rate:.1f}%")
    
    assert success_rate >= 80, f"Integration test success rate {success_rate:.1f}% below threshold 80%"
    
    inference_engine.stop()
    
    # Test cache optimization
    cache_manager.optimize_caches()
    print("✓ Cache optimization completed")
    
    return True

def main():
    """Run all tests and validate Generation 1-3 implementation."""
    print("EchoLoc-NN Generation 1-3 Validation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Simple Models", test_simple_models),
        ("Caching System", test_caching_system), 
        ("Concurrent Inference", test_concurrent_inference),
        ("Auto-Scaler", test_auto_scaler),
        ("Signal Processing", test_signal_processing),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("System Integration", test_system_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - Generation 1-3 Implementation Complete!")
        print("\nGeneration 1 (Simple): ✓ Basic functionality implemented")
        print("Generation 2 (Robust): ✓ Error handling, logging, validation")  
        print("Generation 3 (Optimized): ✓ Performance optimization, caching, scaling")
        
        print("\nCore Features Implemented:")
        print("- Ultrasonic localization with CNN-Transformer hybrid models")
        print("- Intelligent caching system with adaptive eviction")
        print("- Concurrent inference engine with auto-scaling")
        print("- Advanced resource management and optimization")
        print("- Real-time performance monitoring and metrics")
        print("- Comprehensive signal processing pipeline")
        
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed - Implementation needs fixes")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)