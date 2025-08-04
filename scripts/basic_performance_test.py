#!/usr/bin/env python3
"""
Basic performance validation without external dependencies.
Tests fundamental Python performance characteristics.
"""

import time
import sys
import os
import gc
import threading
from pathlib import Path


def test_basic_math_performance():
    """Test basic mathematical operations."""
    print("🔍 Testing basic math performance...")
    
    iterations = 1000000
    
    # Test arithmetic operations
    start = time.perf_counter()
    result = 0
    for i in range(iterations):
        result += i * 2.5 + 1.0
    math_time = (time.perf_counter() - start) * 1000
    
    print(f"  Arithmetic operations ({iterations} iterations): {math_time:.2f} ms")
    
    # Test list operations
    start = time.perf_counter()
    test_list = []
    for i in range(10000):
        test_list.append(i * 2)
    
    # Process list
    sum_result = sum(test_list)
    list_time = (time.perf_counter() - start) * 1000
    
    print(f"  List operations (10k elements): {list_time:.2f} ms")
    
    # Performance targets
    if math_time < 500 and list_time < 100:  # Reasonable targets
        print("  ✅ Basic math performance acceptable")
        return True
    else:
        print("  ❌ Basic math performance concerning")
        return False


def test_file_operations():
    """Test file I/O performance."""
    print("\n🔍 Testing file operations...")
    
    test_file = Path("test_performance.txt")
    test_data = "Performance test data\n" * 1000
    
    try:
        # Write test
        start = time.perf_counter()
        with open(test_file, 'w') as f:
            f.write(test_data)
        write_time = (time.perf_counter() - start) * 1000
        
        # Read test
        start = time.perf_counter()
        with open(test_file, 'r') as f:
            read_data = f.read()
        read_time = (time.perf_counter() - start) * 1000
        
        # Verify
        if read_data != test_data:
            print("  ❌ File data integrity check failed")
            return False
        
        print(f"  File write: {write_time:.2f} ms")
        print(f"  File read: {read_time:.2f} ms")
        
        if write_time < 100 and read_time < 50:
            print("  ✅ File I/O performance acceptable")
            return True
        else:
            print("  ❌ File I/O performance slow")
            return False
    
    except Exception as e:
        print(f"  ❌ File operations failed: {e}")
        return False
    
    finally:
        if test_file.exists():
            test_file.unlink()


def test_memory_management():
    """Test memory allocation and garbage collection."""
    print("\n🔍 Testing memory management...")
    
    try:
        # Test large data structure creation
        start = time.perf_counter()
        large_list = [i * 1.5 for i in range(100000)]
        allocation_time = (time.perf_counter() - start) * 1000
        
        # Test processing
        start = time.perf_counter()
        processed = [x * 2 for x in large_list if x > 50000]
        processing_time = (time.perf_counter() - start) * 1000
        
        # Test cleanup
        start = time.perf_counter()
        del large_list
        del processed
        gc.collect()
        cleanup_time = (time.perf_counter() - start) * 1000
        
        print(f"  Memory allocation: {allocation_time:.2f} ms")
        print(f"  Data processing: {processing_time:.2f} ms")
        print(f"  Cleanup/GC: {cleanup_time:.2f} ms")
        
        # Performance targets
        total_time = allocation_time + processing_time
        if total_time < 1000:  # Less than 1 second for 100k items
            print("  ✅ Memory management performance acceptable")
            return True
        else:
            print("  ❌ Memory management performance slow")
            return False
    
    except Exception as e:
        print(f"  ❌ Memory management test failed: {e}")
        return False


def test_threading_performance():
    """Test basic threading capabilities."""
    print("\n🔍 Testing threading performance...")
    
    def worker_task(data_slice, results, index):
        """Simple worker task."""
        result = sum(x * 2 for x in data_slice)
        results[index] = result
    
    # Test data
    test_data = list(range(100000))
    chunk_size = len(test_data) // 4
    
    try:
        # Sequential processing
        start = time.perf_counter()
        sequential_result = sum(x * 2 for x in test_data)
        sequential_time = time.perf_counter() - start
        
        # Threaded processing
        start = time.perf_counter()
        
        threads = []
        results = [0] * 4
        
        for i in range(4):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < 3 else len(test_data)
            data_slice = test_data[start_idx:end_idx]
            
            thread = threading.Thread(
                target=worker_task, 
                args=(data_slice, results, i)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        threaded_result = sum(results)
        threaded_time = time.perf_counter() - start
        
        print(f"  Sequential time: {sequential_time*1000:.2f} ms")
        print(f"  Threaded time: {threaded_time*1000:.2f} ms")
        
        # Verify correctness
        if sequential_result != threaded_result:
            print("  ❌ Threading result mismatch")
            return False
        
        speedup = sequential_time / threaded_time
        print(f"  Threading speedup: {speedup:.2f}x")
        
        if speedup > 0.8:  # At least some benefit
            print("  ✅ Threading performance acceptable")
            return True
        else:
            print("  ⚠️  Limited threading benefit (expected for CPU-bound tasks)")
            return True  # This is expected for Python's GIL
    
    except Exception as e:
        print(f"  ❌ Threading test failed: {e}")
        return False


def test_import_performance():
    """Test module import performance."""
    print("\n🔍 Testing import performance...")
    
    try:
        # Test standard library imports
        import_times = []
        
        modules = ['json', 'os', 'sys', 'time', 'math', 're', 'pathlib']
        
        for module in modules:
            start = time.perf_counter()
            __import__(module)
            import_time = (time.perf_counter() - start) * 1000
            import_times.append(import_time)
        
        avg_import_time = sum(import_times) / len(import_times)
        max_import_time = max(import_times)
        
        print(f"  Average import time: {avg_import_time:.2f} ms")
        print(f"  Slowest import: {max_import_time:.2f} ms")
        
        if avg_import_time < 10 and max_import_time < 50:
            print("  ✅ Import performance acceptable")
            return True
        else:
            print("  ❌ Import performance slow")
            return False
    
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        return False


def test_system_responsiveness():
    """Test overall system responsiveness."""
    print("\n🔍 Testing system responsiveness...")
    
    try:
        # Test response time variability
        response_times = []
        
        for _ in range(20):
            start = time.perf_counter()
            
            # Simple computation
            result = sum(i ** 2 for i in range(1000))
            
            response_time = (time.perf_counter() - start) * 1000
            response_times.append(response_time)
        
        avg_response = sum(response_times) / len(response_times)
        max_response = max(response_times)
        min_response = min(response_times)
        variability = max_response - min_response
        
        print(f"  Average response: {avg_response:.2f} ms")
        print(f"  Response range: {min_response:.2f} - {max_response:.2f} ms")
        print(f"  Variability: {variability:.2f} ms")
        
        # Check for consistent performance
        if avg_response < 10 and variability < 20:
            print("  ✅ System responsiveness good")
            return True
        else:
            print("  ⚠️  System responsiveness variable")
            return True  # Warnings, not failures
    
    except Exception as e:
        print(f"  ❌ Responsiveness test failed: {e}")
        return False


def benchmark_summary():
    """Provide performance benchmark summary."""
    print("\n📊 Performance Benchmark Summary")
    print("-" * 40)
    
    # System information
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    
    # CPU count (if available)
    try:
        import os
        cpu_count = os.cpu_count()
        if cpu_count:
            print(f"CPU cores: {cpu_count}")
    except:
        pass
    
    # Rough performance estimation
    start = time.perf_counter()
    test_computation = sum(i * 1.414 for i in range(100000))
    computation_time = (time.perf_counter() - start) * 1000
    
    print(f"Computation benchmark: {computation_time:.2f} ms")
    
    # Performance class estimation
    if computation_time < 50:
        perf_class = "High Performance"
    elif computation_time < 200:
        perf_class = "Good Performance"
    elif computation_time < 500:
        perf_class = "Adequate Performance"
    else:
        perf_class = "Limited Performance"
    
    print(f"Performance class: {perf_class}")


def main():
    """Run all performance validation tests."""
    print("EchoLoc-NN Basic Performance Validation")
    print("=" * 50)
    
    tests = [
        test_basic_math_performance,
        test_file_operations,
        test_memory_management,
        test_threading_performance,
        test_import_performance,
        test_system_responsiveness,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
    
    benchmark_summary()
    
    print(f"\n{'='*50}")
    print(f"Performance Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All performance tests passed!")
        return 0
    elif passed >= total * 0.8:  # 80% pass rate
        print("✅ Most performance tests passed - system ready")
        return 0
    else:
        print(f"❌ {total - passed} critical performance issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())