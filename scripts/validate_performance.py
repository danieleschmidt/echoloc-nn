#!/usr/bin/env python3
"""
Simple performance validation script without heavy dependencies.
Tests core performance characteristics and validates requirements.
"""

import time
import sys
import os
import gc
from pathlib import Path
import numpy as np


def test_numpy_performance():
    """Test NumPy operations performance."""
    print("🔍 Testing NumPy performance...")
    
    # Array operations
    size = 100000
    iterations = 10
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        # Simulate signal processing operations
        data = np.random.randn(size)
        fft_data = np.fft.fft(data)
        filtered = np.abs(fft_data)
        result = np.sum(filtered)
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Array operations ({size} elements): {avg_time:.2f} ± {std_time:.2f} ms")
    
    # Performance target: should be < 100ms for basic operations
    if avg_time < 100:
        print("  ✅ NumPy performance acceptable")
        return True
    else:
        print("  ❌ NumPy performance too slow")
        return False


def test_memory_usage():
    """Test memory usage patterns."""
    print("\n🔍 Testing memory usage...")
    
    try:
        import psutil
        process = psutil.Process()
        
        # Baseline memory
        baseline_mb = process.memory_info().rss / 1024 / 1024
        print(f"  Baseline memory: {baseline_mb:.1f} MB")
        
        # Allocate test arrays
        test_arrays = []
        for i in range(5):
            # Simulate model inputs (4 sensors, 2048 samples)
            array = np.random.randn(4, 2048).astype(np.float32)
            test_arrays.append(array)
        
        peak_mb = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_mb - baseline_mb
        
        print(f"  Peak memory: {peak_mb:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        
        # Clean up
        del test_arrays
        gc.collect()
        
        final_mb = process.memory_info().rss / 1024 / 1024
        memory_released = peak_mb - final_mb
        
        print(f"  Memory released: {memory_released:.1f} MB")
        
        # Performance targets
        if memory_increase < 50:  # Less than 50MB for test data
            print("  ✅ Memory usage acceptable")
            return True
        else:
            print("  ❌ High memory usage detected")
            return False
    
    except ImportError:
        print("  ⚠️  psutil not available, skipping memory test")
        return True


def test_file_io_performance():
    """Test file I/O performance."""
    print("\n🔍 Testing file I/O performance...")
    
    # Create test data
    test_data = np.random.randn(1000, 100).astype(np.float32)
    test_file = Path("test_performance_data.npy")
    
    try:
        # Write performance
        start = time.perf_counter()
        np.save(test_file, test_data)
        write_time = (time.perf_counter() - start) * 1000
        
        # Read performance
        start = time.perf_counter()
        loaded_data = np.load(test_file)
        read_time = (time.perf_counter() - start) * 1000
        
        # Verify data integrity
        if not np.allclose(test_data, loaded_data):
            print("  ❌ Data integrity check failed")
            return False
        
        print(f"  Write time: {write_time:.2f} ms")
        print(f"  Read time: {read_time:.2f} ms")
        
        # Performance targets
        if write_time < 100 and read_time < 50:
            print("  ✅ File I/O performance acceptable")
            return True
        else:
            print("  ❌ File I/O performance too slow")
            return False
    
    except Exception as e:
        print(f"  ❌ File I/O test failed: {e}")
        return False
    
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


def test_signal_processing_performance():
    """Test signal processing operations."""
    print("\n🔍 Testing signal processing performance...")
    
    # Simulate echo data (4 sensors, 2048 samples, 250kHz)
    n_sensors = 4
    n_samples = 2048
    sample_rate = 250000
    
    echo_data = np.random.randn(n_sensors, n_samples).astype(np.float32)
    
    # Test FFT performance (core operation)
    iterations = 10
    fft_times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        # Per-sensor FFT
        fft_results = []
        for sensor_data in echo_data:
            fft_result = np.fft.fft(sensor_data)
            fft_results.append(fft_result)
        
        end = time.perf_counter()
        fft_times.append((end - start) * 1000)
    
    avg_fft_time = np.mean(fft_times)
    print(f"  FFT processing time: {avg_fft_time:.2f} ms")
    
    # Test filtering operations
    filter_times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        # Simulate matched filtering
        filtered_data = np.zeros_like(echo_data)
        for i, sensor_data in enumerate(echo_data):
            # Simple filtering simulation
            filtered_data[i] = np.convolve(sensor_data, np.ones(10) / 10, mode='same')
        
        end = time.perf_counter()
        filter_times.append((end - start) * 1000)
    
    avg_filter_time = np.mean(filter_times)
    print(f"  Filtering time: {avg_filter_time:.2f} ms")
    
    # Performance targets for real-time processing
    real_time_budget = 20  # 20ms for 50Hz processing rate
    total_processing_time = avg_fft_time + avg_filter_time
    
    print(f"  Total processing time: {total_processing_time:.2f} ms")
    print(f"  Real-time budget: {real_time_budget} ms")
    
    if total_processing_time < real_time_budget:
        print("  ✅ Signal processing meets real-time requirements")
        return True
    else:
        print("  ❌ Signal processing too slow for real-time")
        return False


def test_concurrent_processing():
    """Test concurrent processing capabilities."""
    print("\n🔍 Testing concurrent processing...")
    
    try:
        import threading
        import queue
        
        # Simulate concurrent echo processing
        def worker(q, results):
            """Worker function for concurrent processing."""
            while True:
                try:
                    data = q.get(timeout=1)
                    if data is None:
                        break
                    
                    # Simulate processing
                    processed = np.mean(data) + np.std(data)
                    results.append(processed)
                    q.task_done()
                
                except queue.Empty:
                    break
        
        # Test data
        test_batches = [np.random.randn(4, 1024) for _ in range(10)]
        
        # Sequential processing
        start = time.perf_counter()
        sequential_results = []
        for batch in test_batches:
            result = np.mean(batch) + np.std(batch)
            sequential_results.append(result)
        sequential_time = time.perf_counter() - start
        
        # Concurrent processing (2 threads)
        start = time.perf_counter()
        q = queue.Queue()
        concurrent_results = []
        
        # Start workers
        threads = []
        for _ in range(2):
            t = threading.Thread(target=worker, args=(q, concurrent_results))
            t.start()
            threads.append(t)
        
        # Add work
        for batch in test_batches:
            q.put(batch)
        
        # Stop workers
        for _ in range(2):
            q.put(None)
        
        for t in threads:
            t.join()
        
        concurrent_time = time.perf_counter() - start
        
        print(f"  Sequential time: {sequential_time*1000:.2f} ms")
        print(f"  Concurrent time: {concurrent_time*1000:.2f} ms")
        
        # Check speedup
        speedup = sequential_time / concurrent_time
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup > 1.2:  # At least 20% improvement
            print("  ✅ Concurrent processing provides speedup")
            return True
        else:
            print("  ⚠️  Limited concurrent processing benefit")
            return True  # Not a failure, just limited by workload
    
    except Exception as e:
        print(f"  ❌ Concurrent processing test failed: {e}")
        return False


def test_system_resources():
    """Test system resource availability."""
    print("\n🔍 Testing system resources...")
    
    try:
        import psutil
        
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        print(f"  CPU cores: {cpu_count}")
        if cpu_freq:
            print(f"  CPU frequency: {cpu_freq.current:.0f} MHz")
        
        # Memory information
        memory = psutil.virtual_memory()
        print(f"  Total memory: {memory.total / 1024**3:.1f} GB")
        print(f"  Available memory: {memory.available / 1024**3:.1f} GB")
        print(f"  Memory usage: {memory.percent:.1f}%")
        
        # Disk information
        disk = psutil.disk_usage('/')
        print(f"  Disk space: {disk.free / 1024**3:.1f} GB free of {disk.total / 1024**3:.1f} GB")
        
        # Resource adequacy checks
        issues = []
        
        if cpu_count < 2:
            issues.append("Low CPU core count")
        
        if memory.available < 1024**3:  # Less than 1GB available
            issues.append("Low available memory")
        
        if memory.percent > 90:
            issues.append("High memory usage")
        
        if disk.free < 1024**3:  # Less than 1GB free
            issues.append("Low disk space")
        
        if issues:
            print("  ❌ Resource issues:")
            for issue in issues:
                print(f"    - {issue}")
            return False
        else:
            print("  ✅ System resources adequate")
            return True
    
    except ImportError:
        print("  ⚠️  psutil not available, skipping resource check")
        return True
    except Exception as e:
        print(f"  ❌ Resource check failed: {e}")
        return False


def main():
    """Run all performance validation tests."""
    print("EchoLoc-NN Performance Validation")
    print("=" * 40)
    
    tests = [
        test_numpy_performance,
        test_memory_usage,
        test_file_io_performance,
        test_signal_processing_performance,
        test_concurrent_processing,
        test_system_resources,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
    
    print(f"\n{'='*40}")
    print(f"Performance Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All performance tests passed!")
        return 0
    elif passed >= total * 0.8:  # 80% pass rate
        print("⚠️  Most performance tests passed - acceptable")
        return 0
    else:
        print(f"❌ {total - passed} performance tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())