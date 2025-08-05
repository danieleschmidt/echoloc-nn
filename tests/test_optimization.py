"""
Test Suite for Optimization Components

Tests quantum acceleration, resource pooling, and auto-scaling functionality.
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import optimization components
from echoloc_nn.optimization.quantum_accelerator import (
    QuantumAccelerator, AccelerationConfig, CacheConfig
)
from echoloc_nn.optimization.resource_pool import (
    ResourcePool, ResourceSpec, ResourceType, LoadBalancer
)
from echoloc_nn.optimization.auto_scaler import (
    AutoScaler, QuantumAwareAutoScaler, ScalingConfig, ResourceMonitor
)

class TestQuantumAccelerator:
    """Test suite for QuantumAccelerator."""
    
    def test_accelerator_creation(self):
        """Test accelerator initialization."""
        config = AccelerationConfig(enable_gpu=False)
        accelerator = QuantumAccelerator(config)
        
        assert accelerator.config == config
        assert accelerator.device is not None
        assert accelerator.num_cores > 0
        
    def test_vectorized_operations(self):
        """Test vectorized quantum operations."""
        config = AccelerationConfig(enable_gpu=False, enable_vectorization=True)
        accelerator = QuantumAccelerator(config)
        
        # Test matrix operations
        matrix_a = np.random.rand(10, 10)
        matrix_b = np.random.rand(10, 10)
        
        result = accelerator.accelerated_matrix_multiply(matrix_a, matrix_b)
        
        assert result.shape == (10, 10)
        # Verify correctness
        expected = np.dot(matrix_a, matrix_b)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        
    def test_parallel_processing(self):
        """Test parallel processing capabilities."""
        config = AccelerationConfig(enable_gpu=False, max_threads=4)
        accelerator = QuantumAccelerator(config)
        
        # Test parallel function execution
        def test_function(x):
            return x ** 2
            
        inputs = list(range(100))
        results = accelerator.parallel_execute(test_function, inputs)
        
        assert len(results) == 100
        assert all(results[i] == i ** 2 for i in range(100))
        
    def test_caching_functionality(self):
        """Test result caching."""
        cache_config = CacheConfig(enable_cache=True, max_cache_size=10)
        config = AccelerationConfig(
            enable_gpu=False,
            cache_config=cache_config
        )
        accelerator = QuantumAccelerator(config)
        
        # Test cache hit/miss
        key = "test_key"
        value = np.array([1, 2, 3, 4])
        
        # First call should be cache miss
        assert not accelerator.cache_get(key)
        accelerator.cache_set(key, value)
        
        # Second call should be cache hit
        cached_value = accelerator.cache_get(key)
        assert cached_value is not None
        np.testing.assert_array_equal(cached_value, value)
        
    def test_quantum_state_evolution(self):
        """Test quantum state evolution acceleration."""
        config = AccelerationConfig(enable_gpu=False)
        accelerator = QuantumAccelerator(config)
        
        # Initial quantum state
        initial_state = np.random.rand(8) + 1j * np.random.rand(8)
        initial_state = initial_state / np.linalg.norm(initial_state)
        
        # Evolution operator (unitary matrix)
        evolution_operator = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
        evolution_operator = evolution_operator / np.linalg.norm(evolution_operator)
        
        # Evolve state
        evolved_state = accelerator.evolve_quantum_state(initial_state, evolution_operator)
        
        assert evolved_state.shape == initial_state.shape
        assert abs(np.linalg.norm(evolved_state) - 1.0) < 1e-10  # Should remain normalized


class TestResourcePool:
    """Test suite for ResourcePool."""
    
    def test_pool_creation(self):
        """Test resource pool initialization."""
        pool = ResourcePool()
        
        assert pool.load_balancer is not None
        assert len(pool.resources) == 0
        assert pool.enable_auto_scaling == True
        
    def test_resource_registration(self):
        """Test resource registration."""
        pool = ResourcePool()
        
        spec = ResourceSpec(
            resource_type=ResourceType.CPU,
            capacity=4.0,
            capabilities={'compute', 'parallel'}
        )
        
        resource_id = pool.register_resource(spec)
        
        assert resource_id is not None
        assert len(pool.resources) == 1
        assert resource_id in pool.resources
        
    def test_resource_allocation(self):
        """Test resource allocation and deallocation."""
        pool = ResourcePool()
        
        # Register CPU resource
        cpu_spec = ResourceSpec(
            resource_type=ResourceType.CPU,
            capacity=4.0,
            capabilities={'compute'}
        )
        cpu_id = pool.register_resource(cpu_spec)
        
        # Allocate resource
        allocation_id = pool.allocate_resource(
            resource_type=ResourceType.CPU,
            required_capacity=2.0,
            required_capabilities={'compute'}
        )
        
        assert allocation_id is not None
        
        # Check resource usage
        stats = pool.get_pool_stats()
        assert stats['total_allocations'] == 1
        
        # Deallocate resource
        success = pool.deallocate_resource(allocation_id)
        assert success == True
        
        # Check resource freed
        updated_stats = pool.get_pool_stats()
        assert updated_stats['total_allocations'] == 0
        
    def test_load_balancing(self):
        """Test load balancing across resources."""
        pool = ResourcePool()
        
        # Register multiple CPU resources
        for i in range(3):
            spec = ResourceSpec(
                resource_type=ResourceType.CPU,
                capacity=2.0,
                capabilities={'compute'}
            )
            pool.register_resource(spec)
            
        # Make multiple allocations
        allocations = []
        for _ in range(6):
            allocation_id = pool.allocate_resource(
                resource_type=ResourceType.CPU,
                required_capacity=0.5,
                required_capabilities={'compute'}
            )
            if allocation_id:
                allocations.append(allocation_id)
                
        # Should have distributed load across resources
        assert len(allocations) == 6
        
        # Check load distribution
        resource_loads = {}
        for resource_id, resource in pool.resources.items():
            resource_loads[resource_id] = resource.current_load
            
        # Load should be relatively balanced
        load_values = list(resource_loads.values())
        assert max(load_values) - min(load_values) <= 1.0  # Reasonable distribution
        
    def test_resource_discovery(self):
        """Test resource discovery functionality."""
        pool = ResourcePool()
        
        # Register different types of resources
        cpu_spec = ResourceSpec(ResourceType.CPU, 4.0, {'compute'})
        gpu_spec = ResourceSpec(ResourceType.GPU, 1.0, {'parallel', 'ml'})
        memory_spec = ResourceSpec(ResourceType.MEMORY, 1024.0, {'storage'})
        
        pool.register_resource(cpu_spec)
        pool.register_resource(gpu_spec) 
        pool.register_resource(memory_spec)
        
        # Discover compute resources
        compute_resources = pool.find_resources(
            resource_type=ResourceType.CPU,
            required_capabilities={'compute'}
        )
        assert len(compute_resources) == 1
        
        # Discover ML-capable resources
        ml_resources = pool.find_resources(required_capabilities={'ml'})
        assert len(ml_resources) == 1
        
        # Discover all resources
        all_resources = pool.find_resources()
        assert len(all_resources) == 3


class TestAutoScaler:
    """Test suite for AutoScaler."""
    
    def test_scaler_creation(self):
        """Test auto-scaler initialization."""
        config = ScalingConfig(min_workers=1, max_workers=4)
        scaler = AutoScaler(config)
        
        assert scaler.config == config
        assert scaler.current_workers == config.min_workers
        assert scaler.resource_monitor is not None
        
    def test_resource_monitoring(self):
        """Test resource monitoring functionality."""
        monitor = ResourceMonitor(window_size=5)
        
        # Simulate resource usage
        monitor._collect_metrics()
        monitor.update_queue_size(10)
        monitor.update_processing_metrics(50.0, 20.0)
        
        # Check metrics
        assert monitor.get_avg_cpu() >= 0
        assert monitor.get_avg_memory() >= 0
        assert monitor.get_avg_queue_size() == 10.0
        assert monitor.get_avg_processing_time() == 50.0
        
    def test_scaling_decisions(self):
        """Test scaling decision logic."""
        config = ScalingConfig(
            cpu_scale_up_threshold=70.0,
            cpu_scale_down_threshold=30.0,
            min_workers=1,
            max_workers=8
        )
        scaler = AutoScaler(config)
        
        # Test scale up decision
        scale_decision = scaler._make_scaling_decision(80.0, 60.0, 10.0)  # High CPU
        assert scale_decision['action'] == 'scale_up'
        assert 'High CPU usage' in scale_decision['reasons'][0]
        
        # Test scale down decision
        scale_decision = scaler._make_scaling_decision(20.0, 40.0, 5.0)  # Low usage
        assert scale_decision['action'] == 'scale_down'
        
        # Test no scaling needed
        scale_decision = scaler._make_scaling_decision(50.0, 60.0, 20.0)  # Moderate usage
        assert scale_decision['action'] == 'none'
        
    def test_force_scaling(self):
        """Test manual scaling override."""
        config = ScalingConfig(min_workers=1, max_workers=8)
        scaler = AutoScaler(config)
        
        # Force scale to 4 workers
        scaler.force_scale(4, "Manual test scaling")
        
        assert scaler.current_workers == 4
        assert len(scaler.scaling_history) == 1
        assert scaler.scaling_history[0].event_type == 'manual'
        
    def test_scaling_rate_limiting(self):
        """Test scaling rate limiting."""
        config = ScalingConfig(
            min_workers=1,
            max_workers=8,
            max_scale_events_per_hour=2,
            stabilization_time=5.0  # Short for testing
        )
        scaler = AutoScaler(config)
        
        # Force multiple scaling operations
        scaler.force_scale(2, "Test 1")
        scaler.force_scale(3, "Test 2")
        
        # This should be rate limited
        initial_workers = scaler.current_workers
        scaler.force_scale(4, "Test 3")  # Should still work (not time-based)
        
        assert scaler.current_workers == 4  # Force scaling bypasses rate limiting
        
    def test_scaling_callbacks(self):
        """Test scaling event callbacks."""
        config = ScalingConfig(min_workers=1, max_workers=4)
        scaler = AutoScaler(config)
        
        # Add callback
        callback_events = []
        def scaling_callback(event):
            callback_events.append(event)
            
        scaler.add_scaling_callback(scaling_callback)
        
        # Trigger scaling
        scaler.force_scale(3, "Callback test")
        
        # Check callback was called
        assert len(callback_events) == 1
        assert callback_events[0].new_workers == 3


class TestQuantumAwareAutoScaler:
    """Test suite for QuantumAwareAutoScaler."""
    
    def test_quantum_scaler_creation(self):
        """Test quantum-aware auto-scaler initialization."""
        config = ScalingConfig(min_workers=1, max_workers=4)
        mock_metrics = Mock()
        
        scaler = QuantumAwareAutoScaler(config, planning_metrics=mock_metrics)
        
        assert scaler.config == config
        assert scaler.planning_metrics == mock_metrics
        assert hasattr(scaler, 'quantum_thresholds')
        
    def test_quantum_scaling_evaluation(self):
        """Test quantum-specific scaling evaluation."""
        config = ScalingConfig(min_workers=1, max_workers=4)
        mock_metrics = Mock()
        
        # Mock metrics responses
        mock_metrics.get_planning_performance.return_value = {
            'avg_planning_time': 3000.0,  # Below threshold
            'no_data': False
        }
        mock_metrics.get_quantum_metrics.return_value = {
            'avg_quantum_coherence': 0.8,  # Above critical threshold
            'quantum_advantage_score': 0.6,
            'no_data': False
        }
        
        scaler = QuantumAwareAutoScaler(config, planning_metrics=mock_metrics)
        
        # Should not trigger scaling (metrics within bounds)
        initial_workers = scaler.current_workers
        scaler._evaluate_quantum_scaling()
        
        assert scaler.current_workers == initial_workers
        
    def test_quantum_coherence_scaling(self):
        """Test scaling based on quantum coherence degradation."""
        config = ScalingConfig(min_workers=1, max_workers=4)
        mock_metrics = Mock()
        
        # Mock low coherence (should trigger scaling)
        mock_metrics.get_planning_performance.return_value = {
            'avg_planning_time': 1000.0,
            'no_data': False
        }
        mock_metrics.get_quantum_metrics.return_value = {
            'avg_quantum_coherence': 0.05,  # Below critical threshold
            'quantum_advantage_score': 0.3,
            'no_data': False
        }
        
        scaler = QuantumAwareAutoScaler(config, planning_metrics=mock_metrics)
        scaler.current_workers = 2  # Start with 2 workers
        
        # Should trigger quantum scaling
        scaler._evaluate_quantum_scaling()
        
        # Should have scaled up due to low coherence
        assert scaler.current_workers == 3
        assert len(scaler.scaling_history) == 1
        assert 'quantum' in scaler.scaling_history[0].event_type
        
    def test_quantum_metrics_tracking(self):
        """Test quantum metrics tracking in scaling history."""
        config = ScalingConfig(min_workers=1, max_workers=4)
        mock_metrics = Mock()
        
        scaler = QuantumAwareAutoScaler(config, planning_metrics=mock_metrics)
        
        # Add quantum metrics to history
        quantum_data = {
            'timestamp': time.time(),
            'planning_time': 2000.0,
            'coherence': 0.7,
            'quantum_advantage': 0.8
        }
        scaler.quantum_history.append(quantum_data)
        
        # Get quantum stats
        stats = scaler.get_quantum_scaling_stats()
        
        assert 'quantum_metrics' in stats
        assert stats['quantum_metrics']['avg_planning_time'] == 2000.0
        assert stats['quantum_metrics']['avg_coherence'] == 0.7


class TestIntegratedOptimization:
    """Integration tests for optimization components."""
    
    def test_accelerator_pool_integration(self):
        """Test integration between accelerator and resource pool."""
        # Create accelerator
        config = AccelerationConfig(enable_gpu=False)
        accelerator = QuantumAccelerator(config)
        
        # Create resource pool with accelerator
        pool = ResourcePool()
        
        # Register accelerator as compute resource
        spec = ResourceSpec(
            resource_type=ResourceType.CPU,
            capacity=accelerator.num_cores,
            capabilities={'compute', 'quantum'}
        )
        resource_id = pool.register_resource(spec)
        
        # Allocate resource for quantum computation
        allocation_id = pool.allocate_resource(
            resource_type=ResourceType.CPU,
            required_capacity=2.0,
            required_capabilities={'quantum'}
        )
        
        assert allocation_id is not None
        
        # Use accelerator for quantum operations
        matrix = np.random.rand(5, 5)
        result = accelerator.accelerated_matrix_multiply(matrix, matrix)
        
        assert result.shape == (5, 5)
        
        # Deallocate when done
        pool.deallocate_resource(allocation_id)
        
    def test_scaler_pool_integration(self):
        """Test integration between auto-scaler and resource pool."""
        # Create resource pool
        pool = ResourcePool()
        
        # Register initial resources
        for i in range(2):
            spec = ResourceSpec(
                resource_type=ResourceType.CPU,
                capacity=2.0,
                capabilities={'compute'}
            )
            pool.register_resource(spec)
            
        # Create auto-scaler with pool reference
        config = ScalingConfig(min_workers=2, max_workers=6)
        scaler = AutoScaler(config, processor_pool=pool)
        
        # Get initial stats
        initial_stats = pool.get_pool_stats()
        initial_resources = initial_stats['total_resources']
        
        # Force scaling (should trigger pool expansion in real implementation)
        scaler.force_scale(4, "Integration test")
        
        # Verify scaler tracked the change
        assert scaler.current_workers == 4
        
    def test_quantum_optimization_workflow(self):
        """Test complete quantum optimization workflow."""
        # Initialize all components
        accelerator_config = AccelerationConfig(enable_gpu=False)
        accelerator = QuantumAccelerator(accelerator_config)
        
        pool = ResourcePool()
        
        scaling_config = ScalingConfig(min_workers=1, max_workers=4)
        mock_metrics = Mock()
        scaler = QuantumAwareAutoScaler(scaling_config, planning_metrics=mock_metrics)
        
        # Register quantum-capable resources
        for i in range(2):
            spec = ResourceSpec(
                resource_type=ResourceType.CPU,
                capacity=4.0,
                capabilities={'compute', 'quantum'}
            )
            pool.register_resource(spec)
            
        # Simulate quantum computation workload
        matrices = [np.random.rand(4, 4) for _ in range(10)]
        
        # Process with accelerator
        results = []
        for matrix in matrices:
            result = accelerator.accelerated_matrix_multiply(matrix, matrix.T)
            results.append(result)
            
        # Verify all computations completed
        assert len(results) == 10
        for result in results:
            assert result.shape == (4, 4)
            
        # Check resource utilization
        stats = pool.get_pool_stats()
        assert stats['total_resources'] == 2


# Performance benchmarks
class TestOptimizationPerformance:
    """Performance benchmark tests for optimization components."""
    
    def test_accelerator_performance(self):
        """Benchmark accelerator performance."""
        config = AccelerationConfig(enable_gpu=False, enable_vectorization=True)
        accelerator = QuantumAccelerator(config)
        
        # Benchmark matrix multiplication
        sizes = [10, 50, 100]
        performance_results = []
        
        for size in sizes:
            matrix_a = np.random.rand(size, size)
            matrix_b = np.random.rand(size, size)
            
            # Time accelerated operation
            start_time = time.time()
            result = accelerator.accelerated_matrix_multiply(matrix_a, matrix_b)
            accelerated_time = time.time() - start_time
            
            # Time standard numpy operation
            start_time = time.time()
            expected = np.dot(matrix_a, matrix_b)
            numpy_time = time.time() - start_time
            
            performance_results.append({
                'size': size,
                'accelerated_time': accelerated_time,
                'numpy_time': numpy_time,
                'speedup': numpy_time / accelerated_time if accelerated_time > 0 else 1.0
            })
            
            # Verify correctness
            np.testing.assert_allclose(result, expected, rtol=1e-10)
            
        # Check that acceleration provides reasonable performance
        for result in performance_results:
            # Accelerated version should not be significantly slower
            assert result['speedup'] >= 0.1  # At least 10% of numpy performance
            
    def test_resource_pool_scalability(self):
        """Test resource pool scalability with many resources."""
        pool = ResourcePool()
        
        # Register many resources
        n_resources = 100
        resource_ids = []
        
        start_time = time.time()
        for i in range(n_resources):
            spec = ResourceSpec(
                resource_type=ResourceType.CPU,
                capacity=1.0,
                capabilities={'compute'}
            )
            resource_id = pool.register_resource(spec)
            resource_ids.append(resource_id)
            
        registration_time = time.time() - start_time
        
        # Make many allocations
        n_allocations = 200
        allocations = []
        
        start_time = time.time()
        for i in range(n_allocations):
            allocation_id = pool.allocate_resource(
                resource_type=ResourceType.CPU,
                required_capacity=0.1,
                required_capabilities={'compute'}
            )
            if allocation_id:
                allocations.append(allocation_id)
                
        allocation_time = time.time() - start_time
        
        # Performance should be reasonable
        assert registration_time < 5.0  # Should register 100 resources in < 5 seconds
        assert allocation_time < 10.0   # Should make 200 allocations in < 10 seconds
        assert len(allocations) > n_allocations * 0.8  # Should succeed on most allocations
        
        # Cleanup
        for allocation_id in allocations:
            pool.deallocate_resource(allocation_id)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])