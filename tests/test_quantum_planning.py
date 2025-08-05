"""
Comprehensive Test Suite for Quantum-Inspired Task Planning

Provides extensive testing coverage for quantum planning algorithms,
optimization components, and integration systems.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
import threading
from typing import Dict, List, Any

# Import components to test
from echoloc_nn.quantum_planning import (
    QuantumTaskPlanner, PlanningConfig, TaskGraph, Task, TaskType,
    QuantumOptimizer, PlanningMetrics, EchoLocPlanningBridge
)
from echoloc_nn.quantum_planning.planner import PlanningStrategy
from echoloc_nn.quantum_planning.optimizer import OptimizationResults, AnnealingSchedule
from echoloc_nn.optimization import QuantumAccelerator, ResourcePool, ResourceType, ResourceSpec
from echoloc_nn.utils.validation import QuantumPlanningValidator, ValidationResult
from echoloc_nn.utils.error_handling import QuantumPlanningError, ErrorHandler

class TestTaskGraph:
    """Test suite for TaskGraph functionality."""
    
    def test_task_creation(self):
        """Test basic task creation and properties."""
        task = Task(
            name="Test Task",
            task_type=TaskType.COMPUTE,
            estimated_duration=5.0,
            priority=2
        )
        
        assert task.name == "Test Task"
        assert task.task_type == TaskType.COMPUTE
        assert task.estimated_duration == 5.0
        assert task.priority == 2
        assert task.id is not None
        assert len(task.id) > 0
        
    def test_task_graph_creation(self):
        """Test task graph creation and basic operations."""
        graph = TaskGraph("Test Graph")
        
        assert graph.name == "Test Graph"
        assert len(graph.tasks) == 0
        assert len(graph.dependencies) == 0
        
    def test_add_task_to_graph(self):
        """Test adding tasks to graph."""
        graph = TaskGraph()
        task = Task(name="Task 1", estimated_duration=2.0)
        
        task_id = graph.add_task(task)
        
        assert task_id == task.id
        assert len(graph.tasks) == 1
        assert graph.has_task(task_id)
        assert graph.get_task(task_id) == task
        
    def test_add_dependency(self):
        """Test adding dependencies between tasks."""
        graph = TaskGraph()
        
        task1 = Task(name="Task 1", estimated_duration=1.0)
        task2 = Task(name="Task 2", estimated_duration=2.0)
        
        graph.add_task(task1)
        graph.add_task(task2)
        
        dependency = graph.add_dependency(task1.id, task2.id)
        
        assert dependency is not None
        assert dependency.predecessor_id == task1.id
        assert dependency.successor_id == task2.id
        assert graph.has_dependency(task1.id, task2.id)
        
    def test_cycle_detection(self):
        """Test cycle detection in task dependencies."""
        graph = TaskGraph()
        
        task1 = Task(name="Task 1")
        task2 = Task(name="Task 2")
        task3 = Task(name="Task 3")
        
        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_task(task3)
        
        # Create linear dependency chain
        graph.add_dependency(task1.id, task2.id)
        graph.add_dependency(task2.id, task3.id)
        
        # This should create a cycle
        assert graph.would_create_cycle(task3.id, task1.id)
        
        # This should not create a cycle  
        assert not graph.would_create_cycle(task1.id, task3.id)
        
    def test_ready_tasks(self):
        """Test identification of ready-to-execute tasks."""
        graph = TaskGraph()
        
        task1 = Task(name="Independent Task")
        task2 = Task(name="Dependent Task")
        task3 = Task(name="Another Independent")
        
        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_task(task3)
        
        # task2 depends on task1
        graph.add_dependency(task1.id, task2.id)
        
        ready_tasks = graph.get_ready_tasks()
        ready_ids = [t.id for t in ready_tasks]
        
        # Only task1 and task3 should be ready initially
        assert task1.id in ready_ids
        assert task3.id in ready_ids
        assert task2.id not in ready_ids
        
    def test_graph_serialization(self):
        """Test task graph serialization and deserialization."""
        original_graph = TaskGraph("Serialization Test")
        
        task1 = Task(name="Task 1", estimated_duration=1.0, priority=1)
        task2 = Task(name="Task 2", estimated_duration=2.0, priority=2)
        
        original_graph.add_task(task1)
        original_graph.add_task(task2)
        original_graph.add_dependency(task1.id, task2.id)
        
        # Serialize to dict
        serialized = original_graph.to_dict()
        
        # Deserialize
        restored_graph = TaskGraph.from_dict(serialized)
        
        assert restored_graph.name == original_graph.name
        assert len(restored_graph.tasks) == len(original_graph.tasks)
        assert len(restored_graph.dependencies) == len(original_graph.dependencies)
        
class TestQuantumOptimizer:
    """Test suite for QuantumOptimizer functionality."""
    
    def test_optimizer_creation(self):
        """Test quantum optimizer initialization."""
        optimizer = QuantumOptimizer()
        
        assert optimizer.enable_parallel == True
        assert optimizer.max_threads == 4
        assert optimizer.quantum_register is None
        
    def test_annealing_schedule(self):
        """Test annealing schedule functionality."""
        schedule = AnnealingSchedule(
            initial_temp=100.0,
            final_temp=0.1,
            schedule_type="exponential",
            max_iterations=1000
        )
        
        # Test temperature at different iterations
        temp_start = schedule.get_temperature(0)
        temp_mid = schedule.get_temperature(500)
        temp_end = schedule.get_temperature(1000)
        
        assert temp_start == 100.0
        assert temp_end <= 0.1
        assert temp_mid < temp_start and temp_mid > temp_end
        
    def test_simple_optimization(self):
        """Test basic optimization functionality."""
        optimizer = QuantumOptimizer()
        
        # Simple quadratic cost function
        def cost_function(x):
            return (x - 5) ** 2
            
        def state_generator(x):
            return x + np.random.normal(0, 0.5)
            
        schedule = AnnealingSchedule(
            initial_temp=10.0,
            final_temp=0.01,
            max_iterations=100
        )
        
        result = optimizer.quantum_annealing(
            cost_function=cost_function,
            initial_state=0.0,
            state_generator=state_generator,
            schedule=schedule
        )
        
        assert isinstance(result, OptimizationResults)
        assert result.energy >= 0
        assert len(result.energy_history) > 0
        # Should converge close to x=5 (minimum of quadratic)
        assert abs(result.assignment - 5.0) < 2.0
        
    def test_superposition_search(self):
        """Test superposition-based optimization."""
        optimizer = QuantumOptimizer()
        
        def cost_function(x):
            return x ** 2  # Minimum at x=0
            
        def state_generator(x):
            return x + np.random.uniform(-0.5, 0.5)
            
        initial_states = [1.0, -1.0, 2.0, -2.0]  # Multiple starting points
        
        result = optimizer.superposition_search(
            cost_function=cost_function,
            initial_states=initial_states,
            state_generator=state_generator,
            max_iterations=50
        )
        
        assert isinstance(result, OptimizationResults)
        assert result.superposition_states == len(initial_states)
        # Should converge close to x=0
        assert abs(result.assignment) < 1.0
        
class TestQuantumTaskPlanner:
    """Test suite for QuantumTaskPlanner functionality."""
    
    def test_planner_creation(self):
        """Test quantum task planner initialization."""
        config = PlanningConfig(
            strategy=PlanningStrategy.QUANTUM_ANNEALING,
            max_iterations=100
        )
        
        planner = QuantumTaskPlanner(config)
        
        assert planner.config == config
        assert planner.optimizer is not None
        assert planner.metrics is not None
        assert planner.current_plan is None
        
    def test_basic_planning(self):
        """Test basic task planning functionality."""
        planner = QuantumTaskPlanner()
        
        # Create simple task graph
        graph = TaskGraph("Test Planning")
        task1 = Task(name="Task 1", estimated_duration=1.0)
        task2 = Task(name="Task 2", estimated_duration=2.0)
        
        graph.add_task(task1)
        graph.add_task(task2)
        
        # Simple resources
        resources = {
            'cpu': {'type': 'compute', 'capacity': 1.0},
            'memory': {'type': 'storage', 'capacity': 1024}
        }
        
        # Plan tasks
        result = planner.plan_tasks(graph, resources)
        
        assert isinstance(result, OptimizationResults)
        assert result.assignment is not None
        assert result.energy >= 0
        assert len(result.execution_plan) > 0
        
    def test_planning_with_constraints(self):
        """Test planning with additional constraints."""
        config = PlanningConfig(
            max_iterations=50,
            energy_threshold=1e-3
        )
        planner = QuantumTaskPlanner(config)
        
        # Create task graph with dependencies
        graph = TaskGraph()
        task1 = Task(name="First", estimated_duration=1.0, priority=3)
        task2 = Task(name="Second", estimated_duration=2.0, priority=2)
        task3 = Task(name="Third", estimated_duration=1.5, priority=1)
        
        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_task(task3)
        
        # Dependencies: task1 -> task2 -> task3
        graph.add_dependency(task1.id, task2.id)
        graph.add_dependency(task2.id, task3.id)
        
        resources = {'worker': {'type': 'compute', 'capacity': 1.0}}
        constraints = {'max_time': 10.0}
        
        result = planner.plan_tasks(graph, resources, constraints)
        
        assert result is not None
        assert len(result.execution_plan) == 3
        
        # Check that dependencies are respected in execution plan
        plan_tasks = {step['task_id']: step for step in result.execution_plan}
        
        task1_step = plan_tasks[task1.id]
        task2_step = plan_tasks[task2.id]
        task3_step = plan_tasks[task3.id]
        
        # task1 should start before task2, task2 before task3
        assert task1_step['start_time'] < task2_step['start_time']
        assert task2_step['start_time'] < task3_step['start_time']
        
class TestPlanningMetrics:
    """Test suite for PlanningMetrics functionality."""
    
    def test_metrics_creation(self):
        """Test metrics system initialization."""
        metrics = PlanningMetrics(history_size=100)
        
        assert len(metrics.planning_cycles) == 0
        assert len(metrics.execution_metrics) == 0
        assert metrics.total_plans_generated == 0
        
    def test_record_planning_cycle(self):
        """Test recording planning cycle metrics."""
        metrics = PlanningMetrics()
        
        metrics.record_planning_cycle(
            planning_time=1.5,
            final_energy=10.5,
            convergence_iterations=50,
            n_tasks=3,
            n_resources=2,
            strategy_used="quantum_annealing"
        )
        
        assert len(metrics.planning_cycles) == 1
        assert metrics.total_plans_generated == 1
        
        cycle = metrics.planning_cycles[0]
        assert cycle.planning_time == 1.5
        assert cycle.final_energy == 10.5
        assert cycle.convergence_iterations == 50
        
    def test_record_execution_metrics(self):
        """Test recording task execution metrics."""
        metrics = PlanningMetrics()
        
        metrics.record_task_execution(
            task_id="task_1",
            planned_start=0.0,
            actual_start=0.1,
            planned_duration=2.0,
            actual_duration=2.2,
            resource_used="cpu",
            success=True
        )
        
        assert len(metrics.execution_metrics) == 1
        assert metrics.total_tasks_executed == 1
        
        execution = metrics.execution_metrics[0]
        assert execution.task_id == "task_1"
        assert execution.success == True
        assert execution.start_deviation == 0.1
        assert execution.duration_deviation == 0.2
        
    def test_performance_summary(self):
        """Test performance summary generation."""
        metrics = PlanningMetrics()
        
        # Record multiple planning cycles
        for i in range(5):
            metrics.record_planning_cycle(
                planning_time=1.0 + i * 0.1,
                final_energy=10.0 - i,
                convergence_iterations=100 - i * 10,
                strategy_used="quantum_annealing"
            )
            
        summary = metrics.get_planning_performance()
        
        assert 'total_cycles' in summary
        assert summary['total_cycles'] == 5
        assert 'avg_planning_time' in summary
        assert 'avg_final_energy' in summary
        assert summary['avg_planning_time'] > 0
        
class TestQuantumPlanningValidator:
    """Test suite for validation functionality."""
    
    def test_validator_creation(self):
        """Test validator initialization."""
        validator = QuantumPlanningValidator()
        
        assert validator.validation_level is not None
        assert validator.position_bounds == (-1000.0, 1000.0)
        assert validator.duration_bounds == (0.001, 86400.0)
        
    def test_task_graph_validation(self):
        """Test task graph validation."""
        validator = QuantumPlanningValidator()
        
        # Valid task graph
        graph = TaskGraph()
        task1 = Task(name="Valid Task", estimated_duration=1.0)
        graph.add_task(task1)
        
        result = validator.validate_task_graph(graph)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid == True
        assert len(result.errors) == 0
        
    def test_invalid_task_graph_validation(self):
        """Test validation of invalid task graph."""
        validator = QuantumPlanningValidator()
        
        # Create mock invalid graph
        invalid_graph = Mock()
        invalid_graph.tasks = []  # Empty tasks
        invalid_graph.dependencies = []
        
        result = validator.validate_task_graph(invalid_graph)
        
        # Should pass validation for empty graph (warning only)
        assert isinstance(result, ValidationResult)
        
class TestErrorHandling:
    """Test suite for error handling functionality."""
    
    def test_quantum_planning_error(self):
        """Test quantum planning error creation."""
        error = QuantumPlanningError(
            "Test error message",
            context={'test_key': 'test_value'}
        )
        
        assert str(error) == "Test error message"
        assert error.context['test_key'] == 'test_value'
        assert error.timestamp > 0
        
    def test_error_handler_creation(self):
        """Test error handler initialization."""
        handler = ErrorHandler(enable_auto_recovery=True)
        
        assert handler.enable_auto_recovery == True
        assert len(handler.error_history) == 0
        assert len(handler.recovery_strategies) > 0
        
    def test_error_handling_flow(self):
        """Test basic error handling flow."""
        handler = ErrorHandler(enable_auto_recovery=False)
        
        # Test handling a standard exception
        test_error = ValueError("Test validation error")
        
        with pytest.raises(ValueError):
            handler.handle_error(test_error, {'context': 'test'})
            
        # Should have recorded the error
        assert len(handler.error_history) == 1
        
class TestPerformanceOptimization:
    """Test suite for performance optimization components."""
    
    def test_quantum_accelerator_creation(self):
        """Test quantum accelerator initialization."""
        from echoloc_nn.optimization.quantum_accelerator import QuantumAccelerator, AccelerationConfig
        
        config = AccelerationConfig(enable_gpu=False)  # Disable GPU for testing
        accelerator = QuantumAccelerator(config)
        
        assert accelerator.config == config
        assert accelerator.device is not None
        assert accelerator.num_cores > 0
        
    def test_resource_pool_creation(self):
        """Test resource pool initialization."""
        pool = ResourcePool()
        
        assert pool.load_balancer is not None
        assert len(pool.resources) == 0
        assert pool.enable_auto_scaling == True
        
    def test_resource_registration(self):
        """Test resource registration in pool."""
        pool = ResourcePool()
        
        spec = ResourceSpec(
            resource_type=ResourceType.CPU,
            capacity=1.0,
            capabilities={'compute'}
        )
        
        resource_id = pool.register_resource(spec)
        
        assert resource_id is not None
        assert len(pool.resources) == 1
        assert resource_id in pool.resources
        
class TestIntegration:
    """Integration tests for complete system functionality."""
    
    def test_end_to_end_planning(self):
        """Test complete end-to-end planning workflow."""
        # Create planner
        config = PlanningConfig(
            strategy=PlanningStrategy.QUANTUM_ANNEALING,
            max_iterations=50
        )
        planner = QuantumTaskPlanner(config)
        
        # Create task graph
        graph = TaskGraph("E2E Test")
        
        # Add realistic tasks
        navigate_task = Task(
            name="Navigate to Position",
            task_type=TaskType.ACTUATOR,
            estimated_duration=3.0,
            priority=3
        )
        
        scan_task = Task(
            name="Ultrasonic Scan",
            task_type=TaskType.SENSOR,
            estimated_duration=2.0,
            priority=2
        )
        
        process_task = Task(
            name="Process Data",
            task_type=TaskType.COMPUTE,
            estimated_duration=1.0,
            priority=1
        )
        
        graph.add_task(navigate_task)
        graph.add_task(scan_task)
        graph.add_task(process_task)
        
        # Add dependencies
        graph.add_dependency(navigate_task.id, scan_task.id)
        graph.add_dependency(scan_task.id, process_task.id)
        
        # Define resources
        resources = {
            'mobile_robot': {
                'type': 'actuator',
                'position': [0, 0, 0],
                'capabilities': ['movement', 'navigation']
            },
            'ultrasonic_array': {
                'type': 'sensor',
                'position': [0, 0, 0],
                'capabilities': ['scanning', 'localization']
            },
            'edge_computer': {
                'type': 'compute',
                'cpu_cores': 4,
                'capabilities': ['processing', 'ml_inference']
            }
        }
        
        # Execute planning
        result = planner.plan_tasks(graph, resources)
        
        # Validate result
        assert result is not None
        assert result.assignment is not None
        assert len(result.execution_plan) == 3
        assert result.energy >= 0
        
        # Validate execution order respects dependencies
        execution_times = {}
        for step in result.execution_plan:
            execution_times[step['task_id']] = step['start_time']
            
        assert execution_times[navigate_task.id] < execution_times[scan_task.id]
        assert execution_times[scan_task.id] < execution_times[process_task.id]
        
    def test_metrics_integration(self):
        """Test metrics integration with planning system."""
        planner = QuantumTaskPlanner()
        
        # Create simple task graph
        graph = TaskGraph()
        task = Task(name="Test Task", estimated_duration=1.0)
        graph.add_task(task)
        
        resources = {'cpu': {'type': 'compute'}}
        
        # Execute planning (should record metrics)
        result = planner.plan_tasks(graph, resources)
        
        # Check that metrics were recorded
        metrics = planner.get_metrics()
        performance = metrics.get_planning_performance()
        
        assert not performance.get('no_data', False)
        assert performance['total_cycles'] > 0
        
    def test_validation_integration(self):
        """Test validation integration with planning workflow."""
        from echoloc_nn.utils.validation import get_global_quantum_validator
        
        validator = get_global_quantum_validator()
        planner = QuantumTaskPlanner()
        
        # Create task graph
        graph = TaskGraph()
        task = Task(name="Validated Task", estimated_duration=1.0)
        graph.add_task(task)
        
        # Validate before planning
        validation_result = validator.validate_task_graph(graph)
        assert validation_result.is_valid
        
        # Execute planning
        resources = {'cpu': {'type': 'compute'}}
        planning_result = planner.plan_tasks(graph, resources)
        
        assert planning_result is not None
        
# Benchmark tests for performance validation
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_planning_performance_benchmark(self):
        """Benchmark planning performance for various problem sizes."""
        planner = QuantumTaskPlanner(PlanningConfig(max_iterations=100))
        
        problem_sizes = [5, 10, 15]
        performance_results = []
        
        for n_tasks in problem_sizes:
            # Create graph with n_tasks
            graph = TaskGraph(f"Benchmark_{n_tasks}")
            
            for i in range(n_tasks):
                task = Task(
                    name=f"Task_{i}",
                    estimated_duration=np.random.uniform(0.5, 3.0),
                    priority=np.random.randint(1, 4)
                )
                graph.add_task(task)
                
            # Add some random dependencies
            tasks = list(graph.tasks)
            for i in range(min(n_tasks // 2, 5)):
                task1 = np.random.choice(tasks)
                task2 = np.random.choice(tasks)
                if task1.id != task2.id and not graph.has_dependency(task1.id, task2.id):
                    if not graph.would_create_cycle(task1.id, task2.id):
                        graph.add_dependency(task1.id, task2.id)
                        
            # Create resources
            resources = {
                f'resource_{i}': {'type': 'compute', 'capacity': 1.0}
                for i in range(max(2, n_tasks // 3))
            }
            
            # Benchmark planning time
            start_time = time.time()
            result = planner.plan_tasks(graph, resources)
            planning_time = time.time() - start_time
            
            performance_results.append({
                'n_tasks': n_tasks,
                'planning_time': planning_time,
                'final_energy': result.energy,
                'convergence_iterations': result.convergence
            })
            
            # Ensure reasonable performance
            assert planning_time < 10.0  # Should complete within 10 seconds
            assert result.energy >= 0
            
        # Verify that performance scales reasonably
        # (This is a simple check - more sophisticated analysis could be added)
        for i in range(1, len(performance_results)):
            current = performance_results[i]
            previous = performance_results[i-1]
            
            # Planning time should not grow exponentially
            time_ratio = current['planning_time'] / max(0.001, previous['planning_time'])
            task_ratio = current['n_tasks'] / previous['n_tasks']
            
            # Rough heuristic: time growth should be less than quadratic
            assert time_ratio < (task_ratio ** 2) * 2
            
# Fixture for common test data
@pytest.fixture
def sample_task_graph():
    """Create a sample task graph for testing."""
    graph = TaskGraph("Sample Graph")
    
    task1 = Task(name="Prepare", estimated_duration=1.0, priority=3)
    task2 = Task(name="Execute", estimated_duration=2.0, priority=2)
    task3 = Task(name="Cleanup", estimated_duration=0.5, priority=1)
    
    graph.add_task(task1)
    graph.add_task(task2)
    graph.add_task(task3)
    
    graph.add_dependency(task1.id, task2.id)
    graph.add_dependency(task2.id, task3.id)
    
    return graph
    
@pytest.fixture
def sample_resources():
    """Create sample resources for testing."""
    return {
        'cpu': {'type': 'compute', 'capacity': 1.0},
        'memory': {'type': 'storage', 'capacity': 1024},
        'sensor': {'type': 'sensor', 'position': [0, 0, 0]}
    }
    
# Parameterized tests for different strategies
@pytest.mark.parametrize("strategy", [
    PlanningStrategy.QUANTUM_ANNEALING,
    PlanningStrategy.SUPERPOSITION_SEARCH,
    PlanningStrategy.HYBRID_CLASSICAL,
    PlanningStrategy.ADAPTIVE
])
def test_planning_strategies(strategy, sample_task_graph, sample_resources):
    """Test different planning strategies."""
    config = PlanningConfig(
        strategy=strategy,
        max_iterations=50  # Reduce for faster testing
    )
    
    planner = QuantumTaskPlanner(config)
    result = planner.plan_tasks(sample_task_graph, sample_resources)
    
    assert result is not None
    assert result.assignment is not None
    assert len(result.execution_plan) == 3
    assert result.energy >= 0
    
# Stress tests
class TestStressScenarios:
    """Stress tests for edge cases and high load scenarios."""
    
    def test_large_task_graph(self):
        """Test planning with large task graph."""
        n_tasks = 50
        graph = TaskGraph("Large Graph")
        
        # Create tasks
        tasks = []
        for i in range(n_tasks):
            task = Task(
                name=f"Task_{i}",
                estimated_duration=np.random.uniform(0.1, 2.0),
                priority=np.random.randint(1, 6)
            )
            tasks.append(task)
            graph.add_task(task)
            
        # Add random dependencies (avoid cycles)
        for i in range(n_tasks // 3):
            task1_idx = np.random.randint(0, n_tasks // 2)
            task2_idx = np.random.randint(n_tasks // 2, n_tasks)
            
            task1 = tasks[task1_idx]
            task2 = tasks[task2_idx]
            
            if not graph.has_dependency(task1.id, task2.id):
                graph.add_dependency(task1.id, task2.id)
                
        # Create resources
        resources = {
            f'worker_{i}': {'type': 'compute', 'capacity': 1.0}
            for i in range(10)
        }
        
        # Plan with reduced iterations for performance
        config = PlanningConfig(max_iterations=100)
        planner = QuantumTaskPlanner(config)
        
        start_time = time.time()
        result = planner.plan_tasks(graph, resources)
        planning_time = time.time() - start_time
        
        assert result is not None
        assert len(result.execution_plan) == n_tasks
        assert planning_time < 30.0  # Should complete within 30 seconds
        
    def test_resource_constraints(self):
        """Test planning under severe resource constraints."""
        # Many tasks, few resources
        graph = TaskGraph("Constrained")
        
        for i in range(20):
            task = Task(name=f"Task_{i}", estimated_duration=1.0)
            graph.add_task(task)
            
        # Only one resource
        resources = {'single_cpu': {'type': 'compute', 'capacity': 1.0}}
        
        planner = QuantumTaskPlanner()
        result = planner.plan_tasks(graph, resources)
        
        assert result is not None
        # All tasks should be assigned to the single resource
        resource_assignments = [step['resource'] for step in result.execution_plan]
        assert all(res == 'single_cpu' for res in resource_assignments)
        
    def test_concurrent_planning(self):
        """Test concurrent planning operations."""
        def plan_task_graph(graph_id):
            graph = TaskGraph(f"Concurrent_{graph_id}")
            
            for i in range(5):
                task = Task(name=f"Task_{graph_id}_{i}", estimated_duration=1.0)
                graph.add_task(task)
                
            resources = {'cpu': {'type': 'compute'}}
            
            planner = QuantumTaskPlanner(PlanningConfig(max_iterations=50))
            return planner.plan_tasks(graph, resources)
            
        # Run multiple planning operations concurrently
        threads = []
        results = []
        
        def worker(graph_id):
            result = plan_task_graph(graph_id)
            results.append(result)
            
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join(timeout=10.0)
            
        # All planning operations should succeed
        assert len(results) == 3
        for result in results:
            assert result is not None
            assert len(result.execution_plan) == 5
            
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])