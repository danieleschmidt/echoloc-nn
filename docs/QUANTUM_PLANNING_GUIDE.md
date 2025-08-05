# Quantum-Inspired Task Planning Guide

## Overview

The EchoLoc-NN Quantum Planning system provides advanced task scheduling and resource allocation using quantum-inspired optimization algorithms. This guide covers the core concepts, implementation details, and usage patterns.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Quick Start](#quick-start)
3. [Task Graph Construction](#task-graph-construction)
4. [Optimization Strategies](#optimization-strategies)
5. [Resource Management](#resource-management)
6. [Performance Tuning](#performance-tuning)
7. [Integration Patterns](#integration-patterns)
8. [Advanced Features](#advanced-features)

## Core Concepts

### Quantum-Inspired Optimization

The system uses quantum computing principles adapted for classical computers:

- **Superposition**: Multiple solution candidates explored simultaneously
- **Entanglement**: Task dependencies modeled as quantum correlations
- **Tunneling**: Ability to escape local optima through quantum effects
- **Measurement**: Probabilistic collapse to optimal solution

### Task Representation

Tasks are modeled with quantum properties:

```python
from echoloc_nn.quantum_planning import Task, TaskType

task = Task(
    name="Navigate to Position A",
    task_type=TaskType.ACTUATOR,
    estimated_duration=5.0,
    priority=3,
    # Quantum properties
    superposition_weight=1.0,      # Weight in quantum superposition
    entanglement_strength=0.8,     # Correlation with dependent tasks
    measurement_probability=0.95   # Success probability
)
```

### Energy Function

The system minimizes a composite energy function:

```
E = α·makespan + β·resource_variance + γ·dependency_violations + δ·position_penalty
```

Where:
- **makespan**: Total execution time
- **resource_variance**: Load balancing metric
- **dependency_violations**: Constraint satisfaction
- **position_penalty**: Spatial optimization (with EchoLoc integration)

## Quick Start

### Basic Usage

```python
from echoloc_nn.quantum_planning import (
    QuantumTaskPlanner, PlanningConfig, TaskGraph, Task, TaskType
)

# Create planner
config = PlanningConfig(
    strategy=PlanningStrategy.QUANTUM_ANNEALING,
    max_iterations=1000,
    initial_temperature=100.0,
    final_temperature=0.01
)
planner = QuantumTaskPlanner(config)

# Create task graph
graph = TaskGraph("Example Tasks")

# Add tasks
nav_task = Task("Navigate", TaskType.ACTUATOR, estimated_duration=3.0, priority=3)
scan_task = Task("Scan Area", TaskType.SENSOR, estimated_duration=2.0, priority=2)
process_task = Task("Process Data", TaskType.COMPUTE, estimated_duration=1.0, priority=1)

graph.add_task(nav_task)
graph.add_task(scan_task)
graph.add_task(process_task)

# Add dependencies
graph.add_dependency(nav_task.id, scan_task.id)
graph.add_dependency(scan_task.id, process_task.id)

# Define resources
resources = {
    'robot': {'type': 'actuator', 'capacity': 1.0, 'position': [0, 0, 0]},
    'sensor_array': {'type': 'sensor', 'capacity': 1.0, 'position': [0, 0, 0]},
    'edge_computer': {'type': 'compute', 'capacity': 4.0}
}

# Execute planning
result = planner.plan_tasks(graph, resources)

# Access results
print(f"Optimization energy: {result.energy:.2f}")
print(f"Convergence iterations: {result.convergence}")
print(f"Execution plan: {len(result.execution_plan)} steps")

for step in result.execution_plan:
    print(f"  {step['task_name']} -> {step['resource']} "
          f"(t={step['start_time']:.1f}s)")
```

### Integration with EchoLoc Positioning

```python
from echoloc_nn import create_locator
from echoloc_nn.quantum_planning import EchoLocPlanningBridge

# Create localization system
locator = create_locator()

# Create planning bridge
bridge = EchoLocPlanningBridge(planner, locator)

# Enable position-aware planning
planner.config.use_position_feedback = True

# Planning will now consider physical positions
result = bridge.plan_with_positioning(graph, resources)
```

## Task Graph Construction

### Task Types

The system supports different task categories:

```python
class TaskType(Enum):
    COMPUTE = "compute"     # CPU-intensive processing
    IO = "io"              # File/network operations
    NETWORK = "network"    # Communication tasks
    SENSOR = "sensor"      # Data collection
    ACTUATOR = "actuator"  # Physical actions
    HYBRID = "hybrid"      # Multi-modal tasks
```

### Dependencies

Model task relationships with different dependency types:

```python
# Finish-to-start (default)
graph.add_dependency(task1.id, task2.id, "finish_to_start")

# Start-to-start (parallel execution)
graph.add_dependency(task1.id, task2.id, "start_to_start")

# With delay constraints
graph.add_dependency(task1.id, task2.id, delay=2.0)  # 2-second delay
```

### Resource Requirements

Specify detailed resource needs:

```python
task.add_resource_requirement("cpu", quantity=2.0, duration=5.0)
task.add_resource_requirement("memory", quantity=1024, duration=5.0)
task.set_position_requirement(x=2.0, y=3.0, z=0.0)  # Physical location
```

### Validation

Ensure graph consistency:

```python
from echoloc_nn.utils.validation import get_global_quantum_validator

validator = get_global_quantum_validator()
result = validator.validate_task_graph(graph)

if not result.is_valid:
    print("Validation errors:", result.errors)
else:
    print("Task graph is valid")
```

## Optimization Strategies

### Quantum Annealing

Best for complex optimization landscapes:

```python
config = PlanningConfig(
    strategy=PlanningStrategy.QUANTUM_ANNEALING,
    max_iterations=1000,
    initial_temperature=100.0,
    final_temperature=0.01,
    quantum_tunneling_rate=0.1,  # Tunneling probability
    temperature_schedule="exponential"
)
```

**Advantages:**
- Escapes local optima through tunneling
- Handles complex constraint landscapes
- Proven convergence properties

**Best for:**
- Large task graphs (20+ tasks)
- Complex resource constraints
- Multi-objective optimization

### Superposition Search

Explores multiple solutions simultaneously:

```python
config = PlanningConfig(
    strategy=PlanningStrategy.SUPERPOSITION_SEARCH,
    max_iterations=500,
    measurement_collapse_threshold=0.8,  # Keep top 80%
    enable_superposition=True
)
```

**Advantages:**
- Parallel exploration of solution space
- Natural diversity preservation
- Fast convergence for simple problems

**Best for:**
- Small to medium task graphs (5-15 tasks)
- Well-defined optimization landscapes
- Real-time planning requirements

### Hybrid Classical

Combines quantum and classical methods:

```python
config = PlanningConfig(
    strategy=PlanningStrategy.HYBRID_CLASSICAL,
    max_iterations=200,
    adaptive_learning_rate=0.01
)
```

**Process:**
1. Quantum annealing for global exploration
2. Classical hill climbing for local refinement
3. Iterative improvement

### Adaptive Strategy

Automatically selects the best approach:

```python
config = PlanningConfig(
    strategy=PlanningStrategy.ADAPTIVE,
    # System chooses based on problem characteristics
)
```

**Selection Criteria:**
- Problem size (number of tasks/resources)
- Constraint complexity
- Available computation time

## Resource Management

### Resource Pool

Dynamic resource allocation with load balancing:

```python
from echoloc_nn.optimization import ResourcePool, ResourceSpec, ResourceType

pool = ResourcePool()

# Register resources
cpu_spec = ResourceSpec(
    resource_type=ResourceType.CPU,
    capacity=4.0,
    capabilities={'compute', 'parallel'}
)
pool.register_resource(cpu_spec)

# Allocate resources
allocation_id = pool.allocate_resource(
    resource_type=ResourceType.CPU,
    required_capacity=2.0,
    required_capabilities={'compute'}
)

# Monitor utilization
stats = pool.get_pool_stats()
print(f"Pool utilization: {stats['avg_utilization']:.1%}")
```

### Auto-Scaling

Automatic resource scaling based on demand:

```python
from echoloc_nn.optimization import AutoScaler, ScalingConfig

scaling_config = ScalingConfig(
    min_workers=1,
    max_workers=8,
    cpu_scale_up_threshold=70.0,
    cpu_scale_down_threshold=30.0
)

scaler = AutoScaler(scaling_config, processor_pool=pool)
scaler.start()

# Add scaling callback
def on_scaling_event(event):
    print(f"Scaled {event.old_workers} -> {event.new_workers} workers")

scaler.add_scaling_callback(on_scaling_event)
```

### Quantum-Aware Scaling

Advanced scaling with quantum metrics:

```python
from echoloc_nn.optimization import QuantumAwareAutoScaler

quantum_scaler = QuantumAwareAutoScaler(
    scaling_config, 
    processor_pool=pool,
    planning_metrics=planner.get_metrics()
)

# Scales based on:
# - Traditional CPU/memory metrics
# - Quantum planning performance
# - Coherence degradation
# - Algorithm convergence rates
```

## Performance Tuning

### Configuration Optimization

Tune parameters for your workload:

```python
# For real-time systems
config = PlanningConfig(
    max_iterations=100,        # Reduce for faster planning
    energy_threshold=1e-3,     # Early termination
    parallel_threads=4         # Use available cores
)

# For high-quality solutions
config = PlanningConfig(
    max_iterations=2000,       # More exploration
    initial_temperature=200.0,  # Higher initial energy
    quantum_tunneling_rate=0.2  # More tunneling
)
```

### Caching and Acceleration

Enable performance optimizations:

```python
from echoloc_nn.optimization import QuantumAccelerator, AccelerationConfig

accel_config = AccelerationConfig(
    enable_gpu=True,           # Use GPU acceleration
    enable_vectorization=True, # SIMD operations
    enable_cache=True,         # Memoization
    max_cache_size=1000
)

accelerator = QuantumAccelerator(accel_config)
planner.set_accelerator(accelerator)
```

### Monitoring and Metrics

Track performance metrics:

```python
# Get planning metrics
metrics = planner.get_metrics()
perf = metrics.get_planning_performance()

print(f"Average planning time: {perf['avg_planning_time']:.1f}ms")
print(f"Average final energy: {perf['avg_final_energy']:.3f}")
print(f"Convergence rate: {perf['avg_convergence_iterations']:.0f}")

# Get quantum-specific metrics
quantum_metrics = metrics.get_quantum_metrics()
print(f"Quantum coherence: {quantum_metrics['avg_quantum_coherence']:.2f}")
print(f"Tunneling events: {quantum_metrics['total_tunneling_events']}")
```

## Integration Patterns

### REST API Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
planner = QuantumTaskPlanner()

@app.route('/api/v1/planning/optimize', methods=['POST'])
def optimize_tasks():
    data = request.json
    
    # Build task graph from request
    graph = TaskGraph.from_dict(data['task_graph'])
    resources = data['resources']
    constraints = data.get('constraints', {})
    
    # Execute planning
    result = planner.plan_tasks(graph, resources, constraints)
    
    return jsonify({
        'status': 'success',
        'energy': result.energy,
        'execution_plan': result.execution_plan,
        'metrics': {
            'planning_time': result.optimization_time,
            'convergence': result.convergence
        }
    })
```

### Message Queue Integration

```python
import asyncio
from echoloc_nn.quantum_planning import AsyncQuantumPlanner

async def process_planning_requests():
    planner = AsyncQuantumPlanner()
    
    async for message in message_queue.consume('planning_requests'):
        try:
            graph = TaskGraph.from_dict(message['task_graph'])
            resources = message['resources']
            
            result = await planner.plan_tasks_async(graph, resources)
            
            await message_queue.publish('planning_results', {
                'request_id': message['request_id'],
                'result': result.to_dict()
            })
            
        except Exception as e:
            await message_queue.publish('planning_errors', {
                'request_id': message['request_id'],
                'error': str(e)
            })
```

### Database Integration

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class PlanningResult(Base):
    __tablename__ = 'planning_results'
    
    id = Column(Integer, primary_key=True)
    task_graph_id = Column(String(50))
    energy = Column(Float)
    convergence_iterations = Column(Integer)
    execution_plan = Column(Text)  # JSON serialized
    planning_time_ms = Column(Float)

# Save results
def save_planning_result(result, task_graph_id):
    session = SessionLocal()
    
    db_result = PlanningResult(
        task_graph_id=task_graph_id,
        energy=result.energy,
        convergence_iterations=result.convergence,
        execution_plan=json.dumps(result.execution_plan),
        planning_time_ms=result.optimization_time * 1000
    )
    
    session.add(db_result)
    session.commit()
    session.close()
```

## Advanced Features

### Custom Optimization Objectives

Define custom energy functions:

```python
class CustomQuantumPlanner(QuantumTaskPlanner):
    def _calculate_system_energy(self, assignment, task_graph, resources, constraints):
        base_energy = super()._calculate_system_energy(assignment, task_graph, resources, constraints)
        
        # Add custom objectives
        energy_efficiency = self._calculate_energy_efficiency(assignment)
        user_satisfaction = self._calculate_user_satisfaction(assignment)
        
        return base_energy + 0.1 * energy_efficiency + 0.2 * user_satisfaction
    
    def _calculate_energy_efficiency(self, assignment):
        # Custom energy efficiency calculation
        return sum(task['duration'] * task['priority'] for task in assignment.values())
    
    def _calculate_user_satisfaction(self, assignment):
        # Custom user satisfaction metric
        total_wait_time = max(task['start_time'] for task in assignment.values())
        return total_wait_time * 0.1
```

### Multi-Objective Optimization

Optimize multiple objectives simultaneously:

```python
from echoloc_nn.quantum_planning import MultiObjectiveConfig

multi_config = MultiObjectiveConfig(
    objectives=[
        ('makespan', 0.4),      # 40% weight on completion time
        ('resource_balance', 0.3), # 30% weight on load balancing
        ('energy_efficiency', 0.2), # 20% weight on energy
        ('user_satisfaction', 0.1)  # 10% weight on satisfaction
    ],
    pareto_optimization=True    # Find Pareto optimal solutions
)

planner = QuantumTaskPlanner(multi_config)
```

### Reinforcement Learning Integration

Adaptive strategy learning:

```python
from echoloc_nn.quantum_planning import RLQuantumPlanner

rl_planner = RLQuantumPlanner(
    base_config=config,
    learning_rate=0.001,
    exploration_rate=0.1,
    memory_size=10000
)

# Learn from planning experiences
for episode in range(1000):
    graph = generate_random_task_graph()
    resources = generate_random_resources()
    
    result = rl_planner.plan_tasks_with_learning(graph, resources)
    
    # System learns which strategies work best for different problem types
```

### Distributed Planning

Scale across multiple nodes:

```python
from echoloc_nn.quantum_planning import DistributedQuantumPlanner

distributed_planner = DistributedQuantumPlanner(
    nodes=['node1:8080', 'node2:8080', 'node3:8080'],
    coordination_strategy='consensus',
    fault_tolerance=True
)

# Automatically distributes planning work across nodes
result = distributed_planner.plan_tasks(large_task_graph, resources)
```

## Best Practices

### Graph Design

1. **Keep graphs connected**: Ensure all tasks have paths to completion
2. **Balance priorities**: Use meaningful priority values (1-10 scale)
3. **Realistic durations**: Estimate task durations accurately
4. **Resource requirements**: Specify precise resource needs

### Performance Optimization

1. **Start simple**: Begin with basic configurations, then optimize
2. **Profile regularly**: Monitor planning times and convergence
3. **Cache results**: Reuse solutions for similar problems
4. **Batch similar tasks**: Group related work items

### Error Handling

1. **Validate inputs**: Always validate task graphs before planning
2. **Handle failures gracefully**: Implement retry logic with backoff
3. **Monitor resource health**: Check resource availability
4. **Log security events**: Track unauthorized access attempts

### Testing

1. **Unit test components**: Test individual algorithms thoroughly
2. **Integration testing**: Verify end-to-end workflows
3. **Performance testing**: Benchmark different configurations
4. **Stress testing**: Test with large graphs and resource constraints

This guide provides comprehensive coverage of the Quantum-Inspired Task Planning system. For additional examples and tutorials, see the `examples/` directory in the repository.