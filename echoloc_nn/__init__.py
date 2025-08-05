"""
EchoLoc-NN: End-to-end ultrasonic localization toolkit

This package provides CNN-Transformer hybrid models for processing ultrasonic
chirp echoes to enable GPS-denied indoor positioning with commodity hardware.

Main Components:
- models: Neural network architectures for echo processing
- signal_processing: Chirp generation and echo enhancement
- hardware: Ultrasonic array interfaces and Arduino/RPi integration
- training: Model training pipelines and physics simulation
- inference: Real-time localization engine
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core imports for user convenience
from .models import EchoLocModel, CNNTransformerHybrid
from .inference import EchoLocator, InferenceConfig
from .hardware import UltrasonicArray, SensorConfig
from .signal_processing import ChirpGenerator, EchoProcessor
from .training import EchoLocTrainer, TrainingConfig, EchoSimulator

# Quantum Planning imports
from .quantum_planning import (
    QuantumTaskPlanner, PlanningConfig, TaskGraph, Task, 
    QuantumOptimizer, PlanningMetrics, EchoLocPlanningBridge
)

# Quick start functions
def create_locator(model_path=None, device="auto", **kwargs):
    """
    Create a ready-to-use EchoLocator instance.
    
    Args:
        model_path: Path to trained model (optional)
        device: Computing device ("auto", "cpu", "cuda")
        **kwargs: Additional configuration options
        
    Returns:
        Configured EchoLocator instance
    """
    config = InferenceConfig(model_path=model_path, device=device, **kwargs)
    return EchoLocator(config=config)

def create_square_array(spacing=0.1, port=None):
    """
    Create a 4-sensor square array configuration.
    
    Args:
        spacing: Sensor spacing in meters
        port: Serial port for hardware connection
        
    Returns:
        Configured UltrasonicArray instance
    """
    array = UltrasonicArray.create_square_array(spacing=spacing)
    if port:
        array.port = port
    return array

def quantum_planning_demo():
    """
    Run a quantum-inspired task planning demonstration.
    
    Shows integration between EchoLoc positioning and task planning.
    """
    import numpy as np
    from .quantum_planning import QuantumTaskPlanner, TaskGraph, Task, TaskType
    from .signal_processing import ChirpGenerator
    
    print("EchoLoc-NN Quantum Planning Demo")
    print("================================")
    
    # Create components
    print("1. Creating quantum task planner...")
    planner = QuantumTaskPlanner()
    
    # Create sample task graph
    print("2. Creating task graph...")
    graph = TaskGraph("Demo Tasks")
    
    # Add sample tasks
    task1 = Task(name="Navigate to Position A", task_type=TaskType.ACTUATOR, 
                estimated_duration=5.0, priority=3)
    task1.parameters = {'target_position': [2.0, 3.0, 0.0], 'requires_movement': True}
    
    task2 = Task(name="Ultrasonic Scan", task_type=TaskType.SENSOR,
                estimated_duration=3.0, priority=2)
    task2.parameters = {'target_position': [2.0, 3.0, 0.0]}
    
    task3 = Task(name="Data Processing", task_type=TaskType.COMPUTE,
                estimated_duration=2.0, priority=1)
    
    graph.add_task(task1)
    graph.add_task(task2) 
    graph.add_task(task3)
    
    # Add dependencies
    graph.add_dependency(task1.id, task2.id)  # Navigate before scan
    graph.add_dependency(task2.id, task3.id)  # Scan before processing
    
    # Create resources
    print("3. Setting up resources...")
    resources = {
        'mobile_robot': {'type': 'actuator', 'position': [0, 0, 0]},
        'ultrasonic_array': {'type': 'sensor', 'position': [0, 0, 0]},
        'edge_computer': {'type': 'compute', 'cpu_cores': 4}
    }
    
    # Execute quantum planning
    print("4. Running quantum-inspired optimization...")
    result = planner.plan_tasks(graph, resources)
    
    print(f"5. Results:")
    print(f"   Optimization energy: {result.energy:.2f}")
    print(f"   Convergence iterations: {result.convergence}")
    print(f"   Planning time: {result.optimization_time:.3f}s")
    print(f"   Tasks in execution plan: {len(result.execution_plan)}")
    
    print("\n   Execution sequence:")
    for i, step in enumerate(result.execution_plan):
        print(f"   {i+1}. {step['task_name']} -> {step['resource']} "
              f"(t={step['start_time']:.1f}s, d={step['duration']:.1f}s)")
    
    print("\n✓ Quantum planning demo completed!")
    return result

def quick_demo():
    """
    Run a quick demonstration of the system.
    
    Shows basic usage with synthetic data and quantum planning.
    """
    import numpy as np
    from .signal_processing import ChirpGenerator
    
    print("EchoLoc-NN Quick Demo")
    print("===================")
    
    # Create components
    print("1. Creating localization system...")
    locator = create_locator()
    array = create_square_array()
    chirp_gen = ChirpGenerator()
    
    # Generate synthetic echo data
    print("2. Generating synthetic echo data...")
    t, chirp = chirp_gen.generate_lfm_chirp(35000, 45000, 0.005)
    
    # Simulate echo (simplified)
    echo_data = np.random.randn(4, 2048) * 0.1  # 4 sensors, 2048 samples
    
    # Add delayed chirp echo
    delay_samples = 500
    for i in range(4):
        echo_data[i, delay_samples:delay_samples+len(chirp)] += chirp * (0.5 + i*0.1)
    
    # Localize
    print("3. Performing localization...")
    position, confidence = locator.locate(echo_data)
    
    print(f"4. Results:")
    print(f"   Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) meters")
    print(f"   Confidence: {confidence:.2%}")
    
    # Performance stats
    stats = locator.get_performance_stats()
    if not stats.get('no_data', False):
        print(f"   Inference time: {stats['avg_inference_time_ms']:.1f} ms")
    
    print("\n5. Running quantum planning demonstration...")
    planning_result = quantum_planning_demo()
    
    print("\n✓ Complete demo finished successfully!")
    print("  See examples/ directory for more detailed usage")

# Make key classes available at package level
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "EchoLocModel",
    "CNNTransformerHybrid",
    "EchoLocator",
    "InferenceConfig",
    "UltrasonicArray",
    "SensorConfig",
    "ChirpGenerator",
    "EchoProcessor",
    "EchoLocTrainer",
    "TrainingConfig",
    "EchoSimulator",
    # Quantum Planning exports
    "QuantumTaskPlanner",
    "PlanningConfig",
    "TaskGraph",
    "Task",
    "QuantumOptimizer",
    "PlanningMetrics",
    "EchoLocPlanningBridge",
    # Demo functions
    "create_locator",
    "create_square_array",
    "quick_demo",
    "quantum_planning_demo"
]