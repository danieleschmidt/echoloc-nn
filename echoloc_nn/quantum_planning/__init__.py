"""
Quantum-Inspired Task Planning Module

This module implements quantum-inspired optimization algorithms for task planning
and scheduling, integrated with the EchoLoc-NN positioning system.

Key Components:
- QuantumTaskPlanner: Main planning engine using quantum annealing principles
- TaskGraph: DAG representation of tasks with dependencies
- QuantumOptimizer: Optimization using quantum-inspired algorithms
- PlanningMetrics: Performance evaluation and monitoring
"""

try:
    from .planner import QuantumTaskPlanner, PlanningConfig
    from .task_graph import TaskGraph, Task, TaskDependency, TaskType
    from .optimizer import QuantumOptimizer, AnnealingSchedule
    from .metrics import PlanningMetrics
    from .optimizer import OptimizationResults
    from .integration import EchoLocPlanningBridge
except ImportError:
    # Quantum planning modules not available without dependencies
    QuantumTaskPlanner = None
    PlanningConfig = None
    TaskGraph = None
    Task = None
    TaskDependency = None
    TaskType = None
    QuantumOptimizer = None
    AnnealingSchedule = None
    PlanningMetrics = None
    OptimizationResults = None
    EchoLocPlanningBridge = None

__all__ = [
    'QuantumTaskPlanner',
    'PlanningConfig', 
    'TaskGraph',
    'Task',
    'TaskDependency',
    'TaskType',
    'QuantumOptimizer',
    'AnnealingSchedule',
    'PlanningMetrics',
    'OptimizationResults',
    'EchoLocPlanningBridge'
]