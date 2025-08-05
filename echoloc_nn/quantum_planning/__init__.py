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

from .planner import QuantumTaskPlanner, PlanningConfig
from .task_graph import TaskGraph, Task, TaskDependency
from .optimizer import QuantumOptimizer, AnnealingSchedule
from .metrics import PlanningMetrics, OptimizationResults
from .integration import EchoLocPlanningBridge

__all__ = [
    'QuantumTaskPlanner',
    'PlanningConfig', 
    'TaskGraph',
    'Task',
    'TaskDependency',
    'QuantumOptimizer',
    'AnnealingSchedule',
    'PlanningMetrics',
    'OptimizationResults',
    'EchoLocPlanningBridge'
]