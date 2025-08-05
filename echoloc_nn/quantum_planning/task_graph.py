"""
Task Graph Implementation

Provides DAG (Directed Acyclic Graph) representation of tasks with dependencies,
resource requirements, and scheduling constraints.
"""

import uuid
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import numpy as np
from datetime import datetime, timedelta

class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskType(Enum):
    COMPUTE = "compute"
    IO = "io"
    NETWORK = "network"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    HYBRID = "hybrid"

@dataclass
class ResourceRequirement:
    """Resource requirement specification for a task."""
    resource_type: str
    quantity: float
    duration: float
    exclusive: bool = False
    location: Optional[Tuple[float, float, float]] = None  # (x, y, z) position requirement
    
@dataclass
class Task:
    """Individual task in the task graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    task_type: TaskType = TaskType.COMPUTE
    estimated_duration: float = 1.0  # seconds
    priority: int = 1  # Higher number = higher priority
    deadline: Optional[datetime] = None
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Quantum-inspired properties
    superposition_weight: float = 1.0  # Weight in quantum superposition
    entanglement_strength: float = 0.0  # Strength of entanglement with other tasks
    measurement_probability: float = 1.0  # Probability of successful measurement/execution
    
    def __post_init__(self):
        if not self.name:
            self.name = f"Task_{self.id[:8]}"
            
    def add_resource_requirement(self, resource_type: str, quantity: float, duration: float = None):
        """Add a resource requirement to this task."""
        if duration is None:
            duration = self.estimated_duration
            
        req = ResourceRequirement(
            resource_type=resource_type,
            quantity=quantity,
            duration=duration
        )
        self.resource_requirements.append(req)
        
    def set_position_requirement(self, x: float, y: float, z: float = 0.0):
        """Set position requirement for this task."""
        for req in self.resource_requirements:
            req.location = (x, y, z)
            
    def update_status(self, new_status: TaskStatus, error_message: str = None):
        """Update task status with timestamp tracking."""
        self.status = new_status
        
        if new_status == TaskStatus.RUNNING and self.started_at is None:
            self.started_at = datetime.now()
        elif new_status == TaskStatus.COMPLETED and self.completed_at is None:
            self.completed_at = datetime.now()
        elif new_status == TaskStatus.FAILED:
            self.error_message = error_message
            
    def get_actual_duration(self) -> Optional[float]:
        """Get actual execution duration if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
        
    def is_overdue(self) -> bool:
        """Check if task is overdue based on deadline."""
        if self.deadline is None:
            return False
        return datetime.now() > self.deadline
        
@dataclass
class TaskDependency:
    """Dependency relationship between tasks."""
    predecessor_id: str
    successor_id: str
    dependency_type: str = "finish_to_start"  # finish_to_start, start_to_start, etc.
    delay: float = 0.0  # Minimum delay between tasks (seconds)
    
    # Quantum-inspired properties
    entanglement_coefficient: complex = complex(1.0, 0.0)  # Complex coupling strength
    coherence_time: float = float('inf')  # Time before decoherence
    
class TaskGraph:
    """
    Directed Acyclic Graph (DAG) representation of tasks with dependencies.
    
    Provides quantum-inspired task scheduling with superposition and entanglement
    concepts for optimal resource allocation and execution planning.
    """
    
    def __init__(self, name: str = "TaskGraph"):
        self.name = name
        self.tasks: List[Task] = []
        self.dependencies: List[TaskDependency] = []
        self.graph = nx.DiGraph()
        self.metadata = {}
        
        # Quantum state tracking
        self.quantum_coherence = True
        self.superposition_active = False
        self.entanglement_matrix = None
        
    def add_task(self, task: Task) -> str:
        """Add a task to the graph."""
        if any(t.id == task.id for t in self.tasks):
            raise ValueError(f"Task with ID {task.id} already exists")
            
        self.tasks.append(task)
        self.graph.add_node(task.id, task=task)
        
        # Update entanglement matrix if needed
        self._update_entanglement_matrix()
        
        return task.id
        
    def add_dependency(self, predecessor_id: str, successor_id: str, 
                      dependency_type: str = "finish_to_start", delay: float = 0.0) -> TaskDependency:
        """Add a dependency between two tasks."""
        # Validate tasks exist
        if not self.has_task(predecessor_id):
            raise ValueError(f"Predecessor task {predecessor_id} not found")
        if not self.has_task(successor_id):
            raise ValueError(f"Successor task {successor_id} not found")
            
        # Check for cycles
        if self.would_create_cycle(predecessor_id, successor_id):
            raise ValueError(f"Adding dependency {predecessor_id} -> {successor_id} would create a cycle")
            
        dependency = TaskDependency(
            predecessor_id=predecessor_id,
            successor_id=successor_id,
            dependency_type=dependency_type,
            delay=delay
        )
        
        self.dependencies.append(dependency)
        self.graph.add_edge(predecessor_id, successor_id, dependency=dependency)
        
        # Update quantum entanglement
        self._update_task_entanglement(predecessor_id, successor_id)
        
        return dependency
        
    def remove_task(self, task_id: str):
        """Remove a task and all its dependencies."""
        if not self.has_task(task_id):
            raise ValueError(f"Task {task_id} not found")
            
        # Remove dependencies
        self.dependencies = [d for d in self.dependencies 
                           if d.predecessor_id != task_id and d.successor_id != task_id]
        
        # Remove from tasks list
        self.tasks = [t for t in self.tasks if t.id != task_id]
        
        # Remove from graph
        self.graph.remove_node(task_id)
        
        self._update_entanglement_matrix()
        
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
        
    def has_task(self, task_id: str) -> bool:
        """Check if task exists in graph."""
        return any(t.id == task_id for t in self.tasks)
        
    def has_dependency(self, predecessor_id: str, successor_id: str) -> bool:
        """Check if direct dependency exists."""
        return any(d.predecessor_id == predecessor_id and d.successor_id == successor_id 
                  for d in self.dependencies)
        
    def has_transitive_dependency(self, predecessor_id: str, successor_id: str) -> bool:
        """Check if transitive dependency exists (path through other tasks)."""
        return nx.has_path(self.graph, predecessor_id, successor_id)
        
    def would_create_cycle(self, predecessor_id: str, successor_id: str) -> bool:
        """Check if adding dependency would create a cycle."""
        # Temporarily add edge and check for cycles
        temp_graph = self.graph.copy()
        temp_graph.add_edge(predecessor_id, successor_id)
        return not nx.is_directed_acyclic_graph(temp_graph)
        
    def get_dependencies(self, task_id: str) -> List[str]:
        """Get list of direct predecessor task IDs."""
        return [d.predecessor_id for d in self.dependencies if d.successor_id == task_id]
        
    def get_dependents(self, task_id: str) -> List[str]:
        """Get list of direct successor task IDs."""
        return [d.successor_id for d in self.dependencies if d.predecessor_id == task_id]
        
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready_tasks = []
        
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
                
            dependencies = self.get_dependencies(task.id)
            if not dependencies:  # No dependencies
                task.status = TaskStatus.READY
                ready_tasks.append(task)
            else:
                # Check if all dependencies are completed
                all_deps_completed = True
                for dep_id in dependencies:
                    dep_task = self.get_task(dep_id)
                    if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                        all_deps_completed = False
                        break
                        
                if all_deps_completed:
                    task.status = TaskStatus.READY
                    ready_tasks.append(task)
                    
        return ready_tasks
        
    def get_critical_path(self) -> List[str]:
        """Get critical path (longest path through the graph)."""
        if not self.tasks:
            return []
            
        # Find critical path using longest path algorithm
        try:
            # NetworkX longest_path works on DAGs
            path = nx.dag_longest_path(self.graph, weight='duration')
            return path
        except:
            # Fallback: simple topological sort
            return list(nx.topological_sort(self.graph))
            
    def get_execution_order(self) -> List[str]:
        """Get topologically sorted execution order."""
        return list(nx.topological_sort(self.graph))
        
    def calculate_total_duration(self) -> float:
        """Calculate total duration along critical path."""
        critical_path = self.get_critical_path()
        total_duration = 0.0
        
        for task_id in critical_path:
            task = self.get_task(task_id)
            if task:
                total_duration += task.estimated_duration
                
        return total_duration
        
    def validate_graph(self) -> List[str]:
        """Validate graph consistency and return list of issues."""
        issues = []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            issues.append("Graph contains cycles")
            
        # Check for orphaned dependencies
        for dep in self.dependencies:
            if not self.has_task(dep.predecessor_id):
                issues.append(f"Dependency references non-existent predecessor: {dep.predecessor_id}")
            if not self.has_task(dep.successor_id):
                issues.append(f"Dependency references non-existent successor: {dep.successor_id}")
                
        # Check for tasks with impossible deadlines
        for task in self.tasks:
            if task.deadline and task.created_at > task.deadline:
                issues.append(f"Task {task.id} has deadline before creation time")
                
        return issues
        
    def get_resource_requirements(self) -> Dict[str, float]:
        """Get aggregated resource requirements across all tasks."""
        resource_totals = {}
        
        for task in self.tasks:
            for req in task.resource_requirements:
                if req.resource_type not in resource_totals:
                    resource_totals[req.resource_type] = 0.0
                resource_totals[req.resource_type] += req.quantity
                
        return resource_totals
        
    def create_quantum_superposition(self) -> np.ndarray:
        """Create quantum superposition state for all tasks."""
        n_tasks = len(self.tasks)
        if n_tasks == 0:
            return np.array([])
            
        # Initialize superposition weights
        weights = np.array([task.superposition_weight for task in self.tasks])
        
        # Normalize to unit probability
        weights = weights / np.linalg.norm(weights) if np.linalg.norm(weights) > 0 else weights
        
        # Create complex superposition state
        phases = np.random.uniform(0, 2*np.pi, n_tasks)
        superposition = weights * np.exp(1j * phases)
        
        self.superposition_active = True
        return superposition
        
    def measure_quantum_state(self, superposition: np.ndarray) -> int:
        """Measure quantum state to collapse to specific task selection."""
        if len(superposition) == 0:
            return -1
            
        # Calculate measurement probabilities
        probabilities = np.abs(superposition) ** 2
        
        # Collapse to single state
        selected_idx = np.random.choice(len(probabilities), p=probabilities)
        
        self.superposition_active = False
        return selected_idx
        
    def _update_entanglement_matrix(self):
        """Update quantum entanglement matrix between tasks."""
        n_tasks = len(self.tasks)
        if n_tasks == 0:
            self.entanglement_matrix = None
            return
            
        self.entanglement_matrix = np.zeros((n_tasks, n_tasks), dtype=complex)
        
        # Set entanglement based on dependencies
        for i, task_i in enumerate(self.tasks):
            for j, task_j in enumerate(self.tasks):
                if i == j:
                    self.entanglement_matrix[i, j] = 1.0 + 0j
                elif self.has_dependency(task_i.id, task_j.id):
                    # Strong entanglement for direct dependencies
                    self.entanglement_matrix[i, j] = 0.8 + 0.6j
                elif self.has_transitive_dependency(task_i.id, task_j.id):
                    # Weaker entanglement for indirect dependencies
                    self.entanglement_matrix[i, j] = 0.3 + 0.2j
                    
    def _update_task_entanglement(self, predecessor_id: str, successor_id: str):
        """Update entanglement strength between specific tasks."""
        predecessor = self.get_task(predecessor_id)
        successor = self.get_task(successor_id)
        
        if predecessor and successor:
            # Update entanglement strength based on dependency
            base_strength = 0.5
            priority_factor = (predecessor.priority + successor.priority) / 20.0
            predecessor.entanglement_strength = min(1.0, base_strength + priority_factor)
            successor.entanglement_strength = min(1.0, base_strength + priority_factor)
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'n_tasks': len(self.tasks),
            'n_dependencies': len(self.dependencies),
            'n_ready_tasks': len(self.get_ready_tasks()),
            'total_estimated_duration': sum(t.estimated_duration for t in self.tasks),
            'critical_path_duration': self.calculate_total_duration(),
            'quantum_coherence': self.quantum_coherence,
            'superposition_active': self.superposition_active,
            'resource_types': list(self.get_resource_requirements().keys()),
            'status_distribution': {status.value: len([t for t in self.tasks if t.status == status]) 
                                  for status in TaskStatus}
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            'name': self.name,
            'tasks': [
                {
                    'id': task.id,
                    'name': task.name,
                    'task_type': task.task_type.value,
                    'estimated_duration': task.estimated_duration,
                    'priority': task.priority,
                    'status': task.status.value,
                    'resource_requirements': [
                        {
                            'resource_type': req.resource_type,
                            'quantity': req.quantity,
                            'duration': req.duration,
                            'exclusive': req.exclusive,
                            'location': req.location
                        } for req in task.resource_requirements
                    ]
                } for task in self.tasks
            ],
            'dependencies': [
                {
                    'predecessor_id': dep.predecessor_id,
                    'successor_id': dep.successor_id,
                    'dependency_type': dep.dependency_type,
                    'delay': dep.delay
                } for dep in self.dependencies
            ],
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskGraph':
        """Deserialize graph from dictionary."""
        graph = cls(name=data.get('name', 'TaskGraph'))
        graph.metadata = data.get('metadata', {})
        
        # Add tasks
        for task_data in data.get('tasks', []):
            task = Task(
                id=task_data['id'],
                name=task_data['name'],
                task_type=TaskType(task_data['task_type']),
                estimated_duration=task_data['estimated_duration'],
                priority=task_data['priority'],
                status=TaskStatus(task_data['status'])
            )
            
            # Add resource requirements
            for req_data in task_data.get('resource_requirements', []):
                req = ResourceRequirement(
                    resource_type=req_data['resource_type'],
                    quantity=req_data['quantity'],
                    duration=req_data['duration'],
                    exclusive=req_data.get('exclusive', False),
                    location=tuple(req_data['location']) if req_data.get('location') else None
                )
                task.resource_requirements.append(req)
                
            graph.add_task(task)
            
        # Add dependencies
        for dep_data in data.get('dependencies', []):
            graph.add_dependency(
                predecessor_id=dep_data['predecessor_id'],
                successor_id=dep_data['successor_id'],
                dependency_type=dep_data.get('dependency_type', 'finish_to_start'),
                delay=dep_data.get('delay', 0.0)
            )
            
        return graph
        
    def clone(self) -> 'TaskGraph':
        """Create a deep copy of the task graph."""
        return TaskGraph.from_dict(self.to_dict())
        
    def __len__(self) -> int:
        return len(self.tasks)
        
    def __contains__(self, task_id: str) -> bool:
        return self.has_task(task_id)
        
    def __repr__(self) -> str:
        return f"TaskGraph('{self.name}', {len(self.tasks)} tasks, {len(self.dependencies)} dependencies)"