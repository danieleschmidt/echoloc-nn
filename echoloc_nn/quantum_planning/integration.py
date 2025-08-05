"""
EchoLoc-NN Integration Bridge

Provides seamless integration between quantum-inspired task planning
and the EchoLoc-NN ultrasonic localization system.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import logging
from threading import Thread, Event, Lock
from queue import Queue, Empty

from .planner import QuantumTaskPlanner, PlanningConfig
from .task_graph import TaskGraph, Task, TaskType
from .metrics import PlanningMetrics
from ..inference.locator import EchoLocator
from ..hardware.ultrasonic_array import UltrasonicArray

logger = logging.getLogger(__name__)

@dataclass
class PositionBasedTask(Task):
    """Task with position-aware requirements."""
    target_position: Optional[Tuple[float, float, float]] = None
    position_tolerance: float = 0.1  # meters
    requires_movement: bool = False
    movement_priority: int = 1
    
class LocationAwareResource:
    """Resource with position information."""
    
    def __init__(self, resource_id: str, resource_type: str, 
                 position: Tuple[float, float, float],
                 capabilities: List[str],
                 max_range: float = 5.0):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.position = np.array(position)
        self.capabilities = capabilities
        self.max_range = max_range
        self.is_busy = False
        self.current_task = None
        
    def can_execute(self, task: Task, current_position: np.ndarray) -> bool:
        """Check if resource can execute task from current position."""
        if self.is_busy:
            return False
            
        # Check capability match
        if hasattr(task, 'required_capabilities'):
            if not all(cap in self.capabilities for cap in task.required_capabilities):
                return False
                
        # Check position requirements
        if isinstance(task, PositionBasedTask) and task.target_position:
            target_pos = np.array(task.target_position)
            distance_to_target = np.linalg.norm(self.position - target_pos)
            
            if distance_to_target > self.max_range:
                return False
                
            # Check if movement is required and possible
            if task.requires_movement:
                movement_distance = np.linalg.norm(current_position - target_pos)
                if movement_distance > task.position_tolerance:
                    return False
                    
        return True
        
    def get_execution_cost(self, task: Task, current_position: np.ndarray) -> float:
        """Calculate execution cost including movement."""
        base_cost = task.estimated_duration
        
        if isinstance(task, PositionBasedTask) and task.target_position:
            target_pos = np.array(task.target_position)
            
            # Movement cost
            if task.requires_movement:
                movement_distance = np.linalg.norm(current_position - target_pos)
                movement_time = movement_distance / 1.0  # Assume 1 m/s movement speed
                base_cost += movement_time
                
            # Distance penalty
            resource_distance = np.linalg.norm(self.position - target_pos)
            distance_penalty = resource_distance * 0.1
            base_cost += distance_penalty
            
        return base_cost

class EchoLocPlanningBridge:
    """
    Integration bridge between EchoLoc-NN positioning and quantum task planning.
    
    Provides position-aware task scheduling that optimizes for both task
    execution efficiency and movement costs in the physical environment.
    """
    
    def __init__(self, 
                 locator: Optional[EchoLocator] = None,
                 array: Optional[UltrasonicArray] = None,
                 planning_config: Optional[PlanningConfig] = None):
        # Core components
        self.locator = locator
        self.array = array
        self.planner = QuantumTaskPlanner(planning_config or PlanningConfig())
        self.metrics = PlanningMetrics()
        
        # Position tracking
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.position_history = []
        self.position_confidence = 1.0
        self.last_position_update = time.time()
        
        # Resource management
        self.location_aware_resources: Dict[str, LocationAwareResource] = {}
        self.resource_positions: Dict[str, np.ndarray] = {}
        
        # Real-time integration
        self.position_update_thread = None
        self.shutdown_event = Event()
        self.position_lock = Lock() 
        self.position_queue = Queue(maxsize=100)
        
        # Planning optimization
        self.movement_cost_weight = 0.3
        self.position_uncertainty_penalty = 0.1
        self.realtime_replanning = True
        self.replanning_threshold = 0.5  # meters
        
        logger.info("EchoLocPlanningBridge initialized")
        
    def start_position_tracking(self):
        """Start real-time position tracking."""
        if not self.locator or not self.array:
            logger.warning("Cannot start position tracking: locator or array not configured")
            return
            
        if self.position_update_thread and self.position_update_thread.is_alive():
            logger.warning("Position tracking already running")
            return
            
        self.shutdown_event.clear()
        self.position_update_thread = Thread(target=self._position_tracking_loop)
        self.position_update_thread.daemon = True
        self.position_update_thread.start()
        
        logger.info("Position tracking started")
        
    def stop_position_tracking(self):
        """Stop real-time position tracking."""
        if self.position_update_thread:
            self.shutdown_event.set()
            self.position_update_thread.join(timeout=5.0)
            
        logger.info("Position tracking stopped")
        
    def add_location_aware_resource(self, resource: LocationAwareResource):
        """Add a location-aware resource."""
        self.location_aware_resources[resource.resource_id] = resource
        self.resource_positions[resource.resource_id] = resource.position.copy()
        
        logger.info(f"Added location-aware resource: {resource.resource_id} at {resource.position}")
        
    def plan_position_aware_tasks(self, 
                                 task_graph: TaskGraph,
                                 optimization_constraints: Optional[Dict[str, Any]] = None) -> 'OptimizationResults':
        """Plan tasks with position awareness and movement optimization."""
        
        # Get current position
        with self.position_lock:
            current_pos = self.current_position.copy()
            pos_confidence = self.position_confidence
            
        # Enhance task graph with position information
        enhanced_graph = self._enhance_task_graph_with_positions(task_graph, current_pos)
        
        # Create position-aware resource map
        position_aware_resources = self._create_position_aware_resource_map(current_pos)
        
        # Configure constraints with position awareness
        enhanced_constraints = optimization_constraints or {}
        enhanced_constraints.update({
            'current_position': current_pos,
            'position_confidence': pos_confidence,
            'movement_cost_weight': self.movement_cost_weight,
            'position_uncertainty_penalty': self.position_uncertainty_penalty
        })
        
        # Update planner with position feedback
        position_feedback = {
            'current_position': current_pos.tolist(),
            'confidence': pos_confidence,
            'timestamp': time.time(),
            'resource_positions': {rid: pos.tolist() for rid, pos in self.resource_positions.items()}
        }
        self.planner.update_position_feedback(position_feedback)
        
        # Execute planning
        result = self.planner.plan_tasks(enhanced_graph, position_aware_resources, enhanced_constraints)
        
        # Post-process result with movement optimization
        optimized_result = self._optimize_movement_sequence(result, current_pos)
        
        return optimized_result
        
    def execute_plan_with_positioning(self, 
                                    optimization_result: 'OptimizationResults',
                                    execution_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute planned tasks with real-time position feedback."""
        
        execution_results = {
            'completed_tasks': [],
            'failed_tasks': [],
            'total_movement_distance': 0.0,
            'actual_execution_time': 0.0,
            'position_accuracy': 1.0
        }
        
        start_time = time.time()
        previous_position = None
        
        for step in optimization_result.execution_plan:
            step_start_time = time.time()
            
            # Get current position
            with self.position_lock:
                current_pos = self.current_position.copy()
                pos_confidence = self.position_confidence
                
            # Calculate movement if needed
            if previous_position is not None:
                movement_distance = np.linalg.norm(current_pos - previous_position)
                execution_results['total_movement_distance'] += movement_distance
                
            # Check if replanning is needed
            if self.realtime_replanning and self._should_replan(step, current_pos):
                logger.info(f"Replanning triggered for task {step['task_id']}")
                # TODO: Implement dynamic replanning
                
            # Execute task step
            try:
                if execution_callback:
                    step_result = execution_callback(step, current_pos, pos_confidence)
                else:
                    step_result = self._default_task_execution(step, current_pos)
                    
                execution_results['completed_tasks'].append({
                    'task_id': step['task_id'],
                    'position': current_pos.tolist(),
                    'confidence': pos_confidence,
                    'execution_time': time.time() - step_start_time,
                    'result': step_result
                })
                
                # Record execution metrics
                self.metrics.record_task_execution(
                    task_id=step['task_id'],
                    planned_start=step['start_time'],
                    actual_start=step_start_time - start_time,
                    planned_duration=step['duration'],
                    actual_duration=time.time() - step_start_time,
                    resource_used=step['resource'],
                    success=True
                )
                
            except Exception as e:
                logger.error(f"Task execution failed: {step['task_id']}, error: {e}")
                execution_results['failed_tasks'].append({
                    'task_id': step['task_id'],
                    'error': str(e),
                    'position': current_pos.tolist()
                })
                
                self.metrics.record_task_execution(
                    task_id=step['task_id'],
                    planned_start=step['start_time'],
                    actual_start=step_start_time - start_time,
                    planned_duration=step['duration'],
                    actual_duration=time.time() - step_start_time,
                    resource_used=step['resource'],
                    success=False,
                    error_message=str(e)
                )
                
            previous_position = current_pos.copy()
            
        execution_results['actual_execution_time'] = time.time() - start_time
        execution_results['position_accuracy'] = self._calculate_position_accuracy()
        
        return execution_results
        
    def get_position_aware_metrics(self) -> Dict[str, Any]:
        """Get metrics including position-aware performance."""
        base_metrics = self.metrics.get_comprehensive_report()
        
        position_metrics = {
            'current_position': self.current_position.tolist(),
            'position_confidence': self.position_confidence,
            'position_update_frequency': self._calculate_position_update_frequency(),
            'movement_efficiency': self._calculate_movement_efficiency(),
            'resource_accessibility': self._calculate_resource_accessibility()
        }
        
        base_metrics['position_metrics'] = position_metrics
        return base_metrics
        
    def _position_tracking_loop(self):
        """Background loop for position tracking."""
        logger.info("Position tracking loop started")
        
        try:
            for echo_data in self.array.stream_chirps():
                if self.shutdown_event.is_set():
                    break
                    
                try:
                    # Get position from EchoLoc
                    position, confidence = self.locator.locate(echo_data)
                    
                    # Update position with thread safety
                    with self.position_lock:
                        self.current_position = np.array(position)
                        self.position_confidence = confidence
                        self.last_position_update = time.time()
                        
                    # Record position history
                    self.position_history.append({
                        'position': position,
                        'confidence': confidence,
                        'timestamp': time.time()
                    })
                    
                    # Limit history size
                    if len(self.position_history) > 1000:
                        self.position_history = self.position_history[-500:]
                        
                    # Add to queue for real-time processing
                    try:
                        self.position_queue.put_nowait({
                            'position': position,
                            'confidence': confidence,
                            'timestamp': time.time()
                        })
                    except:
                        pass  # Queue full, skip
                        
                except Exception as e:
                    logger.warning(f"Position update failed: {e}")
                    
        except Exception as e:
            logger.error(f"Position tracking loop error: {e}")
        finally:
            logger.info("Position tracking loop stopped")
            
    def _enhance_task_graph_with_positions(self, task_graph: TaskGraph, current_position: np.ndarray) -> TaskGraph:
        """Enhance task graph with position information."""
        enhanced_graph = task_graph.clone()
        
        # Convert regular tasks to position-aware tasks where applicable
        for task in enhanced_graph.tasks:
            if hasattr(task, 'parameters') and 'target_position' in task.parameters:
                # Create position-based task
                position_task = PositionBasedTask(
                    id=task.id,
                    name=task.name,
                    task_type=task.task_type,
                    estimated_duration=task.estimated_duration,
                    priority=task.priority,
                    target_position=tuple(task.parameters['target_position']),
                    position_tolerance=task.parameters.get('position_tolerance', 0.1),
                    requires_movement=task.parameters.get('requires_movement', False)
                )
                
                # Calculate movement cost
                if position_task.target_position:
                    target_pos = np.array(position_task.target_position)
                    movement_distance = np.linalg.norm(current_position - target_pos)
                    movement_time = movement_distance / 1.0  # 1 m/s speed assumption
                    
                    # Adjust estimated duration
                    position_task.estimated_duration += movement_time * self.movement_cost_weight
                    
        return enhanced_graph
        
    def _create_position_aware_resource_map(self, current_position: np.ndarray) -> Dict[str, Any]:
        """Create resource map with position awareness."""
        resource_map = {}
        
        for resource_id, resource in self.location_aware_resources.items():
            # Calculate accessibility score
            distance = np.linalg.norm(current_position - resource.position)
            accessibility = max(0.0, 1.0 - distance / resource.max_range)
            
            resource_map[resource_id] = {
                'type': resource.resource_type,
                'position': resource.position.tolist(),
                'capabilities': resource.capabilities,
                'accessibility': accessibility,
                'is_busy': resource.is_busy,
                'max_range': resource.max_range
            }
            
        return resource_map
        
    def _optimize_movement_sequence(self, result: 'OptimizationResults', current_position: np.ndarray) -> 'OptimizationResults':
        """Optimize execution sequence to minimize movement."""
        if not result.execution_plan:
            return result
            
        # Simple greedy optimization: sort by distance to current position
        optimized_plan = []
        remaining_tasks = result.execution_plan.copy()
        current_pos = current_position.copy()
        
        while remaining_tasks:
            # Find nearest task that respects dependencies
            best_task = None
            best_distance = float('inf')
            
            for task in remaining_tasks:
                # Check if dependencies are satisfied
                deps_satisfied = all(
                    any(completed['task_id'] == dep_id for completed in optimized_plan)
                    for dep_id in task.get('dependencies', [])
                )
                
                if deps_satisfied:
                    # Calculate distance
                    if 'target_position' in task:
                        target_pos = np.array(task['target_position'])
                        distance = np.linalg.norm(current_pos - target_pos)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_task = task
                    else:
                        # No position requirement, prioritize
                        if best_task is None:
                            best_task = task
                            
            if best_task:
                optimized_plan.append(best_task)
                remaining_tasks.remove(best_task)
                
                # Update current position
                if 'target_position' in best_task:
                    current_pos = np.array(best_task['target_position'])
            else:
                # No valid task found, add first available
                optimized_plan.append(remaining_tasks.pop(0))
                
        result.execution_plan = optimized_plan
        return result
        
    def _should_replan(self, current_step: Dict[str, Any], current_position: np.ndarray) -> bool:
        """Determine if replanning is needed based on current conditions."""
        # Check position deviation
        if 'expected_position' in current_step:
            expected_pos = np.array(current_step['expected_position'])
            position_deviation = np.linalg.norm(current_position - expected_pos)
            
            if position_deviation > self.replanning_threshold:
                return True
                
        # Check resource availability changes
        required_resource = current_step.get('resource')
        if required_resource and required_resource in self.location_aware_resources:
            resource = self.location_aware_resources[required_resource]
            if resource.is_busy or not resource.can_execute(current_step, current_position):
                return True
                
        return False
        
    def _default_task_execution(self, step: Dict[str, Any], position: np.ndarray) -> Dict[str, Any]:
        """Default task execution implementation."""
        # Simulate task execution
        time.sleep(min(0.1, step.get('duration', 1.0)))  # Simulate work
        
        return {
            'success': True,
            'execution_time': step.get('duration', 1.0),
            'position': position.tolist()
        }
        
    def _calculate_position_accuracy(self) -> float:
        """Calculate overall position accuracy from history."""
        if not self.position_history:
            return 1.0
            
        recent_confidences = [p['confidence'] for p in self.position_history[-20:]]
        return sum(recent_confidences) / len(recent_confidences)
        
    def _calculate_position_update_frequency(self) -> float:
        """Calculate position update frequency (Hz)."""
        if len(self.position_history) < 2:
            return 0.0
            
        recent_updates = self.position_history[-10:]
        time_span = recent_updates[-1]['timestamp'] - recent_updates[0]['timestamp']
        
        if time_span > 0:
            return len(recent_updates) / time_span
        return 0.0
        
    def _calculate_movement_efficiency(self) -> float:
        """Calculate movement efficiency based on position history."""
        if len(self.position_history) < 2:
            return 1.0
            
        total_distance = 0.0
        for i in range(1, len(self.position_history)):
            pos1 = np.array(self.position_history[i-1]['position'])
            pos2 = np.array(self.position_history[i]['position'])
            total_distance += np.linalg.norm(pos2 - pos1)
            
        # Efficiency based on smoothness of movement
        time_span = (self.position_history[-1]['timestamp'] - 
                    self.position_history[0]['timestamp'])
        
        if time_span > 0 and total_distance > 0:
            avg_speed = total_distance / time_span
            # Normalize efficiency (lower variance in speed = higher efficiency)
            return min(1.0, 1.0 / (1.0 + avg_speed))
            
        return 1.0
        
    def _calculate_resource_accessibility(self) -> Dict[str, float]:
        """Calculate accessibility score for each resource."""
        accessibility = {}
        
        with self.position_lock:
            current_pos = self.current_position.copy()
            
        for resource_id, resource in self.location_aware_resources.items():
            distance = np.linalg.norm(current_pos - resource.position)
            accessibility[resource_id] = max(0.0, 1.0 - distance / resource.max_range)
            
        return accessibility
        
    def shutdown(self):
        """Clean shutdown of the integration bridge."""
        logger.info("Shutting down EchoLocPlanningBridge")
        
        # Stop position tracking
        self.stop_position_tracking()
        
        # Clear resources
        self.location_aware_resources.clear()
        self.resource_positions.clear()
        
        logger.info("EchoLocPlanningBridge shutdown complete")
        
    def __enter__(self):
        self.start_position_tracking()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()