"""
Dynamic Resource Pool Management

Provides intelligent resource pooling and load balancing for
scalable quantum-inspired task planning and execution.
"""

import time
import threading
import queue
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from collections import defaultdict, deque
import weakref
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    QUANTUM_PLANNER = "quantum_planner"
    ULTRASONIC_ARRAY = "ultrasonic_array"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    
class ResourceStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    RESERVED = "reserved"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    
@dataclass
class ResourceSpec:
    """Resource specification and requirements."""
    resource_type: ResourceType
    capacity: float
    location: Optional[Tuple[float, float, float]] = None
    capabilities: Set[str] = field(default_factory=set)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    
@dataclass
class ResourceInstance:
    """Individual resource instance."""
    id: str
    spec: ResourceSpec
    status: ResourceStatus = ResourceStatus.AVAILABLE
    current_load: float = 0.0
    max_load: float = 1.0
    last_used: float = 0.0
    total_usage_time: float = 0.0
    failure_count: int = 0
    performance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
            
@dataclass
class ResourceRequest:
    """Resource allocation request."""
    id: str
    requester_id: str
    resource_types: List[ResourceType]
    requirements: Dict[str, Any]
    priority: int = 1
    timeout: float = 30.0
    exclusive: bool = False
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
            
@dataclass
class ResourceAllocation:
    """Resource allocation result."""
    request_id: str
    allocated_resources: List[ResourceInstance]
    allocation_time: float
    estimated_duration: float
    
class LoadBalancer:
    """Intelligent load balancing for resource allocation."""
    
    def __init__(self, strategy: str = "least_loaded"):
        self.strategy = strategy
        self.allocation_history = deque(maxlen=1000)
        
    def select_resources(self, 
                        available_resources: List[ResourceInstance],
                        request: ResourceRequest) -> List[ResourceInstance]:
        """Select optimal resources based on load balancing strategy."""
        
        if self.strategy == "least_loaded":
            return self._least_loaded_selection(available_resources, request)
        elif self.strategy == "round_robin":
            return self._round_robin_selection(available_resources, request)
        elif self.strategy == "performance_based":
            return self._performance_based_selection(available_resources, request)
        elif self.strategy == "location_aware":
            return self._location_aware_selection(available_resources, request)
        else:
            return self._default_selection(available_resources, request)
            
    def _least_loaded_selection(self, resources: List[ResourceInstance], 
                               request: ResourceRequest) -> List[ResourceInstance]:
        """Select resources with lowest current load."""
        # Sort by current load (ascending)
        sorted_resources = sorted(resources, key=lambda r: r.current_load)
        
        selected = []
        for resource_type in request.resource_types:
            for resource in sorted_resources:
                if (resource.spec.resource_type == resource_type and 
                    resource not in selected):
                    selected.append(resource)
                    break
                    
        return selected
        
    def _performance_based_selection(self, resources: List[ResourceInstance],
                                   request: ResourceRequest) -> List[ResourceInstance]:
        """Select resources based on performance scores."""
        # Sort by performance score (descending) and load (ascending)
        sorted_resources = sorted(resources, 
                                key=lambda r: (-r.performance_score, r.current_load))
        
        selected = []
        for resource_type in request.resource_types:
            for resource in sorted_resources:
                if (resource.spec.resource_type == resource_type and 
                    resource not in selected):
                    selected.append(resource)
                    break
                    
        return selected
        
    def _location_aware_selection(self, resources: List[ResourceInstance],
                                request: ResourceRequest) -> List[ResourceInstance]:
        """Select resources based on location proximity."""
        target_location = request.requirements.get('location')
        if not target_location:
            return self._least_loaded_selection(resources, request)
            
        # Calculate distances and sort
        def distance_score(resource):
            if not resource.spec.location:
                return float('inf')
            return np.linalg.norm(np.array(resource.spec.location) - np.array(target_location))
            
        sorted_resources = sorted(resources, key=distance_score)
        
        selected = []
        for resource_type in request.resource_types:
            for resource in sorted_resources:
                if (resource.spec.resource_type == resource_type and 
                    resource not in selected):
                    selected.append(resource)
                    break
                    
        return selected
        
    def _round_robin_selection(self, resources: List[ResourceInstance],
                             request: ResourceRequest) -> List[ResourceInstance]:
        """Round-robin resource selection."""
        # Simple round-robin based on usage count
        selected = []
        for resource_type in request.resource_types:
            type_resources = [r for r in resources if r.spec.resource_type == resource_type]
            if type_resources:
                # Select least recently used
                selected_resource = min(type_resources, key=lambda r: r.last_used)
                selected.append(selected_resource)
                
        return selected
        
    def _default_selection(self, resources: List[ResourceInstance],
                          request: ResourceRequest) -> List[ResourceInstance]:
        """Default resource selection strategy."""
        return self._least_loaded_selection(resources, request)
        
class ResourcePool:
    """
    Dynamic resource pool with intelligent allocation and scaling.
    
    Features:
    - Automatic resource discovery and registration
    - Load-balanced resource allocation
    - Dynamic scaling based on demand
    - Health monitoring and fault tolerance
    - Performance optimization
    """
    
    def __init__(self, 
                 load_balancer: Optional[LoadBalancer] = None,
                 enable_auto_scaling: bool = True,
                 max_pool_size: int = 100):
        
        self.load_balancer = load_balancer or LoadBalancer()
        self.enable_auto_scaling = enable_auto_scaling
        self.max_pool_size = max_pool_size
        
        # Resource management
        self.resources: Dict[str, ResourceInstance] = {}
        self.resource_locks: Dict[str, threading.RLock] = {}
        
        # Request management
        self.pending_requests: queue.PriorityQueue = queue.PriorityQueue()
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        
        # Monitoring and stats
        self.allocation_stats = defaultdict(int)
        self.performance_metrics = deque(maxlen=1000)
        
        # Background threads
        self.allocation_thread = None
        self.monitoring_thread = None
        self.scaling_thread = None
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.avg_allocation_time = 0.0
        self.allocation_success_rate = 1.0
        self.resource_utilization = 0.0
        
        logger.info("ResourcePool initialized")
        
    def start(self):
        """Start resource pool background operations."""
        if self.allocation_thread and self.allocation_thread.is_alive():
            return
            
        self.shutdown_event.clear()
        
        # Start background threads
        self.allocation_thread = threading.Thread(target=self._allocation_loop, daemon=True)
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        
        if self.enable_auto_scaling:
            self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
            
        self.allocation_thread.start()
        self.monitoring_thread.start()
        
        if self.scaling_thread:
            self.scaling_thread.start()
            
        logger.info("ResourcePool started")
        
    def stop(self):
        """Stop resource pool operations."""
        self.shutdown_event.set()
        
        # Wait for threads to complete
        for thread in [self.allocation_thread, self.monitoring_thread, self.scaling_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
                
        logger.info("ResourcePool stopped")
        
    def register_resource(self, spec: ResourceSpec, 
                         instance_data: Optional[Dict[str, Any]] = None) -> str:
        """Register a new resource in the pool."""
        
        resource = ResourceInstance(
            id=str(uuid.uuid4()),
            spec=spec,
            metadata=instance_data or {}
        )
        
        self.resources[resource.id] = resource
        self.resource_locks[resource.id] = threading.RLock()
        
        logger.info(f"Registered resource: {resource.id} ({spec.resource_type.value})")
        return resource.id
        
    def unregister_resource(self, resource_id: str):
        """Remove resource from pool."""
        if resource_id in self.resources:
            # Check if resource is currently allocated
            active_allocations = [a for a in self.active_allocations.values() 
                                if any(r.id == resource_id for r in a.allocated_resources)]
            
            if active_allocations:
                # Mark for removal after current allocations complete
                self.resources[resource_id].status = ResourceStatus.MAINTENANCE
                logger.warning(f"Resource {resource_id} marked for removal (has active allocations)")
            else:
                # Safe to remove immediately
                del self.resources[resource_id]
                del self.resource_locks[resource_id]
                logger.info(f"Unregistered resource: {resource_id}")
                
    def request_resources(self, request: ResourceRequest) -> Optional[str]:
        """Request resource allocation."""
        
        # Add to pending queue with priority
        priority = -request.priority  # Higher priority = lower number in queue
        self.pending_requests.put((priority, time.time(), request))
        
        logger.debug(f"Resource request queued: {request.id}")
        return request.id
        
    def release_resources(self, allocation_id: str):
        """Release allocated resources."""
        if allocation_id not in self.active_allocations:
            logger.warning(f"Unknown allocation ID: {allocation_id}")
            return
            
        allocation = self.active_allocations[allocation_id]
        
        # Release each resource
        for resource in allocation.allocated_resources:
            with self.resource_locks[resource.id]:
                resource.status = ResourceStatus.AVAILABLE
                resource.current_load = 0.0
                
                # Update usage statistics
                usage_duration = time.time() - allocation.allocation_time
                resource.total_usage_time += usage_duration
                
        del self.active_allocations[allocation_id]
        logger.debug(f"Released resources for allocation: {allocation_id}")
        
    def get_allocation(self, allocation_id: str) -> Optional[ResourceAllocation]:
        """Get allocation details."""
        return self.active_allocations.get(allocation_id)
        
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource pool status."""
        total_resources = len(self.resources)
        available_resources = len([r for r in self.resources.values() 
                                 if r.status == ResourceStatus.AVAILABLE])
        busy_resources = len([r for r in self.resources.values() 
                            if r.status == ResourceStatus.BUSY])
        
        # Resource type distribution
        type_distribution = defaultdict(int)
        for resource in self.resources.values():
            type_distribution[resource.spec.resource_type.value] += 1
            
        return {
            'total_resources': total_resources,
            'available_resources': available_resources,
            'busy_resources': busy_resources,
            'pending_requests': self.pending_requests.qsize(),
            'active_allocations': len(self.active_allocations),
            'resource_utilization': self.resource_utilization,
            'avg_allocation_time_ms': self.avg_allocation_time * 1000,
            'allocation_success_rate': self.allocation_success_rate,
            'resource_type_distribution': dict(type_distribution)
        }
        
    def optimize_pool(self):
        """Optimize resource pool configuration based on usage patterns."""
        if len(self.performance_metrics) < 10:
            return  # Need more data
            
        recent_metrics = list(self.performance_metrics)[-10:]
        
        # Analyze utilization patterns
        avg_utilization = np.mean([m['utilization'] for m in recent_metrics])
        avg_wait_time = np.mean([m['avg_wait_time'] for m in recent_metrics])
        
        # Optimization decisions
        if avg_utilization > 0.8 and avg_wait_time > 1.0:
            # High utilization and wait time - need more resources
            self._trigger_scaling_up()
        elif avg_utilization < 0.3 and len(self.resources) > 5:
            # Low utilization - can scale down
            self._trigger_scaling_down()
            
        # Optimize load balancing strategy
        self._optimize_load_balancing(recent_metrics)
        
    def _allocation_loop(self):
        """Background thread for processing resource allocation requests."""
        while not self.shutdown_event.is_set():
            try:
                # Get next request with timeout
                try:
                    priority, timestamp, request = self.pending_requests.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Check request timeout
                if time.time() - request.created_at > request.timeout:
                    logger.warning(f"Request {request.id} timed out")
                    continue
                    
                # Process allocation
                allocation = self._process_allocation_request(request)
                
                if allocation:
                    self.active_allocations[allocation.request_id] = allocation
                    logger.debug(f"Allocated resources for request: {request.id}")
                else:
                    # Re-queue if no resources available
                    self.pending_requests.put((priority, timestamp, request))
                    time.sleep(0.1)  # Brief delay before retry
                    
            except Exception as e:
                logger.error(f"Error in allocation loop: {e}")
                
    def _process_allocation_request(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Process individual resource allocation request."""
        start_time = time.time()
        
        # Find available resources
        available_resources = [
            resource for resource in self.resources.values()
            if resource.status == ResourceStatus.AVAILABLE and
               self._resource_matches_request(resource, request)
        ]
        
        if len(available_resources) < len(request.resource_types):
            return None  # Not enough resources
            
        # Select optimal resources using load balancer
        selected_resources = self.load_balancer.select_resources(available_resources, request)
        
        if len(selected_resources) != len(request.resource_types):
            return None  # Couldn't find all required resource types
            
        # Lock and allocate resources
        allocated_resources = []
        try:
            for resource in selected_resources:
                with self.resource_locks[resource.id]:
                    if resource.status == ResourceStatus.AVAILABLE:
                        resource.status = ResourceStatus.BUSY
                        resource.current_load = request.requirements.get('load', 1.0)
                        resource.last_used = time.time()
                        allocated_resources.append(resource)
                    else:
                        # Resource became unavailable, rollback
                        for allocated in allocated_resources:
                            allocated.status = ResourceStatus.AVAILABLE
                            allocated.current_load = 0.0
                        return None
                        
        except Exception as e:
            logger.error(f"Error during resource allocation: {e}")
            # Rollback allocations
            for allocated in allocated_resources:
                allocated.status = ResourceStatus.AVAILABLE
                allocated.current_load = 0.0
            return None
            
        allocation_time = time.time() - start_time
        
        # Update statistics
        self.allocation_stats['successful'] += 1
        self.avg_allocation_time = (self.avg_allocation_time * 0.9 + allocation_time * 0.1)
        
        return ResourceAllocation(
            request_id=request.id,
            allocated_resources=allocated_resources,
            allocation_time=time.time(),
            estimated_duration=request.requirements.get('duration', 60.0)
        )
        
    def _resource_matches_request(self, resource: ResourceInstance, 
                                request: ResourceRequest) -> bool:
        """Check if resource matches request requirements."""
        
        # Check resource type
        if resource.spec.resource_type not in request.resource_types:
            return False
            
        # Check capabilities
        required_capabilities = request.requirements.get('capabilities', set())
        if not required_capabilities.issubset(resource.spec.capabilities):
            return False
            
        # Check capacity
        required_capacity = request.requirements.get('capacity', 0.0)
        if required_capacity > resource.spec.capacity:
            return False
            
        # Check location constraints
        if 'location' in request.requirements and resource.spec.location:
            max_distance = request.requirements.get('max_distance', float('inf'))
            distance = np.linalg.norm(
                np.array(resource.spec.location) - 
                np.array(request.requirements['location'])
            )
            if distance > max_distance:
                return False
                
        return True
        
    def _monitoring_loop(self):
        """Background monitoring and health checking."""
        while not self.shutdown_event.is_set():
            try:
                # Monitor resource health
                self._check_resource_health()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Clean up expired allocations
                self._cleanup_expired_allocations()
                
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
    def _check_resource_health(self):
        """Check health of all resources."""
        for resource in self.resources.values():
            try:
                # Simple health check - would be extended for specific resource types
                if resource.failure_count > 5:
                    resource.status = ResourceStatus.FAILED
                    logger.warning(f"Resource {resource.id} marked as failed")
                elif resource.status == ResourceStatus.FAILED and resource.failure_count <= 2:
                    # Attempt recovery
                    resource.status = ResourceStatus.AVAILABLE
                    logger.info(f"Resource {resource.id} recovered")
                    
            except Exception as e:
                resource.failure_count += 1
                logger.error(f"Health check failed for resource {resource.id}: {e}")
                
    def _update_performance_metrics(self):
        """Update pool performance metrics."""
        
        # Calculate current utilization
        total_resources = len(self.resources)
        busy_resources = len([r for r in self.resources.values() 
                            if r.status == ResourceStatus.BUSY])
        
        current_utilization = busy_resources / max(1, total_resources)
        self.resource_utilization = current_utilization
        
        # Calculate average wait time
        pending_count = self.pending_requests.qsize()
        avg_wait_time = pending_count * self.avg_allocation_time
        
        # Record metrics
        metrics = {
            'timestamp': time.time(),
            'utilization': current_utilization,
            'avg_wait_time': avg_wait_time,
            'pending_requests': pending_count,
            'active_allocations': len(self.active_allocations)
        }
        
        self.performance_metrics.append(metrics)
        
    def _cleanup_expired_allocations(self):
        """Clean up allocations that have exceeded their estimated duration."""
        current_time = time.time()
        expired_allocations = []
        
        for allocation_id, allocation in self.active_allocations.items():
            if (current_time - allocation.allocation_time) > (allocation.estimated_duration * 2):
                expired_allocations.append(allocation_id)
                
        for allocation_id in expired_allocations:
            logger.warning(f"Force-releasing expired allocation: {allocation_id}")
            self.release_resources(allocation_id)
            
    def _scaling_loop(self):
        """Background auto-scaling thread."""
        while not self.shutdown_event.is_set():
            try:
                time.sleep(30.0)  # Check every 30 seconds
                
                if len(self.performance_metrics) >= 5:
                    self._evaluate_scaling_needs()
                    
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                
    def _evaluate_scaling_needs(self):
        """Evaluate if scaling up or down is needed."""
        recent_metrics = list(self.performance_metrics)[-5:]
        
        avg_utilization = np.mean([m['utilization'] for m in recent_metrics])
        avg_wait_time = np.mean([m['avg_wait_time'] for m in recent_metrics])
        
        # Scale up conditions
        if (avg_utilization > 0.8 and avg_wait_time > 2.0 and 
            len(self.resources) < self.max_pool_size):
            self._trigger_scaling_up()
            
        # Scale down conditions  
        elif (avg_utilization < 0.2 and avg_wait_time < 0.5 and 
              len(self.resources) > 2):
            self._trigger_scaling_down()
            
    def _trigger_scaling_up(self):
        """Trigger scaling up by adding resources."""
        # Determine which resource types to add based on demand
        type_demand = defaultdict(int)
        for _, _, request in list(self.pending_requests.queue):
            for resource_type in request.resource_types:
                type_demand[resource_type] += 1
                
        # Add resources for most demanded types
        for resource_type, demand in sorted(type_demand.items(), key=lambda x: x[1], reverse=True):
            if len(self.resources) >= self.max_pool_size:
                break
                
            # Create new virtual resource (simplified)
            spec = ResourceSpec(
                resource_type=resource_type,
                capacity=1.0,
                capabilities={'auto_scaled'}
            )
            
            self.register_resource(spec)
            logger.info(f"Auto-scaled up: added {resource_type.value} resource")
            break  # Add one at a time
            
    def _trigger_scaling_down(self):
        """Trigger scaling down by removing underutilized resources."""
        # Find resources that are consistently underutilized
        candidates = [
            resource for resource in self.resources.values()
            if (resource.status == ResourceStatus.AVAILABLE and
                'auto_scaled' in resource.spec.capabilities and
                time.time() - resource.last_used > 300)  # 5 minutes idle
        ]
        
        if candidates:
            resource_to_remove = candidates[0]
            self.unregister_resource(resource_to_remove.id)
            logger.info(f"Auto-scaled down: removed {resource_to_remove.spec.resource_type.value} resource")
            
    def _optimize_load_balancing(self, metrics: List[Dict[str, Any]]):
        """Optimize load balancing strategy based on performance."""
        # Analyze allocation patterns and adjust strategy if needed
        if len(metrics) < 5:
            return
            
        # Simple strategy optimization - could be more sophisticated
        avg_wait_time = np.mean([m['avg_wait_time'] for m in metrics])
        
        if avg_wait_time > 1.0 and self.load_balancer.strategy != "performance_based":
            self.load_balancer.strategy = "performance_based"
            logger.info("Switched load balancing to performance_based strategy")
        elif avg_wait_time < 0.2 and self.load_balancer.strategy != "least_loaded":
            self.load_balancer.strategy = "least_loaded"
            logger.info("Switched load balancing to least_loaded strategy")
            
# Global resource pool instance
_global_resource_pool = None

def get_global_resource_pool() -> ResourcePool:
    """Get or create global resource pool."""
    global _global_resource_pool
    if _global_resource_pool is None:
        _global_resource_pool = ResourcePool()
        _global_resource_pool.start()
    return _global_resource_pool
    
def shutdown_global_resource_pool():
    """Shutdown global resource pool."""
    global _global_resource_pool
    if _global_resource_pool:
        _global_resource_pool.stop()
        _global_resource_pool = None