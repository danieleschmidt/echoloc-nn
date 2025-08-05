"""
Quantum-Inspired Optimization Algorithms

Implements quantum annealing, superposition search, and hybrid optimization
algorithms for task scheduling and resource allocation problems.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class AnnealingScheduleType(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    GEOMETRIC = "geometric"
    CUSTOM = "custom"

@dataclass
class OptimizationResults:
    """Results from quantum-inspired optimization."""
    assignment: Dict[str, Any]
    energy: float
    convergence: int
    energy_history: List[float]
    execution_plan: List[Dict[str, Any]]
    optimization_time: float = 0.0
    quantum_coherence: float = 1.0
    measurement_count: int = 0
    superposition_states: int = 1
    
    def get_best_energy(self) -> float:
        """Get the best (minimum) energy achieved."""
        return min(self.energy_history) if self.energy_history else self.energy
        
    def get_convergence_rate(self) -> float:
        """Calculate convergence rate based on energy improvement."""
        if len(self.energy_history) < 2:
            return 0.0
            
        initial_energy = self.energy_history[0]
        final_energy = self.energy_history[-1]
        
        if initial_energy == final_energy:
            return 0.0
            
        improvement = (initial_energy - final_energy) / abs(initial_energy)
        convergence_rate = improvement / len(self.energy_history)
        
        return max(0.0, convergence_rate)
        
    def is_converged(self, threshold: float = 1e-6) -> bool:
        """Check if optimization has converged."""
        if len(self.energy_history) < 10:
            return False
            
        recent_energies = self.energy_history[-10:]
        energy_variance = np.var(recent_energies)
        
        return energy_variance < threshold

class AnnealingSchedule:
    """Temperature schedule for quantum annealing."""
    
    def __init__(self, 
                 initial_temp: float = 100.0,
                 final_temp: float = 0.01,
                 schedule_type: str = "exponential",
                 max_iterations: int = 1000,
                 custom_schedule: Optional[Callable[[int, int], float]] = None):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.schedule_type = AnnealingScheduleType(schedule_type)
        self.max_iterations = max_iterations
        self.custom_schedule = custom_schedule
        
        # Precompute schedule parameters
        self._setup_schedule()
        
    def _setup_schedule(self):
        """Setup schedule-specific parameters."""
        if self.schedule_type == AnnealingScheduleType.EXPONENTIAL:
            self.decay_rate = math.pow(self.final_temp / self.initial_temp, 1.0 / self.max_iterations)
        elif self.schedule_type == AnnealingScheduleType.GEOMETRIC:
            self.decay_rate = math.pow(self.final_temp / self.initial_temp, 1.0 / (self.max_iterations - 1))
            
    def get_temperature(self, iteration: int) -> float:
        """Get temperature for given iteration."""
        if iteration >= self.max_iterations:
            return self.final_temp
            
        progress = iteration / self.max_iterations
        
        if self.schedule_type == AnnealingScheduleType.LINEAR:
            return self.initial_temp * (1 - progress) + self.final_temp * progress
            
        elif self.schedule_type == AnnealingScheduleType.EXPONENTIAL:
            return self.initial_temp * (self.decay_rate ** iteration)
            
        elif self.schedule_type == AnnealingScheduleType.LOGARITHMIC:
            return self.initial_temp / (1 + math.log(1 + iteration))
            
        elif self.schedule_type == AnnealingScheduleType.GEOMETRIC:
            return self.initial_temp * (self.decay_rate ** iteration)
            
        elif self.schedule_type == AnnealingScheduleType.CUSTOM and self.custom_schedule:
            return self.custom_schedule(iteration, self.max_iterations)
            
        else:
            return self.initial_temp * math.exp(-iteration / (self.max_iterations / 5))

class QuantumOptimizer:
    """
    Quantum-inspired optimization engine for task scheduling problems.
    
    Implements various quantum-inspired algorithms including:
    - Quantum annealing with tunneling effects
    - Superposition-based parallel search
    - Quantum interference optimization
    - Hybrid quantum-classical approaches
    """
    
    def __init__(self, 
                 enable_parallel: bool = True,
                 max_threads: int = 4,
                 random_seed: Optional[int] = None):
        self.enable_parallel = enable_parallel
        self.max_threads = max_threads
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Quantum state tracking
        self.quantum_register = None
        self.coherence_time = float('inf')
        self.decoherence_rate = 0.0
        
        logger.info(f"Initialized QuantumOptimizer (parallel={enable_parallel}, threads={max_threads})")
        
    def quantum_annealing(self,
                         cost_function: Callable[[Any], float],
                         initial_state: Any,
                         state_generator: Callable[[Any], Any],
                         schedule: AnnealingSchedule,
                         tunneling_rate: float = 0.1) -> OptimizationResults:
        """
        Quantum annealing optimization with tunneling effects.
        
        Args:
            cost_function: Function to minimize
            initial_state: Starting state
            state_generator: Function to generate neighboring states
            schedule: Temperature annealing schedule
            tunneling_rate: Probability of quantum tunneling moves
            
        Returns:
            OptimizationResults with best solution found
        """
        start_time = time.time()
        
        current_state = initial_state
        current_energy = cost_function(current_state)
        
        best_state = current_state
        best_energy = current_energy
        
        energy_history = [current_energy]
        
        for iteration in range(schedule.max_iterations):
            temperature = schedule.get_temperature(iteration)
            
            # Generate candidate state
            if np.random.random() < tunneling_rate:
                # Quantum tunneling: large state change
                candidate_state = self._quantum_tunneling_move(current_state, state_generator)
            else:
                # Classical move: local perturbation
                candidate_state = state_generator(current_state)
                
            candidate_energy = cost_function(candidate_state)
            
            # Quantum-enhanced acceptance probability
            if self._quantum_accept(current_energy, candidate_energy, temperature, tunneling_rate):
                current_state = candidate_state
                current_energy = candidate_energy
                
                if current_energy < best_energy:
                    best_state = current_state
                    best_energy = current_energy
                    
            energy_history.append(current_energy)
            
            # Apply decoherence
            if self.decoherence_rate > 0:
                self._apply_decoherence(iteration)
                
        optimization_time = time.time() - start_time
        
        return OptimizationResults(
            assignment=best_state,
            energy=best_energy,
            convergence=len(energy_history),
            energy_history=energy_history,
            execution_plan=[],  # To be filled by caller
            optimization_time=optimization_time,
            quantum_coherence=self._calculate_coherence(),
            measurement_count=len(energy_history)
        )
        
    def superposition_search(self,
                           cost_function: Callable[[Any], float],
                           initial_states: List[Any],
                           state_generator: Callable[[Any], Any],
                           max_iterations: int = 1000,
                           collapse_threshold: float = 0.8) -> OptimizationResults:
        """
        Superposition-based parallel search maintaining multiple solution candidates.
        
        Args:
            cost_function: Function to minimize
            initial_states: List of initial candidate states
            state_generator: Function to generate neighboring states
            max_iterations: Maximum optimization iterations
            collapse_threshold: Fraction of best states to keep after measurement
            
        Returns:
            OptimizationResults with best solution found
        """
        start_time = time.time()
        
        # Initialize superposition
        superposition_states = initial_states.copy()
        superposition_energies = [cost_function(state) for state in superposition_states]
        n_states = len(superposition_states)
        
        energy_history = [min(superposition_energies)]
        measurement_count = 0
        
        for iteration in range(max_iterations):
            # Evolve each state in superposition
            if self.enable_parallel and n_states > 1:
                new_states, new_energies = self._parallel_evolution(
                    superposition_states, cost_function, state_generator
                )
            else:
                new_states, new_energies = self._sequential_evolution(
                    superposition_states, cost_function, state_generator
                )
                
            # Apply quantum interference
            interfered_states = self._apply_quantum_interference(new_states, new_energies)
            interfered_energies = [cost_function(state) for state in interfered_states]
            
            # Measurement and collapse
            if iteration % 10 == 0:  # Periodic measurement
                measurement_count += 1
                sorted_indices = np.argsort(interfered_energies)
                n_keep = max(1, int(len(sorted_indices) * collapse_threshold))
                
                superposition_states = [interfered_states[i] for i in sorted_indices[:n_keep]]
                superposition_energies = [interfered_energies[i] for i in sorted_indices[:n_keep]]
                
                # Restore superposition by adding perturbed states
                while len(superposition_states) < n_states:
                    base_idx = np.random.randint(len(superposition_states))
                    base_state = superposition_states[base_idx]
                    perturbed_state = state_generator(base_state)
                    perturbed_energy = cost_function(perturbed_state)
                    
                    superposition_states.append(perturbed_state)
                    superposition_energies.append(perturbed_energy)
            else:
                superposition_states = interfered_states
                superposition_energies = interfered_energies
                
            energy_history.append(min(superposition_energies))
            
        # Final measurement
        best_idx = np.argmin(superposition_energies)
        best_state = superposition_states[best_idx]
        best_energy = superposition_energies[best_idx]
        
        optimization_time = time.time() - start_time
        
        return OptimizationResults(
            assignment=best_state,
            energy=best_energy,
            convergence=len(energy_history),
            energy_history=energy_history,
            execution_plan=[],
            optimization_time=optimization_time,
            quantum_coherence=self._calculate_coherence(),
            measurement_count=measurement_count,
            superposition_states=len(initial_states)
        )
        
    def hybrid_optimization(self,
                          cost_function: Callable[[Any], float],
                          initial_state: Any,
                          state_generator: Callable[[Any], Any],
                          quantum_iterations: int = 500,
                          classical_iterations: int = 500) -> OptimizationResults:
        """
        Hybrid quantum-classical optimization.
        
        Combines quantum annealing for global exploration with classical
        local search for refinement.
        """
        # Phase 1: Quantum annealing for global exploration
        schedule = AnnealingSchedule(
            initial_temp=100.0,
            final_temp=1.0,
            schedule_type="exponential",
            max_iterations=quantum_iterations
        )
        
        quantum_result = self.quantum_annealing(
            cost_function, initial_state, state_generator, schedule
        )
        
        # Phase 2: Classical local search for refinement
        current_state = quantum_result.assignment
        current_energy = quantum_result.energy
        
        classical_energy_history = []
        
        for _ in range(classical_iterations):
            candidate_state = state_generator(current_state)
            candidate_energy = cost_function(candidate_state)
            
            if candidate_energy < current_energy:
                current_state = candidate_state
                current_energy = candidate_energy
                
            classical_energy_history.append(current_energy)
            
        # Combine results
        combined_energy_history = quantum_result.energy_history + classical_energy_history
        
        return OptimizationResults(
            assignment=current_state,
            energy=current_energy,
            convergence=len(combined_energy_history),
            energy_history=combined_energy_history,
            execution_plan=[],
            optimization_time=quantum_result.optimization_time,
            quantum_coherence=quantum_result.quantum_coherence,
            measurement_count=quantum_result.measurement_count
        )
        
    def _quantum_tunneling_move(self, current_state: Any, state_generator: Callable) -> Any:
        """
        Generate quantum tunneling move for large state space exploration.
        """
        # Apply multiple perturbations to simulate tunneling through barriers
        tunneled_state = current_state
        
        n_tunneling_steps = np.random.poisson(3) + 1  # Average 3 steps
        for _ in range(n_tunneling_steps):
            tunneled_state = state_generator(tunneled_state)
            
        return tunneled_state
        
    def _quantum_accept(self, current_energy: float, candidate_energy: float, 
                       temperature: float, tunneling_rate: float) -> bool:
        """
        Quantum-enhanced acceptance probability.
        """
        if candidate_energy <= current_energy:
            return True
            
        if temperature <= 0:
            return False
            
        delta_energy = candidate_energy - current_energy
        
        # Classical Boltzmann factor
        classical_prob = math.exp(-delta_energy / temperature)
        
        # Quantum tunneling enhancement
        tunneling_enhancement = 1 + tunneling_rate * math.exp(-delta_energy / (temperature * 0.5))
        
        quantum_prob = min(1.0, classical_prob * tunneling_enhancement)
        
        return np.random.random() < quantum_prob
        
    def _parallel_evolution(self, states: List[Any], cost_function: Callable, 
                          state_generator: Callable) -> tuple:
        """
        Evolve states in parallel using thread pool.
        """
        new_states = []
        new_energies = []
        
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Submit evolution tasks
            future_to_state = {executor.submit(self._evolve_single_state, state, 
                                             cost_function, state_generator): state 
                              for state in states}
            
            # Collect results
            for future in as_completed(future_to_state):
                try:
                    evolved_state, evolved_energy = future.result()
                    new_states.append(evolved_state)
                    new_energies.append(evolved_energy)
                except Exception as e:
                    logger.warning(f"Error in parallel evolution: {e}")
                    original_state = future_to_state[future]
                    new_states.append(original_state)
                    new_energies.append(cost_function(original_state))
                    
        return new_states, new_energies
        
    def _sequential_evolution(self, states: List[Any], cost_function: Callable,
                            state_generator: Callable) -> tuple:
        """
        Evolve states sequentially.
        """
        new_states = []
        new_energies = []
        
        for state in states:
            evolved_state, evolved_energy = self._evolve_single_state(
                state, cost_function, state_generator
            )
            new_states.append(evolved_state)
            new_energies.append(evolved_energy)
            
        return new_states, new_energies
        
    def _evolve_single_state(self, state: Any, cost_function: Callable,
                           state_generator: Callable) -> tuple:
        """
        Evolve a single state.
        """
        evolved_state = state_generator(state)
        evolved_energy = cost_function(evolved_state)
        
        return evolved_state, evolved_energy
        
    def _apply_quantum_interference(self, states: List[Any], energies: List[float]) -> List[Any]:
        """
        Apply quantum interference effects between states.
        """
        if len(states) < 2:
            return states
            
        interfered_states = []
        
        for i, state in enumerate(states):
            # Select interfering state based on energy similarity
            energy_diffs = [abs(energies[i] - energies[j]) for j in range(len(energies)) if j != i]
            
            if energy_diffs:
                # Find most similar energy state for interference
                min_diff_idx = np.argmin(energy_diffs)
                if min_diff_idx >= i:
                    min_diff_idx += 1  # Adjust for skipped index
                    
                # Apply interference (simplified as weighted combination)
                if np.random.random() < 0.1:  # 10% chance of interference
                    interfered_state = self._interfere_states(state, states[min_diff_idx])
                    interfered_states.append(interfered_state)
                else:
                    interfered_states.append(state)
            else:
                interfered_states.append(state)
                
        return interfered_states
        
    def _interfere_states(self, state1: Any, state2: Any) -> Any:
        """
        Create interference between two quantum states.
        
        This is a simplified implementation - in practice, this would depend
        on the specific representation of the state space.
        """
        # For dict-based states (common in task scheduling)
        if isinstance(state1, dict) and isinstance(state2, dict):
            interfered_state = state1.copy()
            
            # Randomly blend some assignments
            for key in state1.keys():
                if key in state2 and np.random.random() < 0.3:
                    if isinstance(state1[key], dict) and 'start_time' in state1[key]:
                        # Blend start times
                        interfered_state[key] = state1[key].copy()
                        interfered_state[key]['start_time'] = (
                            0.7 * state1[key]['start_time'] + 0.3 * state2[key]['start_time']
                        )
                        
            return interfered_state
        else:
            # Default: return one of the states
            return state1 if np.random.random() < 0.5 else state2
            
    def _apply_decoherence(self, iteration: int):
        """
        Apply quantum decoherence effects.
        """
        # Exponential decay of coherence
        self.coherence_time *= (1 - self.decoherence_rate)
        
    def _calculate_coherence(self) -> float:
        """
        Calculate current quantum coherence level.
        """
        if self.coherence_time == float('inf'):
            return 1.0
        return max(0.0, min(1.0, self.coherence_time / 1000.0))
        
    def set_decoherence_rate(self, rate: float):
        """
        Set quantum decoherence rate.
        
        Args:
            rate: Decoherence rate (0.0 = no decoherence, 1.0 = immediate decoherence)
        """
        self.decoherence_rate = max(0.0, min(1.0, rate))
        
    def reset_quantum_state(self):
        """
        Reset quantum state to initial conditions.
        """
        self.quantum_register = None
        self.coherence_time = float('inf')
        self.decoherence_rate = 0.0
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization algorithm statistics.
        """
        return {
            'parallel_enabled': self.enable_parallel,
            'max_threads': self.max_threads,
            'quantum_coherence': self._calculate_coherence(),
            'decoherence_rate': self.decoherence_rate
        }
        
class AdaptiveOptimizer(QuantumOptimizer):
    """
    Adaptive optimizer that selects optimization strategy based on problem characteristics.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problem_history = []
        self.performance_history = []
        
    def optimize(self, 
                cost_function: Callable[[Any], float],
                initial_state: Any,
                state_generator: Callable[[Any], Any],
                problem_size: int,
                complexity_score: float = 1.0) -> OptimizationResults:
        """
        Adaptively select and apply optimization strategy.
        
        Args:
            cost_function: Function to minimize
            initial_state: Starting state
            state_generator: Function to generate neighboring states
            problem_size: Size of the problem (number of variables/tasks)
            complexity_score: Estimated problem complexity (0.0-1.0)
            
        Returns:
            OptimizationResults from selected strategy
        """
        strategy = self._select_strategy(problem_size, complexity_score)
        
        if strategy == "quantum_annealing":
            schedule = AnnealingSchedule(
                initial_temp=100.0 * complexity_score,
                final_temp=0.01,
                max_iterations=min(2000, problem_size * 100)
            )
            result = self.quantum_annealing(cost_function, initial_state, state_generator, schedule)
            
        elif strategy == "superposition_search":
            # Create multiple initial states
            initial_states = [initial_state]
            for _ in range(min(16, problem_size)):
                perturbed_state = state_generator(initial_state)
                initial_states.append(perturbed_state)
                
            result = self.superposition_search(
                cost_function, initial_states, state_generator,
                max_iterations=min(1000, problem_size * 50)
            )
            
        else:  # hybrid
            result = self.hybrid_optimization(
                cost_function, initial_state, state_generator,
                quantum_iterations=min(1000, problem_size * 30),
                classical_iterations=min(500, problem_size * 20)
            )
            
        # Record performance for future adaptive decisions
        self.problem_history.append({
            'size': problem_size,
            'complexity': complexity_score,
            'strategy': strategy
        })
        self.performance_history.append({
            'convergence_rate': result.get_convergence_rate(),
            'optimization_time': result.optimization_time,
            'final_energy': result.energy
        })
        
        return result
        
    def _select_strategy(self, problem_size: int, complexity_score: float) -> str:
        """
        Select optimization strategy based on problem characteristics and history.
        """
        # Simple heuristic-based selection
        if problem_size <= 5 and complexity_score < 0.3:
            return "superposition_search"
        elif problem_size <= 20 and complexity_score < 0.7:
            return "quantum_annealing"
        else:
            return "hybrid"
            
        # TODO: Implement ML-based strategy selection using performance history