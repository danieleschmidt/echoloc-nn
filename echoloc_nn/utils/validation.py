"""
Enhanced Data Validation and Sanitization for EchoLoc-NN.

Provides comprehensive validation for quantum planning parameters,
ultrasonic positioning data, and system configurations.
"""

from typing import Union, List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import re
import math
from .exceptions import ValidationError

class ValidationLevel(Enum):
    BASIC = "basic"
    STRICT = "strict" 
    PARANOID = "paranoid"

@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Any = None
    
    def raise_if_invalid(self):
        """Raise ValidationError if validation failed."""
        if not self.is_valid:
            raise ValidationError(f"Validation failed: {'; '.join(self.errors)}")


class EchoDataValidator:
    """Validator for ultrasonic echo data."""
    
    @staticmethod
    def validate_echo_data(
        echo_data: Union[np.ndarray, torch.Tensor],
        expected_sensors: Optional[int] = None,
        expected_samples: Optional[int] = None,
        amplitude_range: Tuple[float, float] = (-10.0, 10.0),
        check_finite: bool = True
    ) -> bool:
        """
        Validate echo data array.
        
        Args:
            echo_data: Echo data array (n_sensors, n_samples)
            expected_sensors: Expected number of sensors
            expected_samples: Expected number of samples
            amplitude_range: Expected amplitude range
            check_finite: Check for infinite/NaN values
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Convert to numpy if torch tensor
        if isinstance(echo_data, torch.Tensor):
            data = echo_data.detach().cpu().numpy()
        else:
            data = echo_data
            
        # Check basic properties
        if not isinstance(data, np.ndarray):
            raise ValidationError(
                "Echo data must be numpy array or torch tensor",
                error_code="INVALID_TYPE"
            )
        
        if data.ndim != 2:
            raise ValidationError(
                f"Echo data must be 2D array, got {data.ndim}D",
                error_code="INVALID_DIMENSIONS",
                details={"shape": data.shape}
            )
        
        n_sensors, n_samples = data.shape
        
        # Check sensor count
        if expected_sensors is not None and n_sensors != expected_sensors:
            raise ValidationError(
                f"Expected {expected_sensors} sensors, got {n_sensors}",
                error_code="SENSOR_COUNT_MISMATCH",
                details={"expected": expected_sensors, "actual": n_sensors}
            )
        
        # Check sample count
        if expected_samples is not None and n_samples != expected_samples:
            raise ValidationError(
                f"Expected {expected_samples} samples, got {n_samples}",
                error_code="SAMPLE_COUNT_MISMATCH",
                details={"expected": expected_samples, "actual": n_samples}
            )
        
        # Check for finite values
        if check_finite and not np.all(np.isfinite(data)):
            non_finite_count = np.sum(~np.isfinite(data))
            raise ValidationError(
                f"Echo data contains {non_finite_count} non-finite values",
                error_code="NON_FINITE_VALUES",
                details={"non_finite_count": non_finite_count}
            )
        
        # Check amplitude range
        min_val, max_val = np.min(data), np.max(data)
        if min_val < amplitude_range[0] or max_val > amplitude_range[1]:
            raise ValidationError(
                f"Echo data amplitude [{min_val:.3f}, {max_val:.3f}] "
                f"outside expected range {amplitude_range}",
                error_code="AMPLITUDE_OUT_OF_RANGE",
                details={
                    "actual_range": (float(min_val), float(max_val)),
                    "expected_range": amplitude_range
                }
            )
        
        # Check for constant signals (likely hardware failure)
        for i in range(n_sensors):
            if np.std(data[i]) < 1e-6:
                raise ValidationError(
                    f"Sensor {i} data appears constant (std={np.std(data[i]):.2e})",
                    error_code="CONSTANT_SIGNAL",
                    details={"sensor_id": i, "std": float(np.std(data[i]))}
                )
        
        return True
    
    @staticmethod
    def validate_chirp_signal(
        chirp_signal: Union[np.ndarray, torch.Tensor],
        expected_length: Optional[int] = None,
        frequency_range: Optional[Tuple[float, float]] = None,
        sample_rate: int = 250000
    ) -> bool:
        """
        Validate chirp signal.
        
        Args:
            chirp_signal: Chirp waveform
            expected_length: Expected signal length
            frequency_range: Expected frequency range (Hz)
            sample_rate: Sample rate for frequency analysis
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Convert to numpy if needed
        if isinstance(chirp_signal, torch.Tensor):
            signal = chirp_signal.detach().cpu().numpy()
        else:
            signal = chirp_signal
            
        if not isinstance(signal, np.ndarray):
            raise ValidationError(
                "Chirp signal must be numpy array or torch tensor",
                error_code="INVALID_TYPE"
            )
        
        if signal.ndim != 1:
            raise ValidationError(
                f"Chirp signal must be 1D array, got {signal.ndim}D",
                error_code="INVALID_DIMENSIONS"
            )
        
        # Check length
        if expected_length is not None and len(signal) != expected_length:
            raise ValidationError(
                f"Expected chirp length {expected_length}, got {len(signal)}",
                error_code="LENGTH_MISMATCH",
                details={"expected": expected_length, "actual": len(signal)}
            )
        
        # Check for finite values
        if not np.all(np.isfinite(signal)):
            raise ValidationError(
                "Chirp signal contains non-finite values",
                error_code="NON_FINITE_VALUES"
            )
        
        # Frequency domain validation
        if frequency_range is not None and len(signal) > 64:
            fft_signal = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft_signal)**2
            dominant_freq_idx = np.argmax(power_spectrum[:len(freqs)//2])
            dominant_freq = abs(freqs[dominant_freq_idx])
            
            if dominant_freq < frequency_range[0] or dominant_freq > frequency_range[1]:
                raise ValidationError(
                    f"Dominant frequency {dominant_freq:.0f} Hz outside "
                    f"expected range {frequency_range}",
                    error_code="FREQUENCY_OUT_OF_RANGE",
                    details={
                        "dominant_frequency": float(dominant_freq),
                        "expected_range": frequency_range
                    }
                )
        
        return True


class PositionValidator:
    """Validator for position data."""
    
    @staticmethod
    def validate_position(
        position: Union[np.ndarray, torch.Tensor, List, Tuple],
        dimensions: int = 3,
        bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
        check_finite: bool = True
    ) -> bool:
        """
        Validate position coordinates.
        
        Args:
            position: Position coordinates
            dimensions: Expected number of dimensions
            bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            check_finite: Check for finite values
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Convert to numpy array
        if isinstance(position, (list, tuple)):
            pos = np.array(position)
        elif isinstance(position, torch.Tensor):
            pos = position.detach().cpu().numpy()
        else:
            pos = position
            
        if not isinstance(pos, np.ndarray):
            raise ValidationError(
                "Position must be array-like",
                error_code="INVALID_TYPE"
            )
        
        # Check dimensions
        if pos.shape != (dimensions,):
            raise ValidationError(
                f"Expected position shape ({dimensions},), got {pos.shape}",
                error_code="INVALID_DIMENSIONS",
                details={"expected_shape": (dimensions,), "actual_shape": pos.shape}
            )
        
        # Check finite values
        if check_finite and not np.all(np.isfinite(pos)):
            raise ValidationError(
                "Position contains non-finite values",
                error_code="NON_FINITE_VALUES",
                details={"position": pos.tolist()}
            )
        
        # Check bounds
        if bounds is not None:
            if len(bounds) != dimensions:
                raise ValidationError(
                    f"Bounds must have {dimensions} tuples, got {len(bounds)}",
                    error_code="INVALID_BOUNDS"
                )
                
            for i, (pos_val, (min_val, max_val)) in enumerate(zip(pos, bounds)):
                if pos_val < min_val or pos_val > max_val:
                    coord_names = ['x', 'y', 'z']
                    coord_name = coord_names[i] if i < len(coord_names) else f'dim{i}'
                    
                    raise ValidationError(
                        f"Position {coord_name}={pos_val:.3f} outside "
                        f"bounds [{min_val}, {max_val}]",
                        error_code="POSITION_OUT_OF_BOUNDS",
                        details={
                            "coordinate": coord_name,
                            "value": float(pos_val),
                            "bounds": (min_val, max_val)
                        }
                    )
        
        return True
    
    @staticmethod
    def validate_sensor_positions(
        sensor_positions: Union[np.ndarray, List],
        n_sensors: Optional[int] = None,
        dimensions: int = 2,
        min_baseline: float = 0.01  # 1cm minimum spacing
    ) -> bool:
        """
        Validate sensor position array.
        
        Args:
            sensor_positions: Array of sensor positions (n_sensors, dimensions)
            n_sensors: Expected number of sensors
            dimensions: Expected position dimensions
            min_baseline: Minimum distance between sensors
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Convert to numpy array
        if isinstance(sensor_positions, list):
            positions = np.array(sensor_positions)
        else:
            positions = sensor_positions
            
        if not isinstance(positions, np.ndarray):
            raise ValidationError(
                "Sensor positions must be array-like",
                error_code="INVALID_TYPE"
            )
        
        # Check shape
        if positions.ndim != 2:
            raise ValidationError(
                f"Sensor positions must be 2D array, got {positions.ndim}D",
                error_code="INVALID_DIMENSIONS"
            )
        
        actual_sensors, actual_dims = positions.shape
        
        if n_sensors is not None and actual_sensors != n_sensors:
            raise ValidationError(
                f"Expected {n_sensors} sensors, got {actual_sensors}",
                error_code="SENSOR_COUNT_MISMATCH",
                details={"expected": n_sensors, "actual": actual_sensors}
            )
        
        if actual_dims != dimensions:
            raise ValidationError(
                f"Expected {dimensions}D positions, got {actual_dims}D",
                error_code="DIMENSION_MISMATCH",
                details={"expected": dimensions, "actual": actual_dims}
            )
        
        # Check finite values
        if not np.all(np.isfinite(positions)):
            raise ValidationError(
                "Sensor positions contain non-finite values",
                error_code="NON_FINITE_VALUES"
            )
        
        # Check minimum baseline
        for i in range(actual_sensors):
            for j in range(i + 1, actual_sensors):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < min_baseline:
                    raise ValidationError(
                        f"Sensors {i} and {j} too close: {distance:.4f}m "
                        f"(minimum: {min_baseline}m)",
                        error_code="INSUFFICIENT_BASELINE",
                        details={
                            "sensor_pair": (i, j),
                            "distance": float(distance),
                            "minimum": min_baseline
                        }
                    )
        
        return True


class SensorConfigValidator:
    """Validator for sensor configuration."""
    
    @staticmethod
    def validate_sensor_config(config: Dict[str, Any]) -> bool:
        """
        Validate sensor configuration dictionary.
        
        Args:
            config: Sensor configuration
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        required_fields = ['id', 'position', 'frequency']
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                raise ValidationError(
                    f"Missing required field: {field}",
                    error_code="MISSING_FIELD",
                    details={"field": field}
                )
        
        # Validate sensor ID
        sensor_id = config['id']
        if not isinstance(sensor_id, int) or sensor_id < 0:
            raise ValidationError(
                f"Sensor ID must be non-negative integer, got {sensor_id}",
                error_code="INVALID_SENSOR_ID",
                details={"sensor_id": sensor_id}
            )
        
        # Validate position
        PositionValidator.validate_position(
            config['position'], 
            dimensions=2,  # 2D positions for sensors
            bounds=((-10.0, 10.0), (-10.0, 10.0))  # Reasonable room bounds
        )
        
        # Validate frequency
        frequency = config['frequency']
        if not isinstance(frequency, (int, float)) or frequency <= 0:
            raise ValidationError(
                f"Frequency must be positive number, got {frequency}",
                error_code="INVALID_FREQUENCY",
                details={"frequency": frequency}
            )
        
        # Check frequency range (ultrasonic)
        if frequency < 20000 or frequency > 100000:
            raise ValidationError(
                f"Frequency {frequency} Hz outside ultrasonic range [20kHz, 100kHz]",
                error_code="FREQUENCY_OUT_OF_RANGE",
                details={"frequency": frequency}
            )
        
        # Validate optional fields
        if 'beam_width' in config:
            beam_width = config['beam_width']
            if not isinstance(beam_width, (int, float)) or beam_width <= 0 or beam_width > 180:
                raise ValidationError(
                    f"Beam width must be in (0, 180] degrees, got {beam_width}",
                    error_code="INVALID_BEAM_WIDTH",
                    details={"beam_width": beam_width}
                )
        
        if 'max_range' in config:
            max_range = config['max_range']
            if not isinstance(max_range, (int, float)) or max_range <= 0:
                raise ValidationError(
                    f"Max range must be positive, got {max_range}",
                    error_code="INVALID_MAX_RANGE",
                    details={"max_range": max_range}
                )
        
        if 'min_range' in config:
            min_range = config['min_range']
            if not isinstance(min_range, (int, float)) or min_range < 0:
                raise ValidationError(
                    f"Min range must be non-negative, got {min_range}",
                    error_code="INVALID_MIN_RANGE",
                    details={"min_range": min_range}
                )
            
            # Check range consistency
            if 'max_range' in config and min_range >= config['max_range']:
                raise ValidationError(
                    f"Min range {min_range} >= max range {config['max_range']}",
                    error_code="INVALID_RANGE_ORDER",
                    details={"min_range": min_range, "max_range": config['max_range']}
                )
        
        return True

class QuantumPlanningValidator:
    """
    Comprehensive validator for quantum-inspired task planning systems.
    
    Validates:
    - Task graph structure and dependencies
    - Resource allocation parameters
    - Quantum optimization settings
    - Position and movement constraints
    - Hardware configuration
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        
        # Validation thresholds
        self.position_bounds = (-1000.0, 1000.0)  # meters
        self.duration_bounds = (0.001, 86400.0)   # seconds (1ms to 24h)
        self.priority_bounds = (1, 100)
        self.confidence_bounds = (0.0, 1.0)
        self.coherence_bounds = (0.0, 1.0)
        self.temperature_bounds = (0.001, 1000.0)
        
    def validate_task_graph(self, task_graph) -> ValidationResult:
        """Validate task graph structure and consistency."""
        errors = []
        warnings = []
        
        # Check basic structure
        if not hasattr(task_graph, 'tasks') or not hasattr(task_graph, 'dependencies'):
            errors.append("Task graph missing required attributes")
            return ValidationResult(False, errors, warnings)
            
        tasks = task_graph.tasks
        dependencies = task_graph.dependencies
        
        # Validate tasks
        task_validation = self._validate_tasks(tasks)
        errors.extend(task_validation.errors)
        warnings.extend(task_validation.warnings)
        
        # Validate dependencies
        dep_validation = self._validate_dependencies(dependencies, tasks)
        errors.extend(dep_validation.errors)
        warnings.extend(dep_validation.warnings)
        
        # Check for cycles (critical)
        if hasattr(task_graph, 'validate_graph'):
            graph_issues = task_graph.validate_graph()
            for issue in graph_issues:
                if 'cycle' in issue.lower():
                    errors.append(f"Graph validation: {issue}")
                else:
                    warnings.append(f"Graph validation: {issue}")
                    
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def validate_quantum_config(self, config) -> ValidationResult:
        """Validate quantum planning configuration."""
        errors = []
        warnings = []
        
        # Validate strategy
        if hasattr(config, 'strategy'):
            valid_strategies = ['quantum_annealing', 'superposition_search', 'hybrid_classical', 'adaptive']
            strategy_name = config.strategy.value if hasattr(config.strategy, 'value') else str(config.strategy)
            if strategy_name not in valid_strategies:
                errors.append(f"Invalid planning strategy: {strategy_name}")
                
        # Validate quantum coherence parameters
        if hasattr(config, 'quantum_tunneling_rate'):
            rate_result = self._validate_probability(config.quantum_tunneling_rate)
            if not rate_result.is_valid:
                errors.extend([f"quantum_tunneling_rate: {e}" for e in rate_result.errors])
                
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def _validate_tasks(self, tasks) -> ValidationResult:
        """Validate list of tasks."""
        errors = []
        warnings = []
        
        if not tasks:
            warnings.append("Empty task list")
            return ValidationResult(True, errors, warnings)
            
        # Check for duplicate IDs
        task_ids = [task.id for task in tasks if hasattr(task, 'id')]
        if len(task_ids) != len(set(task_ids)):
            errors.append("Duplicate task IDs found")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def _validate_dependencies(self, dependencies, tasks) -> ValidationResult:
        """Validate task dependencies."""
        errors = []
        warnings = []
        
        if not dependencies:
            return ValidationResult(True, errors, warnings)
            
        task_ids = {task.id for task in tasks if hasattr(task, 'id')}
        
        for dep in dependencies:
            # Check dependency structure
            if not hasattr(dep, 'predecessor_id') or not hasattr(dep, 'successor_id'):
                errors.append("Dependency missing required attributes")
                continue
                
            # Check task existence
            if dep.predecessor_id not in task_ids:
                errors.append(f"Dependency references non-existent predecessor: {dep.predecessor_id}")
            if dep.successor_id not in task_ids:
                errors.append(f"Dependency references non-existent successor: {dep.successor_id}")
                
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def _validate_probability(self, probability) -> ValidationResult:
        """Validate probability value."""
        errors = []
        warnings = []
        
        if not isinstance(probability, (int, float)):
            errors.append(f"Probability must be numeric, got {type(probability)}")
            return ValidationResult(False, errors, warnings)
            
        if not 0.0 <= probability <= 1.0:
            errors.append(f"Probability must be in range [0, 1], got {probability}")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

# Global validator instances
_global_quantum_validator = QuantumPlanningValidator()

def get_global_quantum_validator() -> QuantumPlanningValidator:
    """Get the global quantum planning validator instance."""
    return _global_quantum_validator


class EnhancedModelValidator:
    """Validator for model configurations and states."""
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> bool:
        """
        Validate model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Check n_sensors
        if 'n_sensors' in config:
            n_sensors = config['n_sensors']
            if not isinstance(n_sensors, int) or n_sensors < 1 or n_sensors > 16:
                raise ValidationError(
                    f"n_sensors must be integer in [1, 16], got {n_sensors}",
                    error_code="INVALID_N_SENSORS",
                    details={"n_sensors": n_sensors}
                )
        
        # Check chirp_length
        if 'chirp_length' in config:
            chirp_length = config['chirp_length']
            if not isinstance(chirp_length, int) or chirp_length < 64 or chirp_length > 16384:
                raise ValueError(
                    f"chirp_length must be integer in [64, 16384], got {chirp_length}",
                    error_code="INVALID_CHIRP_LENGTH",
                    details={"chirp_length": chirp_length}
                )
        
        # Check model_size if present
        if 'model_size' in config:
            model_size = config['model_size']
            valid_sizes = ['tiny', 'base', 'large']
            if model_size not in valid_sizes:
                raise ValidationError(
                    f"model_size must be one of {valid_sizes}, got {model_size}",
                    error_code="INVALID_MODEL_SIZE",
                    details={"model_size": model_size, "valid_sizes": valid_sizes}
                )
        
        return True
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> bool:
        """
        Validate training configuration.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Check batch_size
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 1024:
                raise ValidationError(
                    f"batch_size must be integer in [1, 1024], got {batch_size}",
                    error_code="INVALID_BATCH_SIZE",
                    details={"batch_size": batch_size}
                )
        
        # Check learning_rate
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1.0:
                raise ValidationError(
                    f"learning_rate must be in (0, 1], got {lr}",
                    error_code="INVALID_LEARNING_RATE",
                    details={"learning_rate": lr}
                )
        
        # Check max_epochs
        if 'max_epochs' in config:
            epochs = config['max_epochs']
            if not isinstance(epochs, int) or epochs < 1 or epochs > 10000:
                raise ValidationError(
                    f"max_epochs must be integer in [1, 10000], got {epochs}",
                    error_code="INVALID_MAX_EPOCHS",
                    details={"max_epochs": epochs}
                )
        
        return True