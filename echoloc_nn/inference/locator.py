"""
Real-time ultrasonic localization engine.
"""

from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import time
import numpy as np
try:
    import torch
except ImportError:
    from ..utils.torch_fallback import get_torch
    torch = get_torch()
import threading
from queue import Queue, Empty
import logging

try:
    from ..models.hybrid_architecture import EchoLocModel
    from ..models.pretrained import load_pretrained_model
except ImportError:
    # Fallback when models not available
    EchoLocModel = None
    load_pretrained_model = None
from ..signal_processing.preprocessing import PreProcessor
from ..hardware.ultrasonic_array import UltrasonicArray
from ..utils.robust_inference import RobustInferenceEngine
from ..optimization.performance_engine import PerformanceEngine, PerformanceConfig
try:
    from ..globalization.i18n import get_message
    from ..globalization.compliance import ComplianceManager, DataRetentionPolicy, PrivacyMetadata
    GLOBAL_FEATURES_AVAILABLE = True
except ImportError:
    GLOBAL_FEATURES_AVAILABLE = False
    def get_message(category, key, **kwargs):
        return f"{category}.{key}"


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    
    # Model parameters
    model_path: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda
    
    # Preprocessing
    preprocess_config: Dict[str, Any] = None
    
    # Performance
    batch_size: int = 1
    max_queue_size: int = 10
    inference_timeout: float = 0.1  # seconds
    
    # Output filtering
    position_smoothing: bool = True
    smoothing_alpha: float = 0.3
    confidence_threshold: float = 0.5
    
    # Real-time constraints  
    max_latency_ms: float = 50.0
    target_fps: float = 20.0
    
    # Performance optimization
    enable_caching: bool = True
    enable_concurrent_processing: bool = True
    cache_size: int = 500
    max_workers: int = 4
    
    def __post_init__(self):
        if self.preprocess_config is None:
            self.preprocess_config = {
                'remove_dc': True,
                'bandpass': {
                    'low_freq': 35000,
                    'high_freq': 45000,
                    'order': 4
                },
                'normalize': {'method': 'max'},
                'target_length': 2048
            }


class EchoLocator:
    """
    Real-time ultrasonic localization engine.
    
    Provides high-level interface for real-time position estimation
    with preprocessing, model inference, and output post-processing.
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[InferenceConfig] = None
    ):
        self.config = config or InferenceConfig()
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Load model
        if model is not None:
            self.model = model
        elif self.config.model_path:
            self.model = self._load_model(self.config.model_path)
        else:
            # Default model for demo
            if EchoLocModel is not None:
                self.model = EchoLocModel(n_sensors=4, chirp_length=2048)
            else:
                # Fallback mock model
                self.model = None
            
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        
        # Initialize components
        self.preprocessor = PreProcessor()
        self.robust_engine = RobustInferenceEngine(max_failures=5, failure_timeout=60.0)
        
        # Initialize performance engine
        perf_config = PerformanceConfig(
            cache_size=self.config.cache_size,
            enable_cache=self.config.enable_caching,
            enable_concurrent_processing=self.config.enable_concurrent_processing,
            max_workers=self.config.max_workers,
            target_latency_ms=self.config.max_latency_ms,
            enable_auto_scaling=True
        )
        self.performance_engine = PerformanceEngine(perf_config)
        
        # State tracking
        self.last_position = np.array([0.0, 0.0, 0.0])
        self.position_history = []
        self.is_initialized = False
        
        # Performance monitoring
        self.inference_times = []
        self.total_inferences = 0
        
        # Logging
        self.logger = logging.getLogger('EchoLocator')
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"EchoLocator initialized on {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        
    def _load_model(self, model_path: str) -> EchoLocModel:
        """Load model from checkpoint."""
        try:
            if hasattr(EchoLocModel, 'load_model'):
                return EchoLocModel.load_model(model_path, str(self.device))
            else:
                # Fallback loading
                checkpoint = torch.load(model_path, map_location=self.device)
                model = EchoLocModel()
                model.load_state_dict(checkpoint['state_dict'])
                return model
        except Exception as e:
            self.logger.warning(f"Failed to load model from {model_path}: {e}")
            self.logger.info("Using default model")
            return EchoLocModel(n_sensors=4, model_size="base")
    
    def locate(
        self,
        echo_data: np.ndarray,
        sensor_positions: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate position from echo data.
        
        Args:
            echo_data: Echo data (n_sensors, n_samples)
            sensor_positions: Sensor positions (n_sensors, 2)
            timestamp: Optional timestamp for the measurement
            
        Returns:
            Tuple of (position, confidence)
            - position: (x, y, z) coordinates in meters
            - confidence: Confidence score [0, 1]
        """
        start_time = time.time()
        
        def _optimized_inference(data: np.ndarray) -> Tuple[np.ndarray, float]:
            """Wrapper for optimized inference."""
            # Preprocess echo data
            processed_echo = self.preprocessor.preprocess_pipeline(
                data, self.config.preprocess_config
            )
            
            # Robust model inference with comprehensive error handling
            position, confidence, metadata = self.robust_engine.robust_inference(
                self.model, processed_echo, fallback_position=self.last_position
            )
            
            return position, confidence
        
        try:
            # Use performance-optimized inference
            position, confidence, perf_metadata = self.performance_engine.optimized_inference(
                _optimized_inference,
                echo_data,
                use_cache=True,
                use_concurrent=False  # Single request, no need for concurrency
            )
            
            # Log performance optimizations
            if perf_metadata.get('cache_hit'):
                self.logger.debug("Cache hit - using cached result")
            if perf_metadata.get('concurrent_execution'):
                self.logger.debug("Used concurrent processing")
                
            # Log optimization time
            opt_time = perf_metadata.get('optimization_time_ms', 0)
            if opt_time > 5:  # Only log if significant
                self.logger.debug(f"Performance optimization took {opt_time:.1f}ms")
            
            # Post-process results
            if self.config.position_smoothing and self.is_initialized:
                position = self._smooth_position(position)
            
            # Apply confidence threshold
            if confidence < self.config.confidence_threshold:
                self.logger.debug(f"Low confidence prediction: {confidence:.3f}")
                
            # Update state
            self.last_position = position
            self.is_initialized = True
            
            # Performance tracking
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times.append(inference_time)
            self.total_inferences += 1
            
            # Keep only recent inference times
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            # Log performance warnings
            if inference_time > self.config.max_latency_ms:
                self.logger.warning(
                    f"Inference latency {inference_time:.1f}ms exceeds "
                    f"target {self.config.max_latency_ms}ms"
                )
            
            return position, confidence
            
        except Exception as e:
            self.logger.error(f"Critical inference failure: {e}")
            # Return last known position with zero confidence
            return self.last_position, 0.0
    
    def locate_realtime(
        self,
        echo_data: np.ndarray,
        sensor_positions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Real-time localization with extended information.
        
        Args:
            echo_data: Echo data (n_sensors, n_samples)
            sensor_positions: Sensor positions (n_sensors, 2)
            
        Returns:
            Dictionary with position, confidence, and metadata
        """
        timestamp = time.time()
        position, confidence = self.locate(echo_data, sensor_positions, timestamp)
        
        # Calculate velocity estimate
        velocity = self._estimate_velocity()
        
        # Performance metrics
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        return {
            'position': position,
            'confidence': confidence,
            'velocity': velocity,
            'timestamp': timestamp,
            'inference_time_ms': self.inference_times[-1] if self.inference_times else 0,
            'avg_inference_time_ms': avg_inference_time,
            'total_inferences': self.total_inferences
        }
    
    def _smooth_position(self, new_position: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to position estimates."""
        alpha = self.config.smoothing_alpha
        smoothed = alpha * new_position + (1 - alpha) * self.last_position
        return smoothed
    
    def _estimate_velocity(self) -> np.ndarray:
        """Estimate velocity from position history."""
        if len(self.position_history) < 2:
            return np.array([0.0, 0.0, 0.0])
        
        # Simple finite difference
        dt = 1.0 / self.config.target_fps  # Assume regular sampling
        velocity = (self.last_position - self.position_history[-1]) / dt
        
        return velocity
    
    def start_continuous_localization(
        self,
        array: UltrasonicArray,
        output_queue: Optional[Queue] = None,
        stop_event: Optional[threading.Event] = None
    ):
        """
        Start continuous localization from ultrasonic array.
        
        Args:
            array: Connected ultrasonic array
            output_queue: Queue for position results
            stop_event: Event to stop continuous localization
        """
        if not array.is_connected:
            raise RuntimeError("Ultrasonic array not connected")
        
        if output_queue is None:
            output_queue = Queue(maxsize=self.config.max_queue_size)
            
        if stop_event is None:
            stop_event = threading.Event()
        
        self.logger.info("Starting continuous localization")
        
        # Start array streaming
        array.start_streaming(self.config.target_fps)
        
        try:
            for echo_data_dict in array.stream_chirps():
                if stop_event.is_set():
                    break
                
                # Extract data
                echo_data = echo_data_dict['echo_data']
                sensor_positions = echo_data_dict['sensor_positions']
                timestamp = echo_data_dict['timestamp']
                
                # Localize
                result = self.locate_realtime(echo_data, sensor_positions)
                result['array_timestamp'] = timestamp
                
                # Put in output queue
                try:
                    output_queue.put_nowait(result)
                except:
                    # Queue full, remove oldest item
                    try:
                        output_queue.get_nowait()
                        output_queue.put_nowait(result)
                    except Empty:
                        pass
                        
        except KeyboardInterrupt:
            self.logger.info("Continuous localization interrupted")
        finally:
            array.stop_streaming()
            self.logger.info("Continuous localization stopped")
    
    def calibrate_with_known_positions(
        self,
        calibration_data: List[Dict[str, Any]],
        method: str = "least_squares"
    ) -> Dict[str, Any]:
        """
        Calibrate localization system with known positions.
        
        Args:
            calibration_data: List of {echo_data, true_position} dictionaries
            method: Calibration method ("least_squares", "robust")
            
        Returns:
            Calibration results and error statistics
        """
        predicted_positions = []
        true_positions = []
        confidences = []
        
        self.logger.info(f"Calibrating with {len(calibration_data)} samples")
        
        for sample in calibration_data:
            echo_data = sample['echo_data']
            true_pos = sample['true_position']
            
            pred_pos, confidence = self.locate(echo_data)
            
            predicted_positions.append(pred_pos)
            true_positions.append(true_pos)
            confidences.append(confidence)
        
        predicted_positions = np.array(predicted_positions)
        true_positions = np.array(true_positions)
        confidences = np.array(confidences)
        
        # Compute errors
        errors = np.linalg.norm(predicted_positions - true_positions, axis=1)
        errors_cm = errors * 100  # Convert to cm
        
        # Error statistics
        calibration_results = {
            'mean_error_cm': np.mean(errors_cm),
            'median_error_cm': np.median(errors_cm),
            'std_error_cm': np.std(errors_cm),
            'max_error_cm': np.max(errors_cm),
            'p95_error_cm': np.percentile(errors_cm, 95),
            'accuracy_5cm': np.mean(errors_cm < 5.0) * 100,
            'accuracy_10cm': np.mean(errors_cm < 10.0) * 100,
            'mean_confidence': np.mean(confidences),
            'n_samples': len(calibration_data)
        }
        
        # Bias correction (if requested)
        if method == "least_squares":
            bias = np.mean(predicted_positions - true_positions, axis=0)
            calibration_results['bias'] = bias
            
            self.logger.info(f"Position bias: {bias}")
            # Could apply bias correction to future predictions
        
        # Log results
        self.logger.info("Calibration Results:")
        self.logger.info(f"  Mean error: {calibration_results['mean_error_cm']:.2f} cm")
        self.logger.info(f"  Median error: {calibration_results['median_error_cm']:.2f} cm")
        self.logger.info(f"  95th percentile: {calibration_results['p95_error_cm']:.2f} cm")
        self.logger.info(f"  Accuracy <5cm: {calibration_results['accuracy_5cm']:.1f}%")
        self.logger.info(f"  Accuracy <10cm: {calibration_results['accuracy_10cm']:.1f}%")
        
        return calibration_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.inference_times:
            return {'no_data': True}
        
        inference_times = np.array(self.inference_times)
        
        return {
            'total_inferences': self.total_inferences,
            'avg_inference_time_ms': np.mean(inference_times),
            'median_inference_time_ms': np.median(inference_times),
            'p95_inference_time_ms': np.percentile(inference_times, 95),
            'max_inference_time_ms': np.max(inference_times),
            'current_fps': 1000.0 / np.mean(inference_times) if len(inference_times) > 0 else 0,
            'target_fps': self.config.target_fps,
            'meets_latency_target': np.mean(inference_times) < self.config.max_latency_ms
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        performance_stats = self.get_performance_stats()
        robust_health = self.robust_engine.get_health_status()
        perf_engine_stats = self.performance_engine.get_performance_stats()
        
        # Combine statistics
        health_status = {
            **performance_stats,
            **robust_health,
            **perf_engine_stats,
            "model_type": self.model.__class__.__name__,
            "device": str(self.device),
            "preprocessor_initialized": self.preprocessor is not None,
            "performance_engine_initialized": self.performance_engine is not None,
            "last_position": self.last_position.tolist(),
        }
        
        # Overall health assessment
        issues = []
        if robust_health.get('circuit_breaker_open'):
            issues.append("Circuit breaker is open")
        if robust_health.get('success_rate', 1.0) < 0.8:
            issues.append("Low success rate")
        if performance_stats.get('avg_inference_time_ms', 0) > self.config.max_latency_ms:
            issues.append("High latency")
        
        # Performance optimization health
        cache_hit_rate = perf_engine_stats.get('cache_hit_rate', 0)
        if cache_hit_rate < 0.1 and perf_engine_stats.get('cache_total_requests', 0) > 10:
            issues.append("Low cache efficiency")
        
        health_status['issues'] = issues
        health_status['overall_status'] = 'healthy' if not issues else 'degraded'
        
        return health_status
    
    def reset_state(self):
        """Reset internal state."""
        self.last_position = np.array([0.0, 0.0, 0.0])
        self.position_history = []
        self.is_initialized = False
        self.inference_times = []
        self.total_inferences = 0
        
        self.logger.info("EchoLocator state reset")
    
    def optimize_for_device(self, target_device: str = "cpu"):
        """Optimize model for target device."""
        self.logger.info(f"Optimizing model for {target_device}")
        
        if target_device == "cpu":
            # Convert to CPU and optimize
            self.model = self.model.cpu()
            self.device = torch.device("cpu")
            
            # Apply CPU-specific optimizations
            torch.set_num_threads(4)  # Limit thread usage
            
        elif target_device == "cuda":
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = torch.device("cuda")
                
                # Enable cuDNN benchmarking
                torch.backends.cudnn.benchmark = True
            else:
                self.logger.warning("CUDA not available, staying on CPU")
        
        self.logger.info(f"Model now on {self.device}")
    
    def save_session(self, session_path: str):
        """Save current session state."""
        session_data = {
            'config': self.config,
            'last_position': self.last_position,
            'position_history': self.position_history,
            'performance_stats': self.get_performance_stats(),
            'is_initialized': self.is_initialized
        }
        
        import pickle
        with open(session_path, 'wb') as f:
            pickle.dump(session_data, f)
            
        self.logger.info(f"Session saved to {session_path}")
    
    def load_session(self, session_path: str):
        """Load session state."""
        import pickle
        with open(session_path, 'rb') as f:
            session_data = pickle.load(f)
        
        # Restore state
        self.last_position = session_data['last_position']
        self.position_history = session_data['position_history']
        self.is_initialized = session_data['is_initialized']
        
        self.logger.info(f"Session loaded from {session_path}")


class MultiSensorLocator:
    """
    Multi-sensor fusion localization system.
    
    Combines multiple EchoLocator instances for improved
    accuracy and redundancy.
    """
    
    def __init__(self, locators: List[EchoLocator], fusion_method: str = "weighted_average"):
        self.locators = locators
        self.fusion_method = fusion_method
        self.logger = logging.getLogger('MultiSensorLocator')
        
    def locate(self, echo_data_list: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Fused localization from multiple sensors.
        
        Args:
            echo_data_list: List of echo data from different sensor arrays
            
        Returns:
            Fused position and confidence
        """
        if len(echo_data_list) != len(self.locators):
            raise ValueError("Number of echo data arrays must match number of locators")
        
        positions = []
        confidences = []
        
        # Get predictions from all locators
        for locator, echo_data in zip(self.locators, echo_data_list):
            pos, conf = locator.locate(echo_data)
            positions.append(pos)
            confidences.append(conf)
        
        positions = np.array(positions)
        confidences = np.array(confidences)
        
        # Fusion
        if self.fusion_method == "weighted_average":
            # Weight by confidence
            weights = confidences / np.sum(confidences)
            fused_position = np.sum(positions * weights[:, np.newaxis], axis=0)
            fused_confidence = np.mean(confidences)
            
        elif self.fusion_method == "median":
            fused_position = np.median(positions, axis=0)
            fused_confidence = np.median(confidences)
            
        elif self.fusion_method == "best_confidence":
            best_idx = np.argmax(confidences)
            fused_position = positions[best_idx]
            fused_confidence = confidences[best_idx]
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_position, fused_confidence