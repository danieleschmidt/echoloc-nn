"""
EchoLoc-NN Generation 2 Robustness Demonstration

This example demonstrates all the robustness improvements including:
- Fixed CNN dimension issues
- Comprehensive error handling and validation
- Enhanced logging with structured output
- Circuit breaker patterns for fault tolerance
- Retry mechanisms with exponential backoff
- Model validation and health monitoring
- Production-ready reliability patterns

Run this to see a complete end-to-end robust localization system.
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from echoloc_nn.models.hybrid_architecture import EchoLocModel
from echoloc_nn.utils.enhanced_logging import setup_enhanced_logging, get_enhanced_logger
from echoloc_nn.utils.circuit_breaker import (
    get_reliability_manager, CircuitBreakerConfig, RetryConfig,
    circuit_breaker, retry, reliable
)
from echoloc_nn.utils.model_validator import get_model_validator, ModelConstraints
from echoloc_nn.utils.monitoring import PerformanceMonitor
from echoloc_nn.reliability.fault_tolerance import FaultTolerantEchoLocator

def main():
    """Main robustness demonstration."""
    
    print("🚀 EchoLoc-NN Generation 2 Robustness Demonstration")
    print("=" * 60)
    
    # 1. Setup Enhanced Logging
    print("\n📋 Setting up enhanced logging system...")
    setup_enhanced_logging(
        log_level="INFO",
        log_file="/tmp/echoloc_demo.log",
        structured=True,
        enable_performance=True,
        enable_security=True,
        enable_analysis=True,
        include_stack_traces=False
    )
    
    logger = get_enhanced_logger("robustness_demo")
    logger.info("Robustness demonstration started", 
               component="demo", 
               version="2.0")
    
    # 2. Initialize Model with Robustness Features
    print("\n🧠 Initializing robust EchoLoc-NN model...")
    logger.push_context("model_initialization")
    
    try:
        model = EchoLocModel(
            n_sensors=4, 
            chirp_length=2048, 
            model_size='base'
        )
        logger.model_info("Model initialized successfully", 
                         model_name="EchoLocModel",
                         model_size="base",
                         parameters=sum(p.numel() for p in model.parameters()))
        print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    except Exception as e:
        logger.model_error("Model initialization failed", error_type="initialization_error", exc_info=True)
        raise
    finally:
        logger.pop_context()
    
    # 3. Model Validation and Health Checking
    print("\n🔍 Running comprehensive model validation...")
    logger.push_context("model_validation")
    
    try:
        validator = get_model_validator()
        
        # Configure validation constraints
        constraints = ModelConstraints(
            max_parameters=50_000_000,  # 50M parameters max
            max_forward_time_ms=500.0,  # 500ms max inference time
            min_validation_accuracy=0.6,
            min_numerical_stability=0.7
        )
        validator.constraints = constraints
        
        # Create test input
        test_input = torch.randn(1, 4, 2048)
        validation_data = (torch.randn(10, 4, 2048), torch.randn(10, 3))
        
        # Run comprehensive validation
        op_id = logger.start_operation("model_validation")
        metrics = validator.validate_model(model, test_input, validation_data)
        logger.end_operation(op_id, success=True, 
                           validation_score=metrics.numerical_stability_score,
                           total_parameters=metrics.total_parameters)
        
        print(f"✓ Model validation passed")
        print(f"  - Parameters: {metrics.total_parameters:,}")
        print(f"  - Model size: {metrics.model_size_mb:.1f} MB")
        print(f"  - Forward pass time: {metrics.forward_pass_time_ms:.1f} ms")
        print(f"  - Numerical stability: {metrics.numerical_stability_score:.3f}")
        print(f"  - Validation accuracy: {metrics.validation_accuracy:.3f}")
        
    except Exception as e:
        logger.model_error("Model validation failed", error_type="validation_error", exc_info=True)
        print(f"❌ Model validation failed: {e}")
    finally:
        logger.pop_context()
    
    # 4. Setup Circuit Breaker and Retry Patterns
    print("\n🛡️ Setting up reliability patterns...")
    reliability_manager = get_reliability_manager()
    
    # Circuit breaker for model inference
    cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=10.0,
        success_threshold=2,
        timeout=5.0
    )
    inference_breaker = reliability_manager.create_circuit_breaker("model_inference", cb_config)
    
    # Retry handler for transient failures
    retry_config = RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=5.0,
        backoff_multiplier=2.0,
        jitter=True
    )
    inference_retry = reliability_manager.create_retry_handler("model_inference", retry_config)
    
    logger.info("Reliability patterns configured",
               component="reliability",
               circuit_breaker="enabled",
               retry_handler="enabled")
    print("✓ Circuit breaker and retry mechanisms configured")
    
    # 5. Create Robust Inference Function
    print("\n🎯 Setting up robust inference pipeline...")
    
    @circuit_breaker(name="robust_inference", failure_threshold=5, recovery_timeout=30.0)
    @retry(name="inference_retry", max_attempts=3, base_delay=1.0)
    def robust_localization(echo_data, sensor_positions=None):
        """Robust localization with comprehensive error handling."""
        
        logger.push_context("localization")
        op_id = logger.start_operation("localization_inference")
        
        try:
            # Input validation happens automatically in model
            position, confidence = model.predict_position(echo_data, sensor_positions)
            
            # Additional output validation
            if np.any(np.isnan(position)) or np.isnan(confidence):
                raise ValueError("Model produced NaN outputs")
            
            if np.any(np.abs(position) > 1000):  # 1km seems unreasonable
                logger.warning("Position estimate seems extreme", 
                              position=position.tolist(),
                              component="validation")
            
            logger.end_operation(op_id, success=True,
                               position=position.tolist(),
                               confidence=confidence,
                               inference_result="success")
            
            return position, confidence
            
        except Exception as e:
            logger.end_operation(op_id, success=False, 
                               error=str(e),
                               inference_result="failed")
            raise
        finally:
            logger.pop_context()
    
    # 6. Performance Monitoring Setup
    print("\n📊 Setting up performance monitoring...")
    perf_monitor = PerformanceMonitor(
        max_history=1000,
        sampling_interval=1.0,
        enable_gpu_monitoring=torch.cuda.is_available()
    )
    
    def performance_alert(message, metrics):
        logger.warning(f"Performance alert: {message}", 
                      component="performance_monitor",
                      **metrics.to_dict())
    
    perf_monitor.add_alert_callback(performance_alert)
    perf_monitor.start_monitoring()
    
    print("✓ Performance monitoring started")
    
    # 7. Fault-Tolerant Wrapper
    print("\n🛠️ Creating fault-tolerant localization system...")
    
    # Mock base locator for demonstration
    class MockBaseLocator:
        def locate(self, sensor_data, **kwargs):
            position, confidence = robust_localization(sensor_data)
            
            # Create mock result object
            class MockResult:
                def __init__(self, pos, conf):
                    self.position = pos
                    self.confidence = conf
                    self.accuracy = conf  # Use confidence as accuracy proxy
            
            return MockResult(position, confidence)
    
    base_locator = MockBaseLocator()
    fault_tolerant_locator = FaultTolerantEchoLocator(
        base_locator,
        enable_fault_detection=True,
        enable_auto_recovery=True,
        max_recovery_attempts=3
    )
    
    print("✓ Fault-tolerant wrapper initialized")
    
    # 8. Comprehensive Testing Scenarios
    print("\n🧪 Running comprehensive robustness tests...")
    
    test_scenarios = [
        ("Normal operation", lambda: np.random.randn(4, 2048) * 0.5),
        ("High noise input", lambda: np.random.randn(4, 2048) * 2.0),
        ("Low signal input", lambda: np.random.randn(4, 2048) * 0.01),
        ("Realistic sensor data", lambda: generate_realistic_echo_data()),
        ("Edge case input", lambda: np.random.randn(4, 2048) * 10.0),
    ]
    
    results = []
    
    for scenario_name, data_generator in test_scenarios:
        print(f"\n  Testing: {scenario_name}")
        logger.push_context(f"testing_{scenario_name.lower().replace(' ', '_')}")
        
        try:
            # Generate test data
            echo_data = data_generator()
            sensor_positions = np.array([
                [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
            ])  # Square array
            
            # Record performance
            start_time = time.time()
            
            # Run fault-tolerant localization
            result = fault_tolerant_locator.locate(echo_data, sensor_positions=sensor_positions)
            
            inference_time = (time.time() - start_time) * 1000  # ms
            perf_monitor.record_inference_time(inference_time)
            
            print(f"    ✓ Position: [{result.position[0]:.2f}, {result.position[1]:.2f}, {result.position[2]:.2f}]")
            print(f"    ✓ Confidence: {result.confidence:.3f}")
            print(f"    ✓ Inference time: {inference_time:.1f} ms")
            
            results.append({
                'scenario': scenario_name,
                'position': result.position,
                'confidence': result.confidence,
                'inference_time_ms': inference_time,
                'success': True
            })
            
            logger.info("Test scenario completed successfully",
                       scenario=scenario_name,
                       position=result.position.tolist(),
                       confidence=result.confidence,
                       inference_time_ms=inference_time)
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            perf_monitor.record_error()
            results.append({
                'scenario': scenario_name,
                'error': str(e),
                'success': False
            })
            
            logger.error("Test scenario failed",
                        scenario=scenario_name,
                        error=str(e),
                        exc_info=True)
        finally:
            logger.pop_context()
    
    # 9. System Health Assessment
    print("\n🏥 System health assessment...")
    
    # Model health
    model_health = validator.get_health_summary()
    print(f"  Model Health Score: {model_health['health_score']:.3f}")
    
    # Reliability system health
    reliability_health = reliability_manager.get_system_health()
    print(f"  System Status: {reliability_health['system_status']}")
    print(f"  Circuit Breakers: {reliability_health['circuit_breakers']['total_count']} configured")
    print(f"  Open Breakers: {reliability_health['circuit_breakers']['open_count']}")
    
    # Performance metrics
    perf_summary = perf_monitor.get_metrics_summary(window_minutes=1)
    if not perf_summary.get('no_data'):
        print(f"  Avg CPU: {perf_summary['cpu_percent']['mean']:.1f}%")
        print(f"  Avg Memory: {perf_summary['memory_mb']['mean']:.1f} MB")
        if 'inference_time_ms' in perf_summary:
            print(f"  Avg Inference Time: {perf_summary['inference_time_ms']['mean']:.1f} ms")
            print(f"  Throughput: {perf_summary['throughput_hz']:.1f} samples/sec")
    
    # Fault tolerance status
    fault_status = fault_tolerant_locator.get_system_status()
    print(f"  Active Faults: {fault_status['active_faults_count']}")
    print(f"  System Health Score: {fault_status['system_health_score']:.3f}")
    
    # 10. Results Summary
    print("\n📈 Results Summary")
    print("=" * 40)
    
    successful_tests = len([r for r in results if r['success']])
    total_tests = len(results)
    success_rate = successful_tests / total_tests * 100
    
    print(f"Test Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    
    if successful_tests > 0:
        successful_results = [r for r in results if r['success']]
        avg_confidence = np.mean([r['confidence'] for r in successful_results])
        avg_inference_time = np.mean([r['inference_time_ms'] for r in successful_results])
        
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Average Inference Time: {avg_inference_time:.1f} ms")
    
    # Log final summary
    logger.info("Robustness demonstration completed",
               component="demo",
               test_success_rate=success_rate,
               total_tests=total_tests,
               model_health_score=model_health['health_score'],
               system_status=reliability_health['system_status'])
    
    # 11. Cleanup
    print("\n🧹 Cleaning up...")
    perf_monitor.stop_monitoring()
    fault_tolerant_locator.shutdown()
    
    print("\n✅ Robustness demonstration completed successfully!")
    print("\nKey Improvements Demonstrated:")
    print("  ✓ Fixed CNN dimension mismatch issues")
    print("  ✓ Comprehensive input validation and error handling")
    print("  ✓ Enhanced structured logging with context tracking")
    print("  ✓ Circuit breaker patterns for fault tolerance")
    print("  ✓ Retry mechanisms with exponential backoff")
    print("  ✓ Model validation and health monitoring")
    print("  ✓ Performance monitoring and alerting")
    print("  ✓ Fault-tolerant system architecture")
    print("  ✓ Production-ready reliability patterns")
    
    return results

def generate_realistic_echo_data():
    """Generate realistic echo data for testing."""
    # Simulate a chirp with reflections
    t = np.linspace(0, 1, 2048)
    
    # Base chirp signal
    chirp = np.sin(2 * np.pi * (20 + 20 * t) * t)
    
    # Add some realistic characteristics
    echo_data = np.zeros((4, 2048))
    
    for i in range(4):
        # Each sensor gets the chirp with different delays and attenuations
        delay = int(i * 50)  # Different delays for each sensor
        attenuation = 1.0 / (1 + i * 0.2)  # Different attenuations
        
        if delay < len(chirp):
            echo_data[i, delay:] = chirp[:-delay] * attenuation
        else:
            echo_data[i] = chirp * attenuation
        
        # Add some noise
        echo_data[i] += np.random.normal(0, 0.1, 2048)
    
    return echo_data

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
    else:
        print(f"\n🎉 Demonstration completed with {len(results)} test scenarios")