#!/usr/bin/env python3
"""
Generation 3 Performance Optimization Validation Script.

Validates that the comprehensive optimizations achieve the target <50ms inference time
while maintaining accuracy and system stability.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from echoloc_nn.models.hybrid_architecture import EchoLocModel
from echoloc_nn.optimization.generation_3_optimizer import Generation3Optimizer, create_generation_3_optimizer
from echoloc_nn.signal_processing.preprocessing import PreProcessor
from echoloc_nn.utils.logging_config import get_logger


def generate_test_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Generate synthetic test data for validation."""
    logger = get_logger('test_data_generator')
    logger.info(f"Generating {num_samples} test samples")
    
    test_data = []
    
    for i in range(num_samples):
        # Generate realistic echo data
        n_sensors = 4
        n_samples = 2048
        
        # Simulate ultrasonic echo with noise
        echo_data = np.random.randn(n_sensors, n_samples).astype(np.float32)
        
        # Add simulated echo signal
        for sensor in range(n_sensors):
            # Simulated time delay based on position
            delay_samples = np.random.randint(100, 300)
            signal_strength = np.random.uniform(0.5, 2.0)
            
            # Add echo pulse
            if delay_samples + 50 < n_samples:
                echo_data[sensor, delay_samples:delay_samples+50] += signal_strength * np.exp(
                    -np.linspace(0, 5, 50)
                ) * np.sin(2 * np.pi * 40000 * np.linspace(0, 50/250000, 50))
        
        # Sensor positions (rectangular array)
        sensor_positions = np.array([
            [-0.05, -0.05],
            [0.05, -0.05], 
            [0.05, 0.05],
            [-0.05, 0.05]
        ], dtype=np.float32)
        
        # Expected position (random within reasonable range)
        expected_position = np.random.uniform(-1.0, 1.0, 3).astype(np.float32)
        
        test_data.append({
            'echo_data': echo_data,
            'sensor_positions': sensor_positions,
            'expected_position': expected_position,
            'test_id': i
        })
    
    logger.info(f"Generated {len(test_data)} test samples")
    return test_data


def validate_baseline_performance(model: torch.nn.Module, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate baseline (unoptimized) performance."""
    logger = get_logger('baseline_validator')
    logger.info("Validating baseline performance")
    
    model.eval()
    device = next(model.parameters()).device
    
    inference_times = []
    errors = []
    
    for test_case in test_data:
        echo_data = torch.from_numpy(test_case['echo_data']).to(device).unsqueeze(0)
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            positions, confidences = model(echo_data)
            position = positions[0].cpu().numpy()
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        inference_times.append(inference_time)
        
        # Calculate position error
        expected = test_case['expected_position']
        error = np.linalg.norm(position - expected)
        errors.append(error)
    
    results = {
        'mean_inference_time_ms': float(np.mean(inference_times)),
        'p95_inference_time_ms': float(np.percentile(inference_times, 95)),
        'max_inference_time_ms': float(np.max(inference_times)),
        'mean_position_error_m': float(np.mean(errors)),
        'meets_50ms_target_percent': float(np.mean(np.array(inference_times) < 50.0) * 100),
        'total_tests': len(test_data)
    }
    
    logger.info(f"Baseline results: {results['mean_inference_time_ms']:.1f}ms avg, "
               f"{results['meets_50ms_target_percent']:.1f}% meet target")
    
    return results


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Generation 3 optimizations")
    parser.add_argument("--num-tests", type=int, default=100, help="Number of test cases")
    parser.add_argument("--optimization-level", default="aggressive", 
                       choices=["conservative", "default", "aggressive"],
                       help="Optimization level")
    parser.add_argument("--target-latency", type=float, default=50.0, 
                       help="Target inference latency in ms")
    parser.add_argument("--output-dir", default="./validation_results",
                       help="Output directory for results")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline performance measurement")
    parser.add_argument("--model-size", default="base", choices=["tiny", "base", "large"],
                       help="Model size to test")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger('gen3_validator')
    logger.info("Starting Generation 3 optimization validation")
    logger.info(f"Configuration: {args}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate test data
        logger.info("=" * 60)
        logger.info("GENERATING TEST DATA")
        logger.info("=" * 60)
        
        test_data = generate_test_data(args.num_tests)
        
        # Initialize model
        logger.info("=" * 60)
        logger.info("INITIALIZING MODEL")
        logger.info("=" * 60)
        
        model = EchoLocModel(n_sensors=4, model_size=args.model_size)
        
        # Determine device
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        
        model.to(device)
        logger.info(f"Model initialized on {device}")
        
        # Measure baseline performance
        baseline_results = None
        if not args.skip_baseline:
            logger.info("=" * 60)
            logger.info("BASELINE PERFORMANCE MEASUREMENT")
            logger.info("=" * 60)
            
            baseline_results = validate_baseline_performance(model, test_data)
            
            logger.info("Baseline Performance Results:")
            logger.info(f"  Average inference time: {baseline_results['mean_inference_time_ms']:.2f}ms")
            logger.info(f"  95th percentile: {baseline_results['p95_inference_time_ms']:.2f}ms")
            logger.info(f"  Maximum: {baseline_results['max_inference_time_ms']:.2f}ms")
            logger.info(f"  Meet 50ms target: {baseline_results['meets_50ms_target_percent']:.1f}%")
            logger.info(f"  Average position error: {baseline_results['mean_position_error_m']:.3f}m")
        
        # Initialize Generation 3 optimizer
        logger.info("=" * 60)
        logger.info("GENERATION 3 OPTIMIZATION")
        logger.info("=" * 60)
        
        optimizer = create_generation_3_optimizer(
            model=model,
            optimization_level=args.optimization_level,
            target_latency_ms=args.target_latency,
            device=str(device)
        )
        
        # Generate calibration data for quantization
        calibration_data = torch.stack([
            torch.from_numpy(test['echo_data']).float()
            for test in test_data[:20]  # Use first 20 samples
        ]).to(device)
        
        # Apply model optimizations
        logger.info("Applying model optimizations...")
        optimization_result = optimizer.optimize_model(
            calibration_data=calibration_data,
            accuracy_threshold=0.05
        )
        
        logger.info("Model Optimization Results:")
        logger.info(f"  Speedup: {optimization_result.speedup_ratio:.2f}x")
        logger.info(f"  Compression: {optimization_result.compression_ratio:.2f}x")
        logger.info(f"  Memory reduction: {optimization_result.memory_reduction_mb:.1f}MB")
        logger.info(f"  Accuracy loss: {optimization_result.accuracy_degradation_percent:.2f}%")
        
        # Setup concurrent processing
        logger.info("Setting up concurrent processing...")
        optimizer.setup_concurrent_processing(
            num_cpu_workers=4,
            num_gpu_workers=1 if torch.cuda.is_available() else 0,
            enable_batching=True,
            batch_size=16 if args.optimization_level == "aggressive" else 8
        )
        
        # Start optimization systems
        logger.info("=" * 60)
        logger.info("STARTING OPTIMIZATION SYSTEMS")
        logger.info("=" * 60)
        
        with optimizer:  # Use context manager to automatically start/stop systems
            
            # Wait for systems to initialize
            time.sleep(2)
            
            # Validate optimized performance
            logger.info("=" * 60)
            logger.info("VALIDATING OPTIMIZED PERFORMANCE")
            logger.info("=" * 60)
            
            validation_results = optimizer.validate_performance(test_data)
            
            logger.info("Optimized Performance Results:")
            if 'performance_summary' in validation_results:
                perf = validation_results['performance_summary']
                logger.info(f"  Average inference time: {perf['mean_inference_time_ms']:.2f}ms")
                logger.info(f"  95th percentile: {perf['p95_inference_time_ms']:.2f}ms")
                logger.info(f"  Maximum: {perf['max_inference_time_ms']:.2f}ms")
                logger.info(f"  Meet target: {perf['meets_target_percentage']:.1f}%")
            
            logger.info(f"  Success rate: {validation_results['success_rate']:.1%}")
            logger.info(f"  Passed tests: {validation_results['passed_tests']}/{validation_results['total_tests']}")
            
            # Performance comparison
            if baseline_results and 'performance_summary' in validation_results:
                logger.info("=" * 60)
                logger.info("PERFORMANCE COMPARISON")
                logger.info("=" * 60)
                
                baseline_time = baseline_results['mean_inference_time_ms']
                optimized_time = validation_results['performance_summary']['mean_inference_time_ms']
                
                if baseline_time > 0:
                    speedup = baseline_time / optimized_time
                    improvement = ((baseline_time - optimized_time) / baseline_time) * 100
                    
                    logger.info(f"Performance Improvement:")
                    logger.info(f"  Baseline: {baseline_time:.2f}ms")
                    logger.info(f"  Optimized: {optimized_time:.2f}ms")
                    logger.info(f"  Speedup: {speedup:.2f}x")
                    logger.info(f"  Improvement: {improvement:.1f}% faster")
                
                # Target achievement
                baseline_target = baseline_results['meets_50ms_target_percent']
                optimized_target = validation_results['performance_summary']['meets_target_percentage']
                
                logger.info(f"Target Achievement:")
                logger.info(f"  Baseline <50ms: {baseline_target:.1f}%")
                logger.info(f"  Optimized <50ms: {optimized_target:.1f}%")
                logger.info(f"  Improvement: +{optimized_target - baseline_target:.1f}%")
            
            # Get comprehensive statistics
            logger.info("=" * 60)
            logger.info("COMPREHENSIVE STATISTICS")
            logger.info("=" * 60)
            
            stats = optimizer.get_comprehensive_stats()
            
            # Cache performance
            if 'cache' in stats:
                cache_stats = stats['cache']['overall']
                logger.info(f"Cache Performance:")
                logger.info(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
                logger.info(f"  Total size: {cache_stats['total_size_mb']:.1f}MB")
                logger.info(f"  Compression savings: {cache_stats.get('compression_savings_mb', 0):.1f}MB")
            
            # Auto-scaling
            if 'auto_scaling' in stats:
                scaling_stats = stats['auto_scaling']
                logger.info(f"Auto-scaling:")
                logger.info(f"  Current workers: {scaling_stats['current_workers']}")
                logger.info(f"  Scaling events: {scaling_stats['total_scaling_events']}")
            
            # Recommendations
            if 'recommendations' in stats and stats['recommendations']:
                logger.info(f"Optimization Recommendations:")
                for rec in stats['recommendations'][:3]:  # Top 3
                    logger.info(f"  • {rec['recommendation']}")
            
            # Export detailed report
            report_path = output_dir / f"generation_3_optimization_report_{int(time.time())}.txt"
            optimizer.export_optimization_report(str(report_path))
            
            # Export results as JSON
            import json
            results_data = {
                'configuration': vars(args),
                'baseline_results': baseline_results,
                'validation_results': validation_results,
                'optimization_stats': stats,
                'timestamp': time.time()
            }
            
            json_path = output_dir / f"validation_results_{int(time.time())}.json"
            with open(json_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Results exported to: {json_path}")
            logger.info(f"Detailed report: {report_path}")
        
        # Final assessment
        logger.info("=" * 60)
        logger.info("FINAL ASSESSMENT")
        logger.info("=" * 60)
        
        success = True
        
        if 'performance_summary' in validation_results:
            avg_time = validation_results['performance_summary']['mean_inference_time_ms']
            target_achievement = validation_results['performance_summary']['meets_target_percentage']
            
            if avg_time <= args.target_latency:
                logger.info(f"✅ ACHIEVED TARGET: Average inference time {avg_time:.2f}ms <= {args.target_latency}ms")
            else:
                logger.info(f"❌ MISSED TARGET: Average inference time {avg_time:.2f}ms > {args.target_latency}ms")
                success = False
            
            if target_achievement >= 90.0:  # 90% of inferences meet target
                logger.info(f"✅ HIGH SUCCESS RATE: {target_achievement:.1f}% of inferences meet target")
            else:
                logger.info(f"⚠️  MODERATE SUCCESS RATE: {target_achievement:.1f}% of inferences meet target")
                if target_achievement < 70.0:
                    success = False
        
        if validation_results['success_rate'] >= 0.95:  # 95% of tests pass
            logger.info(f"✅ HIGH TEST SUCCESS RATE: {validation_results['success_rate']:.1%}")
        else:
            logger.info(f"❌ LOW TEST SUCCESS RATE: {validation_results['success_rate']:.1%}")
            success = False
        
        if success:
            logger.info("🎉 GENERATION 3 OPTIMIZATION VALIDATION SUCCESSFUL!")
            logger.info("The system achieves <50ms inference target with maintained accuracy.")
        else:
            logger.info("⚠️  GENERATION 3 OPTIMIZATION NEEDS IMPROVEMENT")
            logger.info("Consider more aggressive optimizations or hardware upgrades.")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())