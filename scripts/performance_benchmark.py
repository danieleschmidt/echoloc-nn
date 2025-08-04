#!/usr/bin/env python3
"""
Performance benchmarking and validation script for EchoLoc-NN.

Performs comprehensive performance testing including:
- Model inference benchmarks
- Signal processing performance
- Hardware interface latency
- Memory usage profiling
- Throughput measurements
- Scalability testing
"""

import os
import sys
import time
import gc
import json
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from echoloc_nn.models import EchoLocModel
from echoloc_nn.signal_processing import ChirpGenerator, EchoProcessor
from echoloc_nn.inference import EchoLocator, InferenceConfig
from echoloc_nn.utils.monitoring import PerformanceMonitor, PerformanceMetrics
from echoloc_nn.utils.logging_config import get_logger


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""
    
    # Test parameters
    num_warmup_iterations: int = 10
    num_test_iterations: int = 100
    batch_sizes: List[int] = None
    input_sizes: List[Tuple[int, int]] = None  # (n_sensors, n_samples)
    
    # System configuration
    use_gpu: bool = True
    enable_amp: bool = True  # Automatic Mixed Precision
    num_workers: int = 4
    
    # Benchmarks to run
    run_inference_benchmark: bool = True
    run_signal_processing_benchmark: bool = True
    run_memory_benchmark: bool = True
    run_concurrent_benchmark: bool = True
    run_scaling_benchmark: bool = True
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.input_sizes is None:
            self.input_sizes = [(4, 1024), (4, 2048), (4, 4096), (8, 2048)]


@dataclass
class BenchmarkResults:
    """Results from a single benchmark test."""
    
    test_name: str
    config: Dict[str, Any]
    
    # Timing results (milliseconds)
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    
    # Throughput
    throughput_hz: float
    throughput_samples_per_sec: Optional[float] = None
    
    # Resource usage
    peak_memory_mb: float
    avg_cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    
    # Additional metrics
    success_rate: float = 1.0
    error_count: int = 0
    additional_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for EchoLoc-NN.
    
    Tests inference speed, memory usage, throughput, and scalability
    across different configurations and scenarios.
    """
    
    def __init__(self, config: BenchmarkConfig, project_root: str = "."):
        self.config = config
        self.project_root = Path(project_root)
        self.logger = get_logger('performance_benchmark')
        
        # Initialize performance monitor
        self.perf_monitor = PerformanceMonitor()
        
        # Results storage
        self.results: Dict[str, BenchmarkResults] = {}
        
        # Device configuration
        self.device = self._setup_device()
        
        # Initialize components for testing
        self._initialize_components()
    
    def _setup_device(self) -> torch.device:
        """Setup computing device for benchmarks."""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU for benchmarks")
        
        return device
    
    def _initialize_components(self):
        """Initialize models and components for testing."""
        self.logger.info("Initializing benchmark components")
        
        # Create test model
        self.test_model = EchoLocModel(n_sensors=4, model_size="base")
        self.test_model.to(self.device)
        self.test_model.eval()
        
        # Create inference engine
        inference_config = InferenceConfig(
            device=str(self.device),
            enable_optimization=True,
            batch_size=1
        )
        self.locator = EchoLocator(config=inference_config, model=self.test_model)
        
        # Create signal processing components
        self.chirp_generator = ChirpGenerator()
        self.echo_processor = EchoProcessor()
    
    def run_all_benchmarks(self) -> Dict[str, BenchmarkResults]:
        """Run all configured benchmarks."""
        self.logger.info("Starting comprehensive performance benchmarks")
        
        # Start performance monitoring
        self.perf_monitor.start_monitoring()
        
        try:
            if self.config.run_inference_benchmark:
                self.results.update(self._run_inference_benchmarks())
            
            if self.config.run_signal_processing_benchmark:
                self.results.update(self._run_signal_processing_benchmarks())
            
            if self.config.run_memory_benchmark:
                self.results.update(self._run_memory_benchmarks())
            
            if self.config.run_concurrent_benchmark:
                self.results.update(self._run_concurrent_benchmarks())
            
            if self.config.run_scaling_benchmark:
                self.results.update(self._run_scaling_benchmarks())
        
        finally:
            # Stop performance monitoring
            self.perf_monitor.stop_monitoring()
        
        self.logger.info(f"Completed {len(self.results)} benchmark tests")
        return self.results
    
    def _run_inference_benchmarks(self) -> Dict[str, BenchmarkResults]:
        """Run model inference benchmarks."""
        self.logger.info("Running inference benchmarks")
        results = {}
        
        for batch_size in self.config.batch_sizes:
            for n_sensors, n_samples in self.config.input_sizes:
                test_name = f"inference_batch{batch_size}_sensors{n_sensors}_samples{n_samples}"
                
                self.logger.info(f"Running test: {test_name}")
                
                # Generate test data
                test_data = torch.randn(batch_size, n_sensors, n_samples, device=self.device)
                
                # Benchmark configuration
                test_config = {
                    'batch_size': batch_size,
                    'n_sensors': n_sensors,
                    'n_samples': n_samples,
                    'device': str(self.device),
                    'amp_enabled': self.config.enable_amp
                }
                
                # Run benchmark
                benchmark_result = self._benchmark_function(
                    test_name=test_name,
                    func=self._inference_test_function,
                    test_config=test_config,
                    test_data=test_data
                )
                
                # Calculate additional metrics
                benchmark_result.throughput_samples_per_sec = (
                    batch_size * n_samples * benchmark_result.throughput_hz
                )
                
                results[test_name] = benchmark_result
        
        return results
    
    def _inference_test_function(self, test_data: torch.Tensor, **kwargs) -> Any:
        """Test function for inference benchmarks."""
        with torch.no_grad():
            if self.config.enable_amp:
                with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                    return self.test_model(test_data)
            else:
                return self.test_model(test_data)
    
    def _run_signal_processing_benchmarks(self) -> Dict[str, BenchmarkResults]:
        """Run signal processing benchmarks."""
        self.logger.info("Running signal processing benchmarks")
        results = {}
        
        # Test chirp generation
        test_name = "chirp_generation"
        test_config = {
            'chirp_type': 'lfm',
            'f_start': 35000,
            'f_end': 45000,
            'duration': 0.005,
            'sample_rate': 250000
        }
        
        results[test_name] = self._benchmark_function(
            test_name=test_name,
            func=self._chirp_generation_test,
            test_config=test_config
        )
        
        # Test echo processing
        for n_sensors, n_samples in self.config.input_sizes:
            test_name = f"echo_processing_sensors{n_sensors}_samples{n_samples}"
            
            # Generate test echo data
            test_echo_data = np.random.randn(n_sensors, n_samples).astype(np.float32)
            
            test_config = {
                'n_sensors': n_sensors,
                'n_samples': n_samples,
                'sample_rate': 250000
            }
            
            results[test_name] = self._benchmark_function(
                test_name=test_name,
                func=self._echo_processing_test,
                test_config=test_config,
                test_data=test_echo_data
            )
        
        return results
    
    def _chirp_generation_test(self, **config) -> Any:
        """Test function for chirp generation."""
        return self.chirp_generator.generate_lfm_chirp(
            f_start=config['f_start'],
            f_end=config['f_end'],
            duration=config['duration'],
            sample_rate=config.get('sample_rate', 250000)
        )
    
    def _echo_processing_test(self, test_data: np.ndarray, **config) -> Any:
        """Test function for echo processing."""
        return self.echo_processor.enhance_echoes(
            test_data,
            sample_rate=config['sample_rate']
        )
    
    def _run_memory_benchmarks(self) -> Dict[str, BenchmarkResults]:
        """Run memory usage benchmarks."""
        self.logger.info("Running memory benchmarks")
        results = {}
        
        # Test memory usage scaling with batch size
        for batch_size in [1, 8, 16, 32, 64]:
            test_name = f"memory_batch{batch_size}"
            
            test_config = {
                'batch_size': batch_size,
                'n_sensors': 4,
                'n_samples': 2048
            }
            
            # Measure memory before test
            initial_memory = self._get_memory_usage()
            
            # Generate large test batch
            test_data = torch.randn(batch_size, 4, 2048, device=self.device)
            
            # Run inference and measure peak memory
            with torch.no_grad():
                _ = self.test_model(test_data)
            
            peak_memory = self._get_memory_usage()
            
            # Clean up
            del test_data
            torch.cuda.empty_cache() if self.device.type == 'cuda' else gc.collect()
            
            # Create benchmark result
            results[test_name] = BenchmarkResults(
                test_name=test_name,
                config=test_config,
                mean_time_ms=0,  # Not measuring time for memory test
                std_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                p95_time_ms=0,
                p99_time_ms=0,
                throughput_hz=0,
                peak_memory_mb=peak_memory['total_mb'],
                avg_cpu_percent=0,
                gpu_memory_mb=peak_memory.get('gpu_mb'),
                additional_metrics={
                    'initial_memory_mb': initial_memory['total_mb'],
                    'memory_increase_mb': peak_memory['total_mb'] - initial_memory['total_mb'],
                    'memory_per_sample_kb': (peak_memory['total_mb'] - initial_memory['total_mb']) * 1024 / (batch_size * 4 * 2048) if batch_size > 0 else 0
                }
            )
        
        return results
    
    def _run_concurrent_benchmarks(self) -> Dict[str, BenchmarkResults]:
        """Run concurrent processing benchmarks."""
        self.logger.info("Running concurrent benchmarks")
        results = {}
        
        for num_threads in [1, 2, 4, 8]:
            test_name = f"concurrent_threads{num_threads}"
            
            test_config = {
                'num_threads': num_threads,
                'batch_size': 1,
                'n_sensors': 4,
                'n_samples': 2048
            }
            
            # Prepare test data for each thread
            test_data_list = [
                torch.randn(1, 4, 2048, device=self.device)
                for _ in range(num_threads)
            ]
            
            results[test_name] = self._benchmark_concurrent_function(
                test_name=test_name,
                func=self._inference_test_function,
                test_config=test_config,
                test_data_list=test_data_list,
                num_threads=num_threads
            )
        
        return results
    
    def _run_scaling_benchmarks(self) -> Dict[str, BenchmarkResults]:
        """Run scalability benchmarks."""
        self.logger.info("Running scaling benchmarks")
        results = {}
        
        # Test input size scaling
        input_sizes = [(4, 512), (4, 1024), (4, 2048), (4, 4096), (4, 8192)]
        
        for n_sensors, n_samples in input_sizes:
            test_name = f"scaling_samples{n_samples}"
            
            test_data = torch.randn(1, n_sensors, n_samples, device=self.device)
            
            test_config = {
                'n_sensors': n_sensors,
                'n_samples': n_samples,
                'complexity_factor': n_samples / 1024  # Relative to baseline
            }
            
            results[test_name] = self._benchmark_function(
                test_name=test_name,
                func=self._inference_test_function,
                test_config=test_config,
                test_data=test_data
            )
        
        return results
    
    def _benchmark_function(
        self,
        test_name: str,
        func: callable,
        test_config: Dict[str, Any],
        test_data: Any = None,
        **kwargs
    ) -> BenchmarkResults:
        """Benchmark a single function."""
        
        # Warmup
        for _ in range(self.config.num_warmup_iterations):
            try:
                if test_data is not None:
                    func(test_data, **test_config, **kwargs)
                else:
                    func(**test_config, **kwargs)
            except Exception as e:
                self.logger.warning(f"Warmup iteration failed: {e}")
        
        # Clear caches
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        else:
            gc.collect()
        
        # Measure initial memory
        initial_memory = self._get_memory_usage()
        
        # Benchmark iterations
        times = []
        errors = 0
        peak_memory_mb = initial_memory['total_mb']
        
        for i in range(self.config.num_test_iterations):
            try:
                # Start timing
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                # Run function
                if test_data is not None:
                    result = func(test_data, **test_config, **kwargs)
                else:
                    result = func(**test_config, **kwargs)
                
                # End timing
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                # Record time
                iteration_time_ms = (end_time - start_time) * 1000
                times.append(iteration_time_ms)
                
                # Record peak memory
                current_memory = self._get_memory_usage()
                peak_memory_mb = max(peak_memory_mb, current_memory['total_mb'])
                
            except Exception as e:
                self.logger.warning(f"Benchmark iteration {i} failed: {e}")
                errors += 1
        
        # Calculate statistics
        if times:
            times_array = np.array(times)
            mean_time_ms = float(np.mean(times_array))
            std_time_ms = float(np.std(times_array))
            min_time_ms = float(np.min(times_array))
            max_time_ms = float(np.max(times_array))
            p95_time_ms = float(np.percentile(times_array, 95))
            p99_time_ms = float(np.percentile(times_array, 99))
            throughput_hz = 1000.0 / mean_time_ms
            success_rate = len(times) / self.config.num_test_iterations
        else:
            mean_time_ms = std_time_ms = min_time_ms = max_time_ms = 0
            p95_time_ms = p99_time_ms = throughput_hz = 0
            success_rate = 0
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        gpu_memory_mb = None
        if self.device.type == 'cuda':
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        return BenchmarkResults(
            test_name=test_name,
            config=test_config,
            mean_time_ms=mean_time_ms,
            std_time_ms=std_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            p95_time_ms=p95_time_ms,
            p99_time_ms=p99_time_ms,
            throughput_hz=throughput_hz,
            peak_memory_mb=peak_memory_mb,
            avg_cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory_mb,
            success_rate=success_rate,
            error_count=errors
        )
    
    def _benchmark_concurrent_function(
        self,
        test_name: str,
        func: callable,
        test_config: Dict[str, Any],
        test_data_list: List[Any],
        num_threads: int
    ) -> BenchmarkResults:
        """Benchmark concurrent function execution."""
        
        def worker_function(test_data):
            """Worker function for concurrent execution."""
            try:
                return func(test_data, **test_config)
            except Exception as e:
                self.logger.warning(f"Worker function failed: {e}")
                return None
        
        # Warmup
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(self.config.num_warmup_iterations):
                futures = [executor.submit(worker_function, data) for data in test_data_list]
                for future in futures:
                    future.result()
        
        # Benchmark
        times = []
        errors = 0
        initial_memory = self._get_memory_usage()
        peak_memory_mb = initial_memory['total_mb']
        
        for i in range(self.config.num_test_iterations):
            try:
                start_time = time.perf_counter()
                
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(worker_function, data) for data in test_data_list]
                    results = [future.result() for future in futures]
                
                end_time = time.perf_counter()
                
                # Check for failures
                failed_results = sum(1 for r in results if r is None)
                if failed_results > 0:
                    errors += failed_results
                
                # Record successful iteration time
                if failed_results < len(results):
                    iteration_time_ms = (end_time - start_time) * 1000
                    times.append(iteration_time_ms)
                
                # Update peak memory
                current_memory = self._get_memory_usage()
                peak_memory_mb = max(peak_memory_mb, current_memory['total_mb'])
                
            except Exception as e:
                self.logger.warning(f"Concurrent benchmark iteration {i} failed: {e}")
                errors += 1
        
        # Calculate statistics
        if times:
            times_array = np.array(times)
            mean_time_ms = float(np.mean(times_array))
            std_time_ms = float(np.std(times_array))
            min_time_ms = float(np.min(times_array))
            max_time_ms = float(np.max(times_array))
            p95_time_ms = float(np.percentile(times_array, 95))
            p99_time_ms = float(np.percentile(times_array, 99))
            throughput_hz = num_threads * 1000.0 / mean_time_ms  # Concurrent throughput
            success_rate = len(times) / self.config.num_test_iterations
        else:
            mean_time_ms = std_time_ms = min_time_ms = max_time_ms = 0
            p95_time_ms = p99_time_ms = throughput_hz = 0
            success_rate = 0
        
        cpu_percent = psutil.cpu_percent()
        gpu_memory_mb = None
        if self.device.type == 'cuda':
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        return BenchmarkResults(
            test_name=test_name,
            config=test_config,
            mean_time_ms=mean_time_ms,
            std_time_ms=std_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            p95_time_ms=p95_time_ms,
            p99_time_ms=p99_time_ms,
            throughput_hz=throughput_hz,
            peak_memory_mb=peak_memory_mb,
            avg_cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory_mb,
            success_rate=success_rate,
            error_count=errors,
            additional_metrics={
                'concurrent_threads': num_threads,
                'effective_throughput_per_thread': throughput_hz / num_threads if num_threads > 0 else 0
            }
        )
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = psutil.virtual_memory()
        
        result = {
            'total_mb': memory_info.used / 1024 / 1024,
            'percent': memory_info.percent
        }
        
        if self.device.type == 'cuda':
            result['gpu_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        
        return result
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate performance benchmark report."""
        report_lines = [
            "EchoLoc-NN Performance Benchmark Report",
            "=" * 50,
            f"Benchmark Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Device: {self.device}",
            f"Test Iterations: {self.config.num_test_iterations}",
            f"Warmup Iterations: {self.config.num_warmup_iterations}",
            ""
        ]
        
        # System information
        if self.device.type == 'cuda':
            report_lines.extend([
                "GPU Information:",
                f"  Name: {torch.cuda.get_device_name()}",
                f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
                ""
            ])
        
        # Results summary
        report_lines.extend([
            "PERFORMANCE SUMMARY",
            "-" * 30
        ])
        
        # Group results by category
        categories = {
            'Inference': [k for k in self.results.keys() if k.startswith('inference')],
            'Signal Processing': [k for k in self.results.keys() if 'chirp' in k or 'echo_processing' in k],
            'Memory': [k for k in self.results.keys() if k.startswith('memory')],
            'Concurrent': [k for k in self.results.keys() if k.startswith('concurrent')],
            'Scaling': [k for k in self.results.keys() if k.startswith('scaling')]
        }
        
        for category, test_names in categories.items():
            if not test_names:
                continue
                
            report_lines.extend([f"\n{category} Benchmarks:", "-" * (len(category) + 12)])
            
            for test_name in test_names:
                result = self.results[test_name]
                
                report_lines.extend([
                    f"\n{result.test_name}:",
                    f"  Mean Time: {result.mean_time_ms:.2f} ± {result.std_time_ms:.2f} ms",
                    f"  Throughput: {result.throughput_hz:.1f} Hz",
                    f"  Memory Peak: {result.peak_memory_mb:.1f} MB",
                    f"  Success Rate: {result.success_rate:.1%}"
                ])
                
                if result.gpu_memory_mb:
                    report_lines.append(f"  GPU Memory: {result.gpu_memory_mb:.1f} MB")
                
                if result.additional_metrics:
                    for key, value in result.additional_metrics.items():
                        if isinstance(value, (int, float)):
                            report_lines.append(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        
        # Performance targets and recommendations
        report_lines.extend([
            "",
            "PERFORMANCE ANALYSIS",
            "-" * 30
        ])
        
        # Analyze inference performance
        inference_results = [r for k, r in self.results.items() if k.startswith('inference')]
        if inference_results:
            fastest_inference = min(inference_results, key=lambda r: r.mean_time_ms)
            slowest_inference = max(inference_results, key=lambda r: r.mean_time_ms)
            
            report_lines.extend([
                f"Fastest Inference: {fastest_inference.test_name} ({fastest_inference.mean_time_ms:.2f} ms)",
                f"Slowest Inference: {slowest_inference.test_name} ({slowest_inference.mean_time_ms:.2f} ms)",
            ])
            
            # Performance targets
            real_time_target_ms = 50  # 50ms for real-time processing
            good_performance_count = sum(1 for r in inference_results if r.mean_time_ms <= real_time_target_ms)
            
            report_lines.extend([
                f"Real-time Performance ({real_time_target_ms}ms target): {good_performance_count}/{len(inference_results)} tests",
                ""
            ])
        
        # Memory analysis
        memory_results = [r for k, r in self.results.items() if k.startswith('memory')]
        if memory_results:
            max_memory = max(memory_results, key=lambda r: r.peak_memory_mb)
            report_lines.extend([
                f"Peak Memory Usage: {max_memory.peak_memory_mb:.1f} MB ({max_memory.test_name})",
                ""
            ])
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 20
        ])
        
        recommendations = []
        
        # Check inference performance
        if inference_results:
            avg_inference_time = np.mean([r.mean_time_ms for r in inference_results])
            if avg_inference_time > 100:
                recommendations.append("⚠️  Consider model optimization (quantization, pruning) for faster inference")
            
            if any(r.success_rate < 1.0 for r in inference_results):
                recommendations.append("⚠️  Address inference failures - check error handling")
        
        # Check memory usage
        if memory_results:
            max_memory_usage = max(r.peak_memory_mb for r in memory_results)
            if max_memory_usage > 2000:  # 2GB
                recommendations.append("⚠️  High memory usage detected - consider batch size optimization")
        
        # Check GPU utilization
        if self.device.type == 'cuda':
            gpu_results = [r for r in self.results.values() if r.gpu_memory_mb is not None]
            if gpu_results:
                avg_gpu_memory = np.mean([r.gpu_memory_mb for r in gpu_results])
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 * 1024  # MB
                gpu_utilization = avg_gpu_memory / total_gpu_memory
                
                if gpu_utilization < 0.3:
                    recommendations.append("💡 Low GPU memory utilization - consider larger batch sizes")
                elif gpu_utilization > 0.9:
                    recommendations.append("⚠️  High GPU memory utilization - monitor for OOM errors")
        
        if not recommendations:
            recommendations.append("✅ Performance looks good - no specific recommendations")
        
        report_lines.extend(recommendations)
        report_lines.extend(["", ""])
        
        report = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report)
            self.logger.info(f"Performance report written to: {output_file}")
        
        return report
    
    def export_results_json(self, output_file: str):
        """Export benchmark results as JSON."""
        json_data = {
            'benchmark_config': asdict(self.config),
            'system_info': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / 1024**3
            },
            'results': {
                name: asdict(result) for name, result in self.results.items()
            },
            'timestamp': time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark results exported to: {output_file}")


def main():
    """Main entry point for performance benchmarks."""
    parser = argparse.ArgumentParser(description="EchoLoc-NN Performance Benchmark")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--output",
        help="Output file for benchmark report"
    )
    parser.add_argument(
        "--json-output",
        help="JSON output file for raw results"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of test iterations (default: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run benchmarks on CPU only"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks (fewer test cases)"
    )
    
    args = parser.parse_args()
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        num_test_iterations=args.iterations,
        num_warmup_iterations=args.warmup,
        use_gpu=not args.cpu_only
    )
    
    if args.quick:
        config.batch_sizes = [1, 8, 16]
        config.input_sizes = [(4, 1024), (4, 2048)]
        config.num_test_iterations = 50
    
    # Initialize and run benchmarks
    benchmark = PerformanceBenchmark(config, args.project_root)
    
    try:
        print("Running EchoLoc-NN Performance Benchmarks...")
        print("=" * 50)
        
        # Run benchmarks
        results = benchmark.run_all_benchmarks()
        
        # Generate and display report
        report = benchmark.generate_report(args.output)
        if not args.output:
            print(report)
        
        # Export JSON results if requested
        if args.json_output:
            benchmark.export_results_json(args.json_output)
        
        print(f"\n✅ Benchmarks completed: {len(results)} tests run")
        
        # Check for performance issues
        failed_tests = [name for name, result in results.items() if result.success_rate < 1.0]
        if failed_tests:
            print(f"⚠️  {len(failed_tests)} tests had failures: {failed_tests}")
        
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()