"""
Tests for signal processing components.
"""

import pytest
import numpy as np
import scipy.signal as signal
from unittest.mock import patch, MagicMock

from echoloc_nn.signal_processing import (
    ChirpGenerator,
    ChirpDesigner,
    EchoProcessor,
    EchoEnhancer,
    PreProcessor
)
from echoloc_nn.signal_processing.chirp_generator import ChirpType
from echoloc_nn.utils.exceptions import ValidationError


class TestChirpGenerator:
    """Test chirp generation functionality."""
    
    def test_initialization(self):
        """Test chirp generator initialization."""
        generator = ChirpGenerator(sample_rate=250000)
        assert generator.sample_rate == 250000
    
    def test_lfm_chirp_generation(self):
        """Test linear frequency modulated chirp generation."""
        generator = ChirpGenerator(sample_rate=250000)
        
        t, chirp = generator.generate_lfm_chirp(
            start_freq=35000,
            end_freq=45000,
            duration=0.005,
            window="hann"
        )
        
        # Check time array
        assert len(t) == int(0.005 * 250000)
        assert np.isclose(t[-1], 0.005, rtol=1e-3)
        
        # Check chirp signal
        assert len(chirp) == len(t)
        assert np.all(np.isfinite(chirp))
        assert not np.all(chirp == 0)
        
        # Check frequency content
        freqs = np.fft.fftfreq(len(chirp), 1/250000)
        fft_chirp = np.fft.fft(chirp)
        power = np.abs(fft_chirp)**2
        
        # Should have significant power in the chirp frequency range
        freq_mask = (np.abs(freqs) >= 35000) & (np.abs(freqs) <= 45000)
        in_band_power = np.sum(power[freq_mask])
        total_power = np.sum(power)
        
        assert in_band_power / total_power > 0.5  # Most power in band
    
    def test_hyperbolic_chirp_generation(self):
        """Test hyperbolic chirp generation."""
        generator = ChirpGenerator()
        
        t, chirp = generator.generate_hyperbolic_chirp(
            center_freq=40000,
            bandwidth=10000,
            duration=0.005
        )
        
        assert len(chirp) == len(t)
        assert np.all(np.isfinite(chirp))
        assert np.std(chirp) > 0  # Should not be constant
    
    def test_logarithmic_chirp_generation(self):
        """Test logarithmic chirp generation."""
        generator = ChirpGenerator()
        
        t, chirp = generator.generate_logarithmic_chirp(
            start_freq=35000,
            end_freq=45000,
            duration=0.005
        )
        
        assert len(chirp) == len(t)
        assert np.all(np.isfinite(chirp))
    
    def test_coded_chirp_generation(self):
        """Test coded chirp generation."""
        generator = ChirpGenerator()
        
        t, chirp = generator.generate_coded_chirp(
            code="barker13",
            carrier_freq=40000,
            chip_duration=0.0001
        )
        
        assert len(chirp) == len(t)
        assert np.all(np.isfinite(chirp))
        
        # Test different codes
        codes = ["barker13", "barker11", "barker7", "barker5"]
        for code in codes:
            t, chirp = generator.generate_coded_chirp(
                code=code,
                carrier_freq=40000,
                chip_duration=0.0001
            )
            assert len(chirp) > 0
    
    def test_invalid_code(self):
        """Test handling of invalid code types."""
        generator = ChirpGenerator()
        
        with pytest.raises(ValueError):
            generator.generate_coded_chirp(
                code="invalid_code",
                carrier_freq=40000,
                chip_duration=0.0001
            )
    
    def test_window_functions(self):
        """Test different window functions."""
        generator = ChirpGenerator()
        
        windows = ["hann", "hamming", "blackman", None]
        
        for window in windows:
            t, chirp = generator.generate_lfm_chirp(
                start_freq=35000,
                end_freq=45000,
                duration=0.005,
                window=window
            )
            
            assert len(chirp) > 0
            assert np.all(np.isfinite(chirp))
            
            if window is not None:
                # Windowed signals should taper to near zero at edges
                edge_samples = int(0.1 * len(chirp))  # 10% of signal
                assert np.abs(chirp[0]) < 0.5 * np.max(np.abs(chirp))
                assert np.abs(chirp[-1]) < 0.5 * np.max(np.abs(chirp))


class TestChirpDesigner:
    """Test chirp design optimization."""
    
    def test_initialization(self):
        """Test chirp designer initialization."""
        designer = ChirpDesigner(speed_of_sound=343.0)
        assert designer.speed_of_sound == 343.0
    
    def test_optimal_chirp_design(self):
        """Test optimal chirp parameter calculation."""
        designer = ChirpDesigner()
        
        optimal_params = designer.design_optimal_chirp(
            max_range=5.0,
            range_resolution=0.05,
            max_velocity=2.0,
            sample_rate=250000,
            transducer_bandwidth=(35000, 45000)
        )
        
        # Check required fields
        required_fields = [
            'start_freq', 'end_freq', 'center_freq', 'bandwidth',
            'duration', 'theoretical_range_resolution', 'recommended_chirp_type'
        ]
        
        for field in required_fields:
            assert field in optimal_params
        
        # Check parameter validity
        assert optimal_params['start_freq'] < optimal_params['end_freq']
        assert optimal_params['bandwidth'] > 0
        assert optimal_params['duration'] > 0
        assert optimal_params['theoretical_range_resolution'] > 0
    
    def test_chirp_analysis(self):
        """Test chirp signal analysis."""
        designer = ChirpDesigner()
        generator = ChirpGenerator()
        
        # Generate a test chirp
        t, chirp = generator.generate_lfm_chirp(35000, 45000, 0.005)
        
        analysis = designer.analyze_chirp_properties(chirp, 250000)
        
        # Check analysis results
        assert 'duration' in analysis
        assert 'bandwidth' in analysis
        assert 'range_resolution' in analysis
        assert 'peak_sidelobe_ratio' in analysis
        
        assert analysis['duration'] > 0
        assert analysis['bandwidth'] > 0
        assert analysis['range_resolution'] > 0
    
    def test_chirp_type_recommendation(self):
        """Test chirp type recommendation logic."""
        designer = ChirpDesigner()
        
        # Low time-bandwidth product
        chirp_type = designer._recommend_chirp_type(5.0)
        assert chirp_type == "linear"
        
        # Medium time-bandwidth product
        chirp_type = designer._recommend_chirp_type(50.0)
        assert chirp_type == "hyperbolic"
        
        # High time-bandwidth product
        chirp_type = designer._recommend_chirp_type(200.0)
        assert chirp_type == "coded"


class TestEchoProcessor:
    """Test echo processing functionality."""
    
    def test_initialization(self):
        """Test echo processor initialization."""
        processor = EchoProcessor(sample_rate=250000, speed_of_sound=343.0)
        assert processor.sample_rate == 250000
        assert processor.speed_of_sound == 343.0
    
    def test_matched_filtering(self):
        """Test matched filtering implementation."""
        processor = EchoProcessor()
        
        # Create test signal and template
        template = np.random.randn(100)
        received_signal = np.random.randn(4, 1000)  # 4 sensors
        
        # Add delayed template to one sensor
        delay = 200
        received_signal[0, delay:delay+100] += template * 0.5
        
        filtered_signal, time_delays = processor.matched_filter(
            received_signal, template, normalize=True
        )
        
        assert filtered_signal.shape == received_signal.shape
        assert len(time_delays) == 4
        assert np.all(np.isfinite(time_delays))
    
    def test_time_of_flight_extraction(self):
        """Test time-of-flight extraction methods."""
        processor = EchoProcessor()
        
        # Create test echo signal with known peak
        echo_signal = np.zeros((2, 1000))
        echo_signal[0, 300] = 1.0  # Peak at sample 300
        echo_signal[1, 450] = 0.8  # Peak at sample 450
        
        # Add noise
        echo_signal += np.random.randn(2, 1000) * 0.1
        
        methods = ["peak", "energy", "cfd"]
        
        for method in methods:
            tof, confidence = processor.extract_time_of_flight(
                echo_signal, threshold=0.3, method=method
            )
            
            assert len(tof) == 2
            assert len(confidence) == 2
            assert np.all(tof >= 0)
            assert np.all(confidence >= 0)
            assert np.all(confidence <= 1)
    
    def test_range_estimation(self):
        """Test range estimation from time-of-flight."""
        processor = EchoProcessor(speed_of_sound=343.0)
        
        # Test time-of-flight values
        tof = np.array([0.001, 0.002, 0.005])  # 1ms, 2ms, 5ms
        
        ranges = processor.estimate_range_from_tof(tof)
        
        expected_ranges = 343.0 * tof / 2  # Round trip
        assert np.allclose(ranges, expected_ranges)
    
    def test_constant_fraction_discriminator(self):
        """Test constant fraction discriminator."""
        processor = EchoProcessor()
        
        # Create signal with clear peak
        signal_data = np.zeros(1000)
        signal_data[500:520] = np.linspace(0, 1, 20)  # Rising edge
        signal_data[520:540] = np.linspace(1, 0, 20)  # Falling edge
        
        tof, confidence = processor._constant_fraction_discriminator(
            signal_data, fraction=0.3
        )
        
        assert tof > 0
        assert confidence > 0


class TestEchoEnhancer:
    """Test echo enhancement functionality."""
    
    def test_initialization(self):
        """Test echo enhancer initialization."""
        enhancer = EchoEnhancer(sample_rate=250000)
        assert enhancer.sample_rate == 250000
    
    def test_adaptive_denoising(self):
        """Test adaptive noise cancellation."""
        enhancer = EchoEnhancer()
        
        # Create noisy signal
        clean_signal = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, 10000))
        noise = np.random.randn(10000) * 0.5
        noisy_signal = clean_signal + noise
        
        # Reshape for multi-sensor
        signal_data = np.array([noisy_signal, noisy_signal * 0.8])
        
        denoised = enhancer.adaptive_denoise(signal_data)
        
        assert denoised.shape == signal_data.shape
        assert np.all(np.isfinite(denoised))
    
    def test_multipath_suppression(self):
        """Test multipath interference suppression."""
        enhancer = EchoEnhancer()
        
        # Create signal with multipath
        signal_data = np.zeros((2, 2000))
        signal_data[0, 500:600] = 1.0  # Direct path
        signal_data[0, 800:900] = 0.3  # Multipath reflection
        signal_data[1, :] = signal_data[0, :] * 0.9
        
        suppressed = enhancer.suppress_multipath(
            signal_data,
            direct_path_delay=500 / 250000,  # Convert to seconds
            suppression_factor=0.8
        )
        
        assert suppressed.shape == signal_data.shape
        # Multipath should be reduced
        assert np.max(suppressed[:, 800:900]) < np.max(signal_data[:, 800:900])
    
    def test_spectral_subtraction(self):
        """Test spectral subtraction noise reduction."""
        enhancer = EchoEnhancer()
        
        # Create test signal
        t = np.linspace(0, 1, 10000)
        clean_signal = np.sin(2 * np.pi * 1000 * t)
        noise = np.random.randn(10000) * 0.3
        noisy_signal = clean_signal + noise
        
        # Reshape for multi-sensor
        signal_data = np.array([noisy_signal])
        
        enhanced = enhancer.spectral_subtraction(signal_data, alpha=2.0, beta=0.01)
        
        assert enhanced.shape == signal_data.shape
        assert np.all(np.isfinite(enhanced))
    
    def test_wiener_filtering(self):
        """Test Wiener filter implementation."""
        enhancer = EchoEnhancer()
        
        # Create test signal
        signal_data = np.random.randn(5000)
        
        filtered = enhancer._wiener_filter(signal_data)
        
        assert len(filtered) == len(signal_data)
        assert np.all(np.isfinite(filtered))


class TestPreProcessor:
    """Test signal preprocessing functionality."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        processor = PreProcessor(sample_rate=250000)
        assert processor.sample_rate == 250000
    
    def test_bandpass_filtering(self):
        """Test bandpass filtering."""
        processor = PreProcessor(sample_rate=250000)
        
        # Create test signal with multiple frequencies
        t = np.linspace(0, 1, 250000)
        signal_data = (np.sin(2 * np.pi * 1000 * t) +  # Low freq
                      np.sin(2 * np.pi * 40000 * t) +  # In band
                      np.sin(2 * np.pi * 80000 * t))   # High freq
        
        filtered = processor.bandpass_filter(
            signal_data,
            low_freq=35000,
            high_freq=45000,
            order=4
        )
        
        assert len(filtered) == len(signal_data)
        
        # Check frequency content
        freqs = np.fft.fftfreq(len(filtered), 1/250000)
        fft_filtered = np.fft.fft(filtered)
        
        # Should have attenuated out-of-band frequencies
        low_freq_power = np.abs(fft_filtered[np.abs(freqs - 1000) < 100]).mean()
        in_band_power = np.abs(fft_filtered[np.abs(freqs - 40000) < 100]).mean()
        
        assert in_band_power > low_freq_power * 10  # At least 10x attenuation
    
    def test_dc_offset_removal(self):
        """Test DC offset removal."""
        processor = PreProcessor()
        
        # Create signal with DC offset
        signal_with_dc = np.random.randn(1000) + 5.0  # DC offset of 5
        
        signal_no_dc = processor.remove_dc_offset(signal_with_dc)
        
        assert len(signal_no_dc) == len(signal_with_dc)
        assert np.abs(np.mean(signal_no_dc)) < 1e-10  # Should be close to zero
    
    def test_amplitude_normalization(self):
        """Test amplitude normalization methods."""
        processor = PreProcessor()
        
        signal_data = np.random.randn(2, 1000) * 10  # Large amplitude
        
        methods = ["max", "rms", "zscore"]
        
        for method in methods:
            normalized = processor.normalize_amplitude(signal_data, method=method)
            
            assert normalized.shape == signal_data.shape
            assert np.all(np.isfinite(normalized))
            
            if method == "max":
                assert np.abs(np.max(np.abs(normalized))) <= 1.0
            elif method == "zscore":
                # Each row should have approximately zero mean and unit variance
                for i in range(normalized.shape[0]):
                    assert np.abs(np.mean(normalized[i])) < 0.1
                    assert np.abs(np.std(normalized[i]) - 1.0) < 0.1
    
    def test_windowing(self):
        """Test windowing functions."""
        processor = PreProcessor()
        
        signal_data = np.ones(1000)  # Constant signal
        
        windows = ["hann", "hamming", "blackman", "kaiser"]
        
        for window in windows:
            windowed = processor.apply_windowing(signal_data, window_type=window)
            
            assert len(windowed) == len(signal_data)
            # Windowed signal should taper at edges
            assert windowed[0] < windowed[len(windowed)//2]
            assert windowed[-1] < windowed[len(windowed)//2]
    
    def test_resampling(self):
        """Test signal resampling."""
        processor = PreProcessor(sample_rate=250000)
        
        # Create test signal
        signal_data = np.random.randn(2500)  # 0.01 seconds at 250kHz
        
        # Resample to different rates
        target_rates = [125000, 500000]
        
        for target_rate in target_rates:
            resampled = processor.resample_signal(signal_data, target_rate)
            
            expected_length = int(len(signal_data) * target_rate / 250000)
            assert len(resampled) == expected_length
            assert np.all(np.isfinite(resampled))
    
    def test_length_adjustment(self):
        """Test signal length trimming and padding."""
        processor = PreProcessor()
        
        signal_data = np.random.randn(3, 1000)
        
        # Test trimming
        trimmed = processor.trim_to_length(signal_data, 500, method="center")
        assert trimmed.shape == (3, 500)
        
        # Test padding
        padded = processor.trim_to_length(signal_data, 1500, method="center")
        assert padded.shape == (3, 1500)
        
        # Original data should be preserved in center
        start_idx = (1500 - 1000) // 2
        end_idx = start_idx + 1000
        assert np.allclose(padded[:, start_idx:end_idx], signal_data)
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        processor = PreProcessor(sample_rate=250000)
        
        # Create test signal
        signal_data = np.random.randn(4, 3000) + 2.0  # With DC offset
        
        config = {
            'remove_dc': True,
            'bandpass': {
                'low_freq': 35000,
                'high_freq': 45000,
                'order': 4
            },
            'target_length': 2048,
            'normalize': {'method': 'max'},
            'window': {'type': 'hann'}
        }
        
        processed = processor.preprocess_pipeline(signal_data, config)
        
        assert processed.shape == (4, 2048)
        assert np.all(np.isfinite(processed))
        assert np.max(np.abs(processed)) <= 1.0  # Normalized
        
        # Should have removed DC offset
        for i in range(4):
            assert np.abs(np.mean(processed[i])) < 0.1


class TestSignalProcessingEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_signal_handling(self):
        """Test handling of empty signals."""
        processor = EchoProcessor()
        
        # Empty signal should raise appropriate error or return empty result
        empty_signal = np.array([]).reshape(1, 0)
        template = np.random.randn(100)
        
        with pytest.raises((ValueError, IndexError)):
            processor.matched_filter(empty_signal, template)
    
    def test_mismatched_dimensions(self):
        """Test handling of dimension mismatches."""
        processor = EchoProcessor()
        
        signal_2d = np.random.randn(4, 1000)
        template_1d = np.random.randn(100)
        
        # This should work (correct usage)
        result = processor.matched_filter(signal_2d, template_1d)
        assert result[0].shape == signal_2d.shape
        
        # Test with 1D signal
        signal_1d = np.random.randn(1000)
        result = processor.matched_filter(signal_1d, template_1d)
        assert result[0].shape == (1, 1000)  # Should be reshaped
    
    def test_extreme_parameter_values(self):
        """Test handling of extreme parameter values."""
        generator = ChirpGenerator()
        
        # Very short duration
        t, chirp = generator.generate_lfm_chirp(35000, 45000, 0.0001)
        assert len(chirp) > 0
        
        # Very long duration
        t, chirp = generator.generate_lfm_chirp(35000, 45000, 1.0)
        assert len(chirp) > 0
        
        # Very close frequencies
        t, chirp = generator.generate_lfm_chirp(40000, 40001, 0.005)
        assert len(chirp) > 0
    
    def test_invalid_filter_parameters(self):
        """Test handling of invalid filter parameters."""
        processor = PreProcessor()
        
        signal_data = np.random.randn(1000)
        
        # Invalid frequency ranges
        with pytest.raises((ValueError, Warning)):
            processor.bandpass_filter(signal_data, 50000, 30000)  # high < low
        
        # Frequencies above Nyquist
        with pytest.raises((ValueError, Warning)):
            processor.bandpass_filter(signal_data, 200000, 300000)  # Above Nyquist


@pytest.mark.performance
class TestSignalProcessingPerformance:
    """Performance tests for signal processing."""
    
    def test_large_signal_processing(self):
        """Test processing of large signals."""
        processor = PreProcessor()
        
        # Large signal (10 seconds at 250kHz)
        large_signal = np.random.randn(4, 2500000)
        
        import time
        start_time = time.time()
        
        # Should complete in reasonable time
        filtered = processor.bandpass_filter(large_signal, 35000, 45000)
        
        processing_time = time.time() - start_time
        assert processing_time < 10.0  # Should complete in less than 10 seconds
        assert filtered.shape == large_signal.shape
    
    def test_chirp_generation_speed(self):
        """Test chirp generation performance."""
        generator = ChirpGenerator()
        
        import time
        
        # Generate many chirps
        start_time = time.time()
        
        for _ in range(100):
            t, chirp = generator.generate_lfm_chirp(35000, 45000, 0.005)
        
        generation_time = time.time() - start_time
        assert generation_time < 5.0  # Should be fast
        
        print(f"Generated 100 chirps in {generation_time:.2f} seconds")


if __name__ == "__main__":
    pytest.main([__file__])