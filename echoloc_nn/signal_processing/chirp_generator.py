"""
Chirp generation and design utilities.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import scipy.signal as signal
from enum import Enum


class ChirpType(Enum):
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic" 
    HYPERBOLIC = "hyperbolic"
    QUADRATIC = "quadratic"


class ChirpGenerator:
    """
    Generator for various types of ultrasonic chirp signals.
    
    Supports linear frequency modulation (LFM), logarithmic, hyperbolic,
    and quadratic chirps optimized for ultrasonic ranging applications.
    """
    
    def __init__(self, sample_rate: int = 250000):
        self.sample_rate = sample_rate
        
    def generate_lfm_chirp(
        self,
        start_freq: float,
        end_freq: float,
        duration: float,
        window: Optional[str] = "hann"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Linear Frequency Modulated (LFM) chirp.
        
        Args:
            start_freq: Starting frequency in Hz
            end_freq: Ending frequency in Hz  
            duration: Duration in seconds
            window: Window function ("hann", "hamming", "blackman", None)
            
        Returns:
            Tuple of (time_array, chirp_signal)
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Generate LFM chirp
        chirp_signal = signal.chirp(
            t, start_freq, duration, end_freq, method='linear'
        )
        
        # Apply window function
        if window:
            window_func = signal.get_window(window, len(chirp_signal))
            chirp_signal = chirp_signal * window_func
            
        return t, chirp_signal
    
    def generate_hyperbolic_chirp(
        self,
        center_freq: float,
        bandwidth: float,
        duration: float,
        window: Optional[str] = "hann"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate hyperbolic chirp with better Doppler tolerance.
        
        Args:
            center_freq: Center frequency in Hz
            bandwidth: Frequency bandwidth in Hz
            duration: Duration in seconds
            window: Window function
            
        Returns:
            Tuple of (time_array, chirp_signal)
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Hyperbolic frequency modulation
        # f(t) = f0 / (1 + beta * t)
        beta = bandwidth / (center_freq * duration)
        instantaneous_freq = center_freq / (1 + beta * t)
        
        # Integrate to get phase
        phase = np.cumsum(instantaneous_freq) * 2 * np.pi / self.sample_rate
        chirp_signal = np.cos(phase)
        
        # Apply window function
        if window:
            window_func = signal.get_window(window, len(chirp_signal))
            chirp_signal = chirp_signal * window_func
            
        return t, chirp_signal
    
    def generate_logarithmic_chirp(
        self,
        start_freq: float,
        end_freq: float,
        duration: float,
        window: Optional[str] = "hann"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate logarithmic chirp.
        
        Args:
            start_freq: Starting frequency in Hz
            end_freq: Ending frequency in Hz
            duration: Duration in seconds
            window: Window function
            
        Returns:
            Tuple of (time_array, chirp_signal)
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Generate logarithmic chirp
        chirp_signal = signal.chirp(
            t, start_freq, duration, end_freq, method='logarithmic'
        )
        
        # Apply window function
        if window:
            window_func = signal.get_window(window, len(chirp_signal))
            chirp_signal = chirp_signal * window_func
            
        return t, chirp_signal
    
    def generate_coded_chirp(
        self,
        code: str,
        carrier_freq: float,
        chip_duration: float,
        window: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate coded chirp using binary sequences.
        
        Args:
            code: Code type ("barker13", "barker11", "gold", "mls")  
            carrier_freq: Carrier frequency in Hz
            chip_duration: Duration of each chip in seconds
            window: Window function
            
        Returns:
            Tuple of (time_array, chirp_signal)
        """
        # Get binary code sequence
        code_sequence = self._get_code_sequence(code)
        
        # Generate coded signal
        chip_samples = int(self.sample_rate * chip_duration)
        total_samples = len(code_sequence) * chip_samples
        
        t = np.linspace(0, len(code_sequence) * chip_duration, total_samples, False)
        carrier = np.cos(2 * np.pi * carrier_freq * t)
        
        # Apply code modulation
        coded_signal = np.zeros(total_samples)
        for i, bit in enumerate(code_sequence):
            start_idx = i * chip_samples
            end_idx = (i + 1) * chip_samples
            coded_signal[start_idx:end_idx] = bit * carrier[start_idx:end_idx]
        
        # Apply window function
        if window:
            window_func = signal.get_window(window, len(coded_signal))
            coded_signal = coded_signal * window_func
            
        return t, coded_signal
    
    def _get_code_sequence(self, code: str) -> np.ndarray:
        """Get binary code sequence."""
        codes = {
            "barker13": np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]),
            "barker11": np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]),
            "barker7": np.array([1, 1, 1, -1, -1, 1, -1]),
            "barker5": np.array([1, 1, 1, -1, 1])
        }
        
        if code in codes:
            return codes[code]
        else:
            raise ValueError(f"Unknown code type: {code}")


class ChirpDesigner:
    """
    Design optimal chirp parameters for ultrasonic ranging.
    
    Provides methods to optimize chirp parameters based on
    application requirements and hardware constraints.
    """
    
    def __init__(self, speed_of_sound: float = 343.0):
        self.speed_of_sound = speed_of_sound
        
    def design_optimal_chirp(
        self,
        max_range: float,
        range_resolution: float,
        max_velocity: float = 1.0,
        sample_rate: int = 250000,
        transducer_bandwidth: Tuple[float, float] = (35000, 45000)
    ) -> Dict[str, Any]:
        """
        Design optimal chirp parameters for given requirements.
        
        Args:
            max_range: Maximum detection range in meters
            range_resolution: Required range resolution in meters
            max_velocity: Maximum target velocity in m/s
            sample_rate: ADC sample rate in Hz
            transducer_bandwidth: Transducer frequency range (f_min, f_max)
            
        Returns:
            Dictionary with optimal chirp parameters
        """
        f_min, f_max = transducer_bandwidth
        bandwidth = f_max - f_min
        center_freq = (f_min + f_max) / 2
        
        # Calculate required bandwidth for range resolution
        # Range resolution = c / (2 * B)
        required_bandwidth = self.speed_of_sound / (2 * range_resolution)
        
        if required_bandwidth > bandwidth:
            print(f"Warning: Required bandwidth ({required_bandwidth:.0f} Hz) "
                  f"exceeds transducer bandwidth ({bandwidth:.0f} Hz)")
            bandwidth = min(required_bandwidth, bandwidth)
            
        # Calculate minimum chirp duration for Doppler tolerance
        # Doppler shift = 2 * v * f / c
        max_doppler_shift = 2 * max_velocity * center_freq / self.speed_of_sound
        
        # Duration should be >> 1/Doppler_shift for good tolerance
        min_duration = 10 / max_doppler_shift  # Conservative factor of 10
        
        # Calculate duration based on range ambiguity
        # Maximum unambiguous range = c * T / 2
        max_duration = 2 * max_range / self.speed_of_sound
        
        # Choose conservative duration
        duration = min(max_duration * 0.8, max(min_duration, 0.001))  # At least 1ms
        
        # Time-bandwidth product
        time_bandwidth_product = bandwidth * duration
        
        return {
            "start_freq": center_freq - bandwidth / 2,
            "end_freq": center_freq + bandwidth / 2,
            "center_freq": center_freq,
            "bandwidth": bandwidth,
            "duration": duration,
            "time_bandwidth_product": time_bandwidth_product,
            "theoretical_range_resolution": self.speed_of_sound / (2 * bandwidth),
            "max_unambiguous_range": self.speed_of_sound * duration / 2,
            "doppler_tolerance": max_doppler_shift,
            "recommended_chirp_type": self._recommend_chirp_type(time_bandwidth_product)
        }
    
    def _recommend_chirp_type(self, time_bandwidth_product: float) -> str:
        """Recommend chirp type based on time-bandwidth product."""
        if time_bandwidth_product < 10:
            return "linear"  # Simple LFM for low TB products
        elif time_bandwidth_product < 100:
            return "hyperbolic"  # Better Doppler tolerance
        else:
            return "coded"  # Better sidelobe suppression
    
    def analyze_chirp_properties(
        self, 
        chirp_signal: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        Analyze properties of a generated chirp signal.
        
        Args:
            chirp_signal: Generated chirp waveform
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with chirp analysis results
        """
        duration = len(chirp_signal) / sample_rate
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(
            chirp_signal, sample_rate, nperseg=256, noverlap=128
        )
        
        # Find frequency extent
        energy_threshold = 0.1 * np.max(Sxx)
        freq_mask = np.any(Sxx > energy_threshold, axis=1)
        f_min = f[freq_mask][0] if np.any(freq_mask) else 0
        f_max = f[freq_mask][-1] if np.any(freq_mask) else sample_rate/2
        bandwidth = f_max - f_min
        
        # Range resolution
        range_resolution = self.speed_of_sound / (2 * bandwidth)
        
        # Compute autocorrelation for sidelobe analysis
        autocorr = np.correlate(chirp_signal, chirp_signal, mode='full')
        autocorr = autocorr / np.max(autocorr)
        
        # Find peak and sidelobes
        center_idx = len(autocorr) // 2
        peak_width = self._find_peak_width(autocorr, center_idx)
        max_sidelobe = self._find_max_sidelobe(autocorr, center_idx, peak_width)
        
        return {
            "duration": duration,
            "bandwidth": bandwidth,
            "center_frequency": (f_min + f_max) / 2,
            "time_bandwidth_product": bandwidth * duration,
            "range_resolution": range_resolution,
            "peak_sidelobe_ratio": -20 * np.log10(max_sidelobe) if max_sidelobe > 0 else np.inf,
            "energy": np.sum(chirp_signal**2)
        }
    
    def _find_peak_width(self, autocorr: np.ndarray, center_idx: int) -> int:
        """Find the width of the main peak in autocorrelation."""
        peak_val = autocorr[center_idx]
        threshold = peak_val * 0.5  # -3dB width
        
        # Find left and right edges
        left_idx = center_idx
        while left_idx > 0 and autocorr[left_idx] > threshold:
            left_idx -= 1
            
        right_idx = center_idx
        while right_idx < len(autocorr) - 1 and autocorr[right_idx] > threshold:
            right_idx += 1
            
        return right_idx - left_idx
    
    def _find_max_sidelobe(
        self, 
        autocorr: np.ndarray, 
        center_idx: int, 
        peak_width: int
    ) -> float:
        """Find maximum sidelobe level."""
        half_width = peak_width // 2
        
        # Exclude main peak region
        left_sidelobes = autocorr[:center_idx - half_width]
        right_sidelobes = autocorr[center_idx + half_width:]
        
        max_left = np.max(np.abs(left_sidelobes)) if len(left_sidelobes) > 0 else 0
        max_right = np.max(np.abs(right_sidelobes)) if len(right_sidelobes) > 0 else 0
        
        return max(max_left, max_right)