"""
Echo processing and enhancement utilities.
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d


class EchoProcessor:
    """
    Basic echo processing for ultrasonic localization.
    
    Provides fundamental signal processing operations including
    matched filtering, time-of-flight extraction, and basic
    noise reduction techniques.
    """
    
    def __init__(self, sample_rate: int = 250000, speed_of_sound: float = 343.0):
        self.sample_rate = sample_rate
        self.speed_of_sound = speed_of_sound
        
    def matched_filter(
        self,
        received_signal: np.ndarray,
        template: np.ndarray,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply matched filtering to enhance echo detection.
        
        Args:
            received_signal: Received echo signal (n_sensors, n_samples)
            template: Template chirp signal (n_samples,)
            normalize: Whether to normalize the output
            
        Returns:
            Tuple of (filtered_signal, time_delays)
        """
        if received_signal.ndim == 1:
            received_signal = received_signal.reshape(1, -1)
            
        n_sensors, n_samples = received_signal.shape
        filtered_signals = np.zeros_like(received_signal)
        time_delays = np.zeros(n_sensors)
        
        for i in range(n_sensors):
            # Cross-correlation (matched filter)
            correlation = signal.correlate(
                received_signal[i], template, mode='full'
            )
            
            if normalize:
                # Normalize by template energy
                template_energy = np.sum(template**2)
                correlation = correlation / np.sqrt(template_energy)
                
            # Find peak (time delay)
            peak_idx = np.argmax(np.abs(correlation))
            time_delays[i] = (peak_idx - len(template) + 1) / self.sample_rate
            
            # Extract filtered signal (same length as input)
            start_idx = len(template) - 1
            filtered_signals[i] = correlation[start_idx:start_idx + n_samples]
            
        return filtered_signals, time_delays
    
    def extract_time_of_flight(
        self,
        echo_signal: np.ndarray,
        threshold: float = 0.3,
        method: str = "peak"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract time-of-flight from echo signals.
        
        Args:
            echo_signal: Echo signal (n_sensors, n_samples)
            threshold: Detection threshold (0-1)
            method: Detection method ("peak", "energy", "cfd")
            
        Returns:
            Tuple of (time_of_flight, confidence)
        """
        if echo_signal.ndim == 1:
            echo_signal = echo_signal.reshape(1, -1)
            
        n_sensors, n_samples = echo_signal.shape
        time_of_flight = np.zeros(n_sensors)
        confidence = np.zeros(n_sensors)
        
        for i in range(n_sensors):
            signal_abs = np.abs(echo_signal[i])
            
            if method == "peak":
                # Simple peak detection
                peak_idx = np.argmax(signal_abs)
                peak_val = signal_abs[peak_idx]
                
                if peak_val > threshold * np.max(signal_abs):
                    time_of_flight[i] = peak_idx / self.sample_rate
                    confidence[i] = peak_val / np.max(signal_abs)
                    
            elif method == "energy":
                # Energy-based detection
                energy_envelope = self._compute_energy_envelope(signal_abs)
                threshold_val = threshold * np.max(energy_envelope)
                
                first_crossing = np.where(energy_envelope > threshold_val)[0]
                if len(first_crossing) > 0:
                    time_of_flight[i] = first_crossing[0] / self.sample_rate
                    confidence[i] = energy_envelope[first_crossing[0]] / np.max(energy_envelope)
                    
            elif method == "cfd":
                # Constant Fraction Discriminator
                tof, conf = self._constant_fraction_discriminator(
                    signal_abs, threshold
                )
                time_of_flight[i] = tof
                confidence[i] = conf
                
        return time_of_flight, confidence
    
    def _compute_energy_envelope(self, signal_data: np.ndarray, window_size: int = 64) -> np.ndarray:
        """Compute energy envelope of signal."""
        signal_squared = signal_data**2
        envelope = np.convolve(
            signal_squared, 
            np.ones(window_size) / window_size, 
            mode='same'
        )
        return envelope
    
    def _constant_fraction_discriminator(
        self, 
        signal_data: np.ndarray, 
        fraction: float = 0.3
    ) -> Tuple[float, float]:
        """Constant fraction discriminator for precise timing."""
        # Find global maximum
        max_idx = np.argmax(signal_data)
        max_val = signal_data[max_idx]
        
        # Find fraction point on leading edge
        threshold_val = fraction * max_val
        
        # Search backwards from peak for threshold crossing
        for i in range(max_idx, 0, -1):
            if signal_data[i] <= threshold_val and signal_data[i-1] <= threshold_val:
                # Linear interpolation for sub-sample precision
                if signal_data[i+1] != signal_data[i]:
                    frac = (threshold_val - signal_data[i]) / (signal_data[i+1] - signal_data[i])
                    precise_time = (i + frac) / self.sample_rate
                else:
                    precise_time = i / self.sample_rate
                    
                confidence = max_val / np.max(signal_data)
                return precise_time, confidence
                
        return 0.0, 0.0
    
    def estimate_range_from_tof(self, time_of_flight: np.ndarray) -> np.ndarray:
        """Convert time-of-flight to range estimates."""
        # Range = speed_of_sound * time_of_flight / 2 (round trip)
        return self.speed_of_sound * time_of_flight / 2


class EchoEnhancer:
    """
    Advanced echo enhancement and noise reduction.
    
    Provides sophisticated signal enhancement techniques including
    adaptive filtering, multipath suppression, and spectral cleaning.
    """
    
    def __init__(self, sample_rate: int = 250000):
        self.sample_rate = sample_rate
        
    def adaptive_denoise(
        self,
        signal_data: np.ndarray,
        noise_profile: Optional[np.ndarray] = None,
        adaptation_rate: float = 0.01,
        filter_length: int = 64
    ) -> np.ndarray:
        """
        Adaptive noise cancellation using LMS algorithm.
        
        Args:
            signal_data: Input signal (n_sensors, n_samples)
            noise_profile: Noise reference signal
            adaptation_rate: LMS adaptation rate
            filter_length: Adaptive filter length
            
        Returns:
            Denoised signal
        """
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(1, -1)
            
        n_sensors, n_samples = signal_data.shape
        denoised = np.zeros_like(signal_data)
        
        for i in range(n_sensors):
            if noise_profile is not None:
                # Use provided noise reference
                denoised[i] = self._lms_filter(
                    signal_data[i], noise_profile, adaptation_rate, filter_length
                )
            else:
                # Estimate noise from signal statistics
                denoised[i] = self._wiener_filter(signal_data[i])
                
        return denoised
    
    def _lms_filter(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        mu: float,
        filter_length: int
    ) -> np.ndarray:
        """Least Mean Squares adaptive filter."""
        n_samples = len(primary)
        weights = np.zeros(filter_length)
        output = np.zeros(n_samples)
        
        for n in range(filter_length, n_samples):
            # Extract reference vector
            x = reference[n-filter_length:n][::-1]  # Reverse for convolution
            
            # Filter output
            y = np.dot(weights, x)
            
            # Error signal  
            error = primary[n] - y
            output[n] = error
            
            # Update weights
            weights += mu * error * x
            
        return output
    
    def _wiener_filter(self, signal_data: np.ndarray, noise_var: Optional[float] = None) -> np.ndarray:
        """Wiener filter for noise reduction."""
        # Estimate noise variance from signal if not provided
        if noise_var is None:
            # Use median absolute deviation as robust noise estimator
            noise_var = (np.median(np.abs(signal_data)) / 0.6745)**2
            
        # Compute power spectral density
        f, psd = signal.welch(signal_data, self.sample_rate, nperseg=1024)
        
        # Wiener filter transfer function
        signal_var = np.var(signal_data)
        wiener_filter = psd / (psd + noise_var)
        
        # Apply filter in frequency domain
        fft_signal = np.fft.fft(signal_data)
        
        # Interpolate filter to match FFT length
        filter_interp = np.interp(
            np.linspace(0, len(wiener_filter)-1, len(fft_signal)//2 + 1),
            np.arange(len(wiener_filter)),
            wiener_filter
        )
        
        # Create full filter (symmetric for real signals)
        full_filter = np.concatenate([filter_interp, filter_interp[-2:0:-1]])
        
        # Apply filter
        filtered_fft = fft_signal * full_filter
        filtered_signal = np.real(np.fft.ifft(filtered_fft))
        
        return filtered_signal
    
    def suppress_multipath(
        self,
        signal_data: np.ndarray,
        direct_path_delay: float,
        suppression_factor: float = 0.8,
        window_size: float = 0.001  # 1ms window
    ) -> np.ndarray:
        """
        Suppress multipath reflections while preserving direct path.
        
        Args:
            signal_data: Input signal (n_sensors, n_samples)
            direct_path_delay: Estimated direct path delay in seconds
            suppression_factor: Suppression strength (0-1)
            window_size: Suppression window size in seconds
            
        Returns:
            Signal with suppressed multipath
        """
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(1, -1)
            
        n_sensors, n_samples = signal_data.shape
        processed = signal_data.copy()
        
        window_samples = int(window_size * self.sample_rate)
        direct_sample = int(direct_path_delay * self.sample_rate)
        
        for i in range(n_sensors):
            # Create suppression mask
            mask = np.ones(n_samples)
            
            # Preserve region around direct path
            preserve_start = max(0, direct_sample - window_samples // 2)
            preserve_end = min(n_samples, direct_sample + window_samples // 2)
            
            # Apply suppression to later arrivals (multipath)
            if preserve_end < n_samples:
                mask[preserve_end:] *= (1 - suppression_factor)
                
            processed[i] *= mask
            
        return processed
    
    def spectral_subtraction(
        self,
        signal_data: np.ndarray,
        noise_spectrum: Optional[np.ndarray] = None,
        alpha: float = 2.0,
        beta: float = 0.01
    ) -> np.ndarray:
        """
        Spectral subtraction for noise reduction.
        
        Args:
            signal_data: Input signal (n_sensors, n_samples)
            noise_spectrum: Estimated noise spectrum
            alpha: Over-subtraction factor
            beta: Spectral floor factor
            
        Returns:
            Enhanced signal
        """
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(1, -1)
            
        n_sensors, n_samples = signal_data.shape
        enhanced = np.zeros_like(signal_data)
        
        for i in range(n_sensors):
            # Compute signal spectrum
            fft_signal = np.fft.fft(signal_data[i])
            magnitude = np.abs(fft_signal)
            phase = np.angle(fft_signal)
            
            # Estimate noise spectrum if not provided
            if noise_spectrum is None:
                # Use first 10% of signal as noise estimate
                noise_samples = signal_data[i][:n_samples//10]
                noise_fft = np.fft.fft(noise_samples, n=n_samples)
                noise_mag = np.abs(noise_fft)
            else:
                noise_mag = noise_spectrum
                
            # Spectral subtraction
            subtracted_mag = magnitude - alpha * noise_mag
            
            # Apply spectral floor
            floor_mask = subtracted_mag < beta * magnitude
            subtracted_mag[floor_mask] = beta * magnitude[floor_mask]
            
            # Reconstruct signal
            enhanced_fft = subtracted_mag * np.exp(1j * phase)
            enhanced[i] = np.real(np.fft.ifft(enhanced_fft))
            
        return enhanced