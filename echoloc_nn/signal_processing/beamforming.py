"""
Beamforming algorithms for ultrasonic array processing.

Implements various beamforming techniques for spatial signal processing
and interference suppression in ultrasonic localization systems.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import scipy.signal as signal
from scipy.spatial.distance import cdist


class BeamFormer:
    """
    Base class for beamforming algorithms.
    
    Provides common functionality for spatial filtering and
    direction-of-arrival processing with ultrasonic arrays.
    """
    
    def __init__(
        self,
        array_geometry: np.ndarray,
        sample_rate: float = 250000.0,
        sound_speed: float = 343.0
    ):
        """
        Initialize beamformer.
        
        Args:
            array_geometry: Sensor positions (n_sensors, 2) or (n_sensors, 3)
            sample_rate: Sample rate in Hz
            sound_speed: Speed of sound in m/s
        """
        self.array_geometry = np.array(array_geometry)
        self.sample_rate = sample_rate
        self.sound_speed = sound_speed
        self.n_sensors = self.array_geometry.shape[0]
        
        # Validate array geometry
        if self.array_geometry.shape[1] not in [2, 3]:
            raise ValueError("Array geometry must be (n_sensors, 2) or (n_sensors, 3)")
    
    def calculate_delays(
        self,
        target_direction: np.ndarray,
        reference_sensor: int = 0
    ) -> np.ndarray:
        """
        Calculate time delays for steering to target direction.
        
        Args:
            target_direction: Target direction vector (2D or 3D)
            reference_sensor: Reference sensor index for delay calculation
            
        Returns:
            Time delays in seconds (n_sensors,)
        """
        target_direction = np.array(target_direction)
        target_direction = target_direction / np.linalg.norm(target_direction)
        
        # Ensure compatible dimensions
        if len(target_direction) != self.array_geometry.shape[1]:
            if len(target_direction) == 2 and self.array_geometry.shape[1] == 3:
                target_direction = np.append(target_direction, 0)
            elif len(target_direction) == 3 and self.array_geometry.shape[1] == 2:
                target_direction = target_direction[:2]
        
        # Calculate path differences relative to reference sensor
        ref_pos = self.array_geometry[reference_sensor]
        path_differences = np.dot(self.array_geometry - ref_pos, target_direction)
        
        # Convert to time delays
        time_delays = path_differences / self.sound_speed
        
        return time_delays
    
    def calculate_steering_vector(
        self,
        target_direction: np.ndarray,
        frequency: float,
        reference_sensor: int = 0
    ) -> np.ndarray:
        """
        Calculate complex steering vector for target direction.
        
        Args:
            target_direction: Target direction vector
            frequency: Signal frequency in Hz
            reference_sensor: Reference sensor index
            
        Returns:
            Complex steering vector (n_sensors,)
        """
        time_delays = self.calculate_delays(target_direction, reference_sensor)
        
        # Phase delays at specified frequency
        phase_delays = 2 * np.pi * frequency * time_delays
        steering_vector = np.exp(-1j * phase_delays)
        
        return steering_vector
    
    def apply_beamforming(
        self,
        signal_data: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Apply beamforming weights to multi-sensor signal.
        
        Args:
            signal_data: Multi-sensor signal (n_sensors, n_samples)
            weights: Beamforming weights (n_sensors,) - can be complex
            
        Returns:
            Beamformed signal (n_samples,)
        """
        if signal_data.shape[0] != len(weights):
            raise ValueError("Number of sensors must match number of weights")
        
        # Apply weights and sum
        if np.iscomplexobj(weights):
            # For complex weights, assume signal_data is complex (e.g., analytic signal)
            beamformed = np.sum(weights[:, np.newaxis] * signal_data, axis=0)
        else:
            # For real weights
            beamformed = np.sum(weights[:, np.newaxis] * signal_data, axis=0)
        
        return beamformed
    
    def scan_directions(
        self,
        signal_data: np.ndarray,
        scan_angles: np.ndarray,
        method: str = "delay_sum"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan multiple directions for source localization.
        
        Args:
            signal_data: Multi-sensor signal (n_sensors, n_samples)
            scan_angles: Angles to scan in radians (n_angles,)
            method: Beamforming method ("delay_sum", "mvdr", "music")
            
        Returns:
            Tuple of (scan_angles, power_spectrum)
        """
        power_spectrum = np.zeros(len(scan_angles))
        
        for i, angle in enumerate(scan_angles):
            # Convert angle to direction vector (2D)
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            if method == "delay_sum":
                # Simple delay-and-sum beamforming
                delays = self.calculate_delays(direction)
                beamformed = self._delay_and_sum_steer(signal_data, delays)
                power_spectrum[i] = np.mean(beamformed**2)
                
            else:
                raise ValueError(f"Unsupported scanning method: {method}")
        
        return scan_angles, power_spectrum
    
    def _delay_and_sum_steer(
        self,
        signal_data: np.ndarray,
        delays: np.ndarray
    ) -> np.ndarray:
        """
        Apply delay-and-sum steering to signal data.
        
        Args:
            signal_data: Multi-sensor signal (n_sensors, n_samples)
            delays: Time delays in seconds (n_sensors,)
            
        Returns:
            Steered signal (n_samples,)
        """
        # Convert time delays to sample delays
        sample_delays = delays * self.sample_rate
        
        # Apply fractional delays using interpolation
        steered_signals = np.zeros_like(signal_data)
        
        for i, delay in enumerate(sample_delays):
            if abs(delay) < 0.1:  # Skip very small delays
                steered_signals[i] = signal_data[i]
            else:
                # Use scipy's resample for fractional delays
                delay_samples = int(np.round(delay))
                if delay_samples > 0:
                    # Positive delay - pad at beginning
                    steered_signals[i] = np.pad(
                        signal_data[i][:-delay_samples] if delay_samples < len(signal_data[i]) else np.zeros(len(signal_data[i])),
                        (delay_samples, 0),
                        mode='constant'
                    )
                elif delay_samples < 0:
                    # Negative delay - pad at end
                    delay_samples = abs(delay_samples)
                    steered_signals[i] = np.pad(
                        signal_data[i][delay_samples:],
                        (0, delay_samples),
                        mode='constant'
                    )
                else:
                    steered_signals[i] = signal_data[i]
        
        # Sum all steered signals
        return np.sum(steered_signals, axis=0) / self.n_sensors
    
    def estimate_snr(
        self,
        beamformed_signal: np.ndarray,
        noise_samples: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate signal-to-noise ratio of beamformed output.
        
        Args:
            beamformed_signal: Beamformed signal
            noise_samples: Optional noise-only samples for comparison
            
        Returns:
            SNR estimate in dB
        """
        signal_power = np.mean(beamformed_signal**2)
        
        if noise_samples is not None:
            noise_power = np.mean(noise_samples**2)
        else:
            # Estimate noise from signal statistics
            # Use median absolute deviation as robust noise estimate
            mad = np.median(np.abs(beamformed_signal - np.median(beamformed_signal)))
            noise_power = (mad / 0.6745)**2  # Convert MAD to variance
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')
        
        return snr_db


class DelayAndSum(BeamFormer):
    """
    Delay-and-Sum beamforming implementation.
    
    Classical beamforming technique that aligns signals from different
    sensors by applying appropriate time delays and summing.
    """
    
    def __init__(
        self,
        array_geometry: np.ndarray,
        sample_rate: float = 250000.0,
        sound_speed: float = 343.0,
        interpolation_method: str = "linear"
    ):
        """
        Initialize Delay-and-Sum beamformer.
        
        Args:
            array_geometry: Sensor positions
            sample_rate: Sample rate in Hz
            sound_speed: Speed of sound in m/s
            interpolation_method: Interpolation method for fractional delays
        """
        super().__init__(array_geometry, sample_rate, sound_speed)
        self.interpolation_method = interpolation_method
    
    def beamform(
        self,
        signal_data: np.ndarray,
        target_direction: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Apply delay-and-sum beamforming.
        
        Args:
            signal_data: Multi-sensor signal (n_sensors, n_samples)
            target_direction: Target direction for plane wave assumption
            target_position: Target position for point source assumption
            normalize: Whether to normalize output by number of sensors
            
        Returns:
            Beamformed signal (n_samples,)
        """
        if target_direction is not None:
            # Plane wave beamforming
            delays = self.calculate_delays(target_direction)
        elif target_position is not None:
            # Point source beamforming
            delays = self.calculate_point_source_delays(target_position)
        else:
            raise ValueError("Must specify either target_direction or target_position")
        
        # Apply delays and sum
        beamformed = self._apply_delays_and_sum(signal_data, delays, normalize)
        
        return beamformed
    
    def calculate_point_source_delays(
        self,
        target_position: np.ndarray,
        reference_sensor: int = 0
    ) -> np.ndarray:
        """
        Calculate delays for point source at given position.
        
        Args:
            target_position: Target position coordinates
            reference_sensor: Reference sensor index
            
        Returns:
            Time delays in seconds (n_sensors,)
        """
        target_position = np.array(target_position)
        
        # Ensure compatible dimensions
        if len(target_position) == 2 and self.array_geometry.shape[1] == 3:
            target_position = np.append(target_position, 0)
        elif len(target_position) == 3 and self.array_geometry.shape[1] == 2:
            target_position = target_position[:2]
        
        # Calculate distances from target to each sensor
        distances = np.linalg.norm(
            self.array_geometry - target_position[np.newaxis, :],
            axis=1
        )
        
        # Reference distance
        ref_distance = distances[reference_sensor]
        
        # Convert distance differences to time delays
        time_delays = (distances - ref_distance) / self.sound_speed
        
        return time_delays
    
    def _apply_delays_and_sum(
        self,
        signal_data: np.ndarray,
        delays: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Apply time delays and sum signals.
        
        Args:
            signal_data: Multi-sensor signal (n_sensors, n_samples)
            delays: Time delays in seconds (n_sensors,)
            normalize: Whether to normalize by number of sensors
            
        Returns:
            Beamformed signal (n_samples,)
        """
        n_sensors, n_samples = signal_data.shape
        delayed_signals = np.zeros_like(signal_data)
        
        # Convert time delays to sample indices
        sample_delays = delays * self.sample_rate
        
        for i in range(n_sensors):
            delay_samples = sample_delays[i]
            
            if abs(delay_samples) < 0.1:
                # No significant delay
                delayed_signals[i] = signal_data[i]
            else:
                delayed_signals[i] = self._apply_fractional_delay(
                    signal_data[i], delay_samples
                )
        
        # Sum delayed signals
        beamformed = np.sum(delayed_signals, axis=0)
        
        if normalize:
            beamformed /= n_sensors
        
        return beamformed
    
    def _apply_fractional_delay(
        self,
        signal: np.ndarray,
        delay_samples: float
    ) -> np.ndarray:
        """
        Apply fractional sample delay to signal.
        
        Args:
            signal: Input signal
            delay_samples: Delay in samples (can be fractional)
            
        Returns:
            Delayed signal
        """
        if self.interpolation_method == "linear":
            return self._linear_interpolation_delay(signal, delay_samples)
        elif self.interpolation_method == "sinc":
            return self._sinc_interpolation_delay(signal, delay_samples)
        else:
            # Simple integer delay fallback
            int_delay = int(np.round(delay_samples))
            if int_delay > 0:
                return np.pad(signal[:-int_delay] if int_delay < len(signal) else np.zeros_like(signal), (int_delay, 0), mode='constant')
            elif int_delay < 0:
                int_delay = abs(int_delay)
                return np.pad(signal[int_delay:], (0, int_delay), mode='constant')
            else:
                return signal
    
    def _linear_interpolation_delay(
        self,
        signal: np.ndarray,
        delay_samples: float
    ) -> np.ndarray:
        """Apply fractional delay using linear interpolation."""
        n_samples = len(signal)
        
        # Create output array
        delayed_signal = np.zeros_like(signal)
        
        # Integer and fractional parts of delay
        int_delay = int(np.floor(delay_samples))
        frac_delay = delay_samples - int_delay
        
        # Apply delay with linear interpolation
        for i in range(n_samples):
            src_idx = i - int_delay
            
            if 0 <= src_idx < n_samples - 1:
                # Linear interpolation between adjacent samples
                delayed_signal[i] = (1 - frac_delay) * signal[src_idx] + frac_delay * signal[src_idx + 1]
            elif src_idx == n_samples - 1:
                # Edge case - use last sample
                delayed_signal[i] = signal[src_idx]
            # else: leave as zero (padding)
        
        return delayed_signal
    
    def _sinc_interpolation_delay(
        self,
        signal: np.ndarray,
        delay_samples: float,
        window_size: int = 16
    ) -> np.ndarray:
        """Apply fractional delay using windowed sinc interpolation."""
        n_samples = len(signal)
        delayed_signal = np.zeros_like(signal)
        
        # Sinc interpolation with windowing
        for i in range(n_samples):
            value = 0.0
            
            for k in range(-window_size//2, window_size//2):
                src_idx = i - int(delay_samples) + k
                
                if 0 <= src_idx < n_samples:
                    # Windowed sinc kernel
                    t = k - (delay_samples - int(delay_samples))
                    if abs(t) < 1e-10:
                        sinc_val = 1.0
                    else:
                        sinc_val = np.sin(np.pi * t) / (np.pi * t)
                    
                    # Hamming window
                    window_val = 0.54 - 0.46 * np.cos(2 * np.pi * (k + window_size//2) / window_size)
                    
                    value += signal[src_idx] * sinc_val * window_val
            
            delayed_signal[i] = value
        
        return delayed_signal
    
    def adaptive_beamform(
        self,
        signal_data: np.ndarray,
        search_directions: np.ndarray,
        method: str = "power_scan"
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Adaptive beamforming - automatically find best direction.
        
        Args:
            signal_data: Multi-sensor signal (n_sensors, n_samples)
            search_directions: Directions to search (n_directions, 2 or 3)
            method: Search method ("power_scan", "cross_correlation")
            
        Returns:
            Tuple of (beamformed_signal, direction_powers, best_direction_idx)
        """
        n_directions = len(search_directions)
        direction_powers = np.zeros(n_directions)
        beamformed_signals = []
        
        for i, direction in enumerate(search_directions):
            # Beamform in this direction
            beamformed = self.beamform(signal_data, target_direction=direction)
            beamformed_signals.append(beamformed)
            
            if method == "power_scan":
                # Measure output power
                direction_powers[i] = np.mean(beamformed**2)
            elif method == "cross_correlation":
                # Use cross-correlation with reference signal
                # (would need a reference signal for this method)
                direction_powers[i] = np.mean(beamformed**2)
        
        # Find best direction
        best_idx = np.argmax(direction_powers)
        best_beamformed = beamformed_signals[best_idx]
        
        return best_beamformed, direction_powers, best_idx
    
    def calibrate_delays(
        self,
        calibration_signals: List[np.ndarray],
        known_directions: List[np.ndarray],
        method: str = "cross_correlation"
    ) -> Dict[str, Any]:
        """
        Calibrate array delays using known source directions.
        
        Args:
            calibration_signals: List of multi-sensor signals
            known_directions: List of corresponding source directions
            method: Calibration method
            
        Returns:
            Calibration results including delay corrections
        """
        if len(calibration_signals) != len(known_directions):
            raise ValueError("Number of signals must match number of directions")
        
        # Initialize delay corrections
        delay_corrections = np.zeros(self.n_sensors)
        calibration_errors = []
        
        for signal_data, true_direction in zip(calibration_signals, known_directions):
            # Calculate theoretical delays
            theoretical_delays = self.calculate_delays(true_direction)
            
            if method == "cross_correlation":
                # Estimate actual delays using cross-correlation
                estimated_delays = self._estimate_delays_xcorr(signal_data)
                
                # Calculate correction
                delay_error = estimated_delays - theoretical_delays
                calibration_errors.append(delay_error)
        
        # Average delay corrections across calibration signals
        if calibration_errors:
            delay_corrections = np.mean(calibration_errors, axis=0)
        
        calibration_results = {
            'delay_corrections': delay_corrections,
            'calibration_errors': calibration_errors,
            'rms_error': np.sqrt(np.mean(delay_corrections**2)),
            'max_error': np.max(np.abs(delay_corrections))
        }
        
        return calibration_results
    
    def _estimate_delays_xcorr(self, signal_data: np.ndarray, reference_channel: int = 0) -> np.ndarray:
        """Estimate relative delays using cross-correlation."""
        n_sensors, n_samples = signal_data.shape
        estimated_delays = np.zeros(n_sensors)
        
        reference_signal = signal_data[reference_channel]
        
        for i in range(n_sensors):
            if i == reference_channel:
                estimated_delays[i] = 0.0
            else:
                # Cross-correlation with reference
                xcorr = np.correlate(reference_signal, signal_data[i], mode='full')
                
                # Find peak
                peak_idx = np.argmax(np.abs(xcorr))
                delay_samples = peak_idx - (len(xcorr) - 1) // 2
                
                # Convert to time
                estimated_delays[i] = delay_samples / self.sample_rate
        
        return estimated_delays