"""
Signal preprocessing utilities.
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import scipy.signal as signal
from scipy.stats import zscore


class PreProcessor:
    """
    Signal preprocessing for ultrasonic localization.
    
    Provides standardized preprocessing pipeline including filtering,
    normalization, and feature preparation for ML models.
    """
    
    def __init__(self, sample_rate: int = 250000):
        self.sample_rate = sample_rate
        
    def bandpass_filter(
        self,
        signal_data: np.ndarray,
        low_freq: float,
        high_freq: float,
        order: int = 4,
        filter_type: str = "butterworth"
    ) -> np.ndarray:
        """
        Apply bandpass filter to remove out-of-band noise.
        
        Args:
            signal_data: Input signal (n_sensors, n_samples)
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz
            order: Filter order
            filter_type: Filter type ("butterworth", "chebyshev1", "elliptic")
            
        Returns:
            Filtered signal
        """
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        if filter_type == "butterworth":
            b, a = signal.butter(order, [low, high], btype='band')
        elif filter_type == "chebyshev1":
            b, a = signal.cheby1(order, 1, [low, high], btype='band')
        elif filter_type == "elliptic":
            b, a = signal.ellip(order, 1, 40, [low, high], btype='band')
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
            
        if signal_data.ndim == 1:
            return signal.filtfilt(b, a, signal_data)
        else:
            filtered = np.zeros_like(signal_data)
            for i in range(signal_data.shape[0]):
                filtered[i] = signal.filtfilt(b, a, signal_data[i])
            return filtered
    
    def remove_dc_offset(self, signal_data: np.ndarray) -> np.ndarray:
        """Remove DC offset from signal."""
        if signal_data.ndim == 1:
            return signal_data - np.mean(signal_data)
        else:
            return signal_data - np.mean(signal_data, axis=1, keepdims=True)
    
    def normalize_amplitude(
        self, 
        signal_data: np.ndarray, 
        method: str = "max"
    ) -> np.ndarray:
        """
        Normalize signal amplitude.
        
        Args:
            signal_data: Input signal (n_sensors, n_samples)
            method: Normalization method ("max", "rms", "zscore")
            
        Returns:
            Normalized signal
        """
        if method == "max":
            if signal_data.ndim == 1:
                max_val = np.max(np.abs(signal_data))
                return signal_data / max_val if max_val > 0 else signal_data
            else:
                max_vals = np.max(np.abs(signal_data), axis=1, keepdims=True)
                max_vals[max_vals == 0] = 1  # Avoid division by zero
                return signal_data / max_vals
                
        elif method == "rms":
            if signal_data.ndim == 1:
                rms_val = np.sqrt(np.mean(signal_data**2))
                return signal_data / rms_val if rms_val > 0 else signal_data
            else:
                rms_vals = np.sqrt(np.mean(signal_data**2, axis=1, keepdims=True))
                rms_vals[rms_vals == 0] = 1
                return signal_data / rms_vals
                
        elif method == "zscore":
            if signal_data.ndim == 1:
                return zscore(signal_data)
            else:
                normalized = np.zeros_like(signal_data)
                for i in range(signal_data.shape[0]):
                    normalized[i] = zscore(signal_data[i])
                return normalized
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def apply_windowing(
        self,
        signal_data: np.ndarray,
        window_type: str = "hann",
        window_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply windowing function to reduce spectral leakage.
        
        Args:
            signal_data: Input signal (n_sensors, n_samples)
            window_type: Window type ("hann", "hamming", "blackman", "kaiser")
            window_length: Window length (default: signal length)
            
        Returns:
            Windowed signal
        """
        if signal_data.ndim == 1:
            length = window_length or len(signal_data)
            window = signal.get_window(window_type, length)
            if len(window) != len(signal_data):
                window = np.resize(window, len(signal_data))
            return signal_data * window
        else:
            windowed = np.zeros_like(signal_data)
            for i in range(signal_data.shape[0]):
                length = window_length or signal_data.shape[1]
                window = signal.get_window(window_type, length)
                if len(window) != signal_data.shape[1]:
                    window = np.resize(window, signal_data.shape[1])
                windowed[i] = signal_data[i] * window
            return windowed
    
    def resample_signal(
        self,
        signal_data: np.ndarray,
        target_rate: int,
        method: str = "scipy"
    ) -> np.ndarray:
        """
        Resample signal to target sample rate.
        
        Args:
            signal_data: Input signal (n_sensors, n_samples)
            target_rate: Target sample rate in Hz
            method: Resampling method ("scipy", "polyphase")
            
        Returns:
            Resampled signal
        """
        if target_rate == self.sample_rate:
            return signal_data
            
        ratio = target_rate / self.sample_rate
        
        if signal_data.ndim == 1:
            if method == "scipy":
                return signal.resample(signal_data, int(len(signal_data) * ratio))
            elif method == "polyphase":
                return signal.resample_poly(signal_data, target_rate, self.sample_rate)
        else:
            resampled = []
            for i in range(signal_data.shape[0]):
                if method == "scipy":
                    resampled_ch = signal.resample(
                        signal_data[i], int(signal_data.shape[1] * ratio)
                    )
                elif method == "polyphase":
                    resampled_ch = signal.resample_poly(
                        signal_data[i], target_rate, self.sample_rate
                    )
                resampled.append(resampled_ch)
            return np.array(resampled)
    
    def trim_to_length(
        self,
        signal_data: np.ndarray,
        target_length: int,
        method: str = "center"
    ) -> np.ndarray:
        """
        Trim or pad signal to target length.
        
        Args:
            signal_data: Input signal (n_sensors, n_samples)
            target_length: Target signal length
            method: Trimming method ("center", "start", "end")
            
        Returns:
            Trimmed/padded signal
        """
        if signal_data.ndim == 1:
            current_length = len(signal_data)
        else:
            current_length = signal_data.shape[1]
            
        if current_length == target_length:
            return signal_data
        elif current_length > target_length:
            # Trim signal
            if method == "center":
                start_idx = (current_length - target_length) // 2
                end_idx = start_idx + target_length
            elif method == "start":
                start_idx = 0
                end_idx = target_length
            elif method == "end":
                start_idx = current_length - target_length
                end_idx = current_length
            else:
                raise ValueError(f"Unknown trimming method: {method}")
                
            if signal_data.ndim == 1:
                return signal_data[start_idx:end_idx]
            else:
                return signal_data[:, start_idx:end_idx]
        else:
            # Pad signal
            pad_length = target_length - current_length
            
            if signal_data.ndim == 1:
                if method == "center":
                    pad_left = pad_length // 2
                    pad_right = pad_length - pad_left
                    return np.pad(signal_data, (pad_left, pad_right), mode='constant')
                elif method == "start":
                    return np.pad(signal_data, (pad_length, 0), mode='constant')
                elif method == "end":
                    return np.pad(signal_data, (0, pad_length), mode='constant')
            else:
                if method == "center":
                    pad_left = pad_length // 2
                    pad_right = pad_length - pad_left
                    return np.pad(signal_data, ((0, 0), (pad_left, pad_right)), mode='constant')
                elif method == "start":
                    return np.pad(signal_data, ((0, 0), (pad_length, 0)), mode='constant')
                elif method == "end":
                    return np.pad(signal_data, ((0, 0), (0, pad_length)), mode='constant')
    
    def preprocess_pipeline(
        self,
        signal_data: np.ndarray,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            signal_data: Input signal (n_sensors, n_samples)
            config: Preprocessing configuration dictionary
            
        Returns:
            Preprocessed signal
        """
        processed = signal_data.copy()
        
        # Remove DC offset
        if config.get('remove_dc', True):
            processed = self.remove_dc_offset(processed)
        
        # Bandpass filtering
        if 'bandpass' in config:
            bp_config = config['bandpass']
            processed = self.bandpass_filter(
                processed,
                bp_config['low_freq'],
                bp_config['high_freq'],
                bp_config.get('order', 4),
                bp_config.get('filter_type', 'butterworth')
            )
        
        # Resampling
        if 'target_sample_rate' in config:
            processed = self.resample_signal(
                processed,
                config['target_sample_rate'],
                config.get('resample_method', 'scipy')
            )
            self.sample_rate = config['target_sample_rate']
        
        # Length normalization
        if 'target_length' in config:
            processed = self.trim_to_length(
                processed,
                config['target_length'],
                config.get('trim_method', 'center')
            )
        
        # Windowing
        if 'window' in config:
            processed = self.apply_windowing(
                processed,
                config['window'].get('type', 'hann'),
                config['window'].get('length')
            )
        
        # Amplitude normalization
        if 'normalize' in config:
            processed = self.normalize_amplitude(
                processed,
                config['normalize'].get('method', 'max')
            )
        
        return processed


class SignalNormalizer:
    """
    Signal normalization and standardization utilities.
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
        
    def fit(self, signal_data: np.ndarray):
        """Compute normalization statistics from training data."""
        if signal_data.ndim == 3:  # (batch, sensors, samples)
            # Compute statistics across batch and time dimensions
            self.mean_ = np.mean(signal_data, axis=(0, 2), keepdims=True)
            self.std_ = np.std(signal_data, axis=(0, 2), keepdims=True)
            self.min_ = np.min(signal_data, axis=(0, 2), keepdims=True)
            self.max_ = np.max(signal_data, axis=(0, 2), keepdims=True)
        elif signal_data.ndim == 2:  # (sensors, samples)
            self.mean_ = np.mean(signal_data, axis=1, keepdims=True)
            self.std_ = np.std(signal_data, axis=1, keepdims=True)
            self.min_ = np.min(signal_data, axis=1, keepdims=True)
            self.max_ = np.max(signal_data, axis=1, keepdims=True)
        
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1
        
    def transform(self, signal_data: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Apply normalization transformation."""
        if self.mean_ is None:
            raise ValueError("Must call fit() before transform()")
            
        if method == "zscore":
            return (signal_data - self.mean_) / self.std_
        elif method == "minmax":
            return (signal_data - self.min_) / (self.max_ - self.min_)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def fit_transform(self, signal_data: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Fit normalizer and transform data in one step."""
        self.fit(signal_data)
        return self.transform(signal_data, method)
    
    def inverse_transform(self, normalized_data: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Inverse transformation to original scale."""
        if method == "zscore":
            return normalized_data * self.std_ + self.mean_
        elif method == "minmax":
            return normalized_data * (self.max_ - self.min_) + self.min_
        else:
            raise ValueError(f"Unknown normalization method: {method}")