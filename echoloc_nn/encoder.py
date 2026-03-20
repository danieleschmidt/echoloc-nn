"""
ChirpEncoder: 1D convolutional encoder for raw chirp/echo signals.

Implements a multi-filter-bank 1D convolution using numpy correlate,
followed by activation and pooling to produce a fixed-size feature vector.
"""

import numpy as np
from typing import Optional


class ChirpEncoder:
    """
    Encode a raw acoustic signal into a fixed-size feature vector.

    Architecture:
    1. Multi-scale filter bank (1D convolution via numpy.correlate)
    2. ReLU activation
    3. Max pooling per filter → fixed-size output
    4. L2 normalization

    Parameters
    ----------
    n_filters : int
        Number of filters in the filter bank. Default: 16.
    filter_sizes : list of int, optional
        Kernel sizes for each filter scale. If None, uses
        [32, 64, 128, 256] repeated to fill n_filters.
    pool_size : int
        Number of output values per filter after pooling. Default: 8.
    seed : int, optional
        Random seed for reproducible filter initialization.
    """

    def __init__(
        self,
        n_filters: int = 16,
        filter_sizes: Optional[list] = None,
        pool_size: int = 8,
        seed: int = 42,
    ):
        self.n_filters = n_filters
        self.pool_size = pool_size

        if filter_sizes is None:
            base_sizes = [32, 64, 128, 256]
            filter_sizes = [base_sizes[i % len(base_sizes)] for i in range(n_filters)]
        self.filter_sizes = filter_sizes

        rng = np.random.default_rng(seed)
        self.filters = []
        for size in self.filter_sizes:
            # Initialize with band-pass-like filters (sinusoidal * window)
            t = np.linspace(0, 1, size)
            freq = rng.uniform(0.1, 0.9)  # normalized frequency
            filt = np.sin(2 * np.pi * freq * t) * np.hanning(size)
            filt /= np.linalg.norm(filt) + 1e-8
            self.filters.append(filt)

        self.feature_size = n_filters * pool_size

    def encode(self, signal: np.ndarray) -> np.ndarray:
        """
        Encode a raw acoustic signal into a feature vector.

        Parameters
        ----------
        signal : np.ndarray
            1-D input signal of shape (n_samples,).

        Returns
        -------
        np.ndarray
            Feature vector of shape (n_filters * pool_size,).
        """
        if signal.ndim != 1:
            raise ValueError(f"signal must be 1-D, got shape {signal.shape}")
        if len(signal) == 0:
            raise ValueError("signal must not be empty")

        features = []

        for filt in self.filters:
            # 1D cross-correlation (same mode → preserves length)
            conv_out = np.correlate(signal, filt, mode="same")

            # ReLU activation
            activated = np.maximum(conv_out, 0.0)

            # Max pooling: split into pool_size chunks, take max of each
            pooled = self._max_pool(activated, self.pool_size)
            features.append(pooled)

        feature_vec = np.concatenate(features)

        # L2 normalize
        norm = np.linalg.norm(feature_vec)
        if norm > 1e-8:
            feature_vec = feature_vec / norm

        return feature_vec.astype(np.float64)

    def _max_pool(self, x: np.ndarray, out_size: int) -> np.ndarray:
        """
        1D max pooling that reduces x to exactly out_size values.

        Parameters
        ----------
        x : np.ndarray
            1-D input array.
        out_size : int
            Desired output size.

        Returns
        -------
        np.ndarray
            Pooled array of shape (out_size,).
        """
        n = len(x)
        if n == 0:
            return np.zeros(out_size)

        # Split into out_size roughly equal chunks
        indices = np.array_split(np.arange(n), out_size)
        result = np.array([x[idx].max() if len(idx) > 0 else 0.0 for idx in indices])
        return result

    @property
    def output_size(self) -> int:
        """Size of the output feature vector."""
        return self.feature_size

    def __repr__(self) -> str:
        return (
            f"ChirpEncoder(n_filters={self.n_filters}, "
            f"pool_size={self.pool_size}, "
            f"output_size={self.feature_size})"
        )
