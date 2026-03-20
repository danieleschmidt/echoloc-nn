"""
encoder.py — ChirpEncoder: cosine-modulated Gaussian filter bank.
"""

import numpy as np
from typing import List


class ChirpEncoder:
    """
    Encode a 1-D signal into a compact feature vector using a bank of
    cosine-modulated Gaussian (Gabor-like) filters.

    Parameters
    ----------
    n_filters : int
        Number of filters in the bank (default 16).
    filter_size : int
        Length of each filter in samples (default 64).
    """

    def __init__(self, n_filters: int = 16, filter_size: int = 64):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self._filters = self._build_filters()

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _build_filters(self) -> np.ndarray:
        """
        Build a bank of cosine-modulated Gaussian filters.

        Filter k has centre frequency  f_k = (k+1) / (n_filters + 1)
        (normalised, range 0–0.5 Nyquist) and a fixed Gaussian envelope.

        Returns
        -------
        np.ndarray of shape (n_filters, filter_size)
        """
        t = np.linspace(-1, 1, self.filter_size)
        sigma = 0.4  # controls bandwidth
        filters = np.zeros((self.n_filters, self.filter_size))
        for k in range(self.n_filters):
            # Normalised frequency: evenly spaced between 0.05 and 0.45
            f_k = 0.05 + 0.40 * k / max(self.n_filters - 1, 1)
            envelope = np.exp(-0.5 * (t / sigma) ** 2)
            carrier = np.cos(2.0 * np.pi * f_k * self.filter_size * t / 2.0)
            filt = envelope * carrier
            # Unit-normalise
            norm = np.linalg.norm(filt)
            filters[k] = filt / norm if norm > 0 else filt
        return filters

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def encode(self, signal: np.ndarray) -> np.ndarray:
        """
        Encode a single 1-D signal into a feature vector.

        Applies each filter via valid-mode convolution, then takes the
        absolute value and max-pools across time.

        Parameters
        ----------
        signal : np.ndarray
            1-D input array.

        Returns
        -------
        np.ndarray of shape (n_filters,)
        """
        features = np.zeros(self.n_filters)
        for k, filt in enumerate(self._filters):
            conv = np.convolve(signal, filt, mode="valid")
            features[k] = np.max(np.abs(conv)) if len(conv) > 0 else 0.0
        return features

    def encode_batch(self, signals: List[np.ndarray]) -> np.ndarray:
        """
        Encode a list (or array) of signals.

        Parameters
        ----------
        signals : list of np.ndarray

        Returns
        -------
        np.ndarray of shape (len(signals), n_filters)
        """
        return np.vstack([self.encode(s) for s in signals])
