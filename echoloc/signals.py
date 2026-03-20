"""
signals.py — ChirpSignal: linear frequency-swept pulse with Gaussian envelope.
"""

import numpy as np


class ChirpSignal:
    """
    Generate a linear chirp (frequency-swept) signal with a Gaussian envelope.

    Parameters
    ----------
    f_start : float
        Start frequency in Hz (default 1000).
    f_end : float
        End frequency in Hz (default 10000).
    duration : float
        Duration of the pulse in seconds (default 0.01).
    sample_rate : int
        Samples per second (default 44100).
    """

    def __init__(
        self,
        f_start: float = 1000.0,
        f_end: float = 10000.0,
        duration: float = 0.01,
        sample_rate: int = 44100,
    ):
        self.f_start = f_start
        self.f_end = f_end
        self._duration = duration
        self._sample_rate = sample_rate

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def n_samples(self) -> int:
        return int(self._duration * self._sample_rate)

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #

    def generate(self) -> np.ndarray:
        """
        Generate the chirp signal.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with values in [-1, 1].

        Formula
        -------
        s(t) = A(t) * sin(2π * (f_start * t + (f_end - f_start) / (2 * duration) * t²))

        where A(t) is a Gaussian envelope centred at t = duration / 2.
        """
        t = np.linspace(0, self._duration, self.n_samples, endpoint=False)
        phase = 2.0 * np.pi * (
            self.f_start * t
            + (self.f_end - self.f_start) / (2.0 * self._duration) * t ** 2
        )
        raw = np.sin(phase)

        # Gaussian envelope: σ = duration / 6  (covers ±3σ over the window)
        sigma = self._duration / 6.0
        t_center = self._duration / 2.0
        envelope = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)

        signal = envelope * raw

        # Normalise to [-1, 1]
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal /= peak

        return signal
