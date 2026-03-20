"""
ChirpSignal: Linear frequency-modulated (LFM) chirp signal generator.

Inspired by bat echolocation — bats emit chirps that sweep from high to low
frequency, allowing them to resolve range and velocity of targets.
"""

import numpy as np
from typing import Optional


class ChirpSignal:
    """
    Generate linear frequency-modulated (LFM) chirp signals.

    A chirp sweeps from frequency f0 to f1 over a given duration.
    The instantaneous frequency is: f(t) = f0 + (f1 - f0) * t / duration

    Parameters
    ----------
    window : str, optional
        Window function to apply ('hann', 'hamming', 'blackman', or None).
        Default is 'hann' to reduce spectral leakage.
    """

    SPEED_OF_SOUND = 340.0  # m/s

    def __init__(self, window: Optional[str] = "hann"):
        self.window = window

    def generate(
        self,
        duration: float,
        f0: float,
        f1: float,
        sr: int = 44100,
    ) -> np.ndarray:
        """
        Generate a linear chirp waveform.

        Parameters
        ----------
        duration : float
            Duration of the chirp in seconds.
        f0 : float
            Start frequency in Hz.
        f1 : float
            End frequency in Hz.
        sr : int
            Sample rate in Hz. Default: 44100.

        Returns
        -------
        np.ndarray
            1-D float64 waveform of shape (n_samples,), normalized to [-1, 1].
        """
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        if f0 <= 0 or f1 <= 0:
            raise ValueError(f"Frequencies must be positive, got f0={f0}, f1={f1}")
        if sr <= 0:
            raise ValueError(f"Sample rate must be positive, got {sr}")
        if f0 > sr / 2 or f1 > sr / 2:
            raise ValueError(
                f"Frequencies must be below Nyquist ({sr/2} Hz), "
                f"got f0={f0}, f1={f1}"
            )

        n_samples = int(duration * sr)
        t = np.linspace(0.0, duration, n_samples, endpoint=False)

        # Linear chirp: phase = 2*pi * (f0*t + 0.5*(f1-f0)/duration * t^2)
        sweep_rate = (f1 - f0) / (2.0 * duration)
        phase = 2.0 * np.pi * (f0 * t + sweep_rate * t**2)
        chirp = np.sin(phase)

        # Apply window if requested
        if self.window == "hann":
            chirp *= np.hanning(n_samples)
        elif self.window == "hamming":
            chirp *= np.hamming(n_samples)
        elif self.window == "blackman":
            chirp *= np.blackman(n_samples)
        elif self.window is not None:
            raise ValueError(f"Unknown window: {self.window}. Use hann/hamming/blackman/None.")

        return chirp.astype(np.float64)

    def instantaneous_frequency(
        self,
        duration: float,
        f0: float,
        f1: float,
        sr: int = 44100,
    ) -> np.ndarray:
        """
        Return the instantaneous frequency at each sample.

        Returns
        -------
        np.ndarray
            Instantaneous frequency array of shape (n_samples,).
        """
        n_samples = int(duration * sr)
        t = np.linspace(0.0, duration, n_samples, endpoint=False)
        return f0 + (f1 - f0) * t / duration

    def max_range(self, duration: float) -> float:
        """
        Maximum unambiguous range for a chirp of given duration (meters).

        Parameters
        ----------
        duration : float
            Chirp duration in seconds.

        Returns
        -------
        float
            Max range in meters: speed_of_sound * duration / 2
        """
        return self.SPEED_OF_SOUND * duration / 2.0
