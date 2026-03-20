"""
EchoSimulator: Simulate acoustic echoes from reflective targets.

Models how a chirp signal bounces off objects at various distances,
amplitudes, and angles — like bat sonar returning echoes.
"""

import numpy as np
from typing import List, Tuple


# Type alias: (distance_m, amplitude, angle_deg)
Reflection = Tuple[float, float, float]


class EchoSimulator:
    """
    Simulate multi-path acoustic echo returns.

    Given a transmitted chirp and a list of reflections, computes the
    received signal as a sum of attenuated, delayed copies of the chirp.

    Parameters
    ----------
    speed_of_sound : float
        Speed of sound in m/s. Default: 340.0
    noise_std : float
        Standard deviation of additive Gaussian noise. Default: 0.0
    sr : int
        Sample rate in Hz. Default: 44100
    """

    def __init__(
        self,
        speed_of_sound: float = 340.0,
        noise_std: float = 0.0,
        sr: int = 44100,
        seed: int = None,
    ):
        self.speed_of_sound = speed_of_sound
        self.noise_std = noise_std
        self.sr = sr
        self._rng = np.random.default_rng(seed)

    def simulate(
        self,
        chirp: np.ndarray,
        reflections: List[Reflection],
    ) -> np.ndarray:
        """
        Simulate the received echo signal.

        Parameters
        ----------
        chirp : np.ndarray
            Transmitted chirp waveform, shape (n_samples,).
        reflections : list of (distance, amplitude, angle) tuples
            Each reflection:
              - distance (float): target distance in meters (> 0)
              - amplitude (float): reflection coefficient in [0, 1]
              - angle (float): azimuth angle in degrees [-90, 90]

        Returns
        -------
        np.ndarray
            Received signal of shape (n_samples + max_delay,).
            Contains the sum of all delayed/attenuated echoes plus noise.
        """
        if chirp.ndim != 1:
            raise ValueError(f"chirp must be 1-D, got shape {chirp.shape}")
        if not reflections:
            raise ValueError("reflections list must not be empty")

        # Compute the maximum delay to size the output buffer
        max_delay_samples = 0
        echo_params = []
        for distance, amplitude, angle in reflections:
            if distance <= 0:
                raise ValueError(f"distance must be positive, got {distance}")
            # Two-way travel time (emit → target → receiver)
            travel_time = 2.0 * distance / self.speed_of_sound
            delay_samples = int(np.round(travel_time * self.sr))
            max_delay_samples = max(max_delay_samples, delay_samples)
            echo_params.append((delay_samples, amplitude, angle))

        output_len = len(chirp) + max_delay_samples
        received = np.zeros(output_len, dtype=np.float64)

        # Add each echo
        for delay_samples, amplitude, angle in echo_params:
            # Angle-dependent gain: cosine weighting (max at 0°, zero at ±90°)
            angle_rad = np.deg2rad(angle)
            directional_gain = np.cos(angle_rad) ** 2

            # 1/r² spreading loss (normalized by reference distance 1 m)
            # distance already baked into amplitude by caller, but we apply
            # a basic directional gain here
            effective_amp = amplitude * directional_gain

            end = delay_samples + len(chirp)
            received[delay_samples:end] += effective_amp * chirp

        # Add noise
        if self.noise_std > 0:
            noise = self._rng.normal(0.0, self.noise_std, size=output_len)
            received += noise

        return received

    def delay_samples(self, distance: float) -> int:
        """
        Compute the round-trip delay in samples for a given distance.

        Parameters
        ----------
        distance : float
            Distance to target in meters.

        Returns
        -------
        int
            Number of samples for the round-trip delay.
        """
        travel_time = 2.0 * distance / self.speed_of_sound
        return int(np.round(travel_time * self.sr))

    def delay_seconds(self, distance: float) -> float:
        """
        Round-trip travel time for a given distance.

        Parameters
        ----------
        distance : float
            Distance in meters.

        Returns
        -------
        float
            Round-trip time in seconds.
        """
        return 2.0 * distance / self.speed_of_sound
