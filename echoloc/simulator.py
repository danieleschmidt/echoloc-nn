"""
simulator.py — EchoSimulator: physics-based echo simulation.
"""

import numpy as np
from typing import List, Tuple
from .signals import ChirpSignal


class EchoSimulator:
    """
    Simulate the received signal from a collection of point reflectors.

    Parameters
    ----------
    max_range : float
        Maximum useful range in metres (default 10.0).
    sample_rate : int
        Samples per second (default 44100).
    speed_of_sound : float
        Speed of sound in m/s (default 343.0).
    """

    def __init__(
        self,
        max_range: float = 10.0,
        sample_rate: int = 44100,
        speed_of_sound: float = 343.0,
    ):
        self.max_range = max_range
        self.sample_rate = sample_rate
        self.speed_of_sound = speed_of_sound
        self._reflectors: List[Tuple[float, float, float]] = []

    def add_reflector(
        self, distance: float, angle_deg: float, reflectivity: float = 1.0
    ) -> None:
        """
        Register a point reflector.

        Parameters
        ----------
        distance : float
            Distance from the transducer in metres.
        angle_deg : float
            Bearing of the reflector in degrees.
        reflectivity : float
            Energy reflectivity in (0, 1] (default 1.0).
        """
        self._reflectors.append((float(distance), float(angle_deg), float(reflectivity)))

    def simulate(self, chirp_signal: ChirpSignal) -> np.ndarray:
        """
        Simulate the total received signal for the given chirp.

        Each reflector contributes a delayed, attenuated copy of the chirp.
        Delay  = 2 * distance / speed_of_sound  (round-trip travel time).
        Gain   = reflectivity / distance²

        Parameters
        ----------
        chirp_signal : ChirpSignal
            The transmitted pulse.

        Returns
        -------
        np.ndarray
            Received signal of the same length as the transmitted pulse.
        """
        emitted = chirp_signal.generate()
        n = len(emitted)
        received = np.zeros(n)

        for distance, _angle_deg, reflectivity in self._reflectors:
            if distance <= 0:
                continue
            delay_s = 2.0 * distance / self.speed_of_sound
            delay_samples = int(round(delay_s * self.sample_rate))
            if delay_samples >= n:
                continue  # echo arrives after the window — ignore

            attenuation = reflectivity / (distance ** 2)
            echo = attenuation * emitted
            # Shift the echo by delay_samples
            end = n - delay_samples
            received[delay_samples:] += echo[:end]

        return received
