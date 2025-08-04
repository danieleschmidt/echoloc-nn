"""
Physics-based simulation for training data generation.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import scipy.signal as signal
from dataclasses import dataclass
import random


@dataclass
class MaterialProperties:
    """Acoustic properties of materials."""
    
    name: str
    absorption_coefficient: float  # [0, 1]
    reflection_coefficient: float  # [0, 1] 
    scattering_coefficient: float  # [0, 1]
    impedance: float  # Acoustic impedance
    
    def __post_init__(self):
        # Ensure energy conservation
        total_coeff = self.absorption_coefficient + self.reflection_coefficient + self.scattering_coefficient
        if total_coeff > 1.0:
            # Normalize coefficients
            self.absorption_coefficient /= total_coeff
            self.reflection_coefficient /= total_coeff
            self.scattering_coefficient /= total_coeff


class RoomAcoustics:
    """
    Physics-based room acoustics simulation.
    
    Models ultrasonic wave propagation, reflections, and multipath
    effects in indoor environments.
    """
    
    def __init__(
        self,
        room_dimensions: Tuple[float, float, float],
        speed_of_sound: float = 343.0,
        temperature: float = 20.0,
        humidity: float = 50.0
    ):
        self.room_dimensions = room_dimensions  # (width, height, depth)
        self.speed_of_sound = speed_of_sound
        self.temperature = temperature
        self.humidity = humidity
        
        # Material properties database
        self.materials = self._create_material_database()
        
        # Room surfaces (walls, floor, ceiling)
        self.surfaces = self._initialize_surfaces()
        
        # Atmospheric attenuation
        self.attenuation_coeff = self._calculate_atmospheric_attenuation()
        
    def _create_material_database(self) -> Dict[str, MaterialProperties]:
        """Create database of common material properties."""
        return {
            'concrete': MaterialProperties(
                name='concrete',
                absorption_coefficient=0.05,
                reflection_coefficient=0.90,
                scattering_coefficient=0.05,
                impedance=8.0e6
            ),
            'drywall': MaterialProperties(
                name='drywall',
                absorption_coefficient=0.15,
                reflection_coefficient=0.80,
                scattering_coefficient=0.05,
                impedance=2.0e5
            ),
            'glass': MaterialProperties(
                name='glass',
                absorption_coefficient=0.02,
                reflection_coefficient=0.95,
                scattering_coefficient=0.03,
                impedance=1.2e7
            ),
            'wood': MaterialProperties(
                name='wood',
                absorption_coefficient=0.25,
                reflection_coefficient=0.70,
                scattering_coefficient=0.05,
                impedance=4.0e5
            ),
            'fabric': MaterialProperties(
                name='fabric',
                absorption_coefficient=0.60,
                reflection_coefficient=0.35,
                scattering_coefficient=0.05,
                impedance=1.0e4
            ),
            'metal': MaterialProperties(
                name='metal',
                absorption_coefficient=0.01,
                reflection_coefficient=0.98,
                scattering_coefficient=0.01,
                impedance=4.0e7
            )
        }
    
    def _initialize_surfaces(self) -> List[Dict[str, Any]]:
        """Initialize room surfaces with default materials."""
        w, h, d = self.room_dimensions
        
        surfaces = [
            # Floor
            {
                'vertices': [(0, 0, 0), (w, 0, 0), (w, 0, d), (0, 0, d)],
                'normal': (0, 1, 0),
                'material': self.materials['concrete']
            },
            # Ceiling  
            {
                'vertices': [(0, h, 0), (0, h, d), (w, h, d), (w, h, 0)],
                'normal': (0, -1, 0),
                'material': self.materials['drywall']
            },
            # Left wall
            {
                'vertices': [(0, 0, 0), (0, 0, d), (0, h, d), (0, h, 0)],
                'normal': (1, 0, 0),
                'material': self.materials['drywall']
            },
            # Right wall
            {
                'vertices': [(w, 0, 0), (w, h, 0), (w, h, d), (w, 0, d)],
                'normal': (-1, 0, 0),
                'material': self.materials['drywall']
            },
            # Front wall
            {
                'vertices': [(0, 0, 0), (0, h, 0), (w, h, 0), (w, 0, 0)],
                'normal': (0, 0, 1),
                'material': self.materials['drywall']
            },
            # Back wall
            {
                'vertices': [(0, 0, d), (w, 0, d), (w, h, d), (0, h, d)],
                'normal': (0, 0, -1),
                'material': self.materials['drywall']
            }
        ]
        
        return surfaces
    
    def _calculate_atmospheric_attenuation(self) -> float:
        """Calculate atmospheric attenuation coefficient."""
        # Simplified model for ultrasonic attenuation in air
        # Based on temperature and humidity
        frequency = 40000  # 40 kHz
        
        # Classical attenuation (viscosity and thermal conduction)
        classical = 1.84e-11 * (frequency**2) / 101325
        
        # Molecular relaxation attenuation
        # Simplified - full model would include O2 and N2 relaxation
        molecular = 0.01 * frequency**1.5 / 1e6
        
        return classical + molecular
    
    def simulate_propagation(
        self,
        source_pos: Tuple[float, float, float],
        receiver_pos: Tuple[float, float, float],
        chirp_signal: np.ndarray,
        sample_rate: int,
        max_reflections: int = 3,
        min_amplitude: float = 0.01
    ) -> np.ndarray:
        """
        Simulate ultrasonic wave propagation with multipath effects.
        
        Args:
            source_pos: Source position (x, y, z)
            receiver_pos: Receiver position (x, y, z)
            chirp_signal: Transmitted chirp signal
            sample_rate: Sampling rate in Hz
            max_reflections: Maximum number of reflections to consider
            min_amplitude: Minimum amplitude threshold
            
        Returns:
            Received signal with multipath components
        """
        # Initialize output signal
        max_time = 0.2  # 200ms maximum propagation time
        max_samples = int(max_time * sample_rate)
        received_signal = np.zeros(max_samples)
        
        # Find all propagation paths (direct + reflections)
        paths = self._find_propagation_paths(
            source_pos, receiver_pos, max_reflections, min_amplitude
        )
        
        # Add each path to the received signal
        for path in paths:
            path_signal = self._apply_path_effects(
                chirp_signal, path, sample_rate
            )
            
            # Add to received signal with proper delay
            delay_samples = int(path['delay'] * sample_rate)
            end_idx = min(delay_samples + len(path_signal), max_samples)
            
            if delay_samples < max_samples:
                signal_length = end_idx - delay_samples
                received_signal[delay_samples:end_idx] += path_signal[:signal_length]
        
        return received_signal
    
    def _find_propagation_paths(
        self,
        source_pos: Tuple[float, float, float],
        receiver_pos: Tuple[float, float, float],
        max_reflections: int,
        min_amplitude: float
    ) -> List[Dict[str, Any]]:
        """Find all significant propagation paths using ray tracing."""
        paths = []
        
        # Direct path
        direct_distance = np.linalg.norm(np.array(receiver_pos) - np.array(source_pos))
        direct_delay = direct_distance / self.speed_of_sound
        direct_amplitude = self._calculate_path_amplitude(direct_distance, 0)
        
        if direct_amplitude >= min_amplitude:
            paths.append({
                'type': 'direct',
                'distance': direct_distance,
                'delay': direct_delay,
                'amplitude': direct_amplitude,
                'reflections': [],
                'path_points': [source_pos, receiver_pos]
            })
        
        # First-order reflections (single bounce)
        if max_reflections >= 1:
            for surface in self.surfaces:
                reflected_path = self._calculate_reflection_path(
                    source_pos, receiver_pos, surface
                )
                
                if reflected_path and reflected_path['amplitude'] >= min_amplitude:
                    paths.append(reflected_path)
        
        # Higher-order reflections (multiple bounces)
        if max_reflections >= 2:
            # Simplified: only consider a few important multi-bounce paths
            # Full implementation would use recursive ray tracing
            for i, surface1 in enumerate(self.surfaces[:4]):  # Only walls
                for j, surface2 in enumerate(self.surfaces[:4]):
                    if i != j:
                        double_reflection = self._calculate_double_reflection(
                            source_pos, receiver_pos, surface1, surface2
                        )
                        
                        if double_reflection and double_reflection['amplitude'] >= min_amplitude:
                            paths.append(double_reflection)
        
        return paths
    
    def _calculate_reflection_path(
        self,
        source_pos: Tuple[float, float, float],
        receiver_pos: Tuple[float, float, float],
        surface: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Calculate single reflection path."""
        # Mirror source position across the surface
        mirrored_source = self._mirror_point_across_surface(source_pos, surface)
        
        # Check if direct line from mirrored source to receiver intersects surface
        intersection = self._line_surface_intersection(mirrored_source, receiver_pos, surface)
        
        if intersection is None:
            return None
        
        # Calculate path properties
        source_to_surface = np.linalg.norm(np.array(intersection) - np.array(source_pos))
        surface_to_receiver = np.linalg.norm(np.array(receiver_pos) - np.array(intersection))
        total_distance = source_to_surface + surface_to_receiver
        
        delay = total_distance / self.speed_of_sound
        amplitude = self._calculate_path_amplitude(total_distance, 1, surface['material'])
        
        return {
            'type': 'single_reflection',
            'distance': total_distance,
            'delay': delay,
            'amplitude': amplitude,
            'reflections': [surface],
            'path_points': [source_pos, intersection, receiver_pos]
        }
    
    def _calculate_double_reflection(
        self,
        source_pos: Tuple[float, float, float],
        receiver_pos: Tuple[float, float, float],
        surface1: Dict[str, Any],
        surface2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Calculate double reflection path (simplified)."""
        # This is a simplified implementation
        # Full implementation would require more complex geometric calculations
        
        # Estimate path through room center as intermediate point
        room_center = tuple(d/2 for d in self.room_dimensions)
        
        # Approximate path distance
        dist1 = np.linalg.norm(np.array(room_center) - np.array(source_pos))
        dist2 = np.linalg.norm(np.array(receiver_pos) - np.array(room_center))
        total_distance = dist1 + dist2 + 1.0  # Add extra for reflections
        
        delay = total_distance / self.speed_of_sound
        amplitude = self._calculate_path_amplitude(total_distance, 2, surface1['material'])
        amplitude *= surface2['material'].reflection_coefficient
        
        if amplitude < 0.001:  # Very weak double reflections
            return None
        
        return {
            'type': 'double_reflection',
            'distance': total_distance,
            'delay': delay,
            'amplitude': amplitude,
            'reflections': [surface1, surface2],
            'path_points': [source_pos, room_center, receiver_pos]
        }
    
    def _mirror_point_across_surface(
        self,
        point: Tuple[float, float, float],
        surface: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """Mirror a point across a surface."""
        # Simplified for axis-aligned surfaces
        normal = surface['normal']
        
        if normal == (0, 1, 0):  # Floor
            return (point[0], -point[1], point[2])
        elif normal == (0, -1, 0):  # Ceiling
            return (point[0], 2*self.room_dimensions[1] - point[1], point[2])
        elif normal == (1, 0, 0):  # Left wall
            return (-point[0], point[1], point[2])
        elif normal == (-1, 0, 0):  # Right wall
            return (2*self.room_dimensions[0] - point[0], point[1], point[2])
        elif normal == (0, 0, 1):  # Front wall
            return (point[0], point[1], -point[2])
        elif normal == (0, 0, -1):  # Back wall
            return (point[0], point[1], 2*self.room_dimensions[2] - point[2])
        
        return point
    
    def _line_surface_intersection(
        self,
        point1: Tuple[float, float, float],
        point2: Tuple[float, float, float],
        surface: Dict[str, Any]
    ) -> Optional[Tuple[float, float, float]]:
        """Find intersection of line with surface."""
        # Simplified for axis-aligned surfaces
        normal = surface['normal']
        
        # Line parametric form: P = point1 + t * (point2 - point1)
        direction = np.array(point2) - np.array(point1)
        
        if normal == (0, 1, 0):  # Floor (y = 0)
            if abs(direction[1]) < 1e-6:
                return None
            t = -point1[1] / direction[1]
        elif normal == (0, -1, 0):  # Ceiling
            if abs(direction[1]) < 1e-6:
                return None
            t = (self.room_dimensions[1] - point1[1]) / direction[1]
        elif normal == (1, 0, 0):  # Left wall (x = 0)
            if abs(direction[0]) < 1e-6:
                return None
            t = -point1[0] / direction[0]
        elif normal == (-1, 0, 0):  # Right wall
            if abs(direction[0]) < 1e-6:
                return None
            t = (self.room_dimensions[0] - point1[0]) / direction[0]
        elif normal == (0, 0, 1):  # Front wall (z = 0)
            if abs(direction[2]) < 1e-6:
                return None
            t = -point1[2] / direction[2]
        elif normal == (0, 0, -1):  # Back wall
            if abs(direction[2]) < 1e-6:
                return None
            t = (self.room_dimensions[2] - point1[2]) / direction[2]
        else:
            return None
        
        # Check if intersection is within line segment and surface bounds
        if t < 0 or t > 1:
            return None
        
        intersection = np.array(point1) + t * direction
        
        # Check if intersection is within surface bounds
        if (0 <= intersection[0] <= self.room_dimensions[0] and
            0 <= intersection[1] <= self.room_dimensions[1] and
            0 <= intersection[2] <= self.room_dimensions[2]):
            return tuple(intersection)
        
        return None
    
    def _calculate_path_amplitude(
        self,
        distance: float,
        num_reflections: int,
        material: Optional[MaterialProperties] = None
    ) -> float:
        """Calculate amplitude for a propagation path."""
        # Geometric spreading loss (1/r)
        geometric_loss = 1.0 / (4 * np.pi * distance**2)
        
        # Atmospheric attenuation
        atmospheric_loss = np.exp(-self.attenuation_coeff * distance)
        
        # Reflection losses
        reflection_loss = 1.0
        if material and num_reflections > 0:
            reflection_loss = material.reflection_coefficient ** num_reflections
        
        return geometric_loss * atmospheric_loss * reflection_loss
    
    def _apply_path_effects(
        self,
        signal: np.ndarray,
        path: Dict[str, Any],
        sample_rate: int
    ) -> np.ndarray:
        """Apply path-specific effects to signal."""
        # Start with scaled signal
        path_signal = signal * path['amplitude']
        
        # Add Doppler effects (if receiver/source moving - not implemented)
        
        # Add phase shifts from reflections
        for reflection in path['reflections']:
            # Simple phase shift for reflection
            phase_shift = np.pi  # 180 degree phase shift
            path_signal = -path_signal  # Simplified phase inversion
        
        # Add frequency-dependent attenuation
        if len(path['reflections']) > 0:
            # High frequencies attenuate more in reflections
            path_signal = self._apply_frequency_attenuation(path_signal, sample_rate)
        
        return path_signal
    
    def _apply_frequency_attenuation(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply frequency-dependent attenuation."""
        # Simple low-pass filtering to simulate high-frequency attenuation
        cutoff = 35000  # Hz
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, signal)


class EchoSimulator:
    """
    High-level echo simulation for training data generation.
    
    Combines room acoustics simulation with sensor modeling
    and noise injection to create realistic training datasets.
    """
    
    def __init__(
        self,
        room_dimensions: Tuple[float, float, float] = (5.0, 3.0, 4.0),
        wall_materials: List[str] = ['drywall', 'drywall', 'drywall', 'drywall'],
        sample_rate: int = 250000,
        noise_floor: float = -60  # dB
    ):
        self.room_acoustics = RoomAcoustics(room_dimensions)
        self.sample_rate = sample_rate
        self.noise_floor = noise_floor
        
        # Set wall materials
        if len(wall_materials) >= 4:
            for i, material_name in enumerate(wall_materials[:4]):
                if material_name in self.room_acoustics.materials:
                    self.room_acoustics.surfaces[i+2]['material'] = \
                        self.room_acoustics.materials[material_name]
    
    def generate_training_sample(
        self,
        target_position: Tuple[float, float, float],
        sensor_positions: List[Tuple[float, float, float]],
        chirp_signal: np.ndarray,
        snr_db: float = 20.0,
        add_clutter: bool = True
    ) -> Dict[str, Any]:
        """
        Generate single training sample.
        
        Args:
            target_position: Target position to localize
            sensor_positions: List of sensor positions
            chirp_signal: Chirp signal to transmit
            snr_db: Signal-to-noise ratio in dB
            add_clutter: Whether to add clutter reflections
            
        Returns:
            Dictionary with echo data and ground truth
        """
        n_sensors = len(sensor_positions)
        
        # Simulate echo for each sensor
        echo_data = []
        
        for sensor_pos in sensor_positions:
            # Forward path: transmitter (sensor) to target
            forward_signal = self.room_acoustics.simulate_propagation(
                sensor_pos, target_position, chirp_signal, self.sample_rate
            )
            
            # Backward path: target to receiver (same sensor)
            # Use impulse response and convolve with forward signal
            impulse_response = np.array([1.0])  # Simplified
            echo_signal = np.convolve(forward_signal, impulse_response, mode='same')
            
            # Add noise
            echo_with_noise = self._add_noise(echo_signal, snr_db)
            
            # Add clutter if requested
            if add_clutter:
                echo_with_noise = self._add_clutter(echo_with_noise, sensor_pos)
            
            echo_data.append(echo_with_noise)
        
        # Convert to numpy array
        max_length = max(len(echo) for echo in echo_data)
        echo_array = np.zeros((n_sensors, max_length))
        
        for i, echo in enumerate(echo_data):
            echo_array[i, :len(echo)] = echo
        
        return {
            'echo_data': echo_array,
            'position': np.array(target_position),
            'sensor_positions': np.array(sensor_positions),
            'snr_db': snr_db
        }
    
    def generate_dataset(
        self,
        n_samples: int,
        sensor_positions: List[Tuple[float, float, float]],
        position_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
        chirp_params: Optional[Dict[str, Any]] = None,
        snr_range: Tuple[float, float] = (10.0, 30.0)
    ) -> List[Dict[str, Any]]:
        """
        Generate training dataset.
        
        Args:
            n_samples: Number of samples to generate
            sensor_positions: Fixed sensor positions
            position_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            chirp_params: Chirp generation parameters
            snr_range: SNR range for noise injection
            
        Returns:
            List of training samples
        """
        if position_bounds is None:
            # Default to room interior
            w, h, d = self.room_acoustics.room_dimensions
            position_bounds = ((0.5, w-0.5), (0.2, h-0.2), (0.5, d-0.5))
        
        if chirp_params is None:
            chirp_params = {
                'duration': 0.005,  # 5ms
                'start_freq': 35000,
                'end_freq': 45000
            }
        
        # Generate chirp signal
        t = np.linspace(0, chirp_params['duration'], 
                       int(chirp_params['duration'] * self.sample_rate))
        chirp_signal = signal.chirp(
            t, chirp_params['start_freq'], chirp_params['duration'], 
            chirp_params['end_freq'], method='linear'
        )
        
        # Apply window
        window = signal.hann(len(chirp_signal))
        chirp_signal = chirp_signal * window
        
        dataset = []
        
        for i in range(n_samples):
            # Random target position
            target_pos = (
                random.uniform(*position_bounds[0]),
                random.uniform(*position_bounds[1]),
                random.uniform(*position_bounds[2])
            )
            
            # Random SNR
            snr = random.uniform(*snr_range)
            
            # Generate sample
            sample = self.generate_training_sample(
                target_pos, sensor_positions, chirp_signal, snr
            )
            
            dataset.append(sample)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")
        
        return dataset
    
    def _add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add realistic noise to signal."""
        # Calculate signal power
        signal_power = np.mean(signal**2)
        
        # Calculate noise power from SNR
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        
        # Add 1/f noise component (electronic noise)
        if len(signal) > 100:
            f_noise = self._generate_pink_noise(len(signal)) * np.sqrt(noise_power) * 0.3
            noise += f_noise
        
        return signal + noise
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink (1/f) noise."""
        # Simple approximation using filtered white noise
        white_noise = np.random.randn(length)
        
        # Apply 1/f filter in frequency domain
        freqs = np.fft.fftfreq(length)
        freqs[0] = 1e-10  # Avoid division by zero
        
        # 1/f amplitude response
        filter_response = 1.0 / np.sqrt(np.abs(freqs))
        filter_response[0] = filter_response[1]  # Fix DC component
        
        # Apply filter
        fft_noise = np.fft.fft(white_noise)
        filtered_fft = fft_noise * filter_response
        pink_noise = np.real(np.fft.ifft(filtered_fft))
        
        return pink_noise
    
    def _add_clutter(self, signal: np.ndarray, sensor_pos: Tuple[float, float, float]) -> np.ndarray:
        """Add clutter reflections from room features."""
        # Simple clutter model - add weak random reflections
        clutter_signal = signal.copy()
        
        # Add a few random clutter peaks
        for _ in range(random.randint(1, 3)):
            delay_samples = random.randint(100, len(signal) - 100)
            amplitude = random.uniform(0.05, 0.2) * np.max(np.abs(signal))
            
            # Add exponentially decaying clutter
            decay_length = random.randint(50, 200)
            decay_envelope = np.exp(-np.arange(decay_length) / (decay_length / 3))
            
            end_idx = min(delay_samples + decay_length, len(signal))
            clutter_length = end_idx - delay_samples
            
            clutter_signal[delay_samples:end_idx] += \
                amplitude * decay_envelope[:clutter_length] * \
                np.random.randn(clutter_length) * 0.5
        
        return clutter_signal