"""
Secure Random Number Generation for EchoLoc-NN

Provides cryptographically secure random number generation for quantum
algorithms and security-sensitive operations.
"""

import secrets
import hashlib
import hmac
import os
from typing import List, Union, Optional
import numpy as np

class SecureRandom:
    """
    Cryptographically secure random number generator.
    
    This class provides secure alternatives to numpy.random for use in
    quantum algorithms and security-sensitive contexts.
    """
    
    def __init__(self, seed: Optional[bytes] = None):
        """Initialize secure random generator with optional seed."""
        self._entropy_pool = secrets.token_bytes(32)
        if seed:
            self._entropy_pool = hmac.new(self._entropy_pool, seed, hashlib.sha256).digest()
    
    def random(self) -> float:
        """Generate a secure random float in [0, 1)."""
        random_bytes = secrets.token_bytes(8)
        return int.from_bytes(random_bytes, 'big') / (2**64)
    
    def randint(self, low: int, high: int) -> int:
        """Generate a secure random integer in [low, high)."""
        if low >= high:
            raise ValueError("low must be less than high")
        return secrets.randbelow(high - low) + low
    
    def choice(self, sequence: List) -> any:
        """Securely choose a random element from a sequence."""
        if not sequence:
            raise ValueError("sequence must not be empty")
        return sequence[secrets.randbelow(len(sequence))]
    
    def uniform(self, low: float, high: float) -> float:
        """Generate a secure random float in [low, high)."""
        return low + (high - low) * self.random()
    
    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Generate a secure random number from normal distribution."""
        # Box-Muller transform using secure random
        u1 = self.random()
        u2 = self.random()
        
        # Ensure u1 is not zero to avoid log(0)
        while u1 == 0:
            u1 = self.random()
        
        z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return mean + std * z0
    
    def exponential(self, rate: float = 1.0) -> float:
        """Generate a secure random number from exponential distribution."""
        u = self.random()
        while u == 0:  # Avoid log(0)
            u = self.random()
        return -np.log(u) / rate
    
    def quantum_phase(self) -> float:
        """Generate a secure random quantum phase in [0, 2π)."""
        return self.uniform(0, 2 * np.pi)
    
    def quantum_state_vector(self, dimension: int) -> np.ndarray:
        """Generate a secure random quantum state vector."""
        # Generate random complex amplitudes
        real_parts = np.array([self.normal() for _ in range(dimension)])
        imag_parts = np.array([self.normal() for _ in range(dimension)])
        
        # Combine into complex vector
        state_vector = real_parts + 1j * imag_parts
        
        # Normalize to unit vector
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
        
        return state_vector
    
    def token_hex(self, nbytes: int = 16) -> str:
        """Generate a secure random token as hex string."""
        return secrets.token_hex(nbytes)
    
    def token_urlsafe(self, nbytes: int = 16) -> str:
        """Generate a secure random URL-safe token."""
        return secrets.token_urlsafe(nbytes)


# Global secure random instance
_secure_random = SecureRandom()

# Secure alternatives to numpy.random functions
def secure_random() -> float:
    """Secure alternative to numpy.random.random()."""
    return _secure_random.random()

def secure_randint(low: int, high: int) -> int:
    """Secure alternative to numpy.random.randint()."""
    return _secure_random.randint(low, high)

def secure_choice(sequence: List) -> any:
    """Secure alternative to numpy.random.choice()."""
    return _secure_random.choice(sequence)

def secure_uniform(low: float, high: float) -> float:
    """Secure alternative to numpy.random.uniform()."""
    return _secure_random.uniform(low, high)

def secure_normal(mean: float = 0.0, std: float = 1.0) -> float:
    """Secure alternative to numpy.random.normal()."""
    return _secure_random.normal(mean, std)

def secure_quantum_phase() -> float:
    """Generate secure random quantum phase."""
    return _secure_random.quantum_phase()

def secure_quantum_state(dimension: int) -> np.ndarray:
    """Generate secure random quantum state vector."""
    return _secure_random.quantum_state_vector(dimension)


class QuantumSecureRandom:
    """
    Quantum-aware secure random number generator.
    
    Provides secure random number generation with quantum-specific features
    for quantum algorithms and quantum state manipulation.
    """
    
    def __init__(self):
        self.secure = SecureRandom()
        
    def quantum_tunneling_probability(self, energy_barrier: float, temperature: float) -> float:
        """Calculate secure quantum tunneling probability."""
        if temperature <= 0:
            return 0.0
        
        # Use secure random for quantum effects
        base_prob = np.exp(-energy_barrier / (temperature + 1e-10))
        quantum_noise = self.secure.normal(0, 0.01)  # Small quantum fluctuation
        
        return max(0.0, min(1.0, base_prob + quantum_noise))
    
    def quantum_measurement(self, state_probabilities: np.ndarray) -> int:
        """Perform secure quantum measurement."""
        # Normalize probabilities
        probabilities = np.abs(state_probabilities) ** 2
        probabilities = probabilities / np.sum(probabilities)
        
        # Secure cumulative selection
        cumulative = np.cumsum(probabilities)
        random_value = self.secure.random()
        
        for i, cum_prob in enumerate(cumulative):
            if random_value <= cum_prob:
                return i
        
        return len(probabilities) - 1  # Fallback to last state
    
    def quantum_interference_factor(self) -> complex:
        """Generate secure quantum interference factor."""
        magnitude = self.secure.uniform(0.5, 1.0)
        phase = self.secure.quantum_phase()
        return magnitude * np.exp(1j * phase)
    
    def quantum_decoherence_noise(self, coherence_time: float, current_time: float) -> float:
        """Generate secure decoherence noise."""
        if coherence_time <= 0:
            return 1.0  # Complete decoherence
        
        # Exponential decay with secure random fluctuations
        base_decoherence = np.exp(-current_time / coherence_time)
        noise = self.secure.normal(0, 0.05)  # Small fluctuation
        
        return max(0.0, min(1.0, base_decoherence + noise))


# Global quantum secure random instance
_quantum_secure_random = QuantumSecureRandom()

def secure_quantum_tunneling(energy_barrier: float, temperature: float) -> float:
    """Get secure quantum tunneling probability."""
    return _quantum_secure_random.quantum_tunneling_probability(energy_barrier, temperature)

def secure_quantum_measurement(state_probabilities: np.ndarray) -> int:
    """Perform secure quantum measurement."""
    return _quantum_secure_random.quantum_measurement(state_probabilities)

def secure_quantum_interference() -> complex:
    """Generate secure quantum interference factor."""
    return _quantum_secure_random.quantum_interference_factor()

def secure_quantum_decoherence(coherence_time: float, current_time: float) -> float:
    """Generate secure decoherence noise."""
    return _quantum_secure_random.quantum_decoherence_noise(coherence_time, current_time)