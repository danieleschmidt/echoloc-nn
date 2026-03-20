"""
TransformerLocator: Self-attention based localization network.

Takes an encoded feature vector and predicts the distance (m) and
azimuth angle (degrees) of the nearest/strongest echo target.

Implemented entirely in numpy — no pytorch/tensorflow required.
"""

import numpy as np
from typing import Tuple


class TransformerLocator:
    """
    Predict target distance and angle from an encoded feature vector.

    Architecture:
    1. Linear projection to d_model dimensions (sequence of tokens)
    2. Multi-head self-attention (manual Q/K/V matrices)
    3. Feed-forward layer (ReLU activation)
    4. Output head: linear projection → [distance, angle]

    The feature vector is reshaped into a sequence of tokens for
    self-attention processing.

    Parameters
    ----------
    input_size : int
        Size of the input feature vector (from ChirpEncoder).
    d_model : int
        Internal model dimension. Default: 32.
    n_heads : int
        Number of attention heads. Default: 4.
    seq_len : int
        Number of tokens to split features into. Default: 8.
    max_distance : float
        Maximum predicted distance in meters. Default: 50.0.
    seed : int
        Random seed for weight initialization. Default: 0.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        seq_len: int = 8,
        max_distance: float = 50.0,
        seed: int = 0,
    ):
        self.input_size = input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.max_distance = max_distance

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads

        rng = np.random.default_rng(seed)

        def glorot(shape):
            fan_in, fan_out = shape[-2], shape[-1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, shape)

        # Token projection: (seq_len, token_dim) → (seq_len, d_model)
        self.token_dim = input_size // seq_len
        self.W_proj = glorot((self.token_dim, d_model))
        self.b_proj = np.zeros(d_model)

        # Multi-head attention weights per head: Q, K, V
        self.W_Q = [glorot((d_model, self.d_head)) for _ in range(n_heads)]
        self.W_K = [glorot((d_model, self.d_head)) for _ in range(n_heads)]
        self.W_V = [glorot((d_model, self.d_head)) for _ in range(n_heads)]
        self.W_O = glorot((d_model, d_model))  # output projection

        # Feed-forward network
        ff_dim = d_model * 4
        self.W_ff1 = glorot((d_model, ff_dim))
        self.b_ff1 = np.zeros(ff_dim)
        self.W_ff2 = glorot((ff_dim, d_model))
        self.b_ff2 = np.zeros(d_model)

        # Output head: mean-pooled d_model → 2 (distance, angle)
        self.W_out = glorot((d_model, 2))
        self.b_out = np.zeros(2)

    def forward(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Predict distance and angle from an encoded feature vector.

        Parameters
        ----------
        features : np.ndarray
            Feature vector of shape (input_size,).

        Returns
        -------
        (distance, angle) : tuple of float
            distance in meters in [0, max_distance]
            angle in degrees in [-90, 90]
        """
        if features.ndim != 1:
            raise ValueError(f"features must be 1-D, got shape {features.shape}")
        if len(features) < self.seq_len:
            raise ValueError(
                f"features length {len(features)} < seq_len {self.seq_len}"
            )

        # Reshape feature vector into tokens
        usable = self.seq_len * self.token_dim
        tokens = features[:usable].reshape(self.seq_len, self.token_dim)

        # Linear projection → (seq_len, d_model)
        x = tokens @ self.W_proj + self.b_proj  # (seq_len, d_model)

        # Layer norm
        x = self._layer_norm(x)

        # Multi-head self-attention
        attn_out = self._multi_head_attention(x)  # (seq_len, d_model)
        x = self._layer_norm(x + attn_out)  # residual + layer norm

        # Feed-forward
        ff_out = self._feed_forward(x)  # (seq_len, d_model)
        x = self._layer_norm(x + ff_out)  # residual + layer norm

        # Global average pooling over sequence
        pooled = x.mean(axis=0)  # (d_model,)

        # Output head
        raw = pooled @ self.W_out + self.b_out  # (2,)

        # Map to physically meaningful ranges
        distance = self._sigmoid(raw[0]) * self.max_distance  # [0, max_distance]
        angle = np.tanh(raw[1]) * 90.0  # [-90, 90]

        return float(distance), float(angle)

    def _multi_head_attention(self, x: np.ndarray) -> np.ndarray:
        """
        Multi-head self-attention.

        Parameters
        ----------
        x : np.ndarray
            Input of shape (seq_len, d_model).

        Returns
        -------
        np.ndarray
            Attention output of shape (seq_len, d_model).
        """
        head_outputs = []

        for h in range(self.n_heads):
            Q = x @ self.W_Q[h]  # (seq_len, d_head)
            K = x @ self.W_K[h]
            V = x @ self.W_V[h]

            # Scaled dot-product attention
            scale = np.sqrt(self.d_head)
            scores = Q @ K.T / scale  # (seq_len, seq_len)
            weights = self._softmax(scores)  # (seq_len, seq_len)
            head_out = weights @ V  # (seq_len, d_head)
            head_outputs.append(head_out)

        # Concatenate heads: (seq_len, d_model)
        concat = np.concatenate(head_outputs, axis=-1)

        # Output projection
        return concat @ self.W_O

    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Position-wise feed-forward network.

        Parameters
        ----------
        x : np.ndarray
            Input of shape (seq_len, d_model).

        Returns
        -------
        np.ndarray
            Output of shape (seq_len, d_model).
        """
        h = x @ self.W_ff1 + self.b_ff1  # (seq_len, ff_dim)
        h = np.maximum(h, 0.0)  # ReLU
        return h @ self.W_ff2 + self.b_ff2  # (seq_len, d_model)

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer normalization over last dimension."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over last dimension."""
        x_max = x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-float(x)))

    def __repr__(self) -> str:
        return (
            f"TransformerLocator(input_size={self.input_size}, "
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"seq_len={self.seq_len}, max_distance={self.max_distance})"
        )
