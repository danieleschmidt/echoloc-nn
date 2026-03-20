"""
locator.py — TransformerLocator: self-attention-based distance/angle estimator.

Architecture
------------
This is an Extreme Learning Machine (ELM) with a self-attention feature transform:

  1. Random fixed projections W_q, W_k, W_v perform self-attention over the input.
  2. A fixed random hidden layer W_o maps the attended representation to
     hidden_dim activations via tanh.
  3. Output heads (distance, angle) are trained analytically with
     regularised least squares (closed-form, fast, stable).

The iterative `epochs` argument drives an optional feature normalisation
warmup; after the first epoch the solution is exact.
"""

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


class TransformerLocator:
    """
    Self-attention locator implemented in pure NumPy (ELM style).

    Parameters
    ----------
    feature_dim : int
        Length of the input feature vector (default 16).
    n_heads : int
        Number of attention heads (default 4).
    hidden_dim : int
        Size of the random hidden layer (default 32).
    """

    def __init__(
        self,
        feature_dim: int = 16,
        n_heads: int = 4,
        hidden_dim: int = 32,
    ):
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self._rng = np.random.default_rng(42)
        self._init_weights()
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Weight initialisation                                                #
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        d = self.feature_dim
        h = self.hidden_dim
        scale = 0.5

        # Fixed random attention projections
        self.W_q = self._rng.standard_normal((d, d)) * scale
        self.W_k = self._rng.standard_normal((d, d)) * scale
        self.W_v = self._rng.standard_normal((d, d)) * scale

        # Fixed random hidden layer
        self.W_o = self._rng.standard_normal((d, h)) * scale
        self.b_o = self._rng.standard_normal(h) * scale

        # Trained output heads (initialised to zero)
        self.w_dist = np.zeros(h + 1)   # includes bias
        self.w_angle = np.zeros(h + 1)

        # Feature normalisation statistics (set during fit)
        self._feat_mean = None
        self._feat_std = None

    # ------------------------------------------------------------------ #
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #

    def _attention(self, x: np.ndarray) -> np.ndarray:
        """Self-attention over the feature vector as a sequence of tokens."""
        d = self.feature_dim
        head_dim = max(d // self.n_heads, 1)

        Q = (x @ self.W_q).reshape(self.n_heads, head_dim)
        K = (x @ self.W_k).reshape(self.n_heads, head_dim)
        V = (x @ self.W_v).reshape(self.n_heads, head_dim)

        scale = np.sqrt(head_dim + 1e-9)
        scores = (Q @ K.T) / scale
        attn = _softmax(scores)
        out = (attn @ V).reshape(d)
        return out

    def _feature_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Full feature transform: normalise → attention → tanh hidden.

        Returns hidden representation of shape (hidden_dim,).
        """
        # Normalise
        if self._feat_mean is not None:
            x = (x - self._feat_mean) / (self._feat_std + 1e-9)

        attended = self._attention(x)
        hidden = np.tanh(attended @ self.W_o + self.b_o)
        return hidden

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the design matrix H_aug = [H | 1] for all samples."""
        H = np.vstack([self._feature_transform(X[i]) for i in range(len(X))])
        return np.hstack([H, np.ones((len(H), 1))])

    def forward(self, features: np.ndarray):
        """
        Run a single forward pass.

        Parameters
        ----------
        features : np.ndarray of shape (feature_dim,)

        Returns
        -------
        (distance_pred, angle_pred) : tuple of two floats
        """
        features = np.asarray(features, dtype=float).ravel()
        # Pad or truncate to feature_dim
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        else:
            features = features[: self.feature_dim]

        h = self._feature_transform(features)
        h_aug = np.append(h, 1.0)

        dist = float(h_aug @ self.w_dist)
        angle = float(h_aug @ self.w_angle)
        return dist, angle

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_features: np.ndarray,
        y_distances: np.ndarray,
        y_angles: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-2,
        ridge: float = 1e-3,
    ) -> None:
        """
        Train the output heads using regularised least squares (ELM).

        The `epochs` parameter is accepted for API compatibility but the
        ELM solution is computed in a single closed-form step.  We
        repeat it `min(epochs, 3)` times to allow feature normalisation
        to stabilise.

        Parameters
        ----------
        X_features : np.ndarray of shape (N, feature_dim)
        y_distances : np.ndarray of shape (N,)
        y_angles    : np.ndarray of shape (N,)
        epochs : int  (accepted, used minimally for normalisation warmup)
        lr : float    (accepted for API compatibility)
        ridge : float
            L2 regularisation strength (default 1e-3).
        """
        X = np.asarray(X_features, dtype=float)
        yd = np.asarray(y_distances, dtype=float).ravel()
        ya = np.asarray(y_angles, dtype=float).ravel()

        # Compute feature normalisation statistics from raw inputs
        self._feat_mean = X.mean(axis=0)
        self._feat_std = X.std(axis=0)

        # Solve with least squares (2 warmup passes for stability)
        n_passes = min(max(epochs, 1), 3)
        for _ in range(n_passes):
            H_aug = self._design_matrix(X)
            # Regularised normal equations: (H^T H + λI) w = H^T y
            HtH = H_aug.T @ H_aug
            reg = ridge * np.eye(HtH.shape[0])
            Hty_d = H_aug.T @ yd
            Hty_a = H_aug.T @ ya

            self.w_dist = np.linalg.solve(HtH + reg, Hty_d)
            self.w_angle = np.linalg.solve(HtH + reg, Hty_a)

        self._fitted = True

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, X_features: np.ndarray):
        """
        Predict distances and angles for a batch of feature vectors.

        Parameters
        ----------
        X_features : np.ndarray of shape (N, feature_dim)

        Returns
        -------
        (distances, angles) : tuple of np.ndarray of shape (N,)
        """
        X = np.asarray(X_features, dtype=float)
        distances = []
        angles = []
        for x in X:
            d, a = self.forward(x)
            distances.append(d)
            angles.append(a)
        return np.array(distances), np.array(angles)
