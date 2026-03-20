"""
demo.py — End-to-end echolocation demonstration.
"""

import numpy as np
from .signals import ChirpSignal
from .simulator import EchoSimulator
from .encoder import ChirpEncoder
from .locator import TransformerLocator


def demo() -> None:
    """
    Demonstrate the full echolocation pipeline.

    Steps
    -----
    1. Define a set of known reflectors.
    2. Generate synthetic training data by randomising reflector positions.
    3. Train a TransformerLocator on the encoded echoes.
    4. Test on 3 new reflector scenarios and print predictions vs actuals.
    """
    print("=== Neural Echolocation Demo ===\n")

    rng = np.random.default_rng(0)
    encoder = ChirpEncoder(n_filters=16, filter_size=64)
    chirp_template = ChirpSignal(
        f_start=1000, f_end=10000, duration=0.01, sample_rate=44100
    )

    # ---- Generate synthetic training dataset ----
    N_train = 200
    X_train, y_dist_train, y_angle_train = [], [], []

    for _ in range(N_train):
        dist = rng.uniform(0.5, 9.0)
        angle = rng.uniform(-90.0, 90.0)
        refl = rng.uniform(0.5, 1.0)

        sim = EchoSimulator(sample_rate=44100)
        sim.add_reflector(dist, angle, refl)
        received = sim.simulate(chirp_template)
        feat = encoder.encode(received)

        X_train.append(feat)
        y_dist_train.append(dist)
        y_angle_train.append(angle)

    X_train = np.array(X_train)
    y_dist_train = np.array(y_dist_train)
    y_angle_train = np.array(y_angle_train)

    # ---- Train the locator ----
    locator = TransformerLocator(feature_dim=16, n_heads=4, hidden_dim=32)
    print(f"Training on {N_train} synthetic examples for 50 epochs…")
    locator.fit(X_train, y_dist_train, y_angle_train, epochs=50, lr=1e-2)
    print("Training complete.\n")

    # ---- Test on 3 known reflectors ----
    test_cases = [
        (2.0, 30.0, 1.0),
        (5.5, -45.0, 0.8),
        (8.0, 10.0, 0.6),
    ]

    print(f"{'Reflector':<12} {'True dist':>10} {'Pred dist':>10} "
          f"{'True angle':>11} {'Pred angle':>11}")
    print("-" * 58)

    for idx, (dist, angle, refl) in enumerate(test_cases, start=1):
        sim = EchoSimulator(sample_rate=44100)
        sim.add_reflector(dist, angle, refl)
        received = sim.simulate(chirp_template)
        feat = encoder.encode(received).reshape(1, -1)

        pred_dist, pred_angle = locator.predict(feat)
        print(
            f"  #{idx:<9} {dist:>10.2f} {pred_dist[0]:>10.2f} "
            f"{angle:>11.1f} {pred_angle[0]:>11.1f}"
        )

    print("\nDemo finished.")


if __name__ == "__main__":
    demo()
