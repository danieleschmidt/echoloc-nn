"""
End-to-end demo: Echo simulation → Encoding → Localization.

Demonstrates the full echoloc-nn pipeline:
1. Generate a chirp signal (bat-style LFM sweep)
2. Simulate echoes from multiple reflective targets
3. Encode the received signal into feature vectors
4. Use TransformerLocator to predict distance and angle

Run:
    python -m echoloc_nn.demo
"""

import numpy as np
from .chirp import ChirpSignal
from .simulator import EchoSimulator
from .encoder import ChirpEncoder
from .locator import TransformerLocator


def run_demo(verbose: bool = True) -> dict:
    """
    Run the full echoloc-nn pipeline.

    Returns
    -------
    dict
        Results containing chirp, received signal, features, and predictions.
    """
    # ──────────────────────────────────────────────
    # 1. Generate chirp signal
    # ──────────────────────────────────────────────
    sr = 44100
    duration = 0.01  # 10 ms chirp (bat-like)
    f0 = 20_000.0   # 20 kHz start
    f1 = 80_000.0   # 80 kHz end (but will be capped to Nyquist)

    # Stay below Nyquist (sr/2 = 22050 Hz for 44100)
    f1 = min(f1, sr / 2 - 100)

    chirp_gen = ChirpSignal(window="hann")
    chirp = chirp_gen.generate(duration=duration, f0=f0, f1=f1, sr=sr)

    if verbose:
        print(f"[1] Chirp generated: {len(chirp)} samples @ {sr} Hz")
        print(f"    Sweep: {f0:.0f} Hz → {f1:.0f} Hz over {duration*1000:.1f} ms")
        print(f"    Max unambiguous range: {chirp_gen.max_range(duration):.1f} m")
        print()

    # ──────────────────────────────────────────────
    # 2. Define targets and simulate echoes
    # ──────────────────────────────────────────────
    targets = [
        # (distance_m, amplitude, angle_deg)
        (5.0,  0.8,   0.0),   # Strong target directly ahead, 5 m
        (12.5, 0.5,  30.0),   # Mid-range target at 30° right, 12.5 m
        (20.0, 0.3, -20.0),   # Weak target at 20° left, 20 m
    ]

    simulator = EchoSimulator(speed_of_sound=340.0, noise_std=0.01, sr=sr, seed=42)
    received = simulator.simulate(chirp, targets)

    if verbose:
        print(f"[2] Echo simulation: {len(targets)} targets")
        for dist, amp, angle in targets:
            delay_s = simulator.delay_seconds(dist)
            delay_n = simulator.delay_samples(dist)
            print(f"    dist={dist:5.1f}m  amp={amp:.1f}  angle={angle:+.0f}°"
                  f"  → delay={delay_s*1000:.2f}ms ({delay_n} samples)")
        print(f"    Received signal: {len(received)} samples")
        print()

    # ──────────────────────────────────────────────
    # 3. Encode the received signal
    # ──────────────────────────────────────────────
    encoder = ChirpEncoder(n_filters=16, pool_size=8, seed=42)
    features = encoder.encode(received)

    if verbose:
        print(f"[3] Encoding: {encoder}")
        print(f"    Feature vector: shape={features.shape}, "
              f"norm={np.linalg.norm(features):.4f}")
        print(f"    Feature stats: min={features.min():.4f}, "
              f"max={features.max():.4f}, mean={features.mean():.4f}")
        print()

    # ──────────────────────────────────────────────
    # 4. Predict distance and angle
    # ──────────────────────────────────────────────
    locator = TransformerLocator(
        input_size=encoder.output_size,
        d_model=32,
        n_heads=4,
        seq_len=16,
        max_distance=50.0,
        seed=7,
    )
    pred_distance, pred_angle = locator.forward(features)

    if verbose:
        print(f"[4] Localization: {locator}")
        print(f"    Predicted distance: {pred_distance:.2f} m")
        print(f"    Predicted angle:    {pred_angle:+.2f}°")
        print()
        print("    (Note: predictions are from a randomly initialized network;")
        print("     training on labelled data would refine these estimates.)")

    return {
        "chirp": chirp,
        "received": received,
        "features": features,
        "pred_distance": pred_distance,
        "pred_angle": pred_angle,
        "targets": targets,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  echoloc-nn: Bat-Inspired Echolocation Neural Network")
    print("=" * 60)
    print()
    result = run_demo(verbose=True)
    print("Demo complete.")
