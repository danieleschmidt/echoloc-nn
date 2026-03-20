# echoloc-nn

**Neural echolocation in pure Python / NumPy.**

A minimal, dependency-light library that simulates bat-style echolocation and
learns to localise reflectors from frequency-swept chirp signals — no deep-learning
framework required.

---

## Overview

Bats navigate and hunt by emitting ultrasonic chirps and interpreting the
returning echoes. This library recreates that pipeline in software:

| Step | Module | What it does |
|------|--------|--------------|
| 1. Transmit | `ChirpSignal` | Generates a linear frequency-swept pulse with Gaussian envelope |
| 2. Reflect | `EchoSimulator` | Simulates delayed, attenuated echoes from point reflectors |
| 3. Encode | `ChirpEncoder` | Extracts features via a cosine-modulated Gaussian filter bank |
| 4. Localise | `TransformerLocator` | Predicts distance & angle using self-attention + linear regression |

---

## Quick Start

```python
from echoloc import ChirpSignal, EchoSimulator, ChirpEncoder, TransformerLocator

# 1. Create a chirp
chirp = ChirpSignal(f_start=1000, f_end=10000, duration=0.01, sample_rate=44100)
signal = chirp.generate()          # numpy array, values in [-1, 1]

# 2. Simulate echoes from a reflector 3 m away at 20°
sim = EchoSimulator()
sim.add_reflector(distance=3.0, angle_deg=20.0, reflectivity=0.8)
received = sim.simulate(chirp)

# 3. Encode to a feature vector
enc = ChirpEncoder(n_filters=16, filter_size=64)
features = enc.encode(received)    # shape (16,)

# 4. Localise (after training)
loc = TransformerLocator(feature_dim=16, n_heads=4, hidden_dim=32)
# loc.fit(X_train, y_distances, y_angles, epochs=50)
dist_pred, angle_pred = loc.forward(features)
```

Run the built-in demo:

```python
from echoloc.demo import demo
demo()
```

---

## Installation

```bash
pip install -e .
```

Requires: Python ≥ 3.9, NumPy.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Architecture

```
ChirpSignal          →  s(t) = A(t)·sin(2π(f₀t + kt²))
                                A(t) = Gaussian envelope
                                k    = (f_end − f_start) / (2·duration)

EchoSimulator        →  received(t) = Σ_r (ρ_r / d_r²) · s(t − 2d_r/c)
                                d_r  = reflector distance
                                ρ_r  = reflectivity, c = speed of sound

ChirpEncoder         →  f_k = max|conv(signal, g_k)|
                                g_k = Gaussian·cos(2π f_k t)

TransformerLocator   →  H = tanh(Attention(X) · W_o + b_o)
                          d̂ = H · w_dist,  â = H · w_angle
```

---

## License

MIT
