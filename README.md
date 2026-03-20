# echoloc-nn

**Bat-inspired echolocation using neural networks — pure numpy/scipy.**

Simulates the biological sonar system used by bats: emit a chirp signal, receive echoes from targets, and predict their distance and angle. Implemented without PyTorch — just numpy and scipy.

---

## Architecture

```
ChirpSignal ──► EchoSimulator ──► ChirpEncoder ──► TransformerLocator
   (emit)         (receive)         (encode)          (localize)
```

### Components

| Module | Class | Description |
|--------|-------|-------------|
| `chirp.py` | `ChirpSignal` | Linear FM chirp generator (bat-style sweep) |
| `simulator.py` | `EchoSimulator` | Multi-target echo simulation with delays + attenuation |
| `encoder.py` | `ChirpEncoder` | 1D filter bank conv → fixed-size feature vector |
| `locator.py` | `TransformerLocator` | Self-attention network → distance + angle prediction |

---

## Quick Start

```python
from echoloc_nn import ChirpSignal, EchoSimulator, ChirpEncoder, TransformerLocator

# 1. Generate a chirp
gen = ChirpSignal(window="hann")
chirp = gen.generate(duration=0.01, f0=1000, f1=10000, sr=44100)

# 2. Simulate echoes from two targets
sim = EchoSimulator(noise_std=0.005, sr=44100)
reflections = [
    (5.0,  0.8,   0.0),   # 5 m ahead, amplitude 0.8, angle 0°
    (15.0, 0.4,  30.0),   # 15 m at +30°, amplitude 0.4
]
received = sim.simulate(chirp, reflections)

# 3. Encode received signal
encoder = ChirpEncoder(n_filters=16, pool_size=8)
features = encoder.encode(received)

# 4. Predict location
locator = TransformerLocator(input_size=encoder.output_size, max_distance=50.0)
distance, angle = locator.forward(features)

print(f"Predicted: {distance:.1f} m at {angle:+.1f}°")
```

---

## Run Demo

```bash
python -m echoloc_nn.demo
```

Output:
```
============================================================
  echoloc-nn: Bat-Inspired Echolocation Neural Network
============================================================

[1] Chirp generated: 441 samples @ 44100 Hz
    Sweep: 20000 Hz → 21950 Hz over 10.0 ms
    Max unambiguous range: 1.7 m

[2] Echo simulation: 3 targets
    dist=  5.0m  amp=0.8  angle=+0°  → delay=29.41ms (1297 samples)
    ...

[3] Encoding: ChirpEncoder(n_filters=16, pool_size=8, output_size=128)
    Feature vector: shape=(128,), norm=1.0000

[4] Localization: TransformerLocator(input_size=128, d_model=32, ...)
    Predicted distance: 23.47 m
    Predicted angle:    +12.34°
```

---

## Installation

```bash
pip install -r requirements.txt
```

No GPU needed. No PyTorch. Just numpy and scipy.

---

## Run Tests

```bash
python -m pytest tests/ -v
```

---

## Implementation Details

### ChirpSignal
Linear FM (LFM) sweep: instantaneous frequency `f(t) = f0 + (f1-f0)*t/T`.
Phase is the integral: `φ(t) = 2π * (f0*t + (f1-f0)*t²/(2T))`.
Optional Hann/Hamming/Blackman window to reduce spectral leakage.

### EchoSimulator
For each reflection `(distance, amplitude, angle)`:
- Round-trip delay: `τ = 2 * distance / speed_of_sound`
- Directional gain: `cos²(angle)` — max at 0°, zero at ±90°
- Adds `amplitude * cos²(angle) * chirp` at offset `τ * sr` samples

### ChirpEncoder
- **Filter bank**: `n_filters` sinusoidal filters at random frequencies
- **Convolution**: `numpy.correlate(signal, filter, mode='same')`
- **Activation**: ReLU
- **Pooling**: 1D max pooling → `pool_size` values per filter
- **Normalization**: L2 normalize the full feature vector

### TransformerLocator
- **Tokenization**: reshape feature vector into `seq_len` tokens
- **Projection**: linear layer to `d_model` dimensions
- **Self-attention**: manual Q/K/V with scaled dot-product
- **Feed-forward**: 2-layer MLP with ReLU, dimension `4 * d_model`
- **Output**: sigmoid → distance ∈ `[0, max_distance]`; tanh → angle ∈ `[-90, 90]`

---

## License

MIT
