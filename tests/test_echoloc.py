"""
Tests for echoloc-nn: ChirpSignal, EchoSimulator, ChirpEncoder, TransformerLocator.
"""

import pytest
import numpy as np

from echoloc_nn.chirp import ChirpSignal
from echoloc_nn.simulator import EchoSimulator
from echoloc_nn.encoder import ChirpEncoder
from echoloc_nn.locator import TransformerLocator


# ──────────────────────────────────────────────────────────────────────────────
# ChirpSignal Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestChirpSignal:
    """Tests for the ChirpSignal class."""

    def setup_method(self):
        self.sr = 44100
        self.gen = ChirpSignal(window="hann")

    def test_generate_returns_correct_length(self):
        """Chirp should have exactly duration * sr samples."""
        duration = 0.01
        chirp = self.gen.generate(duration=duration, f0=1000, f1=10000, sr=self.sr)
        assert len(chirp) == int(duration * self.sr)

    def test_generate_returns_float64(self):
        """Output should be float64."""
        chirp = self.gen.generate(duration=0.01, f0=1000, f1=5000, sr=self.sr)
        assert chirp.dtype == np.float64

    def test_generate_is_1d(self):
        """Output should be 1-D."""
        chirp = self.gen.generate(duration=0.01, f0=1000, f1=5000, sr=self.sr)
        assert chirp.ndim == 1

    def test_generate_amplitude_bounded(self):
        """With hann window, amplitude should be in [-1, 1]."""
        chirp = self.gen.generate(duration=0.02, f0=500, f1=10000, sr=self.sr)
        assert np.all(chirp >= -1.0)
        assert np.all(chirp <= 1.0)

    def test_frequency_content_start(self):
        """FFT should show energy near f0 at the start of the chirp."""
        sr = 44100
        duration = 0.1
        f0, f1 = 1000.0, 5000.0
        gen = ChirpSignal(window=None)  # no window for clean FFT
        chirp = gen.generate(duration=duration, f0=f0, f1=f1, sr=sr)

        # Check first 10% of samples for f0 energy
        n_check = len(chirp) // 10
        segment = chirp[:n_check]
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
        fft_mag = np.abs(np.fft.rfft(segment))
        peak_freq = freqs[np.argmax(fft_mag)]

        # Peak should be near f0 (within ±500 Hz)
        assert abs(peak_freq - f0) < 500.0, (
            f"Expected peak near {f0} Hz, got {peak_freq:.1f} Hz"
        )

    def test_no_window_higher_amplitude(self):
        """Without windowing, max amplitude should be ~1.0."""
        gen_no_window = ChirpSignal(window=None)
        chirp = gen_no_window.generate(duration=0.01, f0=1000, f1=5000, sr=self.sr)
        assert np.max(np.abs(chirp)) > 0.9

    def test_invalid_duration_raises(self):
        with pytest.raises(ValueError):
            self.gen.generate(duration=-0.01, f0=1000, f1=5000, sr=self.sr)

    def test_invalid_frequency_raises(self):
        with pytest.raises(ValueError):
            self.gen.generate(duration=0.01, f0=-100, f1=5000, sr=self.sr)

    def test_max_range_formula(self):
        """Max range = speed_of_sound * duration / 2."""
        gen = ChirpSignal()
        duration = 0.01
        expected = 340.0 * duration / 2.0
        assert gen.max_range(duration) == pytest.approx(expected)

    def test_instantaneous_frequency_shape(self):
        """Instantaneous frequency should have same length as chirp."""
        duration = 0.01
        chirp = self.gen.generate(duration=duration, f0=1000, f1=5000, sr=self.sr)
        inst_freq = self.gen.instantaneous_frequency(
            duration=duration, f0=1000, f1=5000, sr=self.sr
        )
        assert len(inst_freq) == len(chirp)

    def test_instantaneous_frequency_bounds(self):
        """Instantaneous frequency should go from f0 to f1."""
        f0, f1 = 1000.0, 8000.0
        inst_freq = self.gen.instantaneous_frequency(
            duration=0.05, f0=f0, f1=f1, sr=self.sr
        )
        assert inst_freq[0] == pytest.approx(f0)
        assert inst_freq[-1] == pytest.approx(f1, rel=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# EchoSimulator Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEchoSimulator:
    """Tests for the EchoSimulator class."""

    def setup_method(self):
        self.sr = 44100
        self.sim = EchoSimulator(speed_of_sound=340.0, noise_std=0.0, sr=self.sr)
        gen = ChirpSignal(window=None)
        self.chirp = gen.generate(duration=0.005, f0=1000, f1=5000, sr=self.sr)

    def test_simulate_returns_longer_signal(self):
        """Output should be longer than input chirp due to delays."""
        reflections = [(5.0, 0.8, 0.0)]
        received = self.sim.simulate(self.chirp, reflections)
        assert len(received) > len(self.chirp)

    def test_echo_timing_correct(self):
        """Echo peak should appear at the expected delay sample."""
        distance = 5.0
        reflections = [(distance, 1.0, 0.0)]
        received = self.sim.simulate(self.chirp, reflections)

        expected_delay = self.sim.delay_samples(distance)
        # The peak of the received signal should be near the delay offset
        peak_idx = np.argmax(np.abs(received))
        chirp_peak = np.argmax(np.abs(self.chirp))
        expected_peak = expected_delay + chirp_peak

        # Allow ±5 samples tolerance
        assert abs(peak_idx - expected_peak) <= 5, (
            f"Echo peak at {peak_idx}, expected ~{expected_peak}"
        )

    def test_delay_formula(self):
        """delay_samples should equal round(2*distance/speed * sr)."""
        distance = 10.0
        expected = int(np.round(2.0 * distance / 340.0 * self.sr))
        assert self.sim.delay_samples(distance) == expected

    def test_delay_seconds_formula(self):
        """delay_seconds = 2 * distance / speed."""
        distance = 17.0
        expected = 2.0 * distance / 340.0
        assert self.sim.delay_seconds(distance) == pytest.approx(expected)

    def test_multiple_reflections(self):
        """Multiple reflections should each contribute to the output."""
        reflections = [
            (5.0,  0.8,  0.0),
            (15.0, 0.5, 20.0),
        ]
        received = self.sim.simulate(self.chirp, reflections)
        # Output should extend at least to the delay of the farther target
        min_len = self.sim.delay_samples(15.0) + len(self.chirp)
        assert len(received) >= min_len

    def test_invalid_distance_raises(self):
        with pytest.raises(ValueError):
            self.sim.simulate(self.chirp, [(-1.0, 0.5, 0.0)])

    def test_empty_reflections_raises(self):
        with pytest.raises(ValueError):
            self.sim.simulate(self.chirp, [])

    def test_noise_added_when_std_nonzero(self):
        """With noise_std > 0, output should differ from noiseless."""
        sim_noisy = EchoSimulator(noise_std=0.1, sr=self.sr, seed=1)
        sim_clean = EchoSimulator(noise_std=0.0, sr=self.sr)
        reflections = [(5.0, 0.5, 0.0)]
        noisy = sim_noisy.simulate(self.chirp, reflections)
        clean = sim_clean.simulate(self.chirp, reflections)
        assert not np.allclose(noisy, clean)


# ──────────────────────────────────────────────────────────────────────────────
# ChirpEncoder Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestChirpEncoder:
    """Tests for the ChirpEncoder class."""

    def setup_method(self):
        self.sr = 44100
        self.encoder = ChirpEncoder(n_filters=16, pool_size=8, seed=42)

        gen = ChirpSignal(window="hann")
        chirp = gen.generate(duration=0.01, f0=1000, f1=5000, sr=self.sr)
        sim = EchoSimulator(noise_std=0.0, sr=self.sr)
        self.signal = sim.simulate(chirp, [(5.0, 0.8, 0.0)])

    def test_output_shape_correct(self):
        """Feature vector should have shape (n_filters * pool_size,)."""
        features = self.encoder.encode(self.signal)
        assert features.shape == (16 * 8,)

    def test_output_is_float64(self):
        features = self.encoder.encode(self.signal)
        assert features.dtype == np.float64

    def test_output_is_normalized(self):
        """L2 norm of output should be ~1.0."""
        features = self.encoder.encode(self.signal)
        norm = np.linalg.norm(features)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_output_size_property(self):
        assert self.encoder.output_size == 16 * 8

    def test_different_signals_give_different_features(self):
        """Two different signals should produce different feature vectors."""
        gen = ChirpSignal(window="hann")
        chirp1 = gen.generate(duration=0.01, f0=1000, f1=5000, sr=self.sr)
        chirp2 = gen.generate(duration=0.01, f0=3000, f1=8000, sr=self.sr)

        sim = EchoSimulator(noise_std=0.0, sr=self.sr)
        sig1 = sim.simulate(chirp1, [(5.0, 0.8, 0.0)])
        sig2 = sim.simulate(chirp2, [(5.0, 0.8, 0.0)])

        f1 = self.encoder.encode(sig1)
        f2 = self.encoder.encode(sig2)
        assert not np.allclose(f1, f2)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            self.encoder.encode(np.zeros((10, 10)))


# ──────────────────────────────────────────────────────────────────────────────
# TransformerLocator Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTransformerLocator:
    """Tests for the TransformerLocator class."""

    def setup_method(self):
        self.sr = 44100
        self.encoder = ChirpEncoder(n_filters=16, pool_size=8, seed=42)
        self.locator = TransformerLocator(
            input_size=self.encoder.output_size,
            d_model=32,
            n_heads=4,
            seq_len=16,
            max_distance=50.0,
            seed=0,
        )

        gen = ChirpSignal(window="hann")
        chirp = gen.generate(duration=0.01, f0=1000, f1=5000, sr=self.sr)
        sim = EchoSimulator(noise_std=0.0, sr=self.sr)
        received = sim.simulate(chirp, [(10.0, 0.7, 15.0)])
        self.features = self.encoder.encode(received)

    def test_output_distance_in_range(self):
        """Predicted distance should be in [0, max_distance]."""
        dist, angle = self.locator.forward(self.features)
        assert 0.0 <= dist <= 50.0

    def test_output_angle_in_range(self):
        """Predicted angle should be in [-90, 90]."""
        dist, angle = self.locator.forward(self.features)
        assert -90.0 <= angle <= 90.0

    def test_output_is_float(self):
        """Outputs should be Python floats."""
        dist, angle = self.locator.forward(self.features)
        assert isinstance(dist, float)
        assert isinstance(angle, float)

    def test_deterministic(self):
        """Same input should give same output."""
        dist1, angle1 = self.locator.forward(self.features)
        dist2, angle2 = self.locator.forward(self.features)
        assert dist1 == dist2
        assert angle1 == angle2

    def test_different_features_different_predictions(self):
        """Different inputs should generally give different outputs."""
        gen = ChirpSignal(window="hann")
        chirp_a = gen.generate(duration=0.01, f0=1000, f1=5000, sr=self.sr)
        chirp_b = gen.generate(duration=0.01, f0=4000, f1=8000, sr=self.sr)
        sim = EchoSimulator(noise_std=0.0, sr=self.sr)

        fa = self.encoder.encode(sim.simulate(chirp_a, [(5.0, 0.8, 0.0)]))
        fb = self.encoder.encode(sim.simulate(chirp_b, [(20.0, 0.3, -45.0)]))

        da, aa = self.locator.forward(fa)
        db, ab = self.locator.forward(fb)

        # At least one prediction should differ
        assert not (da == pytest.approx(db) and aa == pytest.approx(ab))

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            self.locator.forward(np.zeros((8, 8)))


# ──────────────────────────────────────────────────────────────────────────────
# Integration Test
# ──────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    """Integration test: full pipeline from chirp to localization."""

    def test_full_pipeline(self):
        """Run the complete pipeline and verify all shapes and ranges."""
        sr = 44100

        # 1. Generate chirp
        gen = ChirpSignal(window="hann")
        chirp = gen.generate(duration=0.01, f0=1000, f1=10000, sr=sr)
        assert len(chirp) == int(0.01 * sr)

        # 2. Simulate echoes
        sim = EchoSimulator(noise_std=0.005, sr=sr, seed=99)
        targets = [(8.0, 0.9, 10.0), (25.0, 0.4, -30.0)]
        received = sim.simulate(chirp, targets)
        assert len(received) > len(chirp)

        # 3. Encode
        encoder = ChirpEncoder(n_filters=16, pool_size=8, seed=5)
        features = encoder.encode(received)
        assert features.shape == (128,)
        assert abs(np.linalg.norm(features) - 1.0) < 1e-6

        # 4. Localize
        locator = TransformerLocator(
            input_size=128, d_model=32, n_heads=4, seq_len=16, max_distance=50.0
        )
        dist, angle = locator.forward(features)

        assert 0.0 <= dist <= 50.0
        assert -90.0 <= angle <= 90.0

    def test_demo_runs(self):
        """Demo script should execute without errors."""
        from echoloc_nn.demo import run_demo
        result = run_demo(verbose=False)

        assert "chirp" in result
        assert "received" in result
        assert "features" in result
        assert "pred_distance" in result
        assert "pred_angle" in result
        assert 0.0 <= result["pred_distance"] <= 50.0
        assert -90.0 <= result["pred_angle"] <= 90.0
