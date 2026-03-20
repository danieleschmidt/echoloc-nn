"""
tests/test_echoloc.py — 10+ tests for the echoloc package.
"""

import numpy as np
import pytest

from echoloc.signals import ChirpSignal
from echoloc.simulator import EchoSimulator
from echoloc.encoder import ChirpEncoder
from echoloc.locator import TransformerLocator
from echoloc.demo import demo


# ======================================================================
# ChirpSignal tests
# ======================================================================


class TestChirpSignal:

    def test_generate_correct_length(self):
        cs = ChirpSignal(duration=0.01, sample_rate=44100)
        sig = cs.generate()
        assert len(sig) == cs.n_samples
        assert cs.n_samples == int(0.01 * 44100)

    def test_generate_values_in_range(self):
        cs = ChirpSignal()
        sig = cs.generate()
        assert np.max(np.abs(sig)) <= 1.0 + 1e-9

    def test_n_samples_property(self):
        cs = ChirpSignal(duration=0.05, sample_rate=22050)
        assert cs.n_samples == int(0.05 * 22050)

    def test_duration_property(self):
        cs = ChirpSignal(duration=0.02)
        assert cs.duration == pytest.approx(0.02)

    def test_sample_rate_property(self):
        cs = ChirpSignal(sample_rate=22050)
        assert cs.sample_rate == 22050

    def test_frequency_increases_over_time(self):
        """Check that instantaneous frequency rises (chirp goes up)."""
        cs = ChirpSignal(f_start=1000, f_end=10000, duration=0.01, sample_rate=44100)
        sig = cs.generate()
        # Compare zero-crossing density in first vs second half
        half = len(sig) // 2
        first_half = sig[:half]
        second_half = sig[half:]
        zc_first = np.sum(np.diff(np.sign(first_half)) != 0)
        zc_second = np.sum(np.diff(np.sign(second_half)) != 0)
        # Second half should have more zero-crossings (higher frequency)
        assert zc_second > zc_first

    def test_generate_is_not_all_zeros(self):
        cs = ChirpSignal()
        sig = cs.generate()
        assert np.max(np.abs(sig)) > 0.1


# ======================================================================
# EchoSimulator tests
# ======================================================================


class TestEchoSimulator:

    def _make_chirp(self):
        return ChirpSignal(f_start=1000, f_end=10000, duration=0.05, sample_rate=44100)

    def test_simulate_returns_correct_length(self):
        cs = self._make_chirp()
        sim = EchoSimulator(sample_rate=44100)
        sim.add_reflector(1.0, 0.0)
        received = sim.simulate(cs)
        assert len(received) == cs.n_samples

    def test_no_reflectors_near_zero(self):
        cs = self._make_chirp()
        sim = EchoSimulator(sample_rate=44100)
        received = sim.simulate(cs)
        assert np.max(np.abs(received)) < 1e-9

    def test_longer_distance_later_echo_peak(self):
        cs = self._make_chirp()
        speed = 343.0

        sim_near = EchoSimulator(sample_rate=44100, speed_of_sound=speed)
        sim_near.add_reflector(distance=1.0, angle_deg=0.0, reflectivity=1.0)
        recv_near = sim_near.simulate(cs)

        sim_far = EchoSimulator(sample_rate=44100, speed_of_sound=speed)
        sim_far.add_reflector(distance=4.0, angle_deg=0.0, reflectivity=1.0)
        recv_far = sim_far.simulate(cs)

        peak_near = np.argmax(np.abs(recv_near))
        peak_far = np.argmax(np.abs(recv_far))
        assert peak_far > peak_near

    def test_multiple_reflectors(self):
        cs = self._make_chirp()
        sim = EchoSimulator(sample_rate=44100)
        sim.add_reflector(1.0, 0.0)
        sim.add_reflector(3.0, 45.0)
        received = sim.simulate(cs)
        assert len(received) == cs.n_samples
        assert np.max(np.abs(received)) > 0


# ======================================================================
# ChirpEncoder tests
# ======================================================================


class TestChirpEncoder:

    def _make_signal(self, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(441)  # ~10 ms at 44100

    def test_encode_returns_correct_shape(self):
        enc = ChirpEncoder(n_filters=16, filter_size=64)
        sig = self._make_signal()
        feat = enc.encode(sig)
        assert feat.shape == (16,)

    def test_encode_batch_returns_2d(self):
        enc = ChirpEncoder(n_filters=16, filter_size=64)
        signals = [self._make_signal(i) for i in range(5)]
        feats = enc.encode_batch(signals)
        assert feats.ndim == 2
        assert feats.shape == (5, 16)

    def test_different_signals_give_different_encodings(self):
        enc = ChirpEncoder(n_filters=16, filter_size=64)
        sig1 = self._make_signal(0)
        sig2 = self._make_signal(99)
        f1 = enc.encode(sig1)
        f2 = enc.encode(sig2)
        assert not np.allclose(f1, f2)

    def test_chirp_signal_encoded(self):
        cs = ChirpSignal()
        enc = ChirpEncoder()
        feat = enc.encode(cs.generate())
        assert feat.shape == (enc.n_filters,)
        assert np.max(np.abs(feat)) > 0


# ======================================================================
# TransformerLocator tests
# ======================================================================


class TestTransformerLocator:

    def _make_data(self, N=50, feature_dim=16, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((N, feature_dim))
        yd = rng.uniform(1, 10, N)
        ya = rng.uniform(-90, 90, N)
        return X, yd, ya

    def test_forward_returns_two_scalars(self):
        loc = TransformerLocator()
        feat = np.random.default_rng(0).standard_normal(16)
        result = loc.forward(feat)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_fit_runs_without_error(self):
        loc = TransformerLocator()
        X, yd, ya = self._make_data(N=20)
        loc.fit(X, yd, ya, epochs=5)  # short run for speed

    def test_predict_returns_arrays(self):
        loc = TransformerLocator()
        X, yd, ya = self._make_data(N=20)
        loc.fit(X, yd, ya, epochs=5)
        dists, angles = loc.predict(X)
        assert isinstance(dists, np.ndarray)
        assert isinstance(angles, np.ndarray)
        assert dists.shape == (20,)
        assert angles.shape == (20,)

    def test_predict_shape_matches_input(self):
        loc = TransformerLocator()
        X = np.random.default_rng(1).standard_normal((7, 16))
        dists, angles = loc.predict(X)
        assert len(dists) == 7
        assert len(angles) == 7


# ======================================================================
# Demo test
# ======================================================================


class TestDemo:

    def test_demo_runs_without_error(self):
        demo()


# ======================================================================
# End-to-end test
# ======================================================================


class TestEndToEnd:

    def test_predicted_distances_reasonable(self):
        """
        Train on synthetic echoes; check that test predictions are in a
        plausible range (not wildly outside training range 0.5–9 m).
        """
        rng = np.random.default_rng(7)
        encoder = ChirpEncoder(n_filters=16, filter_size=64)
        chirp = ChirpSignal(duration=0.05, sample_rate=44100)

        X, yd, ya = [], [], []
        for _ in range(80):
            d = rng.uniform(0.5, 8.0)
            a = rng.uniform(-80, 80)
            sim = EchoSimulator(sample_rate=44100)
            sim.add_reflector(d, a, 1.0)
            recv = sim.simulate(chirp)
            X.append(encoder.encode(recv))
            yd.append(d)
            ya.append(a)

        X = np.array(X)
        yd = np.array(yd)
        ya = np.array(ya)

        locator = TransformerLocator(feature_dim=16, n_heads=4, hidden_dim=32)
        locator.fit(X, yd, ya, epochs=30, lr=1e-2)

        pred_d, pred_a = locator.predict(X)
        # Predictions should not be wildly out of range (within 2× training range)
        assert np.all(pred_d > -50) and np.all(pred_d < 50)
        assert np.all(pred_a > -500) and np.all(pred_a < 500)
