"""Tests for the MLP model: forward pass, persistence and trainability."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mnist import nn
from mnist.model import MLP


def separable_dataset(n_per_class: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A tiny, linearly separable stand-in for MNIST (10 classes, 784 features)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(10):
        centre = np.zeros(784, dtype=np.float32)
        centre[c * 70 : (c + 1) * 70] = 1.0
        X.append(centre + 0.01 * rng.standard_normal((n_per_class, 784)).astype(np.float32))
        y.append(np.full(n_per_class, c, dtype=np.int64))
    return np.concatenate(X), np.concatenate(y)


class TestForwardPass(unittest.TestCase):
    def test_output_shapes(self) -> None:
        model = MLP(hidden_sizes=(32, 16), seed=0)
        X = np.random.default_rng(0).random((7, 784), dtype=np.float32)
        logits = model.forward(X)
        self.assertEqual(logits.shape, (7, 10))
        self.assertEqual(model.predict(X).shape, (7,))

    def test_probabilities_sum_to_one(self) -> None:
        model = MLP(hidden_sizes=(8,), seed=1)
        X = np.random.default_rng(1).random((5, 784), dtype=np.float32)
        probs = model.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_linear_model_has_no_hidden_layers(self) -> None:
        model = MLP(hidden_sizes=(), seed=0)
        self.assertEqual(len(model.layers), 1)
        self.assertEqual(len(model.trainable_layers()), 1)

    def test_dropout_only_active_in_training_mode(self) -> None:
        model = MLP(hidden_sizes=(64,), dropout=0.5, seed=2)
        X = np.random.default_rng(2).random((32, 784), dtype=np.float32)
        np.testing.assert_allclose(model.forward(X, training=False), model.forward(X, training=False))
        self.assertFalse(np.allclose(model.forward(X, training=True), model.forward(X, training=True)))

    def test_batched_prediction_matches_single_batch(self) -> None:
        model = MLP(hidden_sizes=(16,), seed=3)
        X = np.random.default_rng(3).random((40, 784), dtype=np.float32)
        np.testing.assert_allclose(
            model.predict_proba(X, batch_size=8), model.predict_proba(X, batch_size=1000), atol=1e-6
        )


class TestPersistence(unittest.TestCase):
    def test_save_load_round_trip(self) -> None:
        model = MLP(hidden_sizes=(24, 12), dropout=0.1, init="xavier", seed=5)
        X = np.random.default_rng(5).random((10, 784), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "model.npz"
            model.save(path)
            self.assertTrue(path.exists())
            restored = MLP.load(path)
        self.assertEqual(restored.config(), model.config())
        np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X), atol=1e-6)

    def test_load_rejects_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            MLP.load("/nonexistent/model.npz")


class TestTraining(unittest.TestCase):
    def test_model_can_fit_separable_data(self) -> None:
        X, y = separable_dataset()
        model = MLP(hidden_sizes=(32,), seed=0)
        optimizer = nn.SGD(model.trainable_layers(), lr=0.1, momentum=0.9)
        rng = np.random.default_rng(0)
        for _ in range(60):
            logits = model.forward(X, training=True)
            _, dlogits = nn.softmax_cross_entropy(logits, y)
            model.backward(dlogits)
            optimizer.step()
            rng.permutation(len(X))
        self.assertGreater(float((model.predict(X) == y).mean()), 0.95)

    def test_loss_decreases(self) -> None:
        X, y = separable_dataset()
        model = MLP(hidden_sizes=(16,), seed=1)
        optimizer = nn.Adam(model.trainable_layers(), lr=0.01)
        first, _ = model.loss(X, y)
        for _ in range(50):
            _, dlogits = model.loss(X, y, training=True)
            model.backward(dlogits)
            optimizer.step()
        last, _ = model.loss(X, y)
        self.assertLess(last, first)


if __name__ == "__main__":
    unittest.main()
