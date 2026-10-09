"""Tests for the mini-batch training helpers."""

from __future__ import annotations

import unittest

import numpy as np

from mnist import nn
from mnist.model import MLP
from mnist.train import build_optimizer, evaluate, iterate_minibatches, train_epoch


class TestMinibatchIterator(unittest.TestCase):
    def test_covers_every_sample_exactly_once(self) -> None:
        X = np.arange(10 * 784, dtype=np.float32).reshape(10, 784)
        y = np.arange(10)
        rng = np.random.default_rng(0)
        seen = []
        for xb, yb in iterate_minibatches(X, y, batch_size=4, rng=rng, shuffle=True):
            self.assertEqual(len(xb), len(yb))
            seen.extend(yb.tolist())
        self.assertEqual(sorted(seen), list(range(10)))

    def test_no_shuffle_is_sequential(self) -> None:
        X = np.zeros((6, 784), dtype=np.float32)
        y = np.arange(6)
        rng = np.random.default_rng(0)
        batches = list(iterate_minibatches(X, y, batch_size=2, rng=rng, shuffle=False))
        np.testing.assert_array_equal(np.concatenate([b[1] for b in batches]), y)


class TestOptimizerFactory(unittest.TestCase):
    def test_builds_known_optimizers(self) -> None:
        model = MLP(hidden_sizes=(4,), seed=0)
        self.assertIsInstance(build_optimizer(model, "sgd", 0.1, 0.9, 0.0), nn.SGD)
        self.assertIsInstance(build_optimizer(model, "adam", 0.01, 0.9, 0.0), nn.Adam)

    def test_rejects_unknown_optimizer(self) -> None:
        model = MLP(hidden_sizes=(4,), seed=0)
        with self.assertRaises(ValueError):
            build_optimizer(model, "rmsprop", 0.1, 0.9, 0.0)


class TestEpochLoop(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        centres = rng.standard_normal((10, 784)).astype(np.float32) * 2
        labels = np.repeat(np.arange(10), 20)
        self.X = centres[labels] + 0.1 * rng.standard_normal((len(labels), 784)).astype(np.float32)
        self.y = labels.astype(np.int64)

    def test_epoch_improves_accuracy(self) -> None:
        model = MLP(hidden_sizes=(64,), seed=0)
        optimizer = nn.SGD(model.trainable_layers(), lr=0.1, momentum=0.9)
        rng = np.random.default_rng(0)
        _, acc_before = evaluate(model, self.X, self.y)
        for _ in range(10):
            train_epoch(model, optimizer, self.X, self.y, batch_size=32, rng=rng)
        _, acc_after = evaluate(model, self.X, self.y)
        self.assertGreater(acc_after, acc_before)
        self.assertGreater(acc_after, 0.8)

    def test_evaluate_returns_loss_and_accuracy(self) -> None:
        model = MLP(hidden_sizes=(8,), seed=0)
        loss, acc = evaluate(model, self.X[:50], self.y[:50], batch_size=16)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
