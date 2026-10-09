"""Tests for the from-scratch layers, losses and optimizers."""

from __future__ import annotations

import unittest

import numpy as np

from mnist import nn


class TestSoftmaxCrossEntropy(unittest.TestCase):
    def test_softmax_is_a_distribution(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        probs = nn.softmax(logits)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue((probs >= 0).all())

    def test_softmax_is_shift_invariant(self) -> None:
        logits = np.random.default_rng(0).normal(size=(4, 5)).astype(np.float32)
        np.testing.assert_allclose(nn.softmax(logits), nn.softmax(logits + 100.0), atol=1e-6)

    def test_loss_matches_manual_computation(self) -> None:
        logits = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        y = np.array([0, 1])
        loss, dlogits = nn.softmax_cross_entropy(logits, y)
        probs = nn.softmax(logits)
        expected = -np.log(probs[0, 0]) - np.log(probs[1, 1])
        self.assertAlmostEqual(loss, expected / 2, places=6)
        # Rows of (softmax - onehot) sum to zero.
        np.testing.assert_allclose(dlogits.sum(axis=1), 0.0, atol=1e-6)


class TestLayers(unittest.TestCase):
    def test_linear_shapes(self) -> None:
        rng = np.random.default_rng(0)
        layer = nn.Linear(4, 3, rng)
        x = rng.normal(size=(5, 4)).astype(np.float32)
        out = layer.forward(x)
        self.assertEqual(out.shape, (5, 3))
        self.assertEqual(layer.backward(np.ones_like(out)).shape, (5, 4))
        self.assertEqual(layer.dW.shape, (4, 3))
        self.assertEqual(layer.db.shape, (3,))

    def test_relu_masks_negatives(self) -> None:
        relu = nn.ReLU()
        x = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(relu.forward(x), [[0.0, 0.0, 2.0]])
        np.testing.assert_allclose(relu.backward(np.ones_like(x)), [[0.0, 0.0, 1.0]])

    def test_dropout_is_identity_when_inactive(self) -> None:
        rng = np.random.default_rng(0)
        layer = nn.Dropout(0.5, rng)
        x = rng.normal(size=(32, 32)).astype(np.float32)
        layer.training = False
        np.testing.assert_allclose(layer.forward(x), x)

    def test_dropout_preserves_expectation(self) -> None:
        rng = np.random.default_rng(0)
        layer = nn.Dropout(0.25, rng)
        x = np.ones((1000, 1000), dtype=np.float32)
        self.assertAlmostEqual(float(layer.forward(x).mean()), 1.0, places=1)


def _relative_gradient_error(build, seed: int = 0, eps: float = 1e-6) -> float:
    """Max relative error between analytic and central-difference gradients.

    Parameters are promoted to float64 so the finite differences are not
    swamped by float32 round-off.
    """
    rng = np.random.default_rng(seed)
    layers = build(rng)
    for layer in layers:
        if isinstance(layer, nn.Linear):
            layer.W = layer.W.astype(np.float64)
            layer.b = layer.b.astype(np.float64)
    x = rng.normal(size=(6, 784)).astype(np.float64)
    y = rng.integers(0, 10, size=6)
    dropouts = [layer for layer in layers if isinstance(layer, nn.Dropout)]

    def forward():
        # Re-seed so every evaluation sees the *same* dropout mask, otherwise the
        # finite differences compare two different random networks.
        for layer in dropouts:
            layer.rng = np.random.default_rng(1234)
        out = x
        for layer in layers:
            out = layer.forward(out)
        return out

    def loss() -> float:
        value, _ = nn.softmax_cross_entropy(forward(), y)
        return value

    _, dlogits = nn.softmax_cross_entropy(forward(), y)
    grad = dlogits
    for layer in reversed(layers):
        grad = layer.backward(grad)

    worst = 0.0
    for layer in layers:
        for param, analytic in zip(layer.params(), layer.grads(), strict=True):
            flat, gflat = param.reshape(-1), analytic.reshape(-1)
            for i in rng.choice(flat.size, size=min(5, flat.size), replace=False):
                old = float(flat[i])
                flat[i] = old + eps
                hi = loss()
                flat[i] = old - eps
                lo = loss()
                flat[i] = old
                numeric = (hi - lo) / (2 * eps)
                denom = max(1e-8, abs(numeric) + abs(float(gflat[i])))
                worst = max(worst, abs(numeric - float(gflat[i])) / denom)
    return worst


class TestGradients(unittest.TestCase):
    def test_linear_only_gradients(self) -> None:
        self.assertLess(_relative_gradient_error(lambda rng: [nn.Linear(784, 10, rng)]), 1e-3)

    def test_mlp_with_relu_gradients(self) -> None:
        def build(rng):
            return [nn.Linear(784, 32, rng), nn.ReLU(), nn.Linear(32, 10, rng)]

        self.assertLess(_relative_gradient_error(build, seed=3), 1e-3)

    def test_dropout_does_not_break_gradients(self) -> None:
        def build(rng):
            return [nn.Linear(784, 16, rng), nn.ReLU(), nn.Dropout(0.5, rng), nn.Linear(16, 10, rng)]

        self.assertLess(_relative_gradient_error(build, seed=7), 1e-3)


class TestOptimizers(unittest.TestCase):
    @staticmethod
    def _fit_quadratic(make_optimizer, seed: int = 0) -> float:
        """Fit a random linear layer to a constant target; return the final MSE."""
        rng = np.random.default_rng(seed)
        layer = nn.Linear(20, 1, rng)
        optimizer = make_optimizer(layer)
        target = np.ones((16, 1), dtype=np.float32)
        x = rng.normal(size=(16, 20)).astype(np.float32)
        for _ in range(400):
            diff = layer.forward(x) - target
            layer.dW = x.T @ diff / len(x)
            layer.db = diff.mean(axis=0)
            optimizer.step()
        return float(((layer.forward(x) - target) ** 2).mean())

    def test_sgd_reduces_loss(self) -> None:
        loss = self._fit_quadratic(lambda layer: nn.SGD([layer], lr=0.1))
        self.assertLess(loss, 1e-2)

    def test_adam_reduces_loss(self) -> None:
        loss = self._fit_quadratic(lambda layer: nn.Adam([layer], lr=0.05))
        self.assertLess(loss, 1e-2)

    def test_momentum_and_weight_decay_update_parameters(self) -> None:
        layer = nn.Linear(4, 2, np.random.default_rng(0))
        opt = nn.SGD([layer], lr=0.01, momentum=0.9, weight_decay=1e-4)
        layer.forward(np.ones((3, 4), dtype=np.float32))
        layer.backward(np.ones((3, 2), dtype=np.float32))
        before = layer.W.copy()
        opt.step()
        self.assertFalse(np.allclose(before, layer.W))


if __name__ == "__main__":
    unittest.main()

