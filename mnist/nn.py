"""Tiny autograd-free neural-network toolkit implemented with NumPy.

Every layer exposes ``forward``/``backward`` and the ``params``/``grads`` arrays
in a matching order so that optimizers can be written generically.
"""

from __future__ import annotations

import numpy as np


def he_normal(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """He initialisation for ReLU networks: ``N(0, sqrt(2/fan_in))``."""
    fan_in = shape[0]
    return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)


def xavier_uniform(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Glorot/Xavier uniform initialisation."""
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


_INITIALISERS = {"he": he_normal, "xavier": xavier_uniform}


class Layer:
    """Base class: subclasses must provide ``forward``/``backward``."""

    def params(self) -> list[np.ndarray]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def update(self, param: np.ndarray, value: np.ndarray) -> None:
        """Write an updated parameter back (kept in place for simplicity)."""


class Linear(Layer):
    """Affine transform ``y = x @ W + b``."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator,
                 init: str = "he") -> None:
        if init not in _INITIALISERS:
            raise ValueError(f"unknown init {init!r}; expected one of {sorted(_INITIALISERS)}")
        self.W = _INITIALISERS[init]((in_features, out_features), rng).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._x is not None, "backward() called before forward()"
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]  # type: ignore[list-item]

    def update(self, param: np.ndarray, value: np.ndarray) -> None:
        param[...] = value


class ReLU(Layer):
    """Element-wise rectifier."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._mask is not None, "backward() called before forward()"
        return dout * self._mask


class Dropout(Layer):
    """Inverted dropout (active only while ``training`` is true)."""

    def __init__(self, p: float = 0.0, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p
        self.rng = rng or np.random.default_rng()
        self.training = True
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.p
        self._mask = (self.rng.random(x.shape) < keep).astype(x.dtype) / keep
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return dout
        return dout * self._mask


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Return ``(mean_loss, dlogits)`` for integer labels ``y``."""
    probs = softmax(logits)
    n = logits.shape[0]
    eps = 1e-12
    loss = -np.log(probs[np.arange(n), y] + eps).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    return float(loss), dlogits


class Optimizer:
    """Base optimizer holding a flat view of every parameter it updates."""

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def _pairs(self):
        for layer in self.layers:
            for param, grad in zip(layer.params(), layer.grads()):
                if param is not None and grad is not None:
                    yield layer, param, grad

    def step(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay."""

    def __init__(self, layers: list[Layer], lr: float = 0.1, momentum: float = 0.0,
                 weight_decay: float = 0.0) -> None:
        super().__init__(layers)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: dict[int, np.ndarray] = {}

    def step(self) -> None:
        for layer, param, grad in self._pairs():
            g = grad
            if self.weight_decay:
                g = g + self.weight_decay * param
            if self.momentum:
                v = self._velocity.get(id(param))
                v = g if v is None else self.momentum * v + g
                self._velocity[id(param)] = v
                g = v
            layer.update(param, param - self.lr * g)


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015)."""

    def __init__(self, layers: list[Layer], lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        super().__init__(layers)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def step(self) -> None:
        self.t += 1
        b1t = 1.0 - self.beta1 ** self.t
        b2t = 1.0 - self.beta2 ** self.t
        for layer, param, grad in self._pairs():
            g = grad + self.weight_decay * param if self.weight_decay else grad
            m = self._m.get(id(param))
            m = g if m is None else self.beta1 * m + (1 - self.beta1) * g
            v = self._v.get(id(param))
            v = g * g if v is None else self.beta2 * v + (1 - self.beta2) * g * g
            self._m[id(param)] = m
            self._v[id(param)] = v
            m_hat = m / b1t
            v_hat = v / b2t
            layer.update(param, param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
