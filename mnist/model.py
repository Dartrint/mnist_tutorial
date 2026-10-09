"""A configurable multi-layer perceptron for MNIST."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import nn
from .utils import set_seed


class MLP:
    """Fully-connected classifier: ``784 -> hidden... -> 10`` with ReLU.

    Parameters
    ----------
    hidden_sizes:
        Widths of the hidden layers (empty tuple gives a linear softmax model).
    dropout:
        Dropout probability applied after each hidden activation.
    init:
        Weight initialisation, ``"he"`` (default) or ``"xavier"``.
    seed:
        Seed used for weight initialisation and dropout sampling.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (256,),
        dropout: float = 0.0,
        init: str = "he",
        in_features: int = 28 * 28,
        n_classes: int = 10,
        seed: int | None = 0,
    ) -> None:
        self.in_features = int(in_features)
        self.n_classes = int(n_classes)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.dropout = float(dropout)
        self.init = init

        rng = set_seed(seed)
        sizes = (self.in_features, *self.hidden_sizes, self.n_classes)
        self.layers: list[nn.Layer] = []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            is_last = i == len(sizes) - 2
            self.layers.append(nn.Linear(fan_in, fan_out, rng, init=init))
            if not is_last:
                self.layers.append(nn.ReLU())
                if self.dropout > 0:
                    self.layers.append(nn.Dropout(self.dropout, rng))

    # ------------------------------------------------------------------ core
    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Return class logits for a batch ``X`` of shape ``(N, 784)``."""
        out = np.asarray(X, dtype=np.float32)
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                layer.training = training
            out = layer.forward(out)
        return out

    def loss(self, X: np.ndarray, y: np.ndarray, training: bool = False) -> tuple[float, np.ndarray]:
        """Forward pass plus ``(loss, dlogits)``."""
        logits = self.forward(X, training=training)
        return nn.softmax_cross_entropy(logits, np.asarray(y, dtype=np.int64))

    def backward(self, dlogits: np.ndarray) -> None:
        """Backpropagate ``dlogits`` through every layer, filling gradients."""
        grad = dlogits
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # ------------------------------------------------------------- inference
    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Softmax probabilities, computed in batches to bound memory."""
        X = np.asarray(X, dtype=np.float32)
        out = np.empty((len(X), self.n_classes), dtype=np.float32)
        for start in range(0, len(X), batch_size):
            chunk = X[start : start + batch_size]
            out[start : start + len(chunk)] = nn.softmax(self.forward(chunk, training=False))
        return out

    def predict(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Predicted class indices for ``X``."""
        return np.argmax(self.predict_proba(X, batch_size=batch_size), axis=1)

    def trainable_layers(self) -> list[nn.Layer]:
        """Layers that own parameters (used by optimizers)."""
        return [layer for layer in self.layers if layer.params()]

    # ------------------------------------------------------------ persistence
    def config(self) -> dict:
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
            "init": self.init,
            "in_features": self.in_features,
            "n_classes": self.n_classes,
        }

    def save(self, path: str | Path) -> Path:
        """Persist weights + architecture to a single ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"__config__": np.array(json.dumps(self.config()))}
        for i, layer in enumerate(self.layers):
            for j, param in enumerate(layer.params()):
                arrays[f"layer{i}_param{j}"] = param
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        """Reconstruct a model previously written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            cfg = json.loads(str(data["__config__"]))
            model = cls(
                hidden_sizes=tuple(cfg["hidden_sizes"]),
                dropout=cfg["dropout"],
                init=cfg["init"],
                in_features=cfg["in_features"],
                n_classes=cfg["n_classes"],
            )
            for i, layer in enumerate(model.layers):
                for j, param in enumerate(layer.params()):
                    key = f"layer{i}_param{j}"
                    if key in data:
                        param[...] = data[key]
        return model
