"""Shared helpers: seeding, metrics, serialisation and tiny utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int | None) -> np.random.Generator:
    """Seed NumPy globally and return a fresh ``Generator``."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.default_rng(seed)


def one_hot(y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Convert integer labels into a one-hot matrix."""
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), np.asarray(y, dtype=np.int64)] = 1.0
    return out


def accuracy(logits_or_probs: np.ndarray, y: np.ndarray) -> float:
    """Fraction of rows whose arg-max matches ``y``."""
    preds = np.argmax(logits_or_probs, axis=1)
    return float((preds == np.asarray(y)).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Rows = true label, columns = predicted label."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> str:
    """A compact precision/recall/F1 report as a formatted string."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    total = cm.sum()
    lines = ["class   precision   recall   f1-score   support"]
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        lines.append(f"{c:>5}   {precision:9.3f}   {recall:6.3f}   {f1:8.3f}   {cm[c].sum():>7}")
    acc = np.trace(cm) / total if total else 0.0
    lines.append(f"\naccuracy: {acc:.4f}   (n={total})")
    return "\n".join(lines)


class AverageMeter:
    """Track a running mean over the entries of a stream of values."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


class Timer:
    """Minimal context manager that records elapsed wall-clock seconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
