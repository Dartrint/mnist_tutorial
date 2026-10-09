"""Evaluate a trained model: accuracy, confusion matrix, per-class metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA_DIR, load_mnist
from .model import MLP
from .utils import classification_report, configure_stdio, confusion_matrix, save_json


def load_model(path: str | Path):
    """Load a ``.npz`` (NumPy MLP) or ``.pt`` (PyTorch CNN) checkpoint."""
    if Path(path).suffix == ".pt":
        from .torch_cnn import CNNClassifier

        return CNNClassifier.load(path)
    return MLP.load(path)


def evaluate_model(
    model,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str = "test",
    limit: int | None = None,
) -> dict:
    """Compute a metrics dictionary for ``model`` on the chosen split."""
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    X, y = (X_train, y_train) if split == "train" else (X_test, y_test)
    if limit is not None:
        X, y = X[:limit], y[:limit]
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    n_classes = int(getattr(model, "n_classes", probs.shape[1]))
    loss = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())
    acc = float((preds == y).mean())
    cm = confusion_matrix(y, preds, n_classes)
    per_class = {
        str(c): float(cm[c, c] / cm[c].sum()) if cm[c].sum() else 0.0 for c in range(n_classes)
    }
    return {
        "split": split,
        "n": int(len(y)),
        "loss": loss,
        "accuracy": acc,
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y, preds, n_classes),
    }



def format_confusion(cm: np.ndarray) -> str:
    """Render a confusion matrix as a fixed-width table."""
    header = "true\\pred " + "".join(f"{c:>7}" for c in range(cm.shape[1]))
    lines = [header, "-" * len(header)]
    for i, row in enumerate(cm):
        lines.append(f"{i:>9} " + "".join(f"{v:>7}" for v in row))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained MNIST model.")
    p.add_argument("--model", default="models/mlp.npz")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--split", choices=("train", "test"), default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--show-confusion", action="store_true")
    p.add_argument("--json", dest="json_out", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)
    model = load_model(args.model)
    metrics = evaluate_model(
        model, data_dir=args.data_dir, split=args.split, limit=args.limit
    )
    print(f"model:      {args.model}")
    print(f"split:      {metrics['split']}  (n={metrics['n']})")
    print(f"loss:       {metrics['loss']:.4f}")
    print(f"accuracy:   {metrics['accuracy']:.4f}")
    print()
    print(metrics["report"])
    if args.show_confusion:
        print()
        print(format_confusion(np.array(metrics["confusion_matrix"])))
    if args.json_out:
        save_json(metrics, args.json_out)
        print(f"\n[evaluate] wrote metrics -> {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
