"""Optional PyTorch CNN backend (CPU friendly) for ~99% MNIST accuracy.

This module is imported lazily: the core package only needs NumPy. Install the
extra dependency with ``pip install torch --index-url https://download.pytorch.org/whl/cpu``
(see ``requirements-torch.txt``).

The wrapper :class:`CNNClassifier` mirrors the :class:`mnist.model.MLP`
interface (``predict_proba`` / ``predict`` / ``save`` / ``load``) so the shared
``evaluate`` and ``predict`` CLIs work with ``.pt`` checkpoints too.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA_DIR, load_mnist
from .utils import configure_stdio, save_json, set_seed


def _torch():
    """Import torch on demand with a friendly error message."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyTorch is not installed. Install the CPU wheel with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch, nn


def build_cnn(n_classes: int = 10, dropout: float = 0.25):
    """A compact VGG-style CNN: 2 conv blocks + 2 fully-connected layers."""
    _, nn = _torch()

    class CNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(dropout),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout(dropout),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):  # noqa: D102 - torch convention
            return self.classifier(self.features(x))

    return CNN()


class CNNClassifier:
    """Thin wrapper giving the torch CNN the same API as the NumPy MLP."""

    def __init__(self, module=None, n_classes: int = 10, dropout: float = 0.25) -> None:
        self.n_classes = int(n_classes)
        self.dropout = float(dropout)
        self.module = module if module is not None else build_cnn(n_classes, dropout)

    # ------------------------------------------------------------- inference
    def predict_proba(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        torch, _ = _torch()
        X = np.asarray(X, dtype=np.float32).reshape(-1, 1, 28, 28)
        self.module.eval()
        out = np.empty((len(X), self.n_classes), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[start : start + batch_size])
                logits = self.module(batch)
                out[start : start + len(batch)] = torch.softmax(logits, dim=1).numpy()
        return out

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        return np.argmax(self.predict_proba(X, batch_size=batch_size), axis=1)

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        torch, _ = _torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.module.state_dict(), "n_classes": self.n_classes,
             "dropout": self.dropout},
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> CNNClassifier:
        torch, _ = _torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(n_classes=payload.get("n_classes", 10), dropout=payload.get("dropout", 0.25))
        model.module.load_state_dict(payload["state_dict"])
        model.module.eval()
        return model


def load_any(path: str | Path):
    """Load an ``.npz`` (NumPy MLP) or ``.pt`` (torch CNN) checkpoint."""
    path = Path(path)
    if path.suffix == ".pt":
        return CNNClassifier.load(path)
    from .model import MLP

    return MLP.load(path)



def train_cnn(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    dropout: float = 0.25,
    limit: int | None = None,
    val_size: int = 5000,
    seed: int = 0,
    out: str | Path | None = "models/cnn.pt",
    verbose: bool = True,
) -> dict:
    """Train the CNN on MNIST and return a history dictionary."""
    torch, nn = _torch()
    set_seed(seed)
    torch.manual_seed(seed)

    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    if limit is not None:
        X_train, y_train = X_train[:limit], y_train[:limit]

    rng = np.random.default_rng(seed)
    if 0 < val_size < len(X_train):
        perm = rng.permutation(len(X_train))
        val_idx, fit_idx = perm[:val_size], perm[val_size:]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
    else:
        X_fit, y_fit, X_val, y_val = X_train, y_train, X_test, y_test

    def loader(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X.reshape(-1, 1, 28, 28)),
            torch.from_numpy(np.asarray(y, dtype=np.int64)),
        )
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = loader(X_fit, y_fit, True)
    model = CNNClassifier(n_classes=10, dropout=dropout)
    optimizer = torch.optim.AdamW(model.module.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=max(1, epochs * len(train_loader)), pct_start=0.3
    )
    criterion = nn.CrossEntropyLoss()

    history: dict = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "seconds": []}
    best_acc, best_state, best_epoch = -1.0, None, 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.module.train()
        total, seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model.module(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.detach().item() * len(xb)
            seen += len(xb)
        train_loss = total / max(1, seen)
        val_acc = float((model.predict(X_val) == y_val).mean())
        elapsed = time.perf_counter() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(val_acc)  # train_acc filled below for clarity
        history["val_acc"].append(val_acc)
        history["seconds"].append(elapsed)

        if verbose:
            print(
                f"epoch {epoch:>3}/{epochs}  train_loss={train_loss:.4f}"
                f"  val_acc={val_acc:.4f}  ({elapsed:.1f}s)",
                flush=True,
            )
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.module.state_dict().items()}

    if best_state is not None:
        model.module.load_state_dict(best_state)

    test_acc = float((model.predict(X_test) == y_test).mean())
    history.update(best_val_acc=best_acc, best_epoch=best_epoch, test_acc=test_acc,
                   total_seconds=time.perf_counter() - started)
    if out:
        model.save(out)
        if verbose:
            print(f"[cnn] saved model -> {out}")
    return history



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PyTorch CNN on MNIST (CPU friendly).")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--out", default="models/cnn.pt")
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)
    torch, _ = _torch()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    history = train_cnn(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        limit=args.limit,
        val_size=args.val_size,
        seed=args.seed,
        out=args.out or None,
        verbose=not args.quiet,
    )
    print(f"test accuracy: {history['test_acc']:.4f}  (best val {history['best_val_acc']:.4f})")
    if args.metrics_out:
        save_json(history, args.metrics_out)
        print(f"[cnn] wrote metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

