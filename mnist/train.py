"""Mini-batch training loop for the NumPy MLP, plus a command-line interface."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import nn
from .data import DEFAULT_DATA_DIR, load_mnist
from .model import MLP
from .utils import AverageMeter, accuracy, save_json, set_seed


def iterate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator, shuffle: bool = True
):
    """Yield ``(x_batch, y_batch)`` tuples, optionally shuffled each call."""
    n = len(X)
    indices = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        yield X[idx], y[idx]


def build_optimizer(model: MLP, name: str, lr: float, momentum: float, weight_decay: float):
    """Instantiate an optimizer over the model's trainable layers."""
    layers = model.trainable_layers()
    if name == "sgd":
        return nn.SGD(layers, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return nn.Adam(layers, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def train_epoch(model: MLP, optimizer, X, y, batch_size: int, rng) -> tuple[float, float]:
    """Run one epoch; return ``(mean_loss, accuracy)`` over the training set.

    The logits from the forward pass are reused for both the loss and the
    accuracy, so each mini-batch costs a single forward pass.
    """
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for xb, yb in iterate_minibatches(X, y, batch_size, rng, shuffle=True):
        logits = model.forward(xb, training=True)
        loss, dlogits = nn.softmax_cross_entropy(logits, yb)
        model.backward(dlogits)
        optimizer.step()
        loss_meter.update(loss, n=len(xb))
        acc_meter.update(accuracy(logits, yb), n=len(xb))
    return loss_meter.avg, acc_meter.avg


def evaluate(model: MLP, X, y, batch_size: int = 4096) -> tuple[float, float]:
    """Return ``(mean_loss, accuracy)`` for a dataset (single forward per batch)."""
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for start in range(0, len(X), batch_size):
        xb, yb = X[start : start + batch_size], y[start : start + batch_size]
        logits = model.forward(xb, training=False)
        loss, _ = nn.softmax_cross_entropy(logits, yb)
        loss_meter.update(loss, n=len(xb))
        acc_meter.update(accuracy(logits, yb), n=len(xb))
    return loss_meter.avg, acc_meter.avg


def train(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    epochs: int = 10,
    batch_size: int = 128,
    hidden_sizes: tuple[int, ...] = (256,),
    dropout: float = 0.0,
    lr: float = 0.1,
    optimizer_name: str = "sgd",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    init: str = "he",
    seed: int = 0,
    limit: int | None = None,
    val_size: int = 5000,
    patience: int | None = None,
    out: str | Path | None = "models/mlp.npz",
    verbose: bool = True,
) -> dict:
    """Train an :class:`MLP` on MNIST and return a history dictionary."""
    rng = set_seed(seed)
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    if limit is not None:
        X_train, y_train = X_train[:limit], y_train[:limit]

    if 0 < val_size < len(X_train):
        perm = rng.permutation(len(X_train))
        val_idx, fit_idx = perm[:val_size], perm[val_size:]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
    else:
        X_fit, y_fit = X_train, y_train
        X_val, y_val = X_test, y_test

    model = MLP(hidden_sizes=hidden_sizes, dropout=dropout, init=init, seed=seed)
    optimizer = build_optimizer(model, optimizer_name, lr, momentum, weight_decay)

    history: dict = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "seconds": [],
    }
    best_acc, best_weights, best_epoch, stale = -1.0, None, 0, 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_epoch(model, optimizer, X_fit, y_fit, batch_size, rng)
        va_loss, va_acc = evaluate(model, X_val, y_val)
        elapsed = time.perf_counter() - t0

        if not math.isfinite(tr_loss) or not math.isfinite(va_loss):
            print(
                f"[train] loss became non-finite at epoch {epoch} — stopping. "
                "Try a smaller --lr or --weight-decay.",
                file=sys.stderr,
            )
            break

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["seconds"].append(elapsed)

        if verbose:
            print(
                f"epoch {epoch:>3}/{epochs}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}"
                f"  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  ({elapsed:.1f}s)",
                flush=True,
            )

        if va_acc > best_acc:
            best_acc, best_epoch, stale = va_acc, epoch, 0
            best_weights = [p.copy() for layer in model.trainable_layers() for p in layer.params()]
        else:
            stale += 1
            if patience is not None and stale >= patience:
                if verbose:
                    print(f"[train] early stopping at epoch {epoch} (no gain for {stale} epochs)")
                break

    # Restore the best checkpoint observed during training.
    if best_weights is not None:
        flat = [p for layer in model.trainable_layers() for p in layer.params()]
        for target, value in zip(flat, best_weights, strict=True):
            target[...] = value

    test_loss, test_acc = evaluate(model, X_test, y_test)
    history.update(
        best_val_acc=best_acc,
        best_epoch=best_epoch,
        test_loss=test_loss,
        test_acc=test_acc,
        total_seconds=time.perf_counter() - started,
        config=model.config() | {"optimizer": optimizer_name, "lr": lr, "batch_size": batch_size},
    )
    if out:
        model.save(out)
        if verbose:
            print(f"[train] saved model -> {out}")
    return history



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a NumPy MLP on MNIST.")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="where IDX files live")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", default="256", help="comma-separated hidden widths, e.g. 512,256")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--init", choices=("he", "xavier"), default="he")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="use only the first N training images")
    p.add_argument("--val-size", type=int, default=5000, help="hold-out size carved from train")
    p.add_argument("--patience", type=int, default=None, help="early-stopping patience (epochs)")
    p.add_argument("--out", default="models/mlp.npz", help="model output path ('' to skip saving)")
    p.add_argument("--metrics-out", default=None, help="optional JSON file for the history")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hidden = tuple(int(h) for h in str(args.hidden).split(",") if h.strip())
    history = train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_sizes=hidden,
        dropout=args.dropout,
        lr=args.lr,
        optimizer_name=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        init=args.init,
        seed=args.seed,
        limit=args.limit,
        val_size=args.val_size,
        patience=args.patience,
        out=args.out or None,
        verbose=not args.quiet,
    )
    print(f"test accuracy: {history['test_acc']:.4f}  (loss {history['test_loss']:.4f})")
    if args.metrics_out:
        save_json(history, args.metrics_out)
        print(f"[train] wrote metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

