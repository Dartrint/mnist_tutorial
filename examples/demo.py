#!/usr/bin/env python3
"""End-to-end MNIST demo on a small subset — runs in well under a minute.

Usage::

    python examples/demo.py            # train a tiny model and show results
    python examples/demo.py --limit 2000 --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running the file directly from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnist.data import load_mnist, to_images  # noqa: E402
from mnist.evaluate import evaluate_model  # noqa: E402
from mnist.imageio import to_ascii, write_png  # noqa: E402
from mnist.predict import render_grid  # noqa: E402
from mnist.train import train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8000, help="training images to use")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", default="128")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/demo.npz")
    parser.add_argument("--grid", default="reports/demo_grid.png")
    args = parser.parse_args()

    hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())
    print("== 1/4  train ==================================================")
    history = train(
        epochs=args.epochs,
        hidden_sizes=hidden,
        lr=0.1,
        optimizer_name="sgd",
        momentum=0.9,
        batch_size=128,
        limit=args.limit,
        val_size=1000,
        seed=args.seed,
        out=args.out,
    )

    print("\n== 2/4  evaluate ===============================================")
    from mnist.model import MLP

    model = MLP.load(args.out)
    metrics = evaluate_model(model, limit=2000)
    print(metrics["report"])

    print("\n== 3/4  sample predictions =====================================")
    _, _, X_test, y_test = load_mnist()
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X_test), size=8, replace=False)
    probs = model.predict_proba(X_test[idx])
    preds = probs.argmax(axis=1)
    for i, (true, pred) in enumerate(zip(y_test[idx], preds, strict=True)):
        mark = "OK  " if true == pred else "MISS"
        print(f"  [{i}] true={true} pred={pred} p={probs[i, pred]:.3f} {mark}")

    print("\n== 4/4  artefacts ==============================================")
    print("first sample as ASCII art:")
    print(to_ascii(to_images(X_test[idx])[0]))
    grid = render_grid(
        to_images(X_test[idx]),
        [f"{p}x{t}" for p, t in zip(preds, y_test[idx], strict=True)],
    )
    write_png(args.grid, grid)
    print(f"\nwrote sprite sheet -> {args.grid}")
    print(f"test accuracy     -> {history['test_acc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
