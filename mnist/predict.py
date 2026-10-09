"""Run inference on MNIST test images or on your own digit pictures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .data import DEFAULT_DATA_DIR, load_mnist, to_images
from .evaluate import load_model
from .imageio import preprocess_digit, read_image, to_ascii, write_png
from .utils import configure_stdio

if TYPE_CHECKING:  # pragma: no cover
    from .model import MLP


def render_grid(
    images: np.ndarray,
    titles: list[str],
    cols: int = 8,
    cell: int = 28,
    pad: int = 2,
) -> np.ndarray:
    """Compose a grayscale sprite sheet of ``images`` with caption space on top."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 2:
        images = images[None]
    rows = int(np.ceil(len(images) / cols))
    caption = 10
    height = rows * (cell + caption + pad) + pad
    width = cols * (cell + pad) + pad
    canvas = np.zeros((height, width), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        y = pad + r * (cell + caption + pad) + caption
        x = pad + c * (cell + pad)
        tile = (np.clip(img.reshape(cell, cell), 0, 1) * 255).astype(np.uint8)
        canvas[y : y + cell, x : x + cell] = tile
        # 3x5 pixel font strip encoding the caption digits (see _draw_caption).
        _draw_caption(canvas, titles[i] if i < len(titles) else "", x, y - caption + 3)
    return canvas


_FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "v": ("101", "101", "101", "101", "010"),
    "x": ("101", "101", "010", "101", "101"),
    "/": ("001", "001", "010", "100", "100"),
    "-": ("000", "000", "111", "000", "000"),
    " ": ("000", "000", "000", "000", "000"),
}


def _draw_caption(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    """Blit a short caption using a tiny built-in 3x5 bitmap font."""
    cursor = x
    for ch in text.lower():
        glyph = _FONT.get(ch)
        if glyph is None:
            cursor += 4
            continue
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    yy, xx = y + gy, cursor + gx
                    if 0 <= yy < canvas.shape[0] and 0 <= xx < canvas.shape[1]:
                        canvas[yy, xx] = 255
        cursor += 4


def predict_vector(model: MLP, vector: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Predict one flattened digit; return ``(class, confidence, probs)``."""
    probs = model.predict_proba(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs


def predict_paths(
    model: MLP, paths: list[str | Path], invert: bool | None = None, center: bool = True
) -> list[dict]:
    """Predict every image path given on the command line."""
    results = []
    for path in paths:
        image = read_image(path)
        vector = preprocess_digit(image, invert=invert, center=center)
        cls, conf, probs = predict_vector(model, vector)
        results.append({"path": str(path), "digit": cls, "confidence": conf,
                        "probs": probs.tolist(), "image": vector.reshape(28, 28)})
    return results


def sample_test_set(
    data_dir: str | Path = DEFAULT_DATA_DIR, n: int = 16, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` random images (as ``(n, 28, 28)``) and their labels from the test set."""
    _, _, X_test, y_test = load_mnist(data_dir)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=min(n, len(X_test)), replace=False)
    return to_images(X_test[idx]), y_test[idx]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict digits with a trained MNIST model.")
    p.add_argument("--model", default="models/mlp.npz")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--image", action="append", default=[], help="PNG/PGM file (repeatable)")
    p.add_argument("--sample", type=int, default=0, help="predict N random test images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--invert", choices=("auto", "yes", "no"), default="auto")
    p.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        help="tắt cắt+scale+căn khối tâm (chỉ dùng khi ảnh đã đúng chuẩn MNIST)",
    )
    p.add_argument("--ascii", action="store_true", help="print each digit as ASCII art")
    p.add_argument("--save-grid", default=None, help="write a PNG sprite sheet of predictions")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)
    model = load_model(args.model)
    invert = {"auto": None, "yes": True, "no": False}[args.invert]

    images, titles = [], []
    if args.image:
        for item in predict_paths(model, args.image, invert=invert, center=args.center):
            images.append(item["image"])
            titles.append(f"{item['digit']}v{item['confidence'] * 100:.0f}")
            print(f"{item['path']}: digit={item['digit']} confidence={item['confidence']:.3f}")
            if args.ascii:
                print(to_ascii(item["image"]))

    if args.sample:
        X, y = sample_test_set(args.data_dir, n=args.sample, seed=args.seed)
        probs = model.predict_proba(X.reshape(len(X), -1))
        preds = probs.argmax(axis=1)
        correct = int((preds == y).sum())
        print(f"sampled {len(y)} test images — accuracy {correct}/{len(y)}")
        for img, true, pred, pr in zip(X, y, preds, probs, strict=True):
            print(f"  true={true}  pred={pred}  p={pr[pred]:.3f}  {'OK' if true == pred else 'MISS'}")
            if args.ascii:
                print(to_ascii(img))
        images.extend(list(X))
        titles.extend([f"{p}x{t}" for p, t in zip(preds, y, strict=True)])

    if args.save_grid:
        if not images:
            print("[predict] nothing to draw; pass --image or --sample", file=sys.stderr)
            return 2
        grid = render_grid(np.stack(images), titles)
        write_png(args.save_grid, grid)
        print(f"[predict] wrote grid -> {args.save_grid}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

