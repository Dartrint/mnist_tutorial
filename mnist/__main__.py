"""Unified command line entry point: ``python -m mnist <command> [options]``."""

from __future__ import annotations

import importlib
import sys

from .utils import configure_stdio

COMMANDS = {
    "data": ("mnist.data", "tải MNIST IDX files vào data/"),
    "train": ("mnist.train", "huấn luyện MLP bằng NumPy"),
    "cnn": ("mnist.torch_cnn", "huấn luyện CNN bằng PyTorch (cần torch)"),
    "evaluate": ("mnist.evaluate", "đánh giá model: accuracy, confusion matrix"),
    "predict": ("mnist.predict", "dự đoán ảnh hoặc mẫu test, vẽ sprite sheet"),
    "serve": ("mnist.webapp", "chạy website thao tác với model đã train"),
}

USAGE = """usage: python -m mnist <command> [options]

commands:
{commands}

Ví dụ:
  python -m mnist data
  python -m mnist train --epochs 8 --hidden 256,128
  python -m mnist evaluate --model models/mlp.npz --show-confusion
  python -m mnist predict --model models/mlp.npz --sample 16 --ascii
  python -m mnist serve --port 8000

Chạy `python -m mnist <command> --help` để xem tùy chọn của từng lệnh.
"""


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    rendered = "\n".join(f"  {name:<9} {desc}" for name, (_, desc) in COMMANDS.items())

    if not argv or argv[0] in ("-h", "--help", "help"):
        stream = sys.stdout if argv else sys.stderr
        print(USAGE.format(commands=rendered), file=stream)
        return 0 if argv else 2

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"unknown command {command!r}\n", file=sys.stderr)
        print(USAGE.format(commands=rendered), file=sys.stderr)
        return 2

    module = importlib.import_module(COMMANDS[command][0])
    return int(module.main(rest))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
