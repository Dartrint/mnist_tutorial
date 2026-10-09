#!/usr/bin/env python3
"""Cross-platform task runner — the Windows-friendly twin of the Makefile.

Works identically on Windows (PowerShell/cmd), macOS and Linux, and never
needs ``make``, ``bash`` or any shell-specific syntax:

    python scripts/tasks.py                # list every task
    python scripts/tasks.py data
    python scripts/tasks.py train --epochs 8 --hidden 256,128
    python scripts/tasks.py serve --port 8000
    python scripts/tasks.py test

Extra arguments after the task name are appended to the underlying command, so
anything the CLI accepts can be passed straight through.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# The repository root (or MNIST_TASKS_ROOT, which the tests use for isolation).
ROOT = Path(os.environ.get("MNIST_TASKS_ROOT") or Path(__file__).resolve().parents[1])


def _python() -> str:
    return sys.executable or "python"


def _cmd(*args: str) -> list[str]:
    return [_python(), "-m", "mnist", *args]


#: task name -> (description, argv prefix)
TASKS: dict[str, tuple[str, list[str]]] = {
    "install": ("cài dependency runtime (numpy)", [_python(), "-m", "pip", "install", "-r", "requirements.txt"]),
    "install-torch": ("cài PyTorch bản CPU (tùy chọn, cho CNN)", [_python(), "-m", "pip", "install", "-r", "requirements-torch.txt"]),
    "data": ("tải dữ liệu MNIST vào data/", _cmd("data")),
    "train": ("huấn luyện MLP (NumPy)", _cmd("train", "--out", "models/mlp.npz", "--metrics-out", "reports/mlp_history.json")),
    "cnn": ("huấn luyện CNN (PyTorch, cần install-torch)", _cmd("cnn", "--out", "models/cnn.pt", "--metrics-out", "reports/cnn_history.json")),
    "evaluate": ("đánh giá model đã train", _cmd("evaluate", "--model", "models/mlp.npz", "--show-confusion")),
    "predict": ("dự đoán ảnh test + xuất sprite sheet", _cmd("predict", "--model", "models/mlp.npz", "--sample", "16", "--save-grid", "reports/predictions.png")),
    "serve": ("chạy website tại http://127.0.0.1:8000", _cmd("serve", "--port", "8000")),
    "test": ("chạy toàn bộ unit test", [_python(), "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]),
    "lint": ("kiểm tra style bằng ruff", [_python(), "-m", "ruff", "check", "mnist", "tests", "examples"]),
    "demo": ("demo end-to-end dưới 1 phút", [_python(), "examples/demo.py"]),
    "clean": ("xoá cache và ảnh sinh ra", []),  # handled in Python
}

HELP_FLAGS = {"-h", "--help", "help", "list", "-l"}


def _usage() -> str:
    width = max(len(name) for name in TASKS)
    lines = [f"  {name:<{width}}  {desc}" for name, (desc, _) in TASKS.items()]
    return (
        "usage: python scripts/tasks.py <task> [extra args]\n\n"
        "tasks:\n" + "\n".join(lines) + "\n\n"
        "Ví dụ:\n"
        "  python scripts/tasks.py data\n"
        "  python scripts/tasks.py train --epochs 8 --hidden 256,128\n"
        "  python scripts/tasks.py serve --port 8000\n"
    )


def clean() -> int:
    """Remove caches and generated images (keeps models/ and data/)."""
    removed = []
    for cache in list(ROOT.rglob("__pycache__")) + [
        ROOT / ".pytest_cache",
        ROOT / ".ruff_cache",
    ]:
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
            removed.append(cache.relative_to(ROOT).as_posix())
    for png in sorted((ROOT / "reports").glob("*.png")) if (ROOT / "reports").is_dir() else []:
        png.unlink()
        removed.append(png.relative_to(ROOT).as_posix())
    print(f"cleaned {len(removed)} item(s)" + (": " + ", ".join(removed) if removed else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in HELP_FLAGS:
        stream = sys.stdout if argv else sys.stderr
        print(_usage(), file=stream)
        return 0 if argv else 2

    task, extra = argv[0], argv[1:]
    if task not in TASKS:
        print(f"unknown task {task!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    if task == "clean":
        return clean()

    command = [*TASKS[task][1], *extra]
    print(f"[tasks] {task}: {' '.join(command)}", flush=True)
    try:
        return subprocess.call(command, cwd=str(ROOT))
    except FileNotFoundError as exc:
        print(f"[tasks] could not start {command[0]!r}: {exc}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    sys.exit(main())
