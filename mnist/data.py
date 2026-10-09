"""Download and parse the MNIST dataset from the original IDX files.

The files are fetched from a couple of well-known public mirrors, verified
against their published MD5 checksums and cached under ``data/`` so that
subsequent runs are fully offline.

Example
-------
>>> from mnist.data import load_mnist
>>> X_train, y_train, X_test, y_test = load_mnist()
>>> X_train.shape, X_train.dtype
((60000, 784), dtype('float32'))
"""

from __future__ import annotations

import gzip
import hashlib
import os
import ssl
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

#: Canonical file names and their MD5 checksums.
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

#: Mirrors are tried in order until one succeeds.
MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)

DEFAULT_DATA_DIR = Path(os.environ.get("MNIST_DATA_DIR", "data"))


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(name: str, expected_md5: str, data_dir: Path) -> Path:
    """Download ``name`` (unless cached and valid) and return its path."""
    dest = data_dir / name
    if dest.exists() and _md5(dest) == expected_md5:
        return dest

    data_dir.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for base in MIRRORS:
        url = base + name
        for ctx in (ssl.create_default_context(), ssl._create_unverified_context()):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "mnist-toolkit/0.1"})
                with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                    payload = resp.read()
                if hashlib.md5(payload).hexdigest() != expected_md5:
                    raise ValueError(f"checksum mismatch for {url}")
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(payload)
                tmp.replace(dest)
                return dest
            except (urllib.error.URLError, ValueError, OSError) as exc:
                last_err = exc
    raise RuntimeError(f"Could not download {name} from any mirror: {last_err}")


def download(data_dir: str | Path = DEFAULT_DATA_DIR, quiet: bool = False) -> Path:
    """Ensure all four IDX files exist locally and return the data directory."""
    data_dir = Path(data_dir)
    for name, md5 in FILES.items():
        if not quiet:
            print(f"[data] ensuring {name} ...", file=sys.stderr)
        _download_one(name, md5, data_dir)
    return data_dir


def _read_idx(path: Path) -> np.ndarray:
    """Parse an IDX (gzip compressed) file into a ``uint8`` array."""
    with gzip.open(path, "rb") as fh:
        (magic,) = struct.unpack(">I", fh.read(4))
        ndim = magic & 0xFF
        dims = struct.unpack(f">{ndim}I", fh.read(4 * ndim))
        buf = fh.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(dims)


def _load_split(data_dir: Path, images: str, labels: str) -> tuple[np.ndarray, np.ndarray]:
    x = _read_idx(data_dir / images).reshape(-1, 28 * 28).astype(np.float32)
    x /= 255.0  # normalise pixel intensities to [0, 1]
    y = _read_idx(data_dir / labels).astype(np.int64)
    return x, y


def load_mnist(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    download_if_missing: bool = True,
    quiet: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X_train, y_train, X_test, y_test)``.

    ``X`` arrays are ``float32`` of shape ``(N, 784)`` with values in ``[0, 1]``;
    ``y`` arrays hold integer labels in ``[0, 9]``.
    """
    data_dir = Path(data_dir)
    if download_if_missing and not all((data_dir / n).exists() for n in FILES):
        download(data_dir, quiet=quiet)
    if not all((data_dir / n).exists() for n in FILES):
        raise FileNotFoundError(
            f"MNIST files not found in {data_dir!r}; run `python -m mnist.data` first."
        )
    X_train, y_train = _load_split(
        data_dir, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
    )
    X_test, y_test = _load_split(
        data_dir, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    )
    return X_train, y_train, X_test, y_test


def load_split(
    split: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    limit: int | None = None,
    flatten: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a single ``"train"`` or ``"test"`` split.

    ``limit`` truncates to the first *N* examples (handy for quick runs) and
    ``flatten=False`` keeps the ``(N, 28, 28)`` image geometry.
    """
    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    if split == "train":
        X, y = X_train, y_train
    elif split == "test":
        X, y = X_test, y_test
    else:
        raise ValueError("split must be 'train' or 'test'")
    if limit is not None:
        X, y = X[:limit], y[:limit]
    if not flatten:
        X = X.reshape(-1, 28, 28)
    return X, y


def to_images(X: np.ndarray) -> np.ndarray:
    """Reshape a flat ``(N, 784)`` batch back to ``(N, 28, 28)``."""
    return np.asarray(X).reshape(-1, 28, 28)


def main(argv: list[str] | None = None) -> int:
    """Download (or verify) the MNIST files; usable as ``python -m mnist.data``."""
    import argparse

    parser = argparse.ArgumentParser(description="Download the MNIST IDX files.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    path = download(args.data_dir, quiet=args.quiet)
    print(f"MNIST ready in {Path(path).resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
