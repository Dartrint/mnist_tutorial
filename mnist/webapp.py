"""Dependency-free web UI for the trained MNIST models.

Serves a single-page app (``mnist/web/``) plus a small JSON API on top of the
standard library's ``http.server`` — no Flask, no build step:

    python -m mnist serve --port 8000

Endpoints
---------
``GET  /api/health``              service + model list
``GET  /api/models``              checkpoints found in ``models/``
``POST /api/predict``             predict a drawn/uploaded digit
``GET  /api/samples``             random test images with ground-truth labels
``POST /api/evaluate``            accuracy + confusion matrix on a test subset
``GET  /`` and ``/static/*``      the single-page UI
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import __version__
from .data import DEFAULT_DATA_DIR, load_mnist, to_images
from .evaluate import load_model
from .imageio import preprocess_digit, read_png, to_png_bytes
from .utils import classification_report, configure_stdio, confusion_matrix

WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_MODEL_DIR = Path("models")
MAX_BODY_BYTES = 8 * 1024 * 1024  # 8 MB is plenty for a 280x280 canvas PNG
PREVIEW_SCALE = 10  # 28x28 -> 280x280 so the browser can show it crisply

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def png_data_url(image: np.ndarray, scale: int = PREVIEW_SCALE) -> str:
    """Encode a 2-D array in ``[0, 1]`` as a ``data:image/png;base64,...`` URL."""
    arr = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if scale > 1:
        arr = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
    payload = base64.b64encode(to_png_bytes((arr * 255).round().astype(np.uint8)))
    return "data:image/png;base64," + payload.decode("ascii")


def decode_image_payload(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    """Turn an API payload into ``(flat_vector_784, image_28x28)``.

    Accepts either a base64 PNG data URL under ``"image"`` (what the canvas
    sends) or an explicit ``"pixels"`` list of 784 values in ``[0, 1]``.
    """
    center = bool(payload.get("center", True))
    if payload.get("pixels") is not None:
        pixels = np.asarray(payload["pixels"], dtype=np.float32).reshape(-1)
        if pixels.size != 28 * 28:
            raise ValueError(f"expected 784 pixels, got {pixels.size}")
        pixels = np.clip(pixels, 0.0, 1.0)
        return pixels, pixels.reshape(28, 28)

    image = payload.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError("provide either 'image' (base64 PNG) or 'pixels' (784 values)")

    encoded = image.split(",", 1)[1] if image.startswith("data:") else image
    try:
        raw = base64.b64decode(encoded, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"could not decode base64 image: {exc}") from exc

    decoded = read_png(raw)
    vector = preprocess_digit(decoded, center=center)
    return vector, vector.reshape(28, 28)


class ModelRegistry:
    """Discovers checkpoints in ``models/`` and caches the loaded models.

    Loading is lazy and guarded by a lock so the threaded server never loads
    the same checkpoint twice (or half-loads one).
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._cache: dict[str, object] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- discovery
    def names(self) -> list[str]:
        if not self.model_dir.is_dir():
            return []
        return [
            p.name
            for p in sorted(self.model_dir.iterdir())
            if p.suffix in (".npz", ".pt") and p.is_file()
        ]

    def _accuracy_from_reports(self, name: str) -> float | None:
        stem = Path(name).stem
        for candidate in (
            self.model_dir.parent / "reports" / f"{stem}_eval.json",
            self.model_dir.parent / "reports" / f"{stem}_history.json",
        ):
            if candidate.is_file():
                try:
                    data = json.loads(candidate.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                for key in ("accuracy", "test_acc"):
                    if isinstance(data.get(key), (int, float)):
                        return float(data[key])
        return None

    def info(self) -> list[dict]:
        """Metadata for every checkpoint, newest first."""
        entries = []
        for name in self.names():
            path = self.model_dir / name
            stat = path.stat()
            entries.append(
                {
                    "name": name,
                    "kind": "cnn" if path.suffix == ".pt" else "mlp",
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                    "test_accuracy": self._accuracy_from_reports(name),
                }
            )
        entries.sort(key=lambda entry: entry["modified"], reverse=True)
        default = self.default_name()
        for entry in entries:
            entry["default"] = entry["name"] == default
        return entries

    def default_name(self) -> str | None:
        names = self.names()
        for preferred in ("mlp.npz", "cnn.pt"):
            if preferred in names:
                return preferred
        return names[0] if names else None

    # -------------------------------------------------------------- loading
    def get(self, name: str | None):
        """Load (or fetch from cache) a checkpoint; raises ``KeyError`` if absent."""
        name = name or self.default_name()
        if not name:
            raise KeyError("no trained model found in models/")
        if name not in self.names():
            raise KeyError(f"unknown model {name!r}")
        with self._lock:
            if name not in self._cache:
                self._cache[name] = load_model(self.model_dir / name)
            return self._cache[name]

    def predict(self, name: str | None, vector: np.ndarray) -> tuple[np.ndarray, str]:
        """Return ``(probabilities, model_name)`` for a flat 784-vector."""
        model = self.get(name)
        probs = np.asarray(model.predict_proba(vector.reshape(1, -1)))[0]
        return probs, str(name or self.default_name())


class DatasetCache:
    """Lazily loads MNIST (only when the samples/evaluate endpoints are used)."""

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()
        self._data: tuple[np.ndarray, np.ndarray] | None = None

    def test_split(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self._data is None:
                _, _, X_test, y_test = load_mnist(self.data_dir)
                self._data = (X_test, y_test)
            return self._data


class MNISTHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying the shared registry and dataset cache."""

    daemon_threads = True
    # SO_REUSEADDR behaves differently on Windows (it lets a second socket bind
    # an already-used port, silently shadowing the first server), so enable it
    # only on POSIX where it merely avoids "address already in use" on restart.
    allow_reuse_address = os.name != "nt"

    def __init__(self, address, handler, *, registry: ModelRegistry,
                 dataset: DatasetCache, static_dir: Path, quiet: bool = False) -> None:
        super().__init__(address, handler)
        self.registry = registry
        self.dataset = dataset
        self.static_dir = Path(static_dir)
        self.quiet = quiet


class RequestHandler(BaseHTTPRequestHandler):
    """JSON API plus static file handler for the single-page UI."""

    server_version = f"mnist-toolkit/{__version__}"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------- utilities
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib signature
        if not getattr(self.server, "quiet", False):
            print(f"[web] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        self._send(json.dumps(payload).encode("utf-8"), CONTENT_TYPES[".json"], status)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ValueError("invalid Content-Length") from None
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request body too large (limit {MAX_BODY_BYTES} bytes)")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload


    # ------------------------------------------------------------ GET routes
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self._send_json(
                    {
                        "status": "ok",
                        "version": __version__,
                        "models": self.server.registry.names(),
                    }
                )
            elif parsed.path == "/api/models":
                self._send_json(
                    {
                        "models": self.server.registry.info(),
                        "default": self.server.registry.default_name(),
                    }
                )
            elif parsed.path == "/api/samples":
                self._handle_samples(query)
            elif parsed.path.startswith("/api/"):
                self._error(404, f"unknown endpoint {parsed.path!r}")
            elif parsed.path == "/favicon.ico":
                self._send(b"", "image/x-icon", 204)
            else:
                self._serve_static(parsed.path)
        except KeyError as exc:
            self._error(404, str(exc).strip("'"))
        except (ValueError, FileNotFoundError) as exc:
            self._error(400, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self._error(500, f"internal error: {exc}")

    def _handle_samples(self, query: dict) -> None:
        try:
            n = int(query.get("n", ["8"])[0])
            seed = int(query.get("seed", ["0"])[0])
        except ValueError:
            raise ValueError("'n' and 'seed' must be integers") from None
        n = max(1, min(64, n))
        X_test, y_test = self.server.dataset.test_split()
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X_test), size=min(n, len(X_test)), replace=False)
        images = to_images(X_test[indices])
        self._send_json(
            {
                "samples": [
                    {"index": int(i), "label": int(label), "image": png_data_url(image)}
                    for i, label, image in zip(indices, y_test[indices], images, strict=True)
                ]
            }
        )

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path in ("/", "/index.html") else path
        if relative.startswith("/static/"):
            relative = relative[len("/static/") :]
        else:
            relative = relative.lstrip("/")
        root = self.server.static_dir.resolve()
        candidate = (root / relative).resolve()
        # Reject path traversal: the resolved file must live inside web/.
        if candidate != root and root not in candidate.parents:
            self._error(404, "not found")
            return
        if not candidate.is_file():
            self._error(404, f"not found: {path}")
            return
        content_type = CONTENT_TYPES.get(candidate.suffix, "application/octet-stream")
        self._send(candidate.read_bytes(), content_type)


    # ----------------------------------------------------------- POST routes
    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/predict":
                self._handle_predict(payload)
            elif parsed.path == "/api/evaluate":
                self._handle_evaluate(payload)
            elif parsed.path.startswith("/api/"):
                self._error(404, f"unknown endpoint {parsed.path!r}")
            else:
                self._error(405, "only /api/predict and /api/evaluate accept POST")
        except KeyError as exc:
            self._error(404, str(exc).strip("'"))
        except (ValueError, FileNotFoundError) as exc:
            self._error(400, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self._error(500, f"internal error: {exc}")

    def _handle_predict(self, payload: dict) -> None:
        started = time.perf_counter()
        vector, image = decode_image_payload(payload)
        probs, model_name = self.server.registry.predict(payload.get("model"), vector)
        digit = int(np.argmax(probs))
        self._send_json(
            {
                "model": model_name,
                "digit": digit,
                "confidence": float(probs[digit]),
                "probs": [float(p) for p in probs],
                "preview": png_data_url(image),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )

    def _handle_evaluate(self, payload: dict) -> None:
        try:
            limit = int(payload.get("limit", 1000))
        except (TypeError, ValueError):
            raise ValueError("'limit' must be an integer") from None
        limit = max(1, min(5000, limit))
        model_name = payload.get("model") or self.server.registry.default_name()
        model = self.server.registry.get(model_name)

        X_test, y_test = self.server.dataset.test_split()
        X, y = X_test[:limit], y_test[:limit]
        probs = np.asarray(model.predict_proba(X))
        preds = np.argmax(probs, axis=1)
        n_classes = int(getattr(model, "n_classes", probs.shape[1]))
        loss = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())
        cm = confusion_matrix(y, preds, n_classes)
        per_class = {
            str(c): (float(cm[c, c] / cm[c].sum()) if cm[c].sum() else 0.0)
            for c in range(n_classes)
        }
        self._send_json(
            {
                "model": model_name,
                "n": int(len(y)),
                "accuracy": float((preds == y).mean()),
                "loss": loss,
                "confusion_matrix": cm.tolist(),
                "per_class_accuracy": per_class,
                "report": classification_report(y, preds, n_classes),
            }
        )



def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    static_dir: str | Path = WEB_DIR,
    quiet: bool = False,
) -> MNISTHTTPServer:
    """Build (but do not start) a configured server; handy for tests."""
    return MNISTHTTPServer(
        (host, port),
        RequestHandler,
        registry=ModelRegistry(model_dir),
        dataset=DatasetCache(data_dir),
        static_dir=Path(static_dir),
        quiet=quiet,
    )


def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Serve the MNIST web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8000, help="bind port (0 = random free port)")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--quiet", action="store_true", help="suppress request logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)
    httpd = create_server(
        args.host, args.port, model_dir=args.model_dir, data_dir=args.data_dir, quiet=args.quiet
    )
    host, port = httpd.server_address[:2]
    models = httpd.registry.names()
    print(f"MNIST web UI  ->  http://{host}:{port}")
    if models:
        print(f"models available: {', '.join(models)}")
    else:
        print("no models found — train one first:  python -m mnist train")
    print("press Ctrl+C to stop", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())

