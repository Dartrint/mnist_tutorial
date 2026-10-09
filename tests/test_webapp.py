"""End-to-end tests for the stdlib web UI (real HTTP requests, no mocks)."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from mnist.data import load_mnist
from mnist.imageio import to_png_bytes
from mnist.model import MLP
from mnist.webapp import WEB_DIR, create_server, png_data_url

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def data_available() -> bool:
    return (DATA_DIR / "t10k-images-idx3-ubyte.gz").is_file()


class WebAppTestCase(unittest.TestCase):
    """Boots a real server on an ephemeral port with a freshly trained model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = Path(tempfile.mkdtemp(prefix="mnist-web-test-"))
        model_dir = cls._tmp / "models"
        model_dir.mkdir()

        # Train a small model so predictions are meaningful (not random weights).
        cls.X_probe, cls.y_probe = None, None
        if data_available():
            X_train, y_train, X_test, y_test = load_mnist(DATA_DIR)
            cls.X_probe, cls.y_probe = X_train[:40], y_train[:40]
            X_fit, y_fit = X_train[:3000], y_train[:3000]
        else:  # pragma: no cover - only on a machine without data/
            rng = np.random.default_rng(0)
            X_fit = rng.random((300, 784), dtype=np.float32)
            y_fit = rng.integers(0, 10, size=300)

        model = MLP(hidden_sizes=(64,), seed=0)
        from mnist import nn

        optimizer = nn.Adam(model.trainable_layers(), lr=0.01)
        rng = np.random.default_rng(0)
        for _ in range(4):
            for start in range(0, len(X_fit), 128):
                xb, yb = X_fit[start : start + 128], y_fit[start : start + 128]
                _, dlogits = model.loss(xb, yb, training=True)
                model.backward(dlogits)
                optimizer.step()
            rng.permutation(len(X_fit))
        model.save(model_dir / "tiny.npz")

        cls.server = create_server(
            "127.0.0.1", 0, model_dir=model_dir, data_dir=DATA_DIR, static_dir=WEB_DIR, quiet=True
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        shutil.rmtree(cls._tmp, ignore_errors=True)

    # ------------------------------------------------------------- helpers
    def request(self, path: str, payload: dict | None = None, method: str | None = None):
        url = self.base + path
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                status = resp.status
                ctype = resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            body, status, ctype = exc.read(), exc.code, exc.headers.get("Content-Type", "")
        parsed = json.loads(body) if "json" in ctype else body
        return status, parsed, ctype


class TestApi(WebAppTestCase):
    def test_health(self) -> None:
        status, body, _ = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertIn("tiny.npz", body["models"])
        self.assertTrue(body["version"])

    def test_models_lists_checkpoint_with_metadata(self) -> None:
        status, body, _ = self.request("/api/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["default"], "tiny.npz")
        entry = body["models"][0]
        self.assertEqual(entry["name"], "tiny.npz")
        self.assertEqual(entry["kind"], "mlp")
        self.assertGreater(entry["size_bytes"], 0)
        self.assertTrue(entry["default"])

    def test_predict_with_pixels(self) -> None:
        pixels = np.zeros(784, dtype=np.float32).tolist()
        status, body, _ = self.request("/api/predict", {"pixels": pixels})
        self.assertEqual(status, 200)
        self.assertIn(body["digit"], range(10))
        self.assertEqual(len(body["probs"]), 10)
        self.assertAlmostEqual(sum(body["probs"]), 1.0, places=5)
        self.assertAlmostEqual(body["confidence"], body["probs"][body["digit"]], places=6)
        self.assertTrue(body["preview"].startswith("data:image/png;base64,"))
        self.assertEqual(body["model"], "tiny.npz")

    def test_predict_with_png_data_url(self) -> None:
        image = np.zeros((28, 28), dtype=np.float32)
        image[8:20, 10:18] = 1.0
        payload = {"image": png_data_url(image, scale=1), "center": False}
        status, body, _ = self.request("/api/predict", payload)
        self.assertEqual(status, 200)
        # The same input sent as raw pixels must produce identical probabilities.
        status2, body2, _ = self.request("/api/predict", {"pixels": image.reshape(-1).tolist()})
        self.assertEqual(status2, 200)
        np.testing.assert_allclose(body["probs"], body2["probs"], atol=1e-6)

    @unittest.skipUnless(data_available(), "MNIST data not downloaded")
    def test_predict_is_accurate_on_known_digits(self) -> None:
        self.assertIsNotNone(self.X_probe)
        vectors = self.X_probe.reshape(len(self.X_probe), -1)
        correct = 0
        for vector, label in zip(vectors, self.y_probe, strict=True):
            status, body, _ = self.request("/api/predict", {"pixels": vector.tolist()})
            self.assertEqual(status, 200)
            correct += int(body["digit"] == int(label))
        self.assertGreaterEqual(correct, len(vectors) - 2)

    def test_predict_rejects_missing_payload(self) -> None:
        status, body, _ = self.request("/api/predict", {})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_predict_rejects_wrong_pixel_count(self) -> None:
        status, body, _ = self.request("/api/predict", {"pixels": [0.0] * 10})
        self.assertEqual(status, 400)
        self.assertIn("784", body["error"])

    def test_predict_rejects_unknown_model(self) -> None:
        status, body, _ = self.request("/api/predict", {"model": "../../etc/passwd", "pixels": [0.0] * 784})
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    def test_predict_rejects_corrupt_image(self) -> None:
        status, body, _ = self.request("/api/predict", {"image": "data:image/png;base64,bm90YXBuZw=="})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_unknown_api_endpoint_is_json_404(self) -> None:
        status, body, ctype = self.request("/api/does-not-exist")
        self.assertEqual(status, 404)
        self.assertIn("json", ctype)
        self.assertIn("error", body)

    def test_post_to_root_is_rejected(self) -> None:
        status, body, _ = self.request("/", {"anything": 1})
        self.assertEqual(status, 405)
        self.assertIn("error", body)

    def test_invalid_json_body(self) -> None:
        req = urllib.request.Request(
            self.base + "/api/predict", data=b"{not json", method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status, body = resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            status, body = exc.code, json.loads(exc.read())
        self.assertEqual(status, 400)
        self.assertIn("invalid JSON", body["error"])



class TestStaticFiles(WebAppTestCase):
    def test_index_is_served_at_root(self) -> None:
        status, body, ctype = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        html = body.decode("utf-8")
        self.assertIn("/static/app.js", html)
        self.assertIn("/static/style.css", html)

    def test_javascript_and_css_are_served(self) -> None:
        for path, needle in (("/static/app.js", "fetch"), ("/static/style.css", "{")):
            status, body, ctype = self.request(path)
            self.assertEqual(status, 200, path)
            self.assertIn(needle, body.decode("utf-8"), path)
            self.assertIn("javascript" if path.endswith(".js") else "css", ctype)

    def test_missing_asset_is_404(self) -> None:
        status, body, _ = self.request("/static/nope.js")
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    def test_path_traversal_is_blocked(self) -> None:
        for path in ("/static/../webapp.py", "/../pyproject.toml"):
            status, body, _ = self.request(path)
            self.assertEqual(status, 404, path)
            self.assertIn("error", body)

    def test_favicon_is_empty_204(self) -> None:
        status, _, _ = self.request("/favicon.ico")
        self.assertEqual(status, 204)


class TestDatasetEndpoints(WebAppTestCase):
    @unittest.skipUnless(data_available(), "MNIST data not downloaded")
    def test_samples_returns_labelled_images(self) -> None:
        status, body, _ = self.request("/api/samples?n=5&seed=7")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["samples"]), 5)
        for sample in body["samples"]:
            self.assertIn(sample["label"], range(10))
            self.assertGreaterEqual(sample["index"], 0)
            self.assertTrue(sample["image"].startswith("data:image/png;base64,"))

    @unittest.skipUnless(data_available(), "MNIST data not downloaded")
    def test_samples_are_deterministic_for_a_seed(self) -> None:
        _, first, _ = self.request("/api/samples?n=4&seed=123")
        _, second, _ = self.request("/api/samples?n=4&seed=123")
        self.assertEqual([s["index"] for s in first["samples"]],
                         [s["index"] for s in second["samples"]])
        _, other, _ = self.request("/api/samples?n=4&seed=321")
        self.assertNotEqual([s["index"] for s in first["samples"]],
                            [s["index"] for s in other["samples"]])

    @unittest.skipUnless(data_available(), "MNIST data not downloaded")
    def test_samples_clamps_count(self) -> None:
        _, body, _ = self.request("/api/samples?n=999")
        self.assertLessEqual(len(body["samples"]), 64)

    def test_samples_rejects_bad_query(self) -> None:
        status, body, _ = self.request("/api/samples?n=abc")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    @unittest.skipUnless(data_available(), "MNIST data not downloaded")
    def test_evaluate_returns_metrics(self) -> None:
        status, body, _ = self.request("/api/evaluate", {"model": "tiny.npz", "limit": 300})
        self.assertEqual(status, 200)
        self.assertEqual(body["n"], 300)
        self.assertGreaterEqual(body["accuracy"], 0.0)
        self.assertLessEqual(body["accuracy"], 1.0)
        self.assertEqual(np.array(body["confusion_matrix"]).shape, (10, 10))
        self.assertEqual(len(body["per_class_accuracy"]), 10)
        self.assertIn("accuracy:", body["report"])

    @unittest.skipUnless(data_available(), "MNIST data not downloaded")
    def test_evaluate_clamps_limit(self) -> None:
        _, body, _ = self.request("/api/evaluate", {"limit": 99999})
        self.assertLessEqual(body["n"], 5000)

    def test_evaluate_rejects_bad_limit(self) -> None:
        status, body, _ = self.request("/api/evaluate", {"limit": "many"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)



class TestRegistryUnit(unittest.TestCase):
    """Registry behaviour that is easier to check without HTTP."""

    def test_empty_model_dir(self) -> None:
        from mnist.webapp import ModelRegistry

        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(tmp)
            self.assertEqual(registry.names(), [])
            self.assertIsNone(registry.default_name())
            self.assertEqual(registry.info(), [])
            with self.assertRaises(KeyError):
                registry.get(None)

    def test_missing_model_dir_is_not_an_error(self) -> None:
        from mnist.webapp import ModelRegistry

        self.assertEqual(ModelRegistry("/definitely/not/here").names(), [])

    def test_prefers_mlp_then_cnn(self) -> None:
        from mnist.webapp import ModelRegistry

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            model_dir.mkdir()
            (model_dir / "cnn.pt").write_bytes(b"stub")
            (model_dir / "mlp.npz").write_bytes(b"stub")
            registry = ModelRegistry(model_dir)
            self.assertEqual(registry.default_name(), "mlp.npz")
            (model_dir / "mlp.npz").unlink()
            self.assertEqual(registry.default_name(), "cnn.pt")
            kinds = {entry["name"]: entry["kind"] for entry in registry.info()}
            self.assertEqual(kinds["cnn.pt"], "cnn")

    def test_accuracy_is_read_from_reports(self) -> None:
        from mnist.webapp import ModelRegistry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
            (root / "reports").mkdir()
            (root / "models" / "mlp.npz").write_bytes(b"stub")
            (root / "reports" / "mlp_eval.json").write_text(json.dumps({"accuracy": 0.9747}))
            registry = ModelRegistry(root / "models")
            self.assertAlmostEqual(registry.info()[0]["test_accuracy"], 0.9747)


class TestHelpers(unittest.TestCase):
    def test_png_data_url_is_decodable(self) -> None:
        import base64

        from mnist.imageio import read_png

        image = np.zeros((28, 28), dtype=np.float32)
        image[10:18, 10:18] = 1.0
        url = png_data_url(image, scale=10)
        self.assertTrue(url.startswith("data:image/png;base64,"))
        decoded = read_png(base64.b64decode(url.split(",", 1)[1]))
        self.assertEqual(decoded.shape, (280, 280))
        self.assertEqual(int(decoded.max()), 255)

    def test_decode_rejects_bad_pixels(self) -> None:
        from mnist.webapp import decode_image_payload

        with self.assertRaises(ValueError):
            decode_image_payload({"pixels": [1, 2, 3]})

    def test_decode_accepts_plain_base64_without_data_prefix(self) -> None:
        import base64

        from mnist.webapp import decode_image_payload

        image = (np.zeros((28, 28), dtype=np.float32))
        image[5:20, 8:16] = 1.0
        raw = base64.b64encode(to_png_bytes((image * 255).astype(np.uint8))).decode()
        vector, preview = decode_image_payload({"image": raw})
        self.assertEqual(vector.shape, (784,))
        self.assertEqual(preview.shape, (28, 28))


if __name__ == "__main__":
    unittest.main()

