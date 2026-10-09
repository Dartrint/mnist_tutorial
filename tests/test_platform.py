"""Cross-platform / Windows robustness checks.

These guard against regressions that only show up on Windows: legacy console
code pages that cannot encode Vietnamese text, and SO_REUSEADDR semantics.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import unittest
from pathlib import Path

from mnist.utils import configure_stdio
from mnist.webapp import MNISTHTTPServer

ROOT = Path(__file__).resolve().parents[1]


class TestConsoleEncoding(unittest.TestCase):
    """The CLI must degrade, never crash, on a code page without Vietnamese."""

    def test_configure_stdio_accepts_injected_streams(self) -> None:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp437", errors="strict")
        configure_stdio([stream])
        stream.write("Chạy — dự đoán ảnh")  # would raise without errors="replace"
        stream.flush()
        self.assertIn(b"?", buffer.getvalue())

    def test_configure_stdio_tolerates_plain_objects(self) -> None:
        class Odd:
            def write(self, _text):  # no reconfigure() at all
                return 0

        configure_stdio([Odd()])  # must not raise

    def test_help_survives_legacy_code_pages(self) -> None:
        for encoding in ("cp437", "cp1258", "ascii"):
            with self.subTest(encoding=encoding):
                result = subprocess.run(
                    [sys.executable, "-m", "mnist", "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    env={**os.environ, "PYTHONIOENCODING": encoding},
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
                self.assertNotIn(b"UnicodeEncodeError", result.stderr)

    def test_train_help_survives_ascii_only_console(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mnist", "train", "--help"],
            cwd=ROOT,
            capture_output=True,
            env={**os.environ, "PYTHONIOENCODING": "ascii"},
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))


class TestWindowsSpecifics(unittest.TestCase):
    def test_reuse_address_matches_platform(self) -> None:
        # On Windows SO_REUSEADDR lets a second server hijack a live port.
        self.assertEqual(MNISTHTTPServer.allow_reuse_address, os.name != "nt")

    def test_no_posix_only_imports(self) -> None:
        banned = ("import fcntl", "import termios", "import pwd", "import grp", "os.fork")
        for path in sorted((ROOT / "mnist").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for needle in banned:
                self.assertNotIn(needle, text, f"{path.name} uses POSIX-only {needle!r}")

    def test_no_hardcoded_tmp_paths_in_sources(self) -> None:
        # Built at runtime so that this file does not match its own pattern.
        needle = '"' + "/" + "tmp/"
        for folder in ("mnist", "tests", "examples", "scripts"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and path.suffix in (".py", ".mjs"):
                    text = path.read_text(encoding="utf-8")
                    self.assertNotIn(needle, text, f"{path} hard-codes a POSIX temp path")

    def test_paths_are_built_with_pathlib(self) -> None:
        # Windows separators must not leak into stored config or the CLI output.
        import tempfile

        from mnist.model import MLP

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "model.npz"
            MLP(hidden_sizes=(4,), seed=0).save(target)
            self.assertTrue(target.is_file())
            self.assertEqual(MLP.load(target).config()["hidden_sizes"], [4])


if __name__ == "__main__":
    unittest.main()
