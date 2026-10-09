"""Tests for the cross-platform task runner (the Windows twin of the Makefile)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tasks.py"


def load_tasks_module():
    spec = importlib.util.spec_from_file_location("mnist_tasks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTaskRunnerCli(unittest.TestCase):
    def run_script(self, *args: str, cwd: Path | None = None, env: dict | None = None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=str(cwd or ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, **(env or {})},
        )

    def test_no_arguments_prints_usage_and_fails(self) -> None:
        result = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)
        self.assertIn("clean", result.stderr)

    def test_help_prints_usage_and_succeeds(self) -> None:
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout)

    def test_unknown_task_fails_with_message(self) -> None:
        result = self.run_script("definitely-not-a-task")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown task", result.stderr)

    def test_runs_from_any_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_script("--help", cwd=Path(tmp))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)


class TestTaskMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = load_tasks_module()

    def test_every_task_is_documented(self) -> None:
        usage = self.tasks._usage()
        for name in self.tasks.TASKS:
            self.assertIn(name, usage)

    def test_commands_use_the_current_interpreter(self) -> None:
        for name, (_, argv) in self.tasks.TASKS.items():
            if argv:  # 'clean' is implemented in Python, not a subprocess
                self.assertEqual(argv[0], sys.executable, name)

    def test_module_tasks_go_through_python_m_mnist(self) -> None:
        for name in ("data", "train", "cnn", "evaluate", "predict", "serve"):
            argv = self.tasks.TASKS[name][1]
            self.assertEqual(argv[1:3], ["-m", "mnist"], name)
            self.assertEqual(argv[3], name, name)

    def test_extra_arguments_are_appended(self) -> None:
        captured = {}

        def fake_call(command, cwd=None):
            captured["command"], captured["cwd"] = command, cwd
            return 0

        original = self.tasks.subprocess.call
        self.tasks.subprocess.call = fake_call
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                code = self.tasks.main(["train", "--epochs", "2", "--hidden", "64"])
        finally:
            self.tasks.subprocess.call = original

        self.assertEqual(code, 0)
        self.assertEqual(captured["command"][-4:], ["--epochs", "2", "--hidden", "64"])
        self.assertEqual(Path(captured["cwd"]), ROOT)

    def test_missing_interpreter_is_reported(self) -> None:
        def boom(command, cwd=None):
            raise FileNotFoundError("nope")

        original = self.tasks.subprocess.call
        self.tasks.subprocess.call = boom
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = self.tasks.main(["data"])
        finally:
            self.tasks.subprocess.call = original
        self.assertEqual(code, 127)
        self.assertIn("could not start", err.getvalue())


class TestCleanTask(unittest.TestCase):
    """`clean` must be safe: it deletes caches and generated PNGs, never models/data."""

    @staticmethod
    def _fixture(root: Path) -> None:
        (root / "mnist" / "__pycache__").mkdir(parents=True)
        (root / "mnist" / "__pycache__" / "x.pyc").write_bytes(b"cache")
        (root / ".pytest_cache").mkdir()
        (root / "reports").mkdir()
        (root / "reports" / "predictions.png").write_bytes(b"png")
        (root / "reports" / "mlp_eval.json").write_text("{}")
        (root / "models").mkdir()
        (root / "models" / "mlp.npz").write_bytes(b"weights")
        (root / "data").mkdir()
        (root / "data" / "train.gz").write_bytes(b"data")

    def _run_clean(self, root: Path):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "clean"],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "MNIST_TASKS_ROOT": str(root)},
        )

    def test_clean_removes_caches_and_pngs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            result = self._run_clean(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "mnist" / "__pycache__").exists())
            self.assertFalse((root / ".pytest_cache").exists())
            self.assertFalse((root / "reports" / "predictions.png").exists())
            self.assertTrue((root / "reports" / "mlp_eval.json").is_file())
            self.assertTrue((root / "models" / "mlp.npz").is_file())
            self.assertTrue((root / "data" / "train.gz").is_file())

    def test_clean_on_empty_tree_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_clean(Path(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("cleaned 0 item(s)", result.stdout)


class TestMakefileParity(unittest.TestCase):
    """Every Makefile target should have a tasks.py equivalent."""

    def test_targets_are_covered(self) -> None:
        targets = set()
        for line in (ROOT / "Makefile").read_text().splitlines():
            if line and not line.startswith(("\t", " ", "#", ".")) and ":" in line:
                targets.add(line.split(":", 1)[0].strip())
        tasks = set(load_tasks_module().TASKS)
        # `help` is the Makefile's own self-documentation target.
        self.assertTrue(targets - {"help"} <= tasks, sorted(targets - {"help"} - tasks))


if __name__ == "__main__":
    unittest.main()

