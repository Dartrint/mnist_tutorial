"""Tests for the unified CLI dispatcher and argument parsing."""

from __future__ import annotations

import contextlib
import io
import unittest

from mnist import __main__ as dispatcher
from mnist.data import main as data_main
from mnist.evaluate import parse_args as evaluate_args
from mnist.predict import parse_args as predict_args
from mnist.train import parse_args as train_args


class TestDispatcher(unittest.TestCase):
    def test_help_lists_every_command(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(dispatcher.main(["--help"]), 0)
        text = out.getvalue()
        for name in dispatcher.COMMANDS:
            self.assertIn(name, text)

    def test_no_arguments_is_usage_error(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(dispatcher.main([]), 2)
        self.assertIn("usage:", err.getvalue())

    def test_unknown_command(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(dispatcher.main(["nope"]), 2)
        self.assertIn("unknown command", err.getvalue())

    def test_data_command_returns_zero(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(dispatcher.main(["data", "--quiet"]), 0)
        self.assertIn("MNIST ready", out.getvalue())


class TestArgumentParsing(unittest.TestCase):
    def test_train_defaults(self) -> None:
        args = train_args([])
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.optimizer, "sgd")
        self.assertEqual(args.out, "models/mlp.npz")

    def test_train_hidden_and_optimizer_parsing(self) -> None:
        args = train_args(["--hidden", "512,256", "--optimizer", "adam", "--lr", "0.001"])
        self.assertEqual(tuple(int(h) for h in args.hidden.split(",")), (512, 256))
        self.assertEqual(args.optimizer, "adam")
        self.assertAlmostEqual(args.lr, 0.001)

    def test_evaluate_and_predict_defaults(self) -> None:
        self.assertEqual(evaluate_args([]).split, "test")
        self.assertEqual(predict_args([]).invert, "auto")

    def test_data_rejects_unknown_flag(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            data_main(["--definitely-not-a-flag"])


if __name__ == "__main__":
    unittest.main()
