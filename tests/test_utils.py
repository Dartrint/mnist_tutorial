"""Tests for the shared helpers in :mod:`mnist.utils`."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from mnist import utils


class TestSeeding(unittest.TestCase):
    def test_set_seed_makes_runs_reproducible(self) -> None:
        first = utils.set_seed(42).random(3)
        second = utils.set_seed(42).random(3)
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_differ(self) -> None:
        self.assertFalse(np.array_equal(utils.set_seed(1).random(3), utils.set_seed(2).random(3)))


class TestLabelsAndMetrics(unittest.TestCase):
    def test_one_hot(self) -> None:
        encoded = utils.one_hot(np.array([0, 3, 9]), n_classes=10)
        self.assertEqual(encoded.shape, (3, 10))
        np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(3, dtype=np.float32))
        self.assertEqual(encoded[1, 3], 1.0)

    def test_accuracy_uses_argmax(self) -> None:
        scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
        self.assertAlmostEqual(utils.accuracy(scores, np.array([0, 1, 1])), 2 / 3)

    def test_confusion_matrix_layout(self) -> None:
        cm = utils.confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]), n_classes=2)
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm[0, 0], 1)  # true 0 predicted 0
        self.assertEqual(cm[0, 1], 1)  # true 0 predicted 1
        self.assertEqual(cm[1, 1], 1)
        self.assertEqual(int(cm.sum()), 3)

    def test_classification_report_contents(self) -> None:
        y_true = np.array([0, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2])
        report = utils.classification_report(y_true, y_pred, n_classes=3)
        self.assertIn("precision", report)
        self.assertIn("accuracy:", report)
        self.assertIn("0.7500", report)  # 3 of 4 correct


class TestMeterAndTimer(unittest.TestCase):
    def test_average_meter_weights_by_count(self) -> None:
        meter = utils.AverageMeter()
        meter.update(1.0, n=3)
        meter.update(2.0, n=1)
        self.assertAlmostEqual(meter.avg, 1.25)

    def test_average_meter_is_zero_when_empty(self) -> None:
        self.assertEqual(utils.AverageMeter().avg, 0.0)

    def test_timer_records_elapsed_time(self) -> None:
        with utils.Timer() as timer:
            time.sleep(0.02)
        self.assertGreaterEqual(timer.elapsed, 0.02)


class TestSerialisation(unittest.TestCase):
    def test_save_json_round_trip(self) -> None:
        payload = {"accuracy": 0.9747, "epochs": [1, 2, 3], "nested": {"a": None}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            utils.save_json(payload, path)
            self.assertEqual(json.loads(path.read_text()), payload)


if __name__ == "__main__":
    unittest.main()
