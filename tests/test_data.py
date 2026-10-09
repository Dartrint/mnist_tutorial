"""Tests for the IDX parser and dataset helpers."""

from __future__ import annotations

import gzip
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mnist import data as data_mod


def write_idx(path: Path, array: np.ndarray) -> None:
    """Write a synthetic IDX file so the parser can be tested offline."""
    magic = 0x0800 | array.ndim  # 0x0803 for 3-D images, 0x0801 for 1-D labels
    header = struct.pack(">I", magic) + struct.pack(f">{array.ndim}I", *array.shape)
    with gzip.open(path, "wb") as fh:
        fh.write(header)
        fh.write(array.astype(np.uint8).tobytes())


class TestIdxParser(unittest.TestCase):
    def test_reads_images_and_labels(self) -> None:
        images = np.arange(3 * 28 * 28, dtype=np.uint8).reshape(3, 28, 28)
        labels = np.array([1, 5, 9], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "images"
            lbl_path = Path(tmp) / "labels"
            write_idx(img_path, images)
            write_idx(lbl_path, labels)
            np.testing.assert_array_equal(data_mod._read_idx(img_path), images)
            np.testing.assert_array_equal(data_mod._read_idx(lbl_path), labels)

    def test_roundtrip_through_load_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in data_mod.FILES:
                images = np.zeros((4, 28, 28), dtype=np.uint8)
                if "images" in name:
                    write_idx(tmp_path / name, images)
                else:
                    write_idx(tmp_path / name, np.zeros(4, dtype=np.uint8))
            X, y = data_mod.load_split("train", tmp_path, flatten=True)
            self.assertEqual(X.shape, (4, 784))
            self.assertEqual(X.dtype, np.float32)
            self.assertEqual(y.shape, (4,))
            X_img, _ = data_mod.load_split("test", tmp_path, flatten=False)
            self.assertEqual(X_img.shape, (4, 28, 28))

    def test_normalisation_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            images = np.full((2, 28, 28), 255, dtype=np.uint8)
            write_idx(tmp_path / "train-images-idx3-ubyte.gz", images)
            write_idx(tmp_path / "train-labels-idx1-ubyte.gz", np.zeros(2, dtype=np.uint8))
            write_idx(tmp_path / "t10k-images-idx3-ubyte.gz", images)
            write_idx(tmp_path / "t10k-labels-idx1-ubyte.gz", np.zeros(2, dtype=np.uint8))
            X, _ = data_mod.load_split("train", tmp_path)
            np.testing.assert_allclose(X, 1.0)


class TestHelpers(unittest.TestCase):
    def test_to_images_shape(self) -> None:
        X = np.zeros((5, 784), dtype=np.float32)
        self.assertEqual(data_mod.to_images(X).shape, (5, 28, 28))

    def test_load_split_rejects_unknown_split(self) -> None:
        X = np.zeros((2, 784), dtype=np.float32)
        y = np.zeros(2, dtype=np.int64)
        with mock.patch.object(data_mod, "load_mnist", return_value=(X, y, X, y)):
            with self.assertRaises(ValueError):
                data_mod.load_split("validation")

    def test_md5_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob"
            path.write_bytes(b"hello")
            self.assertEqual(data_mod._md5(path), "5d41402abc4b2a76b9719d911017c592")


class TestRealDataset(unittest.TestCase):
    """Only runs when the MNIST files are already present locally."""

    def setUp(self) -> None:
        self.data_dir = Path(__file__).resolve().parents[1] / "data"
        if not all((self.data_dir / name).exists() for name in data_mod.FILES):
            self.skipTest("MNIST files not downloaded (run `python -m mnist.data`)")

    def test_shapes_and_labels(self) -> None:
        X_train, y_train, X_test, y_test = data_mod.load_mnist(self.data_dir)
        self.assertEqual(X_train.shape, (60000, 784))
        self.assertEqual(X_test.shape, (10000, 784))
        self.assertEqual(y_train.min(), 0)
        self.assertEqual(y_train.max(), 9)
        self.assertGreaterEqual(X_train.min(), 0.0)
        self.assertLessEqual(X_train.max(), 1.0)


if __name__ == "__main__":
    unittest.main()
