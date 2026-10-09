"""Tests for the dependency-free image codec and digit preprocessing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mnist import imageio


class TestPngRoundTrip(unittest.TestCase):
    def test_grayscale(self) -> None:
        rng = np.random.default_rng(0)
        image = (rng.random((28, 28)) * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = imageio.write_png(Path(tmp) / "g.png", image)
            loaded = imageio.read_png(path)
        self.assertEqual(loaded.shape, image.shape)
        np.testing.assert_array_equal(loaded, image)

    def test_rgb_and_rgba(self) -> None:
        rng = np.random.default_rng(1)
        for channels in (3, 4):
            image = (rng.random((10, 7, channels)) * 255).astype(np.uint8)
            with tempfile.TemporaryDirectory() as tmp:
                path = imageio.write_png(Path(tmp) / f"c{channels}.png", image)
                loaded = imageio.read_png(path)
            np.testing.assert_array_equal(loaded, image)

    def test_flat_image_compresses(self) -> None:
        image = np.zeros((64, 64), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = imageio.write_png(Path(tmp) / "flat.png", image)
            self.assertLess(path.stat().st_size, 200)
            np.testing.assert_array_equal(imageio.read_png(path), image)

    def test_rejects_unsupported_ndim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                imageio.write_png(Path(tmp) / "bad.png", np.zeros((4, 4, 2), dtype=np.uint8))

    def test_rejects_non_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not.png"
            path.write_bytes(b"definitely not a png")
            with self.assertRaises(ValueError):
                imageio.read_png(path)


class TestPgmAndDispatch(unittest.TestCase):
    def _write_pgm(self, path: Path, array: np.ndarray) -> None:
        header = f"P5\n# a comment\n{array.shape[1]} {array.shape[0]}\n255\n".encode()
        path.write_bytes(header + array.astype(np.uint8).tobytes())

    def test_read_pgm_with_comment(self) -> None:
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.pgm"
            self._write_pgm(path, image)
            np.testing.assert_array_equal(imageio.read_pgm(path), image)
            np.testing.assert_array_equal(imageio.read_image(path), image)

    def test_read_npy(self) -> None:
        image = np.arange(16, dtype=np.uint8).reshape(4, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.npy"
            np.save(path, image)
            np.testing.assert_array_equal(imageio.read_image(path), image)

    def test_rgb_png_is_converted_to_grayscale(self) -> None:
        colour = np.zeros((5, 5, 3), dtype=np.uint8)
        colour[..., 0] = 255  # pure red -> luma ~76
        with tempfile.TemporaryDirectory() as tmp:
            path = imageio.write_png(Path(tmp) / "red.png", colour)
            grey = imageio.read_image(path)
        self.assertEqual(grey.shape, (5, 5))
        self.assertAlmostEqual(int(grey[0, 0]), 76, delta=2)

    def test_unknown_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                imageio.read_image(Path(tmp) / "whatever.bmp")


class TestResizeAndPrepare(unittest.TestCase):
    def test_resize_shapes(self) -> None:
        image = np.zeros((28, 28), dtype=np.uint8)
        self.assertEqual(imageio.resize_nearest(image, 14).shape, (14, 14))
        self.assertEqual(imageio.resize_nearest(image, 56).shape, (56, 56))

    def test_resize_rejects_non_2d(self) -> None:
        with self.assertRaises(ValueError):
            imageio.resize_nearest(np.zeros((4, 4, 3)), 8)

    def test_prepare_digit_infers_polarity(self) -> None:
        # Dark strokes on a light background must be inverted to MNIST layout.
        light_bg = np.full((28, 28), 255, dtype=np.uint8)
        light_bg[10:18, 10:18] = 0
        vector = imageio.prepare_digit(light_bg)
        self.assertEqual(vector.shape, (784,))
        self.assertLessEqual(vector.max(), 1.0)
        self.assertGreaterEqual(vector.min(), 0.0)
        self.assertGreater(vector.mean(), 0.0)  # stroke became the bright part

    def test_prepare_digit_respects_explicit_invert(self) -> None:
        image = np.zeros((28, 28), dtype=np.uint8)
        image[10:18, 10:18] = 255
        kept = imageio.prepare_digit(image, invert=False)
        flipped = imageio.prepare_digit(image, invert=True)
        np.testing.assert_allclose(kept, 1.0 - flipped, atol=1e-6)

    def test_ascii_render(self) -> None:
        image = np.zeros((8, 8), dtype=np.uint8)
        art = imageio.to_ascii(image)
        self.assertEqual(len(art.splitlines()), 8)
        self.assertEqual(set(art) - {"\n"}, {" "})


if __name__ == "__main__":
    unittest.main()
