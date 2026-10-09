"""Tiny dependency-free image I/O (PNG read/write, PGM read) + resizing.

Only what the MNIST tooling needs: 8-bit grayscale/RGB/RGBA PNGs without
interlacing, plus binary PGM. Everything is implemented on top of ``zlib`` and
``struct`` from the standard library so the project needs no Pillow.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_png(source: str | Path | bytes | bytearray) -> np.ndarray:
    """Decode an 8-bit non-interlaced PNG from a path or raw bytes.

    Returns ``(H, W)`` for grayscale or ``(H, W, C)`` for colour images.
    """
    if isinstance(source, (bytes, bytearray)):
        raw, label = bytes(source), "<bytes>"
    else:
        raw, label = Path(source).read_bytes(), str(source)
    if raw[:8] != _PNG_SIG:
        raise ValueError(f"{label}: not a PNG file")
    pos, idat = 8, bytearray()
    width = height = depth = ctype = interlace = 0
    while pos < len(raw):
        (length,) = struct.unpack(">I", raw[pos : pos + 4])
        ctag = raw[pos + 4 : pos + 8]
        body = raw[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctag == b"IHDR":
            width, height, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif ctag == b"IDAT":
            idat += body
        elif ctag == b"IEND":
            break
    if depth != 8:
        raise ValueError(f"{label}: only 8-bit PNGs are supported (got {depth})")
    if interlace:
        raise ValueError(f"{label}: interlaced PNGs are not supported")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(ctype)
    if channels is None:
        raise ValueError(f"{label}: unsupported PNG colour type {ctype}")

    data = zlib.decompress(bytes(idat))
    stride = width * channels
    out = np.empty((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    offset = 0
    for row in range(height):
        ftype = data[offset]
        offset += 1
        line = np.frombuffer(data[offset : offset + stride], dtype=np.uint8).copy()
        offset += stride
        if ftype == 0:
            pass
        elif ftype == 1:
            for i in range(channels, stride):
                line[i] = (int(line[i]) + int(line[i - channels])) & 0xFF
        elif ftype == 2:
            line = (line.astype(np.int32) + prev.astype(np.int32)).astype(np.uint8)
        elif ftype == 3:
            for i in range(stride):
                left = int(line[i - channels]) if i >= channels else 0
                line[i] = (int(line[i]) + (left + int(prev[i])) // 2) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                left = int(line[i - channels]) if i >= channels else 0
                upleft = int(prev[i - channels]) if i >= channels else 0
                line[i] = (int(line[i]) + _paeth(left, int(prev[i]), upleft)) & 0xFF
        else:
            raise ValueError(f"{label}: unknown PNG filter {ftype}")
        out[row] = line
        prev = line

    img = out.reshape(height, width, channels)
    return img[:, :, 0] if channels == 1 else img


def _chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a uint8 grayscale or RGB(A) array as an 8-bit PNG in memory."""
    return _PNG_SIG + b"".join(_png_chunks(image))


def write_png(path: str | Path, image: np.ndarray) -> Path:
    """Write a uint8 grayscale or RGB(A) array as an 8-bit PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_png_bytes(image))
    return path


def _png_chunks(image: np.ndarray) -> list[bytes]:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        height, width = arr.shape
        channels, ctype = 1, 0
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        height, width, channels = arr.shape
        ctype = {3: 2, 4: 6}[channels]
    else:
        raise ValueError("image must be (H, W) or (H, W, 3|4)")
    lines = b"".join(b"\x00" + arr[r].tobytes() for r in range(height))
    return [
        _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, ctype, 0, 0, 0)),
        _chunk(b"IDAT", zlib.compress(lines, 9)),
        _chunk(b"IEND", b""),
    ]


def read_pgm(path: str | Path) -> np.ndarray:
    """Read a binary PGM (``P5``) file into a ``(H, W)`` uint8 array."""
    data = Path(path).read_bytes()
    if not data.startswith(b"P5"):
        raise ValueError(f"{path}: not a binary PGM (P5) file")
    tokens, i = [], 2
    while len(tokens) < 3 and i < len(data):
        while i < len(data) and data[i : i + 1].isspace():
            i += 1
        if data[i : i + 1] == b"#":
            while i < len(data) and data[i] != 0x0A:
                i += 1
            continue
        start = i
        while i < len(data) and not data[i : i + 1].isspace():
            i += 1
        tokens.append(int(data[start:i]))
    i += 1  # single whitespace after the header
    width, height = tokens[0], tokens[1]
    return np.frombuffer(data[i : i + width * height], dtype=np.uint8).reshape(height, width)


def read_image(path: str | Path) -> np.ndarray:
    """Read a PNG, PGM or ``.npy`` file, returning a grayscale ``(H, W)`` uint8 array."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".png":
        img = read_png(path)
    elif suffix in (".pgm", ".pnm"):
        img = read_pgm(path)
    elif suffix == ".npy":
        img = np.load(path)
    else:
        raise ValueError(f"unsupported image type {suffix!r} (use .png, .pgm or .npy)")
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114])  # ITU-R 601 luma
    return arr.astype(np.uint8)


def resize_nearest(image: np.ndarray, size: int = 28) -> np.ndarray:
    """Nearest-neighbour resize of a 2-D array to ``size x size``."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("expected a 2-D array")
    h, w = img.shape
    rows = np.clip((np.arange(size) + 0.5) * h / size, 0, h - 1).astype(int)
    cols = np.clip((np.arange(size) + 0.5) * w / size, 0, w - 1).astype(int)
    return img[rows][:, cols]


def prepare_digit(image: np.ndarray, *, invert: bool | None = None, size: int = 28) -> np.ndarray:
    """Turn an arbitrary digit image into a normalised ``(784,)`` MNIST-style vector.

    MNIST digits are light strokes on a dark background; ``invert`` can force the
    polarity. When left as ``None`` it is inferred from the border pixels.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    if invert is None:
        invert = border.mean() > 0.5
    if invert:
        img = 1.0 - img
    return resize_nearest(img, size).reshape(-1).astype(np.float32)


def to_ascii(image: np.ndarray, width: int | None = None) -> str:
    """Render a grayscale image as ASCII art (handy for terminals)."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("expected a 2-D array")
    if width and width != img.shape[1]:
        img = resize_nearest(img, width)
    if img.max() > 1.0:
        img = img / 255.0
    ramp = " .:-=+*#%@"
    idx = np.clip((img * (len(ramp) - 1)).round().astype(int), 0, len(ramp) - 1)
    return "\n".join("".join(ramp[i] for i in row) for row in idx)



def _resample(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Box-average resample a 2-D array to ``out_h x out_w`` (anti-aliased)."""
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape
    rows = (np.arange(out_h + 1) * h / out_h).round().astype(int)
    cols = (np.arange(out_w + 1) * w / out_w).round().astype(int)
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        # Clamp so that upsampling (out_h > h) never produces an empty slice.
        r0 = min(int(rows[i]), h - 1)
        r1 = min(max(int(rows[i + 1]), r0 + 1), h)
        for j in range(out_w):
            c0 = min(int(cols[j]), w - 1)
            c1 = min(max(int(cols[j + 1]), c0 + 1), w)
            out[i, j] = img[r0:r1, c0:c1].mean()
    return out


def resize_area(image: np.ndarray, size: int = 28) -> np.ndarray:
    """Anti-aliased box downsampling of a 2-D array to ``size x size``.

    Each output pixel is the mean of the input pixels it covers, so smooth
    strokes stay smooth — unlike :func:`resize_nearest`.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("expected a 2-D array")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("cannot resize an empty image")
    return _resample(img, size, size)


def crop_to_content(image: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Crop a grayscale image to the bounding box of its bright ("ink") pixels."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("expected a 2-D array")
    if img.max() > 1.0:
        img = img / 255.0
    mask = img > threshold
    if not mask.any():
        return img
    rows, cols = np.where(mask)
    return img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def center_by_mass(image: np.ndarray, size: int = 28) -> np.ndarray:
    """Normalise a digit the way the original MNIST pipeline does.

    The ink is cropped, scaled to fit a 20x20 box (aspect ratio preserved) and
    pasted into a black ``size``-sized square so that its centre of mass lands
    in the middle. This matches how the training digits were prepared and makes
    a large difference for hand-drawn input.
    """
    img = crop_to_content(image)
    if img.max() > 1.0:
        img = img / 255.0
    h, w = img.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, min(size, int(round(h * scale))))
    new_w = max(1, min(size, int(round(w * scale))))
    resized = _resample(img, new_h, new_w)

    out = np.zeros((size, size), dtype=np.float32)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized

    total = float(out.sum())
    if total > 0:
        ys, xs = np.indices(out.shape)
        cy = float((ys * out).sum()) / total
        cx = float((xs * out).sum()) / total
        shift_y = int(round(size / 2.0 - cy))
        shift_x = int(round(size / 2.0 - cx))
        if shift_y or shift_x:
            out = np.roll(np.roll(out, shift_y, axis=0), shift_x, axis=1)
            # np.roll wraps around; blank the wrapped-in edges to keep ink inside.
            if shift_y > 0:
                out[:shift_y] = 0.0
            elif shift_y < 0:
                out[shift_y:] = 0.0
            if shift_x > 0:
                out[:, :shift_x] = 0.0
            elif shift_x < 0:
                out[:, shift_x:] = 0.0
    return out


def preprocess_digit(
    image: np.ndarray, *, invert: bool | None = None, size: int = 28, center: bool = True
) -> np.ndarray:
    """Full MNIST-style preprocessing for an arbitrary digit image.

    Handles polarity detection, anti-aliased resizing and (by default) the
    crop-scale-centre normalisation of the original dataset. Returns a flattened
    ``(784,)`` ``float32`` vector with values in ``[0, 1]``.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    if invert is None:
        invert = border.mean() > 0.5
    if invert:
        img = 1.0 - img
    normalised = center_by_mass(img, size=size) if center else resize_area(img, size)
    return np.clip(normalised, 0.0, 1.0).reshape(-1).astype(np.float32)

