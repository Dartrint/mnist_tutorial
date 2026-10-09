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


def read_png(path: str | Path) -> np.ndarray:
    """Decode an 8-bit non-interlaced PNG into ``(H, W)`` or ``(H, W, C)`` uint8."""
    raw = Path(path).read_bytes()
    if raw[:8] != _PNG_SIG:
        raise ValueError(f"{path}: not a PNG file")
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
        raise ValueError(f"{path}: only 8-bit PNGs are supported (got {depth})")
    if interlace:
        raise ValueError(f"{path}: interlaced PNGs are not supported")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(ctype)
    if channels is None:
        raise ValueError(f"{path}: unsupported PNG colour type {ctype}")

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
            raise ValueError(f"{path}: unknown PNG filter {ftype}")
        out[row] = line
        prev = line

    img = out.reshape(height, width, channels)
    return img[:, :, 0] if channels == 1 else img


def _chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def write_png(path: str | Path, image: np.ndarray) -> Path:
    """Write a uint8 grayscale or RGB(A) array as an 8-bit PNG."""
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
    chunks = [
        _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, ctype, 0, 0, 0)),
        _chunk(b"IDAT", zlib.compress(lines, 9)),
        _chunk(b"IEND", b""),
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG_SIG + b"".join(chunks))
    return path


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

