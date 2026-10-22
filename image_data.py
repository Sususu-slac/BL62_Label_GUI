"""Prepare original-range display and SAM3 image caches during project import."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _cache_root(path: Path) -> Path:
    # Keep generated files on the same drive as the source dataset.
    return path.parent / ".bl62_sam3_cache"


def _signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_source(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode in ("P", "PA", "1", "CMYK", "YCbCr", "LAB", "HSV"):
            return np.asarray(image.convert("RGB")).copy()
        return np.asarray(image).copy()


def _gray_values(source: np.ndarray) -> np.ndarray:
    values = np.asarray(source)
    if values.ndim == 2:
        return values.astype(np.float64, copy=False)
    if values.ndim == 3:
        channels = values[..., :3].astype(np.float64, copy=False)
        if channels.shape[2] == 1:
            return channels[..., 0]
        if channels.shape[2] >= 3:
            return channels[..., 0] * 0.299 + channels[..., 1] * 0.587 + channels[..., 2] * 0.114
    raise ValueError(f"Unsupported image shape: {values.shape}")


def _rgb8(source: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(source)
    if values.ndim == 2:
        channels = values[..., None]
    elif values.ndim == 3 and values.shape[2] in (1, 2, 3, 4):
        channels = values[..., :3] if values.shape[2] >= 3 else values[..., :1]
    else:
        raise ValueError(f"Unsupported image shape: {values.shape}")
    work = channels.astype(np.float32, copy=False)
    if high <= low:
        mapped = np.zeros(work.shape, dtype=np.uint8)
    else:
        mapped = np.clip((np.nan_to_num(work, nan=low, posinf=high, neginf=low) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    if mapped.shape[2] == 1:
        mapped = np.repeat(mapped, 3, axis=2)
    return np.ascontiguousarray(mapped[..., :3])


def _display_rgb8(source: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(source)
    if values.dtype == np.uint8:
        if values.ndim == 2:
            values = np.repeat(values[..., None], 3, axis=2)
        elif values.ndim == 3 and values.shape[2] == 1:
            values = np.repeat(values, 3, axis=2)
        return np.ascontiguousarray(values[..., :3])
    return _rgb8(values, low, high)


@dataclass
class PreparedImage:
    display_rgb8: np.ndarray
    sam_path: Path
    sam_rgb8: np.ndarray | None
    source_dtype: str
    histogram: np.ndarray
    intensity_min: float
    intensity_max: float
    cache_dir: Path

    @property
    def nbytes(self) -> int:
        return self.display_rgb8.nbytes + self.histogram.nbytes + (0 if self.sam_rgb8 is None else self.sam_rgb8.nbytes)


def prepare_image(path: Path, force: bool = False, cancelled=None) -> PreparedImage:
    """Create a reusable display preview, original-range histogram, and SAM3 input."""
    def check_cancelled():
        if cancelled and cancelled():
            raise InterruptedError("Image preparation cancelled")

    check_cancelled()
    path = Path(os.path.abspath(path))
    signature = _signature(path)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()
    folder = _cache_root(path) / key
    metadata_path = folder / "metadata.json"
    display_path = folder / "display.npy"
    sam_path = folder / "sam.npy" if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"} else path
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature and display_path.is_file() and (metadata.get("sam_is_source") or (folder / "sam.npy").is_file()):
            check_cancelled()
            display = np.load(display_path, allow_pickle=False)
            return PreparedImage(display, path if metadata["sam_is_source"] else folder / "sam.npy", None,
                metadata["source_dtype"], np.asarray(metadata["histogram"], dtype=np.int64),
                float(metadata["intensity_min"]), float(metadata["intensity_max"]), folder)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    source = _read_source(path)
    check_cancelled()
    gray = _gray_values(source)
    check_cancelled()
    finite = np.isfinite(gray)
    if not finite.any():
        low, high = 0.0, 0.0
        histogram = np.zeros(256, dtype=np.int64)
    else:
        low, high = float(np.min(gray[finite])), float(np.max(gray[finite]))
        histogram = np.histogram(gray[finite], bins=256, range=(low, high) if high > low else (low - 0.5, high + 0.5))[0].astype(np.int64)
    display = _display_rgb8(source, low, high)
    check_cancelled()
    is_uint8 = source.dtype == np.uint8
    if is_uint8:
        # 8-bit files are already in SAM3's supported format; retain the source path.
        sam_output = None
        cached_sam_path = path
    else:
        sam_gray = gray[np.isfinite(gray)]
        if sam_gray.size:
            sam_low, sam_high = np.percentile(sam_gray, (0.1, 99.9))
            if sam_high <= sam_low:
                sam_low, sam_high = low, high
        else:
            sam_low, sam_high = low, high
        sam_output = _rgb8(source, float(sam_low), float(sam_high))
        check_cancelled()
        cached_sam_path = folder / "sam.npy"
    folder.mkdir(parents=True, exist_ok=True)
    check_cancelled()
    display_tmp = folder / "display.tmp.npy"
    with display_tmp.open("wb") as handle:
        np.save(handle, display, allow_pickle=False)
    os.replace(display_tmp, display_path)
    if sam_output is not None:
        sam_tmp = folder / "sam.tmp.npy"
        with sam_tmp.open("wb") as handle:
            np.save(handle, sam_output, allow_pickle=False)
        os.replace(sam_tmp, folder / "sam.npy")
    metadata = {"signature": signature, "source_dtype": str(source.dtype), "histogram": histogram.tolist(),
        "intensity_min": low, "intensity_max": high, "sam_is_source": is_uint8}
    metadata_tmp = folder / "metadata.tmp.json"
    metadata_tmp.write_text(json.dumps(metadata), encoding="utf-8")
    os.replace(metadata_tmp, metadata_path)
    return PreparedImage(np.load(display_path, allow_pickle=False), cached_sam_path,
        sam_output, str(source.dtype), histogram, low, high, folder)


class ImageDataCache:
    """Small in-memory LRU over import-time prepared image previews."""

    def __init__(self, max_bytes: int = 256 * 1024 * 1024):
        self.max_bytes = max(1, int(max_bytes))
        self._items: OrderedDict[str, PreparedImage] = OrderedDict()
        self._bytes = 0

    def get(self, path: Path) -> PreparedImage:
        path = Path(os.path.abspath(path))
        key = str(path)
        item = self._items.pop(key, None)
        if item is None:
            item = prepare_image(path)
            self._bytes += item.nbytes
        else:
            self._bytes -= item.nbytes
        self._items[key] = item
        while self._bytes > self.max_bytes and len(self._items) > 1:
            _, evicted = self._items.popitem(last=False)
            self._bytes -= evicted.nbytes
        return item

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0

    @property
    def cached_bytes(self) -> int:
        return self._bytes

    @property
    def cached_images(self) -> int:
        return len(self._items)
