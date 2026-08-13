"""Image loading and writing.

JPEGs are used for anything speed-sensitive (grouping, masking, detection,
timelapse). RAF files go through rawpy for the stacks, where the extra shadow
latitude actually matters. Everything downstream works in float32 0..1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

RAW_SUFFIXES = (".RAF", ".raf")
JPEG_SUFFIXES = (".JPG", ".jpg", ".JPEG", ".jpeg")


def find_raw_for(jpeg_path: Path) -> Optional[Path]:
    """Return the RAF sitting next to a JPEG, if there is one."""
    for suffix in RAW_SUFFIXES:
        candidate = jpeg_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def load_jpeg(path: Path, as_gray: bool = False) -> np.ndarray:
    """Load a JPEG as float32 in 0..1."""
    with Image.open(path) as im:
        im = im.convert("L" if as_gray else "RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def load_jpeg_proxy(path: Path, width: int, as_gray: bool = True) -> np.ndarray:
    """Load a downscaled JPEG. Uses PIL's draft mode, which lets libjpeg do the
    downscale during decode - far faster than decoding full size then resizing.
    """
    with Image.open(path) as im:
        im.draft("L" if as_gray else "RGB", (width, width))
        im = im.convert("L" if as_gray else "RGB")
        if im.width != width:
            height = max(1, round(im.height * width / im.width))
            im = im.resize((width, height), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def load_raw(path: Path, use_camera_wb: bool = True) -> np.ndarray:
    """Demosaic a RAF to linear-ish 16-bit RGB, returned as float32 0..1.

    ``no_auto_bright`` matters: any per-frame auto brightness would break the
    photometric consistency the stack depends on.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=use_camera_wb,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
            output_color=__import__("rawpy").ColorSpace.sRGB,
        )
    return rgb.astype(np.float32) / 65535.0


def load_for_stack(jpeg_path: Path, prefer_raw: bool = True,
                   use_camera_wb: bool = True) -> tuple[np.ndarray, str]:
    """Load the best available version of a frame for stacking.

    Returns ``(image, source)`` where source is "raw" or "jpeg".
    """
    if prefer_raw:
        raw_path = find_raw_for(jpeg_path)
        if raw_path is not None:
            try:
                return load_raw(raw_path, use_camera_wb=use_camera_wb), "raw"
            except Exception as exc:  # pragma: no cover - depends on raw file
                log.warning("rawpy failed on %s (%s); using the JPEG", raw_path.name, exc)
    return load_jpeg(jpeg_path), "jpeg"


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


def save_tiff16(path: Path, img: np.ndarray) -> None:
    """Write a 16-bit TIFF for finishing in Lightroom."""
    import tifffile

    arr = np.clip(img, 0.0, 1.0)
    tifffile.imwrite(str(path), (arr * 65535.0 + 0.5).astype(np.uint16),
                     photometric="rgb" if arr.ndim == 3 else "minisblack")


def save_jpeg(path: Path, img: np.ndarray, quality: int = 92) -> None:
    arr = np.clip(img, 0.0, 1.0)
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    mode = "RGB" if arr8.ndim == 3 else "L"
    Image.fromarray(arr8, mode=mode).save(path, quality=quality, subsampling=0)


def save_png(path: Path, img: np.ndarray) -> None:
    arr = np.clip(img, 0.0, 1.0)
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    mode = "RGB" if arr8.ndim == 3 else "L"
    Image.fromarray(arr8, mode=mode).save(path)


def autostretch(img: np.ndarray, black_pct: float = 0.5, white_pct: float = 99.7,
                gamma: float = 0.55) -> np.ndarray:
    """A display stretch for previews only - never fed back into the stacks."""
    lo = np.percentile(img, black_pct)
    hi = np.percentile(img, white_pct)
    if hi <= lo:
        return np.clip(img, 0.0, 1.0)
    out = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return np.power(out, gamma, dtype=np.float32)
