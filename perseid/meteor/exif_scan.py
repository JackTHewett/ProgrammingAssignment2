"""Stage 0 - read EXIF from every frame and work out the processing groups.

Two things split frames into groups:

* **Exposure settings.** Shutter/ISO changes mean a different noise and sky
  level, so those frames stack separately.
* **Framing.** The camera sat on a table, so it drifted and got knocked. A
  deliberate refocus can shift the framing far enough that one alignment
  cannot cover the whole night; that starts a new framing group.

Nothing here is expensive - EXIF plus a small grayscale proxy per frame - so it
runs over the full set without asking first.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .config import Config
from .imageio_utils import JPEG_SUFFIXES, find_raw_for, load_jpeg_proxy

log = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    index: int
    name: str
    path: str
    raw_path: Optional[str]
    timestamp: Optional[str]
    exposure_s: Optional[float]
    iso: Optional[int]
    aperture: Optional[float]
    focal_length_mm: Optional[float]
    camera: Optional[str]
    width: Optional[int]
    height: Optional[int]
    # filled in by the framing pass
    shift_x: float = 0.0
    shift_y: float = 0.0
    step_shift: float = 0.0
    exposure_group: int = -1
    framing_group: int = -1
    group: str = ""


def _ratio(value) -> Optional[float]:
    """Coerce an EXIF scalar to a float.

    Cameras write these as RATIONAL, which exifread hands back as a Ratio in a
    one-element list. Some writers emit the numerator and denominator as two
    SHORTs instead, so a bare two-element pair is read as num/den rather than
    silently taken as its first element - that is the difference between 1/4s
    and "4 seconds".
    """
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return _ratio(value[0])
            if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
                return float(value[0]) / float(value[1]) if value[1] else None
            return _ratio(value[0]) if value else None
        if hasattr(value, "num") and hasattr(value, "den"):
            return float(value.num) / float(value.den) if value.den else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_exif(path: Path) -> dict:
    """Read EXIF from a JPEG or RAF. exifread handles both."""
    import exifread

    with open(path, "rb") as fh:
        tags = exifread.process_file(fh, details=False)

    def get(key: str):
        tag = tags.get(key)
        return tag.values if tag is not None else None

    def first(key: str):
        v = get(key)
        if isinstance(v, list):
            return v[0] if v else None
        return v

    # These go through _ratio whole, so a two-SHORT num/den pair is not mistaken
    # for a scalar.
    exposure = _ratio(get("EXIF ExposureTime"))
    iso = first("EXIF ISOSpeedRatings")
    aperture = _ratio(get("EXIF FNumber"))
    focal = _ratio(get("EXIF FocalLength"))
    stamp = get("EXIF DateTimeOriginal") or get("Image DateTime")
    camera_make = get("Image Make")
    camera_model = get("Image Model")

    timestamp = None
    if stamp:
        text = str(stamp).strip()
        try:
            timestamp = datetime.strptime(text, "%Y:%m:%d %H:%M:%S").isoformat()
        except ValueError:
            log.debug("unparsed timestamp %r in %s", text, path.name)

    camera = " ".join(str(x).strip() for x in (camera_make, camera_model) if x) or None

    width = first("EXIF ExifImageWidth")
    height = first("EXIF ExifImageLength")

    return {
        "timestamp": timestamp,
        "exposure_s": exposure,
        "iso": int(iso) if iso is not None else None,
        "aperture": aperture,
        "focal_length_mm": focal,
        "camera": camera,
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
    }


def discover_frames(source: Path) -> list[Path]:
    """All JPEGs in the folder, in shot order.

    Sorted by the numeric part of the filename so DSCF999 precedes DSCF1000,
    which a plain lexicographic sort gets wrong.
    """
    files: list[Path] = []
    for suffix in JPEG_SUFFIXES:
        files.extend(source.glob(f"*{suffix}"))
    files = sorted(set(files))

    def sort_key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return (int(m.group(1)) if m else 0, p.stem)

    return sorted(files, key=sort_key)


def _exposure_groups(frames: list[FrameInfo], tolerance: float) -> None:
    """Assign an exposure group id per distinct (exposure, ISO) setting."""
    seen: list[tuple[Optional[float], Optional[int]]] = []

    for f in frames:
        match = -1
        for i, (exp, iso) in enumerate(seen):
            if iso != f.iso:
                continue
            if exp is None or f.exposure_s is None:
                if exp is f.exposure_s:
                    match = i
                    break
                continue
            if abs(exp - f.exposure_s) <= tolerance * max(exp, f.exposure_s):
                match = i
                break
        if match < 0:
            seen.append((f.exposure_s, f.iso))
            match = len(seen) - 1
        f.exposure_group = match


def _framing_groups(frames: list[FrameInfo], paths: list[Path], cfg: Config) -> None:
    """Track framing drift and split where the camera jumped.

    Frame-to-frame translation comes from phase correlation on small grayscale
    proxies. Stars move a little between frames, but they are point sources
    against a mostly static gradient and roofline, so the dominant correlation
    peak follows the framing rather than the sky.
    """
    import cv2

    width = cfg.grouping.proxy_width
    break_px = cfg.grouping.framing_break_fraction * width

    prev: Optional[np.ndarray] = None
    cum_x = cum_y = 0.0
    group = 0

    for f, path in zip(frames, paths):
        proxy = load_jpeg_proxy(path, width, as_gray=True)
        # A Hann window keeps the frame edges from ringing in the correlation.
        window = cv2.createHanningWindow((proxy.shape[1], proxy.shape[0]), cv2.CV_32F)
        current = np.ascontiguousarray(proxy, dtype=np.float32)

        if prev is not None:
            (dx, dy), _response = cv2.phaseCorrelate(prev, current, window)
            step = float(np.hypot(dx, dy))
            cum_x += float(dx)
            cum_y += float(dy)
            if step > break_px:
                group += 1
                log.info("framing break before %s (%.1f px on the %d px proxy)",
                         f.name, step, width)
        else:
            step = 0.0

        f.step_shift = round(step, 3)
        f.shift_x = round(cum_x, 3)
        f.shift_y = round(cum_y, 3)
        f.framing_group = group
        prev = current


def scan(cfg: Config) -> list[dict]:
    """Read every frame's EXIF, then group by settings and framing."""
    paths = discover_frames(cfg.paths.source)
    if not paths:
        raise SystemExit(f"no JPEGs found in {cfg.paths.source.resolve()}")

    log.info("reading EXIF from %d frames", len(paths))
    frames: list[FrameInfo] = []
    for i, path in enumerate(paths):
        meta = read_exif(path)
        raw = find_raw_for(path)
        frames.append(FrameInfo(
            index=i,
            name=path.name,
            path=str(path),
            raw_path=str(raw) if raw else None,
            **meta,
        ))

    _exposure_groups(frames, cfg.grouping.exposure_tolerance)
    log.info("measuring framing drift")
    _framing_groups(frames, paths, cfg)

    for f in frames:
        f.group = f"e{f.exposure_group}f{f.framing_group}"

    return [asdict(f) for f in frames]


# --- reporting ------------------------------------------------------------

def _fmt_exposure(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    if seconds >= 1:
        return f"{seconds:g}s"
    return f"1/{round(1 / seconds)}s"


def _span(values: Iterable[Optional[str]]) -> tuple[Optional[str], Optional[str]]:
    stamps = sorted(v for v in values if v)
    return (stamps[0], stamps[-1]) if stamps else (None, None)


def summarise(frames: list[dict], cfg: Config) -> str:
    """The human-readable session summary printed before any processing."""
    lines: list[str] = []
    n = len(frames)

    cameras = sorted({f["camera"] for f in frames if f["camera"]})
    focals = sorted({f["focal_length_mm"] for f in frames if f["focal_length_mm"]})
    apertures = sorted({f["aperture"] for f in frames if f["aperture"]})
    start, end = _span(f["timestamp"] for f in frames)
    with_raw = sum(1 for f in frames if f["raw_path"])

    lines.append("=" * 72)
    lines.append("SHOOTING SESSION")
    lines.append("=" * 72)
    lines.append(f"  frames            {n}")
    lines.append(f"  with RAF alongside {with_raw}"
                 + ("" if with_raw == n else f"  ({n - with_raw} JPEG-only)"))
    lines.append(f"  camera            {', '.join(cameras) or 'unknown'}")
    lines.append(f"  focal length      {', '.join(f'{v:g}mm' for v in focals) or 'unknown'}")
    lines.append(f"  aperture          {', '.join(f'f/{v:g}' for v in apertures) or 'unknown'}")
    if start and end:
        t0 = datetime.fromisoformat(start)
        t1 = datetime.fromisoformat(end)
        span = t1 - t0
        hours, rem = divmod(int(span.total_seconds()), 3600)
        lines.append(f"  first frame       {t0:%Y-%m-%d %H:%M:%S}")
        lines.append(f"  last frame        {t1:%Y-%m-%d %H:%M:%S}")
        lines.append(f"  elapsed           {hours}h {rem // 60}m")

    # --- exposure groups ---
    lines.append("")
    lines.append("EXPOSURE GROUPS")
    lines.append("-" * 72)
    lines.append(f"  {'id':<4}{'shutter':>10}{'ISO':>8}{'frames':>8}   {'range':<24}")
    by_exp: dict[int, list[dict]] = {}
    for f in frames:
        by_exp.setdefault(f["exposure_group"], []).append(f)
    for gid in sorted(by_exp):
        members = by_exp[gid]
        head = members[0]
        lines.append(
            f"  {gid:<4}{_fmt_exposure(head['exposure_s']):>10}"
            f"{head['iso'] if head['iso'] else '?':>8}{len(members):>8}   "
            f"{members[0]['name']} .. {members[-1]['name']}"
        )

    # --- framing groups ---
    lines.append("")
    lines.append("FRAMING GROUPS  (camera movement between frames)")
    lines.append("-" * 72)
    lines.append(f"  {'id':<4}{'frames':>8}{'drift px':>10}   {'range':<24}")
    by_frm: dict[int, list[dict]] = {}
    for f in frames:
        by_frm.setdefault(f["framing_group"], []).append(f)
    for gid in sorted(by_frm):
        members = by_frm[gid]
        drift = max(abs(m["shift_x"] - members[0]["shift_x"]) for m in members)
        drift_y = max(abs(m["shift_y"] - members[0]["shift_y"]) for m in members)
        lines.append(
            f"  {gid:<4}{len(members):>8}{max(drift, drift_y):>10.1f}   "
            f"{members[0]['name']} .. {members[-1]['name']}"
        )
    lines.append(f"  (drift measured on a {cfg.grouping.proxy_width}px proxy; "
                 f"break threshold {cfg.grouping.framing_break_fraction * cfg.grouping.proxy_width:.0f}px)")

    # --- combined processing groups ---
    lines.append("")
    lines.append("PROCESSING GROUPS  (exposure x framing - each stacks separately)")
    lines.append("-" * 72)
    by_group: dict[str, list[dict]] = {}
    for f in frames:
        by_group.setdefault(f["group"], []).append(f)
    lines.append(f"  {'group':<8}{'frames':>8}{'shutter':>10}{'ISO':>8}   {'range':<24}")
    for gid in sorted(by_group):
        members = by_group[gid]
        head = members[0]
        flag = "" if len(members) >= cfg.grouping.min_group_size else "   <- too small, skipped"
        lines.append(
            f"  {gid:<8}{len(members):>8}{_fmt_exposure(head['exposure_s']):>10}"
            f"{head['iso'] if head['iso'] else '?':>8}   "
            f"{members[0]['name']} .. {members[-1]['name']}{flag}"
        )

    usable = sum(len(v) for v in by_group.values() if len(v) >= cfg.grouping.min_group_size)
    lines.append("")
    lines.append(f"  {usable}/{n} frames sit in a group large enough to stack.")
    lines.append("=" * 72)
    return "\n".join(lines)


def largest_group(frames: list[dict], cfg: Config) -> str:
    by_group: dict[str, list[dict]] = {}
    for f in frames:
        by_group.setdefault(f["group"], []).append(f)
    return max(by_group, key=lambda g: len(by_group[g]))


def group_members(frames: list[dict], group: str) -> list[dict]:
    return [f for f in frames if f["group"] == group]


def processable_groups(frames: list[dict], cfg: Config) -> list[str]:
    by_group: dict[str, list[dict]] = {}
    for f in frames:
        by_group.setdefault(f["group"], []).append(f)
    return sorted(g for g, m in by_group.items() if len(m) >= cfg.grouping.min_group_size)
