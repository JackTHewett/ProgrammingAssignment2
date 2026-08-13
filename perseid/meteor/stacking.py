"""Stage 5 - stack and composite.

Three products come out of here:

* the **sky stack** - frames warped into the reference frame and combined with
  a sigma-clipped mean, then gradient-subtracted,
* the **ground stack** - the same frames combined without alignment, which is
  the only way the roofline stays sharp,
* the **composite** - the two blended through the feathered foreground mask.

The meteor composite reuses that background and lighten-blends each confirmed
trail on top, working in the aligned frame so the streaks land in the right
place against the stars.

Stacking reads RAF through rawpy when one sits next to the JPEG. Frames are
accumulated one at a time rather than held as a single array - 174 full-size
frames would be tens of gigabytes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .cache import Cache
from .config import Config
from .imageio_utils import load_for_stack
from .registration import apply_transform

log = logging.getLogger(__name__)


def _frames(paths: list[Path], cfg: Config) -> Iterator[tuple[Path, np.ndarray, str]]:
    for path in paths:
        image, source = load_for_stack(path, prefer_raw=cfg.stack.prefer_raw,
                                       use_camera_wb=cfg.stack.raw_use_camera_wb)
        yield path, image, source


def _sigma_clipped_mean(paths: list[Path], cfg: Config,
                        transforms: Optional[dict[str, np.ndarray]] = None,
                        label: str = "stack") -> tuple[np.ndarray, dict]:
    """Two-pass sigma-clipped mean.

    Pass one accumulates the mean and the sum of squares to get a per-pixel
    standard deviation; pass two re-accumulates while rejecting outliers. Two
    passes cost double the decode time but avoid holding every frame in memory,
    and rejecting outliers is what removes satellites, aircraft and cosmic-ray
    hits from the background.
    """
    scfg = cfg.stack
    n = 0
    total: Optional[np.ndarray] = None
    total_sq: Optional[np.ndarray] = None
    sources: dict[str, int] = {}

    for path, image, source in _frames(paths, cfg):
        if transforms is not None:
            image = apply_transform(image, transforms[path.name])
        if total is None:
            total = np.zeros_like(image, dtype=np.float64)
            total_sq = np.zeros_like(image, dtype=np.float64)
        elif image.shape != total.shape:
            log.warning("%s has shape %s, expected %s - skipped",
                        path.name, image.shape, total.shape)
            continue
        total += image
        total_sq += image.astype(np.float64) ** 2
        n += 1
        sources[source] = sources.get(source, 0) + 1
        log.debug("%s pass 1 %d/%d", label, n, len(paths))

    if total is None or n == 0:
        raise RuntimeError(f"no frames stacked for {label}")

    mean = total / n
    if scfg.method != "sigma_clip" or n < 4:
        log.info("%s: plain mean of %d frames", label, n)
        return mean.astype(np.float32), {"frames": n, "sources": sources, "method": "mean"}

    variance = np.maximum(total_sq / n - mean ** 2, 0.0)
    sigma = np.sqrt(variance)

    accum = np.zeros_like(mean)
    counts = np.zeros(mean.shape, dtype=np.float32)
    rejected = 0.0
    for path, image, _source in _frames(paths, cfg):
        if transforms is not None:
            image = apply_transform(image, transforms[path.name])
        if image.shape != mean.shape:
            continue
        deviation = image - mean
        keep = (deviation <= scfg.sigma_high * sigma) & (deviation >= -scfg.sigma_low * sigma)
        # Warped frames have zero-filled corners; those must not count.
        if transforms is not None:
            keep &= image > 0
        accum += np.where(keep, image, 0.0)
        counts += keep
        rejected += float((~keep).sum())

    # Where every sample was rejected, fall back to the unclipped mean.
    result = np.where(counts > 0, accum / np.maximum(counts, 1), mean)
    fraction = rejected / (n * mean.size)
    log.info("%s: sigma-clipped mean of %d frames (%.2f%% of samples rejected)",
             label, n, fraction * 100)
    return result.astype(np.float32), {
        "frames": n,
        "sources": sources,
        "method": "sigma_clip",
        "rejected_fraction": round(fraction, 5),
    }


def stack_sky(paths: list[Path], registration: dict, cfg: Config, cache: Cache,
              group: str) -> tuple[np.ndarray, dict]:
    """Star-aligned stack of every frame that registered."""
    key = f"sky_{group}"
    meta_key = f"sky_meta_{group}"
    if cache.has_array("stack", key) and cache.exists("stack", meta_key):
        log.info("reusing cached sky stack for group %s", group)
        return cache.load_array("stack", key), cache.load_json("stack", meta_key)

    usable, transforms = [], {}
    for path in paths:
        result = registration["results"].get(path.name, {})
        if result.get("status") in ("ok", "reference"):
            usable.append(path)
            transforms[path.name] = np.asarray(result["matrix"], dtype=float)

    if not usable:
        raise RuntimeError(f"group {group}: no frames registered, nothing to stack")

    image, meta = _sigma_clipped_mean(usable, cfg, transforms, label=f"sky {group}")
    cache.save_array("stack", key, image)
    cache.save_json("stack", meta_key, meta)
    return image, meta


def stack_ground(paths: list[Path], cfg: Config, cache: Cache,
                 group: str) -> tuple[np.ndarray, dict]:
    """Unaligned stack, so the buildings stay sharp.

    A median would be the safer choice for rejecting the sky, but it needs every
    frame resident at once. The sigma-clipped mean gets close for a fraction of
    the memory, and the sky half of this stack is thrown away anyway.
    """
    key = f"ground_{group}"
    meta_key = f"ground_meta_{group}"
    if cache.has_array("stack", key) and cache.exists("stack", meta_key):
        log.info("reusing cached ground stack for group %s", group)
        return cache.load_array("stack", key), cache.load_json("stack", meta_key)

    image, meta = _sigma_clipped_mean(paths, cfg, None, label=f"ground {group}")
    cache.save_array("stack", key, image)
    cache.save_json("stack", meta_key, meta)
    return image, meta


def composite(sky: np.ndarray, ground: np.ndarray, foreground_alpha: np.ndarray) -> np.ndarray:
    """Blend the aligned sky with the unaligned ground through a soft mask."""
    alpha = foreground_alpha
    if sky.ndim == 3 and alpha.ndim == 2:
        alpha = alpha[..., None]
    return np.clip(sky * (1.0 - alpha) + ground * alpha, 0.0, 1.0).astype(np.float32)


def _trail_alpha(shape: tuple[int, int], candidate: dict, cfg: Config) -> np.ndarray:
    """Soft mask around one trail, so only the streak is blended in."""
    import cv2

    scfg = cfg.stack
    mask = np.zeros(shape, dtype=np.float32)
    p1 = (int(round(candidate["x1"])), int(round(candidate["y1"])))
    p2 = (int(round(candidate["x2"])), int(round(candidate["y2"])))
    cv2.line(mask, p1, p2, 1.0, thickness=2 * scfg.meteor_blend_margin_px)
    k = 2 * scfg.meteor_blend_feather_px + 1
    return cv2.GaussianBlur(mask, (k, k), scfg.meteor_blend_feather_px / 2.0)


def meteor_composite(background: np.ndarray, confirmed: list[dict], paths_by_name: dict[str, Path],
                     registration: dict, cfg: Config) -> tuple[np.ndarray, int]:
    """Lighten-blend every confirmed meteor onto the background.

    Each trail comes from its own frame, warped into the reference frame so the
    streak sits correctly against the stars, and is restricted to a feathered
    region around the detection so the rest of that frame's noise is not pulled
    in with it.
    """
    result = background.copy()
    blended = 0

    for candidate in confirmed:
        name = candidate["frame"]
        path = paths_by_name.get(name)
        entry = registration["results"].get(name, {})
        if path is None or entry.get("status") not in ("ok", "reference"):
            log.warning("cannot blend %s from %s - frame not registered",
                        candidate.get("classification", "candidate"), name)
            continue

        image, _source = load_for_stack(path, prefer_raw=cfg.stack.prefer_raw,
                                        use_camera_wb=cfg.stack.raw_use_camera_wb)
        aligned = apply_transform(image, np.asarray(entry["matrix"], dtype=float))
        if aligned.shape != result.shape:
            log.warning("%s does not match the stack shape - skipped", name)
            continue

        alpha = _trail_alpha(result.shape[:2], candidate, cfg)
        if aligned.ndim == 3:
            alpha = alpha[..., None]

        # Lighten: keep whichever is brighter, but only inside the trail region.
        lightened = np.maximum(result, aligned)
        result = result * (1.0 - alpha) + lightened * alpha
        blended += 1
        log.info("blended %s trail from %s", candidate.get("classification", "candidate"), name)

    return np.clip(result, 0.0, 1.0).astype(np.float32), blended
