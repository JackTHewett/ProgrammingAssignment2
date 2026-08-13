"""Stage 2 - align frames on the stars.

astroalign matches asterisms (star triangles), so it recovers the rotation from
Earth's spin as well as the translation from table bumps - which a plain
cross-correlation cannot do.

It only ever sees the sky: the foreground is zeroed first, otherwise the fixed
roofline dominates and the solver happily "aligns" the building instead.

When astroalign cannot find enough stars, a DAOStarFinder pass supplies the
source list and astroalign solves from those coordinates instead. Frames that
still fail are skipped and reported rather than dropped into the stack.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .cache import Cache
from .config import Config
from .imageio_utils import load_jpeg, to_gray

log = logging.getLogger(__name__)


class RegistrationError(RuntimeError):
    pass


def _masked_sky(gray: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    """Zero the foreground and flatten the background so stars stand out."""
    import cv2

    out = gray.copy()
    # A large median subtraction removes the light pollution gradient, which
    # otherwise biases star detection towards the bright corner.
    background = cv2.medianBlur((out * 255).astype(np.uint8), 31).astype(np.float32) / 255.0
    out = np.clip(out - background, 0.0, None)
    out[~sky_mask] = 0.0
    return np.ascontiguousarray(out, dtype=np.float32)


def _find_stars(img: np.ndarray, cfg: Config) -> np.ndarray:
    """DAOStarFinder fallback source list, brightest first, as (x, y)."""
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    rcfg = cfg.registration
    _mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    if std <= 0:
        return np.empty((0, 2), dtype=float)
    finder = DAOStarFinder(fwhm=rcfg.fallback_fwhm,
                           threshold=median + rcfg.fallback_threshold_sigma * std)
    table = finder(img)
    if table is None or len(table) == 0:
        return np.empty((0, 2), dtype=float)
    table.sort("flux", reverse=True)
    table = table[: rcfg.fallback_max_stars]
    # photutils 3.0 renamed these columns; support both rather than relying on
    # the compatibility shim, which emits a deprecation warning per call and
    # buries the run log.
    x_col = "x_centroid" if "x_centroid" in table.colnames else "xcentroid"
    y_col = "y_centroid" if "y_centroid" in table.colnames else "ycentroid"
    return np.transpose((np.asarray(table[x_col]), np.asarray(table[y_col])))


def _degenerate(points: np.ndarray, shape: tuple[int, int], cfg: Config) -> Optional[str]:
    """Reject control-point sets whose geometry cannot constrain a rotation.

    If every matched star lies along one line - which happens when a railing or
    roof edge shows through a slightly loose mask - then a 180 degree flip about
    that line fits just as well as the identity, and the solver will happily
    return it with a sub-pixel residual. Checking the residual alone does not
    catch this; the point spread has to be genuinely two-dimensional.
    """
    rcfg = cfg.registration
    if len(points) < 3:
        return f"only {len(points)} control points"

    centred = points - points.mean(axis=0)
    # Singular values are the spread along the principal axes, in pixels.
    spread = np.linalg.svd(centred, compute_uv=False) / math.sqrt(len(points))
    major, minor = float(spread[0]), float(spread[1])
    if major <= 0:
        return "control points coincide"

    if minor / major < rcfg.min_point_axis_ratio:
        return (f"control points nearly collinear "
                f"(axis ratio {minor / major:.3f})")

    if major < rcfg.min_point_spread_fraction * max(shape):
        return (f"control points cover too little of the frame "
                f"({major:.0f}px spread)")
    return None


def _validate(matrix: np.ndarray, shape: tuple[int, int], cfg: Config) -> Optional[str]:
    """Reject transforms that are not physically plausible for this session."""
    rcfg = cfg.registration
    a = matrix[:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(a))))
    if not np.isfinite(scale) or abs(scale - 1.0) > rcfg.max_scale_deviation:
        return f"implausible scale {scale:.4f}"

    rotation = float(np.degrees(np.arctan2(a[1, 0], a[0, 0])))
    if abs(rotation) > rcfg.max_rotation_deg:
        return f"implausible rotation {rotation:.2f} deg"

    shift = float(np.hypot(matrix[0, 2], matrix[1, 2]))
    limit = rcfg.max_translation_fraction * max(shape)
    if shift > limit:
        return f"implausible shift {shift:.0f}px (limit {limit:.0f}px)"
    return None


def solve_transform(source: np.ndarray, target: np.ndarray,
                    cfg: Config) -> tuple[np.ndarray, dict]:
    """Find the transform mapping ``source`` onto ``target``.

    Both inputs are already sky-masked. Returns the 3x3 matrix plus stats.
    """
    import astroalign

    rcfg = cfg.registration
    astroalign.MAX_CONTROL_POINTS = rcfg.max_control_points
    astroalign.MIN_MATCHES_FRACTION = 0.6
    astroalign.PIXEL_TOL = 2
    astroalign.NUM_NEAREST_NEIGHBORS = 5

    method = "astroalign"
    try:
        transform, (src_pts, tgt_pts) = astroalign.find_transform(
            source, target,
            detection_sigma=rcfg.detection_sigma,
            min_area=rcfg.min_area,
        )
    except Exception as exc:
        log.debug("astroalign image solve failed (%s); trying DAOStarFinder", exc)
        src_stars = _find_stars(source, cfg)
        tgt_stars = _find_stars(target, cfg)
        if len(src_stars) < rcfg.min_matched_stars or len(tgt_stars) < rcfg.min_matched_stars:
            raise RegistrationError(
                f"too few stars ({len(src_stars)} source, {len(tgt_stars)} target)"
            ) from exc
        try:
            transform, (src_pts, tgt_pts) = astroalign.find_transform(src_stars, tgt_stars)
        except Exception as exc2:
            raise RegistrationError(f"no asterism match ({exc2})") from exc2
        method = "daostarfinder"

    matched = int(len(src_pts))
    if matched < rcfg.min_matched_stars:
        raise RegistrationError(f"only {matched} stars matched")

    problem = _degenerate(np.asarray(tgt_pts, dtype=float), target.shape, cfg)
    if problem:
        raise RegistrationError(problem)

    matrix = np.asarray(transform.params, dtype=float)

    # Residual scatter of the matched stars after applying the solution.
    ones = np.ones((len(src_pts), 1))
    projected = (np.hstack([np.asarray(src_pts), ones]) @ matrix.T)[:, :2]
    residuals = np.linalg.norm(projected - np.asarray(tgt_pts), axis=1)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    if rms > rcfg.max_residual_px:
        raise RegistrationError(f"residual {rms:.2f}px over limit")

    problem = _validate(matrix, source.shape, cfg)
    if problem:
        raise RegistrationError(problem)

    return matrix, {"method": method, "matched_stars": matched, "residual_px": round(rms, 3)}


def apply_transform(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Warp an image into the reference frame.

    astroalign's matrix maps source coordinates to target, so warping the source
    needs its inverse - which is what ``skimage.transform.warp`` expects.
    """
    from skimage.transform import ProjectiveTransform, warp

    transform = ProjectiveTransform(matrix=matrix)
    out = warp(img, transform.inverse, order=1, mode="constant", cval=0.0,
               preserve_range=True)
    return out.astype(np.float32)


def register_group(paths: list[Path], sky_mask_proxy: np.ndarray, cfg: Config,
                   cache: Cache, group: str) -> dict:
    """Solve every frame in a group against a reference frame.

    The reference is the middle frame, which keeps the largest rotation at
    either end of the sequence to roughly half what an end-frame reference
    would give.
    """
    from .masking import resize_mask

    key = f"transforms_{group}"
    if cache.exists("registration", key):
        cached = cache.load_json("registration", key)
        if cached.get("frames") == [p.name for p in paths]:
            log.info("reusing cached registration for group %s", group)
            return cached

    ref_index = len(paths) // 2
    ref_gray = to_gray(load_jpeg(paths[ref_index]))
    sky_mask = resize_mask(sky_mask_proxy, ref_gray.shape)
    ref_prepared = _masked_sky(ref_gray, sky_mask)

    log.info("registering %d frames in group %s against %s",
             len(paths), group, paths[ref_index].name)

    results: dict[str, dict] = {}
    ok = failed = 0
    for i, path in enumerate(paths):
        if i == ref_index:
            results[path.name] = {
                "status": "reference",
                "matrix": np.eye(3).tolist(),
                "matched_stars": None,
                "residual_px": 0.0,
                "method": "reference",
            }
            ok += 1
            continue
        try:
            gray = to_gray(load_jpeg(path))
            prepared = _masked_sky(gray, sky_mask)
            matrix, stats = solve_transform(prepared, ref_prepared, cfg)
            results[path.name] = {"status": "ok", "matrix": matrix.tolist(), **stats}
            ok += 1
            log.debug("%s aligned (%d stars, %.2fpx, %s)",
                      path.name, stats["matched_stars"], stats["residual_px"], stats["method"])
        except RegistrationError as exc:
            results[path.name] = {"status": "failed", "reason": str(exc)}
            failed += 1
            log.warning("%s skipped: %s", path.name, exc)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the run
            results[path.name] = {"status": "failed", "reason": f"unexpected: {exc}"}
            failed += 1
            log.warning("%s skipped: unexpected error: %s", path.name, exc)

    payload = {
        "group": group,
        "reference": paths[ref_index].name,
        "frames": [p.name for p in paths],
        "aligned": ok,
        "failed": failed,
        "results": results,
    }
    cache.save_json("registration", key, payload)
    log.info("group %s: %d/%d aligned, %d skipped", group, ok, len(paths), failed)
    return payload


def aligned_frames(registration: dict) -> list[str]:
    return [name for name, r in registration["results"].items()
            if r["status"] in ("ok", "reference")]
