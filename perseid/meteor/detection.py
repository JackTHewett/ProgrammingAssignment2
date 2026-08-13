"""Stage 3 - find linear transients and say what they probably are.

For each frame, a temporal median of its neighbours is subtracted. Stars and
skyglow are in every frame so they cancel; anything present in only one frame
survives. A probabilistic Hough transform then picks linear features out of the
residual, and each candidate is scored on length, straightness and its
brightness profile.

The classifier separates three things:

* **meteor** - one frame only, smoothly brightens and fades, does not repeat.
* **aircraft** - dashed or dotted from strobes, roughly uniform along its
  length, and usually continues into the next frame along the same vector.
* **satellite** - faint, thin, very long, and also continues across frames.

Aircraft and satellites are flagged, not silently dropped: everything reaches
the contact sheet with the classifier's guess so it can be overridden by eye.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .cache import Cache
from .config import Config
from .imageio_utils import load_jpeg, to_gray

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    frame: str
    frame_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_deg: float
    straightness_rms: float
    peak_sigma: float
    mean_sigma: float
    peak_ratio: float
    gap_fraction: float
    peak_count: int = 0
    profile: list[float] = field(default_factory=list)
    continues_in: list[str] = field(default_factory=list)
    classification: str = "unknown"
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    crop: Optional[str] = None
    # filled in from the review CSV
    verdict: str = ""

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def _erode_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    """Shrink a boolean mask by a few pixels."""
    import cv2

    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


# --- residual construction -------------------------------------------------

def temporal_residual(index: int, grays: dict[int, np.ndarray], order: list[int],
                      cfg: Config,
                      valid: dict[int, np.ndarray] | None = None
                      ) -> tuple[np.ndarray, float, np.ndarray]:
    """Frame minus the median of its neighbours, with a validity mask.

    Neighbours are taken in shot order and exclude the frame itself, so a real
    single-frame transient is never part of its own reference.

    The validity mask is essential, not a nicety. Registration warps each frame,
    which leaves zero-filled wedges along the edges; where one frame has data and
    its neighbours do not, the residual shows a hard step running the full length
    of the frame border. The Hough transform reads that as a magnificent, perfectly
    straight trail. Restricting detection to pixels where the frame *and* every
    contributing neighbour hold real data removes the artefact at its source.
    """
    half = cfg.detection.neighbour_halfwidth
    position = order.index(index)
    lo = max(0, position - half)
    hi = min(len(order), position + half + 1)
    neighbours = [order[p] for p in range(lo, hi) if order[p] != index]

    if len(neighbours) < 2:
        raise ValueError(f"frame {index} has too few neighbours for a median")

    reference = np.median(np.stack([grays[n] for n in neighbours], axis=0), axis=0)
    residual = grays[index] - reference

    if valid is not None:
        combined = valid[index].copy()
        for n in neighbours:
            combined &= valid[n]
    else:
        combined = np.ones(residual.shape, dtype=bool)

    # Robust noise estimate over valid pixels only; a bright trail barely moves
    # the MAD, but a border step would.
    sample = residual[combined] if combined.any() else residual.ravel()
    sigma = float(np.median(np.abs(sample - np.median(sample))) * 1.4826)
    return residual, max(sigma, 1e-6), combined


# --- Hough and segment merging --------------------------------------------

def _hough_segments(residual: np.ndarray, sigma: float, sky_mask: np.ndarray,
                    cfg: Config) -> list[tuple[float, float, float, float]]:
    import cv2

    dcfg = cfg.detection
    binary = (residual > dcfg.threshold_sigma * sigma) & sky_mask
    binary = binary.astype(np.uint8)

    # Close small gaps so a trail broken by noise still reads as one line, but
    # keep the kernel small enough that genuine strobe dashes survive.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(
        binary,
        rho=dcfg.hough_rho,
        theta=math.radians(dcfg.hough_theta_deg),
        threshold=dcfg.hough_threshold,
        minLineLength=dcfg.min_line_length,
        maxLineGap=dcfg.max_line_gap,
    )
    if lines is None or len(lines) == 0:
        return []
    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4). Reshaping covers both.
    return [tuple(float(v) for v in row) for row in np.asarray(lines).reshape(-1, 4)]


def _line_params(seg: tuple[float, float, float, float]) -> tuple[float, float]:
    """Angle in degrees (mod 180) and perpendicular distance from the origin."""
    x1, y1, x2, y2 = seg
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    theta = math.radians(angle)
    # Normal form: x*sin - y*cos = rho
    rho = x1 * math.sin(theta) - y1 * math.cos(theta)
    return angle, rho


def _merge_segments(segments: list[tuple[float, float, float, float]],
                    cfg: Config) -> list[tuple[float, float, float, float]]:
    """Collapse Hough fragments lying on the same line into one trail."""
    dcfg = cfg.detection
    groups: list[list[tuple[float, float, float, float]]] = []
    params: list[tuple[float, float]] = []

    for seg in segments:
        angle, rho = _line_params(seg)
        placed = False
        for i, (gangle, grho) in enumerate(params):
            dangle = abs(angle - gangle)
            dangle = min(dangle, 180.0 - dangle)
            if dangle <= dcfg.merge_angle_deg and abs(rho - grho) <= dcfg.merge_offset_px:
                groups[i].append(seg)
                n = len(groups[i])
                params[i] = (gangle + (angle - gangle) / n, grho + (rho - grho) / n)
                placed = True
                break
        if not placed:
            groups.append([seg])
            params.append((angle, rho))

    merged = []
    for group in groups:
        points = np.array([[s[0], s[1]] for s in group] + [[s[2], s[3]] for s in group])
        # The trail runs along the group's principal axis; its extent is the
        # spread of the endpoints projected onto that axis.
        centre = points.mean(axis=0)
        centred = points - centre
        _u, _s, vh = np.linalg.svd(centred, full_matrices=False)
        direction = vh[0]
        t = centred @ direction
        start = centre + direction * t.min()
        end = centre + direction * t.max()
        merged.append((float(start[0]), float(start[1]), float(end[0]), float(end[1])))
    return merged


# --- profiling -------------------------------------------------------------

def _sample_profile(residual: np.ndarray, seg: tuple[float, float, float, float],
                    cfg: Config) -> np.ndarray:
    """Brightness sampled along a trail, averaged across its width."""
    dcfg = cfg.detection
    x1, y1, x2, y2 = seg
    n = dcfg.profile_samples
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)

    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy) or 1.0
    # Unit normal, to average across the trail rather than sampling one thin line.
    nx, ny = -dy / norm, dx / norm

    h, w = residual.shape
    offsets = np.arange(-dcfg.profile_halfwidth, dcfg.profile_halfwidth + 1)
    samples = np.zeros((len(offsets), n), dtype=np.float32)
    for i, off in enumerate(offsets):
        px = np.clip(np.round(xs + nx * off).astype(int), 0, w - 1)
        py = np.clip(np.round(ys + ny * off).astype(int), 0, h - 1)
        samples[i] = residual[py, px]
    return samples.mean(axis=0)


def _straightness(residual: np.ndarray, seg: tuple[float, float, float, float],
                  sigma: float, cfg: Config) -> float:
    """RMS deviation of the bright pixels from the fitted straight line."""
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm

    # Pixels above threshold within a corridor around the segment.
    pad = 6
    xlo, xhi = int(min(x1, x2)) - pad, int(max(x1, x2)) + pad
    ylo, yhi = int(min(y1, y2)) - pad, int(max(y1, y2)) + pad
    h, w = residual.shape
    xlo, xhi = max(0, xlo), min(w, xhi)
    ylo, yhi = max(0, ylo), min(h, yhi)
    if xhi <= xlo or yhi <= ylo:
        return float("inf")

    window = residual[ylo:yhi, xlo:xhi]
    ys, xs = np.nonzero(window > cfg.detection.threshold_sigma * sigma)
    if len(xs) < 8:
        return float("inf")
    px = xs + xlo - x1
    py = ys + ylo - y1
    # Perpendicular distance from the line through (x1, y1) with direction u.
    perp = np.abs(px * uy - py * ux)
    # Only pixels that actually project onto the segment count.
    along = px * ux + py * uy
    keep = (along >= -pad) & (along <= norm + pad)
    if keep.sum() < 8:
        return float("inf")
    return float(np.sqrt(np.mean(perp[keep] ** 2)))


# --- classification --------------------------------------------------------

def _count_peaks(profile: np.ndarray, sigma: float) -> int:
    """Number of distinct maxima along the trail.

    This is the direct expression of "smoothly brightens and fades": a meteor's
    profile has exactly one broad peak, while a strobe-lit aircraft has one per
    flash. Peak-to-mean ratio does not capture it - sharp strobe dashes score a
    *higher* peak ratio than a real meteor, so on its own that feature points
    the wrong way for the most common kind of aircraft trail.
    """
    if len(profile) < 8:
        return 0
    # Light smoothing so noise ripples on one flank are not counted as peaks.
    kernel = np.array([0.25, 0.5, 0.25])
    smooth = np.convolve(profile, kernel, mode="same")
    threshold = 1.5 * sigma

    peaks = 0
    above = False
    for value in smooth:
        if value > threshold and not above:
            peaks += 1
            above = True
        elif value <= threshold:
            above = False
    return peaks


def _gap_fraction(profile: np.ndarray, sigma: float) -> float:
    """Fraction of the trail's interior that drops back to background.

    Strobe-lit aircraft read as a dotted line, so a good share of the samples
    between the endpoints sit at zero. A meteor's profile never breaks.
    """
    if len(profile) < 8:
        return 0.0
    interior = profile[2:-2]
    if len(interior) == 0:
        return 0.0
    return float(np.mean(interior < 1.5 * sigma))


def _continuation(candidate: Candidate, others: list[Candidate], cfg: Config,
                  order: list[str]) -> list[str]:
    """Trails in adjacent frames lying along the same vector.

    An object that reappears in the next frame on the same track is not a
    meteor - meteors last well under one 8s exposure and never repeat.
    """
    dcfg = cfg.detection
    try:
        position = order.index(candidate.frame)
    except ValueError:
        return []
    neighbours = set()
    for offset in (-1, 1):
        p = position + offset
        if 0 <= p < len(order):
            neighbours.add(order[p])

    angle_a, rho_a = _line_params((candidate.x1, candidate.y1, candidate.x2, candidate.y2))
    hits = []
    for other in others:
        if other.frame not in neighbours:
            continue
        angle_b, rho_b = _line_params((other.x1, other.y1, other.x2, other.y2))
        dangle = abs(angle_a - angle_b)
        dangle = min(dangle, 180.0 - dangle)
        if dangle <= dcfg.continuation_angle_deg and abs(rho_a - rho_b) <= dcfg.continuation_offset_px:
            hits.append(other.frame)
    return sorted(hits)


def classify(candidate: Candidate, cfg: Config) -> None:
    """Assign a classification, a confidence and the reasons behind it."""
    dcfg = cfg.detection
    reasons: list[str] = []
    scores = {"meteor": 0.0, "aircraft": 0.0, "satellite": 0.0}

    if candidate.continues_in:
        scores["aircraft"] += 2.0
        scores["satellite"] += 2.0
        scores["meteor"] -= 3.0
        reasons.append(f"continues into {', '.join(candidate.continues_in)}")

    if candidate.gap_fraction >= dcfg.dash_gap_fraction:
        scores["aircraft"] += 2.5
        scores["meteor"] -= 1.5
        reasons.append(f"dashed profile ({candidate.gap_fraction:.0%} gaps)")
    else:
        scores["meteor"] += 0.5

    if candidate.peak_count > dcfg.max_meteor_peaks:
        scores["aircraft"] += 2.0
        scores["meteor"] -= 2.0
        reasons.append(f"{candidate.peak_count} separate flashes")
    elif candidate.peak_count == 1:
        scores["meteor"] += 1.0
        reasons.append("single smooth peak")

    if candidate.peak_ratio < dcfg.uniform_peak_ratio:
        scores["aircraft"] += 1.0
        scores["satellite"] += 1.0
        scores["meteor"] -= 1.0
        reasons.append(f"uniform brightness (peak/mean {candidate.peak_ratio:.2f})")
    else:
        # Only a genuinely single-peaked profile is a brighten-and-fade; a high
        # ratio from strobe dashes must not be described that way on the sheet.
        scores["meteor"] += 1.5
        if candidate.peak_count <= dcfg.max_meteor_peaks:
            reasons.append(f"brightens and fades (peak/mean {candidate.peak_ratio:.2f})")
        else:
            reasons.append(f"peaked profile (peak/mean {candidate.peak_ratio:.2f})")

    if (candidate.length >= dcfg.satellite_min_length
            and candidate.peak_sigma <= dcfg.satellite_max_peak_sigma):
        scores["satellite"] += 2.0
        reasons.append(f"long and faint ({candidate.length:.0f}px, "
                       f"{candidate.peak_sigma:.1f} sigma)")

    if candidate.straightness_rms > dcfg.max_straight_rms:
        scores["meteor"] -= 1.0
        reasons.append(f"not straight (RMS {candidate.straightness_rms:.1f}px)")

    if not candidate.continues_in:
        scores["meteor"] += 1.0
        reasons.append("single frame")

    best = max(scores, key=lambda k: scores[k])
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1]

    candidate.classification = best if scores[best] > 0 else "unknown"
    candidate.confidence = round(min(1.0, max(0.0, margin / 4.0)), 2)
    candidate.reasons = reasons


# --- driver ----------------------------------------------------------------

def detect_group(paths: list[Path], registration: dict, sky_mask_proxy: np.ndarray,
                 cfg: Config, cache: Cache, group: str) -> list[dict]:
    """Run detection over every aligned frame in a group."""
    from .masking import resize_mask
    from .registration import apply_transform

    key = f"candidates_{group}"
    if cache.exists("detection", key):
        log.info("reusing cached detections for group %s", group)
        return cache.load_json("detection", key)

    usable = [p for p in paths if registration["results"].get(p.name, {}).get("status")
              in ("ok", "reference")]
    if len(usable) < 2 * cfg.detection.neighbour_halfwidth + 1:
        log.warning("group %s has too few aligned frames to build a temporal median", group)
        return []

    log.info("loading %d aligned frames for detection in group %s", len(usable), group)
    grays: dict[int, np.ndarray] = {}
    valid: dict[int, np.ndarray] = {}
    order: list[int] = []
    names: dict[int, str] = {}
    for i, path in enumerate(usable):
        gray = to_gray(load_jpeg(path))
        matrix = np.asarray(registration["results"][path.name]["matrix"], dtype=float)
        grays[i] = apply_transform(gray, matrix)
        # Warping a field of ones marks exactly which pixels came from real data.
        # Eroding trims the interpolated fringe at the boundary.
        warped_ones = apply_transform(np.ones_like(gray), matrix)
        valid[i] = _erode_bool(warped_ones > 0.999, cfg.detection.valid_erode_px)
        order.append(i)
        names[i] = path.name

    shape = grays[order[0]].shape
    sky_mask = resize_mask(sky_mask_proxy, shape)

    candidates: list[Candidate] = []
    for i in order:
        try:
            residual, sigma, frame_valid = temporal_residual(i, grays, order, cfg, valid)
        except ValueError as exc:
            log.debug("%s: %s", names[i], exc)
            continue

        segments = _hough_segments(residual, sigma, sky_mask & frame_valid, cfg)
        merged = _merge_segments(segments, cfg) if segments else []

        frame_candidates: list[Candidate] = []
        for seg in merged:
            x1, y1, x2, y2 = seg
            length = math.hypot(x2 - x1, y2 - y1)
            if length < cfg.detection.min_trail_length:
                continue

            profile = _sample_profile(residual, seg, cfg)
            peak = float(profile.max())
            mean = float(profile.mean())
            if mean <= 0:
                continue

            frame_candidates.append(Candidate(
                frame=names[i],
                frame_index=i,
                x1=x1, y1=y1, x2=x2, y2=y2,
                length=round(length, 1),
                angle_deg=round(math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0, 1),
                straightness_rms=round(_straightness(residual, seg, sigma, cfg), 2),
                peak_sigma=round(peak / sigma, 1),
                mean_sigma=round(mean / sigma, 1),
                peak_ratio=round(peak / mean, 2),
                gap_fraction=round(_gap_fraction(profile, sigma), 3),
                peak_count=_count_peaks(profile, sigma),
                profile=[round(float(v / sigma), 2) for v in profile],
            ))

        if len(frame_candidates) > cfg.detection.max_candidates_per_frame:
            frame_candidates.sort(key=lambda c: c.peak_sigma, reverse=True)
            log.warning("%s produced %d candidates - keeping the %d brightest; "
                        "consider raising detection.threshold_sigma",
                        names[i], len(frame_candidates), cfg.detection.max_candidates_per_frame)
            frame_candidates = frame_candidates[: cfg.detection.max_candidates_per_frame]

        candidates.extend(frame_candidates)

    # Cross-frame continuation, then classify.
    name_order = [names[i] for i in order]
    for candidate in candidates:
        candidate.continues_in = _continuation(candidate, candidates, cfg, name_order)
        classify(candidate, cfg)

    payload = [asdict(c) for c in candidates]
    cache.save_json("detection", key, payload)

    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.classification] = counts.get(c.classification, 0) + 1
    log.info("group %s: %d candidates (%s)", group, len(candidates),
             ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "none")
    if counts.get("meteor", 0) > 25:
        log.warning("%d meteor candidates is far more than a normal Perseid night - "
                    "the detector is probably catching aircraft. Raise "
                    "detection.threshold_sigma or min_trail_length.",
                    counts["meteor"])
    return payload
