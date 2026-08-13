"""Stage 6 - contact sheet of every candidate, for confirming by eye.

Each candidate gets a crop around its trail, labelled with the source filename,
the classifier's guess and the measurements behind it. Alongside the sheet goes
a CSV with one row per candidate and a ``verdict`` column: edit it to ``meteor``
or ``reject`` and the composite stage uses that instead of the classifier.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import numpy as np

from .config import Config
from .imageio_utils import autostretch, load_jpeg, save_jpeg

log = logging.getLogger(__name__)

CROP_SIZE = 320
COLUMNS = 5
LABEL_HEIGHT = 58
PADDING = 8

CLASS_COLOURS = {
    "meteor": (0.35, 1.0, 0.45),
    "aircraft": (1.0, 0.55, 0.3),
    "satellite": (0.5, 0.7, 1.0),
    "unknown": (0.75, 0.75, 0.75),
}


def _crop_around(image: np.ndarray, candidate: dict, size: int) -> np.ndarray:
    """Square crop centred on the trail, scaled so the whole trail fits."""
    import cv2

    h, w = image.shape[:2]
    cx = (candidate["x1"] + candidate["x2"]) / 2.0
    cy = (candidate["y1"] + candidate["y2"]) / 2.0
    extent = max(candidate["length"] * 1.4, size * 0.6)
    half = extent / 2.0

    x0 = int(round(max(0, min(w - 1, cx - half))))
    x1 = int(round(max(1, min(w, cx + half))))
    y0 = int(round(max(0, min(h - 1, cy - half))))
    y1 = int(round(max(1, min(h, cy + half))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return np.zeros((size, size, 3), dtype=np.float32)

    patch = image[y0:y1, x0:x1]
    patch = autostretch(patch, black_pct=1.0, white_pct=99.9, gamma=0.6)
    if patch.ndim == 2:
        patch = np.dstack([patch] * 3)
    resized = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)

    # Mark the detected trail so it is obvious what was measured.
    scale_x = size / max(x1 - x0, 1)
    scale_y = size / max(y1 - y0, 1)
    p1 = (int((candidate["x1"] - x0) * scale_x), int((candidate["y1"] - y0) * scale_y))
    p2 = (int((candidate["x2"] - x0) * scale_x), int((candidate["y2"] - y0) * scale_y))
    colour = CLASS_COLOURS.get(candidate.get("classification", "unknown"), (1, 1, 1))
    overlay = resized.copy()
    cv2.line(overlay, p1, p2, colour, 1, lineType=cv2.LINE_AA)
    return np.clip(resized * 0.82 + overlay * 0.18, 0.0, 1.0).astype(np.float32)


def _label(draw, x: int, y: int, candidate: dict, index: int, font, small_font) -> None:
    cls = candidate.get("classification", "unknown")
    colour = tuple(int(c * 255) for c in CLASS_COLOURS.get(cls, (1, 1, 1)))
    draw.text((x, y), f"[{index}] {candidate['frame']}", fill=(235, 235, 235), font=font)
    draw.text((x, y + 17), f"{cls}  ({candidate.get('confidence', 0):.2f})",
              fill=colour, font=font)
    detail = (f"{candidate['length']:.0f}px  peak {candidate['peak_sigma']:.0f}s  "
              f"pk/mn {candidate['peak_ratio']:.2f}  gaps {candidate['gap_fraction']:.0%}")
    draw.text((x, y + 34), detail, fill=(150, 150, 150), font=small_font)


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_contact_sheet(candidates: list[dict], paths_by_name: dict[str, Path],
                        registration: dict, cfg: Config, group: str) -> tuple[Path, Path]:
    """Write the contact sheet image and the review CSV."""
    from PIL import Image, ImageDraw

    from .registration import apply_transform

    if not candidates:
        log.warning("group %s: no candidates, no contact sheet", group)
        return None, None

    ordered = sorted(candidates, key=lambda c: (c["frame"], -c["peak_sigma"]))

    # Crops come from the aligned frames, matching what the detector measured.
    cache: dict[str, np.ndarray] = {}
    crops = []
    for candidate in ordered:
        name = candidate["frame"]
        if name not in cache:
            path = paths_by_name.get(name)
            if path is None:
                continue
            entry = registration["results"].get(name, {})
            image = load_jpeg(path)
            if entry.get("status") in ("ok", "reference"):
                image = apply_transform(image, np.asarray(entry["matrix"], dtype=float))
            cache[name] = image
            if len(cache) > 24:  # keep memory bounded on long runs
                cache.pop(next(iter(cache)))
        crop = _crop_around(cache[name], candidate, CROP_SIZE)
        crops.append((candidate, crop))
        crop_path = cfg.paths.crops / f"{group}_{name.split('.')[0]}_{len(crops):03d}.jpg"
        save_jpeg(crop_path, crop, quality=88)
        candidate["crop"] = str(crop_path)

    rows = math.ceil(len(crops) / COLUMNS)
    cell_w = CROP_SIZE + PADDING
    cell_h = CROP_SIZE + LABEL_HEIGHT + PADDING
    sheet = Image.new("RGB", (COLUMNS * cell_w + PADDING, rows * cell_h + PADDING + 34),
                      (14, 14, 18))
    draw = ImageDraw.Draw(sheet)
    font = _font(13)
    small = _font(11)
    title = _font(16)

    counts: dict[str, int] = {}
    for candidate, _crop in crops:
        cls = candidate.get("classification", "unknown")
        counts[cls] = counts.get(cls, 0) + 1
    summary = "   ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    draw.text((PADDING, 9), f"group {group}   {len(crops)} candidates     {summary}",
              fill=(235, 235, 235), font=title)

    for i, (candidate, crop) in enumerate(crops):
        col, row = i % COLUMNS, i // COLUMNS
        x = PADDING + col * cell_w
        y = 34 + PADDING + row * cell_h
        arr = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
        sheet.paste(Image.fromarray(arr), (x, y))
        _label(draw, x + 2, y + CROP_SIZE + 4, candidate, i + 1, font, small)

    sheet_path = cfg.paths.output / f"contact_sheet_{group}.jpg"
    sheet.save(sheet_path, quality=90)
    log.info("wrote %s (%d candidates)", sheet_path, len(crops))

    csv_path = cfg.paths.output / f"candidates_{group}.csv"
    # Verdicts already entered by hand must survive a rerun of this stage -
    # otherwise reviewing the sheet and then regenerating it silently throws the
    # review away.
    previous = load_verdicts(csv_path)
    if previous:
        log.info("preserving %d verdict(s) already entered in %s",
                 len(previous), csv_path.name)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "index", "frame", "classification", "confidence", "verdict",
            "length_px", "angle_deg", "peak_sigma", "peak_ratio", "gap_fraction",
            "peak_count", "straightness_rms", "continues_in", "reasons", "crop",
        ])
        for i, (candidate, _crop) in enumerate(crops, start=1):
            writer.writerow([
                i,
                candidate["frame"],
                candidate.get("classification", "unknown"),
                candidate.get("confidence", 0),
                previous.get((candidate["frame"], i), ""),
                candidate["length"],
                candidate["angle_deg"],
                candidate["peak_sigma"],
                candidate["peak_ratio"],
                candidate["gap_fraction"],
                candidate.get("peak_count", 0),
                candidate["straightness_rms"],
                ";".join(candidate.get("continues_in", [])),
                "; ".join(candidate.get("reasons", [])),
                candidate.get("crop", ""),
            ])
    log.info("wrote %s - set the verdict column to 'meteor' or 'reject'", csv_path)
    return sheet_path, csv_path


def load_verdicts(csv_path: Path) -> dict[tuple[str, int], str]:
    """Read hand-entered verdicts, keyed by (frame, index)."""
    verdicts: dict[tuple[str, int], str] = {}
    if not csv_path.exists():
        return verdicts
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            verdict = (row.get("verdict") or "").strip().lower()
            if verdict:
                verdicts[(row["frame"], int(row["index"]))] = verdict
    return verdicts


def confirmed_meteors(candidates: list[dict], csv_path: Path) -> list[dict]:
    """Candidates to blend: hand verdicts win, else the classifier's guess."""
    verdicts = load_verdicts(csv_path)
    ordered = sorted(candidates, key=lambda c: (c["frame"], -c["peak_sigma"]))

    confirmed = []
    for i, candidate in enumerate(ordered, start=1):
        verdict = verdicts.get((candidate["frame"], i))
        if verdict:
            candidate["verdict"] = verdict
            if verdict in ("meteor", "yes", "y", "confirm", "accept"):
                confirmed.append(candidate)
        elif candidate.get("classification") == "meteor":
            confirmed.append(candidate)
    return confirmed
