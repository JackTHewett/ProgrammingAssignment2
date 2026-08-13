#!/usr/bin/env python3
"""Generate a synthetic night for testing the pipeline without real photos.

The scene deliberately reproduces every awkward property of the real session:

* a rotating starfield (Earth's spin) plus small random jitter (table bumps),
* one large framing jump partway through (the refocus),
* two exposure settings, written into real EXIF,
* a static roofline and railing along the bottom,
* a strong light pollution gradient with a lamp near the frame edge,
* single-frame meteors that smoothly brighten and fade,
* dashed aircraft trails that continue across consecutive frames.

Ground truth is written to ``truth.json`` so detection can be scored.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 1600, 1067
N_STARS = 420
RNG = np.random.default_rng(20250812)

# frame index -> (x1, y1, x2, y2) in image coordinates
METEORS = {6: (300, 180, 470, 300), 19: (900, 120, 1120, 340), 31: (500, 400, 660, 250)}
# first frame index of each aircraft pass; it continues for 3 frames
AIRCRAFT = {11: (120, 500, 520, 380), 26: (1400, 200, 1000, 460)}


def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """sRGB transfer curve, matching meteor.imageio_utils.linear_to_srgb.

    Kept local so the generator runs standalone.
    """
    a = np.clip(img, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92,
                    1.055 * np.power(a, 1 / 2.4) - 0.055).astype(np.float32)


def _gaussian_blob(canvas: np.ndarray, x: float, y: float, flux: float, sigma: float) -> None:
    """Add a Gaussian point source, touching only the pixels that matter."""
    radius = int(math.ceil(sigma * 3.5))
    x0, x1 = max(0, int(x) - radius), min(canvas.shape[1], int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(canvas.shape[0], int(y) + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] += flux * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)))


def _make_stars() -> tuple[np.ndarray, np.ndarray]:
    """Star positions in sky coordinates, with a realistic brightness spread."""
    margin = 400  # stars can rotate into frame from outside
    xs = RNG.uniform(-margin, WIDTH + margin, N_STARS)
    ys = RNG.uniform(-margin, HEIGHT + margin, N_STARS)
    # Many faint, few bright.
    flux = RNG.pareto(1.8, N_STARS) * 0.045 + 0.02
    return np.column_stack([xs, ys]), np.clip(flux, 0, 1.1)


def _sky_transform(frame: int, break_at: int) -> tuple[float, float, float]:
    """Rotation and translation of the sky for a given frame."""
    # Earth's rotation, exaggerated so alignment is actually exercised.
    angle = math.radians(0.022 * frame)
    # Table bumps: a slow drift plus per-frame jitter.
    jitter = RNG.normal(0, 1.4, 2)
    dx = 0.35 * frame + jitter[0]
    dy = -0.18 * frame + jitter[1]
    # The deliberate refocus knocks the framing hard.
    if frame >= break_at:
        dx += 78.0
        dy += -46.0
    return angle, dx, dy


def _render_sky(frame: int, stars: np.ndarray, flux: np.ndarray, break_at: int,
                exposure: float) -> np.ndarray:
    canvas = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    angle, dx, dy = _sky_transform(frame, break_at)

    # Rotate about a pole placed well off frame, as a real polar axis would be.
    pole = np.array([WIDTH * 0.5, HEIGHT * 2.4])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    positions = (stars - pole) @ rotation.T + pole + np.array([dx, dy])

    scale = exposure / 8.0  # longer exposure collects more signal
    for (x, y), f in zip(positions, flux):
        if -20 <= x < WIDTH + 20 and -20 <= y < HEIGHT + 20:
            _gaussian_blob(canvas, x, y, f * scale, sigma=1.5)
    return canvas


def _light_pollution() -> np.ndarray:
    """Smooth gradient plus a bright lamp near the left edge."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    nx, ny = xx / WIDTH, yy / HEIGHT
    gradient = 0.030 + 0.075 * (1 - ny) ** 1.6 + 0.028 * nx + 0.012 * nx * (1 - ny)
    lamp = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    _gaussian_blob(lamp, 40, HEIGHT * 0.62, 0.55, sigma=190)
    return (gradient + lamp).astype(np.float32)


def _foreground() -> tuple[np.ndarray, np.ndarray]:
    """Static roof and railing, in linear light. Returns (image, mask).

    These values are deliberately far below the sky level: a building against a
    light-polluted sky is a silhouette, which is the contrast the mask stage
    depends on.
    """
    img = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)

    roof_y = int(HEIGHT * 0.78)
    # A sloped roofline, so the mask boundary is not a trivial straight cut.
    for x in range(WIDTH):
        top = roof_y + int(26 * math.sin(x / WIDTH * math.pi * 1.3))
        mask[top:, x] = True
        img[top:, x] = 0.0065

    # Railing: vertical bars above the roof with a horizontal rail.
    rail_top = roof_y - 88
    for x in range(60, WIDTH - 60, 96):
        mask[rail_top:roof_y, x:x + 9] = True
        img[rail_top:roof_y, x:x + 9] = 0.0094
    mask[rail_top:rail_top + 11, :] = True
    img[rail_top:rail_top + 11, :] = 0.0094

    # A couple of lit windows, so the foreground is not uniformly dark.
    img[roof_y + 60:roof_y + 110, 300:360] = 0.10
    img[roof_y + 60:roof_y + 110, 900:960] = 0.08
    return img, mask


def _add_meteor(canvas: np.ndarray, seg: tuple[float, float, float, float]) -> None:
    """A meteor: smooth brighten-and-fade along a continuous streak."""
    x1, y1, x2, y2 = seg
    n = 400
    for i in range(n):
        t = i / (n - 1)
        # Rises fast, fades slower - the classic meteor light curve.
        envelope = math.sin(math.pi * t) ** 0.7 * (1.0 - 0.35 * t)
        _gaussian_blob(canvas, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t,
                       0.030 * envelope, sigma=1.7)


def _add_aircraft(canvas: np.ndarray, seg: tuple[float, float, float, float],
                  phase: float) -> None:
    """An aircraft: uniform brightness broken into strobe dashes."""
    x1, y1, x2, y2 = seg
    n = 400
    for i in range(n):
        t = i / (n - 1)
        # Dash pattern: on for roughly a third of each cycle.
        if ((t * 13 + phase) % 1.0) > 0.36:
            continue
        _gaussian_blob(canvas, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t,
                       0.020, sigma=1.6)


def _write_exif(path: Path, array: np.ndarray, exposure: float, iso: int,
                stamp: datetime) -> None:
    from PIL.ExifTags import Base
    from PIL.TiffImagePlugin import IFDRational

    image = Image.fromarray(array, mode="RGB")
    exif = image.getexif()
    exif[Base.Make.value] = "FUJIFILM"
    exif[Base.Model.value] = "X-T30 III"

    # IFDRational makes Pillow emit a real RATIONAL tag. A plain (num, den)
    # tuple is written as two SHORTs, which is not what a camera produces.
    ifd = exif.get_ifd(0x8769)
    ifd[Base.DateTimeOriginal.value] = stamp.strftime("%Y:%m:%d %H:%M:%S")
    ifd[Base.ExposureTime.value] = IFDRational(int(exposure * 1000), 1000)
    ifd[Base.ISOSpeedRatings.value] = iso
    ifd[Base.FNumber.value] = IFDRational(35, 10)
    ifd[Base.FocalLength.value] = IFDRational(130, 10)
    ifd[Base.ExifImageWidth.value] = WIDTH
    ifd[Base.ExifImageHeight.value] = HEIGHT

    image.save(path, quality=94, subsampling=0, exif=exif.tobytes())


def generate(out_dir: Path, n_frames: int = 40, break_at: int = 25,
             exposure_switch: int = 15) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stars, flux = _make_stars()
    pollution = _light_pollution()
    fg_img, fg_mask = _foreground()

    start = datetime(2025, 8, 12, 23, 41, 0)
    truth = {"meteors": [], "aircraft": [], "framing_break_at": break_at,
             "exposure_switch_at": exposure_switch, "frames": []}
    clock = start

    for frame in range(n_frames):
        if frame < exposure_switch:
            exposure, iso = 15.0, 800
        else:
            exposure, iso = 8.0, 3200

        sky = _render_sky(frame, stars, flux, break_at, exposure)
        canvas = sky + pollution * (exposure / 10.0)

        if frame in METEORS:
            _add_meteor(canvas, METEORS[frame])
            truth["meteors"].append({"frame_index": frame, "segment": METEORS[frame]})

        for first, seg in AIRCRAFT.items():
            if first <= frame < first + 3:
                step = frame - first
                x1, y1, x2, y2 = seg
                # The aircraft advances along its own vector between frames.
                shift = 0.34 * step
                dx, dy = (x2 - x1) * shift, (y2 - y1) * shift
                moved = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                _add_aircraft(canvas, moved, phase=0.27 * step)
                truth["aircraft"].append({"frame_index": frame, "segment": moved})

        # Foreground sits in front of everything.
        canvas = np.where(fg_mask, fg_img, canvas)
        # Sensor noise belongs in linear light, before the transfer curve.
        canvas += RNG.normal(0, 0.0075, canvas.shape).astype(np.float32)

        # Everything above is linear light, as a sensor records it. The sRGB
        # transfer curve goes on last, which is what the camera's JPEG engine
        # does. Building the scene in encoded space instead would make the
        # skyglow polynomial in the wrong domain and quietly bias any test of
        # the gradient fit.
        rgb_linear = np.clip(np.dstack([canvas * 0.94, canvas * 0.97, canvas * 1.06]), 0, 1)
        array = (_linear_to_srgb(rgb_linear) * 255 + 0.5).astype(np.uint8)

        name = f"DSCF{4954 + frame:04d}.JPG"
        _write_exif(out_dir / name, array, exposure, iso, clock)
        truth["frames"].append({"index": frame, "name": name, "exposure": exposure,
                                "iso": iso, "timestamp": clock.isoformat()})
        clock += timedelta(seconds=exposure + 2)

    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2))
    np.save(out_dir / "truth_mask.npy", fg_mask)
    print(f"wrote {n_frames} frames to {out_dir}")
    print(f"  meteors at frames {sorted(METEORS)}")
    print(f"  aircraft starting at frames {sorted(AIRCRAFT)} (3 frames each)")
    print(f"  exposure switch at frame {exposure_switch}, framing break at {break_at}")
    return truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()
    generate(args.out, n_frames=args.frames)
