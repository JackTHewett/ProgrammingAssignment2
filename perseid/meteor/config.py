"""Tunables and paths for the Perseid pipeline.

Everything the pipeline might reasonably need to adjust lives here so the
processing modules stay free of magic numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Paths:
    source: Path = Path(".")
    output: Path = Path("output")
    cache: Path = Path("output/.cache")

    @property
    def previews(self) -> Path:
        return self.output / "previews"

    @property
    def crops(self) -> Path:
        return self.output / "crops"

    def ensure(self) -> None:
        for d in (self.output, self.cache, self.previews, self.crops):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class GroupingConfig:
    """Stage 0: how frames get split into processing groups."""

    # Exposures within this relative tolerance count as the same setting.
    exposure_tolerance: float = 0.05
    # Width of the downsampled proxy used for framing correlation.
    proxy_width: int = 512
    # A frame-to-frame shift beyond this fraction of image width starts a new
    # framing group. Small table bumps fall well under it; a refocus does not.
    framing_break_fraction: float = 0.02
    # Groups smaller than this are reported but not processed.
    min_group_size: int = 5


@dataclass
class MaskConfig:
    """Stage 1: static foreground (roof/railing) versus sky."""

    proxy_width: int = 1024
    # Scale of the smoothed sky model, in proxy pixels. Must stay well above the
    # foreground's own size or the model tracks the building itself.
    sky_model_sigma: float = 120.0
    # How far below the local sky level a pixel must sit to read as silhouette,
    # in robust sigma. Signed on purpose: it must not catch the lamp.
    dark_sigma: float = 1.5
    # Foreground must connect to the image border to be accepted, which keeps
    # bright stars and the lamp glow from being labelled as building.
    border_band: int = 8
    # Morphological cleanup, in proxy pixels. Keep open_radius small: it erases
    # anything thinner than its diameter, and the railing bars are only a few
    # proxy pixels wide.
    close_radius: int = 9
    open_radius: int = 2
    # Smallest accepted foreground blob, as a fraction of frame area.
    min_blob_fraction: float = 0.002
    # Feather width for the final composite blend, in full-resolution pixels.
    feather_px: int = 48
    # Registration ignores this much extra margin around the foreground.
    registration_dilate_px: int = 24


@dataclass
class RegistrationConfig:
    """Stage 2: star alignment."""

    # astroalign settings.
    detection_sigma: float = 5.0
    max_control_points: int = 60
    min_area: int = 5
    # A solve is rejected if astroalign matches fewer than this many stars.
    min_matched_stars: int = 8
    # ...or if the residual scatter of matched stars exceeds this, in pixels.
    max_residual_px: float = 2.5
    # Sanity bounds on the recovered transform. A table bump plus Earth rotation
    # over one group stays far inside these.
    max_scale_deviation: float = 0.02
    max_rotation_deg: float = 8.0
    max_translation_fraction: float = 0.25
    # Guards against degenerate control-point geometry. Points strung along a
    # single line - a railing left showing through the mask, say - fit a
    # rotation just as well upside down, so the solve must be rejected.
    min_point_axis_ratio: float = 0.10
    min_point_spread_fraction: float = 0.12
    # DAOStarFinder fallback.
    fallback_fwhm: float = 3.5
    fallback_threshold_sigma: float = 4.0
    fallback_max_stars: int = 120


@dataclass
class DetectionConfig:
    """Stage 3: meteor / aircraft / satellite candidates."""

    # Number of neighbouring frames each side used for the temporal median.
    neighbour_halfwidth: int = 3
    # Residual detection threshold, in robust sigma above the residual noise.
    threshold_sigma: float = 5.0
    # Pixels trimmed from the edge of each frame's valid (non-zero-filled) region
    # after warping, to clear the interpolated fringe.
    valid_erode_px: int = 6
    # Probabilistic Hough parameters, in full-resolution pixels.
    hough_rho: float = 1.0
    hough_theta_deg: float = 0.5
    hough_threshold: int = 40
    min_line_length: int = 60
    # Strobe-lit aircraft are dashed, and a real strobe flashes about once a
    # second across an 8s exposure, so the bridging gap has to be generous or
    # dashed trails are missed entirely rather than flagged.
    max_line_gap: int = 30
    # Segments closer than this in angle and offset are merged into one trail.
    merge_angle_deg: float = 4.0
    merge_offset_px: float = 18.0
    # A trail shorter than this is discarded as noise or a hot pixel cluster.
    min_trail_length: int = 70
    # Sampling of the brightness profile along a trail.
    profile_samples: int = 96
    profile_halfwidth: int = 3

    # --- classifier thresholds ---
    # Fraction of profile samples below background at which a trail reads as
    # dashed (strobe-lit aircraft).
    dash_gap_fraction: float = 0.18
    # Peak-to-mean ratio below which a trail reads as uniform (a steady nav
    # light) rather than a brighten-and-fade. Weak evidence on its own: sharp
    # strobe dashes score a higher ratio than a real meteor, so gap fraction and
    # peak count carry the discrimination.
    uniform_peak_ratio: float = 1.25
    # More separate maxima than this means strobe flashes, not one meteor.
    max_meteor_peaks: int = 2
    # Straightness: RMS deviation from the fitted line, in pixels.
    max_straight_rms: float = 2.5
    # A trail in an adjacent frame within this angle and offset counts as the
    # same object continuing, which rules out a meteor.
    continuation_angle_deg: float = 6.0
    continuation_offset_px: float = 60.0
    # Satellites: long, thin, faint.
    satellite_min_length: int = 400
    satellite_max_peak_sigma: float = 12.0
    # Guard rail. If a single frame yields more than this, thresholds are loose.
    max_candidates_per_frame: int = 12


@dataclass
class GradientConfig:
    """Stage 4: light pollution gradient."""

    poly_order: int = 3
    # Sky pixels above this many sigma are stars and excluded from the fit.
    star_reject_sigma: float = 2.0
    # Iterations of fit-and-reject.
    iterations: int = 4
    # The fit is done on a decimated grid for speed.
    sample_step: int = 8


@dataclass
class StackConfig:
    """Stage 5: stacking and compositing."""

    method: str = "sigma_clip"  # "sigma_clip" or "mean"
    sigma_low: float = 3.0
    sigma_high: float = 2.5
    # Prefer RAF via rawpy when a matching raw file sits next to the JPEG.
    prefer_raw: bool = True
    raw_output_bps: int = 16
    raw_use_camera_wb: bool = True
    # Meteors are blended within this margin around the detected trail.
    meteor_blend_margin_px: int = 40
    meteor_blend_feather_px: int = 12


@dataclass
class TimelapseConfig:
    fps: int = 24
    width: int = 1920
    crf: int = 18
    codec: str = "libx264"


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    gradient: GradientConfig = field(default_factory=GradientConfig)
    stack: StackConfig = field(default_factory=StackConfig)
    timelapse: TimelapseConfig = field(default_factory=TimelapseConfig)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["paths"] = {k: str(v) for k, v in payload["paths"].items()}
        return json.dumps(payload, indent=2)

    def save(self, path: Path) -> None:
        path.write_text(self.to_json())
