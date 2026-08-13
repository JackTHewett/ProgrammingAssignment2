"""Stage 1 - separate the static foreground from the sky.

The roofline and railing are fixed in the frame while the sky rotates through
it, so a temporal median over the unaligned frames reconstructs the foreground
cleanly. The test for "building, not sky" is that the pixel sits *below* a
smoothed model of the sky: under heavy light pollution the structure is a
silhouette. Being signed is what keeps the lamp out, since the lamp is just as
static but sits above the sky level.

See ``build_mask`` for the two cues that were tried and rejected.

This mask matters twice over. Registration must ignore the foreground, because
a fixed roofline is a far stronger correlation signal than the stars and would
otherwise pin the alignment to the ground. Compositing needs it again to take
the ground from the unaligned stack.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .cache import Cache
from .config import Config
from .imageio_utils import autostretch, load_jpeg_proxy, save_png

log = logging.getLogger(__name__)


def _temporal_median(paths: list[Path], width: int) -> np.ndarray:
    """Median across the unaligned frames, on small proxies.

    Stars move between exposures, so they largely cancel; what survives is the
    static foreground plus the smooth skyglow.
    """
    stack = np.stack([load_jpeg_proxy(p, width, as_gray=True) for p in paths], axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def _touching_border(mask: np.ndarray, band: int) -> np.ndarray:
    """Keep only components that reach the frame border.

    The foreground enters from an edge; a star does not.
    """
    import cv2

    num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return np.zeros_like(mask, dtype=bool)

    border = np.zeros(mask.shape, dtype=bool)
    border[:band, :] = border[-band:, :] = True
    border[:, :band] = border[:, -band:] = True

    keep = set(np.unique(labels[border & mask])) - {0}
    return np.isin(labels, list(keep)) if keep else np.zeros_like(mask, dtype=bool)


def build_mask(paths: list[Path], cfg: Config) -> np.ndarray:
    """Return a proxy-resolution boolean mask: True where the foreground is.

    The test is that the foreground is *darker* than the local sky. Under heavy
    light pollution the roof and railing are silhouettes, so they sit well below
    a smoothed model of the sky, while the lamp sits equally far above it. Being
    signed is the whole point: it separates the building from the lamp, which is
    just as static and just as far from the sky level.

    Two cues were tried and dropped, both measured on synthetic frames with a
    known mask:

    * *temporal spread* - reads as sensor noise across ~99% of the frame, since
      stars are too sparse to lift a given sky pixel's MAD. It separated nothing.
    * *local edge energy* - only 38% of what it selected was really foreground.
      A steep skyglow gradient produces as much edge response as a roofline, and
      the building's dark interior produces almost none.

    Darkness alone scores ~99% purity, and hole filling recovers the interior.
    """
    import cv2

    mcfg = cfg.mask
    median = _temporal_median(paths, mcfg.proxy_width)

    # Smooth model of the sky, covering both the gradient and the lamp's glow.
    # The scale has to stay well above the foreground's own size, or the model
    # follows the building and its interior stops looking dark.
    sky_model = cv2.GaussianBlur(median, (0, 0), mcfg.sky_model_sigma)
    residual = median - sky_model
    scatter = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
    scatter = max(scatter, 1e-6)

    candidate = residual < -mcfg.dark_sigma * scatter

    candidate = _touching_border(candidate, mcfg.border_band)

    def disk(radius: int) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    m = candidate.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(mcfg.close_radius))
    # The opening radius has to stay small. Railing bars are only a few proxy
    # pixels wide, and an opening wide enough to be a useful speck filter erases
    # them outright - which leaves a periodic row of bars in the "sky" and hands
    # registration a degenerate, collinear set of control points. Speck removal
    # is the connected-component area filter's job below.
    if mcfg.open_radius > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(mcfg.open_radius))

    # Fill interior holes so windows and gaps in the railing count as foreground.
    filled = m.copy()
    h, w = m.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, flood, (0, 0), 1)
    m = m | (1 - filled)

    # Drop specks.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    min_area = mcfg.min_blob_fraction * m.size
    keep = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    mask = np.isin(labels, keep) if keep else np.zeros_like(m, dtype=bool)

    coverage = mask.mean()
    log.info("foreground mask covers %.1f%% of the frame", coverage * 100)
    if coverage > 0.6:
        log.warning("mask covers most of the frame - check %s and override with "
                    "--mask-file if it is wrong", "previews/mask_overlay.png")
    elif coverage < 0.005:
        log.warning("almost nothing masked as foreground - if the roofline is in "
                    "shot, check previews/mask_overlay.png")
    return mask


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def feathered_alpha(mask: np.ndarray, feather_px: int) -> np.ndarray:
    """Soft 0..1 foreground alpha, so the composite seam is not a hard cut."""
    import cv2

    m = mask.astype(np.uint8)
    if feather_px <= 0:
        return m.astype(np.float32)
    k = 2 * feather_px + 1
    return cv2.GaussianBlur(m.astype(np.float32), (k, k), feather_px / 2.5)


def sky_mask_for_registration(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Sky-only mask with a margin, used to blank the ground before alignment."""
    import cv2

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
        grown = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    else:
        grown = mask
    return ~grown


def write_preview(paths: list[Path], mask: np.ndarray, cfg: Config,
                  group: str = "") -> None:
    """Overlay the mask on the median frame so it can be checked by eye."""
    median = _temporal_median(paths[: min(len(paths), 40)], cfg.mask.proxy_width)
    base = autostretch(median)
    rgb = np.dstack([base, base, base])
    resized = resize_mask(mask, base.shape)
    rgb[..., 0] = np.where(resized, np.minimum(1.0, rgb[..., 0] * 0.4 + 0.6), rgb[..., 0])
    rgb[..., 1] = np.where(resized, rgb[..., 1] * 0.4, rgb[..., 1])
    rgb[..., 2] = np.where(resized, rgb[..., 2] * 0.4, rgb[..., 2])
    suffix = f"_{group}" if group else ""
    out = cfg.paths.previews / f"mask_overlay{suffix}.png"
    save_png(out, rgb)
    log.info("wrote %s - red is the masked foreground", out)


def load_override(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a hand-painted mask (white = foreground) at the given proxy shape."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L").resize((shape[1], shape[0]), Image.NEAREST)
        return np.asarray(im) > 127


def get_mask(paths: list[Path], cfg: Config, cache: Cache, group: str,
             override: Path | None = None) -> np.ndarray:
    """Cached mask for a group, or a user-supplied override."""
    if override is not None:
        proxy = load_jpeg_proxy(paths[0], cfg.mask.proxy_width, as_gray=True)
        log.info("using mask override %s", override)
        return load_override(override, proxy.shape)

    key = f"mask_{group}"
    if cache.has_array("mask", key):
        return cache.load_array("mask", key).astype(bool)
    mask = build_mask(paths, cfg)
    cache.save_array("mask", key, mask.astype(np.uint8))
    return mask
