"""Stage 4 - subtract the light pollution gradient.

A low-order 2D polynomial is fitted to the sky background and subtracted. Stars
and the foreground are excluded from the fit by iterative sigma rejection,
otherwise the polynomial chases the stars instead of the skyglow, and the lamp
near the frame edge drags the whole surface with it.

The frame's median level is added back afterwards, so the result is a flat sky
at a sensible brightness rather than one centred on zero.

What this does and does not fix, measured on synthetic frames with a known
gradient: the broad skyglow ramp drops by roughly 2.6x in large-scale variation
at order 3, which is the point of the exercise. A *compact* source such as a
lamp just outside the frame is a different problem - its glow is far too steep
for any low-order surface, and raising the order enough to chase it starts
eating real sky. If a lamp dominates a corner of the finished stack, mask it out
or crop it rather than turning the polynomial order up.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import Config

log = logging.getLogger(__name__)


def _design_matrix(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Columns for every x^i * y^j term with i + j <= order."""
    columns = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            columns.append((x ** i) * (y ** j))
    return np.stack(columns, axis=-1)


def fit_background(channel: np.ndarray, valid: np.ndarray, cfg: Config) -> np.ndarray:
    """Fit and evaluate the background surface for one channel.

    ``valid`` marks pixels eligible for the fit - sky, not foreground.
    """
    gcfg = cfg.gradient
    h, w = channel.shape
    step = max(1, gcfg.sample_step)

    # Normalised coordinates keep the Vandermonde matrix well conditioned.
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    xs = (xx.ravel() / max(w - 1, 1)) * 2.0 - 1.0
    ys = (yy.ravel() / max(h - 1, 1)) * 2.0 - 1.0
    values = channel[::step, ::step].ravel()
    usable = valid[::step, ::step].ravel()

    xs, ys, values = xs[usable], ys[usable], values[usable]
    if len(values) < 64:
        log.warning("not enough sky pixels for a gradient fit; skipping")
        return np.zeros_like(channel)

    keep = np.ones(len(values), dtype=bool)
    coefficients = None
    for iteration in range(max(1, gcfg.iterations)):
        design = _design_matrix(xs[keep], ys[keep], gcfg.poly_order)
        coefficients, *_ = np.linalg.lstsq(design, values[keep], rcond=None)

        model_all = _design_matrix(xs, ys, gcfg.poly_order) @ coefficients
        residual = values - model_all
        sigma = float(np.std(residual[keep]))
        if sigma <= 0:
            break
        # Reject only positive outliers: stars sit above the background, and
        # clipping the negative side would eat genuine dark sky.
        updated = residual < gcfg.star_reject_sigma * sigma
        if updated.sum() < 64:
            break
        if np.array_equal(updated, keep):
            log.debug("gradient fit converged after %d iterations", iteration + 1)
            break
        keep = updated

    yy_full, xx_full = np.mgrid[0:h, 0:w]
    xs_full = (xx_full / max(w - 1, 1)) * 2.0 - 1.0
    ys_full = (yy_full / max(h - 1, 1)) * 2.0 - 1.0
    surface = _design_matrix(xs_full.ravel(), ys_full.ravel(),
                             gcfg.poly_order) @ coefficients
    return surface.reshape(h, w).astype(np.float32)


def remove_gradient(image: np.ndarray, sky_mask: np.ndarray, cfg: Config) -> np.ndarray:
    """Subtract the fitted background from every channel, preserving level."""
    if image.ndim == 2:
        channels = [image]
    else:
        channels = [image[..., c] for c in range(image.shape[2])]

    output = []
    for channel in channels:
        surface = fit_background(channel, sky_mask, cfg)
        level = float(np.median(channel[sky_mask])) if sky_mask.any() else float(np.median(channel))
        output.append(channel - surface + level)

    result = output[0] if image.ndim == 2 else np.stack(output, axis=-1)
    log.info("gradient removed (order %d polynomial)", cfg.gradient.poly_order)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
