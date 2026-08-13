"""Stage runner.

Every stage is separately invocable and reads its inputs from the cache, so
recompositing after changing a verdict never redoes registration:

    python run.py scan                 # EXIF + grouping report, always cheap
    python run.py mask                 # foreground mask + overlay preview
    python run.py register             # star alignment            (slow)
    python run.py detect               # meteor candidates         (slow)
    python run.py sheet                # contact sheet + review CSV
    python run.py composite            # stacks, gradient, composites (slow)
    python run.py timelapse            # 24fps MP4
    python run.py all                  # everything in order

Slow stages ask before running across the whole set unless ``--yes`` is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


from . import contact_sheet as sheet_mod
from . import detection, exif_scan, gradient, masking, registration, stacking, timelapse
from .cache import Cache
from .config import Config, Paths
from .imageio_utils import autostretch, linear_to_srgb, save_jpeg, save_tiff16

log = logging.getLogger("meteor")


# --- helpers ---------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _confirm(stage: str, frames: int, estimate_s: float, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        log.warning("stage '%s' needs confirmation but stdin is not a terminal; "
                    "rerun with --yes", stage)
        return False
    minutes = estimate_s / 60.0
    print(f"\nStage '{stage}' will process {frames} frames "
          f"(rough estimate {minutes:.0f}-{minutes * 2.5:.0f} min).")
    answer = input("Continue? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def _load_scan(cfg: Config, cache: Cache) -> list[dict]:
    try:
        return cache.load_json("scan", "frames")
    except FileNotFoundError:
        log.info("no cached scan; running it now")
        frames = exif_scan.scan(cfg)
        cache.save_json("scan", "frames", frames)
        return frames


def _group_paths(frames: list[dict], group: str) -> list[Path]:
    return [Path(f["path"]) for f in exif_scan.group_members(frames, group)]


def _target_groups(frames: list[dict], cfg: Config, requested: str | None) -> list[str]:
    available = exif_scan.processable_groups(frames, cfg)
    if not available:
        raise SystemExit("no group is large enough to process; see the scan report")
    if requested:
        if requested not in available:
            raise SystemExit(f"group {requested!r} not in {available}")
        return [requested]
    return available


# --- stages ----------------------------------------------------------------

def stage_scan(cfg: Config, cache: Cache, args) -> list[dict]:
    frames = exif_scan.scan(cfg)
    cache.save_json("scan", "frames", frames)
    print()
    print(exif_scan.summarise(frames, cfg))
    print()
    out = cfg.paths.output / "session_summary.txt"
    out.write_text(exif_scan.summarise(frames, cfg))
    log.info("wrote %s", out)
    return frames


def stage_mask(cfg: Config, cache: Cache, args) -> None:
    frames = _load_scan(cfg, cache)
    override = Path(args.mask_file) if args.mask_file else None
    for group in _target_groups(frames, cfg, args.group):
        paths = _group_paths(frames, group)
        log.info("building foreground mask for group %s from %d frames", group, len(paths))
        mask = masking.get_mask(paths, cfg, cache, group, override)
        masking.write_preview(paths, mask, cfg, group)
        log.info("group %s: foreground covers %.1f%% of the frame",
                 group, mask.mean() * 100)


def stage_register(cfg: Config, cache: Cache, args) -> None:
    frames = _load_scan(cfg, cache)
    groups = _target_groups(frames, cfg, args.group)
    total = sum(len(_group_paths(frames, g)) for g in groups)
    if not _confirm("register", total, total * 2.0, args.yes):
        log.info("skipped")
        return

    for group in groups:
        paths = _group_paths(frames, group)
        mask = masking.get_mask(paths, cfg, cache, group,
                                Path(args.mask_file) if args.mask_file else None)
        sky = masking.sky_mask_for_registration(mask, cfg.mask.registration_dilate_px)
        started = time.time()
        result = registration.register_group(paths, sky, cfg, cache, group)
        log.info("group %s registered in %.1fs", group, time.time() - started)
        for name, entry in result["results"].items():
            if entry["status"] == "failed":
                log.info("  rejected %s: %s", name, entry["reason"])


def stage_detect(cfg: Config, cache: Cache, args) -> None:
    frames = _load_scan(cfg, cache)
    groups = _target_groups(frames, cfg, args.group)
    total = sum(len(_group_paths(frames, g)) for g in groups)
    if not _confirm("detect", total, total * 1.5, args.yes):
        log.info("skipped")
        return

    for group in groups:
        paths = _group_paths(frames, group)
        try:
            reg = cache.load_json("registration", f"transforms_{group}")
        except FileNotFoundError:
            raise SystemExit(f"group {group} is not registered yet - run 'register' first")
        mask = masking.get_mask(paths, cfg, cache, group,
                                Path(args.mask_file) if args.mask_file else None)
        sky = masking.sky_mask_for_registration(mask, cfg.mask.registration_dilate_px)
        detection.detect_group(paths, reg, sky, cfg, cache, group)


def stage_sheet(cfg: Config, cache: Cache, args) -> None:
    frames = _load_scan(cfg, cache)
    for group in _target_groups(frames, cfg, args.group):
        try:
            candidates = cache.load_json("detection", f"candidates_{group}")
            reg = cache.load_json("registration", f"transforms_{group}")
        except FileNotFoundError as exc:
            raise SystemExit(f"{exc} - run 'detect' first")
        paths_by_name = {p.name: p for p in _group_paths(frames, group)}
        sheet_mod.build_contact_sheet(candidates, paths_by_name, reg, cfg, group)


def stage_composite(cfg: Config, cache: Cache, args) -> None:
    frames = _load_scan(cfg, cache)
    groups = _target_groups(frames, cfg, args.group)
    total = sum(len(_group_paths(frames, g)) for g in groups)
    if not _confirm("composite", total, total * 4.0, args.yes):
        log.info("skipped")
        return

    for group in groups:
        paths = _group_paths(frames, group)
        try:
            reg = cache.load_json("registration", f"transforms_{group}")
        except FileNotFoundError:
            raise SystemExit(f"group {group} is not registered yet - run 'register' first")

        mask_proxy = masking.get_mask(paths, cfg, cache, group,
                                      Path(args.mask_file) if args.mask_file else None)

        sky, sky_meta = stacking.stack_sky(paths, reg, cfg, cache, group)
        ground, _ground_meta = stacking.stack_ground(paths, cfg, cache, group)

        mask_full = masking.resize_mask(mask_proxy, sky.shape[:2])

        # Stacking happens in linear light, because averaging frames is
        # arithmetic on photons and because RAF and JPEG have to meet in one
        # space. Flattening happens after the transfer curve is back on, which
        # is not where the physics would put it but is measurably better: the
        # curve compresses a lamp's dynamic range from roughly 18x to 4x, and a
        # low-order polynomial can actually follow that. Measured on the
        # synthetic scene, fitting in display space left 0.029 residual skyglow
        # against 0.055 for the same fit done in linear.
        sky_display = linear_to_srgb(sky)
        ground_display = linear_to_srgb(ground)

        flat_sky = gradient.remove_gradient(sky_display, ~mask_full, cfg)

        alpha = masking.feathered_alpha(mask_full, cfg.mask.feather_px)
        background = stacking.composite(flat_sky, ground_display, alpha)

        save_tiff16(cfg.paths.output / f"sky_stack_{group}.tif", background)
        save_jpeg(cfg.paths.previews / f"sky_stack_{group}.jpg", autostretch(background))
        log.info("group %s: sky stack from %d frames (%s)",
                 group, sky_meta["frames"], sky_meta.get("sources"))

        # --- meteor composite ---
        try:
            candidates = cache.load_json("detection", f"candidates_{group}")
        except FileNotFoundError:
            log.warning("group %s has no detections; skipping the meteor composite", group)
            continue

        csv_path = cfg.paths.output / f"candidates_{group}.csv"
        confirmed = sheet_mod.confirmed_meteors(candidates, csv_path)
        if not confirmed:
            log.warning("group %s: nothing confirmed as a meteor, so the composite "
                        "would match the plain stack; skipping", group)
            continue

        paths_by_name = {p.name: p for p in paths}
        composite, blended = stacking.meteor_composite(
            background, confirmed, paths_by_name, reg, cfg)
        save_tiff16(cfg.paths.output / f"meteor_composite_{group}.tif", composite)
        save_jpeg(cfg.paths.previews / f"meteor_composite_{group}.jpg",
                  autostretch(composite))
        log.info("group %s: meteor composite with %d trails", group, blended)


def stage_timelapse(cfg: Config, cache: Cache, args) -> None:
    frames = _load_scan(cfg, cache)
    # Shot order across the whole night, regardless of grouping - the ground is
    # fixed and unaligned, so a framing break is just a jump cut.
    paths = [Path(f["path"]) for f in sorted(frames, key=lambda f: f["index"])]
    timelapse.build_timelapse(paths, cfg)


# --- summary ---------------------------------------------------------------

def print_summary(cfg: Config, cache: Cache) -> None:
    frames = _load_scan(cfg, cache)
    groups = exif_scan.processable_groups(frames, cfg)

    print()
    print("=" * 72)
    print("RUN SUMMARY")
    print("=" * 72)
    print(f"  frames found        {len(frames)}")

    skipped = len(frames) - sum(len(exif_scan.group_members(frames, g)) for g in groups)
    if skipped:
        print(f"  in groups too small {skipped}  (not processed)")

    total_aligned = total_failed = 0
    reasons: dict[str, int] = {}
    meteors = aircraft = satellites = unknown = 0

    for group in groups:
        try:
            reg = cache.load_json("registration", f"transforms_{group}")
        except FileNotFoundError:
            continue
        total_aligned += reg["aligned"]
        total_failed += reg["failed"]
        for entry in reg["results"].values():
            if entry["status"] == "failed":
                key = entry["reason"].split("(")[0].strip()
                reasons[key] = reasons.get(key, 0) + 1

        try:
            candidates = cache.load_json("detection", f"candidates_{group}")
        except FileNotFoundError:
            continue
        csv_path = cfg.paths.output / f"candidates_{group}.csv"
        confirmed = {id(c) for c in sheet_mod.confirmed_meteors(candidates, csv_path)}
        for candidate in candidates:
            cls = candidate.get("classification", "unknown")
            if cls == "aircraft":
                aircraft += 1
            elif cls == "satellite":
                satellites += 1
            elif cls == "unknown":
                unknown += 1
        meteors += len(confirmed)

    print(f"  frames aligned      {total_aligned}")
    print(f"  frames rejected     {total_failed}")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {count:>3}  {reason}")
    print(f"  meteors confirmed   {meteors}")
    print(f"  aircraft rejected   {aircraft}")
    print(f"  satellites rejected {satellites}")
    if unknown:
        print(f"  unclassified        {unknown}")
    print(f"  output              {cfg.paths.output.resolve()}")
    print("=" * 72)


# --- entry point -----------------------------------------------------------

STAGES = {
    "scan": stage_scan,
    "mask": stage_mask,
    "register": stage_register,
    "detect": stage_detect,
    "sheet": stage_sheet,
    "composite": stage_composite,
    "timelapse": stage_timelapse,
}

ALL_ORDER = ["scan", "mask", "register", "detect", "sheet", "composite", "timelapse"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Perseid meteor shower pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("stage", choices=list(STAGES) + ["all"], help="stage to run")
    parser.add_argument("--source", default=".", help="folder holding the frames")
    parser.add_argument("--output", default="output", help="output folder")
    parser.add_argument("--group", default=None,
                        help="process only this group id (default: all usable groups)")
    parser.add_argument("--mask-file", default=None,
                        help="hand-painted foreground mask, white = foreground")
    parser.add_argument("--force", action="store_true",
                        help="ignore cached results and recompute")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="do not prompt before slow stages")
    parser.add_argument("--jpeg-only", action="store_true",
                        help="stack from JPEGs even when RAF files are present")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    cfg = Config(paths=Paths(source=Path(args.source), output=Path(args.output),
                             cache=Path(args.output) / ".cache"))
    if args.jpeg_only:
        cfg.stack.prefer_raw = False
    cfg.paths.ensure()
    cfg.save(cfg.paths.output / "config_used.json")

    cache = Cache(cfg.paths.cache, force=args.force)

    stages = ALL_ORDER if args.stage == "all" else [args.stage]
    for stage in stages:
        log.info("--- stage: %s ---", stage)
        try:
            STAGES[stage](cfg, cache, args)
        except SystemExit:
            raise
        except Exception:
            log.exception("stage '%s' failed", stage)
            return 1

    if args.stage in ("all", "composite", "sheet"):
        try:
            print_summary(cfg, cache)
        except Exception:
            log.exception("could not build the summary")
    return 0
