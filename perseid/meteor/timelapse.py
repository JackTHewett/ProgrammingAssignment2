"""Stage 7 - timelapse.

Every frame in shot order, unaligned, so the roofline stays put and the sky
turns behind it. Built from the JPEGs, since speed matters more than shadow
detail here.

Frames are streamed into the encoder one at a time rather than assembled into
an array first.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .config import Config
from .imageio_utils import load_jpeg_proxy

log = logging.getLogger(__name__)


def _resized_frames(paths: list[Path], width: int):
    for path in paths:
        arr = load_jpeg_proxy(path, width, as_gray=False)
        # Even dimensions are required by yuv420p.
        h, w = arr.shape[:2]
        if h % 2 or w % 2:
            arr = arr[: h - (h % 2), : w - (w % 2)]
        yield path, (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _build_with_ffmpeg(paths: list[Path], out: Path, cfg: Config) -> Path:
    """Pipe raw RGB into ffmpeg. Avoids writing a few hundred temporary PNGs."""
    tcfg = cfg.timelapse
    stream = _resized_frames(paths, tcfg.width)
    first_path, first = next(stream)
    h, w = first.shape[:2]
    log.info("encoding %d frames at %dx%d, %d fps", len(paths), w, h, tcfg.fps)

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(tcfg.fps),
        "-i", "pipe:0",
        "-an", "-c:v", tcfg.codec, "-preset", "medium",
        "-crf", str(tcfg.crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    written = 0
    try:
        process.stdin.write(first.tobytes())
        written += 1
        for path, frame in stream:
            if frame.shape[:2] != (h, w):
                log.warning("%s is %s, expected %s - skipped from the timelapse",
                            path.name, frame.shape[:2], (h, w))
                continue
            process.stdin.write(frame.tobytes())
            written += 1
    finally:
        process.stdin.close()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with {code}")
    log.info("wrote %s (%d frames)", out, written)
    return out


def _build_with_imageio(paths: list[Path], out: Path, cfg: Config) -> Path:
    import imageio.v2 as imageio

    tcfg = cfg.timelapse
    log.info("ffmpeg not on PATH; using the imageio writer")
    writer = imageio.get_writer(str(out), fps=tcfg.fps, codec=tcfg.codec,
                                quality=None, ffmpeg_params=["-crf", str(tcfg.crf),
                                                             "-pix_fmt", "yuv420p"])
    shape = None
    written = 0
    try:
        for path, frame in _resized_frames(paths, tcfg.width):
            if shape is None:
                shape = frame.shape[:2]
            elif frame.shape[:2] != shape:
                log.warning("%s is the wrong size - skipped", path.name)
                continue
            writer.append_data(frame)
            written += 1
    finally:
        writer.close()
    log.info("wrote %s (%d frames)", out, written)
    return out


def build_timelapse(paths: list[Path], cfg: Config, name: str = "timelapse") -> Path:
    """Encode the timelapse, preferring a direct ffmpeg pipe."""
    cfg.paths.ensure()
    out = cfg.paths.output / f"{name}.mp4"
    if not paths:
        raise RuntimeError("no frames for the timelapse")
    if _ffmpeg_available():
        return _build_with_ffmpeg(paths, out, cfg)
    return _build_with_imageio(paths, out, cfg)
