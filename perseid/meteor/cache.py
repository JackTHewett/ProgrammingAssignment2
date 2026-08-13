"""Disk cache so every stage is resumable.

Each stage writes its products under ``output/.cache/<stage>/`` and records a
small JSON manifest. Rerunning a stage with the same inputs is a no-op unless
``force`` is set, so recompositing never costs a re-registration.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    """Write a .npy through a temp file so an interrupted run leaves no
    half-written cache entry.

    The temp name has to end in .npy: numpy.save silently appends the extension
    to any path that lacks it, and the rename would then miss the real file.
    """
    tmp = path.with_name(path.name + ".tmp.npy")
    with open(tmp, "wb") as fh:
        np.save(fh, value)
    tmp.replace(path)


def fingerprint(*parts: Any) -> str:
    """Stable short hash of whatever identifies a stage's inputs."""
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


class Cache:
    def __init__(self, root: Path, force: bool = False):
        self.root = Path(root)
        self.force = force
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, stage: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- arrays -----------------------------------------------------------
    def array(self, stage: str, key: str, build: Callable[[], np.ndarray]) -> np.ndarray:
        path = self._dir(stage) / f"{key}.npy"
        if path.exists() and not self.force:
            log.debug("cache hit  %s/%s", stage, key)
            return np.load(path, allow_pickle=False)
        log.debug("cache miss %s/%s", stage, key)
        value = build()
        _atomic_save_npy(path, value)
        return value

    def has_array(self, stage: str, key: str) -> bool:
        return (self._dir(stage) / f"{key}.npy").exists() and not self.force

    def load_array(self, stage: str, key: str) -> np.ndarray:
        return np.load(self._dir(stage) / f"{key}.npy", allow_pickle=False)

    def save_array(self, stage: str, key: str, value: np.ndarray) -> None:
        _atomic_save_npy(self._dir(stage) / f"{key}.npy", value)

    # --- json -------------------------------------------------------------
    def json(self, stage: str, key: str, build: Callable[[], Any]) -> Any:
        path = self._dir(stage) / f"{key}.json"
        if path.exists() and not self.force:
            log.debug("cache hit  %s/%s", stage, key)
            return json.loads(path.read_text())
        log.debug("cache miss %s/%s", stage, key)
        value = build()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, indent=2, default=str))
        tmp.replace(path)
        return value

    def load_json(self, stage: str, key: str) -> Any:
        path = self._dir(stage) / f"{key}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"missing cached {stage}/{key}.json - run the earlier stage first"
            )
        return json.loads(path.read_text())

    def save_json(self, stage: str, key: str, value: Any) -> None:
        path = self._dir(stage) / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, indent=2, default=str))
        tmp.replace(path)

    def exists(self, stage: str, key: str, suffix: str = ".json") -> bool:
        """Whether a cached product is present *and* usable.

        This has to honour ``force`` the same way ``has_array`` does. When it did
        not, ``--force`` silently failed to invalidate any stage that gated on
        it, and reruns kept returning stale results.
        """
        if self.force:
            return False
        return (self._dir(stage) / f"{key}{suffix}").exists()
