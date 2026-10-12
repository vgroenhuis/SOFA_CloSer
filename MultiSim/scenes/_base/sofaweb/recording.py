"""Recording of a run, for playback in the browser.

The runner records a frame (the visual models' positions, the probes' values
and the lights) at most `fps` times per simulated second, whatever the
speed of the simulation, so playback at real time looks as smooth as the
live view. A frame only holds the models that moved since the previous one.

Wire format (GET api/recording; gzip-compressed):
    uint32 little-endian header length | header JSON (UTF-8) | padding to a
    multiple of 4 | the frames' position blocks, in header order, each
    padded to a multiple of 4 bytes.

Positions are quantised per model to a grid of `q` (the model's size /
2^18, far below what's visible) and stored as int32 grid coordinates
(enc 0), or as the difference from the model's previous frame, as int16
(enc 1) or int32 (enc 2). The differences compress well.

Runs of preset parameters are kept on disk (see RecordingCache): the next
simulation with the same parameters plays them back instead of simulating.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
GRID_BITS = 18
INT32_MAX = 2**31 - 1
# A session recording stops growing beyond this (raw float32 positions).
MAX_RECORDING_BYTES = 200 * 2**20


class Recording:
    def __init__(self, scales: dict[int, float], fps: float) -> None:
        self.q = {i: max(float(s), 1e-12) / 2**GRID_BITS for i, s in scales.items()}
        self.interval = 1.0 / fps
        self._next_t = 0.0  # when the next frame is due
        self.frames: list[dict] = []  # {"t", "k" (step), "s" (scalars), "l" (lights, when changed)}
        self.arrays: list[dict[int, np.ndarray]] = []
        self.nbytes = 0
        self.truncated = False
        # Recorded from the start with one parameter set: no live change
        # after the first step, never continued past its end.
        self.pure = True
        self.saved = False  # kept on disk, as the run of a preset
        self._dirty: set[int] = set()
        self._lights: Any = None
        self._lights_dirty = False
        self._encoded: Optional[tuple[int, bytes]] = None  # (frame count, gzip bytes from frame 0)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def end_time(self) -> float:
        return self.frames[-1]["t"] if self.frames else 0.0

    def offer(self, t: float, step: int, changed: list[int], positions: dict[int, np.ndarray],
              scalars: dict, lights: Any, force: bool = False) -> None:
        """Called for every frame sent to the browsers; records one when
        enough simulated time has passed (or `force`)."""
        self._dirty.update(changed)
        if lights is not None and lights != self._lights:
            self._lights, self._lights_dirty = lights, True
        if self.truncated:
            return
        if self.frames and (t <= self.frames[-1]["t"] + 1e-12 or (t < self._next_t - 1e-9 and not force)):
            return
        self._append(t, step, positions, scalars)

    def flush(self, t: float, step: int, positions: dict[int, np.ndarray], scalars: dict) -> None:
        """Records the current state if it isn't the last frame yet (when a
        run pauses or ends between recorded frames)."""
        if self.truncated or (self.frames and t <= self.frames[-1]["t"] + 1e-12):
            return
        if self._dirty or self._lights_dirty or not self.frames:
            self._append(t, step, positions, scalars)

    def _append(self, t: float, step: int, positions: dict[int, np.ndarray], scalars: dict) -> None:
        arrays = {i: np.array(positions[i], dtype=np.float32) for i in sorted(self._dirty) if i in positions}
        size = sum(a.nbytes for a in arrays.values())
        if self.nbytes + size > MAX_RECORDING_BYTES:
            self.truncated = True
            log.info("Recording stopped at t=%.3f s: it reached %d MB", t, MAX_RECORDING_BYTES >> 20)
            return
        frame = {"t": t, "k": step, "s": dict(scalars)}
        if self._lights_dirty or not self.frames:
            frame["l"] = self._lights
        self.frames.append(frame)
        self.arrays.append(arrays)
        # On a steady 1/fps grid; a late frame (a live view slower than
        # the simulation) shifts it, rather than crowding the next one.
        self._next_t = max(self._next_t + self.interval, t + 0.5 * self.interval)
        self.nbytes += size
        self._dirty.clear()
        self._lights_dirty = False

    def copy(self) -> "Recording":
        """A snapshot to encode outside the runner's lock (the frames
        themselves are never changed in place)."""
        other = Recording.__new__(Recording)
        other.__dict__.update(self.__dict__)
        other.frames = list(self.frames)
        other.arrays = [dict(a) for a in self.arrays]
        other._dirty = set()
        return other

    def adopt_encoding(self, snapshot: "Recording") -> None:
        if snapshot._encoded and (self._encoded is None or snapshot._encoded[0] > self._encoded[0]) \
                and snapshot._encoded[0] <= len(self.frames):
            self._encoded = snapshot._encoded

    def expand(self, model_id: int, inverse: np.ndarray) -> None:
        """A model's coincident vertices stopped being sent once (they
        drifted apart): turn its recorded positions into full ones too."""
        for arrays in self.arrays:
            if model_id in arrays:
                arrays[model_id] = arrays[model_id][inverse]
        self._encoded = None

    # -- encoding ---------------------------------------------------------

    def _grid(self, model_id: int, positions: np.ndarray) -> np.ndarray:
        grid = np.rint(positions.astype(np.float64) / self.q[model_id])
        return np.clip(grid, -INT32_MAX, INT32_MAX).astype(np.int64)

    def payload(self, start: int = 0, meta: Optional[dict] = None) -> bytes:
        """Frames from `start` on, uncompressed (see the module docstring)."""
        previous: dict[int, np.ndarray] = {}
        for arrays in self.arrays[:start]:
            for i, a in arrays.items():
                previous[i] = a
        previous = {i: self._grid(i, a) for i, a in previous.items()}
        headers, blocks = [], []
        for frame, arrays in zip(self.frames[start:], self.arrays[start:]):
            models = []
            for i, a in arrays.items():
                grid = self._grid(i, a)
                before = previous.get(i)
                if before is not None and before.shape == grid.shape:
                    delta = grid - before
                    peak = int(np.abs(delta).max()) if delta.size else 0
                    if peak < 2**15:
                        enc, data = 1, delta.astype("<i2").tobytes()
                    else:
                        enc, data = 2, delta.astype("<i4").tobytes()
                else:
                    enc, data = 0, grid.astype("<i4").tobytes()
                previous[i] = grid
                blocks.append(data + b"\0" * ((-len(data)) % 4))
                models.append([i, int(a.shape[0]), enc])
            headers.append({**frame, "m": models})
        header = {
            "v": FORMAT_VERSION,
            "q": {str(i): q for i, q in self.q.items()},
            "start": start,
            "frames": headers,
            **(meta or {}),
        }
        head = json.dumps(header, separators=(",", ":")).encode()
        pad = (-(4 + len(head))) % 4
        return b"".join([struct.pack("<I", len(head)), head, b" " * pad, *blocks])

    def encoded(self, start: int = 0) -> bytes:
        """gzip of payload(start); the full recording is cached until it grows."""
        if start == 0 and self._encoded and self._encoded[0] == len(self.frames):
            return self._encoded[1]
        data = gzip.compress(self.payload(start), compresslevel=6, mtime=0)
        if start == 0:
            self._encoded = (len(self.frames), data)
        return data


def _normalise(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.12g}")
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in sorted(value.items())}
    return value


def code_hash(paths: list[Path]) -> str:
    """Fingerprint of the scene's code: a recording is only reused while
    the scene script, its folder and its sofaweb configuration are unchanged."""
    digest = hashlib.sha256(f"format {FORMAT_VERSION}".encode())
    for path in paths:
        files = sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts) if path.is_dir() else [path]
        for f in files:
            if f.suffix in (".pyc",):
                continue
            digest.update(str(f.relative_to(path.parent)).encode())
            digest.update(f.read_bytes())
    return digest.hexdigest()


class RecordingCache:
    """Recordings of preset runs, one pair of files per parameter set in
    MSD_RECORDINGS_DIR (set by the orchestrator, per scene):
    <key>.rec.gz (the encoded recording) and <key>.json (what it is)."""

    def __init__(self, directory: Optional[Path], code: str) -> None:
        self.dir = directory
        self.code = code
        if self.dir is not None:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                log.warning("No recording cache: %s", exc)
                self.dir = None

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def key(self, effective_params: dict) -> str:
        text = json.dumps({"params": _normalise(effective_params), "code": self.code}, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    def load(self, key: str) -> Optional[tuple[dict, bytes]]:
        if self.dir is None:
            return None
        try:
            meta = json.loads((self.dir / f"{key}.json").read_text(encoding="utf-8"))
            data = (self.dir / f"{key}.rec.gz").read_bytes()
        except (OSError, ValueError):
            return None
        return meta, data

    def save(self, key: str, meta: dict, data: bytes) -> bool:
        if self.dir is None:
            return False
        try:
            for name, content in ((f"{key}.rec.gz", data), (f"{key}.json", json.dumps(meta, indent=1).encode())):
                tmp = self.dir / f".{name}.{os.getpid()}.tmp"
                tmp.write_bytes(content)
                os.replace(tmp, self.dir / name)
            return True
        except OSError as exc:
            log.warning("Saving the recording failed: %s", exc)
            return False

    def discard(self, key: str) -> None:
        if self.dir is None:
            return
        for name in (f"{key}.rec.gz", f"{key}.json"):
            (self.dir / name).unlink(missing_ok=True)

    def prune(self, keep: set[str]) -> None:
        """Removes recordings of parameter sets that are no longer a preset
        (or of older scene code)."""
        if self.dir is None:
            return
        for path in self.dir.glob("*.json"):
            key = path.name[: -len(".json")]
            if key not in keep:
                self.discard(key)
        cutoff = time.time() - 3600
        for path in self.dir.glob(".*.tmp"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass
