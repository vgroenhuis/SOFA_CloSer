"""Runs an original SOFA scene script in a background thread.

All SOFA calls happen on that one thread; the web app talks to it through a
command queue and receives setup messages / frames through callbacks.

Frames are binary WebSocket messages:
    uint32 little-endian header length | header JSON (UTF-8) | padding to a
    multiple of 4 | float32 vertex positions of the models listed in the
    header, in header order.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import json
import logging
import math
import os
import queue
import shutil
import struct
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation

from .capture import Capture, ambient_color, light_list
from .config import Params, SceneConfig, editor_fields

log = logging.getLogger(__name__)

WORK_ROOT = Path("/tmp/sofaweb")
CSV_MAX_ROWS = 20000
HISTORY_MAX_ROWS = 1500  # chart history sent to a client that connects mid-run
RESTART_PAUSE_SECONDS = 1.5
# Dedup of coincident vertices is dropped for a model once any duplicate
# drifts from its representative by more than this (relative to the
# model's size) -- coincident-at-rest vertices that later separate.
DEDUP_TOLERANCE = 1e-6


class ConsoleBuffer(io.TextIOBase):
    """Collects what the scene script prints (per-step diagnostics etc.) so
    the page can show it like the SOFA GUI's log panel -- and so it doesn't
    flood the container log."""

    def __init__(self, maxlen: int = 400) -> None:
        self.lines: deque[tuple[int, float, str]] = deque(maxlen=maxlen)
        self._seq = itertools.count(1)
        self._partial = ""
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        with self._lock:
            self._partial += text
            *complete, self._partial = self._partial.split("\n")
            for line in complete:
                self.lines.append((next(self._seq), time.time(), line.rstrip("\r")))
        return len(text)

    def since(self, after: int) -> list[dict]:
        with self._lock:
            return [{"seq": s, "ts": ts, "text": t} for s, ts, t in self.lines if s > after]

    def note(self, text: str) -> None:
        self.write(f"[sofaweb] {text}\n")


@dataclass
class Model:
    id: int
    obj: Any
    name: str
    kind: str  # "mesh" | "lines"
    color: list[float]
    line_width: float
    index: np.ndarray  # triangle (n*3) or edge (n*2) indices into the full vertex array
    vertex_count: int
    flat: bool
    rep: Optional[np.ndarray]  # indices of representative vertices when deduplicated
    inverse: Optional[np.ndarray]  # full vertex -> representative position
    scale: float
    bound: float  # |coordinate| beyond which the simulation is considered to have blown up
    last_sent: Optional[np.ndarray] = None  # last full position array sent

    def setup(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "color": self.color,
            "lineWidth": self.line_width,
            "index": self.index.astype(int).tolist(),
            "vertexCount": self.vertex_count,
            "flat": self.flat,
            "inverse": self.inverse.astype(int).tolist() if self.inverse is not None else None,
        }

    def packed(self, positions: np.ndarray) -> np.ndarray:
        return positions[self.rep] if self.rep is not None else positions


def _to_triangles(obj) -> np.ndarray:
    tris = np.asarray(obj.triangles.value, dtype=np.int64).reshape(-1, 3)
    quads = np.asarray(obj.quads.value, dtype=np.int64).reshape(-1, 4)
    if len(quads):
        tris = np.concatenate([tris, quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    return tris


def _build_model(model_id: int, record) -> Optional[Model]:
    obj = record.obj
    positions = np.asarray(obj.position.value, dtype=np.float64).reshape(-1, 3)
    if not len(positions):
        return None
    keep_lines = bool(obj.keepLines.value) if hasattr(obj, "keepLines") else False
    tris = _to_triangles(obj)
    if keep_lines and len(obj.edges.value) and not len(tris):
        kind, index = "lines", np.asarray(obj.edges.value, dtype=np.int64).reshape(-1, 2)
    elif len(tris):
        kind, index = "mesh", tris
    else:
        return None
    flat = kind == "mesh" and np.bincount(index.ravel(), minlength=len(positions)).max() <= 1

    rep = inverse = None
    uniq, first, inv = np.unique(positions, axis=0, return_index=True, return_inverse=True)
    if len(uniq) < 0.8 * len(positions):
        rep, inverse = first, inv.reshape(-1)
    extent = positions.max(axis=0) - positions.min(axis=0)
    return Model(
        id=model_id,
        obj=obj,
        name=obj.getPathName(),
        kind=kind,
        color=record.color,
        line_width=record.line_width,
        index=index.reshape(-1),
        vertex_count=len(positions),
        flat=bool(flat),
        rep=rep,
        inverse=inverse,
        scale=float(max(extent.max(), 1e-9)),
        bound=1000.0 * float(np.abs(positions).max() + max(extent.max(), 1e-9)),
    )


def pack_frame(header: dict, arrays: list[np.ndarray]) -> bytes:
    head = json.dumps(header, separators=(",", ":")).encode()
    pad = (-(4 + len(head))) % 4
    parts = [struct.pack("<I", len(head)), head, b" " * pad]
    parts.extend(np.ascontiguousarray(a, dtype=np.float32).tobytes() for a in arrays)
    return b"".join(parts)


class Runner:
    def __init__(
        self,
        config: SceneConfig,
        base_dir: Path,
        on_message: Callable[[Any], None],
    ) -> None:
        self.cfg = config
        self.base_dir = base_dir
        self.script_path = base_dir / config.script
        self._on_message = on_message  # str (JSON text) or bytes (frame)
        self.console = ConsoleBuffer()

        self.defaults = self._load_defaults()
        self.params: Params = dict(self.defaults)
        self.spec = self._build_spec()
        self.auto_restart = config.auto_restart

        self._commands: "queue.Queue[tuple[str, dict, Future]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sofa", daemon=True)
        self._lock = threading.Lock()  # guards the snapshot / csv state below

        self.generation = 0
        self.status = "building"
        self.error: Optional[str] = None
        self.paused = False
        self.finished = False
        self.end_reason: Optional[str] = None
        self.rtf = 0.0
        self._setup_json: Optional[str] = None
        self._full_positions: dict[int, np.ndarray] = {}
        self._last_scalars: dict[str, float] = {}
        self._last_lights: Optional[list] = None
        self._prev_lights_sent: Optional[list] = None
        self._csv_rows: deque[dict] = deque(maxlen=CSV_MAX_ROWS)

        self._root = None
        self._module = None
        self._capture: Optional[Capture] = None
        self._models: list[Model] = []
        self._sim_time = 0.0
        self._stop_at: Optional[float] = None
        self._last_good_params: Optional[Params] = None
        self._diverged = False

    # -- parameters ---------------------------------------------------------

    def _load_defaults(self) -> Params:
        params: Params = {}
        if self.cfg.params_file:
            path = self.base_dir / self.cfg.params_file
            if path.is_file():
                params = json.loads(path.read_text(encoding="utf-8"))
        return params

    def _build_spec(self) -> list[dict]:
        editor = self.base_dir / self.cfg.editor_file if self.cfg.editor_file else None
        fields = [f for f in editor_fields(editor) if f["key"] in self.defaults]
        known = {f["key"] for f in fields}
        # params.json keys the editor doesn't list still get a plain field.
        for key, value in self.defaults.items():
            if key in known:
                continue
            if isinstance(value, bool):
                fields.append({"key": key, "label": key, "type": "bool"})
            elif isinstance(value, (int, float)):
                fields.append({"key": key, "label": key, "type": "number", "scale": 1, "decimals": 3, "unit": ""})
            elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                fields.append({"key": key, "label": key, "type": "vector", "length": len(value), "decimals": 3, "unit": ""})
        visible = [
            f for f in fields
            if f["key"] not in self.cfg.hidden_params and f["key"] not in self.cfg.forced_params
        ]
        for f in visible:
            f["label"] = self.cfg.labels.get(f["key"], f["label"])
            f["live"] = f["key"] in self.cfg.live_params
            f["integer"] = isinstance(self.defaults[f["key"]], int) and not isinstance(self.defaults[f["key"]], bool)
            if f["key"] in self.cfg.param_limits:
                f["min"], f["max"] = self.cfg.param_limits[f["key"]]
        order = {f["key"]: f for f in visible}
        if self.cfg.sections:
            grouped, used = [], set()
            for title, keys in self.cfg.sections:
                items = [order[k] for k in keys if k in order]
                used.update(k for k in keys if k in order)
                if items:
                    grouped.append({"title": title, "fields": items})
            rest = [f for f in visible if f["key"] not in used]
            if rest:
                grouped.append({"title": "Other", "fields": rest})
            return grouped
        return [{"title": "Parameters", "fields": visible}]

    def visible_keys(self) -> set[str]:
        return {f["key"] for section in self.spec for f in section["fields"]}

    def coerce(self, changes: Params) -> Params:
        """Validates user-supplied values against the defaults' types and
        the configured limits; unknown / hidden keys are dropped."""
        visible = self.visible_keys()
        clean: Params = {}
        for key, value in changes.items():
            if key not in visible:
                continue
            default = self.defaults[key]
            try:
                if isinstance(default, bool):
                    value = bool(value)
                elif isinstance(default, int):
                    value = int(round(float(value)))
                elif isinstance(default, float):
                    value = float(value)
                elif isinstance(default, list):
                    value = [float(v) for v in value][: len(default)]
                    if len(value) != len(default):
                        continue
                else:
                    continue
            except (TypeError, ValueError):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            if key in self.cfg.param_limits and not isinstance(value, (bool, list)):
                lo, hi = self.cfg.param_limits[key]
                value = type(value)(min(hi, max(lo, value)))
            clean[key] = value
        return clean

    def effective_params(self, params: Params) -> Params:
        return {**params, **self.cfg.forced_params}

    # -- public, thread-safe API ------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def request(self, command: str, **kwargs) -> Future:
        future: Future = Future()
        self._commands.put((command, kwargs, future))
        return future

    def snapshot(self) -> list:
        """Messages that bring a newly connected client up to date: setup,
        status, chart history so far, and a full frame."""
        with self._lock:
            messages: list = []
            if self._setup_json:
                messages.append(self._setup_json)
            messages.append(self._status_json())
            rows = list(self._csv_rows)
            if rows:
                step = max(1, len(rows) // HISTORY_MAX_ROWS)
                history = [[r["t"], {k: v for k, v in r.items() if k != "t"}] for r in rows[::step]]
                messages.append(json.dumps(
                    {"type": "history", "generation": self.generation, "rows": history},
                    separators=(",", ":"),
                ))
            if self._setup_json and self._full_positions:
                ids = sorted(self._full_positions)
                header = self._frame_header(ids, full=True)
                messages.append(pack_frame(header, [self._full_positions[i] for i in ids]))
            return messages

    def csv(self) -> str:
        with self._lock:
            rows = list(self._csv_rows)
        if not rows:
            return "t\n"
        keys = list(rows[0].keys())
        lines = [",".join(keys)]
        lines.extend(",".join(f"{row.get(k, float('nan')):.6g}" for k in keys) for row in rows)
        return "\n".join(lines) + "\n"

    def info(self) -> dict:
        root = self._root
        return {
            "title": self.cfg.title,
            "dt": float(root.dt.value) if root is not None else None,
            "simTime": self._sim_time,
            "rtf": self.rtf,
            "stopAt": self._stop_at,
            "status": self.status,
            "generation": self.generation,
            "models": [{"name": m.name, "vertices": m.vertex_count, "kind": m.kind} for m in self._models],
        }

    # -- sim thread -----------------------------------------------------------

    def _emit(self, message: Any) -> None:
        try:
            self._on_message(message)
        except Exception:
            log.exception("Delivering a message failed")

    def _status_json(self) -> str:
        return json.dumps({
            "type": "status",
            "status": self.status,
            "error": self.error,
            "paused": self.paused,
            "finished": self.finished,
            "endReason": self.end_reason,
            "autoRestart": self.auto_restart,
            "stopAt": self._stop_at,
            "generation": self.generation,
        })

    def _emit_status(self) -> None:
        self._emit(self._status_json())

    def _unload(self) -> None:
        if self._root is not None:
            with contextlib.redirect_stdout(self.console):
                try:
                    Sofa.Simulation.unload(self._root)
                except Exception:
                    log.exception("Unloading the previous scene failed")
        self._root = None
        if self._module is not None:
            sys.modules.pop(self._module.__name__, None)
        self._module = None
        self._models = []

    def _build(self, params: Params) -> None:
        self.status = "building"
        self.error = None
        self.finished = False
        self.end_reason = None
        self._emit_status()
        self._unload()

        self.generation += 1
        effective = self.effective_params(params)
        workdir = WORK_ROOT / f"build-{self.generation}"
        if workdir.exists():
            shutil.rmtree(workdir)
        shutil.copytree(self.script_path.parent, workdir)
        (workdir / "params.json").write_text(json.dumps(effective, indent=2), encoding="utf-8")
        old = WORK_ROOT / f"build-{self.generation - 2}"
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        os.environ.update(self.cfg.env)

        self.console.note(f"building scene (build {self.generation})")
        with contextlib.redirect_stdout(self.console):
            spec = importlib.util.spec_from_file_location(f"sofaweb_scene_{self.generation}", workdir / self.script_path.name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
                root = Sofa.Core.Node("root")
                with Capture() as capture:
                    module.createScene(root)
                Sofa.Simulation.init(root)
            except BaseException:
                sys.modules.pop(spec.name, None)
                raise

        self._root, self._module, self._capture = root, module, capture
        self._models = [m for m in (_build_model(i, r) for i, r in enumerate(capture.visuals)) if m is not None]
        self._sim_time = 0.0
        self._diverged = False
        self._stop_at = self.cfg.stop_at(effective) if self.cfg.stop_at else None
        self._last_lights = None
        self._prev_lights_sent = None

        setup = {
            "type": "setup",
            "generation": self.generation,
            "background": self._background(root),
            "camera": self._camera(root),
            "ambient": ambient_color(capture),
            "lights": light_list(capture),
            "view": self.cfg.view,
            "up": list(self.cfg.up),
            "models": [m.setup() for m in self._models],
        }
        full = {m.id: m.packed(np.asarray(m.obj.position.value, dtype=np.float64)) for m in self._models}
        for m in self._models:
            m.last_sent = np.asarray(m.obj.position.value, dtype=np.float64).copy()
        with self._lock:
            self._setup_json = json.dumps(setup, separators=(",", ":"))
            self._full_positions = full
            self._csv_rows.clear()
        self.status = "running"
        self._last_good_params = dict(params)
        self.console.note(
            f"scene ready: {len(self._models)} visual models, "
            f"{sum(m.vertex_count for m in self._models)} vertices, dt={float(root.dt.value):g}s"
        )
        self._emit(self._setup_json)
        self._emit_status()
        self._broadcast_frame(full=True)

    @staticmethod
    def _background(root) -> list[float]:
        for obj in root.objects:
            if obj.getClassName() == "BackgroundSetting":
                return [float(c) for c in obj.color.value]
        return [0.0, 0.0, 0.0, 1.0]

    @staticmethod
    def _camera(root) -> Optional[dict]:
        for obj in root.objects:
            if obj.getClassName() in ("InteractiveCamera", "Camera", "RecordedCamera"):
                return {
                    "position": [float(v) for v in obj.position.value],
                    "orientation": [float(v) for v in obj.orientation.value],
                    "fov": float(obj.fieldOfView.value),
                    "distance": float(obj.distance.value),
                    "projection": str(obj.projectionType.value),
                }
        return None

    def _safe_build(self, params: Params) -> None:
        try:
            self._build(params)
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.console.write(traceback.format_exc())
            self.console.note(f"build failed: {detail}")
            fallback = self._last_good_params
            if fallback is not None and fallback != params:
                self.params = dict(fallback)
                self.console.note("restoring the last working parameters")
                self.error = f"These parameters failed to build ({detail}); restored the previous ones."
                message = self.error
                try:
                    self._build(fallback)
                    self.error = message
                    self._emit_status()
                    return
                except Exception:
                    self.console.write(traceback.format_exc())
            self._unload()
            self.status = "error"
            self.error = f"The scene failed to build: {detail}"
            self._emit_status()

    def _probe(self) -> dict[str, float]:
        scalars: dict[str, float] = {}
        if self.cfg.probes is not None and self._root is not None:
            try:
                scalars = {k: float(v) for k, v in self.cfg.probes(self._root, self._module).items()}
            except Exception as exc:
                scalars = {}
                self.console.note(f"probe failed: {exc}")
        return scalars

    def _frame_header(self, ids: list[int], full: bool) -> dict:
        return {
            "type": "frame",
            "generation": self.generation,
            "t": self._sim_time,
            "rtf": self.rtf,
            "paused": self.paused,
            "finished": self.finished,
            "full": full,
            "scalars": self._last_scalars,
            "lights": self._last_lights,
            "models": [[i, len(self._full_positions[i])] for i in ids],
        }

    def _broadcast_frame(self, full: bool = False) -> None:
        if self._root is None:
            return
        changed: list[int] = []
        resend_setup = False
        for m in self._models:
            positions = np.asarray(m.obj.position.value, dtype=np.float64).reshape(-1, 3)
            if len(positions) != m.vertex_count:
                continue  # topology changed under us; the next rebuild will catch up
            if not np.isfinite(positions).all() or np.abs(positions).max() > m.bound:
                self._diverged = True
                continue  # never send NaNs / runaway values to the browsers
            if m.rep is not None:
                drift = np.abs(positions - positions[m.rep][m.inverse]).max()
                if drift > DEDUP_TOLERANCE * m.scale:
                    m.rep = m.inverse = None
                    resend_setup = True
            if full or resend_setup or m.last_sent is None or not np.array_equal(positions, m.last_sent):
                m.last_sent = positions.copy()
                with self._lock:
                    self._full_positions[m.id] = m.packed(positions)
                changed.append(m.id)
        if resend_setup:
            with self._lock:
                setup = json.loads(self._setup_json)
                setup["models"] = [m.setup() for m in self._models]
                self._setup_json = json.dumps(setup, separators=(",", ":"))
                self._full_positions = {m.id: m.packed(m.last_sent) for m in self._models}
            self._emit(self._setup_json)
            changed = [m.id for m in self._models]
            full = True

        lights = light_list(self._capture) if self._capture else []
        scalars = self._probe()
        with self._lock:
            self._last_scalars = scalars
            self._last_lights = lights
            header = self._frame_header(changed, full)
            if not full and lights == self._prev_lights_sent:
                header["lights"] = None  # unchanged; only a light that moves is re-sent
            self._prev_lights_sent = lights
            arrays = [self._full_positions[i] for i in changed]
            if not full:
                self._csv_rows.append({"t": self._sim_time, **scalars})
        self._emit(pack_frame(header, arrays))

    def _end_run(self, reason: str) -> None:
        self.finished = True
        self.end_reason = reason
        self.console.note(f"run ended at t={self._sim_time:.3f}s: {reason}")
        self._broadcast_frame()
        if self.auto_restart:
            self._emit_status()
            deadline = time.monotonic() + RESTART_PAUSE_SECONDS
            while time.monotonic() < deadline and self._commands.empty() and not self._stop.is_set():
                time.sleep(0.05)
            if self._commands.empty():
                self._safe_build(self.params)
        else:
            self._emit_status()

    def _extend_run(self) -> None:
        """Called on resume-after-finish and on live parameter changes: let
        the (stopped) simulation run on long enough to show the effect."""
        if self.finished or self._stop_at is not None:
            extra = self.cfg.extend_by(self.effective_params(self.params)) if self.cfg.extend_by else None
            if extra is not None:
                self._stop_at = self._sim_time + extra
            elif self.finished:
                self._stop_at = None
        self.finished = False
        self.end_reason = None

    def _handle(self, command: str, kwargs: dict, future: Future) -> None:
        try:
            if command == "reset":
                # A paused simulation stays paused (at its new start) until
                # the owner resumes it.
                self._safe_build(self.params)
                future.set_result(True)
            elif command == "pause":
                self.paused = True
                self._emit_status()
                future.set_result(True)
            elif command == "resume":
                if self._diverged:
                    self.console.note("the run diverged; rebuilding instead of resuming")
                    self.paused = False
                    self._safe_build(self.params)
                    future.set_result(True)
                    return
                if self.finished:
                    self._extend_run()
                self.paused = False
                self._emit_status()
                future.set_result(True)
            elif command == "auto_restart":
                self.auto_restart = bool(kwargs["enabled"])
                self._emit_status()
                future.set_result(True)
            elif command == "params":
                changes: Params = kwargs["changes"]
                rebuild = {k: v for k, v in changes.items() if k not in self.cfg.live_params and v != self.params.get(k)}
                live = {k: v for k, v in changes.items() if k in self.cfg.live_params}
                self.params.update(changes)
                if rebuild or kwargs.get("rebuild") or (live and self._diverged):
                    self._safe_build(self.params)
                elif live and self._root is not None:
                    for key, value in live.items():
                        self.cfg.live_params[key](self._root, value)
                        self.console.note(f"live change: {key} = {value}")
                    self._extend_run()
                    self._emit_status()
                future.set_result(dict(self.params))
            else:
                future.set_exception(ValueError(f"unknown command {command}"))
        except Exception as exc:
            future.set_exception(exc)

    def _run(self) -> None:
        self._safe_build(self.params)
        frame_interval = 1.0 / self.cfg.frames_per_second
        next_frame = time.monotonic()
        wall_anchor = time.monotonic()
        sim_anchor = 0.0
        rtf_window: deque[tuple[float, float]] = deque(maxlen=50)
        was_running = False

        while not self._stop.is_set():
            try:
                command, kwargs, future = self._commands.get_nowait()
                self._handle(command, kwargs, future)
                was_running = False
                continue
            except queue.Empty:
                pass

            runnable = self._root is not None and self.status == "running" and not self.paused and not self.finished
            if not runnable:
                was_running = False
                self.rtf = 0.0
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if not was_running:
                wall_anchor, sim_anchor = now, self._sim_time
                rtf_window.clear()
                was_running = True

            # Real-time cap: never run ahead of the wall clock. When the
            # scene is slower than real time it simply runs flat out
            # (re-anchoring so it doesn't try to catch up in a burst).
            target = sim_anchor + (now - wall_anchor)
            if self._sim_time >= target:
                time.sleep(min(max(target - self._sim_time + 0.001, 0.001), 0.02))
            else:
                dt = float(self._root.dt.value)
                with contextlib.redirect_stdout(self.console):
                    try:
                        Sofa.Simulation.animate(self._root, dt)
                    except Exception as exc:
                        self.console.write(traceback.format_exc())
                        self.status = "error"
                        self.error = f"The simulation step failed: {exc}"
                        self._emit_status()
                        continue
                self._sim_time += dt
                if target - self._sim_time > 0.5:
                    wall_anchor, sim_anchor = time.monotonic(), self._sim_time

            now = time.monotonic()
            if now >= next_frame:
                rtf_window.append((now, self._sim_time))
                if len(rtf_window) >= 2 and rtf_window[-1][0] > rtf_window[0][0]:
                    self.rtf = (rtf_window[-1][1] - rtf_window[0][1]) / (rtf_window[-1][0] - rtf_window[0][0])
                self._broadcast_frame()
                next_frame = now + frame_interval

                if self._diverged:
                    self._end_run(
                        "the simulation became numerically unstable -- try a gentler load, "
                        "a smaller timestep or stiffer material, then Restart"
                    )
                    was_running = False
                elif self._stop_at is not None and self._sim_time >= self._stop_at - 1e-9:
                    self._end_run(f"reached t = {self._stop_at:g} s")
                    was_running = False
                elif self.cfg.end_condition is not None:
                    reason = self.cfg.end_condition(self._root, self._module)
                    if reason:
                        self._end_run(reason)
                        was_running = False

        self._unload()
