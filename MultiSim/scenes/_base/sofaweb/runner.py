"""Runs an original scene script (SOFA, or another engine: see engines.py) in a
background thread.

All simulation calls happen on that one thread; the web app talks to it through a
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

from .config import Params, SceneConfig, editor_fields
from .engines import make_engine
from .recording import Recording, RecordingCache, code_hash

log = logging.getLogger(__name__)

WORK_ROOT = Path("/tmp/sofaweb")
CSV_MAX_ROWS = 20000
HISTORY_MAX_ROWS = 1500  # chart history sent to a client that connects mid-run
RESTART_PAUSE_SECONDS = 1.5
# Dedup of coincident vertices is dropped for a model once any duplicate
# drifts from its representative by more than this (relative to the
# model's size) -- coincident-at-rest vertices that later separate.
DEDUP_TOLERANCE = 1e-6
# A held "hold" control lets go unless the owner's browser confirms it
# within this time (it does every second while the button is down).
HOLD_TIMEOUT_SECONDS = 2.5


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
    kind: str  # "mesh" | "lines" | "tetra" (an FEM mesh: index is 4 node indices per element)
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
    # FEM meshes only: rest positions, so the viewer can colour elements by
    # strain even when it connects mid-run.
    rest: Optional[np.ndarray] = None

    def setup(self) -> dict:
        info = {
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
        if self.rest is not None:
            info["rest"] = [round(float(v), 9) for v in self.rest.reshape(-1)]
        return info

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


def _fem_models(root, first_id: int) -> list[Model]:
    """The scene's tetrahedral FEM meshes, for the viewer's "FEM elements"
    display (SOFA's showForceFields look): every node whose topology has
    tetrahedra, with that node's mechanical state as positions."""
    found: list[Model] = []
    seen: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        path = node.getPathName()
        if path in seen:  # a node with two parents (e.g. Finger 2 with channel's cube)
            continue
        seen.add(path)
        stack.extend(node.children)
        mstate = node.getMechanicalState()
        if mstate is None:
            continue
        for obj in node.objects:
            if not hasattr(obj, "tetrahedra"):
                continue
            tets = np.asarray(obj.tetrahedra.value, dtype=np.int64).reshape(-1, 4)
            positions = np.asarray(mstate.position.value, dtype=np.float64)
            # Only a Vec3 state holds the mesh's nodes (e.g. not the Vec1
            # state of a VolumeMapping that reads a tetrahedral topology).
            if positions.ndim != 2 or positions.shape[1] != 3:
                continue
            if not len(tets) or len(positions) <= tets.max():
                continue
            extent = positions.max(axis=0) - positions.min(axis=0)
            found.append(Model(
                id=first_id + len(found),
                obj=mstate,
                name=f"FEM:{obj.getPathName()}",
                kind="tetra",
                color=[0.0, 0.0, 1.0, 1.0],
                line_width=1.0,
                index=tets.reshape(-1),
                vertex_count=len(positions),
                flat=False,
                rep=None,
                inverse=None,
                scale=float(max(extent.max(), 1e-9)),
                bound=1000.0 * float(np.abs(positions).max() + max(extent.max(), 1e-9)),
                rest=np.asarray(mstate.rest_position.value, dtype=np.float64).reshape(-1, 3),
            ))
            break  # one tetrahedral topology per node
    return found


def _env_saved_defaults() -> dict:
    """MSD_SCENE_DEFAULTS, set by the orchestrator (see Runner.apply_saved_defaults)."""
    try:
        value = json.loads(os.environ.get("MSD_SCENE_DEFAULTS") or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def visible_fields(cfg: SceneConfig, base_dir: Path, defaults: Params) -> list[dict]:
    """The parameter panel's fields (label, type, scale, unit), in the
    editor's order: the original params_editor.py's fields, plus a plain one
    for each params.json key it doesn't list; hidden, forced and display
    keys left out."""
    editor = base_dir / cfg.editor_file if cfg.editor_file else None
    fields = [f for f in editor_fields(editor) if f["key"] in defaults]
    known = {f["key"] for f in fields}
    for key, value in defaults.items():
        if key in known:
            continue
        if isinstance(value, bool):
            fields.append({"key": key, "label": key, "type": "bool"})
        elif isinstance(value, (int, float)):
            fields.append({"key": key, "label": key, "type": "number", "scale": 1, "decimals": 3, "unit": ""})
        elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
            fields.append({"key": key, "label": key, "type": "vector", "length": len(value), "decimals": 3, "unit": ""})
    display_keys = {d["key"] for d in cfg.display}
    visible = [
        f for f in fields
        if f["key"] not in cfg.hidden_params and f["key"] not in cfg.forced_params and f["key"] not in display_keys
    ]
    for f in visible:
        f["label"] = cfg.labels.get(f["key"], f["label"])
    return visible


def preset_entries(cfg: SceneConfig, saved_defaults: dict, section: str, default_values: dict) -> list[dict]:
    """One panel's named presets: "Default" first, then the scene's built-in
    ones, then an admin's own. Saved presets are {"presets": {section:
    {name: values}}} in the saved defaults; a saved name replaces a built-in
    one of the same name, and None deletes it."""
    saved_all = saved_defaults.get("presets")
    saved_all = saved_all if isinstance(saved_all, dict) else {}
    entries = [{"name": "Default", "values": default_values, "builtin": True, "description":
                "The scene's default settings."}]
    saved = saved_all.get(section)
    saved = dict(saved) if isinstance(saved, dict) else {}
    for preset in cfg.presets.get(section, []):
        name = preset["name"]
        if name in saved:
            values = saved.pop(name)
            if not isinstance(values, dict):
                continue  # deleted
            entries.append({"name": name, "values": values, "builtin": True, "modified": True,
                            "description": preset.get("description", "")})
        else:
            entries.append({"name": name, "values": dict(preset.get("values", {})), "builtin": True,
                            "description": preset.get("description", "")})
    for name, values in saved.items():
        if isinstance(values, dict) and name.lower() != "default":
            entries.append({"name": name, "values": values, "builtin": False, "description": ""})
    return entries


def format_value(field: Optional[dict], value: Any) -> str:
    """A parameter value as the panel shows it (display units)."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, float)) and field is not None and field.get("type") == "number":
        shown = value / (field.get("scale") or 1)
        text = f"{shown:.6g}"
        return f"{text} {field['unit']}" if field.get("unit") else text
    if isinstance(value, float):
        return f"{value:.6g}"
    return json.dumps(value)


class ImportSource:
    """Another scene whose parameter presets this one can import: its
    sofaweb config (read, not run), defaults and presets."""

    def __init__(self, scene_id: str, scenes_dir: Path, saved_defaults: dict) -> None:
        path = scenes_dir / scene_id / "sofaweb_scene.py"
        if not path.is_file():
            raise FileNotFoundError(f"no scene {scene_id!r} here ({path})")
        spec = importlib.util.spec_from_file_location(f"sofaweb_import_{scene_id.replace('-', '_')}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.cfg: SceneConfig = module.CONFIG
        self.base_dir = path.parent
        self.title = self.cfg.title
        defaults: Params = {}
        if self.cfg.params_file and (self.base_dir / self.cfg.params_file).is_file():
            defaults = json.loads((self.base_dir / self.cfg.params_file).read_text(encoding="utf-8"))
        saved_params = saved_defaults.get("params")
        self.saved_defaults = saved_defaults
        self.fields = {f["key"]: f for f in visible_fields(self.cfg, self.base_dir, defaults)}
        self.defaults = {**defaults, **{k: v for k, v in (saved_params or {}).items() if k in self.fields}}

    def presets(self) -> list[dict]:
        visible = {k: v for k, v in self.defaults.items() if k in self.fields}
        return preset_entries(self.cfg, self.saved_defaults, "params", visible)

    def preset_values(self, name: str) -> Optional[Params]:
        for entry in self.presets():
            if entry["name"] == name:
                return {**self.defaults, **{k: v for k, v in entry["values"].items() if k in self.fields}}
        return None


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
        self.engine = make_engine(config.engine)

        self.original_defaults = self._load_defaults()  # params.json
        self.defaults = dict(self.original_defaults)
        self.spec = self._build_spec()
        self.saved_defaults: dict = {}
        # Other scenes' admin-saved defaults and presets, for importing their
        # presets: {"<scene id>": {...}} (MSD_SCENE_DEFAULTS' "imports").
        self.import_defaults: dict = {}
        self._import_cache: dict[str, ImportSource] = {}
        self.apply_saved_defaults(_env_saved_defaults())
        self.params: Params = dict(self.defaults)
        self.auto_restart = config.auto_restart

        self._commands: "queue.Queue[tuple[str, dict, Future]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sofa", daemon=True)
        self._lock = threading.Lock()  # guards the snapshot / csv state below

        self.generation = 0
        self.status = "building"
        self.error: Optional[str] = None
        # Start paused: a new simulation waits for the user to press play.
        self.paused = True
        self.finished = False
        self.end_reason: Optional[str] = None
        self.rtf = 0.0
        self.fps = 0.0  # frames sent per second (one per step when the scene is slow)
        self._setup_json: Optional[str] = None
        self._full_positions: dict[int, np.ndarray] = {}
        self._last_scalars: dict[str, float] = {}
        self._last_lights: Optional[list] = None
        self._prev_lights_sent: Optional[list] = None
        self._csv_rows: deque[dict] = deque(maxlen=CSV_MAX_ROWS)

        self._root = None
        self._module = None
        self._capture: Any = None  # what the browser draws (see engines)
        self._models: list[Model] = []
        self._sim_time = 0.0
        self._steps = 0  # animation steps since the scene was (re)built
        self._stop_at: Optional[float] = None
        self._last_good_params: Optional[Params] = None
        self._diverged = False
        # Interactive controls (cfg.controls): their current values, and when
        # each held "hold" control lets go unless confirmed again.
        self.control_values: dict[str, Any] = {}
        self._hold_deadlines: dict[str, float] = {}

        # Playback: the current run's recording, or a preset's run loaded
        # from the cache (then nothing is simulated until asked).
        self._recording: Optional[Recording] = None
        # After Stop (command "reset"): the recording is the run that was
        # stopped, kept for the timeline until the rebuilt scene is set going.
        self._previous_run = False
        self._cached: Optional[tuple[dict, bytes]] = None  # (meta, gzip bytes)
        self._rec_key: Optional[str] = None
        self._rec_serial = 0  # which recording: a browser refetches when it changes
        self._undeduped: set[int] = set()
        recordings_dir = os.environ.get("MSD_RECORDINGS_DIR")
        code_paths = [self.script_path.parent]
        if self.script_path.parent != base_dir:
            code_paths += [p for p in base_dir.glob("*.py") if p.is_file()]
        self._cache = RecordingCache(Path(recordings_dir) if recordings_dir else None,
                                     code_hash(code_paths) if recordings_dir else "")
        self._cache.prune(self._preset_keys())

    # -- parameters ---------------------------------------------------------

    def _load_defaults(self) -> Params:
        params: Params = {}
        if self.cfg.params_file:
            path = self.base_dir / self.cfg.params_file
            if path.is_file():
                params = json.loads(path.read_text(encoding="utf-8"))
        return params

    def apply_saved_defaults(self, saved: Any) -> None:
        """Defaults an admin set with the viewer's "Set as default" buttons
        (handed over by the orchestrator: MSD_SCENE_DEFAULTS at start, or
        POST /api/defaults later): {"params": {...}, "display": {...},
        "lighting": {...}}. Their params replace params.json's as what a
        new simulation starts with and the Defaults button returns to; the
        display/lighting ones are passed on to the browser (/api/scene)."""
        self.saved_defaults = saved if isinstance(saved, dict) else {}
        imports = self.saved_defaults.pop("imports", None)
        if isinstance(imports, dict):  # only at start; a later push concerns this scene alone
            self.import_defaults = imports
            self._import_cache = {}
        params = self.saved_defaults.get("params")
        self.defaults = {**self.original_defaults, **self.coerce(params if isinstance(params, dict) else {})}

    PRESET_SECTIONS = ("params", "display", "lighting")

    def presets(self) -> dict[str, list[dict]]:
        """Each panel's named presets: "Default" first (the saved defaults,
        or params.json's), then the scene's built-in ones, then an admin's
        own (see preset_entries)."""
        visible = self.visible_keys()
        result: dict[str, list[dict]] = {}
        for section in self.PRESET_SECTIONS:
            if section == "params":
                default_values = {k: v for k, v in self.defaults.items() if k in visible}
            else:
                default_values = self.saved_defaults.get(section) or {}
            entries = preset_entries(self.cfg, self.saved_defaults, section, default_values)
            if section == "params":
                for entry in entries:
                    entry["values"] = self.coerce(entry["values"])
            result[section] = entries
        return result

    # -- presets imported from other scenes -----------------------------------

    def _import_source(self, scene_id: str) -> ImportSource:
        source = self._import_cache.get(scene_id)
        if source is None:
            saved = self.import_defaults.get(scene_id)
            source = ImportSource(scene_id, self.base_dir.parent, saved if isinstance(saved, dict) else {})
            self._import_cache[scene_id] = source
        return source

    def import_sources(self) -> list[dict]:
        """For the viewer: the scenes to import parameter presets from, and
        their presets' names."""
        out = []
        for scene_id, spec in self.cfg.import_presets.items():
            try:
                source = self._import_source(scene_id)
            except Exception as exc:  # e.g. not in this container (docker builds one scene folder)
                log.warning("No presets to import from %s: %s", scene_id, exc)
                continue
            out.append({"scene": scene_id, "title": spec.get("title") or source.title,
                        "presets": [{"name": p["name"], "description": p.get("description", "")}
                                    for p in source.presets()]})
        return out

    def import_preset(self, scene_id: str, name: str) -> dict:
        """This scene's full parameter set for another scene's preset, and
        notes on what couldn't be carried over directly."""
        spec = self.cfg.import_presets.get(scene_id)
        if spec is None:
            raise KeyError(f"no presets to import from {scene_id!r}")
        source = self._import_source(scene_id)
        values = source.preset_values(name)
        if values is None:
            raise KeyError(f"{source.title} has no preset {name!r}")
        converted = spec["convert"](dict(values))
        fields = {f["key"]: f for section in self.spec for f in section["fields"]}
        params = {k: v for k, v in self.defaults.items() if k in fields}
        set_keys = set()
        for key, value in converted.get("params", {}).items():
            if key in fields:
                params[key] = value
                set_keys.add(key)
        handled = set(converted.get("handled", ())) | set_keys
        copied = set()
        for key, value in values.items():
            if key in fields and key in source.fields and key not in handled:
                params[key] = value
                copied.add(key)
        notes = []
        noted = set()
        for kind, key, text in converted.get("notes", ()):
            field = source.fields.get(key) if kind == "not imported" else fields.get(key)
            label = field["label"] if field else key
            notes.append({"kind": kind, "key": key, "label": label, "text": text})
            noted.add((kind, key))
        for key, field in source.fields.items():
            if key not in copied and key not in handled and ("not imported", key) not in noted:
                notes.append({"kind": "not imported", "key": key, "label": field["label"],
                              "text": f"{format_value(field, values.get(key))}: no counterpart here."})
        for key, field in fields.items():
            if key not in copied and key not in set_keys and not any(k == key for kind, k in noted if kind != "not imported"):
                notes.append({"kind": "assumed", "key": key, "label": field["label"],
                              "text": f"{format_value(field, params.get(key))}: this scene's default."})
        params = self.coerce(params)
        return {"source": source.title, "preset": name, "params": params, "notes": notes}

    def preset_params(self, name: str) -> Optional[Params]:
        """The full parameter set of a params preset: the defaults with the
        preset's values on top (a preset may list only what differs)."""
        for entry in self.presets()["params"]:
            if entry["name"] == name:
                visible = self.visible_keys()
                return {**{k: v for k, v in self.defaults.items() if k in visible}, **entry["values"]}
        return None

    def _preset_keys(self) -> set[str]:
        """Recording-cache keys of every parameters preset."""
        if not self._cache.enabled:
            return set()
        keys = set()
        for entry in self.presets()["params"]:
            keys.add(self._cache.key(self.effective_params({**self.defaults, **entry["values"]})))
        return keys

    def _build_spec(self) -> list[dict]:
        visible = visible_fields(self.cfg, self.base_dir, self.defaults)
        excludes = {}
        for a, b in self.cfg.exclusive:
            excludes.setdefault(a, []).append(b)
            excludes.setdefault(b, []).append(a)
        for f in visible:
            if f["key"] in excludes:
                f["excludes"] = excludes[f["key"]]  # switching it on switches these off
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
                    if not all(math.isfinite(v) for v in value):
                        continue
                    if len(value) != len(default) and not (key in self.cfg.short_vectors and value):
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
            "dt": self.engine.dt(root) if root is not None else None,
            "simTime": self._sim_time,
            "rtf": self.rtf,
            "fps": self.fps,
            "stopAt": self._stop_at,
            "status": self.status,
            "generation": self.generation,
            "models": [{"name": m.name, "vertices": m.vertex_count, "kind": m.kind} for m in self._models],
            # The FEM meshes: elements (tetrahedra) and nodes of each.
            "fem": [{"name": m.name, "elements": len(m.index) // 4, "nodes": m.vertex_count}
                    for m in self._models if m.kind == "tetra"],
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
            "recording": self._recording_status(),
            "controls": self.control_values,
        })

    def _emit_status(self) -> None:
        self._emit(self._status_json())

    def _unload(self) -> None:
        if self._root is not None:
            with contextlib.redirect_stdout(self.console):
                try:
                    self.engine.unload(self._root)
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
                root, capture = self.engine.build(module)
            except BaseException:
                sys.modules.pop(spec.name, None)
                raise

        self._root, self._module, self._capture = root, module, capture
        self._init_controls()
        self._models = [m for m in (_build_model(i, r) for i, r in enumerate(capture.visuals)) if m is not None]
        if self.engine.name == "sofa":
            self._models += _fem_models(root, len(capture.visuals))
        self._sim_time = 0.0
        self._steps = 0
        self._diverged = False
        self._stop_at = self.cfg.stop_at(effective) if self.cfg.stop_at else None
        self._last_lights = None
        self._prev_lights_sent = None
        self._undeduped = set()
        self._recording = None
        self._previous_run = False
        self._cached = self._load_cached(effective)
        if self._cached is not None:
            self._rec_serial += 1
            # The recording was made with these models' coincident vertices
            # sent separately: send them so from the start.
            for m in self._models:
                if m.id in self._cached[0].get("undeduped", []):
                    m.rep = m.inverse = None
                    self._undeduped.add(m.id)
        else:
            self._recording = self._new_recording()

        setup = {
            "type": "setup",
            "generation": self.generation,
            "background": self.engine.background(root),
            "camera": self.engine.camera(root),
            "ambient": self.engine.ambient(capture),
            "lights": self.engine.lights(capture),
            "view": self.cfg.view,
            "up": list(self.cfg.up),
            "models": [m.setup() for m in self._models],
            "gauge": self._gauge_layout(),
        }
        full = {m.id: m.packed(np.asarray(m.obj.position.value, dtype=np.float64)) for m in self._models}
        for m in self._models:
            m.last_sent = np.asarray(m.obj.position.value, dtype=np.float64).copy()
        with self._lock:
            self._setup_json = json.dumps(setup, separators=(",", ":"))
            self._full_positions = full
            self._csv_rows.clear()
            if self._cached is not None:
                self._csv_rows.extend(self._cached[0].get("rows", []))
        self.status = "running"
        self._last_good_params = dict(params)
        visual = [m for m in self._models if m.kind != "tetra"]
        fem = [m for m in self._models if m.kind == "tetra"]
        fem_text = "".join(f", FEM {len(m.index) // 4} tetrahedra / {m.vertex_count} nodes" for m in fem)
        self.console.note(
            f"scene ready: {len(visual)} visual models, "
            f"{sum(m.vertex_count for m in visual)} vertices{fem_text}, dt={self.engine.dt(root):g}s"
        )
        if self._cached is not None:
            # A preset's run, recorded before: it plays back in the browser;
            # simulating it again is the owner's choice (command "simulate").
            meta = self._cached[0]
            self.finished = True
            self.paused = True
            self.end_reason = meta.get("endReason") or "recorded run"
            self.console.note(f"playing back the recorded run of these parameters "
                              f"({meta.get('frames')} frames up to t={meta.get('end', 0):.2f}s)")
        self._emit(self._setup_json)
        self._emit_status()
        self._broadcast_frame(full=True)

    def _gauge_layout(self) -> Optional[dict]:
        if self.cfg.gauge is None:
            return None
        try:
            return self.cfg.gauge(self._root, self._module)
        except Exception as exc:  # a gauge that can't be laid out is left out, not a failed build
            self.console.note(f"pressure gauge left out: {exc}")
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

    # -- interactive controls --------------------------------------------------

    def _control(self, key: str) -> Optional[dict]:
        return next((c for c in self.cfg.controls if c["key"] == key), None)

    @staticmethod
    def _control_value(control: dict, value: Any) -> Any:
        kind = control["type"]
        if kind == "slider":
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("not a number")
            return min(max(value, float(control["min"])), float(control["max"]))
        if kind in ("toggle", "hold"):
            return bool(value)
        return None  # button

    def _init_controls(self) -> None:
        """Every build starts the controls at their defaults."""
        self.control_values = {}
        self._hold_deadlines = {}
        for control in self.cfg.controls:
            kind = control["type"]
            if kind == "button":
                continue
            value = self._control_value(control, control.get("default", control.get("min", 0.0) if kind == "slider" else False))
            with contextlib.redirect_stdout(self.console):
                control["apply"](self._root, value)
            self.control_values[control["key"]] = value

    def _apply_control(self, key: str, value: Any) -> dict:
        control = self._control(key)
        if control is None:
            raise ValueError(f"unknown control {key!r}")
        if self._root is None or self.status != "running":
            raise RuntimeError("the scene isn't running")
        value = self._control_value(control, value)
        kind = control["type"]
        if kind == "hold" and value:
            self._hold_deadlines[key] = time.monotonic() + HOLD_TIMEOUT_SECONDS
        else:
            self._hold_deadlines.pop(key, None)
        if kind != "button" and self.control_values.get(key) == value:
            return {"key": key, "value": value}  # a hold's keep-alive, or nothing new
        if self._cached is not None:
            self._start_live(paused=self.paused)
        self._leave_previous_run()
        with contextlib.redirect_stdout(self.console):
            if kind == "button":
                control["apply"](self._root)
            else:
                control["apply"](self._root, value)
        if kind != "button":
            self.control_values[key] = value
        if kind != "slider":
            self.console.note(f"control: {control['label']}" + ("" if kind == "button" else f" = {value}"))
        # The run is no longer what its parameters alone would give.
        if self._recording is not None:
            self._recording.pure = False
        if self.finished:
            self._extend_run()
        self._emit_status()
        return {"key": key, "value": value}

    def _expire_holds(self) -> None:
        now = time.monotonic()
        for key, deadline in list(self._hold_deadlines.items()):
            if now >= deadline:
                self.console.note(f"control: {key} released (no word from the browser)")
                try:
                    self._apply_control(key, False)
                except Exception:
                    self._hold_deadlines.pop(key, None)

    # -- recording / playback ------------------------------------------------

    def _new_recording(self) -> Recording:
        self._rec_serial += 1
        return Recording({m.id: m.scale for m in self._models}, self.cfg.frames_per_second)

    def _model_signature(self) -> list:
        return [[m.id, m.vertex_count, m.kind] for m in self._models]

    def _load_cached(self, effective: Params) -> Optional[tuple[dict, bytes]]:
        self._rec_key = self._cache.key(effective) if self._cache.enabled else None
        if self._rec_key is None:
            return None
        found = self._cache.load(self._rec_key)
        if found is None:
            return None
        if found[0].get("models") != self._model_signature():
            self.console.note("the recorded run of these parameters doesn't fit the scene; simulating")
            self._cache.discard(self._rec_key)
            return None
        return found

    def _start_live(self, paused: bool) -> None:
        """From a cached run's playback to simulating it (the scene is
        still at its start)."""
        self._cached = None
        self.finished = False
        self.end_reason = None
        self.paused = paused
        self._recording = self._new_recording()
        with self._lock:
            self._csv_rows.clear()
            self._recording.offer(self._sim_time, self._steps, [m.id for m in self._models],
                                  self._full_positions, self._last_scalars, self._last_lights, force=True)
        self.console.note("simulating live")

    def _leave_previous_run(self) -> None:
        """The scene Stop rebuilt is set going (or changed): from playing back
        the stopped run to recording this one."""
        if not self._previous_run:
            return
        self._previous_run = False
        self._recording = self._new_recording()
        with self._lock:
            self._recording.offer(self._sim_time, self._steps, [m.id for m in self._models],
                                  self._full_positions, self._last_scalars, self._last_lights, force=True)

    def _flush_recording(self) -> None:
        if self._recording is not None:
            with self._lock:
                self._recording.flush(self._sim_time, self._steps, self._full_positions, self._last_scalars)

    def _save_recording(self) -> None:
        """Keeps the run on disk if its parameters are a preset's and it
        was recorded from start to end with them."""
        rec = self._recording
        if (rec is None or rec.saved or not rec.pure or rec.truncated or self._diverged or not self.finished
                or self._rec_key is None or self._rec_key not in self._preset_keys()):
            return
        with self._lock:
            snapshot = rec.copy()
            rows = list(self._csv_rows)
        data = snapshot.encoded(level=6)
        rec.adopt_encoding(snapshot)
        meta = {
            "title": self.cfg.title,
            "key": self._rec_key,
            "params": self.effective_params(self.params),
            "created": time.time(),
            "frames": len(snapshot),
            "end": snapshot.end_time,
            "endReason": self.end_reason,
            "models": self._model_signature(),
            "undeduped": sorted(self._undeduped),
            "rows": rows,
            "bytes": len(data),
        }
        if self._cache.save(self._rec_key, meta, data):
            rec.saved = True
            self.console.note(f"saved this run for playback ({len(data) / 2**20:.1f} MB): "
                              "its parameters are a preset's")

    def recording_blob(self, start: int = 0) -> Optional[dict]:
        """The recording for GET api/recording: gzip bytes of frames from
        `start` on (a cached run always comes whole)."""
        with self._lock:
            generation = self.generation
            cached = self._cached
            rec = self._recording
            snapshot = rec.copy() if rec is not None and cached is None else None
            serial = self._rec_serial
        if cached is not None:
            return {"generation": generation, "serial": serial, "start": 0, "frames": cached[0].get("frames", 0),
                    "data": cached[1]}
        if snapshot is None or not len(snapshot):
            return None
        start = max(0, min(start, len(snapshot)))
        data = snapshot.encoded(start)
        if start == 0:
            with self._lock:
                rec.adopt_encoding(snapshot)
        return {"generation": generation, "serial": serial, "start": start, "frames": len(snapshot), "data": data}

    def _recording_status(self) -> Optional[dict]:
        if self._cached is not None:
            meta = self._cached[0]
            return {"id": self._rec_serial, "frames": meta.get("frames", 0), "end": meta.get("end", 0.0),
                    "cached": True, "saved": True}
        rec = self._recording
        if rec is None:
            return None
        return {"id": self._rec_serial, "frames": len(rec), "end": rec.end_time, "cached": False, "saved": rec.saved,
                "truncated": rec.truncated, "previous": self._previous_run}

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
            "step": self._steps,
            "rtf": self.rtf,
            "fps": self.fps,
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
                    if self._recording is not None:
                        with self._lock:
                            self._recording.expand(m.id, m.inverse)
                    self._undeduped.add(m.id)
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

        lights = self.engine.lights(self._capture) if self._capture else []
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
            if self._recording is not None:
                self._recording.offer(self._sim_time, self._steps, changed, self._full_positions, scalars, lights)
        self._emit(pack_frame(header, arrays))

    def _end_run(self, reason: str) -> None:
        self.finished = True
        self.end_reason = reason
        self.console.note(f"run ended at t={self._sim_time:.3f}s: {reason}")
        self._broadcast_frame()
        self._flush_recording()
        self._save_recording()
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
        if self._recording is not None and (self.finished or self._steps > 0):
            self._recording.pure = False
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
                # the owner resumes it. The run just stopped stays on the
                # timeline until then, if the scene came back the same (same
                # parameters, so the same visual models).
                previous = self._recording if self._cached is None else None
                signature = self._model_signature()
                self._flush_recording()
                self._safe_build(self.params)
                if (previous is not None and len(previous) >= 2 and self._cached is None
                        and self.status == "running" and self._model_signature() == signature):
                    self._recording = previous
                    self._previous_run = True
                    self._emit_status()
                future.set_result(True)
            elif command == "pause":
                self.paused = True
                self._flush_recording()
                self._emit_status()
                future.set_result(True)
            elif command in ("resume", "simulate"):
                if self._cached is not None:
                    self._start_live(paused=False)
                    self._emit_status()
                    future.set_result(True)
                    return
                if self._diverged:
                    self.console.note("the run diverged; rebuilding instead of resuming")
                    self.paused = False
                    self._safe_build(self.params)
                    future.set_result(True)
                    return
                if self.finished:
                    self._extend_run()
                self._leave_previous_run()
                self.paused = False
                self._emit_status()
                future.set_result(True)
            elif command == "presets_changed":
                # An admin saved or deleted a preset: forget recordings of
                # parameters that no longer are one; keep this run if they now are.
                keys = self._preset_keys()
                self._cache.prune(keys)
                if self._recording is not None:
                    if self._recording.saved and self._rec_key not in keys:
                        self._recording.saved = False
                    self._save_recording()
                self._emit_status()
                future.set_result(True)
            elif command == "control":
                future.set_result(self._apply_control(kwargs["key"], kwargs.get("value")))
            elif command == "auto_restart":
                self.auto_restart = bool(kwargs["enabled"])
                self._emit_status()
                future.set_result(True)
            elif command == "params":
                changes: Params = kwargs["changes"]
                rebuild = {k: v for k, v in changes.items() if k not in self.cfg.live_params and v != self.params.get(k)}
                live = {k: v for k, v in changes.items() if k in self.cfg.live_params}
                self.params.update(changes)
                for a, b in self.cfg.exclusive:
                    if self.params.get(a) and self.params.get(b):
                        keep = b if changes.get(b) and not changes.get(a) else a
                        drop = b if keep == a else a
                        self.params[drop] = False
                        self.console.note(f"{drop} switched off: it can't be on together with {keep}")
                if rebuild or kwargs.get("rebuild") or (live and self._diverged):
                    self._safe_build(self.params)
                elif live and self._root is not None:
                    if self._cached is not None:
                        self._start_live(paused=True)
                    self._leave_previous_run()
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
            if self._hold_deadlines:
                self._expire_holds()

            runnable = self._root is not None and self.status == "running" and not self.paused and not self.finished
            if not runnable:
                was_running = False
                self.rtf = 0.0
                self.fps = 0.0
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
                dt = self.engine.dt(self._root)
                step_start = time.monotonic()
                with contextlib.redirect_stdout(self.console):
                    try:
                        self.engine.step(self._root, dt)
                    except Exception as exc:
                        self.console.write(traceback.format_exc())
                        self.status = "error"
                        self.error = f"The simulation step failed: {exc}"
                        self._emit_status()
                        continue
                self._sim_time += dt
                self._steps += 1
                if target - self._sim_time > 0.5:
                    wall_anchor, sim_anchor = time.monotonic(), self._sim_time
                # A step holds the GIL throughout (SOFA's animate doesn't release
                # it), so a scene that steps flat out at ~1 s per step (e.g. a
                # full finger) starved the web server: every request waited a
                # step per GIL hand-over and timed out. Pause briefly after a
                # long step (5 % of it, at most 50 ms) so requests get through.
                step_time = time.monotonic() - step_start
                if step_time > 0.02:
                    time.sleep(min(0.05, 0.05 * step_time))

            now = time.monotonic()
            if now >= next_frame:
                rtf_window.append((now, self._sim_time))
                if len(rtf_window) >= 2 and rtf_window[-1][0] > rtf_window[0][0]:
                    span = rtf_window[-1][0] - rtf_window[0][0]
                    self.rtf = (rtf_window[-1][1] - rtf_window[0][1]) / span
                    self.fps = (len(rtf_window) - 1) / span
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
