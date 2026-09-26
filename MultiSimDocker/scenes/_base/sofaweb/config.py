"""Per-scene configuration: what a scene folder's sofaweb_scene.py declares."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

Params = dict[str, Any]


@dataclass
class SceneConfig:
    title: str
    # The original scene script and its params.json, relative to the
    # config file. The script is run unmodified: its params.json is
    # rewritten and the script re-imported for every (re)build, which is
    # exactly what the SOFA GUI's Reload button does.
    script: str
    params_file: Optional[str] = "params.json"
    # The original Tk params_editor.py, read (never imported) for its
    # field labels / units / display scales.
    editor_file: Optional[str] = "params_editor.py"
    env: dict[str, str] = field(default_factory=dict)

    # Always applied on top of the user's parameters (e.g. console logging
    # off, debug overlays that need GL off).
    forced_params: Params = field(default_factory=dict)
    # Parameters not shown in the web panel (they keep their params.json
    # value unless forced).
    hidden_params: set[str] = field(default_factory=set)
    # (min, max) in stored units -- guards against e.g. a mesh resolution
    # that would take down the container.
    param_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Label overrides for editor fields whose meaning changes on the web.
    labels: dict[str, str] = field(default_factory=dict)
    # Optional grouping of the panel: [(title, [keys])]; unlisted visible
    # keys go into a final "Other" group.
    sections: Optional[list[tuple[str, list[str]]]] = None
    # Parameters applied to the running scene without a rebuild:
    # key -> fn(root, value).
    live_params: dict[str, Callable[[Any, Any], None]] = field(default_factory=dict)

    # Telemetry: fn(root, module) -> {key: float}, sampled every frame.
    probes: Optional[Callable[[Any, Any], dict[str, float]]] = None
    # [{"title", "unit", "series": [{"key", "label", "color"}]}]
    charts: list[dict] = field(default_factory=list)
    # [{"key", "label", "digits", "unit"}]
    readouts: list[dict] = field(default_factory=list)

    # Simulated time at which to stop (pause as "finished"), or None to run
    # on indefinitely: fn(params) -> seconds.
    stop_at: Optional[Callable[[Params], Optional[float]]] = None
    # After a live parameter change or a resume past the end, run this many
    # more simulated seconds: fn(params) -> seconds.
    extend_by: Optional[Callable[[Params], float]] = None
    # fn(root, module) -> reason string when the run should end early.
    end_condition: Optional[Callable[[Any, Any], Optional[str]]] = None
    # When a run ends: restart automatically (after a short pause) or stop.
    auto_restart: bool = False

    # "3d": orbit/pan/zoom; "2d": pan/zoom only, painter's-order drawing.
    view: str = "3d"
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    frames_per_second: float = 25.0
    about: list[str] = field(default_factory=list)
    origin: str = ""


def editor_fields(editor_path: Path) -> list[dict]:
    """Field specs from an original params_editor.py's literal _FIELDS /
    _VECTOR_FIELDS / _BOOL_FIELDS tables, parsed with ast (the editor
    imports tkinter, which the container doesn't have)."""
    if not editor_path or not editor_path.is_file():
        return []
    tree = ast.parse(editor_path.read_text(encoding="utf-8"))
    tables: dict[str, list] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("_FIELDS", "_VECTOR_FIELDS", "_BOOL_FIELDS")
        ):
            try:
                tables[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    specs = []
    for label, key, scale, decimals, unit in tables.get("_FIELDS", []):
        specs.append({"key": key, "label": label, "type": "number", "scale": scale, "decimals": decimals, "unit": unit})
    for label, key, length, decimals, unit in tables.get("_VECTOR_FIELDS", []):
        specs.append({"key": key, "label": label, "type": "vector", "length": length, "decimals": decimals, "unit": unit})
    for label, key, _default in tables.get("_BOOL_FIELDS", []):
        specs.append({"key": key, "label": label, "type": "bool"})
    return specs
