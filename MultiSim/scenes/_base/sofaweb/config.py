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
    # The physics engine that runs `script` (see engines.py): "sofa"
    # (createScene(root) on a SOFA root node) or "mujoco" (createScene()
    # returns a sofaweb.mjscene.MjScene).
    engine: str = "sofa"
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
    # Interactive controls (the Controls panel): act on the running
    # simulation at once, like a hand on the model. Only the owner can use
    # them; watchers see where they are. Each is
    #   {"key", "label", "type": "slider", "min", "max", "step", "default",
    #    "unit", "digits", "apply": fn(root, value)}
    #       -- "wide": True gives the slider a row of its own, the panel's full width
    #          (e.g. a position, finer to set on a touchscreen);
    #   {"key", "label", "type": "toggle", "default", "apply": fn(root, on)}
    #   {"key", "label", "type": "hold", "apply": fn(root, pressed)}
    #       -- on while the button is held down (released automatically if
    #          the owner's browser stops confirming it, e.g. disconnects);
    #   {"key", "label", "type": "button", "apply": fn(root)}
    #       -- one action per click (e.g. a kick).
    # Optional "description" (tooltip). Every build starts them at their
    # default (applied after the scene is built). A run that used them is
    # never kept as a preset's recorded run.
    controls: list[dict] = field(default_factory=list)

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

    # Display controls, applied in each viewer's browser without a rebuild
    # (and per viewer: watchers can change them too). Each is tied to a
    # params.json key, which supplies its initial value and is left out of
    # the parameter panel:
    #   {"key", "label", "type": "opacity" | "color" | "visible", "models": regex}
    #       -- applies to the visual models whose SOFA path matches `models`;
    #   {"key", "label", "type": "mirror", "axis": "X" | "Y" | "Z", "models": regex,
    #    "plane": float (default 0), "requires": param key (optional)}
    #       -- the browser draws reflected copies of the matching models
    #          across that axis's plane (free for the server: shared
    #          geometry, negative scale), shown only while `requires` (e.g.
    #          symmetry_x) is on; combinations of axes (XY) are included;
    #       "planeModel": regex instead of "axis": the plane follows the
    #          visual model that marks it (e.g. a moving far-end plane);
    #       "default": the initial value when the key isn't in params.json;
    #   {"key", "label", "type": "femRegion", "regions": [{"axis": "X" | "Y" | "Z",
    #    "below" | "above": float, "requires": param key (optional)}, ...],
    #    "requires": param key (optional)}
    #       -- a switch for the FEM element overlays: while off, elements
    #          whose rest centre lies in any of the regions (each only
    #          while its own `requires` is on) are hidden, e.g. the
    #          symmetry buffer blocks past a cut plane. Its key needn't be
    #          in params.json (shown by default).
    # Any control with a `requires` is grayed out while that parameter is
    # off (falsy) in the running scene.
    # A "visible" control only has something to show if the model is
    # built, so force its params.json flag on in forced_params (and force a
    # scene's own server-side mirroring off when using "mirror").
    display: list[dict] = field(default_factory=list)

    # Built-in named presets, offered next to "Default" in each panel's
    # preset list: {"params" | "display" | "lighting": [{"name", "values",
    # "description" (optional)}]}. `values` only needs what differs from
    # the defaults (loading a preset starts from the defaults). An admin
    # can overwrite or delete them like their own (stored per scene in the
    # orchestrator).
    presets: dict[str, list[dict]] = field(default_factory=dict)

    # Parameter presets importable from other scenes (sibling folders under
    # scenes/), e.g. to compare two models of the same thing:
    #   {"<scene id>": {"title": "Finger 5", "convert": fn(values) -> {
    #       "params": {key: value}  -- this scene's parameters that differ in
    #                                  name or meaning, converted;
    #       "handled": [source keys] -- source parameters dealt with above;
    #       "notes": [(kind, key, text)]}}}
    # `values` is the source preset's full parameter set (its defaults with
    # the preset on top). Parameters with the same name in both scenes are
    # copied as they are, unless set by `params` or listed in `handled`.
    # kind: "not imported" (a source parameter with no counterpart here, key
    # its key), "estimated" or "assumed" (a parameter of this scene, key its
    # key). Source parameters neither copied nor handled are reported as not
    # imported, and this scene's parameters left at their default as assumed.
    # The scene's scene.json lists the same ids under "importPresetsFrom", so
    # the orchestrator hands over their admin-saved presets too.
    import_presets: dict[str, dict] = field(default_factory=dict)

    # A pressure gauge drawn in the 3D scene by the viewer (an analog manometer
    # with a needle, on a tube into the model), for scenes with a pressure:
    # fn(root, module) -> {
    #   "key": scalar (a probe) the needle shows,     "min", "max": the scale's ends,
    #   "unit": "kPa",  "major": step of the labelled ticks, "minor": step of the small ones,
    #   "center": [x, y, z], "radius": dial radius (all in the scene's units, m),
    #   "right": [x, y, z], "up": [x, y, z]  -- the dial's right and up directions
    #                                           (it faces right x up: towards the viewer),
    #   "tube": [[x, y, z], ...]  -- waypoints of the tube (smoothed), the last one at the
    #                                gauge's casing, "tubeRadius": ...,
    # } (or a list of these, one gauge each) evaluated at every build. Switchable in the Display panel; the needle follows
    # recorded playback too.
    gauge: Optional[Callable[[Any, Any], Optional[dict]]] = None

    # Steering an end-effector in the 3D view (the viewer's manipulator.js): a gizmo at the target pose, dragging
    # the end-effector itself, double-click / double-tap to go somewhere, a height slider, a grip button and the
    # keyboard, all acting on slider controls (and in step with them):
    #   {"position": [x, y, z control keys],  "scale": control units -> scene units (e.g. 1e-3 for mm in m),
    #    "rotation": [roll, pitch, yaw control keys] (degrees, R = Rz(yaw) Ry(pitch) Rx(roll); optional),
    #    "grip": {"key": a slider control, "on": value, "off": value, "label", "releaseLabel"} (optional),
    #    "models": regex of the visual models that can be grabbed and dragged (the end-effector),
    #    "jogSpeed": keyboard speed (control units/s), "jogTurn": keyboard yaw speed (deg/s),
    #    "markerRadius": the target marker's radius (scene units, default 0.008),
    #    "buttons": [{"key": a button control, "label", "title"}] -- extra toolbar buttons (e.g. a repair)}
    manipulator: Optional[dict] = None

    # Vector parameters that may be given fewer values than their default has (at least one),
    # e.g. {"pressures"}: one pressure for every finger, or one each. The scene reads the list.
    short_vectors: set[str] = field(default_factory=set)

    # Pairs of boolean parameters that can't both be on: switching one on in
    # the panel switches the other off (and a change that turns both on keeps
    # the one just switched on).
    exclusive: list[tuple[str, str]] = field(default_factory=list)

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
