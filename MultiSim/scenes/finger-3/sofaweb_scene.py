"""Finger 3 (Finger3.py), run through sofaweb (see scenes/_base)."""

import math

from sofaweb.config import SceneConfig


def _ramp(root):
    return root.getObject("PressureRamp")


def _end_frame(root):
    """The Rigid3 frame holding the +X far-end plane, or None when no duct
    is built or its end constraint is off."""
    finger = root.getChild("finger")
    rigid = finger.getChild("rigidEnds") if finger is not None else None
    return rigid.getObject("dofs") if rigid is not None else None


def probes(root, module):
    ramp = _ramp(root)
    volume = float(ramp.pressure.cavityVolume.value)
    volume0 = float(ramp.pressure.initialCavityVolume.value)
    values = {
        "target": float(ramp.targetPressure.value) / 1000.0,
        "applied": float(ramp.pressure.pressure.value) / 1000.0,
        "volume": volume * 1e6,
        "growth": (volume / volume0 - 1.0) * 100.0 if volume0 else 0.0,
    }
    frame = _end_frame(root)
    if frame is not None:
        position = frame.position.value[0]
        rest = frame.rest_position.value[0]
        qy, qw = float(position[4]), float(position[6])
        angle = math.degrees(2.0 * math.atan2(qy, qw))
        values["end_shift"] = (float(position[0]) - float(rest[0])) * 1000.0
        values["end_angle"] = angle
        # The chamber's midplane (x = 0) stays put and the far-end plane
        # tilts by `angle`: one chamber pitch of the finger bends by twice that.
        values["bend"] = 2.0 * angle
    return values


class _Settled:
    """Ends a run once the pressure has reached its target and the cavity
    volume has stopped changing (less than 0.02 % over 0.3 s)."""

    def __init__(self):
        self.history = []

    def __call__(self, root, module):
        ramp = _ramp(root)
        t = float(root.time.value)
        volume = float(ramp.pressure.cavityVolume.value)
        if self.history and t < self.history[-1][0]:
            self.history = []  # rebuilt
        self.history.append((t, volume))
        self.history = [h for h in self.history if h[0] >= t - 0.3 - 1e-9]
        target = float(ramp.targetPressure.value)
        applied = float(ramp.pressure.pressure.value)
        if abs(applied - target) > 1e-6 * max(1.0, abs(target)) or t - self.history[0][0] < 0.3 - 1e-9:
            return None
        volumes = [v for _, v in self.history]
        if max(volumes) - min(volumes) <= 2e-4 * max(abs(volume), 1e-12):
            return f"Settled (t = {t:.2f} s): the volume no longer changes."
        return None


def _set(data_name, cast):
    def apply(root, value):
        getattr(_ramp(root), data_name).value = cast(value)
    return apply


LIVE = {
    "target_pressure": _set("targetPressure", float),
    "ramp_time": _set("rampTime", float),
    "instant": _set("instant", bool),
}


def stop_time(p):
    # A safety limit; runs normally end earlier, once settled.
    return p["ramp_time"] + 15.0


def extend_time(p):
    return p["ramp_time"] + 10.0


CONFIG = SceneConfig(
    title="Finger 3",
    script="original/Finger3.py",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"FINGER3_NO_EDITOR": "1"},
    # GL-only debug drawing, and the desktop GUI's per-step mirroring (the
    # browser mirrors instead, see `display`).
    forced_params={"show_pressure_overlay": False, "show_fem_elements": False,
                   "mirror_visual_x": False, "mirror_visual_y": False},
    hidden_params={"theme"},
    param_limits={
        "element_size": (2.0, 50.0),
        "min_wall_layers": (1, 4),
        "length": (5.0, 300.0),
        "width": (5.0, 300.0),
        "height": (5.0, 300.0),
        "wall_end": (0.5, 50.0),
        "wall_side": (0.5, 50.0),
        "wall_bottom": (0.5, 50.0),
        "wall_top": (0.5, 50.0),
        "channel_height": (0.0, 100.0),
        "channel_width": (0.0, 300.0),
        "channel_wall": (0.5, 50.0),
        "channel_separation": (0.0, 100.0),
        "trunk_length": (0.0, 300.0),
        "trunk_width": (0.0, 300.0),
        "floor_size": (20.0, 2000.0),
        "symmetry_buffer": (0.0, 50.0),
        "strain_limiting_stiffness": (1e3, 1e8),
        "young_modulus": (1e4, 1e9),
        "poisson_ratio": (0.0, 0.49),
        "density": (10.0, 20000.0),
        "target_pressure": (-3000.0, 100000.0),
        "ramp_time": (0.0, 60.0),
        "wall_alpha": (0.0, 1.0),
        "interior_alpha": (0.0, 1.0),
        "channel_end_alpha": (0.0, 1.0),
    },
    sections=[
        ("Pressure", ["target_pressure", "ramp_time", "instant"]),
        ("Mesh", ["element_size", "min_wall_layers"]),
        ("Geometry (mm)", ["length", "width", "height", "wall_end", "wall_side", "wall_bottom", "wall_top",
                           "trunk_length", "trunk_width", "floor_size"]),
        ("Channel (mm)", ["channel_height", "channel_width", "channel_wall", "channel_separation"]),
        ("Far-end plane (midplane to the next chamber)", ["channel_end_constraint", "channel_end_sliding",
                                                          "prevent_channel_end_crossing"]),
        ("Base", ["clamp_to_trunk", "strain_limiting_layer", "strain_limiting_stiffness"]),
        ("Material", ["hyperelastic", "young_modulus", "poisson_ratio", "density", "gravity_enabled"]),
        ("Symmetry", ["symmetry_x", "symmetry_y", "symmetry_buffer", "prevent_symmetry_crossing",
                      "mirror_visual_x", "mirror_visual_y"]),
        ("Cutaway (rebuilds the skin)", ["cutaway", "interior_cutaway"]),
    ],
    display=[
        {"key": "wall_alpha", "label": "Wall opacity", "type": "opacity", "models": r"(^|/)cube/visu/visual$"},
        {"key": "interior_alpha", "label": "Interior opacity", "type": "opacity",
         "models": r"(^|/)cube/interior_visu/visual$"},
        {"key": "channel_end_alpha", "label": "Far-end plane opacity", "type": "opacity",
         "models": r"^/channelEndPlane"},
        {"key": "floor_alpha", "label": "Floor opacity", "type": "opacity", "models": r"^/floor/visual$"},
        {"key": "axes_alpha", "label": "Coordinate axes (X, Y) opacity", "type": "opacity",
         "models": r"^/(arrow|label)_[xyz]/visual$"},
        {"key": "mirror_visual_x", "label": "Mirror across X symmetry plane", "type": "mirror", "axis": "X",
         "models": r"(^|/)cube/(interior_)?visu/visual$", "requires": "symmetry_x"},
        {"key": "mirror_visual_y", "label": "Mirror across Y symmetry plane", "type": "mirror", "axis": "Y",
         "models": r"(^|/)cube/(interior_)?visu/visual$", "requires": "symmetry_y"},
        {"key": "mirror_far_end", "label": "Mirror across far-end plane", "type": "mirror",
         "planeModel": r"^/channelEndPlanePos/visual$", "default": False,
         "chain": {"countKey": "mirror_far_end_count", "max": 12, "axis": "X", "plane": 0.0, "half": "symmetry_x"},
         "models": r"(^|/)cube/(interior_)?visu/visual$", "requires": "channel_separation"},
        {"key": "show_symmetry_buffer", "label": "Show symmetry buffer elements (FEM)", "type": "femRegion",
         "requires": "symmetry_buffer",
         "regions": [{"axis": "X", "below": 0.0, "requires": "symmetry_x"},
                     {"axis": "Y", "below": 0.0, "requires": "symmetry_y"}]},
    ],
    live_params=LIVE,
    probes=probes,
    charts=[
        {
            "title": "Chamber pressure",
            "unit": "kPa",
            "series": [
                {"key": "applied", "label": "applied", "color": "#e8933c"},
                {"key": "target", "label": "target", "color": "#8f97a3"},
            ],
        },
        {"title": "Chamber volume growth", "unit": "%", "series": [{"key": "growth", "label": "growth", "color": "#2a78d6"}]},
        {
            "title": "Far-end plane",
            "unit": "deg / mm",
            "series": [
                {"key": "end_angle", "label": "tilt (deg)", "color": "#c77dff"},
                {"key": "end_shift", "label": "X shift (mm)", "color": "#1baf7a"},
            ],
        },
    ],
    readouts=[
        {"key": "applied", "label": "pressure", "digits": 3, "unit": "kPa"},
        {"key": "volume", "label": "chamber volume", "digits": 2, "unit": "cm³"},
        {"key": "growth", "label": "volume growth", "digits": 1, "unit": "%"},
        {"key": "end_angle", "label": "far-end plane tilt", "digits": 2, "unit": "deg"},
        {"key": "bend", "label": "finger bend per chamber", "digits": 1, "unit": "deg"},
        {"key": "end_shift", "label": "far-end plane X shift", "digits": 3, "unit": "mm"},
    ],
    stop_at=stop_time,
    extend_by=extend_time,
    end_condition=_Settled(),
    about=[
        "One chamber of a PneuNet soft pneumatic finger, like Better finger, with every dimension given in "
        "millimetres plus a standard element size. Each feature is split into the whole number of equal elements "
        "closest to that size (an 18 mm wall with 10 mm elements becomes 2 x 9 mm), and every wall into at least "
        "'minimum wall layers' elements: linear tetrahedra need more than one layer through a wall to bend "
        "realistically. The mesh is mirror-symmetric, so a model cut at a midplane is exactly half the full one.",
        "The far-end plane (the midplane to the neighbouring chamber, halfway along the duct) is a symmetry plane: "
        "its nodes stay on it but slide freely within it, while the plane itself shifts along X and tilts about Y. "
        "In Better finger those nodes were glued to the plane, which froze the duct's cross-section there. With "
        "sliding, the channel can open up under pressure all the way to the midplane -- and with the default "
        "full-width channel the end wall above it acts as a flap. At about 1 kPa that flap has no equilibrium any more "
        "(it keeps opening until the run diverges). A narrower channel (Channel width, e.g. 16 mm with 4 mm duct "
        "walls) is closer to a real PneuNet and stays stable.",
        "Where the end wall reaches the far-end plane it touches its mirrored neighbour: that contact now pushes "
        "the plane too (before, it only held the wall back), so it can bend the finger.",
        "Options closer to a real finger: a strain-limiting layer (paper) on the bottom face instead of the clamped "
        "trunk, and neo-Hookean hyperelastic silicone for large strains. Finger bend per chamber = twice the "
        "far-end plane's tilt. The run ends once the volume has settled.",
    ],
    origin="Scene script: scenes/finger-3/original/Finger3.py (successor of scenes/better-finger).",
)
