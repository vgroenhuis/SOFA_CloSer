"""Better finger (BetterFinger.py), run through sofaweb (see scenes/_base)."""

import math

from sofaweb.config import SceneConfig


def _ramp(root):
    return root.getObject("PressureRamp")


def _end_frame(root):
    """The Rigid3 frame holding the +X channel far-end plane (the last one
    when both ends are built), or None when channel_end_constraint is off
    or no duct is built."""
    finger = root.getChild("finger")
    rigid = finger.getChild("rigidEnds") if finger is not None else None
    return rigid.getObject("dofs") if rigid is not None else None


def probes(root, module):
    """PressureRamp's own quantities, plus the far-end plane's motion read
    straight from its rigid frame (its only free DOFs: X and rotation about Y)."""
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
        position = frame.position.value[-1]
        rest = frame.rest_position.value[-1]
        qy, qw = float(position[4]), float(position[6])
        values["end_shift"] = (float(position[0]) - float(rest[0])) * 1000.0
        values["end_angle"] = math.degrees(2.0 * math.atan2(qy, qw))
    return values


def _set(data_name, cast):
    def apply(root, value):
        getattr(_ramp(root), data_name).value = cast(value)
    return apply


# PressureRamp's own Data fields -- the ones the original lets you edit live
# in the SOFA GUI's properties panel.
LIVE = {
    "target_pressure": _set("targetPressure", float),
    "ramp_time": _set("rampTime", float),
    "instant": _set("instant", bool),
}


def settle_time(p):
    # The original's headless run length: ramp time plus 2 s.
    return p["ramp_time"] + 2.0


CONFIG = SceneConfig(
    title="Better finger",
    script="original/BetterFinger.py",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"BETTERFINGER_NO_EDITOR": "1"},
    # The first two are GL-only debug drawing (SOFA's force-field display
    # flags). Mirroring is done by the browser (Display panel) instead of the
    # scene's per-step MirrorController, which costs simulation time.
    forced_params={"show_pressure_overlay": False, "show_fem_elements": False,
                   "mirror_visual_x": False, "mirror_visual_y": False},
    hidden_params={"theme"},
    param_limits={
        "block_size": (1.0, 50.0),
        "length_blocks": (3, 24),
        "width_blocks": (3, 40),
        "height_blocks": (3, 24),
        "wall_length_blocks": (1, 6),
        "wall_width_blocks": (1, 6),
        "wall_height_blocks": (1, 6),
        # 0 is only kept with X symmetry and a symmetry buffer (the buffer's
        # floor is then the attachment); otherwise the script makes it 1+.
        "trunk_length_blocks": (0, 24),
        "trunk_width_blocks": (1, 40),
        "floor_size_blocks": (5, 200),
        "channel_height_blocks": (1, 10),
        "channel_separation_blocks": (0.0, 10.0),
        "symmetry_buffer_blocks": (0, 4),
        "young_modulus": (1e4, 1e9),
        "poisson_ratio": (0.0, 0.49),
        "density": (10.0, 20000.0),
        "target_pressure": (-3000.0, 10000.0),
        "ramp_time": (0.0, 60.0),
        "wall_alpha": (0.0, 1.0),
        "interior_alpha": (0.0, 1.0),
        "channel_end_alpha": (0.0, 1.0),
    },
    sections=[
        ("Pressure", ["target_pressure", "ramp_time", "instant"]),
        ("Geometry", ["block_size", "length_blocks", "width_blocks", "height_blocks",
                      "wall_length_blocks", "wall_width_blocks", "wall_height_blocks",
                      "trunk_length_blocks", "trunk_width_blocks", "floor_size_blocks"]),
        ("Channel", ["channel_height_blocks", "channel_separation_blocks", "channel_end_constraint",
                     "prevent_channel_end_crossing"]),
        ("Material", ["young_modulus", "poisson_ratio", "density", "gravity_enabled"]),
        ("Symmetry", ["symmetry_x", "symmetry_y", "symmetry_buffer_blocks", "prevent_symmetry_crossing",
                      "mirror_visual_x", "mirror_visual_y"]),
        ("Cutaway (rebuilds the skin)", ["cutaway", "interior_cutaway"]),
    ],
    display=[
        {"key": "wall_alpha", "label": "Wall opacity", "type": "opacity", "models": r"(^|/)cube/visu/visual$"},
        {"key": "interior_alpha", "label": "Interior opacity", "type": "opacity",
         "models": r"(^|/)cube/interior_visu/visual$"},
        {"key": "channel_end_alpha", "label": "Channel end-plane opacity", "type": "opacity",
         "models": r"^/channelEndPlane"},
        {"key": "floor_alpha", "label": "Floor opacity", "type": "opacity", "models": r"^/floor/visual$"},
        {"key": "axes_alpha", "label": "Coordinate axes (X, Y) opacity", "type": "opacity",
         "models": r"^/(arrow|label)_[xyz]/visual$"},
        {"key": "mirror_visual_x", "label": "Mirror across X symmetry plane", "type": "mirror", "axis": "X",
         "models": r"(^|/)cube/(interior_)?visu/visual$", "requires": "symmetry_x"},
        {"key": "mirror_visual_y", "label": "Mirror across Y symmetry plane", "type": "mirror", "axis": "Y",
         "models": r"(^|/)cube/(interior_)?visu/visual$", "requires": "symmetry_y"},
        # Follows the far-end plane (it shifts and tilts) via its indicator
        # model, which exists whenever the channel duct is built.
        {"key": "mirror_far_end", "label": "Mirror across far-end symmetry plane", "type": "mirror",
         "planeModel": r"^/channelEndPlanePos/visual$", "default": False,
         # Repeats along the finger: up to 12 more chambers (see the viewer).
         "chain": {"countKey": "mirror_far_end_count", "max": 12, "axis": "X", "plane": 0.0, "half": "symmetry_x"},
         "models": r"(^|/)cube/(interior_)?visu/visual$", "requires": "channel_separation_blocks"},
        {"key": "show_symmetry_buffer", "label": "Show symmetry buffer elements (FEM)", "type": "femRegion",
         "requires": "symmetry_buffer_blocks",
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
            "title": "Channel end plane",
            "unit": "deg / mm",
            "series": [
                {"key": "end_angle", "label": "rotation (deg)", "color": "#c77dff"},
                {"key": "end_shift", "label": "X shift (mm)", "color": "#1baf7a"},
            ],
        },
    ],
    readouts=[
        {"key": "applied", "label": "pressure", "digits": 3, "unit": "kPa"},
        {"key": "volume", "label": "chamber volume", "digits": 2, "unit": "cm³"},
        {"key": "growth", "label": "volume growth", "digits": 1, "unit": "%"},
        {"key": "end_angle", "label": "end plane rotation", "digits": 2, "unit": "deg"},
        {"key": "end_shift", "label": "end plane X shift", "digits": 3, "unit": "mm"},
    ],
    stop_at=settle_time,
    extend_by=settle_time,
    about=[
        "One chamber of a PneuNet soft pneumatic finger, as in the PneuNet finger scene, with one improvement: "
        "the far end of each channel duct -- the midpoint to the neighbouring chamber -- is held by a hard "
        "constraint instead of a stiff-spring truss. Its FEM nodes are rigidly mapped from a frame that can only "
        "shift along X and rotate about Y, so that cross-section stays an exact rigid copy of its rest shape "
        "(the spring version deviates by over a millimetre). That is what lets copies of this chamber be placed "
        "side by side to form a whole finger.",
        "The guard that stops the chamber wall from bulging past that far-end plane follows the frame exactly, "
        "tilt included, so the bellows can expand all the way to the tilted plane (in the PneuNet finger scene it "
        "only follows the plane's X shift, and stops the wall at a copy of the rest orientation).",
        "The end plane's rotation and X shift are read directly from that frame and charted. Target pressure, "
        "ramp time and 'instant' are live. The run pauses once the pressure has settled (ramp time + 2 s), and "
        "continues when you change the pressure or press play.",
    ],
    origin="Scene script: scenes/better-finger/original/BetterFinger.py (derived from verilogscripts/SOFA/PneuNetFinger).",
)
