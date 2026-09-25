"""HollowCube (HollowCubeInflation.py), run unmodified through sofaweb (see
scenes/_base)."""

from sofaweb.config import SceneConfig


def _ramp(root):
    return root.getObject("PressureRamp")


def probes(root, module):
    """The same quantities PressureRamp logs every 0.2 s."""
    ramp = _ramp(root)
    volume = float(ramp.pressure.cavityVolume.value)
    volume0 = float(ramp.pressure.initialCavityVolume.value)
    return {
        "target": float(ramp.targetPressure.value) / 1000.0,
        "applied": float(ramp.pressure.pressure.value) / 1000.0,
        "volume": volume * 1e6,
        "growth": (volume / volume0 - 1.0) * 100.0 if volume0 else 0.0,
    }


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
    title="Hollow cube inflation",
    script="original/HollowCubeInflation.py",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"HOLLOWCUBE_NO_EDITOR": "1"},
    # Both are GL-only debug drawing (SOFA's force-field display flags).
    forced_params={"show_pressure_overlay": False, "show_fem_elements": False},
    hidden_params={"theme"},
    param_limits={
        "block_size": (1.0, 50.0),
        "length_blocks": (3, 24),
        "width_blocks": (3, 24),
        "height_blocks": (3, 24),
        "wall_length_blocks": (1, 6),
        "wall_width_blocks": (1, 6),
        "wall_height_blocks": (1, 6),
        "trunk_length_blocks": (1, 24),
        "trunk_width_blocks": (1, 24),
        "floor_size_blocks": (5, 200),
        "symmetry_buffer_blocks": (0, 4),
        "young_modulus": (1e4, 1e9),
        "poisson_ratio": (0.0, 0.49),
        "density": (10.0, 20000.0),
        "target_pressure": (-5000.0, 20000.0),
        "ramp_time": (0.0, 60.0),
        "wall_alpha": (0.0, 1.0),
        "interior_alpha": (0.0, 1.0),
    },
    sections=[
        ("Pressure", ["target_pressure", "ramp_time", "instant"]),
        ("Geometry", ["block_size", "length_blocks", "width_blocks", "height_blocks",
                      "wall_length_blocks", "wall_width_blocks", "wall_height_blocks",
                      "trunk_length_blocks", "trunk_width_blocks", "floor_size_blocks"]),
        ("Material", ["young_modulus", "poisson_ratio", "density", "gravity_enabled"]),
        ("Symmetry", ["symmetry_x", "symmetry_y", "symmetry_z", "symmetry_buffer_blocks",
                      "prevent_symmetry_crossing", "mirror_visual_x", "mirror_visual_y", "mirror_visual_z"]),
        ("Display", ["wall_alpha", "interior_alpha", "cutaway", "interior_cutaway"]),
    ],
    live_params=LIVE,
    probes=probes,
    charts=[
        {
            "title": "Cavity pressure",
            "unit": "kPa",
            "series": [
                {"key": "applied", "label": "applied", "color": "#e8933c"},
                {"key": "target", "label": "target", "color": "#8f97a3"},
            ],
        },
        {"title": "Cavity volume growth", "unit": "%", "series": [{"key": "growth", "label": "growth", "color": "#2a78d6"}]},
    ],
    readouts=[
        {"key": "applied", "label": "pressure", "digits": 3, "unit": "kPa"},
        {"key": "volume", "label": "cavity volume", "digits": 2, "unit": "cm³"},
        {"key": "growth", "label": "volume growth", "digits": 1, "unit": "%"},
    ],
    stop_at=settle_time,
    extend_by=settle_time,
    about=[
        "A hollow silicone box, clamped at the bottom to a rigid trunk, is inflated from the inside: "
        "SoftRobots' SurfacePressureConstraint applies the cavity pressure on a corotational tetrahedral FEM "
        "shell, solved with a Gauss-Seidel constraint solver. Blue is the outer wall, orange the cavity.",
        "Target pressure, ramp time and 'instant' are live: change them and the running simulation follows, "
        "just like editing PressureRamp's Data fields in the SOFA GUI. The run pauses once the pressure has "
        "settled (ramp time + 2 s), and continues when you change the pressure or press Resume.",
    ],
    origin="Original scene: verilogscripts/SOFA/HollowCube/HollowCubeInflation.py.",
)
