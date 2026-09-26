"""AsymmetricSquareTube, run unmodified through sofaweb (see scenes/_base)."""

import math

import numpy as np

from sofaweb.config import SceneConfig


def _ramp_factor(t, start, ramp):
    elapsed = t - start
    if ramp > 0:
        return min(max(elapsed, 0.0) / ramp, 1.0)
    return 1.0 if elapsed >= 0 else 0.0


def probes(root, module):
    """The same quantities TopPlateController prints each step."""
    c = root.getObject("topPlateCtrl")
    pos = np.asarray(c.dofs.position.value)
    top = pos[c.top_indices]
    xs, zs = top[:, 0], top[:, 2]
    x_bar = xs.mean()
    a = float(((xs - x_bar) * (zs - zs.mean())).sum() / ((xs - x_bar) ** 2).sum())
    b = float(zs.mean() - a * x_bar)
    cx = module.WIDTH / 2.0
    probe = c.cavity_probes
    return {
        "platen_z": (a * cx + b) * 1000.0,
        "tilt": math.degrees(math.atan(a)),
        "cavity_w": (pos[probe["right"]][0] - pos[probe["left"]][0]) * 1000.0,
        "cavity_h": (pos[probe["top"]][2] - pos[probe["bottom"]][2]) * 1000.0,
        "top_pressure": module.PRESSURE * _ramp_factor(c.sim_time, module.TOP_RAMP_START, module.TOP_RAMP_TIME) / 1e5,
        "cavity_pressure": module.CAVITY_PRESSURE
        * _ramp_factor(c.sim_time, module.CAVITY_RAMP_START, module.CAVITY_RAMP_TIME)
        / 1e5,
    }


def stop_at(p):
    # Both pressure ramps complete, plus a second to settle.
    return max(p["top_ramp_start"] + p["top_ramp_time"], p["cavity_ramp_start"] + p["cavity_ramp_time"]) + 1.0


CONFIG = SceneConfig(
    title="Asymmetric square tube",
    script="original/AsymmetricSquareTube.py",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"ASYMMETRICSQUARETUBE_NO_EDITOR": "1"},
    # show_fem_elements is SOFA's GL-only debug drawing of the tetrahedra.
    forced_params={"console_log_enabled": False, "show_fem_elements": False},
    hidden_params={"theme", "run_duration"},
    param_limits={
        "outer_width": (0.01, 1.0),
        "outer_height": (0.01, 1.0),
        "divs_x": (3, 60),
        "divs_z": (3, 60),
        "depth": (0.001, 0.5),
        "young_modulus": (1e4, 1e9),
        "poisson_ratio": (0.0, 0.49),
        "density": (10.0, 20000.0),
        "cavity_width": (0.0, 1.0),
        "cavity_height": (0.0, 1.0),
        "cavity_center_x": (0.0, 1.0),
        "cavity_center_z": (0.0, 1.0),
        "top_pressure": (0.0, 1e7),
        "top_ramp_start": (0.0, 60.0),
        "top_ramp_time": (0.0, 60.0),
        "cavity_pressure": (-1e6, 1e7),
        "cavity_ramp_start": (0.0, 60.0),
        "cavity_ramp_time": (0.0, 60.0),
        "gravity": (0.0, 30.0),
        "dt": (0.001, 0.05),
        "rayleigh_stiffness": (0.0, 1.0),
        "rayleigh_mass": (0.0, 1.0),
        "block_alpha": (0.0, 1.0),
        "key_light_rotate_period": (0.5, 120.0),
    },
    sections=[
        ("Geometry", ["outer_width", "outer_height", "divs_x", "divs_z", "depth",
                      "cavity_width", "cavity_height", "cavity_center_x", "cavity_center_z"]),
        ("Material", ["young_modulus", "poisson_ratio", "density", "rayleigh_stiffness", "rayleigh_mass"]),
        ("Loads", ["top_pressure", "top_ramp_start", "top_ramp_time",
                   "cavity_pressure", "cavity_ramp_start", "cavity_ramp_time", "gravity", "gravity_enabled"]),
        ("Simulation", ["dt"]),
        ("Appearance", ["block_fill_color", "block_alpha", "block_show_edges", "block_edge_color", "show_platen"]),
        ("Lighting", ["ambient_color", "key_light_direction", "key_light_color", "fill_light_direction",
                      "fill_light_color", "key_light_rotate_enabled", "key_light_rotate_period"]),
    ],
    probes=probes,
    charts=[
        {
            "title": "Cavity size",
            "unit": "mm",
            "series": [
                {"key": "cavity_w", "label": "width", "color": "#2a78d6"},
                {"key": "cavity_h", "label": "height", "color": "#eb6834"},
            ],
        },
        {"title": "Platen height", "unit": "mm", "series": [{"key": "platen_z", "label": "center", "color": "#8f97a3"}]},
        {
            "title": "Applied pressure",
            "unit": "bar",
            "series": [
                {"key": "top_pressure", "label": "top", "color": "#1baf7a"},
                {"key": "cavity_pressure", "label": "cavity", "color": "#e87ba4"},
            ],
        },
    ],
    readouts=[
        {"key": "platen_z", "label": "platen height", "digits": 3, "unit": "mm"},
        {"key": "tilt", "label": "platen tilt", "digits": 3, "unit": "deg"},
        {"key": "cavity_w", "label": "cavity width", "digits": 3, "unit": "mm"},
        {"key": "cavity_h", "label": "cavity height", "digits": 3, "unit": "mm"},
        {"key": "top_pressure", "label": "top pressure", "digits": 2, "unit": "bar"},
        {"key": "cavity_pressure", "label": "cavity pressure", "digits": 2, "unit": "bar"},
    ],
    stop_at=stop_at,
    about=[
        "A real FEM simulation (corotational tetrahedra, implicit Euler) of a silicone-rubber tube cross-section "
        "with an off-center rectangular cavity, so each wall has a different thickness. The bottom is pinned "
        "vertically; a rigid platen on top presses down with a uniform pressure and is free to tilt, following "
        "a least-squares line fitted to the top row every step. Then the cavity is pressurized.",
        "The run ends one second after both pressure ramps have finished. Resume to keep it going.",
    ],
    origin="Original scene: verilogscripts/SOFA/AsymmetricSquareTube/AsymmetricSquareTube.py.",
)
