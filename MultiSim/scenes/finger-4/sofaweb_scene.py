"""Finger 4 with strain-limiting layer (Finger4.py), run through sofaweb (see scenes/_base)."""

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
        "applied": ramp.applied_pressure() / 1000.0,
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
        applied = ramp.applied_pressure()
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


# PneuNet-like proportions (Mosadegh et al. 2014 scale): an 8 mm chamber
# pitch, 16 x 16 mm cross-section, 1.5 mm walls, a 4 mm base and a 2 x 2 mm
# connecting channel, on a 1 mm mesh with a free base at 50 kPa.
_PNEUNET = {
    "element_size": 1.0, "length": 8.0, "width": 16.0, "height": 16.0,
    "wall_end": 1.5, "wall_side": 1.5, "wall_bottom": 4.0, "wall_top": 1.5,
    "channel_height": 2.0, "channel_width": 2.0, "channel_wall": 1.5, "channel_separation": 1.5,
    "floor_size": 60.0, "clamp_to_trunk": False, "target_pressure": 50000.0,
}

PRESETS = {
    "params": [
        {"name": "No strain-limiting layer (as Finger 3)",
         "description": "Finger 3's model: the base without paper. Bends 48.0 deg per chamber at 30 kPa, "
                        "against 48.3 deg with the paper layer (Default).",
         "values": {"strain_limiting_layer": False}},
        {"name": "Bonded layer that carries compression",
         "description": "The layer as a linear membrane that resists compression as well as tension (a stiff "
                        "sheet fully bonded into the silicone). Bends 40.8 deg per chamber: 15 % less than "
                        "without a layer.",
         "values": {"strain_layer_tension_only": False}},
        {"name": "Thin 2 mm base, free (paper)",
         "description": "A 2 mm base without the clamped trunk. Paper here increases the bend: 69.5 deg per "
                        "chamber, against 61.4 deg without the layer, because it stops the thin base widening.",
         "values": {"wall_bottom": 2.0, "clamp_to_trunk": False}},
        {"name": "PneuNet proportions, no layer",
         "description": "A real PneuNet's proportions: 8 mm chamber pitch, 16 x 16 mm section, 1.5 mm walls, "
                        "4 mm base, 2 x 2 mm channel, 1 mm elements, free base, 50 kPa. No strain-limiting "
                        "layer: 23.7 deg per chamber.",
         "values": {**_PNEUNET, "strain_limiting_layer": False}},
        {"name": "PneuNet proportions, paper layer",
         "description": "As above with 0.1 mm paper under the base: 23.2 deg per chamber, almost no change.",
         "values": {**_PNEUNET, "strain_limiting_layer": True, "strain_layer_tension_only": True}},
        {"name": "PneuNet proportions, bonded layer",
         "description": "As above with a layer that also carries compression: 17.3 deg per chamber, 27 % "
                        "less than without a layer -- the largest effect of a strain-limiting layer here.",
         "values": {**_PNEUNET, "strain_limiting_layer": True, "strain_layer_tension_only": False}},
    ],
    "display": [
        {"name": "Whole finger",
         "description": "The quarter model mirrored into a finger of several chambers, without the far-end planes.",
         "values": {"mirror_visual_x": True, "mirror_visual_y": True, "mirror_far_end": True,
                    "mirror_far_end_count": 2, "channel_end_alpha:on": False, "axes_alpha:on": False}},
        {"name": "See-through walls",
         "description": "Translucent outer walls, showing the cavity and the strain-limiting layer.",
         "values": {"wall_alpha": 0.3, "wall_alpha:on": True, "interior_alpha": 1.0, "interior_alpha:on": True,
                    "mirror_visual_x": True, "mirror_visual_y": True}},
        {"name": "Underside (no floor)",
         "description": "The whole cross-section without the floor, to look at the strain-limiting layer from "
                        "below (orbit underneath; pair with the lighting preset 'Lit from below').",
         "values": {"floor_alpha:on": False, "axes_alpha:on": False, "channel_end_alpha:on": False,
                    "mirror_visual_x": True, "mirror_visual_y": True}},
        {"name": "FEM strain",
         "description": "The tetrahedra coloured by strain, with their edges; surfaces hidden.",
         "values": {"wall_alpha:on": False, "interior_alpha:on": False, "strain_layer_alpha:on": False,
                    "overlay:fem": True, "overlay:femAlpha": 1.0, "overlay:femColor": "strainLit",
                    "overlay:femWire": True, "overlay:femWireAlpha": 0.25}},
    ],
    "lighting": [
        {"name": "Lit from below",
         "description": "Both lights mirrored below the finger, to see the underside and the strain-limiting layer.",
         "values": {"light:0:dir": [-163, -62], "light:1:dir": [22, -29]}},
        {"name": "Soft",
         "description": "More ambient light, softer key light.",
         "values": {"light:ambient:brightness": 0.6, "light:0:brightness": 0.7, "light:1:brightness": 0.45}},
        {"name": "High contrast",
         "description": "Little ambient light and a strong key light, to bring out the shape.",
         "values": {"light:ambient:brightness": 0.12, "light:0:brightness": 1.4, "light:1:brightness": 0.2}},
    ],
}


CONFIG = SceneConfig(
    title="Finger 4 with strain-limiting layer",
    script="original/Finger4.py",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"FINGER4_NO_EDITOR": "1"},
    # GL-only debug drawing, and the desktop GUI's per-step mirroring (the
    # browser mirrors instead, see `display`).
    forced_params={"show_pressure_overlay": False, "show_fem_elements": False,
                   "mirror_visual_x": False, "mirror_visual_y": False},
    hidden_params={"theme"},
    param_limits={
        "element_size": (0.5, 50.0),
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
        "strain_layer_thickness": (0.01, 5.0),
        "strain_layer_young_modulus": (1e5, 1e12),
        "strain_layer_poisson_ratio": (0.0, 0.49),
        "strain_layer_density": (10.0, 20000.0),
        "strain_layer_alpha": (0.0, 1.0),
        "young_modulus": (1e4, 1e9),
        "poisson_ratio": (0.0, 0.49),
        "density": (10.0, 20000.0),
        "target_pressure": (-50000.0, 200000.0),
        "ramp_time": (0.0, 60.0),
        "dt": (0.0001, 0.05),
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
        ("Strain-limiting layer", ["strain_limiting_layer", "strain_layer_thickness", "strain_layer_young_modulus",
                                   "strain_layer_poisson_ratio", "strain_layer_density",
                                   "strain_layer_tension_only"]),
        ("Base", ["clamp_to_trunk"]),
        ("Material", ["hyperelastic", "young_modulus", "poisson_ratio", "density", "gravity_enabled"]),
        ("Simulation", ["dt"]),
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
        {"key": "strain_layer_alpha", "label": "Strain-limiting layer opacity", "type": "opacity",
         "models": r"(^|/)cube/strain_layer_visu/visual$", "requires": "strain_limiting_layer"},
        {"key": "floor_alpha", "label": "Floor opacity", "type": "opacity", "models": r"^/floor/visual$"},
        {"key": "axes_alpha", "label": "Coordinate axes (X, Y) opacity", "type": "opacity",
         "models": r"^/(arrow|label)_[xyz]/visual$"},
        {"key": "mirror_visual_x", "label": "Mirror across X symmetry plane", "type": "mirror", "axis": "X",
         "models": r"(^|/)cube/((interior_)?visu|strain_layer_visu)/visual$", "requires": "symmetry_x"},
        {"key": "mirror_visual_y", "label": "Mirror across Y symmetry plane", "type": "mirror", "axis": "Y",
         "models": r"(^|/)cube/((interior_)?visu|strain_layer_visu)/visual$", "requires": "symmetry_y"},
        {"key": "mirror_far_end", "label": "Mirror across far-end plane", "type": "mirror",
         "planeModel": r"^/channelEndPlanePos/visual$", "default": False,
         "chain": {"countKey": "mirror_far_end_count", "max": 12, "axis": "X", "plane": 0.0, "half": "symmetry_x"},
         "models": r"(^|/)cube/((interior_)?visu|strain_layer_visu)/visual$", "requires": "channel_separation"},
        {"key": "show_symmetry_buffer", "label": "Show symmetry buffer elements (FEM)", "type": "femRegion",
         "requires": "symmetry_buffer",
         "regions": [{"axis": "X", "below": 0.0, "requires": "symmetry_x"},
                     {"axis": "Y", "below": 0.0, "requires": "symmetry_y"}]},
    ],
    presets=PRESETS,
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
        "Finger 3 plus a strain-limiting layer under the base (on by default): an inextensible sheet glued to the "
        "bottom face, as the paper in the original PneuNets. Default: paper, 0.1 mm, E = 6.5 GPa, Poisson's ratio "
        "0.2, 750 kg/m^3 (the values used in the Abaqus PneuNet models). It is a membrane: it keeps the base from "
        "stretching but lets it bend, and by default it wrinkles rather than carrying compression, as paper does. "
        "Switch it off to get Finger 3 exactly.",
        "As in Finger 3: one chamber of a PneuNet soft pneumatic finger, like Finger 2 with channel, with every dimension given in "
        "millimetres plus a standard element size. Each feature is split into the whole number of equal elements "
        "closest to that size (an 18 mm wall with 10 mm elements becomes 2 x 9 mm), and every wall into at least "
        "'minimum wall layers' elements: linear tetrahedra need more than one layer through a wall to bend "
        "realistically. The mesh is mirror-symmetric, so a model cut at a midplane is exactly half the full one.",
        "The far-end plane (the midplane to the neighbouring chamber, halfway along the duct) is a symmetry plane: "
        "its nodes stay on it but slide freely within it, while the plane itself shifts along X and tilts about Y. "
        "In Finger 2 with channel those nodes were glued to the plane, which froze the duct's cross-section there. With "
        "sliding, the channel can open up under pressure all the way to the midplane -- and with the default "
        "full-width channel the end wall above it acts as a flap. At about 100 kPa that flap has no equilibrium any more "
        "(it keeps opening until the run diverges). A narrower channel (Channel width, e.g. 16 mm with 4 mm duct "
        "walls) is closer to a real PneuNet and stays stable.",
        "Where the end wall reaches the far-end plane it touches its mirrored neighbour: that contact now pushes "
        "the plane too (before, it only held the wall back), so it can bend the finger.",
        "Options closer to a real finger: a strain-limiting layer (paper) on the bottom face instead of the clamped "
        "trunk, and neo-Hookean hyperelastic silicone for large strains. Finger bend per chamber = twice the "
        "far-end plane's tilt. The run ends once the volume has settled.",
        "Pressures are true pressures. SoftRobots' SurfacePressureConstraint imposes an impulse per time step, "
        "so a value v acts as a pressure v / dt: the other chamber scenes' '300 Pa' at dt = 0.01 s is really 30 kPa "
        "(checked against SurfacePressureForceField). This scene scales its value by dt, so 30 kPa here is 30 kPa, and "
        "the result no longer depends on the time step.",
        "Presets (top of the Parameters, Display and Lighting panels) load ready-made configurations, e.g. "
        "real PneuNet proportions with no layer, paper, or a bonded layer that also carries compression -- the "
        "case where the layer changes the bend most.",
    ],
    origin="Scene script: scenes/finger-4/original/Finger4.py (a copy of scenes/finger-3 with a strain-limiting layer).",
)
