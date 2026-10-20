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
        params = getattr(module, "PARAMS", {})
        clamp_end = bool(params.get("clamp_proximal_end"))
        if clamp_end:
            # The first chamber: from its clamped end face (no tilt) to the
            # far-end plane.
            values["bend"] = angle
        elif params.get("symmetry_x", True):
            # The chamber's midplane (x = 0) stays put and the far-end plane
            # tilts by `angle`: one chamber pitch of the finger bends by twice that.
            values["bend"] = 2.0 * angle
        elif len(frame.position.value) > 1:
            # The whole chamber, between its two far-end planes.
            q = frame.position.value[1]
            values["bend"] = angle - math.degrees(2.0 * math.atan2(float(q[4]), float(q[6])))
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


def _from_finger5(v):
    """A Finger 5 (full finger) preset as Finger 4 parameters (see
    SceneConfig.import_presets). The chamber geometry, channel, layer,
    material and pressure have the same names and meaning and are copied."""
    trunk = bool(v.get("clamp_to_trunk"))
    clamp_end = bool(v.get("clamp_proximal_end")) or not trunk  # Finger 5 clamps the end when neither is on
    planes = bool(v.get("collision_planes", True))
    n = int(v.get("segments", 3))
    params = {
        "clamp_proximal_end": clamp_end,
        "symmetry_x": not clamp_end,
        "clamp_to_trunk": trunk,
        "prevent_channel_end_crossing": planes,
        "channel_end_constraint": True,
        "channel_end_sliding": True,
        "symmetry_buffer": 0.0,
    }
    if clamp_end:
        which = ("Finger 5's first chamber (its proximal end face clamped); the neighbour beyond the far-end "
                 "plane is taken as its mirror image")
    else:
        which = ("a chamber in the middle of an endless row (X symmetry): compare it with Finger 5's middle "
                 "chamber")
    notes = [
        ("not imported", "segments", f"{n}: Finger 4 models one chamber, here {which}."),
        ("not imported", "plane_hold_stiffness",
         f"{v.get('plane_hold_stiffness', 1e4):g} N/m: Finger 4's far-end plane is a symmetry plane, the duct's "
         "nodes held on it stiffly while sliding within it (in Finger 5 the bend hardly depends on this "
         "stiffness)."),
        ("estimated", "prevent_channel_end_crossing",
         f"{'on' if planes else 'off'}, from Finger 5's collision planes: the end wall touches the far-end plane, "
         "which stands for the neighbouring chamber's wall."),
        ("assumed", "clamp_proximal_end",
         f"{'on' if clamp_end else 'off'}: " + ("Finger 5's mount clamps its proximal end face."
                                                 if clamp_end else "Finger 5 is held by the trunk alone.")),
        ("assumed", "symmetry_x", f"{'off' if clamp_end else 'on'}: " + (
            "the first chamber isn't symmetric about its midplane." if clamp_end else
            "the chamber as one of an endless row.")),
        ("assumed", "channel_end_constraint", "on: the far-end plane shifts along X and tilts about Y, as "
                                              "Finger 5's collision planes."),
        ("assumed", "channel_end_sliding", "on: the duct's nodes slide within the far-end plane."),
        ("assumed", "symmetry_buffer", "0 mm: no material past the symmetry planes."),
    ]
    if trunk:
        notes.append(("estimated", "clamp_to_trunk", "on: in Finger 5 the trunk is under the first chamber only; "
                                                     "here under this chamber" + (
                                                         "." if clamp_end else ", and by the X symmetry under "
                                                         "every chamber of the row.")))
    return {"params": params, "notes": notes,
            "handled": ["segments", "plane_hold_stiffness", "collision_planes", "clamp_proximal_end",
                        "clamp_to_trunk"]}


def _gauge(root, module):
    """The pressure gauge's layout (see SceneConfig.gauge): a manometer above and beside the first chamber, on a
    4 mm tube (at the human-scale size; in proportion to the chamber's height otherwise). The tube drops straight
    down from the gauge beside the mount, bends, and runs horizontally at the height of the cavity to the first
    chamber's proximal end face (through the mount, which hides it there), so it stays clear of the FEM model.
    The scale runs from 0 to the first of 10, 20, 25, 50, 100, ... kPa that is at least 1.25 times the target
    pressure."""
    P = module.PARAMS
    g = module.finger4_mesh.Geometry(P)
    mm = 1e-3
    base = max(4.0, min(g.h, 10.0)) * mm  # the height of the model's base above the world's z = 0
    H, L = g.H * mm, g.L * mm
    R = 0.6 * H  # the dial's radius
    tube_r = 0.125 * H
    roof = base + H
    cavity_z = base + 0.5 * (g.wb + g.H - g.wt) * mm  # the cavity's mid height
    target = abs(P["target_pressure"]) / 1000.0
    top = next((v for v in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000) if v >= 1.25 * target), 2000)
    cx = -L / 2 - max(0.25 * H, 4 * mm) - 1.4 * R  # beside the mount, if there is one
    cz = roof + 2.2 * R
    casing = cz - 1.12 * R
    bend = 3.0 * tube_r  # the radius of the tube's bend
    x_end = -L / 2  # the proximal end face
    tube = [[cx, 0.0, casing], [cx, 0.0, 0.5 * (casing + cavity_z + bend)], [cx, 0.0, cavity_z + bend],
            [cx + bend * (1 - math.cos(math.pi / 4)), 0.0, cavity_z + bend * (1 - math.sin(math.pi / 4))],
            [cx + bend, 0.0, cavity_z], [0.5 * (cx + bend + x_end), 0.0, cavity_z], [x_end, 0.0, cavity_z]]
    return {
        "key": "applied", "min": 0.0, "max": float(top), "unit": "kPa", "major": top / 5.0, "minor": top / 20.0,
        "center": [cx, 0.0, cz], "radius": R, "right": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0],
        "tube": tube, "tubeRadius": tube_r,
    }

LIVE = {
    "target_pressure": _set("targetPressure", float),
    "ramp_start": _set("rampStart", float),
    "ramp_time": _set("rampTime", float),
    "instant": _set("instant", bool),
}


def stop_time(p):
    # A safety limit; runs normally end earlier, once settled.
    return p["ramp_start"] + p["ramp_time"] + 15.0


def extend_time(p):
    return p["ramp_start"] + p["ramp_time"] + 10.0


# PneuNet-like proportions (Mosadegh et al. 2014 scale): an 8 mm chamber
# pitch, 16 x 16 mm cross-section, 1.5 mm walls, a 4 mm base and a 2 x 2 mm
# connecting channel in a full-width duct with a 1.5 mm roof, on a 1 mm mesh
# with a free base at 50 kPa.
_PNEUNET = {
    "element_size": 1.0, "length": 8.0, "width": 16.0, "height": 16.0,
    "wall_end": 1.5, "wall_side": 1.5, "wall_bottom": 4.0, "wall_top": 1.5,
    "channel_height": 2.0, "channel_width": 2.0, "channel_wall": 1.5, "channel_separation": 1.5,
    "floor_size": 60.0, "clamp_to_trunk": False, "target_pressure": 50000.0,
}

# The human-scale index finger chamber: the default, also kept as a preset (every
# parameter that matters listed) so it can be restored if the default changes. See
# the About text for where the dimensions come from.
_HUMAN = {
    "element_size": 2.0, "min_wall_layers": 1,
    "length": 8.0, "width": 20.0, "height": 16.0,
    "wall_end": 1.5, "wall_side": 1.5, "wall_bottom": 3.0, "wall_top": 1.5,
    "channel_height": 2.0, "channel_width": 2.0, "channel_wall": 1.5, "channel_separation": 2.0,
    "trunk_length": 8.0, "trunk_width": 20.0,
    "strain_limiting_layer": True, "strain_layer_thickness": 0.1, "strain_layer_young_modulus": 6.5e9,
    "strain_layer_poisson_ratio": 0.2, "strain_layer_density": 750.0, "strain_layer_tension_only": True,
    "hyperelastic": False, "young_modulus": 1.0e6, "poisson_ratio": 0.45, "density": 1000.0,
    "gravity_enabled": False, "target_pressure": 50000.0, "ramp_start": 0.0, "ramp_time": 0.5, "dt": 0.01,
    "contact_outer_face_only": True,
    "floor_size": 60.0, "clamp_to_trunk": False, "clamp_proximal_end": False, "symmetry_x": True,
    "symmetry_y": True, "prevent_channel_end_crossing": True,
}

# The large block chamber that was the default before the human-scale index finger
# (a demonstration geometry: 8 mm elements, 96 x 160 mm section, 30 kPa, on a trunk).
_BLOCK = {
    "element_size": 8.0, "length": 32.0, "width": 160.0, "height": 96.0,
    "wall_end": 8.0, "wall_side": 8.0, "wall_bottom": 8.0, "wall_top": 8.0,
    "channel_height": 8.0, "channel_width": 0.0, "channel_wall": 8.0, "channel_separation": 8.0,
    "trunk_length": 16.0, "trunk_width": 160.0, "floor_size": 400.0, "target_pressure": 30000.0,
    "ramp_time": 0.2, "clamp_to_trunk": True,
}

PRESETS = {
    "params": [
        {"name": "Human index finger",
         "description": "One chamber of a PneuNet the size of a human index finger (the default at the time of "
                        "writing): 8 mm long, 20 x 16 mm section, 1.5 mm side and roof walls, 3 mm base with the "
                        "paper layer, 2 x 2 mm channel, 2 mm elements, single-wall chambers, 50 kPa. Bends 12.7 deg "
                        "per chamber.",
         "values": dict(_HUMAN)},
        {"name": "No strain-limiting layer (as Finger 3)",
         "description": "On the large block chamber: Finger 3's model, the base without paper. Bends 33.2 deg "
                        "per chamber at 30 kPa, against 33.3 deg with the paper layer.",
         "values": {**_BLOCK, "strain_limiting_layer": False}},
        {"name": "Bonded layer that carries compression",
         "description": "On the large block chamber: the layer as a linear membrane that resists compression as "
                        "well as tension (a stiff sheet fully bonded into the silicone). Bends 29.9 deg per "
                        "chamber: 10 % less than without a layer.",
         "values": {**_BLOCK, "strain_layer_tension_only": False}},
        {"name": "Thin 2 mm base, free (paper)",
         "description": "On the large block chamber: a 2 mm base without the clamped trunk. Paper here increases the bend: 51.1 deg per "
                        "chamber, against 40.4 deg without the layer, because it stops the thin base widening.",
         "values": {**_BLOCK, "wall_bottom": 2.0, "clamp_to_trunk": False}},
        {"name": "Large block chamber (previous default)",
         "description": "The demonstration geometry used before the human-scale finger: a 32 x 160 x 96 mm chamber "
                        "with 8 mm walls and 8 mm elements, on a trunk, 30 kPa. Bends 33.2 deg at 30 kPa "
                        "(46.8 deg with two element layers per wall).",
         "values": dict(_BLOCK)},
        {"name": "Two element layers per wall",
         "description": "Every wall at least 2 elements thick, not 1 (the default): bends more, as a real chamber "
                        "does (linear tetrahedra are stiff when a wall is a single layer), but has about "
                        "three times as many elements and takes three to four times as long.",
         "values": {"min_wall_layers": 2}},
        {"name": "PneuNet proportions, no layer",
         "description": "A real PneuNet's proportions: 8 mm chamber pitch, 16 x 16 mm section, 1.5 mm walls, "
                        "4 mm base, 2 x 2 mm channel in a full-width duct with a 1.5 mm roof, 1 mm elements, "
                        "free base, 50 kPa. No strain-limiting layer: 10.7 deg per chamber.",
         "values": {**_PNEUNET, "strain_limiting_layer": False}},
        {"name": "PneuNet proportions, paper layer",
         "description": "As above with 0.1 mm paper under the base: 10.5 deg per chamber, almost no change.",
         "values": {**_PNEUNET, "strain_limiting_layer": True, "strain_layer_tension_only": True}},
        {"name": "PneuNet proportions, bonded layer",
         "description": "As above with a layer that also carries compression: 7.7 deg per chamber, 29 % "
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
        "ramp_start": (0.0, 60.0),
        "ramp_time": (0.0, 60.0),
        "dt": (0.0001, 0.05),
        "wall_alpha": (0.0, 1.0),
        "interior_alpha": (0.0, 1.0),
        "channel_end_alpha": (0.0, 1.0),
    },
    sections=[
        ("Pressure", ["target_pressure", "ramp_start", "ramp_time", "instant"]),
        ("Mesh", ["element_size", "min_wall_layers"]),
        ("Geometry (mm)", ["length", "width", "height", "wall_end", "wall_side", "wall_bottom", "wall_top",
                           "trunk_length", "trunk_width", "floor_size"]),
        ("Channel (mm)", ["channel_height", "channel_width", "channel_wall", "channel_separation"]),
        ("Far-end plane (midplane to the next chamber)", ["channel_end_constraint", "channel_end_sliding",
                                                          "prevent_channel_end_crossing", "contact_outer_face_only"]),
        ("Strain-limiting layer", ["strain_limiting_layer", "strain_layer_thickness", "strain_layer_young_modulus",
                                   "strain_layer_poisson_ratio", "strain_layer_density",
                                   "strain_layer_tension_only"]),
        ("Base", ["clamp_to_trunk", "clamp_proximal_end"]),
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
    import_presets={"finger-5": {"title": "Finger 5: Full finger", "convert": _from_finger5}},
    exclusive=[("clamp_proximal_end", "symmetry_x")],
    live_params=LIVE,
    probes=probes,
    gauge=_gauge,
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
        "The default is one chamber of a PneuNet the size of a human index finger: about 70 mm long in total "
        "(7 chambers of 8 mm at a 10 mm pitch, as in Finger 5), 20 x 16 mm in section, 1.5 mm side and roof walls, "
        "a 3 mm base with the paper layer under it, a 2 x 2 mm channel, 2 mm elements and 50 kPa. A human index "
        "finger is about 67-72 mm from tip to knuckle crease (GTRI hand anthropometry table, 50th percentile "
        "women / men) and 20-23 mm broad (firefighter-glove survey, mean), and the chamber proportions follow the "
        "Soft Robotics Toolkit's reference PneuNet (8 mm chambers 15 x 15 mm, 1-2 mm walls, 2 mm gaps, "
        "2 x 2 mm channel). The silicone is 1 MPa, as before; the finger depth of 16 mm is an estimate. The "
        "previous default, a 32 x 160 x 96 mm block chamber, is the preset 'Large block chamber'.",
        "Finger 3 plus a strain-limiting layer under the base (on by default): an inextensible sheet glued to the "
        "bottom face, as the paper in the original PneuNets. Default: paper, 0.1 mm, E = 6.5 GPa, Poisson's ratio "
        "0.2, 750 kg/m^3 (the values used in the Abaqus PneuNet models). It is a membrane: it keeps the base from "
        "stretching but lets it bend, and by default it wrinkles rather than carrying compression, as paper does. "
        "Switch it off to get Finger 3 exactly.",
        "As in Finger 3: one chamber of a PneuNet soft pneumatic finger, like Finger 2 with channel, with every dimension given in "
        "millimetres plus a standard element size. Each feature is split into the whole number of equal elements "
        "closest to that size (an 18 mm wall with 10 mm elements becomes 2 x 9 mm), and every wall into at least "
        "'minimum wall layers' elements: linear tetrahedra need more than one layer through a wall to bend "
        "realistically. For speed the default is a single layer (the chamber bends about 30 % less and "
        "runs three to four times faster; the preset 'Two element layers per wall' restores two). "
        "The mesh is mirror-symmetric, so a model cut at a midplane is exactly half the full one.",
        "The far-end plane (the midplane to the neighbouring chamber, halfway along the duct) is a symmetry plane: "
        "its nodes stay on it but slide freely within it, while the plane itself shifts along X and tilts about Y. "
        "In Finger 2 with channel those nodes were glued to the plane, which froze the duct's cross-section there. With "
        "sliding, the channel can open up under pressure all the way to the midplane -- and with the default "
        "full-width channel the end wall above it acts as a flap. At about 100 kPa that flap has no equilibrium any more "
        "(it keeps opening until the run diverges). A narrower channel (Channel width, e.g. 16 mm with a 4 mm duct "
        "roof: 62 deg per chamber at 100 kPa) is closer to a real PneuNet and stays stable. The duct joining "
        "the chambers always spans the full width; 'Channel duct roof thickness' is the material above the "
        "channel.",
        "Where the end wall reaches the far-end plane it touches its mirrored neighbour: that contact now pushes "
        "the plane too (before, it only held the wall back), so it can bend the finger.",
        "Options closer to a real finger: a strain-limiting layer (paper) on the bottom face instead of the clamped "
        "trunk, and neo-Hookean hyperelastic silicone for large strains. Finger bend per chamber = twice the "
        "far-end plane's tilt. The run ends once the volume has settled.",
        "'Ramp start' is when the pressure starts to rise (0 before it): with gravity on, the chamber can first sag "
        "and settle, and the pressure then acts on that state. Editable while the simulation runs.",
        "Pressures are true pressures. SoftRobots' SurfacePressureConstraint imposes an impulse per time step, "
        "so a value v acts as a pressure v / dt: the other chamber scenes' '300 Pa' at dt = 0.01 s is really 30 kPa "
        "(checked against SurfacePressureForceField). This scene scales its value by dt, so 30 kPa here is 30 kPa, and "
        "the result no longer depends on the time step.",
        "'Clamp the proximal end face' makes this chamber the first of a finger, as in Finger 5: its -X end wall "
        "is closed and clamped in a fixture, so the chamber can lean and X symmetry is off (the two exclude each "
        "other). The bend per chamber is then the far-end plane's tilt itself. Import (next to the Parameters "
        "preset) loads a Finger 5 preset converted to this scene, to compare the two models.",
        "Presets (top of the Parameters, Display and Lighting panels) load ready-made configurations, e.g. "
        "real PneuNet proportions with no layer, paper, or a bonded layer that also carries compression -- the "
        "case where the layer changes the bend most.",
    ],
    origin="Scene script: scenes/finger-4/original/Finger4.py (a copy of scenes/finger-3 with a strain-limiting layer).",
)
