"""Finger 5: Full finger (Finger5.py), run through sofaweb (see scenes/_base)."""

import math

import numpy as np

from sofaweb.config import SceneConfig


def _ramp(root):
    return root.getObject("PressureRamp")


def _plane_frames(root):
    """The Rigid3 frames holding the collision planes, or None when they're off."""
    finger = root.getChild("finger")
    planes = finger.getChild("collisionPlanes") if finger is not None else None
    return planes.getObject("dofs") if planes is not None else None


class _Probes:
    """Pressure, volume and the finger's bend. Each chamber's bend is the
    tilt (about Y) of its distal boundary minus that of its proximal one:
    the end faces of the finger (best-fit planes through their nodes) and
    the collision planes between the chambers (their frames; without
    collision planes, best-fit planes through the duct's cross-sections).
    Tilts are followed continuously, so a finger curled past 180 degrees
    still reads right."""

    def __init__(self):
        self.normals = {}
        self.t = None

    def _tilt(self, key, points):
        c = points - points.mean(axis=0)
        n = np.linalg.eigh(c.T @ c)[1][:, 0]
        prev = self.normals.get(key)
        if (prev is None and n[0] < 0) or (prev is not None and float(n @ prev) < 0):
            n = -n
        self.normals[key] = n
        return math.atan2(-n[2], n[0])

    def __call__(self, root, module):
        t = float(root.time.value)
        if self.t is None or t < self.t:
            self.normals = {}  # rebuilt
        self.t = t
        ramp = _ramp(root)
        volume = float(ramp.pressure.cavityVolume.value)
        volume0 = float(ramp.pressure.initialCavityVolume.value)
        values = {
            "target": float(ramp.targetPressure.value) / 1000.0,
            "applied": ramp.applied_pressure() / 1000.0,
            "volume": volume * 1e6,
            "growth": (volume / volume0 - 1.0) * 100.0 if volume0 else 0.0,
        }
        build = module.BUILD
        dofs = root.getChild("finger").getChild("cube").getObject("dofs")
        pos = np.asarray(dofs.position.value)
        tilts = [self._tilt("proximal", pos[build["proximal_face"]])]
        frames = _plane_frames(root)
        if frames is not None:
            prev = tilts[0]
            for q in frames.position.value:
                a = 2.0 * math.atan2(float(q[4]), float(q[6]))
                a += 2 * math.pi * round((prev - a) / (2 * math.pi))  # continuous
                tilts.append(a)
                prev = a
        else:
            tilts += [self._tilt(f"ring{k}", pos[ring]) for k, ring in enumerate(build["rings"])]
        tilts.append(self._tilt("distal", pos[build["distal_face"]]))
        # Unwrap the end faces' tilts too (a best-fit normal's angle wraps at 180 deg).
        for k in range(1, len(tilts)):
            tilts[k] += 2 * math.pi * round((tilts[k - 1] - tilts[k]) / (2 * math.pi))
        bends = [math.degrees(b - a) for a, b in zip(tilts, tilts[1:])]
        self.bends = bends  # every chamber's, for scripts
        values["tip_angle"] = math.degrees(tilts[-1] - tilts[0])
        values["bend_mean"] = values["tip_angle"] / len(bends)
        values["bend_first"] = bends[0]
        values["bend_middle"] = bends[len(bends) // 2]
        values["bend_last"] = bends[-1]
        rest = np.asarray(dofs.rest_position.value)[build["distal_face"]].mean(axis=0)
        values["tip_drop"] = (rest[2] - pos[build["distal_face"]].mean(axis=0)[2]) * 1000.0
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


def _from_finger4(v):
    """A Finger 4 (one chamber) preset as Finger 5 parameters (see
    SceneConfig.import_presets). The chamber geometry, channel, layer,
    material and pressure have the same names and meaning and are copied."""
    clamp_end = bool(v.get("clamp_proximal_end"))
    sym_x = bool(v.get("symmetry_x", True)) and not clamp_end
    trunk = bool(v.get("clamp_to_trunk"))
    contact = bool(v.get("prevent_channel_end_crossing", True))
    params = {"collision_planes": contact}
    notes = [
        ("estimated", "collision_planes",
         f"{'on' if contact else 'off'}, from 'End wall touches the far-end plane': in Finger 4 the far-end plane "
         "stands for the neighbouring chamber's wall."),
        ("assumed", "segments",
         "7 (this scene's default): Finger 4 is one chamber. Compare " + (
             "Finger 4's clamped first chamber with the first chamber here." if clamp_end else
             "Finger 4 (a chamber of an endless row) with the middle chamber here; use more chambers to get "
             "closer to an endless row.")),
        ("assumed", "plane_hold_stiffness",
         "1e4 N/m (this scene's default): Finger 4's far-end plane holds the duct's nodes stiffly; here the bend "
         "hardly depends on it ('Flat duct sections' preset: 1e9 N/m)."),
    ]
    if clamp_end:
        params.update(clamp_proximal_end=True, clamp_to_trunk=trunk)
    elif trunk:
        params.update(clamp_proximal_end=False, clamp_to_trunk=True)
        notes.append(("estimated", "clamp_to_trunk",
                      "on: Finger 4 clamps its chamber's base on the trunk and, by X symmetry, every chamber's; "
                      "here only the first chamber stands on the trunk (the rest of the finger is free)."))
        notes.append(("assumed", "clamp_proximal_end", "off: Finger 4's chamber is held by the trunk alone."))
    else:
        params.update(clamp_proximal_end=True, clamp_to_trunk=False)
        notes.append(("assumed", "clamp_proximal_end",
                      "on: Finger 4's base is free (held by its symmetry planes only); a whole finger needs a mount."))
    notes.append(("not imported", "symmetry_x",
                  f"{'on' if sym_x else 'off'}: there is no X symmetry plane here; the whole length is modelled."))
    if float(v.get("symmetry_buffer", 0.0)) > 0:
        notes.append(("not imported", "symmetry_buffer",
                      f"{v['symmetry_buffer']:g} mm: no symmetry buffer here (no X symmetry plane)."))
    if not v.get("channel_end_constraint", True):
        notes.append(("not imported", "channel_end_constraint",
                      "off: the planes between chambers here always shift and tilt with the finger."))
    if not v.get("channel_end_sliding", True):
        notes.append(("not imported", "channel_end_sliding",
                      "off (glued far end): here the duct is one piece between two chambers."))
    return {"params": params, "notes": notes,
            "handled": ["clamp_proximal_end", "clamp_to_trunk", "symmetry_x", "symmetry_buffer",
                        "channel_end_constraint", "channel_end_sliding", "prevent_channel_end_crossing"]}


def _gauge(root, module):
    """The pressure gauge's layout (see SceneConfig.gauge): a manometer above and beside the first chamber, on a
    4 mm tube (at the human-scale size; in proportion to the chamber's height otherwise). The tube drops straight
    down from the gauge beside the mount, bends, and runs horizontally at the height of the cavity to the first
    chamber's proximal end face (through the mount, which hides it there), so it stays clear of the FEM model.
    The scale runs from 0 to the first of 10, 20, 25, 50, 100, ... kPa that is at least 1.25 times the target
    pressure."""
    P = module.PARAMS
    g = module.finger5_mesh.Geometry(P)
    mm = 1e-3
    base = P["base_height"] * mm  # the height of the model's base above the floor (z = 0)
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


# The human-scale index finger: the default, also kept as a preset (every parameter
# that matters listed) so it can be restored if the default changes. See the About
# text for where the dimensions come from.
_HUMAN = {
    "element_size": 2.0, "min_wall_layers": 1,
    "length": 8.0, "width": 20.0, "height": 16.0,
    "wall_end": 1.5, "wall_side": 1.5, "wall_bottom": 3.0, "wall_top": 1.5,
    "channel_height": 2.0, "channel_width": 2.0, "channel_wall": 1.5, "channel_separation": 2.0,
    "trunk_length": 8.0, "trunk_width": 20.0, "base_height": 60.0,
    "strain_limiting_layer": True, "strain_layer_thickness": 0.1, "strain_layer_young_modulus": 6.5e9,
    "strain_layer_poisson_ratio": 0.2, "strain_layer_density": 750.0, "strain_layer_tension_only": True,
    "hyperelastic": False, "young_modulus": 1.0e6, "poisson_ratio": 0.45, "density": 1000.0,
    "gravity_enabled": False, "target_pressure": 50000.0, "ramp_start": 0.0, "ramp_time": 0.5, "dt": 0.01,
    "contact_outer_face_only": True,
    "floor_size": 120.0, "segments": 7, "clamp_proximal_end": True, "clamp_to_trunk": False,
    "symmetry_y": True, "prevent_symmetry_crossing": True, "collision_planes": True,
    "plane_hold_stiffness": 1.0e4,
}

# The large block chambers that were the default before the human-scale finger
# (a demonstration geometry: 8 mm elements, 32 x 160 x 96 mm chambers, 3 of them, 30 kPa).
_BLOCK = {
    "element_size": 8.0, "length": 32.0, "width": 160.0, "height": 96.0,
    "wall_end": 8.0, "wall_side": 8.0, "wall_bottom": 8.0, "wall_top": 8.0,
    "channel_height": 8.0, "channel_width": 0.0, "channel_wall": 8.0, "channel_separation": 8.0,
    "trunk_length": 16.0, "trunk_width": 160.0, "floor_size": 400.0, "target_pressure": 30000.0,
    "ramp_time": 0.2, "segments": 3, "base_height": 150.0,
}

PRESETS = {
    "params": [
        {"name": "Human index finger",
         "description": "A PneuNet the size of a human index finger (the default at the time of writing): 7 "
                        "chambers of 8 mm at a 10 mm pitch (68 mm), 20 x 16 mm section, 1.5 mm side and roof "
                        "walls, 3 mm base with the paper layer, 2 x 2 mm channel, 2 mm elements, single-wall "
                        "chambers, clamped at its proximal end, 50 kPa. The tip bends 83 deg.",
         "values": dict(_HUMAN)},
        {"name": "No collision planes",
         "description": "Without the planes between the chambers: neighbouring end walls pass through each "
                        "other instead of pushing on each other.",
         "values": {"collision_planes": False}},
        {"name": "Flat duct sections (stiff planes)",
         "description": "The duct's nodes held on their collision plane by stiff springs (1e9 N/m per node), so "
                        "the duct's cross-section between two chambers stays flat, as on Finger 4's symmetry "
                        "plane.",
         "values": {"plane_hold_stiffness": 1.0e9}},
        {"name": "Mounted on a trunk",
         "description": "The first chamber's base clamped to a rigid trunk (as long as a chamber, centred under "
                        "it), as in Finger 4, instead of the clamped end face.",
         "values": {"clamp_proximal_end": False, "clamp_to_trunk": True}},
        {"name": "Two chambers",
         "description": "The shortest finger with a collision plane: about 50 ms per time step, settling after "
                        "1 s of simulated time (a few seconds).",
         "values": {"segments": 2}},
        {"name": "Four chambers",
         "description": "A shorter finger: about 80 ms per time step, settling after 1.6 s of simulated time.",
         "values": {"segments": 4}},
        {"name": "Large block chambers (previous default)",
         "description": "The demonstration geometry used before the human-scale finger: three 32 x 160 x 96 mm "
                        "chambers with 8 mm walls and 8 mm elements, 30 kPa. About 0.12 s per time step, "
                        "settling after 1.4 s; the chambers bend 26-37 deg each.",
         "values": dict(_BLOCK)},
        {"name": "Two element layers per wall",
         "description": "Every wall at least 2 elements thick, not 1 (the default): bends more, as a real chamber "
                        "does (linear tetrahedra are stiff when a wall is a single layer), but has about "
                        "three times as many elements and takes three to four times as long.",
         "values": {"min_wall_layers": 2}},
        {"name": "Bonded layer that carries compression",
         "description": "The strain-limiting layer as a linear membrane that resists compression as well as "
                        "tension (a stiff sheet fully bonded into the silicone).",
         "values": {"strain_layer_tension_only": False}},
    ],
    "display": [
        {"name": "Half model",
         "description": "Only the modelled half (y >= 0), without its mirror image: look into the chambers "
                        "through the cut.",
         "values": {"mirror_visual_y": False}},
        {"name": "See-through walls",
         "description": "Translucent outer walls, showing the cavity and the strain-limiting layer.",
         "values": {"wall_alpha": 0.3, "wall_alpha:on": True, "interior_alpha": 1.0, "interior_alpha:on": True}},
        {"name": "No collision planes shown",
         "description": "The finger without the magenta collision planes.",
         "values": {"channel_end_alpha:on": False}},
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
    title="Finger 5: Full finger",
    script="original/Finger5.py",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"FINGER5_NO_EDITOR": "1"},
    # GL-only debug drawing, and the desktop GUI's per-step mirroring (the
    # browser mirrors instead, see `display`).
    forced_params={"show_pressure_overlay": False, "show_fem_elements": False, "mirror_visual_y": False},
    hidden_params={"theme"},
    param_limits={
        "segments": (1, 8),
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
        "floor_size": (20.0, 4000.0),
        "plane_hold_stiffness": (0.0, 1e10),
        "base_height": (0.0, 2000.0),
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
        ("Finger", ["segments"]),
        ("Chamber geometry (mm)", ["length", "width", "height", "wall_end", "wall_side", "wall_bottom", "wall_top",
                                   "floor_size"]),
        ("Channel (mm)", ["channel_height", "channel_width", "channel_wall", "channel_separation"]),
        ("Collision planes (between neighbouring chambers)", ["collision_planes", "contact_outer_face_only",
                                                              "plane_hold_stiffness"]),
        ("Mount", ["base_height", "clamp_proximal_end", "clamp_to_trunk", "trunk_length", "trunk_width"]),
        ("Strain-limiting layer", ["strain_limiting_layer", "strain_layer_thickness", "strain_layer_young_modulus",
                                   "strain_layer_poisson_ratio", "strain_layer_density",
                                   "strain_layer_tension_only"]),
        ("Material", ["hyperelastic", "young_modulus", "poisson_ratio", "density", "gravity_enabled"]),
        ("Simulation", ["dt"]),
        ("Symmetry", ["symmetry_y", "prevent_symmetry_crossing"]),
        ("Cutaway (rebuilds the skin)", ["cutaway", "interior_cutaway"]),
    ],
    display=[
        {"key": "wall_alpha", "label": "Wall opacity", "type": "opacity", "models": r"(^|/)cube/visu/visual$"},
        {"key": "interior_alpha", "label": "Interior opacity", "type": "opacity",
         "models": r"(^|/)cube/interior_visu/visual$"},
        {"key": "channel_end_alpha", "label": "Collision plane opacity", "type": "opacity",
         "models": r"^/collisionPlane", "requires": "collision_planes"},
        {"key": "strain_layer_alpha", "label": "Strain-limiting layer opacity", "type": "opacity",
         "models": r"(^|/)cube/strain_layer_visu/visual$", "requires": "strain_limiting_layer"},
        {"key": "floor_alpha", "label": "Floor opacity", "type": "opacity", "models": r"^/floor/visual$"},
        {"key": "mount_alpha", "label": "Mount opacity", "type": "opacity", "models": r"^/trunk/visual$"},
        {"key": "axes_alpha", "label": "Coordinate axes (X, Y) opacity", "type": "opacity",
         "models": r"^/(arrow|label)_[xyz]/visual$"},
        {"key": "mirror_visual_y", "label": "Mirror across Y symmetry plane", "type": "mirror", "axis": "Y",
         "models": r"(^|/)cube/((interior_)?visu|strain_layer_visu)/visual$", "requires": "symmetry_y"},
    ],
    presets=PRESETS,
    import_presets={"finger-4": {"title": "Finger 4 with strain-limiting layer", "convert": _from_finger4}},
    live_params=LIVE,
    probes=_Probes(),
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
        {"title": "Cavity volume growth", "unit": "%", "series": [{"key": "growth", "label": "growth", "color": "#2a78d6"}]},
        {"title": "Finger bend (tip angle)", "unit": "deg",
         "series": [{"key": "tip_angle", "label": "tip", "color": "#c77dff"}]},
        {
            "title": "Bend per chamber",
            "unit": "deg",
            "series": [
                {"key": "bend_first", "label": "first", "color": "#1baf7a"},
                {"key": "bend_middle", "label": "middle", "color": "#2a78d6"},
                {"key": "bend_last", "label": "last", "color": "#e8933c"},
            ],
        },
    ],
    readouts=[
        {"key": "applied", "label": "pressure", "digits": 3, "unit": "kPa"},
        {"key": "volume", "label": "cavity volume", "digits": 2, "unit": "cm³"},
        {"key": "growth", "label": "volume growth", "digits": 1, "unit": "%"},
        {"key": "tip_angle", "label": "finger bend (tip angle)", "digits": 1, "unit": "deg"},
        {"key": "bend_mean", "label": "mean bend per chamber", "digits": 1, "unit": "deg"},
        {"key": "bend_first", "label": "first chamber bend", "digits": 1, "unit": "deg"},
        {"key": "bend_middle", "label": "middle chamber bend", "digits": 1, "unit": "deg"},
        {"key": "bend_last", "label": "last chamber bend", "digits": 1, "unit": "deg"},
        {"key": "tip_drop", "label": "tip drop", "digits": 1, "unit": "mm"},
    ],
    stop_at=stop_time,
    extend_by=extend_time,
    end_condition=_Settled(),
    about=[
        "The whole PneuNet finger: Finger 4's chamber repeated along X (3 chambers by default), joined by the "
        "connecting channel and inflated together. Only the Y midplane is a symmetry plane (the browser mirrors "
        "the modelled half). Every chamber is meshed exactly alike, but each can deform in its own way: the "
        "first is held by the mount (its outer end face clamped), the last ends free.",
        "Between each pair of neighbouring chambers sits Finger 4's far-end plane. It is no longer a symmetry "
        "plane but a cheap collision model: both chambers' end walls stay on their own side of it, so where they "
        "bulge into each other they push on the plane and, through it, on each other -- without a contact "
        "search. The plane is held by the duct's cross-section halfway between the chambers (soft springs per "
        "node, normal to the plane), and may shift and tilt in the finger's plane.",
        "As in Finger 4: dimensions in millimetres with a standard element size, a paper strain-limiting layer "
        "under the whole base (tension-only by default), linear corotational or neo-Hookean silicone. The duct "
        "joining two chambers always spans the full width; 'Channel duct roof thickness' is the material above "
        "the channel.",
        "Finger bend = the tilt of the distal end face; each chamber's bend is the tilt between its two "
        "boundaries (end faces or collision planes). The finger curls downwards, so the floor is set below "
        "anything it can reach. The run ends once the volume has settled.",
        "The default is a human-scale index finger: 7 chambers of 8 mm at a 10 mm pitch (68 mm long; a human index "
        "finger is about 67-72 mm from tip to knuckle crease, GTRI hand anthropometry table, 50th percentile "
        "women / men), 20 x 16 mm in section (about 20-23 mm broad, firefighter-glove survey; the depth is an "
        "estimate), 1.5 mm side and roof walls, a 3 mm base with the paper layer, a 2 x 2 mm channel and 2 mm "
        "elements, at 50 kPa, held at its proximal end like a finger on the palm. The chamber proportions follow "
        "the Soft Robotics Toolkit's reference PneuNet (8 mm chambers 15 x 15 mm, 1-2 mm walls, 2 mm gaps, "
        "2 x 2 mm channel). At 50 kPa the tip bends 83 deg. The previous default is the preset 'Large block "
        "chambers'.",
        "Compared with Finger 4, which models one chamber of an endless row (an X symmetry plane through the "
        "chamber's centre): without that plane a chamber can also lean, i.e. rotate as a whole, when a "
        "neighbour presses on it, and the finger's ends act on every chamber. On the default finger the middle "
        "chambers bend as in Finger 4 (13.1 deg per chamber against 12.7 deg) and the first and last less "
        "(8.6 and 9.2 deg: they have one neighbour; Finger 4's clamped first chamber bends 8.4 deg).",
        "Simple models, for indication and speed: the default walls are one element thick (so the chambers bend "
        "less than with two layers; see the preset 'Two element layers per wall'), and only the end walls' outer "
        "faces are checked for contact. The default finger has 8,100 elements: about 0.14 s per time step on this "
        "server, and the run settles after 2 s of simulated time (about 30 s). 2 chambers take 0.05 s per step "
        "(1 s), 4 chambers 0.08 s (1.6 s). With two layers per wall it takes three to four times as long.",
        "Import (next to the Parameters preset) loads a Finger 4 preset converted to this scene, to compare the "
        "two models (Finger 4 can import Finger 5's presets too, and can model the clamped first chamber).",
        "'Ramp start' is when the pressure starts to rise (0 before it): with gravity on, the finger can first sag "
        "and settle, and the pressure then acts on that state. The run ends once the volume has settled after the "
        "pressure has reached its target. Editable while the simulation runs.",
        "Pressures are true pressures (SoftRobots' SurfacePressureConstraint value is scaled by dt, see Finger 4).",
    ],
    origin="Scene script: scenes/finger-5/original/Finger5.py (a copy of scenes/finger-4, extended to the full finger).",
)
