"""Gripper 1: two hanging PneuNet fingers on a rod (Gripper1.py), run through sofaweb (see scenes/_base)."""

import math

import numpy as np

from sofaweb.config import SceneConfig

FINGERS = (1, 2)  # the finger on the +x side (the right one in the default view), and the other


def _ramps(root):
    return [root.getObject(f"PressureRamp{k}") for k in FINGERS]


def _finger_dofs(root, k):
    return root.getChild(f"finger{k}").getChild("cube").getObject("dofs")


def _plane_frames(root, k):
    """The Rigid3 frames holding finger k's collision planes, or None when they're off."""
    planes = root.getChild(f"finger{k}").getChild("collisionPlanes")
    return planes.getObject("dofs") if planes is not None else None


class _Probes:
    """Per finger: pressure, volume growth, bend (the tilt of the distal end face against the clamped
    proximal one, in the finger's own frame, so both fingers read alike: positive curls towards the
    strain-limiting layer, i.e. towards the other finger) and the contact with the rod."""

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
        build = module.BUILD
        params = module.PARAMS
        rod = build.get("rod")
        sym = 2.0 if params.get("symmetry_y") else 1.0  # the model is the +y half: the finger's is twice that
        values = {}
        for k, ramp in zip(FINGERS, _ramps(root)):
            f = build["fingers"][k - 1]
            volume = float(ramp.pressure.cavityVolume.value)
            volume0 = float(ramp.pressure.initialCavityVolume.value)
            values[f"target_{k}"] = float(ramp.targetPressure.value) / 1000.0
            values[f"applied_{k}"] = ramp.applied_pressure() / 1000.0
            values[f"growth_{k}"] = (volume / volume0 - 1.0) * 100.0 if volume0 else 0.0
            world = np.asarray(_finger_dofs(root, k).position.value)
            # Back into the finger's own frame (see Gripper1.FingerPose).
            local = np.stack([f["tz"] - world[:, 2], world[:, 1], f["side"] * world[:, 0] - f["gap"] / 2], axis=1)
            tilts = [self._tilt(f"prox{k}", local[f["proximal_face"]])]
            frames = _plane_frames(root, k)
            if frames is not None:
                rest = 2.0 * math.atan2(module.Q0[1], module.Q0[3])
                prev = tilts[0]
                for q in frames.position.value:
                    a = f["side"] * (2.0 * math.atan2(float(q[4]), float(q[6])) - rest)  # about y, in the finger's frame
                    a += 2 * math.pi * round((prev - a) / (2 * math.pi))
                    tilts.append(a)
                    prev = a
            else:
                tilts += [self._tilt(f"ring{k}_{j}", local[ring]) for j, ring in enumerate(f["rings"])]
            tilts.append(self._tilt(f"distal{k}", local[f["distal_face"]]))
            for j in range(1, len(tilts)):
                tilts[j] += 2 * math.pi * round((tilts[j - 1] - tilts[j]) / (2 * math.pi))
            values[f"tip_angle_{k}"] = math.degrees(tilts[-1] - tilts[0])
            if rod is not None and f["rod_nodes"]:
                nodes = world[f["rod_nodes"]]
                penetration = rod[2] - np.hypot(nodes[:, 0] - rod[0], nodes[:, 2] - rod[1])
                touching = penetration > 1e-6
                values[f"rod_force_{k}"] = float(params["rod_contact_stiffness"] * penetration[touching].sum()) * sym
                values[f"rod_contacts_{k}"] = float(touching.sum()) * sym
        return values


class _Settled:
    """Ends a run once both fingers' pressures have reached their targets and their cavity volumes have stopped
    changing (less than 0.02 % over 0.3 s)."""

    def __init__(self):
        self.history = []

    def __call__(self, root, module):
        ramps = _ramps(root)
        t = float(root.time.value)
        volume = sum(float(r.pressure.cavityVolume.value) for r in ramps)
        if self.history and t < self.history[-1][0]:
            self.history = []  # rebuilt
        self.history.append((t, volume))
        self.history = [h for h in self.history if h[0] >= t - 0.3 - 1e-9]
        for r in ramps:
            target = float(r.targetPressure.value)
            if abs(r.applied_pressure() - target) > 1e-6 * max(1.0, abs(target)):
                return None
        if t - self.history[0][0] < 0.3 - 1e-9:
            return None
        volumes = [v for _, v in self.history]
        if max(volumes) - min(volumes) <= 2e-4 * max(abs(volume), 1e-12):
            return f"Settled (t = {t:.2f} s): the volumes no longer change."
        return None


def _set_all(data_name, cast):
    def apply(root, value):
        for ramp in _ramps(root):
            getattr(ramp, data_name).value = cast(value)
    return apply


def _set_pressures(root, values):
    """One value (kPa) for both fingers, or one each."""
    values = list(values) if isinstance(values, (list, tuple)) and values else [values]
    for k, ramp in zip(FINGERS, _ramps(root)):
        ramp.targetPressure.value = float(values[k - 1] if k - 1 < len(values) else values[0]) * 1000.0


def _gauge(root, module):
    """Pressure gauge(s) (see SceneConfig.gauge), above the end-effector's plate and above
    their finger's cavity, each on a 4 mm tube that goes straight down into the finger's top end face
    (through the plate, which hides it). The scale runs from 0 to the first of 10, 20, 25, 50, 100, ... kPa
    that is at least 1.25 times the largest target pressure."""
    P = module.PARAMS
    g = module.gripper1_mesh.Geometry(dict(P, trunk_length=0.0, trunk_width=0.0, clamp_to_trunk=False))
    mm = 1e-3
    H = g.H * mm
    R = 0.6 * H  # the dial's radius
    tube_r = 0.125 * H
    z_palm = P["effector_height"] * mm
    thick = 0.6 * H
    cz = z_palm + thick + 1.12 * R + 0.6 * H
    casing = cz - 1.12 * R
    cavity_x = P["finger_gap"] * mm / 2 + 0.5 * (g.wb + g.H - g.wt) * mm  # the cavity's mid height across the finger
    pressures = P["pressures"] if isinstance(P["pressures"], (list, tuple)) and P["pressures"] else [P["pressures"]]
    target = max(abs(float(v)) for v in pressures)
    top = next((v for v in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000) if v >= 1.25 * target), 2000)
    # one gauge (in the middle) while both fingers get the same pressure, else one per finger
    same = len({round(abs(float(v)), 9) for v in pressures}) <= 1
    gauges = []
    for k, side in zip(FINGERS, (1.0, -1.0)):
        if same and k != FINGERS[0]:
            break
        x = 0.0 if same else side * cavity_x
        gauges.append({
            "key": f"applied_{k}", "min": 0.0, "max": float(top), "unit": "kPa", "major": top / 5.0, "minor": top / 20.0,
            "center": [x, 0.0, cz], "radius": R, "right": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0],
            "tube": [[x, 0.0, casing], [x, 0.0, 0.5 * (casing + z_palm)], [x, 0.0, z_palm]],
            "tubeRadius": tube_r,
        })
    return gauges


def _from_finger6(v):
    """A Finger 6 preset as Gripper 1 parameters (see SceneConfig.import_presets). The finger (chambers,
    channel, layer, material, collision planes, rod contact) has the same parameters; what hangs where, the
    pressure and the rod's place are this scene's own."""
    kpa = float(v.get("target_pressure", 50000.0)) / 1000.0
    params = {"pressures": [kpa, kpa]}
    notes = [
        ("estimated", "pressures", f"{kpa:g} kPa in both fingers, from Finger 6's target pressure."),
        ("not imported", "base_height", f"{v.get('base_height', 0):g} mm: here the fingers hang from the "
                                        "end-effector (its height is 'End-effector height')."),
        ("not imported", "clamp_proximal_end", "the finger's proximal end face is always clamped to the end-effector here."),
        ("not imported", "clamp_to_trunk", "there is no trunk here: the fingers hang from the end-effector."),
        ("not imported", "trunk_length", "no trunk here."),
        ("not imported", "trunk_width", "no trunk here."),
        ("not imported", "rod_x", f"{v.get('rod_x', 0):g} mm: here the rod sits between the fingers (this scene's default)."),
        ("not imported", "rod_z", f"{v.get('rod_z', 0):g} mm: here the rod sits between the fingers (this scene's default)."),
    ]
    return {"params": params, "notes": notes,
            "handled": ["target_pressure", "base_height", "clamp_proximal_end", "clamp_to_trunk", "trunk_length",
                        "trunk_width", "rod_x", "rod_z"]}


LIVE = {
    "pressures": _set_pressures,
    "ramp_start": _set_all("rampStart", float),
    "ramp_time": _set_all("rampTime", float),
    "instant": _set_all("instant", bool),
}


def stop_time(p):
    # A safety limit; runs normally end earlier, once settled.
    return p["ramp_start"] + p["ramp_time"] + 15.0


def extend_time(p):
    return p["ramp_start"] + p["ramp_time"] + 10.0


# The default setup, also kept as a preset (every parameter that matters listed) so it can be restored.
_DEFAULT = {
    "segments": 5, "element_size": 2.0, "min_wall_layers": 1,
    "length": 8.0, "width": 20.0, "height": 16.0,
    "wall_end": 1.5, "wall_side": 1.5, "wall_bottom": 3.0, "wall_top": 1.5,
    "channel_height": 2.0, "channel_width": 2.0, "channel_wall": 1.5, "channel_separation": 2.0,
    "strain_limiting_layer": True, "strain_layer_thickness": 0.1, "strain_layer_young_modulus": 6.5e9,
    "strain_layer_poisson_ratio": 0.2, "strain_layer_density": 750.0, "strain_layer_tension_only": True,
    "hyperelastic": False, "young_modulus": 1.0e6, "poisson_ratio": 0.45, "density": 1000.0,
    "gravity_enabled": False, "pressures": [50.0, 50.0], "ramp_start": 0.0, "ramp_time": 0.5, "dt": 0.01,
    "contact_outer_face_only": True, "floor_size": 160.0, "finger_gap": 28.0, "effector_height": 80.0,
    "symmetry_y": True, "prevent_symmetry_crossing": True, "collision_planes": True, "plane_hold_stiffness": 1.0e4,
    "rod_enabled": True, "rod_x": 0.0, "rod_z": 36.0, "rod_radius": 8.0,
    "rod_contact_stiffness": 1.0e5, "rod_contact_whole_skin": False,
}

PRESETS = {
    "params": [
        {"name": "Gripper on a rod",
         "description": "The default: two human-scale PneuNet fingers (5 chambers, 48 mm) hanging 28 mm apart from the "
                        "end-effector, an 8 mm radius rod between them, 50 kPa in both: they curl in and close on "
                        "the rod.",
         "values": dict(_DEFAULT)},
        {"name": "Different pressures (30, 60 kPa)",
         "description": "The right finger (+x) at 30 kPa and the left at 60: the stronger finger pushes the rod "
                        "towards the weaker one. A single value in the pressure field sets both.",
         "values": {"pressures": [30.0, 60.0]}},
        {"name": "Left finger only (0, 50 kPa)",
         "description": "Only the left finger is inflated: it curls towards the rod, the right one hangs.",
         "values": {"pressures": [0.0, 50.0]}},
        {"name": "Pressure ramp starts at 1 s, gravity on",
         "description": "Gravity pulls on the fingers for the first second (they sag a little and settle), then "
                        "the pressure ramps up.",
         "values": {"gravity_enabled": True, "ramp_start": 1.0}},
        {"name": "No rod",
         "description": "The rod removed: the fingers close freely and, with the gap given, pass through each other "
                        "(there is no finger-to-finger contact).",
         "values": {"rod_enabled": False}},
        {"name": "Wide gap, big rod",
         "description": "The fingers 40 mm apart with a 14 mm radius rod between them.",
         "values": {"finger_gap": 40.0, "rod_radius": 14.0, "rod_z": 32.0}},
        {"name": "Rod contact on the whole skin",
         "description": "Every node of the fingers' outer skin can touch the rod, not just the underside's.",
         "values": {"rod_contact_whole_skin": True}},
        {"name": "Two element layers per wall",
         "description": "Every wall at least 2 elements thick, not 1 (the default): bends more, as a real chamber "
                        "does, but takes three to four times as long.",
         "values": {"min_wall_layers": 2}},
    ],
    "display": [
        {"name": "Half model",
         "description": "Only the modelled half of each finger (y >= 0), without its mirror image: look into the "
                        "chambers through the cut.",
         "values": {"mirror_visual_y": False}},
        {"name": "See-through walls",
         "description": "Translucent outer walls, showing the cavities and the strain-limiting layers.",
         "values": {"wall_alpha": 0.3, "wall_alpha:on": True, "interior_alpha": 1.0, "interior_alpha:on": True}},
        {"name": "No collision planes shown",
         "description": "The fingers without the magenta collision planes.",
         "values": {"channel_end_alpha:on": False}},
        {"name": "FEM strain",
         "description": "The tetrahedra coloured by strain, with their edges; surfaces hidden.",
         "values": {"wall_alpha:on": False, "interior_alpha:on": False, "strain_layer_alpha:on": False,
                    "overlay:fem": True, "overlay:femAlpha": 1.0, "overlay:femColor": "strainLit",
                    "overlay:femWire": True, "overlay:femWireAlpha": 0.25}},
    ],
    "lighting": [
        {"name": "Soft",
         "description": "More ambient light, softer key light.",
         "values": {"light:ambient:brightness": 0.6, "light:0:brightness": 0.7, "light:1:brightness": 0.45}},
        {"name": "High contrast",
         "description": "Little ambient light and a strong key light, to bring out the shape.",
         "values": {"light:ambient:brightness": 0.12, "light:0:brightness": 1.4, "light:1:brightness": 0.2}},
    ],
}


CONFIG = SceneConfig(
    title="Gripper 1",
    script="original/Gripper1.py",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"GRIPPER1_NO_EDITOR": "1"},
    # GL-only debug drawing, and the desktop GUI's per-step mirroring (the browser mirrors instead, see `display`).
    forced_params={"show_pressure_overlay": False, "show_fem_elements": False, "mirror_visual_y": False},
    hidden_params={"theme"},
    short_vectors={"pressures"},
    param_limits={
        "segments": (1, 6),
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
        "floor_size": (20.0, 4000.0),
        "finger_gap": (0.0, 500.0),
        "effector_height": (0.0, 2000.0),
        "plane_hold_stiffness": (0.0, 1e10),
        "rod_x": (-1000.0, 1000.0),
        "rod_z": (-1000.0, 2000.0),
        "rod_radius": (0.5, 500.0),
        "rod_contact_stiffness": (1e2, 1e9),
        "rod_alpha": (0.0, 1.0),
        "strain_layer_thickness": (0.01, 5.0),
        "strain_layer_young_modulus": (1e5, 1e12),
        "strain_layer_poisson_ratio": (0.0, 0.49),
        "strain_layer_density": (10.0, 20000.0),
        "strain_layer_alpha": (0.0, 1.0),
        "young_modulus": (1e4, 1e9),
        "poisson_ratio": (0.0, 0.49),
        "density": (10.0, 20000.0),
        "ramp_start": (0.0, 60.0),
        "ramp_time": (0.0, 60.0),
        "dt": (0.0001, 0.05),
        "wall_alpha": (0.0, 1.0),
        "interior_alpha": (0.0, 1.0),
        "channel_end_alpha": (0.0, 1.0),
    },
    sections=[
        ("Pressure", ["pressures", "ramp_start", "ramp_time", "instant"]),
        ("Mesh", ["element_size", "min_wall_layers"]),
        ("Fingers (each)", ["segments"]),
        ("Chamber geometry (mm)", ["length", "width", "height", "wall_end", "wall_side", "wall_bottom", "wall_top"]),
        ("Channel (mm)", ["channel_height", "channel_width", "channel_wall", "channel_separation"]),
        ("Gripper layout (mm)", ["finger_gap", "effector_height", "floor_size"]),
        ("Collision planes (between neighbouring chambers)", ["collision_planes", "contact_outer_face_only",
                                                              "plane_hold_stiffness"]),
        ("Rigid rod (mm; the axis along Y)", ["rod_enabled", "rod_x", "rod_z", "rod_radius", "rod_contact_whole_skin",
                                              "rod_contact_stiffness"]),
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
        {"key": "rod_alpha", "label": "Rod opacity", "type": "opacity", "models": r"^/rod/visual$",
         "requires": "rod_enabled"},
        {"key": "mount_alpha", "label": "End-effector opacity", "type": "opacity", "models": r"^/trunk/visual$"},
        {"key": "axes_alpha", "label": "Coordinate axes (X, Y) opacity", "type": "opacity",
         "models": r"^/(arrow|label)_[xyz]/visual$"},
        {"key": "mirror_visual_y", "label": "Mirror across Y symmetry plane", "type": "mirror", "axis": "Y",
         "models": r"(^|/)cube/((interior_)?visu|strain_layer_visu)/visual$", "requires": "symmetry_y"},
    ],
    presets=PRESETS,
    import_presets={"finger-6": {"title": "Finger 6: Finger on a rod", "convert": _from_finger6}},
    live_params=LIVE,
    probes=_Probes(),
    gauge=_gauge,
    charts=[
        {"title": "Finger pressures", "unit": "kPa",
         "series": [{"key": "applied_1", "label": "right finger", "color": "#2a78d6"},
                    {"key": "applied_2", "label": "left finger", "color": "#e8933c"},
                    {"key": "target_1", "label": "target right", "color": "#8f97a3"},
                    {"key": "target_2", "label": "target left", "color": "#5d6470"}]},
        {"title": "Cavity volume growth", "unit": "%",
         "series": [{"key": "growth_1", "label": "right", "color": "#2a78d6"},
                    {"key": "growth_2", "label": "left", "color": "#e8933c"}]},
        {"title": "Finger bend (tip angle)", "unit": "deg",
         "series": [{"key": "tip_angle_1", "label": "right", "color": "#2a78d6"},
                    {"key": "tip_angle_2", "label": "left", "color": "#e8933c"}]},
        {"title": "Force on the rod", "unit": "N",
         "series": [{"key": "rod_force_1", "label": "right finger", "color": "#2a78d6"},
                    {"key": "rod_force_2", "label": "left finger", "color": "#e8933c"}]},
    ],
    readouts=[
        {"key": "applied_1", "label": "pressure, right finger", "digits": 1, "unit": "kPa"},
        {"key": "applied_2", "label": "pressure, left finger", "digits": 1, "unit": "kPa"},
        {"key": "tip_angle_1", "label": "bend, right finger", "digits": 1, "unit": "deg"},
        {"key": "tip_angle_2", "label": "bend, left finger", "digits": 1, "unit": "deg"},
        {"key": "rod_force_1", "label": "force on the rod, right", "digits": 2, "unit": "N"},
        {"key": "rod_force_2", "label": "force on the rod, left", "digits": 2, "unit": "N"},
        {"key": "rod_contacts_1", "label": "nodes touching the rod, right", "digits": 0, "unit": ""},
        {"key": "rod_contacts_2", "label": "nodes touching the rod, left", "digits": 0, "unit": ""},
    ],
    stop_at=stop_time,
    extend_by=extend_time,
    end_condition=_Settled(),
    about=[
        "A gripper: two Finger 6 fingers (Finger 5's human-scale PneuNet index finger with 5 chambers, 48 mm, on a "
        "fixed end-effector) hang vertically from the plate of a fixed end-effector, their strain-limiting layers "
        "facing each other, with a rigid round rod between them. Inflated, each finger curls towards the other and "
        "closes on the rod. Each finger is a separate FEM body with its own cavity, collision planes between its "
        "chambers, and the simplified rod contact of Finger 6; the second finger is the first's mirror image.",
        "Pressure: the field takes one value (kPa) for both fingers, or two separated by a comma: the first for the "
        "right finger (the +x side in the default view), the second for the left. It can be changed while the "
        "simulation runs. 'Ramp start' delays the ramp (e.g. to let gravity act first). Each finger has its own "
        "gauge above the end-effector, on a tube into the finger's top: kPa in blue on the outer scale, bar in "
        "black inside (1 bar = 100 kPa).",
        "There is no contact between the two fingers, only with the rod: without the rod, or with a smaller "
        "gap, they pass through each other. The rod is fixed (it doesn't move when pushed), and its contact has no "
        "friction (see Finger 6).",
        "The layout (world, mm; z = 0 is the floor, the origin): 'End-effector height' is the plate's underside, "
        "where the fingers' proximal end faces are clamped, 'Finger gap' the distance between the fingers' "
        "inner faces. Import (next to the Parameters preset) loads a Finger 6 preset converted to this scene.",
        "As in Finger 5 and 6: dimensions in millimetres with a standard element size, single-wall chambers by "
        "default (the preset 'Two element layers per wall' bends more and takes three to four times as long), a "
        "paper strain-limiting layer, linear corotational or neo-Hookean silicone. Two fingers of 5 chambers have "
        "about 11,000 elements: roughly 0.2-0.3 s per time step on this server.",
    ],
    origin="Scene script: scenes/gripper-1/original/Gripper1.py (a copy of scenes/finger-6, with two hanging fingers).",
)
