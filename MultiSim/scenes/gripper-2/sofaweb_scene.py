"""Gripper 2: two hanging PneuNet fingers that pick up loose cylinders (Gripper2.py), run through sofaweb (see scenes/_base)."""

import math

import numpy as np

from sofaweb.config import SceneConfig

FINGERS = (1, 2)  # the finger on the +x side (the right one in the default view), and the other


def _ramps(root):
    return [root.getObject(f"PressureRamp{k}") for k in FINGERS]


def _finger(root, k):
    return root.getChild("world").getChild(f"finger{k}")


def _finger_dofs(root, k):
    return _finger(root, k).getChild("cube").getObject("dofs")


def _plane_frames(root, k):
    """The Rigid3 frames holding finger k's collision planes, or None when they're off."""
    planes = _finger(root, k).getChild("collisionPlanes")
    return planes.getObject("dofs") if planes is not None else None


def _cylinder_pose(root, j):
    """Position of cylinder j's centre [x, y, z] (metres)."""
    return root.getChild("world").getChild(f"cylinder{j}").getObject("dofs").position.value[0]


class _Probes:
    """Per finger: pressure, volume growth and bend (the tilt of the distal end face against the clamped
    proximal one, in the finger's own frame, so both fingers read alike: positive curls towards the
    strain-limiting layer, i.e. towards the other finger); the flange's position; and per cylinder, where it is
    (x, and how far it's lifted off the floor)."""

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
        values = {}
        for k, ramp in zip(FINGERS, _ramps(root)):
            f = build["fingers"][k - 1]
            volume = float(ramp.pressure.cavityVolume.value)
            volume0 = float(ramp.pressure.initialCavityVolume.value)
            values[f"target_{k}"] = float(ramp.targetPressure.value) / 1000.0
            values[f"applied_{k}"] = ramp.applied_pressure() / 1000.0
            values[f"growth_{k}"] = (volume / volume0 - 1.0) * 100.0 if volume0 else 0.0
            world = np.asarray(_finger_dofs(root, k).position.value)
            # Back into the finger's own frame (see Gripper2.FingerPose).
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
        flange = root.getObject("Flange")
        values["flange_x"] = float(flange.offset[0]) * 1000.0
        values["flange_z"] = module.FLANGE_START[1] + float(flange.offset[1]) * 1000.0
        for j, c in enumerate(build["cylinders"], start=1):
            p = _cylinder_pose(root, j)
            values[f"cyl{j}_x"] = float(p[0]) * 1000.0
            values[f"cyl{j}_height"] = (float(p[2]) - c["r"]) * 1000.0  # how far it's been lifted off the floor
        return values


def _set_all(data_name, cast):
    def apply(root, value):
        for ramp in _ramps(root):
            getattr(ramp, data_name).value = cast(value)
    return apply


def _control_pressure(root, value):
    """The pressure slider (kPa): both fingers' target (the ramp smooths the change)."""
    for ramp in _ramps(root):
        ramp.targetPressure.value = float(value) * 1000.0


def _control_flange_x(root, value):
    root.getObject("Flange").targetX.value = float(value) * 1e-3


def _control_flange_z(root, value):
    root.getObject("Flange").targetZ.value = (float(value) - FLANGE_START_Z) * 1e-3


FLANGE_START_Z = 80.0  # Gripper2.FLANGE_START[1]


def _gauge(root, module):
    """Pressure gauge(s) (see SceneConfig.gauge), above the end-effector's plate and above
    their finger's cavity, each on a 4 mm tube that goes straight down into the finger's top end face (through
    the plate, which hides it). They move with the flange. The scale is 0-100 kPa."""
    P = module.PARAMS
    g = module.gripper2_mesh.Geometry(dict(P, trunk_length=0.0, trunk_width=0.0, clamp_to_trunk=False))
    mm = 1e-3
    H = g.H * mm
    R = 0.6 * H  # the dial's radius
    tube_r = 0.125 * H
    z_palm = module.FLANGE_START[1] * mm
    thick = 0.6 * H
    cz = z_palm + thick + 1.12 * R + 0.6 * H
    casing = cz - 1.12 * R
    cavity_x = P["finger_gap"] * mm / 2 + 0.5 * (g.wb + g.H - g.wt) * mm  # the cavity's mid height across the finger
    # both fingers always get the same pressure: one gauge, in the middle
    gauges = []
    for k, side in zip(FINGERS[:1], (1.0,)):
        x = 0.0
        gauges.append({
            "key": f"applied_{k}", "min": 0.0, "max": 100.0, "unit": "kPa", "major": 20.0, "minor": 5.0,
            "center": [x, 0.0, cz], "radius": R, "right": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0],
            "tube": [[x, 0.0, casing], [x, 0.0, 0.5 * (casing + z_palm)], [x, 0.0, z_palm]],
            "tubeRadius": tube_r,
            # the dials and tubes move with the end-effector plate (a model: its bounding box's centre at rest is given)
            "follow": "^/trunk/visual$", "followRest": list(module.BUILD["flange_rest"]),
        })
    return gauges


def _from_finger6(v):
    """A Finger 6 preset as Gripper 2 parameters (see SceneConfig.import_presets): the finger (chambers, channel,
    layer, material, collision planes) has the same parameters. What hangs where, the pressure (a slider here) and
    the cylinders are this scene's own."""
    notes = [
        ("not imported", "target_pressure", f"{float(v.get('target_pressure', 0)) / 1000:g} kPa: here the pressure is the "
                                            "slider in the Controls panel."),
        ("not imported", "base_height", "the fingers hang from the end-effector here (the Controls panel's flange z)."),
        ("not imported", "clamp_proximal_end", "the fingers' proximal end faces are always clamped to the end-effector here."),
        ("not imported", "clamp_to_trunk", "there is no trunk here."),
        ("not imported", "trunk_length", "no trunk here."),
        ("not imported", "trunk_width", "no trunk here."),
        ("not imported", "rod_enabled", "the fixed rod is replaced by loose cylinders on the floor."),
        ("not imported", "rod_x", "no rod here."),
        ("not imported", "rod_z", "no rod here."),
        ("not imported", "rod_radius", "no rod here (see 'Cylinder radii')."),
        ("not imported", "rod_contact_stiffness", "see 'Contact stiffness'."),
        ("not imported", "rod_contact_whole_skin", "see 'Contact with the cylinders: every node of the outer skin'."),
        ("not imported", "rod_alpha", "no rod here."),
        ("not imported", "gravity_enabled", f"{'on' if v.get('gravity_enabled') else 'off'}: here gravity is on (the "
                                            "cylinders need it)."),
    ]
    return {"params": {}, "notes": notes,
            "handled": ["target_pressure", "base_height", "clamp_proximal_end", "clamp_to_trunk", "trunk_length", "trunk_width",
                        "rod_enabled", "rod_x", "rod_z", "rod_radius", "rod_contact_stiffness", "rod_contact_whole_skin",
                        "rod_alpha", "gravity_enabled", "segments", "floor_size"]}


def _set_flange(data_name):
    def apply(root, value):
        getattr(root.getObject("Flange"), data_name).value = float(value)
    return apply


LIVE = {
    "flange_acceleration": _set_flange("acceleration"),
    "flange_speed": _set_flange("speed"),
    "ramp_start": _set_all("rampStart", float),
    "ramp_time": _set_all("rampTime", float),
    "instant": _set_all("instant", bool),
}


# The default setup, also kept as a preset (every parameter that matters listed) so it can be restored.
_DEFAULT = {
    "segments": 5, "element_size": 2.0, "min_wall_layers": 1,
    "length": 8.0, "width": 20.0, "height": 16.0,
    "wall_end": 1.5, "wall_side": 1.5, "wall_bottom": 3.0, "wall_top": 1.5,
    "channel_height": 2.0, "channel_width": 2.0, "channel_wall": 1.5, "channel_separation": 2.0,
    "strain_limiting_layer": True, "strain_layer_thickness": 0.1, "strain_layer_young_modulus": 6.5e9,
    "strain_layer_poisson_ratio": 0.2, "strain_layer_density": 750.0, "strain_layer_tension_only": True,
    "hyperelastic": False, "young_modulus": 1.0e6, "poisson_ratio": 0.45, "density": 1000.0,
    "gravity_enabled": True, "ramp_start": 0.0, "ramp_time": 0.5, "dt": 0.01,
    "contact_outer_face_only": True, "floor_size": 320.0, "finger_gap": 28.0,
    "symmetry_y": True, "prevent_symmetry_crossing": True, "collision_planes": True, "plane_hold_stiffness": 1.0e4,
    "cylinder_radii": [12.0, 9.0, 6.0], "cylinder_x": [-60.0, 0.0, 60.0], "cylinder_length": 30.0,
    "cylinder_density": 600.0, "cylinder_damping": 0.2, "contact_stiffness": 1.0e3, "contact_whole_skin": False,
    "flange_acceleration": 0.06, "flange_speed": 0.15,
}

PRESETS = {
    "params": [
        {"name": "Three cylinders",
         "description": "The default: cylinders of radius 12, 9 and 6 mm lie on the floor 60 mm apart (a 30 mm long, "
                        "600 kg/m3 each, free to slide and roll along x, axis along y), the gripper hangs over the middle one.",
         "values": dict(_DEFAULT)},
        {"name": "Low element count (16 %)",
         "description": "A crude, fast model with 936 tetrahedra per finger (1,872 in all) instead of 5,736 (11,472): "
                        "6 mm elements, and no channel between the chambers. The chambers stand apart on the common "
                        "base layer with a 2 mm gap between them, like a traditional PneuNet; each has its own cavity, all "
                        "at the same pressure (the channel only equalises the pressure). The silicone is 0.7 MPa to make "
                        "up for the stiffness of such coarse elements (at 50 / 100 kPa the fingertips close to 7.1 / 4.8 mm "
                        "from the middle, against 5.9 / 4.7 mm for the default); softer than about 0.5 MPa it can "
                        "diverge at 100 kPa. About 45-60 ms per time step instead of 220 (0.07x real time in the browser, against 0.03x): "
                        "2-4 times faster. It lifts all three cylinders at 100 kPa.",
         "values": {"element_size": 6.0, "channel_height": 0.0, "channel_width": 0.0, "young_modulus": 7.0e5}},
        {"name": "Heavier cylinders (2000 kg/m3)",
         "description": "The same cylinders, three times heavier: harder to lift (there is no friction, so they are held "
                        "only by the fingers curling under them), steadier on the floor.",
         "values": {"cylinder_density": 2000.0}},
        {"name": "Five cylinders",
         "description": "Cylinders of radius 14, 11, 9, 7 and 5 mm.",
         "values": {"cylinder_radii": [14.0, 11.0, 9.0, 7.0, 5.0], "cylinder_x": [-100.0, -50.0, 0.0, 45.0, 85.0]}},
        {"name": "Wide gap (40 mm)",
         "description": "The fingers 40 mm apart: room for a bigger cylinder, less reach for a small one.",
         "values": {"finger_gap": 40.0, "cylinder_radii": [18.0, 12.0, 6.0]}},
        {"name": "Contact on the whole skin",
         "description": "Every node of the fingers' outer skin can touch a cylinder, not just the underside's.",
         "values": {"contact_whole_skin": True}},
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
    title="Gripper 2",
    script="original/Gripper2.py",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"GRIPPER2_NO_EDITOR": "1"},
    # GL-only debug drawing, and the desktop GUI's per-step mirroring (the browser mirrors instead, see `display`).
    forced_params={"show_pressure_overlay": False, "show_fem_elements": False, "mirror_visual_y": False},
    hidden_params={"theme"},
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
        "plane_hold_stiffness": (0.0, 1e10),
        "flange_acceleration": (0.005, 10.0),
        "flange_speed": (0.005, 1.0),
        "cylinder_length": (2.0, 200.0),
        "cylinder_density": (50.0, 20000.0),
        "cylinder_damping": (0.0, 200.0),
        "contact_stiffness": (10.0, 1e7),
        "cylinder_alpha": (0.0, 1.0),
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
        ("Pressure ramp", ["ramp_start", "ramp_time", "instant"]),
        ("Mesh", ["element_size", "min_wall_layers"]),
        ("Fingers (each)", ["segments"]),
        ("Chamber geometry (mm)", ["length", "width", "height", "wall_end", "wall_side", "wall_bottom", "wall_top"]),
        ("Channel (mm)", ["channel_height", "channel_width", "channel_wall", "channel_separation"]),
        ("Gripper layout (mm)", ["finger_gap", "floor_size"]),
        ("Flange motion", ["flange_acceleration", "flange_speed"]),
        ("Cylinders on the floor", ["cylinder_radii", "cylinder_x", "cylinder_length", "cylinder_density",
                                    "cylinder_damping", "contact_stiffness", "contact_whole_skin"]),
        ("Collision planes (between neighbouring chambers)", ["collision_planes", "contact_outer_face_only",
                                                              "plane_hold_stiffness"]),
        ("Strain-limiting layer", ["strain_limiting_layer", "strain_layer_thickness", "strain_layer_young_modulus",
                                   "strain_layer_poisson_ratio", "strain_layer_density",
                                   "strain_layer_tension_only"]),
        ("Material", ["hyperelastic", "young_modulus", "poisson_ratio", "density", "gravity_enabled"]),
        ("Simulation", ["dt"]),
        ("Symmetry", ["symmetry_y", "prevent_symmetry_crossing"]),
        ("Cutaway (rebuilds the skin)", ["cutaway", "interior_cutaway"]),
    ],
    controls=[
        {"key": "pressure", "label": "Pressure (both fingers)", "type": "slider", "min": 0, "max": 100, "step": 1,
         "default": 0, "unit": "kPa", "digits": 0, "apply": _control_pressure,
         "description": "Target pressure in both fingers; they inflate smoothly towards it."},
        {"key": "flange_x", "label": "Flange x", "type": "slider", "wide": True, "min": -120, "max": 120, "step": 1,
         "default": 0, "unit": "mm", "digits": 0, "apply": _control_flange_x,
         "description": "Move the end-effector sideways (over a cylinder). It moves at a limited speed."},
        {"key": "flange_z", "label": "Flange z", "type": "slider", "wide": True, "min": 50, "max": 160, "step": 1,
         "default": FLANGE_START_Z, "unit": "mm", "digits": 0, "apply": _control_flange_z,
         "description": "Height of the end-effector's underside above the floor: lower it to put the fingers "
                        "around a cylinder, raise it to lift."},
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
        {"key": "cylinder_alpha", "label": "Cylinder opacity", "type": "opacity",
         "models": r"(^|/)cylinder\d+/visu/visual$"},
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
                    {"key": "applied_2", "label": "left finger", "color": "#e8933c"}]},
        {"title": "Finger bend (tip angle)", "unit": "deg",
         "series": [{"key": "tip_angle_1", "label": "right", "color": "#2a78d6"},
                    {"key": "tip_angle_2", "label": "left", "color": "#e8933c"}]},
        {"title": "Flange position", "unit": "mm",
         "series": [{"key": "flange_x", "label": "x", "color": "#2a78d6"},
                    {"key": "flange_z", "label": "z", "color": "#1baf7a"}]},
        {"title": "Cylinders lifted off the floor", "unit": "mm",
         "series": [{"key": "cyl1_height", "label": "cylinder 1", "color": "#e8933c"},
                    {"key": "cyl2_height", "label": "cylinder 2", "color": "#1baf7a"},
                    {"key": "cyl3_height", "label": "cylinder 3", "color": "#c77dff"}]},
    ],
    readouts=[
        {"key": "applied_1", "label": "pressure, right finger", "digits": 1, "unit": "kPa"},
        {"key": "applied_2", "label": "pressure, left finger", "digits": 1, "unit": "kPa"},
        {"key": "flange_x", "label": "flange x", "digits": 1, "unit": "mm"},
        {"key": "flange_z", "label": "flange z", "digits": 1, "unit": "mm"},
        {"key": "cyl1_height", "label": "cylinder 1 lifted", "digits": 1, "unit": "mm"},
        {"key": "cyl2_height", "label": "cylinder 2 lifted", "digits": 1, "unit": "mm"},
        {"key": "cyl3_height", "label": "cylinder 3 lifted", "digits": 1, "unit": "mm"},
    ],
    stop_at=None,  # interactive: it keeps running
    about=[
        "A gripper to pick things up: two PneuNet fingers (Finger 6's, 5 chambers, 48 mm) hang from an end-effector and "
        "face each other, with loose cylinders of different diameters on the floor, their axes parallel to Y. In the "
        "Controls panel, the pressure slider (kPa, the same in both fingers) inflates the fingers, and the flange "
        "sliders move the end-effector sideways (x) and up and down (z) at a limited speed.",
        "To pick up a cylinder: move the flange over it (x), lower it (z) until the fingers hang around the cylinder, "
        "raise the pressure so that the fingers curl in and under it, then raise the flange. The cylinders are rigid "
        "bodies that slide and roll on the floor (no friction with the floor beyond a little damping) and are held "
        "only by the fingers' shape: the contact between the fingers and a cylinder, between the fingers and the floor, "
        "and between two cylinders has no friction. The fingers can't cross the plane between them (they don't touch "
        "otherwise).",
        "The contacts are simplified penalty models like Finger 6's rod contact (each underside node's distance to the "
        "cylinder's centreline, a one-sided spring, a little damping), now acting on the cylinder too. The model is the "
        "+y half (the fingers are half models, mirrored by the browser), so a cylinder is its half too: half the mass. "
        "The contact stiffness is soft (1e3 N/m per node, and less for a lighter cylinder: at most 3e5 N/m per kg of its "
        "mass): the cylinders are light, and a stiffer contact makes them unstable at this time step.",
        "Channel height 0 leaves out the channel between the chambers: they then stand apart on the base layer, each with "
        "its own cavity at the same pressure, as in a traditional PneuNet (the channel only equalises the pressure); the "
        "preset 'Low element count' uses it, with coarse elements, for a model that is 2-4 times faster.",
        "The run takes about 0.2 s per time step on this server for the two fingers, so the controls respond slowly; "
        "the flange accelerates gently (0.06 m/s2, up to 150 mm/s of simulated time; both are live parameters under 'Flange motion') so that the inflated fingers, which swing like pendulums, can keep up. The pressure ramp, the gravity and the rest are the "
        "parameters; Import loads a Finger 6 preset's finger.",
    ],
    origin="Scene script: scenes/gripper-2/original/Gripper2.py (a copy of scenes/gripper-1, with sliders and loose cylinders).",
)
