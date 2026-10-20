"""Gripper 4 on MuJoCo: the scene of ../gripper-4 (a gripper with any number of rigid-segment fingers, moving in X, Y
and Z over grocery items and a box) simulated with MuJoCo (original/Gripper4Mj.py), run through sofaweb's MuJoCo engine
(see scenes/_base/sofaweb/engines.py and mjscene.py)."""

import math

from sofaweb.config import SceneConfig

FLANGE_START_Z = 190.0  # Gripper4Mj.FLANGE_START[2]
ITEM_KEYS = ["apple", "tomato", "egg", "can", "juice", "spice", "yogurt", "tea", "butter"]  # Gripper4Mj.ITEMS
ITEM_LABELS = {"apple": "apple", "tomato": "tomato", "egg": "egg", "can": "soda can", "juice": "juice box",
               "spice": "spice jar", "yogurt": "yogurt cup", "tea": "tea box", "butter": "butter"}
ITEM_COLOURS = {"apple": "#cc1a1a", "tomato": "#eb4020", "egg": "#c9b78f", "can": "#9e1420", "juice": "#f08c1a",
                "spice": "#80471f", "yogurt": "#6f8fd6", "tea": "#338c52", "butter": "#d9b13a"}


def _ramp(root):
    return root.getObject("PressureRamp")


class _Probes:
    """The applied pressure; the fingers' bend (the sum of a finger's hinge angles: the tilt of its tip against the
    end-effector, positive curling inwards; the mean over the fingers, and the least and most); where the end-effector
    is; per item, how far it has been lifted (its centre above where it started on the floor); and how many items
    are in the large box (their centres inside it, below its rim)."""

    def __call__(self, root, module):
        build = module.BUILD
        n = build["segments"] - 1
        values = {}
        ramp = _ramp(root)
        values["target"] = float(ramp.targetPressure) / 1000.0
        values["applied"] = ramp.applied_pressure() / 1000.0
        data = root.data
        angles = [float(data.qpos[a]) for a in build["hinge_qpos"]]
        bends = [math.degrees(sum(angles[k * n:(k + 1) * n])) for k in range(build["fingers"])]
        values["bend"] = sum(bends) / len(bends)
        values["bend_min"], values["bend_max"] = min(bends), max(bends)
        p = data.xpos[build["flange_id"]]
        rest = build["flange_rest0"]
        values["flange_x"] = (float(p[0]) - rest[0]) * 1000.0
        values["flange_y"] = (float(p[1]) - rest[1]) * 1000.0
        values["flange_z"] = FLANGE_START_Z + (float(p[2]) - rest[2]) * 1000.0
        box, inside = build.get("bin"), 0
        for item in build["items"]:
            q = data.xpos[item["body"]]
            values[f"{item['key']}_lift"] = max(0.0, (float(q[2]) - item["rest"][2]) * 1000.0)
            if box and box[0] < float(q[0]) < box[2] and box[1] < float(q[1]) < box[3] and float(q[2]) < box[4]:
                inside += 1
        values["in_box"] = float(inside)
        values["contacts"] = float(data.ncon)
        return values


def _set(data_name, cast):
    def apply(root, value):
        setattr(_ramp(root), data_name, cast(value))
    return apply


def _control_pressure(root, value):
    """The pressure slider (kPa): every finger's target (the ramp smooths the change)."""
    _ramp(root).targetPressure = float(value) * 1000.0


def _control_flange(axis, origin=0.0):
    def apply(root, value):
        setattr(root.getObject("Flange"), f"target{axis}", (float(value) - origin) * 1e-3)
    return apply


def _gauge(root, module):
    """A pressure gauge (see SceneConfig.gauge) above the end-effector's plate, beside its arm, on a 4 mm tube down
    into the plate (every finger gets the same pressure: one gauge). It moves with the end-effector. 0-100 kPa."""
    P = module.PARAMS
    mm = 1e-3
    H = float(P["height"]) * mm
    R = 0.6 * H  # the dial's radius
    tube_r = 0.125 * H
    z_palm = module.FLANGE_START[2] * mm
    thick = 0.6 * H
    cz = z_palm + thick + 1.12 * R + 0.6 * H
    casing = cz - 1.12 * R
    x = module.BUILD["stem"] + 1.25 * R  # on the +x side of the arm, clear of it (the casing is 1.12 R)
    return {
        "key": "applied", "min": 0.0, "max": 100.0, "unit": "kPa", "major": 20.0, "minor": 5.0,
        "center": [x, 0.0, cz], "radius": R, "right": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0],
        "tube": [[x, 0.0, casing], [x, 0.0, 0.5 * (casing + z_palm)], [x, 0.0, z_palm]],
        "tubeRadius": tube_r,
        # the dial and tube move with the end-effector plate (a model: its bounding box's centre at rest is given)
        "follow": "(^|/)trunk/visual$", "followRest": list(module.BUILD["flange_rest"]),
    }


def _set_flange(data_name):
    def apply(root, value):
        setattr(root.getObject("Flange"), data_name, float(value))
    return apply


LIVE = {
    "flange_acceleration": _set_flange("acceleration"),
    "flange_speed": _set_flange("speed"),
    "ramp_start": _set("rampStart", float),
    "ramp_time": _set("rampTime", float),
    "instant": _set("instant", bool),
}


# The default setup, also kept as a preset (every parameter that matters listed) so it can be restored.
_DEFAULT = {
    "fingers": 4, "finger_radius": 45.0, "finger_azimuth": 45.0,
    "segments": 6, "length": 8.0, "width": 20.0, "height": 16.0,
    "wall_end": 1.5, "wall_side": 1.5, "wall_bottom": 3.0, "wall_top": 1.5, "channel_separation": 2.0,
    "floor_size": 600.0,
    "joint_stiffness": 0.34, "joint_damping": 0.01, "joint_open_limit": 90.0, "joint_armature": 2e-6,
    "pressure_torque_gain": 1.0,
    "flange_acceleration": 0.5, "flange_speed": 0.25, "flange_mass": 0.5, "wrist_stiffness": 3000.0,
    **{f"item_{k}": True for k in ITEM_KEYS},
    "item_mass_scale": 1.0, "item_spacing": 1.0, "item_yaw": 45.0,
    "bin_enabled": True, "bin_x": 0.0, "bin_y": 290.0, "bin_length": 220.0, "bin_width": 130.0, "bin_height": 70.0,
    "bin_wall": 8.0,
    "friction": 0.6, "torsional_friction": 0.005, "impratio": 10.0, "finger_collisions": True,
    "density": 1000.0, "gravity_enabled": True, "ramp_start": 0.0, "ramp_time": 0.5, "dt": 0.02, "timestep": 0.001,
}

PRESETS = {
    "params": [
        {"name": "Four fingers (default)",
         "description": "Four fingers of six rigid segments (Gripper 3's hinges), their inner faces 45 mm from the axis, on "
                        "the diagonals; nine grocery items on the floor, turned 45 degrees to face them, and a large box at "
                        "the back to put them in.",
         "values": dict(_DEFAULT)},
        {"name": "Three fingers",
         "description": "Three fingers, one at the back and two at the front sides. Good for round things.",
         "values": {"fingers": 3, "finger_azimuth": 90.0}},
        {"name": "Two fingers (for the butter)",
         "description": "Two fingers facing each other across the diagonal, onto the butter's long sides: it can be "
                        "picked up (lower the fingers to about 60 mm, 100 kPa).",
         "values": {"fingers": 2, "finger_azimuth": 135.0}},
        {"name": "Six fingers",
         "description": "Six fingers, their inner faces 50 mm from the axis: a cage around an item.",
         "values": {"fingers": 6, "finger_radius": 50.0, "finger_azimuth": 0.0}},
        {"name": "Long fingers (nine segments)",
         "description": "Nine segments per finger (88 mm instead of 58), set on a wider circle (55 mm), reaching further "
                        "around a tall item.",
         "values": {"segments": 9, "finger_radius": 55.0}},
        {"name": "Heavier items (twice the mass)",
         "description": "Every item twice as heavy as the real one: harder to hold by friction.",
         "values": {"item_mass_scale": 2.0}},
        {"name": "Slippery (friction 0.3)",
         "description": "Half the friction: items slip out unless the fingers curl under them.",
         "values": {"friction": 0.3}},
        {"name": "Grippy (friction 1.0)",
         "description": "Rubber on cardboard: more friction, so tall items are held with less pressure.",
         "values": {"friction": 1.0}},
        {"name": "Fingers pass through each other",
         "description": "As in the SOFA scene: the fingers don't touch each other (here without its invisible guard "
                        "planes, so fingers closing on nothing cross).",
         "values": {"finger_collisions": False}},
        {"name": "Coarse time step (5 ms)",
         "description": "MuJoCo steps of 5 ms instead of 1 ms: faster, softer contacts; see how the grip changes.",
         "values": {"timestep": 0.005}},
    ],
    "display": [
        {"name": "See-through segments",
         "description": "Translucent segments, showing the base layers and the hinges.",
         "values": {"wall_alpha": 0.3, "wall_alpha:on": True}},
        {"name": "See-through items",
         "description": "Translucent items, to see how the fingers hold them.",
         "values": {"item_alpha": 0.4, "item_alpha:on": True}},
        {"name": "No hinge marks",
         "description": "The fingers without the orange hinge marks.",
         "values": {"joint_alpha:on": False}},
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
    title="Gripper 4 (MuJoCo)",
    script="original/Gripper4Mj.py",
    engine="mujoco",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"GRIPPER4MJ_NO_EDITOR": "1"},
    hidden_params={"theme"},
    param_limits={
        "fingers": (1, 8),
        "finger_radius": (0.0, 200.0),
        "finger_azimuth": (-360.0, 360.0),
        "segments": (2, 15),
        "length": (1.0, 100.0),
        "width": (2.0, 100.0),
        "height": (2.0, 100.0),
        "wall_end": (0.2, 50.0),
        "wall_side": (0.2, 50.0),
        "wall_bottom": (0.2, 50.0),
        "wall_top": (0.2, 50.0),
        "channel_separation": (0.1, 50.0),
        "floor_size": (200.0, 2000.0),
        "joint_stiffness": (1e-3, 100.0),
        "joint_damping": (0.0, 1.0),
        "joint_open_limit": (5.0, 170.0),
        "joint_armature": (0.0, 1e-3),
        "pressure_torque_gain": (0.0, 10.0),
        "flange_acceleration": (0.005, 10.0),
        "flange_speed": (0.005, 1.0),
        "flange_mass": (0.01, 20.0),
        "wrist_stiffness": (100.0, 1e5),
        "item_mass_scale": (0.05, 20.0),
        "item_spacing": (0.6, 3.0),
        "item_yaw": (-360.0, 360.0),
        "friction": (0.0, 3.0),
        "torsional_friction": (0.0, 0.05),
        "impratio": (1.0, 100.0),
        "timestep": (0.0001, 0.01),
        "item_alpha": (0.0, 1.0),
        "bin_x": (-600.0, 600.0),
        "bin_y": (-600.0, 600.0),
        "bin_length": (40.0, 600.0),
        "bin_width": (40.0, 600.0),
        "bin_height": (10.0, 250.0),
        "bin_wall": (2.0, 40.0),
        "density": (10.0, 20000.0),
        "ramp_start": (0.0, 60.0),
        "ramp_time": (0.0, 60.0),
        "dt": (0.002, 0.03),
        "wall_alpha": (0.0, 1.0),
        "joint_alpha": (0.0, 1.0),
    },
    sections=[
        ("Pressure ramp", ["ramp_start", "ramp_time", "instant"]),
        ("Fingers", ["fingers", "finger_radius", "finger_azimuth", "segments"]),
        ("Segment geometry (mm)", ["length", "width", "height", "wall_end", "wall_side", "wall_bottom", "wall_top",
                                   "channel_separation"]),
        ("Joints", ["joint_stiffness", "joint_damping", "joint_open_limit", "joint_armature",
                    "pressure_torque_gain"]),
        ("End-effector motion", ["flange_acceleration", "flange_speed", "flange_mass", "wrist_stiffness"]),
        ("Items on the floor", [f"item_{k}" for k in ITEM_KEYS] + ["item_mass_scale", "item_spacing", "item_yaw",
                                                                   "floor_size"]),
        ("Large box", ["bin_enabled", "bin_x", "bin_y", "bin_length", "bin_width", "bin_height", "bin_wall"]),
        ("Contact", ["friction", "torsional_friction", "impratio", "finger_collisions"]),
        ("Material", ["density", "gravity_enabled"]),
        ("Simulation", ["dt", "timestep"]),
    ],
    controls=[
        {"key": "pressure", "label": "Pressure (all fingers)", "type": "slider", "min": 0, "max": 100, "step": 1,
         "default": 0, "unit": "kPa", "digits": 0, "apply": _control_pressure,
         "description": "Target pressure in every finger; they inflate smoothly towards it."},
        {"key": "flange_x", "label": "Flange x", "type": "slider", "wide": True, "min": -250, "max": 250, "step": 1,
         "default": 0, "unit": "mm", "digits": 0, "apply": _control_flange("X"),
         "description": "Move the end-effector left / right. It moves at a limited speed."},
        {"key": "flange_y", "label": "Flange y", "type": "slider", "wide": True, "min": -250, "max": 330, "step": 1,
         "default": 0, "unit": "mm", "digits": 0, "apply": _control_flange("Y"),
         "description": "Move the end-effector towards the back (+, the large box is at 290 mm) or the front (-)."},
        {"key": "flange_z", "label": "Flange z", "type": "slider", "wide": True, "min": 30, "max": 300, "step": 1,
         "default": FLANGE_START_Z, "unit": "mm", "digits": 0, "apply": _control_flange("Z", FLANGE_START_Z),
         "description": "Height of the end-effector's underside above the floor: lower it to put the fingers "
                        "around an item, raise it to lift. Above 175 mm the fingers clear every item."},
    ],
    display=[
        {"key": "wall_alpha", "label": "Segment opacity", "type": "opacity",
         "models": r"(^|/)bodies/(visu|base_layer)/visual$"},
        {"key": "joint_alpha", "label": "Hinge mark opacity", "type": "opacity", "models": r"(^|/)bodies/joint_marks/visual$"},
        {"key": "item_alpha", "label": "Item opacity", "type": "opacity",
         "models": r"(^|/)items/[^/]+/(visu|detail\d+)/visual$"},
        {"key": "bin_alpha", "label": "Large box opacity", "type": "opacity", "models": r"^/bin/(visu|bottom)/visual$"},
        {"key": "floor_alpha", "label": "Floor opacity", "type": "opacity", "models": r"^/floor/visu/visual$"},
        {"key": "mount_alpha", "label": "End-effector opacity", "type": "opacity", "models": r"(^|/)trunk/visual$"},
        {"key": "axes_alpha", "label": "Coordinate axes (X, Y) opacity", "type": "opacity",
         "models": r"^/(arrow|label)_[xyz]/visual$"},
    ],
    presets=PRESETS,
    live_params=LIVE,
    probes=_Probes(),
    gauge=_gauge,
    charts=[
        {"title": "Pressure", "unit": "kPa",
         "series": [{"key": "target", "label": "target", "color": "#8f97a3"},
                    {"key": "applied", "label": "applied", "color": "#2a78d6"}]},
        {"title": "Finger bend (tip angle)", "unit": "deg",
         "series": [{"key": "bend", "label": "mean", "color": "#2a78d6"},
                    {"key": "bend_min", "label": "least", "color": "#8fb8ef"},
                    {"key": "bend_max", "label": "most", "color": "#1b4f99"}]},
        {"title": "End-effector position", "unit": "mm",
         "series": [{"key": "flange_x", "label": "x", "color": "#d6402a"},
                    {"key": "flange_y", "label": "y", "color": "#1baf7a"},
                    {"key": "flange_z", "label": "z", "color": "#2a78d6"}]},
        {"title": "Items lifted", "unit": "mm",
         "series": [{"key": f"{k}_lift", "label": ITEM_LABELS[k], "color": ITEM_COLOURS[k]} for k in ITEM_KEYS]},
    ],
    readouts=[
        {"key": "applied", "label": "pressure", "digits": 1, "unit": "kPa"},
        {"key": "bend", "label": "bend (mean)", "digits": 1, "unit": "deg"},
        {"key": "flange_x", "label": "flange x", "digits": 1, "unit": "mm"},
        {"key": "flange_y", "label": "flange y", "digits": 1, "unit": "mm"},
        {"key": "flange_z", "label": "flange z", "digits": 1, "unit": "mm"},
        {"key": "in_box", "label": "items in the box", "digits": 0, "unit": ""},
        {"key": "contacts", "label": "contacts", "digits": 0, "unit": ""},
    ] + [{"key": f"{k}_lift", "label": f"{ITEM_LABELS[k]} lifted", "digits": 1, "unit": "mm"} for k in ITEM_KEYS],
    stop_at=None,  # interactive: it keeps running
    about=[
        "Gripper 4 simulated with MuJoCo instead of SOFA, to compare the two: the same fingers, end-effector, grocery "
        "items, box, parameters and controls (see the Gripper 4 scene). In the Controls panel, the pressure slider "
        "inflates every finger, and the three flange sliders move the end-effector at a limited acceleration (0.5 m/s2) "
        "and speed (0.25 m/s). The large box at the back (y = 290 mm) is there to put the items in.",
        "To pick something up: move the flange over the item (x, y), lower it until the fingers hang around it (about "
        "60 mm for the egg, the tomato and the apple; just above the top for a tall item, e.g. 118 mm for the can), "
        "raise the pressure (80 kPa), then raise the flange. To put it in the box: raise the flange until the item clears "
        "the box's walls (300 mm for the can), move it to y = 290 mm, lower it a little and release the pressure.",
        "The model: each finger is a chain of six rigid boxes on MuJoCo hinge joints (the same stiffness, damping and "
        "limits as in SOFA; pressure is a torque of 0.086 N m per hinge at 50 kPa), hanging from an end-effector that "
        "slides on three stiff position actuators (the wrist). The free fingers bend exactly as in SOFA. Contacts are "
        "MuJoCo's soft constraints with an elliptic friction cone (friction 0.6); the items are MuJoCo's own spheres, "
        "ellipsoid (the egg), cylinders and boxes, and a convex mesh for the yogurt cup. Unlike in the SOFA scene, the "
        "fingers touch each other, so they need no invisible guards.",
        "Each 20 ms frame is 20 MuJoCo steps of 1 ms and takes about 2 ms of computing: several times faster than real "
        "time (the SOFA scene needs 8-16 ms per frame).",
    ],
    origin="Scene script: scenes/gripper-4-mujoco/original/Gripper4Mj.py (scenes/gripper-4 on MuJoCo).",
)
