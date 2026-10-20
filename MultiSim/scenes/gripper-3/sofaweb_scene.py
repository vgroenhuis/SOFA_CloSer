"""Gripper 3: two hanging fingers made of rigid segments and compliant joints, picking up loose cylinders (Gripper3.py),
run through sofaweb (see scenes/_base)."""

import math

from sofaweb.config import SceneConfig

FINGERS = (1, 2)  # the finger on the +x side (the right one in the default view), and the other
FLANGE_START_Z = 80.0  # Gripper3.FLANGE_START[1]


def _ramps(root):
    return [root.getObject(f"PressureRamp{k}") for k in FINGERS]


def _bodies(root):
    return root.getChild("world").getChild("bodies").getObject("dofs")


def _cylinder_pose(root, j):
    """Position of cylinder j's centre [x, y, z] (metres)."""
    return root.getChild("world").getChild(f"cylinder{j}").getObject("dofs").position.value[0]


class _Probes:
    """Per finger: the applied pressure and the bend (the sum of the joint angles: the tilt of the distal segment
    against the end-effector, positive curling towards the other finger); the flange's position; and per cylinder,
    where it is (x, and how far it's lifted off the floor)."""

    def __init__(self):
        self.prev = {}
        self.t = None

    def __call__(self, root, module):
        t = float(root.time.value)
        if self.t is None or t < self.t:
            self.prev = {}  # rebuilt
        self.t = t
        build = module.BUILD
        n = build["segments"]
        pos = _bodies(root).position.value
        values = {}
        for k, ramp, side in zip(FINGERS, _ramps(root), (1.0, -1.0)):
            values[f"target_{k}"] = float(ramp.targetPressure.value) / 1000.0
            values[f"applied_{k}"] = ramp.applied_pressure() / 1000.0
            q = pos[build["body_of"][(k - 1, n - 1)]]
            angle = 2.0 * math.atan2(float(q[4]), float(q[6]))  # about the world y axis
            values[f"tip_angle_{k}"] = side * math.degrees(angle)
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


def _gauge(root, module):
    """Pressure gauge(s) (see SceneConfig.gauge), above the end-effector's plate and above
    their finger's cavity, each on a 4 mm tube that goes straight down into the finger's top end face (through
    the plate, which hides it). They move with the flange. The scale is 0-100 kPa."""
    P = module.PARAMS
    mm = 1e-3
    H = float(P["height"]) * mm
    R = 0.6 * H  # the dial's radius
    tube_r = 0.125 * H
    z_palm = module.FLANGE_START[1] * mm
    thick = 0.6 * H
    cz = z_palm + thick + 1.12 * R + 0.6 * H
    casing = cz - 1.12 * R
    cavity_x = P["finger_gap"] * mm / 2 + module.BUILD["cavity_mid"]  # the cavity's mid height across the finger
    # both fingers always get the same pressure: one gauge, on the +x side (clear of the arm up the middle, 0.3 H
    # either side of the axis; the casing is 1.12 R)
    gauges = []
    for k, side in zip(FINGERS[:1], (1.0,)):
        x = max(cavity_x, 0.3 * H + 1.25 * R)
        gauges.append({
            "key": f"applied_{k}", "min": 0.0, "max": 100.0, "unit": "kPa", "major": 20.0, "minor": 5.0,
            "center": [x, 0.0, cz], "radius": R, "right": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0],
            "tube": [[x, 0.0, casing], [x, 0.0, 0.5 * (casing + z_palm)], [x, 0.0, z_palm]],
            "tubeRadius": tube_r,
            # the dials and tubes move with the end-effector plate (a model: its bounding box's centre at rest is given)
            "follow": "(^|/)trunk/visual$", "followRest": list(module.BUILD["flange_rest"]),
        })
    return gauges


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
    "segments": 5, "length": 8.0, "width": 20.0, "height": 16.0,
    "wall_end": 1.5, "wall_side": 1.5, "wall_bottom": 3.0, "wall_top": 1.5, "channel_separation": 2.0,
    "joint_stiffness": 0.34, "joint_damping": 0.01, "joint_open_limit": 90.0, "pin_stiffness": 2.0e4,
    "joint_stop_stiffness": 5.0, "pressure_torque_gain": 1.0,
    "density": 1000.0, "gravity_enabled": True, "ramp_start": 0.0, "ramp_time": 0.5, "dt": 0.01,
    "floor_size": 320.0, "finger_gap": 28.0,
    "cylinder_radii": [12.0, 9.0, 6.0], "cylinder_x": [-60.0, 0.0, 60.0], "cylinder_length": 30.0,
    "cylinder_density": 600.0, "cylinder_damping": 0.2, "contact_stiffness": 1.0e3, "contact_whole_skin": False,
    "flange_acceleration": 0.5, "flange_speed": 0.25,
}

PRESETS = {
    "params": [
        {"name": "Three cylinders (default)",
         "description": "The default: five rigid segments per finger, hinges fitted to Gripper 2's finite-element fingers "
                        "(the tips follow its free-hanging fingers within about 2 mm from 10 to 100 kPa). Cylinders of radius "
                        "12, 9 and 6 mm lie on the floor 60 mm apart (30 mm long, 600 kg/m3 each, free to slide and roll "
                        "along x, axis along y); the gripper hangs over the middle one.",
         "values": dict(_DEFAULT)},
        {"name": "Ten segments (smoother bend)",
         "description": "Ten segments of 4 mm with 1 mm gaps instead of five of 8 mm with 2 mm: the finger bends in a "
                        "smoother curve. The hinge stiffness is raised to 0.77 N m/rad, so that the total bend at a given "
                        "pressure stays the same (nine hinges share it instead of four).",
         "values": {"segments": 10, "length": 4.0, "channel_separation": 1.0, "joint_stiffness": 0.77}},
        {"name": "Stiff hinges",
         "description": "Hinges 70 % stiffer (0.58 N m/rad): the fingers need more pressure to close.",
         "values": {"joint_stiffness": 0.58}},
        {"name": "Soft hinges",
         "description": "Hinges half as stiff (0.17 N m/rad): the fingers close at half the pressure.",
         "values": {"joint_stiffness": 0.17}},
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
        {"name": "Contact on the whole surface",
         "description": "Eight points on each segment's surface can touch a cylinder, not just three on the base face.",
         "values": {"contact_whole_skin": True}},
    ],
    "display": [
        {"name": "See-through segments",
         "description": "Translucent segments, showing the base layers and the hinges.",
         "values": {"wall_alpha": 0.3, "wall_alpha:on": True}},
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
    title="Gripper 3",
    script="original/Gripper3.py",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"GRIPPER3_NO_EDITOR": "1"},
    hidden_params={"theme"},
    param_limits={
        "segments": (1, 20),
        "length": (1.0, 300.0),
        "width": (2.0, 300.0),
        "height": (2.0, 300.0),
        "wall_end": (0.2, 50.0),
        "wall_side": (0.2, 50.0),
        "wall_bottom": (0.2, 50.0),
        "wall_top": (0.2, 50.0),
        "channel_separation": (0.1, 100.0),
        "floor_size": (20.0, 4000.0),
        "finger_gap": (0.0, 500.0),
        "joint_stiffness": (1e-3, 100.0),
        "joint_damping": (0.0, 1.0),
        "joint_open_limit": (5.0, 170.0),
        "pin_stiffness": (100.0, 1e6),
        "joint_stop_stiffness": (0.01, 100.0),
        "pressure_torque_gain": (0.0, 10.0),
        "flange_acceleration": (0.005, 10.0),
        "flange_speed": (0.005, 1.0),
        "cylinder_length": (2.0, 200.0),
        "cylinder_density": (50.0, 20000.0),
        "cylinder_damping": (0.0, 200.0),
        "contact_stiffness": (10.0, 1e7),
        "cylinder_alpha": (0.0, 1.0),
        "density": (10.0, 20000.0),
        "ramp_start": (0.0, 60.0),
        "ramp_time": (0.0, 60.0),
        "dt": (0.001, 0.02),
        "wall_alpha": (0.0, 1.0),
        "joint_alpha": (0.0, 1.0),
    },
    sections=[
        ("Pressure ramp", ["ramp_start", "ramp_time", "instant"]),
        ("Fingers (each)", ["segments"]),
        ("Segment geometry (mm)", ["length", "width", "height", "wall_end", "wall_side", "wall_bottom", "wall_top",
                                   "channel_separation"]),
        ("Joints", ["joint_stiffness", "joint_damping", "joint_open_limit", "pin_stiffness", "joint_stop_stiffness",
                    "pressure_torque_gain"]),
        ("Gripper layout (mm)", ["finger_gap", "floor_size"]),
        ("Flange motion", ["flange_acceleration", "flange_speed"]),
        ("Cylinders on the floor", ["cylinder_radii", "cylinder_x", "cylinder_length", "cylinder_density",
                                    "cylinder_damping", "contact_stiffness", "contact_whole_skin"]),
        ("Material", ["density", "gravity_enabled"]),
        ("Simulation", ["dt"]),
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
        {"key": "wall_alpha", "label": "Segment opacity", "type": "opacity",
         "models": r"(^|/)bodies/(visu|base_layer)/visual$"},
        {"key": "joint_alpha", "label": "Hinge mark opacity", "type": "opacity", "models": r"(^|/)bodies/joint_marks/visual$"},
        {"key": "floor_alpha", "label": "Floor opacity", "type": "opacity", "models": r"^/floor/visual$"},
        {"key": "cylinder_alpha", "label": "Cylinder opacity", "type": "opacity",
         "models": r"(^|/)cylinder\d+/visu/visual$"},
        {"key": "mount_alpha", "label": "End-effector opacity", "type": "opacity", "models": r"(^|/)trunk/visual$"},
        {"key": "axes_alpha", "label": "Coordinate axes (X, Y) opacity", "type": "opacity",
         "models": r"^/(arrow|label)_[xyz]/visual$"},
    ],
    presets=PRESETS,
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
        {"key": "tip_angle_1", "label": "bend, right finger", "digits": 1, "unit": "deg"},
        {"key": "tip_angle_2", "label": "bend, left finger", "digits": 1, "unit": "deg"},
        {"key": "flange_x", "label": "flange x", "digits": 1, "unit": "mm"},
        {"key": "flange_z", "label": "flange z", "digits": 1, "unit": "mm"},
        {"key": "cyl1_height", "label": "cylinder 1 lifted", "digits": 1, "unit": "mm"},
        {"key": "cyl2_height", "label": "cylinder 2 lifted", "digits": 1, "unit": "mm"},
        {"key": "cyl3_height", "label": "cylinder 3 lifted", "digits": 1, "unit": "mm"},
    ],
    stop_at=None,  # interactive: it keeps running
    about=[
        "The gripper of Gripper 2 with a much simpler model of the fingers: no finite elements, but a chain of rigid "
        "segments, one per chamber, joined by hinges that are springs. Two fingers of five segments (8 x 20 x 16 mm, 2 mm "
        "apart, like Gripper 2's chambers) hang from an end-effector and face each other, with loose cylinders of different "
        "diameters on the floor, their axes parallel to Y. In the Controls panel, the pressure slider (kPa, the same in both "
        "fingers) inflates the fingers, and the flange sliders move the end-effector sideways (x) and up and down (z) at a "
        "limited acceleration and speed.",
        "To pick up a cylinder: move the flange over it (x), lower it (z) until the fingers hang around the cylinder, "
        "raise the pressure so that the fingers curl in and under it, then raise the flange. The cylinders are rigid "
        "bodies that slide and roll on the floor and are held only by the fingers' shape: the contact between the fingers "
        "and a cylinder, the floor, and between two cylinders has no friction. The fingers can't cross the plane between them.",
        "A hinge (a SOFA JointSpringForceField between two frames at the base of the gap between two segments) is a pin "
        "that only turns about Y, with a torsional spring (joint stiffness, N m/rad). It is stopped hard when the roofs "
        "of two segments touch (the gap closing) and at the open limit. Pressure acts as a torque on each hinge: the pressure "
        "times the cavity's cross-section (17 x 11.5 mm) is the force on a chamber's end wall, and its mid height above the "
        "base (8.75 mm) the lever: 0.086 N m per hinge at 50 kPa, bending it by torque / stiffness. The default stiffness "
        "(0.34 N m/rad) was fitted so that the free-hanging fingertips follow Gripper 2's finite-element fingers within about "
        "2 mm between 10 and 100 kPa. The model has no ballooning of the walls, no nonlinearity of the silicone and no contact "
        "between a cylinder and the chamber's surface other than at a few points on the base face (three per segment: "
        "'Contact on the whole surface' uses eight).",
        "Everything is planar (x, z and the rotation about y), so a step takes only a few milliseconds: it runs faster than "
        "real time, where Gripper 2 needs 0.2 s. The solver is implicit with a small stiffness-proportional damping ('joint "
        "damping'); the joints' own velocity damping is explicit, which is unstable for bodies this light. The flange "
        "accelerates at up to 0.5 m/s2, up to 250 mm/s (live parameters under 'Flange motion').",
    ],
    origin="Scene script: scenes/gripper-3/original/Gripper3.py (a rigid-segment version of scenes/gripper-2).",
)
