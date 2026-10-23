"""Gripper 5 on MuJoCo: stocking shelves. Gripper 4's fingers (with pointy tips) on an end-effector that moves in six
degrees of freedom in a grocery store: two partly stocked shelf units side by side, crates of groceries on the floor
(original/Gripper5Mj.py), run through sofaweb's MuJoCo engine. Steered with the sliders, or with the mouse, a
touchscreen or the keyboard in the 3D view (SceneConfig.manipulator, the viewer's manipulator.js)."""

import math

from sofaweb.config import SceneConfig

TCP_START = (0.0, 0.0, 330.0)  # Gripper5Mj.TCP_START
PRODUCT_KEYS = ["apple", "tomato", "egg", "can", "juice", "spice", "yogurt", "tea", "butter"]
PRODUCT_LABELS = {"apple": "apples", "tomato": "tomatoes", "egg": "eggs", "can": "soda cans", "juice": "juice boxes",
                  "spice": "spice jars", "yogurt": "yogurt cups", "tea": "tea boxes", "butter": "butter"}


def _ramp(root):
    return root.getObject("PressureRamp")


class _Probes:
    """The applied pressure; the fingers' bend (the sum of a finger's hinge angles; the mean, least and most); where
    the tool point is and how the end-effector is turned (roll, pitch, yaw); and the stocking: how many of the
    crates' items stand in a place for their product, how many are still in the crates, how many shelf items are no
    longer in their place, and how many items lie on the floor."""

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
        f = build["flange_id"]
        p = data.xpos[f]
        values["tcp_x"], values["tcp_y"], values["tcp_z"] = (float(c) * 1000.0 for c in p)
        roll, pitch, yaw = module.matrix_to_euler(data.xmat[f])
        values["roll"], values["pitch"], values["yaw"] = math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
        states = module.item_states(data)
        stocked = in_crates = displaced = floor = 0
        for item, state in zip(build["items"], states):
            if item["origin"] == "shelf":
                displaced += state != "stocked"
            else:
                stocked += state == "stocked"
                in_crates += state == "crate"
            floor += state == "floor"
        values["stocked"] = float(stocked)
        values["in_crates"] = float(in_crates)
        values["displaced"] = float(displaced)
        values["on_floor"] = float(floor)
        values["contacts"] = float(data.ncon)
        return values


def _set(data_name, cast):
    def apply(root, value):
        setattr(_ramp(root), data_name, cast(value))
    return apply


def _control_pressure(root, value):
    """The pressure slider (kPa): every finger's target (the ramp smooths the change)."""
    _ramp(root).targetPressure = float(value) * 1000.0


def _control_position(axis):
    def apply(root, value):
        setattr(root.getObject("Flange"), f"target{axis}", float(value) * 1e-3)
    return apply


def _control_angle(name):
    def apply(root, value):
        setattr(root.getObject("Flange"), name, math.radians(float(value)))
    return apply


def _control_repair(root):
    """The Repair button: straight fingers, the end-effector on its target, everything at rest."""
    root.getObject("Repair")(root)


def _set_flange(data_name, factor=1.0):
    def apply(root, value):
        setattr(root.getObject("Flange"), data_name, float(value) * factor)
    return apply


LIVE = {
    "flange_acceleration": _set_flange("acceleration"),
    "flange_speed": _set_flange("speed"),
    "angular_acceleration": _set_flange("angular_acceleration", math.pi / 180.0),
    "angular_speed": _set_flange("angular_speed", math.pi / 180.0),
    "ramp_start": _set("rampStart", float),
    "ramp_time": _set("rampTime", float),
    "instant": _set("instant", bool),
}


# The default setup, also kept as a preset (every parameter that matters listed) so it can be restored.
_DEFAULT = {
    "fingers": 4, "finger_radius": 45.0, "finger_azimuth": 0.0,
    "segments": 6, "length": 8.0, "width": 20.0, "height": 16.0,
    "wall_end": 1.5, "wall_side": 1.5, "wall_bottom": 3.0, "wall_top": 1.5, "channel_separation": 2.0,
    "tips_enabled": True, "tip_length": 14.0, "tip_width": 3.0, "tip_thickness": 2.0,
    "joint_stiffness": 0.34, "joint_damping": 0.01, "joint_open_limit": 90.0, "side_stiffness": 5.0,
    "side_limit": 20.0, "joint_armature": 2e-6,
    "limit_timeconst": 0.002,
    "pressure_torque_gain": 1.0,
    "tcp_depth": 50.0, "flange_acceleration": 2.0, "flange_speed": 0.4, "angular_acceleration": 720.0,
    "angular_speed": 120.0, "flange_mass": 0.5, "wrist_timeconst": 0.02, "wrist_damping": 1.0,
    "shelf_distance": 300.0, "shelf_length": 1200.0, "shelf_depth": 320.0, "deck_height": 100.0, "level_height": 280.0,
    "board_thickness": 20.0, "lip_height": 8.0, "second_unit": True,
    "crate_length": 360.0, "crate_width": 240.0, "crate_height": 80.0, "crate_wall": 6.0, "crate_spacing": 560.0,
    "item_gap": 14.0, "crate_gap": 40.0, "item_mass_scale": 1.0,
    "friction": 0.8, "torsional_friction": 0.005, "rolling_friction": 0.002, "impratio": 10.0, "noslip_iterations": 10,
    "finger_collisions": True, "sleep": True,
    "density": 1000.0, "gravity_enabled": True, "ramp_start": 0.0, "ramp_time": 0.4, "dt": 0.02, "timestep": 0.001,
}

PRESETS = {
    "params": [
        {"name": "Four fingers (default)",
         "description": "Four fingers of six rigid segments with 14 mm pointy tips, their inner faces 45 mm from the "
                        "axis, facing the shelves' front and sides; two partly stocked shelf units side by side, with three "
                        "crates of groceries on the floor in front of them.",
         "values": dict(_DEFAULT)},
        {"name": "Without pointy tips",
         "description": "Square finger ends, as in Gripper 4: compare how well they get in between the items in a "
                        "crate.",
         "values": {"tips_enabled": False}},
        {"name": "Long tips (24 mm)",
         "description": "Longer, sharper tips: they reach deeper between items, but bend less around them.",
         "values": {"tip_length": 24.0, "tip_width": 2.0, "tip_thickness": 1.5}},
        {"name": "Three fingers",
         "description": "Three fingers, one at the back and two at the front sides. Good for round things.",
         "values": {"fingers": 3, "finger_azimuth": 90.0}},
        {"name": "Two fingers (for the butter)",
         "description": "Two fingers facing each other: they can take a block of butter by its long sides.",
         "values": {"fingers": 2, "finger_azimuth": 90.0}},
        {"name": "Long fingers (nine segments)",
         "description": "Nine segments per finger (88 mm instead of 58) on a wider circle (55 mm), reaching further "
                        "around a tall item.",
         "values": {"segments": 9, "finger_radius": 55.0}},
        {"name": "One shelf unit",
         "description": "Only the left shelf unit (centred): a smaller, lighter scene.",
         "values": {"second_unit": False}},
        {"name": "Crates close to the shelves (225 mm)",
         "description": "The shelves' front 225 mm from the crates' centre line instead of 300: less room to "
                        "manoeuvre between them.",
         "values": {"shelf_distance": 225.0, "crate_width": 200.0}},
        {"name": "Slippery (friction 0.4)",
         "description": "Half the friction: items slip out unless the fingers curl under them.",
         "values": {"friction": 0.4}},
        {"name": "Creeping grip (no-slip solver off)",
         "description": "MuJoCo's plain soft contacts: a held item slowly slides out of the fingers (about 1 mm/s "
                        "for a soda can), as in Gripper 4 (MuJoCo).",
         "values": {"noslip_iterations": 0}},
        {"name": "Lively: fast moves (8 m/s2)",
         "description": "Four times the acceleration and 2.5 times the speed: the fingers bend a few degrees and a "
                        "held can sways about 15 mm when the gripper starts and stops, then wobbles to rest.",
         "values": {"flange_acceleration": 8.0, "flange_speed": 1.0, "angular_acceleration": 2880.0,
                    "angular_speed": 360.0}},
        {"name": "Springy wrist",
         "description": "The gripper on a soft, lightly damped wrist (250 ms, damping ratio 0.15), as on a compliant "
                        "arm: it overshoots and sways after every move, and what it holds swings with it.",
         "values": {"wrist_timeconst": 0.25, "wrist_damping": 0.15, "flange_acceleration": 4.0,
                    "flange_speed": 0.8}},
        {"name": "Rigid sideways (as Gripper 4)",
         "description": "The fingers' joints only curl, with no sideways bending: a held item hardly moves relative to "
                        "the gripper.",
         "values": {"side_stiffness": 0.0}},
    ],
    "display": [
        {"name": "See-through shelves",
         "description": "Translucent shelf panels and boards, to see the gripper inside a shelf.",
         "values": {"shelf_alpha": 0.35, "shelf_alpha:on": True}},
        {"name": "See-through items",
         "description": "Translucent items, to see how the fingers hold them.",
         "values": {"item_alpha": 0.4, "item_alpha:on": True}},
        {"name": "No place marks",
         "description": "The shelves without the coloured marks of the empty places.",
         "values": {"spot_alpha:on": False}},
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
    title="Gripper 5: Stocking shelves (MuJoCo)",
    script="original/Gripper5Mj.py",
    engine="mujoco",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"GRIPPER5MJ_NO_EDITOR": "1"},
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
        "tip_length": (0.5, 60.0),
        "tip_width": (0.2, 50.0),
        "tip_thickness": (0.2, 50.0),
        "joint_stiffness": (1e-3, 100.0),
        "joint_damping": (0.0, 1.0),
        "joint_open_limit": (5.0, 170.0),
        "joint_armature": (0.0, 1e-3),
        "limit_timeconst": (0.0005, 0.5),
        "pressure_torque_gain": (0.0, 10.0),
        "tcp_depth": (0.0, 200.0),
        "flange_acceleration": (0.005, 10.0),
        "flange_speed": (0.005, 2.0),
        "angular_acceleration": (1.0, 10000.0),
        "angular_speed": (1.0, 1000.0),
        "flange_mass": (0.01, 20.0),
        "wrist_timeconst": (0.002, 2.0),
        "wrist_damping": (0.05, 5.0),
        "side_stiffness": (0.0, 1000.0),
        "side_limit": (0.1, 90.0),
        "shelf_distance": (125.0, 2000.0),
        "shelf_length": (400.0, 3000.0),
        "shelf_depth": (120.0, 800.0),
        "deck_height": (20.0, 500.0),
        "level_height": (150.0, 800.0),
        "board_thickness": (5.0, 60.0),
        "lip_height": (0.0, 60.0),
        "crate_length": (80.0, 800.0),
        "crate_width": (60.0, 600.0),
        "crate_height": (20.0, 300.0),
        "crate_wall": (2.0, 40.0),
        "crate_spacing": (100.0, 1500.0),
        "item_gap": (0.0, 100.0),
        "crate_gap": (0.0, 100.0),
        "item_mass_scale": (0.05, 20.0),
        "friction": (0.0, 3.0),
        "torsional_friction": (0.0, 0.05),
        "rolling_friction": (0.0, 0.05),
        "impratio": (1.0, 100.0),
        "noslip_iterations": (0, 100),
        "timestep": (0.0001, 0.01),
        "density": (10.0, 20000.0),
        "ramp_start": (0.0, 60.0),
        "ramp_time": (0.0, 60.0),
        "dt": (0.002, 0.03),
        "item_alpha": (0.0, 1.0),
        "wall_alpha": (0.0, 1.0),
        "joint_alpha": (0.0, 1.0),
        "spot_alpha": (0.0, 1.0),
    },
    sections=[
        ("Pressure ramp", ["ramp_start", "ramp_time", "instant"]),
        ("Fingers", ["fingers", "finger_radius", "finger_azimuth", "segments"]),
        ("Pointy tips", ["tips_enabled", "tip_length", "tip_width", "tip_thickness"]),
        ("Segment geometry (mm)", ["length", "width", "height", "wall_end", "wall_side", "wall_bottom", "wall_top",
                                   "channel_separation"]),
        ("Joints", ["joint_stiffness", "joint_damping", "joint_open_limit", "side_stiffness", "side_limit",
                    "joint_armature", "limit_timeconst",
                    "pressure_torque_gain"]),
        ("End-effector motion (6 DOF)", ["tcp_depth", "flange_acceleration", "flange_speed", "angular_acceleration",
                                         "angular_speed", "flange_mass", "wrist_timeconst", "wrist_damping"]),
        ("Store", ["shelf_distance", "shelf_length", "shelf_depth", "deck_height", "level_height", "board_thickness",
                   "lip_height", "second_unit", "item_gap", "show_spots"]),
        ("Crates", ["crate_length", "crate_width", "crate_height", "crate_wall", "crate_spacing", "crate_gap"]),
        ("Items", ["item_mass_scale"]),
        ("Contact", ["friction", "torsional_friction", "rolling_friction", "impratio", "noslip_iterations",
                     "finger_collisions", "sleep"]),
        ("Material", ["density", "gravity_enabled"]),
        ("Simulation", ["dt", "timestep"]),
    ],
    controls=[
        {"key": "pressure", "label": "Pressure (all fingers)", "type": "slider", "min": 0, "max": 100, "step": 1,
         "default": 0, "unit": "kPa", "digits": 0, "apply": _control_pressure,
         "description": "Target pressure in every finger; they inflate smoothly towards it (the Grip button: 80)."},
        {"key": "tcp_x", "label": "Gripper x", "type": "slider", "wide": True, "min": -1300, "max": 1300, "step": 1,
         "default": TCP_START[0], "unit": "mm", "digits": 0, "apply": _control_position("X"),
         "description": "The tool point (between the fingers) along the shelves: left unit at -, right unit at +."},
        {"key": "tcp_y", "label": "Gripper y", "type": "slider", "wide": True, "min": -700, "max": 700, "step": 1,
         "default": TCP_START[1], "unit": "mm", "digits": 0, "apply": _control_position("Y"),
         "description": "Towards the shelves (+, their front edge at 300 mm) or away from them (-). The crates "
                        "are at 0."},
        {"key": "tcp_z", "label": "Gripper z (height)", "type": "slider", "wide": True, "min": 0, "max": 1100,
         "step": 1, "default": TCP_START[2], "unit": "mm", "digits": 0, "apply": _control_position("Z"),
         "description": "Height of the tool point above the floor (the palm is 50 mm higher, the fingertips about "
                        "20 mm lower). Shelves at 100, 380 and 660 mm."},
        {"key": "roll", "label": "Roll (about x)", "type": "slider", "wide": True, "min": -180, "max": 180, "step": 1,
         "default": 0, "unit": "deg", "digits": 0, "apply": _control_angle("roll"),
         "description": "Tilt about x, along the shelves: +90 points the fingers into the shelves."},
        {"key": "pitch", "label": "Pitch (about y)", "type": "slider", "wide": True, "min": -90, "max": 90, "step": 1,
         "default": 0, "unit": "deg", "digits": 0, "apply": _control_angle("pitch"),
         "description": "Tilt about y: the fingers point along the shelves."},
        {"key": "yaw", "label": "Yaw (about z)", "type": "slider", "wide": True, "min": -180, "max": 180, "step": 1,
         "default": 0, "unit": "deg", "digits": 0, "apply": _control_angle("yaw"),
         "description": "Turn about the vertical: line the fingers up with a box's sides."},
        {"key": "repair", "label": "Repair the gripper", "type": "button", "apply": _control_repair,
         "description": "Straightens every finger and puts the gripper back on its target, at rest: for a finger "
                        "jammed by a hard push into the floor or a wall."},
    ],
    manipulator={
        "position": ["tcp_x", "tcp_y", "tcp_z"], "scale": 1e-3,
        "rotation": ["roll", "pitch", "yaw"],
        "grip": {"key": "pressure", "on": 80, "off": 0, "label": "Grip", "releaseLabel": "Let go"},
        "models": r"(^|/)bodies/(visu|base_layer|tips|joint_marks|trunk)/visual$",
        "jogSpeed": 150, "jogTurn": 60,
        "buttons": [{"key": "repair", "label": "Repair", "title": "Straighten jammed fingers: every finger "
                     "straight, the gripper back on its target, at rest"}],
    },
    display=[
        {"key": "wall_alpha", "label": "Segment opacity", "type": "opacity",
         "models": r"(^|/)bodies/(visu|base_layer|tips)/visual$"},
        {"key": "joint_alpha", "label": "Hinge mark opacity", "type": "opacity", "models": r"(^|/)bodies/joint_marks/visual$"},
        {"key": "item_alpha", "label": "Item opacity", "type": "opacity",
         "models": r"(^|/)items/[^/]+/(visu|detail\d+)/visual$"},
        {"key": "shelf_alpha", "label": "Shelf opacity", "type": "opacity", "models": r"(^|/)store/shelves/[^/]+/visual$"},
        {"key": "spot_alpha", "label": "Empty place marks opacity", "type": "opacity", "models": r"(^|/)store/spots/[^/]+/visual$"},
        {"key": "crate_alpha", "label": "Crate opacity", "type": "opacity", "models": r"(^|/)store/crates/[^/]+/visual$"},
        {"key": "floor_alpha", "label": "Floor opacity", "type": "opacity", "models": r"^/floor/[^/]+/visual$"},
        {"key": "mount_alpha", "label": "End-effector opacity", "type": "opacity", "models": r"(^|/)trunk/visual$"},
        {"key": "axes_alpha", "label": "Coordinate axes (X, Y) opacity", "type": "opacity",
         "models": r"^/arrow_[xy]/visual$"},
    ],
    presets=PRESETS,
    live_params=LIVE,
    probes=_Probes(),
    charts=[
        {"title": "Stocking", "unit": "items",
         "series": [{"key": "stocked", "label": "stocked from the crates", "color": "#1baf7a"},
                    {"key": "in_crates", "label": "still in the crates", "color": "#b86b16"},
                    {"key": "displaced", "label": "shelf items out of place", "color": "#d6402a"},
                    {"key": "on_floor", "label": "on the floor", "color": "#8f97a3"}]},
        {"title": "Pressure", "unit": "kPa",
         "series": [{"key": "target", "label": "target", "color": "#8f97a3"},
                    {"key": "applied", "label": "applied", "color": "#2a78d6"}]},
        {"title": "Tool point position", "unit": "mm",
         "series": [{"key": "tcp_x", "label": "x", "color": "#d6402a"},
                    {"key": "tcp_y", "label": "y", "color": "#1baf7a"},
                    {"key": "tcp_z", "label": "z", "color": "#2a78d6"}]},
        {"title": "Orientation", "unit": "deg",
         "series": [{"key": "roll", "label": "roll", "color": "#d6402a"},
                    {"key": "pitch", "label": "pitch", "color": "#1baf7a"},
                    {"key": "yaw", "label": "yaw", "color": "#2a78d6"}]},
        {"title": "Finger bend (tip angle)", "unit": "deg",
         "series": [{"key": "bend", "label": "mean", "color": "#2a78d6"},
                    {"key": "bend_min", "label": "least", "color": "#8fb8ef"},
                    {"key": "bend_max", "label": "most", "color": "#1b4f99"}]},
    ],
    readouts=[
        {"key": "stocked", "label": "stocked from the crates", "digits": 0, "unit": ""},
        {"key": "in_crates", "label": "still in the crates", "digits": 0, "unit": ""},
        {"key": "displaced", "label": "shelf items out of place", "digits": 0, "unit": ""},
        {"key": "on_floor", "label": "items on the floor", "digits": 0, "unit": ""},
        {"key": "applied", "label": "pressure", "digits": 1, "unit": "kPa"},
        {"key": "tcp_x", "label": "x", "digits": 0, "unit": "mm"},
        {"key": "tcp_y", "label": "y", "digits": 0, "unit": "mm"},
        {"key": "tcp_z", "label": "z", "digits": 0, "unit": "mm"},
        {"key": "roll", "label": "roll", "digits": 0, "unit": "deg"},
        {"key": "pitch", "label": "pitch", "digits": 0, "unit": "deg"},
        {"key": "yaw", "label": "yaw", "digits": 0, "unit": "deg"},
        {"key": "contacts", "label": "contacts", "digits": 0, "unit": ""},
    ],
    stop_at=None,  # interactive: it keeps running
    about=[
        "Stock the shelves: Gripper 4's soft fingers, now with pointy tips, on an end-effector that moves in six "
        "degrees of freedom in a grocery store: two shelf units side by side with empty places, marked on the "
        "shelves in the colour of the product that belongs there (and a price tag of that colour on the shelf's "
        "edge); the three crates on the floor hold those products. Take them out of the crates and put each in an "
        "empty place for its product: the Stocking chart counts them. (A real store would have an aisle between "
        "facing units; here one side is open, for the view.)",
        "Steering, with a mouse (the toolbar under the view, mode Drag): drag the gripper itself to move it over "
        "the floor, Shift-drag to move it up and down; double-click a spot to send the gripper above it. The orange "
        "dot is the target, where the gripper is heading. Axes shows a gizmo at the tool point (between the "
        "fingers): drag its arrows and squares to move along an axis or in a plane; Turn shows its rings, to roll, "
        "pitch and yaw the gripper; World axes / Gripper axes switches the gizmo's frame (gripper axes: push along "
        "the fingers). Keys: arrows or WASD move it as seen from the camera, R / F up and down, Q / E turn it about "
        "the vertical, Space grips or lets go, T switches Drag / Axes / Turn. Dragging anywhere else turns the "
        "view, as always.",
        "On a touchscreen: drag the gripper with a finger to move it; switch on Height to drag it up and down "
        "instead, or use the tall height slider on the left; double-tap a spot to send it there; the gizmo's arrows "
        "and rings work by touch too, and drawn larger. Grip closes the fingers (80 kPa), Let go opens them, Point "
        "down takes out roll and pitch, Repair straightens jammed fingers (after slamming the gripper into the "
        "floor or a wall). Every move also shows in the Controls panel's sliders, which work as well.",
        "How to pick: put the gripper over an item, lower it until the palm is just above the item (the fingertips "
        "reach about 70 mm below the palm; for low items keep the tips off the crate's floor), Grip, wait a moment, "
        "lift. To place: lift the item over the shelf's 8 mm lip, move it in, lower it onto the shelf, Let go, and "
        "rise before pulling back, or the hanging fingers drag it out again. Turn the gripper (yaw) so the fingers "
        "face a box's flat sides; the butter is too long for four fingers (load the two-finger preset). Turning the "
        "gripper (roll 90) lets it reach into a shelf from the side.",
        "The model: each finger is a chain of rigid boxes on MuJoCo hinge joints (Gripper 3's stiffness, damping and "
        "limits; pressure is a torque on every hinge), plus a five times stiffer sideways hinge at every joint, its "
        "last segment ending in a tapered tip (a convex mesh). The "
        "end-effector is a free body held by a soft weld (20 ms, critically damped) to a target that moves with "
        "limited acceleration (2 m/s2) and speed, turning along the shortest rotation. The shelves, lips, panels and crates "
        "are static boxes; contacts have an elliptic friction cone (with 10 no-slip solver passes, or a held item creeps out of the grip), and the round items rolling friction. Resting "
        "items sleep (MuJoCo's island sleeping), so the 60 items cost little until touched: each 20 ms frame is "
        "20 MuJoCo steps of 1 ms, about 2-8 ms of computing.",
        "Dynamics: the segments and items have their real masses, so the fingers bend and a held item sways when the "
        "gripper speeds up or stops. With these light, stiff fingers that is small at the default 2 m/s2 (a held can "
        "about 1.5 mm, settling in a few wobbles); the presets Lively (8 m/s2) and Springy wrist make it plain to "
        "see.",
    ],
    origin="Scene script: scenes/gripper-5-mujoco/original/Gripper5Mj.py (Gripper 4 on MuJoCo, in a grocery store).",
)
