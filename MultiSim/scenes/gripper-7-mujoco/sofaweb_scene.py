"""Gripper 7 on MuJoCo: Gripper 6's mobile manipulator with a choice of gripper (soft PneuNet fingers, rigid parallel
jaws, a three-jaw chuck, a suction cup), arm (UR-like, KUKA LBR iiwa, Franka Panda) and cart (omniwheels, differential
drive, tracks, car-like steering), and automatic stocking: click an item and the robot puts it in its place, or
stock everything in its crate (original/Gripper7Mj.py, run through sofaweb's MuJoCo engine)."""

import math

import mujoco
import numpy as np

from sofaweb.config import SceneConfig

TCP_START = (-150.0, 0.0, 620.0)  # Gripper7Mj.TCP_START (in the platform's frame)


def _ramp(root):
    return root.getObject("PressureRamp")


class _Probes:
    """The applied pressure; the fingers' bend (the sum of a finger's hinge angles; the mean, least and most); the
    platform's pose (x, y, yaw) and speed; where the tool point is and how it is turned (roll, pitch, yaw), in the
    platform's frame; whether its target can be reached (ik_ok); the arm's joint angles; and the stocking: how many
    of the crate's items stand in a place for their product, how many are still in the crate, how many shelf items
    are no longer in their place, and how many items lie on the floor."""

    def __call__(self, root, module):
        build = module.BUILD
        n = build["segments"] - 1
        values = {}
        ramp = _ramp(root)
        values["target"] = float(ramp.targetPressure) / 1000.0
        values["applied"] = ramp.applied_pressure() / 1000.0
        data = root.data
        angles = [float(data.qpos[a]) for a in build["hinge_qpos"]]
        if angles:
            bends = [math.degrees(sum(angles[k * n:(k + 1) * n])) for k in range(build["fingers"])]
            values["bend"] = sum(bends) / len(bends)
            values["bend_min"], values["bend_max"] = min(bends), max(bends)
        jaws = root.getObject("Jaws")
        values["opening"] = jaws.opening(data) * 1000.0 if jaws is not None else 0.0
        f, pid = build["flange_id"], build["platform_id"]
        R = data.xmat[pid].reshape(3, 3)
        base = data.xpos[pid]
        values["base_x"], values["base_y"] = float(base[0]) * 1000.0, float(base[1]) * 1000.0
        values["base_yaw"] = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        drive = root.getObject("Drive")
        values["drive_v"] = math.hypot(*drive.v)
        values["drive_w"] = math.degrees(drive.w)
        p = (data.xpos[f] - base) @ R
        values["tcp_x"], values["tcp_y"], values["tcp_z"] = (float(c) * 1000.0 for c in p)
        roll, pitch, yaw = module.matrix_to_euler(R.T @ data.xmat[f].reshape(3, 3))
        values["roll"], values["pitch"], values["yaw"] = math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
        values["ik_ok"] = 1.0 if root.getObject("Arm").reachable else 0.0
        for i, a in enumerate(build["arm_qpos"]):
            values[f"q{i + 1}"] = math.degrees(float(data.qpos[a]))
        stocker = root.getObject("Stocker")
        values["auto_left"] = float(len(stocker.queue) + (stocker.current is not None))
        values["auto_done"], values["auto_failed"] = float(stocker.done), float(stocker.failed)
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
        # The tactile skin: the contact force on each pad (its finger segment or jaw), all and the most on one.
        skin = build.get("skin_geoms") or {}
        if skin:
            per = dict.fromkeys(build["skin_pads"], 0.0)
            f6 = np.zeros(6)
            for i in range(data.ncon):
                c = data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                pad = skin.get(g1) if g2 not in skin else (skin.get(g2) if g1 not in skin else None)
                if pad is None:
                    continue
                mujoco.mj_contactForce(root.model, data, i, f6)
                per[pad] += abs(float(f6[0]))
            for key, v in per.items():
                values[f"touch_{key}"] = v
            values["touch_total"] = sum(per.values())
            values["touch_max"] = max(per.values())
        vacuum = root.getObject("Vacuum")
        if vacuum is not None:
            values["seal"] = 100.0 * vacuum.seal
            values["cup_force"] = vacuum.force
        return values


def _set(data_name, cast):
    def apply(root, value):
        setattr(_ramp(root), data_name, cast(value))
    return apply


def _user(root):
    """Something the user did: the automatic stocking stops."""
    root.getObject("Stocker").cancel("stopped by hand")


def _control_pressure(root, value):
    """The grip slider (kPa): the fingers' pressure, or the jaws' force, or the suction (the ramp smooths the
    change)."""
    if abs(float(value) * 1000.0 - _ramp(root).targetPressure) > 1.0:
        _user(root)
    _ramp(root).targetPressure = float(value) * 1000.0


def _control_position(axis):
    def apply(root, value):
        flange = root.getObject("Flange")
        if abs(getattr(flange, f"target{axis}") - float(value) * 1e-3) > 1e-4:
            _user(root)
        setattr(flange, f"target{axis}", float(value) * 1e-3)
    return apply


def _control_angle(name):
    def apply(root, value):
        flange = root.getObject("Flange")
        if abs(getattr(flange, name) - math.radians(float(value))) > 1e-3:
            _user(root)
        setattr(flange, name, math.radians(float(value)))
    return apply


def _control_repair(root):
    """The Repair button: straight fingers, the end-effector on its target, everything at rest."""
    root.getObject("Repair")(root)


def _control_drive(name, factor=1.0):
    """A drive slider: the platform's forward or sideways speed [m/s] or turn rate [deg/s] (a deadman: back to zero
    unless the browser keeps sending it)."""
    def apply(root, value):
        if float(value) != 0.0:
            _user(root)
        setattr(root.getObject("Drive"), name, float(value) * factor)
    return apply


_PICK = [0.0, 0.0, 0.0]


def _control_pick(axis):
    """Where the view was clicked (hidden: the manipulator's `pick`), for the next "pick_go"."""
    def apply(root, value):
        _PICK[axis] = float(value) * 1e-3
    return apply


def _control_pick_go(root):
    """A click on an item: the robot stocks it (after what it's doing)."""
    root.getObject("Stocker").request_point(root.data, _PICK)


def _control_stock_all(root):
    root.getObject("Stocker").stock_all(root.data)


def _control_stop(root):
    root.getObject("Stocker").cancel("stopped")


def _sync(root, module):
    """The controls the automatic stocking moves: the tool point's target, its turn, the grip."""
    f = root.getObject("Flange")
    return {"tcp_x": f.targetX * 1000.0, "tcp_y": f.targetY * 1000.0, "tcp_z": f.targetZ * 1000.0,
            "roll": math.degrees(f.roll), "pitch": math.degrees(f.pitch), "yaw": math.degrees(f.yaw),
            "pressure": _ramp(root).targetPressure / 1000.0}


def _set_drive(data_name, factor=1.0):
    def apply(root, value):
        setattr(root.getObject("Drive"), data_name, float(value) * factor)
    return apply


def _set_flange(data_name, factor=1.0):
    def apply(root, value):
        setattr(root.getObject("Flange"), data_name, float(value) * factor)
    return apply


LIVE = {
    "flange_acceleration": _set_flange("acceleration"),
    "flange_speed": _set_flange("speed"),
    "angular_acceleration": _set_flange("angular_acceleration", math.pi / 180.0),
    "angular_speed": _set_flange("angular_speed", math.pi / 180.0),
    "drive_speed": _set_drive("speed"),
    "drive_acceleration": _set_drive("acceleration"),
    "turn_speed": _set_drive("turn_speed", math.pi / 180.0),
    "turn_acceleration": _set_drive("turn_acceleration", math.pi / 180.0),
    "ramp_start": _set("rampStart", float),
    "ramp_time": _set("rampTime", float),
    "instant": _set("instant", bool),
}


# The default setup, also kept as a preset (every parameter that matters listed) so it can be restored.
_DEFAULT = {
    "gripper": "pneunet", "arm": "ur", "cart": "omni", "jaw_stroke": 55.0, "jaw_length": 70.0, "jaw_width": 22.0,
    "jaw_force": 40.0, "suction_force": 30.0, "finray_length": 80.0, "finray_base": 30.0, "finray_stiffness": 0.12,
    "mia_torque": 1.0, "tactile_skin": True, "cup_vacuum": 30.0, "cup_lip_stiffness": 150.0, "sg_spacing": 45.0,
    "sg_open_vacuum": 20.0, "steer_limit": 35.0, "fingers": 4, "finger_radius": 45.0, "finger_azimuth": 0.0,
    "segments": 6, "length": 8.0, "width": 20.0, "height": 16.0, "wall_end": 1.5, "wall_side": 1.5, "wall_bottom":
    3.0, "wall_top": 1.5, "channel_separation": 2.0, "tips_enabled": True, "tip_length": 14.0, "tip_width": 3.0,
    "tip_thickness": 2.0, "joint_stiffness": 0.34, "joint_damping": 0.01, "joint_open_limit": 90.0,
    "side_stiffness": 5.0, "side_limit": 20.0, "joint_armature": 2e-06, "limit_timeconst": 0.002,
    "pressure_torque_gain": 1.0, "tcp_depth": 50.0, "arm_mount_x": 250.0, "arm_mount_y": 120.0, "arm_base_height":
    160.0, "arm_shoulder_offset": 135.0, "arm_upper": 450.0, "arm_elbow_offset": 120.0, "arm_fore": 400.0,
    "arm_wrist1": 100.0, "arm_wrist2": 100.0, "arm_wrist3": 100.0, "arm_gain": 3000.0, "ik_rate": 100.0,
    "ik_max_jump": 30.0, "platform_length": 1000.0, "platform_width": 700.0, "platform_height": 250.0,
    "platform_mass": 60.0, "start_x": -600.0, "start_y": -400.0, "drive_speed": 0.5, "drive_acceleration": 0.6,
    "turn_speed": 60.0, "turn_acceleration": 180.0, "flange_acceleration": 1.5, "flange_speed": 0.4,
    "angular_acceleration": 360.0, "angular_speed": 90.0, "flange_mass": 0.5, "shelf_distance": 300.0,
    "shelf_length": 1200.0, "shelf_depth": 320.0, "deck_height": 100.0, "level_height": 560.0, "board_thickness":
    20.0, "lip_height": 8.0, "second_unit": True, "crate_length": 440.0, "crate_width": 500.0, "crate_height": 80.0,
    "crate_wall": 6.0, "item_gap": 45.0, "crate_gap": 40.0, "item_mass_scale": 1.0, "friction": 0.8,
    "torsional_friction": 0.005, "rolling_friction": 0.002, "impratio": 10.0, "noslip_iterations": 10,
    "finger_collisions": True, "sleep": True, "density": 1000.0, "gravity_enabled": True, "ramp_start": 0.0,
    "ramp_time": 0.4, "dt": 0.02, "timestep": 0.001, "instant": False,
}

PRESETS = {
    "params": [
        {"name": "Default",
         "description": "Gripper 5's four soft fingers on a UR-like arm (about 950 mm reach) on an omniwheel cart with "
                        "a crate of twelve groceries, in front of two shelf units.",
         "values": dict(_DEFAULT)},
        {"name": "Franka, parallel jaws, two-wheel drive",
         "description": "A Franka Emika Panda with rigid parallel jaws on a differential-drive cart (it turns on the "
                        "spot but can't move sideways).",
         "values": {**_DEFAULT, "arm": "franka", "gripper": "parallel", "cart": "diff"}},
        {"name": "KUKA LBR iiwa, suction cup, tracks",
         "description": "A KUKA LBR iiwa 14 with a suction cup on a tracked cart (skid steering).",
         "values": {**_DEFAULT, "arm": "kuka", "gripper": "suction", "cart": "tank"}},
        {"name": "Ocado-style grocery picking",
         "description": "A suction-assisted soft gripper (SoftGripping's centric SoftGripper with a suction cup) on the "
                        "UR-like arm: the cup lifts, the soft fingers steady the item, as in automated grocery "
                        "picking.",
         "values": {**_DEFAULT, "gripper": "softgripping_suction"}},
        {"name": "Prensilia Mia hand, Franka",
         "description": "Prensilia's Mia hand on a Franka Emika Panda.",
         "values": {**_DEFAULT, "gripper": "mia", "arm": "franka"}},
        {"name": "Festo Fin Ray fingers, KUKA",
         "description": "Festo's Fin Ray fingers (DHAS) on a parallel gripper, on a KUKA LBR iiwa.",
         "values": {**_DEFAULT, "gripper": "festo_finray", "arm": "kuka"}},
        {"name": "GripWiq SofTouch cup, two-wheel drive",
         "description": "A GripWiq SofTouch suction cup (ST) on the UR-like arm, on a differential-drive cart.",
         "values": {**_DEFAULT, "gripper": "gripwiq_st", "cart": "diff"}},
        {"name": "Three-jaw chuck, car-like cart",
         "description": "The UR-like arm with a rigid three-jaw chuck, on a cart that steers its front wheels like a "
                        "car (it turns only while moving).",
         "values": {**_DEFAULT, "gripper": "three_jaw", "cart": "car"}},
        {"name": "Fast robot",
         "description": "Twice the drive speeds and accelerations, and a quicker arm: the groceries in the crate slide "
                        "when it brakes hard.",
         "values": {"drive_speed": 1.0, "drive_acceleration": 1.5, "turn_speed": 120.0, "turn_acceleration": 360.0,
                    "flange_acceleration": 3.0, "flange_speed": 0.8, "angular_acceleration": 720.0,
                    "angular_speed": 180.0}},
        {"name": "Long arm (1.2 m reach)",
         "description": "A larger arm (upper arm 600, forearm 500 mm): it reaches the shelves from further away.",
         "values": {"arm": "ur", "arm_upper": 600.0, "arm_fore": 500.0}},
        {"name": "Two fingers (for the butter)",
         "description": "Two fingers facing each other: they can take a block of butter by its long sides.",
         "values": {"gripper": "pneunet", "fingers": 2, "finger_azimuth": 90.0}},
        {"name": "Without pointy tips",
         "description": "Square finger ends, as in Gripper 4.",
         "values": {"tips_enabled": False}},
        {"name": "One shelf unit",
         "description": "Only the left shelf unit (centred): a smaller, lighter scene.",
         "values": {"second_unit": False, "start_x": -250.0}},
        {"name": "Slippery (friction 0.4)",
         "description": "Half the friction: items slip out unless the fingers curl under them, and slide in the "
                        "crate when the robot brakes.",
         "values": {"friction": 0.4}},
    ],
    "display": [
        {"name": "See-through shelves",
         "description": "Translucent shelf panels and boards, to see the gripper inside a shelf.",
         "values": {"shelf_alpha": 0.35, "shelf_alpha:on": True}},
        {"name": "See-through arm",
         "description": "A translucent arm, to see what the gripper is doing behind it.",
         "values": {"arm_alpha": 0.35, "arm_alpha:on": True}},
        {"name": "See-through items",
         "description": "Translucent items, to see how the fingers hold them.",
         "values": {"item_alpha": 0.4, "item_alpha:on": True}},
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

_LIMITS = {
    "fingers": (1, 8), "finger_radius": (0.0, 200.0), "finger_azimuth": (-360.0, 360.0), "segments": (2, 15),
    "length": (1.0, 100.0), "width": (2.0, 100.0), "height": (2.0, 100.0), "wall_end": (0.2, 50.0),
    "wall_side": (0.2, 50.0), "wall_bottom": (0.2, 50.0), "wall_top": (0.2, 50.0), "channel_separation": (0.1, 50.0),
    "tip_length": (0.5, 60.0), "tip_width": (0.2, 50.0), "tip_thickness": (0.2, 50.0),
    "joint_stiffness": (1e-3, 100.0), "joint_damping": (0.0, 1.0), "joint_open_limit": (5.0, 170.0),
    "side_stiffness": (0.0, 1000.0), "side_limit": (0.1, 90.0), "joint_armature": (0.0, 1e-3),
    "limit_timeconst": (0.0005, 0.5), "pressure_torque_gain": (0.0, 10.0), "tcp_depth": (0.0, 200.0),
    "flange_acceleration": (0.005, 10.0), "flange_speed": (0.005, 2.0), "angular_acceleration": (1.0, 10000.0),
    "angular_speed": (1.0, 1000.0), "flange_mass": (0.01, 5.0),
    "arm_mount_x": (-500.0, 500.0), "arm_mount_y": (-400.0, 400.0), "arm_base_height": (20.0, 600.0),
    "arm_shoulder_offset": (0.0, 400.0), "arm_upper": (100.0, 1200.0), "arm_elbow_offset": (0.0, 400.0),
    "arm_fore": (100.0, 1200.0), "arm_wrist1": (10.0, 300.0), "arm_wrist2": (10.0, 300.0), "arm_wrist3": (10.0, 300.0),
    "arm_gain": (10.0, 1e5), "ik_rate": (10.0, 500.0), "ik_max_jump": (1.0, 180.0),
    "platform_length": (300.0, 3000.0), "platform_width": (300.0, 3000.0), "platform_height": (60.0, 1000.0),
    "platform_mass": (1.0, 1000.0), "start_x": (-3000.0, 3000.0), "start_y": (-3000.0, 3000.0),
    "drive_speed": (0.01, 3.0), "drive_acceleration": (0.01, 10.0), "turn_speed": (1.0, 360.0),
    "turn_acceleration": (1.0, 3600.0),
    "shelf_distance": (125.0, 2000.0), "shelf_length": (400.0, 3000.0), "shelf_depth": (120.0, 800.0),
    "deck_height": (20.0, 500.0), "level_height": (150.0, 800.0), "board_thickness": (5.0, 60.0),
    "lip_height": (0.0, 60.0), "crate_length": (80.0, 900.0), "crate_width": (60.0, 700.0),
    "crate_height": (20.0, 300.0), "crate_wall": (2.0, 40.0), "item_gap": (0.0, 100.0), "crate_gap": (0.0, 100.0),
    "item_mass_scale": (0.05, 20.0), "friction": (0.0, 3.0), "torsional_friction": (0.0, 0.05),
    "rolling_friction": (0.0, 0.05), "impratio": (1.0, 100.0), "noslip_iterations": (0, 100),
    "timestep": (0.0001, 0.01), "density": (10.0, 20000.0), "ramp_start": (0.0, 60.0), "ramp_time": (0.0, 60.0),
    "dt": (0.002, 0.03), "item_alpha": (0.0, 1.0), "wall_alpha": (0.0, 1.0), "joint_alpha": (0.0, 1.0),
    "spot_alpha": (0.0, 1.0), "jaw_stroke": (10.0, 120.0), "jaw_length": (20.0, 200.0), "jaw_width": (5.0, 80.0),
    "jaw_force": (0.0, 500.0), "suction_force": (0.0, 500.0), "steer_limit": (5.0, 60.0),
    "cup_vacuum": (0.0, 95.0), "cup_lip_stiffness": (5.0, 5000.0), "sg_spacing": (0.0, 200.0),
    "sg_open_vacuum": (0.0, 80.0), "finray_length": (30.0, 200.0), "finray_base": (10.0, 80.0),
    "finray_stiffness": (0.005, 5.0), "mia_torque": (0.05, 5.0),
}

CONFIG = SceneConfig(
    title="Gripper 7: Robot shop assistant (MuJoCo)",
    script="original/Gripper7Mj.py",
    engine="mujoco",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"GRIPPER7MJ_NO_EDITOR": "1"},
    hidden_params={"theme"},
    param_limits=_LIMITS,
    control_sync=_sync,
    heat=[{"models": r"/bodies/skin/([^/]+)/visual$", "key": "touch_{}", "max": 4.0,
           "label": "Tactile skin heat map (Touchlab / TG0 style)"}],
    sections=[
        ("Robot", ["gripper", "arm", "cart"]),
        ("Rigid jaws, suction cups", ["jaw_stroke", "jaw_length", "jaw_width", "jaw_force", "suction_force",
                                      "cup_vacuum", "cup_lip_stiffness"]),
        ("SoftGripping", ["sg_spacing", "sg_open_vacuum"]),
        ("Festo Fin Ray fingers", ["finray_length", "finray_base", "finray_stiffness"]),
        ("Prensilia Mia hand", ["mia_torque"]),
        ("Tactile skin", ["tactile_skin"]),
        ("Grip ramp", ["ramp_start", "ramp_time", "instant"]),
        ("Cart", ["platform_length", "platform_width", "platform_height", "platform_mass", "start_x", "start_y",
                  "drive_speed", "drive_acceleration", "turn_speed", "turn_acceleration", "steer_limit"]),
        ("Arm (UR-like: its sizes; all: gains, IK)", ["arm_mount_x", "arm_mount_y", "arm_base_height",
                                                     "arm_shoulder_offset", "arm_upper", "arm_elbow_offset",
                                                     "arm_fore", "arm_wrist1", "arm_wrist2", "arm_wrist3",
                                                     "arm_gain", "ik_rate", "ik_max_jump"]),
        ("Tool point motion", ["tcp_depth", "flange_acceleration", "flange_speed", "angular_acceleration",
                               "angular_speed", "flange_mass"]),
        ("Soft fingers (PneuNet)", ["fingers", "finger_radius", "finger_azimuth", "segments"]),
        ("Pointy tips", ["tips_enabled", "tip_length", "tip_width", "tip_thickness"]),
        ("Segment geometry (mm)", ["length", "width", "height", "wall_end", "wall_side", "wall_bottom", "wall_top",
                                   "channel_separation"]),
        ("Joints", ["joint_stiffness", "joint_damping", "joint_open_limit", "side_stiffness", "side_limit",
                    "joint_armature", "limit_timeconst", "pressure_torque_gain"]),
        ("Store", ["shelf_distance", "shelf_length", "shelf_depth", "deck_height", "level_height", "board_thickness",
                   "lip_height", "second_unit", "item_gap", "show_spots"]),
        ("Platform's crate", ["crate_length", "crate_width", "crate_height", "crate_wall", "crate_gap"]),
        ("Items", ["item_mass_scale"]),
        ("Contact", ["friction", "torsional_friction", "rolling_friction", "impratio", "noslip_iterations",
                     "finger_collisions", "sleep"]),
        ("Material", ["density", "gravity_enabled"]),
        ("Simulation", ["dt", "timestep"]),
    ],
    controls=[
        {"key": "pressure", "label": "Grip (pressure)", "type": "slider", "min": 0, "max": 100, "step": 1,
         "default": 0, "unit": "kPa", "digits": 0, "apply": _control_pressure,
         "description": "The soft fingers' pressure; with rigid jaws their force (Jaw force at 100), with the suction "
                        "cup the vacuum (full at 80). The Grip button: 80."},
        {"key": "stock_all", "label": "Stock everything in the crate", "type": "button", "apply": _control_stock_all,
         "description": "The robot puts every item in its crate in an empty place for its product, one after the "
                        "other (click an item to stock just that one). Anything you do stops it."},
        {"key": "auto_stop", "label": "Stop the automatic stocking", "type": "button", "apply": _control_stop,
         "description": "Stops it where it is (an item it holds stays in the gripper)."},
        {"key": "pick_x", "type": "slider", "label": "clicked x", "hidden": True, "min": -10000, "max": 10000,
         "step": 1, "default": 0, "unit": "mm", "digits": 0, "apply": _control_pick(0)},
        {"key": "pick_y", "type": "slider", "label": "clicked y", "hidden": True, "min": -10000, "max": 10000,
         "step": 1, "default": 0, "unit": "mm", "digits": 0, "apply": _control_pick(1)},
        {"key": "pick_z", "type": "slider", "label": "clicked z", "hidden": True, "min": -10000, "max": 10000,
         "step": 1, "default": 0, "unit": "mm", "digits": 0, "apply": _control_pick(2)},
        {"key": "pick_go", "type": "button", "label": "stock the clicked item", "hidden": True,
         "apply": _control_pick_go},
        {"key": "drive_forward", "label": "Drive forward", "type": "slider", "wide": True, "deadman": True,
         "min": -0.5, "max": 0.5, "step": 0.01, "default": 0, "unit": "m/s", "digits": 2,
         "apply": _control_drive("forward"),
         "description": "The platform's speed along its heading (the yellow arrow on its deck). Hold the slider: it "
                        "springs back to 0 when let go (or use the joystick, the arrow keys)."},
        {"key": "drive_sideways", "label": "Drive sideways", "type": "slider", "wide": True, "deadman": True,
         "min": -0.5, "max": 0.5, "step": 0.01, "default": 0, "unit": "m/s", "digits": 2,
         "apply": _control_drive("sideways"),
         "description": "The platform's speed to its left (+) or right (-): omniwheels. Springs back to 0."},
        {"key": "drive_turn", "label": "Turn", "type": "slider", "wide": True, "deadman": True,
         "min": -60, "max": 60, "step": 1, "default": 0, "unit": "deg/s", "digits": 0,
         "apply": _control_drive("turn", math.pi / 180.0),
         "description": "Turning on the spot: + to the left. Springs back to 0."},
        {"key": "tcp_x", "label": "Gripper x (forward)", "type": "slider", "wide": True, "min": -900, "max": 1300,
         "step": 1, "default": TCP_START[0], "unit": "mm", "digits": 0, "apply": _control_position("X"),
         "description": "The tool point (between the fingers), forward of the platform's centre."},
        {"key": "tcp_y", "label": "Gripper y (left)", "type": "slider", "wide": True, "min": -1100, "max": 1100,
         "step": 1, "default": TCP_START[1], "unit": "mm", "digits": 0, "apply": _control_position("Y"),
         "description": "The tool point to the left of the platform's centre line (the shelves are on its left at the "
                        "start)."},
        {"key": "tcp_z", "label": "Gripper z (height)", "type": "slider", "wide": True, "min": 0, "max": 1500,
         "step": 1, "default": TCP_START[2], "unit": "mm", "digits": 0, "apply": _control_position("Z"),
         "description": "Height of the tool point above the floor. The platform's deck is at 250 mm, the shelves at "
                        "100, 380 and 660 mm."},
        {"key": "roll", "label": "Roll (about x)", "type": "slider", "wide": True, "min": -180, "max": 180, "step": 1,
         "default": 0, "unit": "deg", "digits": 0, "apply": _control_angle("roll"),
         "description": "Tilt about the platform's heading: +90 points the fingers to its left (into the shelves at "
                        "the start)."},
        {"key": "pitch", "label": "Pitch (about y)", "type": "slider", "wide": True, "min": -90, "max": 90, "step": 1,
         "default": 0, "unit": "deg", "digits": 0, "apply": _control_angle("pitch"),
         "description": "Tilt about the platform's left: the fingers point forward or back."},
        {"key": "yaw", "label": "Yaw (about z)", "type": "slider", "wide": True, "min": -180, "max": 180, "step": 1,
         "default": 0, "unit": "deg", "digits": 0, "apply": _control_angle("yaw"),
         "description": "Turn about the vertical: line the fingers up with a box's sides."},
        {"key": "repair", "label": "Repair the gripper", "type": "button", "apply": _control_repair,
         "description": "Straightens every finger and puts the arm back on its last solution, at rest: for a finger "
                        "jammed by a hard push."},
    ],
    manipulator={
        "position": ["tcp_x", "tcp_y", "tcp_z"], "scale": 1e-3,
        "rotation": ["roll", "pitch", "yaw"],
        "frame": {"x": "base_x", "y": "base_y", "yaw": "base_yaw", "scale": 1e-3},
        "drive": {"forward": "drive_forward", "sideways": "drive_sideways", "turn": "drive_turn",
                  "hint": "up / down forward and back, left / right sideways (omniwheels) or turn / steer (the other "
                          "carts); ⟲ ⟳ turn it"},
        "pick": {"models": r"(^|/)items/[^/]+/(visu|detail\d+)/visual$", "x": "pick_x", "y": "pick_y", "z": "pick_z",
                 "go": "pick_go", "title": "<b>Click an item</b>: the robot puts it in its place by itself (<b>Stock "
                                           "all</b>: everything in its crate)."},
        "reach": "ik_ok",
        "grip": {"key": "pressure", "on": 80, "off": 0, "label": "Grip", "releaseLabel": "Let go"},
        "models": r"(^|/)bodies/(visu|base_layer|tips|joint_marks|trunk|rail|jaws|jaw_pads|cup|cup_stem|finray_[FBR]|mia_fingers|mia_joints|skin/[^/]+)/visual$",
        "jogSpeed": 150, "jogTurn": 60,
        "buttons": [{"key": "stock_all", "label": "Stock all", "title": "The robot stocks everything in its crate by "
                     "itself (click an item: just that one)"},
                    {"key": "auto_stop", "label": "Stop", "title": "Stop the automatic stocking"},
                    {"key": "repair", "label": "Repair", "title": "Straighten jammed fingers: every finger straight, "
                     "the arm on its last solution, at rest"}],
    },
    display=[
        {"key": "wall_alpha", "label": "Gripper opacity", "type": "opacity",
         "models": r"(^|/)bodies/(visu|base_layer|tips|rail|jaws|jaw_pads|cup|cup_stem|finray_[FBR]|mia_fingers|mia_joints|skin/[^/]+)/visual$"},
        {"key": "joint_alpha", "label": "Hinge mark opacity", "type": "opacity", "models": r"(^|/)bodies/joint_marks/visual$"},
        {"key": "item_alpha", "label": "Item opacity", "type": "opacity",
         "models": r"(^|/)items/[^/]+/(visu|detail\d+)/visual$"},
        {"key": "arm_alpha", "label": "Arm opacity", "type": "opacity", "models": r"(^|/)robot/arm/[^/]+/visual$"},
        {"key": "platform_alpha", "label": "Platform opacity", "type": "opacity",
         "models": r"(^|/)robot/platform/[^/]+/visual$"},
        {"key": "shelf_alpha", "label": "Shelf opacity", "type": "opacity", "models": r"(^|/)store/shelves/[^/]+/visual$"},
        {"key": "spot_alpha", "label": "Empty place marks opacity", "type": "opacity", "models": r"(^|/)store/spots/[^/]+/visual$"},
        {"key": "floor_alpha", "label": "Floor opacity", "type": "opacity", "models": r"^/floor/[^/]+/visual$"},
        {"key": "mount_alpha", "label": "Gripper plate opacity", "type": "opacity", "models": r"(^|/)trunk/visual$"},
        {"key": "axes_alpha", "label": "Coordinate axes (X, Y) opacity", "type": "opacity",
         "models": r"^/arrow_[xy]/visual$"},
    ],
    presets=PRESETS,
    live_params=LIVE,
    probes=_Probes(),
    charts=[
        {"title": "Stocking", "unit": "items",
         "series": [{"key": "stocked", "label": "stocked from the crate", "color": "#1baf7a"},
                    {"key": "in_crates", "label": "still in the crate", "color": "#b86b16"},
                    {"key": "displaced", "label": "shelf items out of place", "color": "#d6402a"},
                    {"key": "on_floor", "label": "on the floor", "color": "#8f97a3"}]},
        {"title": "Platform position", "unit": "mm",
         "series": [{"key": "base_x", "label": "x", "color": "#d6402a"},
                    {"key": "base_y", "label": "y", "color": "#1baf7a"}]},
        {"title": "Tool point (platform frame)", "unit": "mm",
         "series": [{"key": "tcp_x", "label": "x", "color": "#d6402a"},
                    {"key": "tcp_y", "label": "y", "color": "#1baf7a"},
                    {"key": "tcp_z", "label": "z", "color": "#2a78d6"}]},
        {"title": "Arm joints", "unit": "deg",
         "series": [{"key": f"q{i}", "label": label, "color": color} for i, label, color in
                    ((1, "base", "#d6402a"), (2, "shoulder", "#e0912f"), (3, "elbow", "#1baf7a"),
                     (4, "joint 4", "#2a78d6"), (5, "joint 5", "#7a4fd6"), (6, "joint 6", "#8f97a3"),
                     (7, "joint 7", "#c0398f"))]},
        {"title": "Tactile skin", "unit": "N",
         "series": [{"key": "touch_total", "label": "all pads", "color": "#e0912f"},
                    {"key": "touch_max", "label": "the most on one pad", "color": "#d6402a"}]},
        {"title": "Grip", "unit": "kPa",
         "series": [{"key": "target", "label": "target", "color": "#8f97a3"},
                    {"key": "applied", "label": "applied", "color": "#2a78d6"}]},
    ],
    readouts=[
        {"key": "auto_left", "label": "automatic: items to go", "digits": 0, "unit": ""},
        {"key": "auto_done", "label": "automatic: stocked", "digits": 0, "unit": ""},
        {"key": "auto_failed", "label": "automatic: gave up", "digits": 0, "unit": ""},
        {"key": "stocked", "label": "stocked from the crate", "digits": 0, "unit": ""},
        {"key": "in_crates", "label": "still in the crate", "digits": 0, "unit": ""},
        {"key": "displaced", "label": "shelf items out of place", "digits": 0, "unit": ""},
        {"key": "on_floor", "label": "items on the floor", "digits": 0, "unit": ""},
        {"key": "ik_ok", "label": "target reachable (1 yes, 0 no)", "digits": 0, "unit": ""},
        {"key": "base_x", "label": "platform x", "digits": 0, "unit": "mm"},
        {"key": "base_y", "label": "platform y", "digits": 0, "unit": "mm"},
        {"key": "base_yaw", "label": "platform heading", "digits": 0, "unit": "deg"},
        {"key": "drive_v", "label": "platform speed", "digits": 2, "unit": "m/s"},
        {"key": "tcp_x", "label": "tool x (platform)", "digits": 0, "unit": "mm"},
        {"key": "tcp_y", "label": "tool y (platform)", "digits": 0, "unit": "mm"},
        {"key": "tcp_z", "label": "tool z", "digits": 0, "unit": "mm"},
        {"key": "applied", "label": "grip", "digits": 1, "unit": "kPa"},
        {"key": "opening", "label": "jaw opening", "digits": 0, "unit": "mm"},
        {"key": "touch_total", "label": "tactile skin: force", "digits": 1, "unit": "N"},
        {"key": "seal", "label": "vacuum cup: lip sealed", "digits": 0, "unit": "%"},
        {"key": "cup_force", "label": "vacuum cup: pull", "digits": 1, "unit": "N"},
        {"key": "contacts", "label": "contacts", "digits": 0, "unit": ""},
    ],
    stop_at=None,  # interactive: it keeps running
    about=[
        "A robot shop assistant: a gripper on a robot arm on a cart that carries a crate of groceries, in front of two "
        "shelf units with empty places marked in each product's colour. Choose the robot in the Parameters panel "
        "(Robot): the gripper (Gripper 5's soft PneuNet fingers, rigid parallel jaws, a rigid three-jaw chuck or a "
        "suction cup), the arm (UR-like with six axes, a KUKA LBR iiwa 14 or a Franka Emika Panda, both seven axes, "
        "in their real sizes) and the cart (omniwheels that move any way; two-wheel differential drive and tank "
        "tracks that turn on the spot but can't move sideways; a car that steers its front wheels and turns only while "
        "moving). The presets have a few combinations.",
        "CloSer partners' hardware among the grippers: Prensilia's Mia hand (three motors: thumb, index, and the "
        "middle, ring and little fingers together; the thumb's rotation fixed in opposition; the finger joints "
        "approximated as two coupled phalanges), SoftGripping's SoftGripper (pneumatic silicone fingers on a centric "
        "or parallel base, opened wide by vacuum, optionally with a suction cup), GripWiq's SofTouch silicone suction "
        "cups (ST, and the larger CS: a flexible lip of sprung segments, and the vacuum times the area inside it "
        "pulling as far as the lip seals) and Festo's Fin Ray fingers (DHAS: front and back flanks joined at the "
        "tip, with ribs across, so the finger wraps round what presses it). The fingers carry a tactile skin "
        "(Touchlab / TG0 style): each pad lights up by the force on it (Display: the heat map; the Tactile skin "
        "chart). The Ocado-style preset picks groceries with a suction-assisted soft gripper. Sizes are typical ones: "
        "the makers' pages don't give them all.",
        "Automatic stocking: click an item (in the crate, or one out of place on a shelf) and the robot puts it in "
        "an empty place for its product by itself: it drives along the shelves so the arm stands beside the place, "
        "takes the item from above (from a shelf, reaching in under the next one), and puts it down there. Stock "
        "all does every item in the crate, one after the other; Stop (or anything you do: driving, moving the "
        "gripper, gripping) stops it. The Console says what it's doing, and why it gives up on an item (it tries each "
        "twice); the readouts count what it did.",
        "By hand, as in Gripper 6: drive with the joystick at the left of the view or the arrow keys (, . turn), "
        "steer the gripper by dragging it, the gizmo, the height slider, the keys WASD R F Q E, and Space to grip. "
        "The arm follows by inverse kinematics while it can reach (the target dot turns red when it can't).",
        "Gripper 7 has two shelf levels, 560 mm apart, with 45 mm between the places: a gripper pointing down, with "
        "the arm's wrist above it, fits under the next shelf, and soft fingers fit beside a neighbouring item.",
        "The model: the cart is a heavy body on x, y and yaw joints held by stiff position actuators, driven by "
        "integrating the commanded speeds within what its wheels allow; the arms have position actuators on their "
        "joints and gravity-compensated links; inverse kinematics by damped least squares on an arm-only MuJoCo model, "
        "100 times a second (a seven-axis arm also keeps near a comfortable posture). Rigid jaws are slide joints "
        "pushed with a force; the suction cup is MuJoCo's adhesion. The groceries, shelves and contacts are Gripper "
        "5's.",
    ],
    origin="Scene script: scenes/gripper-7-mujoco/original/Gripper7Mj.py (Gripper 6 with a choice of robot, and "
           "automatic stocking).",
)
