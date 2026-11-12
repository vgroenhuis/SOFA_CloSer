"""Gripper 8 on MuJoCo: a grip test line. Every gripper of Gripper 7 / Hagelslag in a row, each over its own copy of
the chosen grocery item (rigid, Hagelslag's packages, or a deformable bag of chips or potatoes), and an automatic
test: grip, lift, a full turn about x, a metre forward, put down, let go (original/Gripper8Mj.py, run through
sofaweb's MuJoCo engine)."""

import mujoco
import numpy as np

from sofaweb.config import SceneConfig

# The grippers in the line, left to right (Gripper8Mj.GRIPPER_CHOICES): (key, short label).
GRIPPERS = [
    ("pneunet", "PneuNet fingers"),
    ("softgripping_centric", "SoftGripping centric"),
    ("softgripping_parallel", "SoftGripping parallel"),
    ("softgripping_suction", "SoftGripping + cup"),
    ("gripwiq_st", "GripWiq ST cup"),
    ("gripwiq_cs", "GripWiq CS cup"),
    ("festo_finray", "Festo Fin Ray"),
    ("mia", "Prensilia Mia hand"),
    ("parallel", "Parallel jaws"),
    ("three_jaw", "Three-jaw chuck"),
    ("suction", "Simple suction cup"),
]
PHASES = ["settle", "down", "grip", "lift", "turn", "hold", "carry", "lower", "release", "rise", "done"]


class _Probes:
    """The test: its phase (0 settle .. 10 done), the turn angle, how many grippers hold their item and how many
    put it down at its goal; per gripper its result (status_<kind>: 1 holding, 2 put down at its goal, -1 dropped,
    -2 not at its goal, 0 not yet) and its grip (applied pressure); the tactile skins' forces (touch_<kind>.<pad>)."""

    def __call__(self, root, module):
        test = root.getObject("Test")
        build = module.BUILD
        data = root.data
        values = {"phase": float(test.phase_index), "roll": float(test.roll),
                  "holding": float(sum(s == 1 for s in test.status)),
                  "placed": float(sum(s == 2 for s in test.status)),
                  "dropped": float(sum(s < 0 for s in test.status))}
        for rig, s in zip(test.rigs, test.status):
            values[f"status_{rig['kind']}"] = float(s)
            values[f"grip_{rig['kind']}"] = rig["ramp"].applied_pressure() / 1000.0
            if rig.get("vacuum") is not None:
                values[f"seal_{rig['kind']}"] = 100.0 * rig["vacuum"].seal
        skin = build.get("skin_geoms") or {}
        if skin:
            per = dict.fromkeys(build["skin_pads"], 0.0)
            f6 = np.zeros(6)
            geoms = data.contact.geom[:data.ncon]
            for i in range(data.ncon):
                g1, g2 = int(geoms[i, 0]), int(geoms[i, 1])
                pad = skin.get(g1) if g2 not in skin else (skin.get(g2) if g1 not in skin else None)
                if pad is None:
                    continue
                mujoco.mj_contactForce(root.model, data, i, f6)
                per[pad] += abs(float(f6[0]))
            for key, v in per.items():
                values[f"touch_{key}"] = v
            values["touch_total"] = sum(per.values())
        values["contacts"] = float(data.ncon)
        return values


_DEFAULT = {
    "item": "g7_egg", "spacing": 400.0, "lift_height": 450.0, "travel": 1000.0, "move_speed": 200.0,
    "turn_speed": 60.0, "grip_pressure": 80.0, "grip_wait": 1.0, "release_wait": 1.5, "chips_length": 220.0,
    "chips_width": 150.0, "chips_thickness": 90.0, "chips_mass": 150.0, "chips_roundness": 0.6, "chips_stiffness": 100.0,
    "chips_poisson": 0.47, "potato_count": 6, "potato_radius": 28.0, "potato_mass": 100.0, "net_slack": 15.0,
    "item_mass_scale": 1.0, "jaw_stroke": 55.0, "jaw_length": 70.0, "jaw_width": 22.0, "jaw_force": 40.0,
    "suction_force": 30.0, "finray_length": 80.0, "finray_base": 30.0, "finray_stiffness": 0.12, "mia_torque": 1.0,
    "tactile_skin": True, "cup_vacuum": 30.0, "cup_lip_stiffness": 150.0, "sg_spacing": 45.0,
    "sg_open_vacuum": 20.0, "fingers": 4, "finger_radius": 45.0, "finger_azimuth": 0.0, "segments": 6,
    "length": 8.0, "width": 20.0, "height": 16.0, "wall_end": 1.5, "wall_side": 1.5, "wall_bottom": 3.0,
    "wall_top": 1.5, "channel_separation": 2.0, "tips_enabled": True, "tip_length": 14.0, "tip_width": 3.0,
    "tip_thickness": 2.0, "joint_stiffness": 0.34, "joint_damping": 0.01, "joint_open_limit": 90.0,
    "side_stiffness": 5.0, "side_limit": 20.0, "joint_armature": 2e-06, "limit_timeconst": 0.002,
    "pressure_torque_gain": 1.0, "tcp_depth": 50.0, "flange_mass": 0.5, "friction": 0.8,
    "torsional_friction": 0.005, "rolling_friction": 0.002, "impratio": 10.0, "noslip_iterations": 10,
    "finger_collisions": True, "density": 1000.0, "gravity_enabled": True, "ramp_start": 0.0, "ramp_time": 0.4,
    "dt": 0.02, "timestep": 0.001, "instant": False,
}

PRESETS = {
    "params": [
        {"name": "Default",
         "description": "Every gripper over Gripper 7's egg, lying on its side (60 g, round and smooth), the test at "
                        "its usual speeds.",
         "values": dict(_DEFAULT)},
        {"name": "Bag of chips 150 g (deformable)",
         "description": "A 150 g bag of chips lying flat: a closed, well-filled bag that keeps its volume, squashed "
                        "where it's gripped and bulging elsewhere. Wider than any gripper's jaws: they pinch its top; "
                        "suction cups do best. Several times slower than real time.",
         "values": {**_DEFAULT, "item": "chips"}},
        {"name": "Bag of chips 25 g (deformable)",
         "description": "A small 25 g bag of chips (130 x 85 mm): narrow enough for the jaws and fingers to grasp "
                        "it across. Slower than real time.",
         "values": {**_DEFAULT, "item": "chips_25"}},
        {"name": "Mini bag of chips 15 g (deformable)",
         "description": "A 15 g mini bag of chips (105 x 70 mm), as from a multipack. Slower than real time.",
         "values": {**_DEFAULT, "item": "chips_15"}},
        {"name": "Bag of potatoes (deformable)",
         "description": "Six round potatoes (600 g) in a net bag: they shift inside it as it's picked up; a suction "
                        "cup on the net leaks.",
         "values": {**_DEFAULT, "item": "potatoes"}},
        {"name": "Soda can",
         "description": "Gripper 7's soda can (380 g), standing.",
         "values": {**_DEFAULT, "item": "g7_can"}},
        {"name": "Banana",
         "description": "Gripper 7's banana: long and curved, taken by the middle of its curve.",
         "values": {**_DEFAULT, "item": "g7_banana"}},
        {"name": "Nutella 750 g",
         "description": "The Hagelslag shelf's biggest Nutella jar (glass): heavy and smooth.",
         "values": {**_DEFAULT, "item": "hs_nut_750"}},
        {"name": "Fast turn",
         "description": "The turn about x three times as fast (180 deg/s): the item's inertia pulls at the grip.",
         "values": {"turn_speed": 180.0}},
        {"name": "Slippery (friction 0.4)",
         "description": "Half the friction: items slip out unless the fingers curl round them.",
         "values": {"friction": 0.4}},
        {"name": "Gentle grip (40 kPa)",
         "description": "Half the grip: the soft fingers bend less, the jaws squeeze with half their force, the cups "
                        "pull with half the vacuum.",
         "values": {"grip_pressure": 40.0}},
    ],
    "display": [
        {"name": "See-through items",
         "description": "Translucent items, to see how the fingers hold them.",
         "values": {"item_alpha": 0.4, "item_alpha:on": True}},
    ],
    "lighting": [
        {"name": "Soft",
         "description": "More ambient light, softer key light.",
         "values": {"light:ambient:brightness": 0.6, "light:0:brightness": 0.7, "light:1:brightness": 0.45}},
    ],
}

_LIMITS = {
    "spacing": (150.0, 1500.0), "lift_height": (150.0, 1500.0), "travel": (-3000.0, 3000.0),
    "move_speed": (20.0, 2000.0), "turn_speed": (5.0, 720.0), "grip_pressure": (0.0, 100.0),
    "grip_wait": (0.0, 10.0), "release_wait": (0.0, 10.0),
    "chips_length": (60.0, 500.0), "chips_width": (40.0, 400.0), "chips_thickness": (10.0, 200.0),
    "chips_mass": (5.0, 2000.0), "chips_roundness": (0.2, 0.95), "chips_stiffness": (0.5, 1000.0), "chips_poisson": (0.0, 0.49),
    "potato_count": (1, 12), "potato_radius": (10.0, 60.0), "potato_mass": (10.0, 1000.0), "net_slack": (0.0, 100.0),
    "fingers": (1, 8), "finger_radius": (0.0, 200.0), "finger_azimuth": (-360.0, 360.0), "segments": (2, 15),
    "length": (1.0, 100.0), "width": (2.0, 100.0), "height": (2.0, 100.0), "wall_end": (0.2, 50.0),
    "wall_side": (0.2, 50.0), "wall_bottom": (0.2, 50.0), "wall_top": (0.2, 50.0), "channel_separation": (0.1, 50.0),
    "tip_length": (0.5, 60.0), "tip_width": (0.2, 50.0), "tip_thickness": (0.2, 50.0),
    "joint_stiffness": (1e-3, 100.0), "joint_damping": (0.0, 1.0), "joint_open_limit": (5.0, 170.0),
    "side_stiffness": (0.0, 1000.0), "side_limit": (0.1, 90.0), "joint_armature": (0.0, 1e-3),
    "limit_timeconst": (0.0005, 0.5), "pressure_torque_gain": (0.0, 10.0), "tcp_depth": (0.0, 200.0),
    "flange_mass": (0.01, 5.0), "item_mass_scale": (0.05, 20.0), "friction": (0.0, 3.0),
    "torsional_friction": (0.0, 0.05), "rolling_friction": (0.0, 0.05), "impratio": (1.0, 100.0),
    "noslip_iterations": (0, 100), "timestep": (0.0001, 0.005), "density": (10.0, 20000.0),
    "ramp_start": (0.0, 60.0), "ramp_time": (0.0, 10.0), "dt": (0.002, 0.03), "item_alpha": (0.0, 1.0),
    "wall_alpha": (0.0, 1.0), "joint_alpha": (0.0, 1.0),
    "jaw_stroke": (10.0, 120.0), "jaw_length": (20.0, 200.0), "jaw_width": (5.0, 80.0),
    "jaw_force": (0.0, 500.0), "suction_force": (0.0, 500.0),
    "cup_vacuum": (0.0, 95.0), "cup_lip_stiffness": (5.0, 5000.0), "sg_spacing": (0.0, 200.0),
    "sg_open_vacuum": (0.0, 80.0), "finray_length": (30.0, 200.0), "finray_base": (10.0, 80.0),
    "finray_stiffness": (0.005, 5.0), "mia_torque": (0.05, 5.0),
}

_STATUS = "1 holding, 2 put down at its goal, -1 dropped, -2 not at its goal or not let go"

CONFIG = SceneConfig(
    title="Gripper 8: Grip test line (MuJoCo)",
    script="original/Gripper8Mj.py",
    engine="mujoco",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"GRIPPER8MJ_NO_EDITOR": "1"},
    hidden_params={"theme"},
    param_limits=_LIMITS,
    heat=[{"models": r"/bodies/skin/([^/]+)/visual$", "key": "touch_{}", "max": 4.0,
           "label": "Tactile skin heat map (Touchlab / TG0 style)"}],
    sections=[
        ("Item", ["item", "item_mass_scale"]),
        ("The test", ["spacing", "lift_height", "travel", "move_speed", "turn_speed", "grip_pressure", "grip_wait",
                      "release_wait"]),
        ("Bags of chips", ["chips_length", "chips_width", "chips_thickness", "chips_mass", "chips_roundness",
                           "chips_stiffness", "chips_poisson"]),
        ("Bag of potatoes", ["potato_count", "potato_radius", "potato_mass", "net_slack"]),
        ("Rigid jaws, suction cups", ["jaw_stroke", "jaw_length", "jaw_width", "jaw_force", "suction_force",
                                      "cup_vacuum", "cup_lip_stiffness"]),
        ("SoftGripping", ["sg_spacing", "sg_open_vacuum"]),
        ("Festo Fin Ray fingers", ["finray_length", "finray_base", "finray_stiffness"]),
        ("Prensilia Mia hand", ["mia_torque"]),
        ("Tactile skin", ["tactile_skin"]),
        ("Grip ramp", ["ramp_start", "ramp_time", "instant"]),
        ("Soft fingers (PneuNet)", ["fingers", "finger_radius", "finger_azimuth", "segments", "tcp_depth",
                                    "flange_mass"]),
        ("Pointy tips", ["tips_enabled", "tip_length", "tip_width", "tip_thickness"]),
        ("Segment geometry (mm)", ["length", "width", "height", "wall_end", "wall_side", "wall_bottom", "wall_top",
                                   "channel_separation"]),
        ("Joints", ["joint_stiffness", "joint_damping", "joint_open_limit", "side_stiffness", "side_limit",
                    "joint_armature", "limit_timeconst", "pressure_torque_gain"]),
        ("Contact", ["friction", "torsional_friction", "rolling_friction", "impratio", "noslip_iterations",
                     "finger_collisions"]),
        ("Material", ["density", "gravity_enabled"]),
        ("Simulation", ["dt", "timestep"]),
    ],
    controls=[],
    display=[
        {"key": "wall_alpha", "label": "Gripper opacity", "type": "opacity",
         "models": r"(^|/)bodies/(visu|base_layer|tips|rail|jaws|jaw_pads|cup|cup_stem|finray_[FBR]|mia_fingers|mia_joints|skin/[^/]+)/visual$"},
        {"key": "joint_alpha", "label": "Hinge mark opacity", "type": "opacity", "models": r"(^|/)bodies/joint_marks/visual$"},
        {"key": "mount_alpha", "label": "Gripper plate opacity", "type": "opacity", "models": r"(^|/)trunk/visual$"},
        {"key": "item_alpha", "label": "Item opacity", "type": "opacity", "models": r"^/world/items/"},
        {"key": "plates_alpha", "label": "Name plates opacity", "type": "opacity", "models": r"^/floor/plates/"},
        {"key": "floor_alpha", "label": "Floor opacity", "type": "opacity",
         "models": r"^/floor/(visu|tiles|marks)/visual$"},
        {"key": "axes_alpha", "label": "Coordinate axes (X, Y) opacity", "type": "opacity",
         "models": r"^/arrow_[xy]/visual$"},
    ],
    presets=PRESETS,
    probes=_Probes(),
    charts=[
        {"title": "Grippers", "unit": "grippers",
         "series": [{"key": "holding", "label": "holding the item", "color": "#2a78d6"},
                    {"key": "placed", "label": "put it down at its goal", "color": "#1baf7a"},
                    {"key": "dropped", "label": "dropped it (or not at its goal)", "color": "#d6402a"}]},
        {"title": "The turn about x", "unit": "deg",
         "series": [{"key": "roll", "label": "angle", "color": "#7a4fd6"}]},
        {"title": "Tactile skin", "unit": "N",
         "series": [{"key": "touch_total", "label": "all pads, all grippers", "color": "#e0912f"}]},
    ],
    readouts=[
        {"key": "phase", "label": "phase (" + ", ".join(f"{i} {p}" for i, p in enumerate(PHASES)) + ")",
         "digits": 0, "unit": ""},
        {"key": "roll", "label": "turn about x", "digits": 0, "unit": "deg"},
        {"key": "holding", "label": "grippers holding the item", "digits": 0, "unit": ""},
        {"key": "placed", "label": "put it down at its goal", "digits": 0, "unit": ""},
        {"key": "dropped", "label": "dropped it, or not at its goal", "digits": 0, "unit": ""},
    ] + [{"key": f"status_{k}", "label": f"{label} ({_STATUS})", "digits": 0, "unit": ""} for k, label in GRIPPERS]
      + [{"key": f"seal_{k}", "label": f"{label}: lip sealed", "digits": 0, "unit": "%"}
         for k, label in GRIPPERS if k in ("softgripping_suction", "gripwiq_st", "gripwiq_cs")]
      + [{"key": "contacts", "label": "contacts", "digits": 0, "unit": ""}],
    stop_at=None,
    about=[
        "A grip test line: every gripper of Gripper 7 and Hagelslag in a row, left to right as on their name plates "
        "(Soft PneuNet fingers, the SoftGripping SoftGripper centric, parallel and with a suction cup, the GripWiq "
        "SofTouch cups ST and CS, Festo's Fin Ray fingers, the Prensilia Mia hand, rigid parallel jaws, a rigid "
        "three-jaw chuck and a simple suction cup), each over its own copy of the same item.",
        "Press play and the test runs by itself, for all of them at once: down to the item, grip, up to 45 cm, a full "
        "turn about the x axis (the item upside down, then on its side), a metre forward along y, down to the floor, "
        "let go and up. The readouts show each gripper's result (1 holding, 2 put down at its goal, -1 dropped, -2 not "
        "at its goal or not let go); the console says where each one dropped its item. Change the item in the "
        "Parameters panel (Item), or pick a preset (the bag of chips, the bag of potatoes, ...).",
        "The items: Gripper 7's groceries, every product of the Hagelslag shelf (with its photographed front), and "
        "deformable ones. The bag of chips (150 g, 40 g, 25 g or a 15 g mini bag; the 150 g one's size is set in "
        "the parameters) is a MuJoCo flex: a rounded, pillow-shaped grid of nodes split into tetrahedra, nearly "
        "incompressible (Poisson's ratio 0.47), so it keeps its volume: squeezed in one place it bulges out "
        "elsewhere. The smaller bags fit between the jaws and fingers and are grasped across; the 150 g one is "
        "pinched on its top. The bag of potatoes holds six round potatoes (spheres) in a net: slack strings between "
        "neighbouring potatoes and to the bag's clip, so they shift inside it as it's picked up. A suction cup on the "
        "net leaks: it can't lift the potatoes.",
        "Each gripper is a free body welded to a target the test moves (as Gripper 5: an exact, very stiff robot), "
        "with Gripper 7's grip controllers. How each one grips is planned as Gripper 7's stocking does: the jaws open "
        "just wider than the item (an item too wide for them, they open fully and try anyway), the cups press on the "
        "middle of its top, the parallel grippers squeeze across its narrow side.",
        "A bag of chips makes the simulation three to ten times slower than real time (eleven deformable bags, and "
        "the flex needs MuJoCo's 'discrete' integrator without its no-slip passes); the other items run about in "
        "real time.",
    ],
    origin="Scene script: scenes/gripper-8-mujoco/original/Gripper8Mj.py (Gripper 7's grippers and groceries, the "
           "Hagelslag shelf's packages).",
)
