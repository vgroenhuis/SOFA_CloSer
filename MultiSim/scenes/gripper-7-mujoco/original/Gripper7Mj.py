# -*- coding: utf-8 -*-
"""
Gripper 7 (MuJoCo): a mobile manipulator stocking shelves. Gripper 5's soft gripper (../../gripper-5-mujoco/original/
Gripper5Mj.py: rigid-segment fingers with pointy tips and sideways compliance) on the flange of a six-axis robot arm
(UR-like: base, shoulder, elbow, three wrist joints) standing on a driving platform, which also carries a crate of
groceries. In front of the two shelf units of Gripper 5, the robot drives (forward speed and turn rate) and the
gripper is steered as before; the arm follows by inverse kinematics.

  - The platform drives on omniwheels (kept simple: it moves forward, sideways and turns, all at once and
    independently): a body on two slide joints and a yaw hinge, each held by a stiff position actuator (a heavy,
    accurate vehicle); a drive controller integrates the commanded speeds (with limited accelerations) into its
    pose. When the platform is blocked (a shelf), its target stays near where it is.
  - The arm: six hinge joints with position actuators (stiff, critically damped), the links' weight compensated
    (gravcomp), collision capsules. Its tool point target is given in the platform's frame (x forward, y left, z up
    from the floor; roll, pitch, yaw with R = Rz Ry Rx, zero pointing down), moved smoothly with limited speeds and
    accelerations as in Gripper 5.
  - Inverse kinematics: damped least squares on a separate arm-only MuJoCo model, 100 times a second, from the last
    solution. A target that can't be reached (no solution within tolerance, or one that would make a joint jump)
    isn't followed: the target stops at the last reachable pose (`ik_ok` 0, the viewer's dot turns red).
  - The gripper, groceries, shelves and contacts are Gripper 5's (no-slip solver, island sleeping, ...).

Units: the scene is SI (metres, kg, s, Pa); params.json takes millimetres for lengths, as labelled.

Run with MuJoCo's viewer:   python Gripper7Mj.py       (pip install mujoco)
Headless check:             python Gripper7Mj.py --headless
Parameters: params.json next to this script (edit by hand, or with params_editor.py).
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import mujoco

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')

# The grocery products: key -> (label, shape, dimensions [mm], mass [g]). Shapes:
#   'sphere' (r), 'egg' (half-length along x, radius), 'cylinder' (r, height), 'cup' (bottom r, top r, height),
#   'box' (x, y, z sizes).
#   'apple' (r, half-height: a turned profile, dimpled at the stem and underneath; an ellipsoid to the physics),
#   'pear' (r, height: a turned profile, upright; its convex hull to the physics), 'banana' (length, r, bend: a
#   curved tube lying along x, bulging to +y; capsules along it to the physics)
PRODUCTS = {
    'apple':  ('Apple',      'apple',    (36.0, 31.0),        170.0),
    'tomato': ('Tomato',     'sphere',   (29.0,),             100.0),
    'egg':    ('Egg',        'egg',      (29.0, 22.0),         60.0),
    'can':    ('Soda can',   'cylinder', (33.0, 115.0),       380.0),
    'juice':  ('Juice box',  'box',      (40.0, 63.0, 103.0), 220.0),
    'spice':  ('Spice jar',  'cylinder', (22.0, 95.0),        120.0),
    'yogurt': ('Yogurt cup', 'cup',      (27.0, 36.0, 72.0),  150.0),
    'tea':    ('Tea box',    'box',      (65.0, 65.0, 75.0),   90.0),
    'butter': ('Butter',     'box',      (110.0, 62.0, 40.0), 250.0),
    'pear':   ('Pear',       'pear',     (34.0, 100.0),       180.0),
    'banana': ('Banana',     'banana',   (175.0, 18.0, 30.0), 120.0),
}
ROUND = {'apple', 'tomato', 'egg'}  # (rolling friction)
EGG_YAW = 90.0  # [deg] eggs lie with their long axis across the shelf (front to back)

# The shelves' stock plan: (gondola, bay, level) -> [(product, places, filled)], left to right as seen from the
# floor in front. 'left' is the unit at -x, 'right' the one at +x (both facing -y, towards the crates and the camera);
# bay 0 is the left half, bay 1 the right; level 0 is the deck on the plinth. `filled` of the `places` hold an item
# at the start (from the left); the empty places are marked on the shelf in the product's colour.
SHELF_PLAN = {  # two levels, 560 mm apart: room for a gripper pointing down (and its wrist) under the next shelf;
    # 45 mm between places: room for the soft fingers (they span about 120 mm) beside a neighbour
    ('left', 0, 0): [('apple', 3, 1), ('tomato', 2, 1)],
    ('left', 0, 1): [('can', 3, 1), ('spice', 2, 1)],
    ('left', 1, 0): [('yogurt', 2, 1), ('egg', 3, 1)],
    ('left', 1, 1): [('juice', 3, 2), ('tea', 2, 2)],
    ('right', 0, 0): [('pear', 2, 1), ('banana', 1, 0)],
    ('right', 0, 1): [('butter', 2, 2), ('spice', 2, 2)],
    ('right', 1, 0): [('tea', 2, 2), ('tomato', 3, 2)],
    ('right', 1, 1): [('can', 2, 2), ('juice', 3, 3)],
}
# No crates on the floor: the platform carries one (rows along the platform, front to back).
CRATES = []
PLATFORM_CRATE = [('banana', 1), ('pear', 1), ('apple', 2), ('tomato', 1), ('egg', 2), ('can', 2), ('spice', 1),
                  ('yogurt', 1), ('juice', 1)]

_DEFAULT_PARAMS = {
    # -- the robot: which gripper, arm and cart --
    'gripper': 'pneunet',        # 'pneunet' (soft fingers), 'parallel' (rigid jaws), 'three_jaw', 'suction'
    'arm': 'ur',                 # 'ur' (UR-like, 6 axes), 'kuka' (LBR iiwa 14), 'franka' (Panda)
    'cart': 'omni',              # 'omni' (omniwheels), 'diff' (two wheels), 'tank' (tracks), 'car' (steering)
    # -- rigid jaws and the suction cup --
    'jaw_stroke': 55.0,          # [mm] each jaw's inner face from the axis, fully open
    'jaw_length': 70.0,          # [mm] below the palm
    'jaw_width': 22.0,           # [mm]
    'jaw_force': 40.0,           # [N] each jaw's grip at 100 kPa (the grip control; the Grip button: 80)
    'suction_force': 30.0,       # [N] the cup's pull at full suction (80 kPa and more)
    'steer_limit': 35.0,         # [deg] the car's largest steering angle
    # -- the fingers --
    'fingers': 4,                # how many, evenly spaced around the end-effector's axis
    'finger_radius': 45.0,       # [mm] from the axis to each finger's inner face (its base layer)
    'finger_azimuth': 0.0,       # [deg] where finger 1 hangs (0: +x, 90: +y)
    # -- each finger's segments [mm] --
    'segments': 6,               # rigid segments (chambers) in each finger
    'length': 8.0,               # one segment, along the finger
    'width': 20.0,               # across it
    'height': 16.0,              # from the base layer to the roof
    'wall_end': 1.5,             # end walls (they set the cavity's cross-section and the mass)
    'wall_side': 1.5,
    'wall_bottom': 3.0,
    'wall_top': 1.5,
    'channel_separation': 2.0,   # gap between neighbouring segments, at the base
    # -- the pointy tips --
    'tips_enabled': True,
    'tip_length': 14.0,          # [mm] beyond the last segment
    'tip_width': 3.0,            # [mm] the point, across the finger
    'tip_thickness': 2.0,        # [mm] the point, from the finger's inner face
    # -- joints --
    'joint_stiffness': 0.34,     # [N m/rad] torsional stiffness of each hinge
    'joint_damping': 0.01,       # [s] stiffness-proportional damping of the joints
    'joint_open_limit': 90.0,    # [deg] largest bend of a hinge (a hard stop beyond it)
    'side_stiffness': 5.0,       # sideways bending of each joint: its stiffness, times joint_stiffness (0: rigid
                                 #   sideways, as in Gripper 4); a silicone finger is stiffer across than along
    'side_limit': 20.0,          # [deg] largest sideways bend of a joint
    'joint_armature': 2e-6,      # [kg m^2] added inertia of each hinge (MuJoCo: keeps the light segments stable)
    'limit_timeconst': 0.002,    # [s] the hinges' limits (MuJoCo solref): small = hard stops; soft ones give way
                                 #   when the gripper is pushed into the floor or a wall
    'pressure_torque_gain': 1.0, # 1: the torque of the pressure on the cavity's cross-section
    # -- the end-effector's motion --
    'tcp_depth': 50.0,           # [mm] the tool point below the palm (the point that moves and turns)
    # -- the arm [mm], UR-like (at zero all links point up; the tool points along the shoulder axis) --
    'arm_mount_x': 250.0,        # where it stands on the platform (UR-like arm): forward of the platform's centre
    'arm_mount_y': 120.0,        #   and to the left (towards the shelves at the start)
    'arm_base_height': 160.0,    # base joint to shoulder joint
    'arm_shoulder_offset': 135.0,  # sideways
    'arm_upper': 450.0,          # shoulder to elbow
    'arm_elbow_offset': 120.0,   # sideways, back
    'arm_fore': 400.0,           # elbow to wrist 1
    'arm_wrist1': 100.0,         # wrist 1 to wrist 2 (sideways)
    'arm_wrist2': 100.0,         # wrist 2 to wrist 3 (up)
    'arm_wrist3': 100.0,         # wrist 3 to the flange (sideways, the tool axis)
    'arm_gain': 3000.0,          # [N m/rad] position gain of the base, shoulder and elbow (wrist: a tenth)
    'ik_rate': 100.0,            # [Hz] inverse kinematics solves per second
    'ik_max_jump': 30.0,         # [deg] a solution that moves a joint more than this at once is refused (a flip)
    # -- the platform [mm] --
    'platform_length': 1000.0,
    'platform_width': 700.0,
    'platform_height': 250.0,    # its deck
    'platform_mass': 60.0,       # [kg]
    'start_x': -600.0,           # where its centre starts (heading +x, along the shelves)
    'start_y': -400.0,
    'drive_speed': 0.5,          # [m/s] largest speed, any direction (omniwheels; the joystick's full stroke)
    'drive_acceleration': 0.6,   # [m/s2]
    'turn_speed': 60.0,          # [deg/s] largest turn rate
    'turn_acceleration': 180.0,  # [deg/s2]
    'flange_acceleration': 1.5,  # [m/s2] largest acceleration of the target
    'flange_speed': 0.4,         # [m/s] its largest speed
    'angular_acceleration': 360.0,  # [deg/s2]
    'angular_speed': 90.0,       # [deg/s]
    'flange_mass': 0.5,          # [kg] the gripper's plate and wrist adapter
    # -- the store [mm] --
    'shelf_distance': 300.0,     # the shelves' front edge (y = this)
    'shelf_length': 1200.0,      # each unit, along x (two bays)
    'shelf_depth': 320.0,
    'deck_height': 100.0,        # the lowest shelf (on the plinth)
    'level_height': 560.0,       # from one shelf's top to the next
    'board_thickness': 20.0,
    'lip_height': 8.0,           # the low edge along the front of every shelf
    'second_unit': True,         # a second shelf unit, beside the first (at +x)
    'crate_length': 440.0,       # the platform's crate, inside, along the platform
    'crate_width': 500.0,        # inside, across it
    'crate_height': 80.0,
    'crate_wall': 6.0,
    'item_gap': 45.0,            # [mm] between neighbouring places on a shelf (room for the gripper)
    'crate_gap': 40.0,           # [mm] between items in a crate (room for the fingers)
    'item_mass_scale': 1.0,      # times each item's mass (see PRODUCTS)
    # -- contacts --
    'friction': 0.8,             # Coulomb friction coefficient of every contact
    'torsional_friction': 0.005, # [m] resistance to twisting about a contact's normal
    'rolling_friction': 0.002,   # [m] round items: resistance to rolling
    'impratio': 10.0,            # MuJoCo: ratio of frictional to normal constraint impedance (higher: less slip)
    'noslip_iterations': 10,     # MuJoCo's no-slip solver passes: without them a held item creeps out (~1 mm/s)
    'finger_collisions': True,   # the fingers touch each other
    'sleep': True,               # resting bodies sleep (MuJoCo islands): cheaper, they wake when touched
    # -- material --
    'density': 1000.0,           # [kg/m^3] of the silicone (the segment's mass: the walls' volume)
    'gravity_enabled': True,
    # -- pressure --
    'ramp_start': 0.0,           # [s] when the pressure starts to rise
    'ramp_time': 0.4,            # [s]
    'dt': 0.02,                  # [s] one frame step (many MuJoCo steps)
    'timestep': 0.001,           # [s] MuJoCo's time step
    'instant': False,
    # -- visuals --
    'wall_alpha': 1.0,
    'joint_alpha': 1.0,
    'item_alpha': 1.0,
    'spot_alpha': 0.45,
    'show_joint_marks': True,
    'show_spots': True,          # the empty places on the shelves, marked in the product's colour
    'theme': 'System',
}


def load_params(path=PARAMS_FILE):
    params = dict(_DEFAULT_PARAMS)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                loaded = json.load(f)
            params.update({k: v for k, v in loaded.items() if k in params})
        except (OSError, ValueError) as e:
            print(f"[params] could not read {path} ({e}); using defaults")
    else:
        try:
            with open(path, 'w') as f:
                json.dump(params, f, indent=2)
        except OSError as e:
            print(f"[params] could not create {path} ({e}); using defaults")
    return params


PARAMS = load_params()

_EDITOR_PID_FILE = os.path.join(_SCRIPT_DIR, '.params_editor.pid')

def _pid_is_running(pid):
    try:
        if os.name == 'nt':
            output = subprocess.check_output(['tasklist', '/FI', f'PID eq {pid}'],
                                             stderr=subprocess.DEVNULL, text=True)
            return str(pid) in output
        os.kill(pid, 0)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def launch_params_editor():
    """Opens params_editor.py in its own process. Skipped when GRIPPER7MJ_NO_EDITOR is set (headless runs)."""
    if os.environ.get('GRIPPER7MJ_NO_EDITOR'):
        return
    if os.path.exists(_EDITOR_PID_FILE):
        try:
            with open(_EDITOR_PID_FILE, 'r') as f:
                if _pid_is_running(int(f.read().strip())):
                    return
        except (OSError, ValueError):
            pass
    try:
        process = subprocess.Popen(['python', os.path.join(_SCRIPT_DIR, 'params_editor.py')],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, close_fds=True)
        with open(_EDITOR_PID_FILE, 'w') as f:
            f.write(str(process.pid))
    except OSError as e:
        print(f"[params] could not launch params_editor.py: {e}")

MM = 1e-3
DT = PARAMS['dt']
TCP_START = (-150.0, 0.0, 620.0)  # the tool point's x, y, z [mm] at the start, in the platform's frame (over the crate)

# ----------------------------------------------------------------------
# Small geometry helpers (visual-only meshes)
# ----------------------------------------------------------------------
def _look_at_quaternion(eye, target, world_up=(0.0, 0.0, 1.0)):
    """Camera orientation [x, y, z, w] looking from `eye` at `target`
    (OpenGL convention: local -Z forward, +Y up), as SOFA's camera uses."""
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def normalize(v):
        n = math.sqrt(sum(c * c for c in v))
        return (v[0] / n, v[1] / n, v[2] / n)

    forward = normalize(sub(target, eye))
    right = normalize(cross(forward, world_up))
    up = cross(right, forward)
    m00, m01, m02 = right[0], up[0], -forward[0]
    m10, m11, m12 = right[1], up[1], -forward[1]
    m20, m21, m22 = right[2], up[2], -forward[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]


def _box(xmin, ymin, zmin, xmax, ymax, zmax, skip=()):
    """Closed axis-aligned box (8 points, outward triangles); `skip` names
    faces to leave out ('+X', '-X', ...)."""
    pts = [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
           [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]]
    faces = {'-Z': (0, 3, 2, 1), '+Z': (4, 5, 6, 7), '-Y': (0, 1, 5, 4),
             '+Y': (3, 7, 6, 2), '-X': (0, 4, 7, 3), '+X': (1, 2, 6, 5)}
    tris = []
    for name, (a, b, c, d) in faces.items():
        if name not in skip:
            tris += [[a, b, c], [a, c, d]]
    return pts, tris


def _flat_shaded(points, tris):
    """Unshared vertices per triangle, so normals come out flat."""
    flat_points, flat_tris = [], []
    for tri in tris:
        base = len(flat_points)
        flat_points += [points[i] for i in tri]
        flat_tris.append([base, base + 1, base + 2])
    return flat_points, flat_tris


def _smooth_shaded(points, tris, crease_deg=50.0):
    """Shared vertices where the surface is meant to be curved, so the browser shades it smoothly: the triangles
    around a vertex share it while their normals are within `crease_deg` of each other (a cylinder's side, a
    sphere); across a sharper edge (a cylinder's rim, a box's edges) each side gets its own copy, so those edges stay
    crisp."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    cos_crease = math.cos(math.radians(crease_deg))
    groups = {}  # vertex -> [(summed normal, new index)]
    out_pts, out_tris = [], []
    for f, tri in enumerate(T):
        new_tri = []
        for v in tri:
            for g in groups.setdefault(int(v), []):
                if np.dot(g[0], n[f]) >= cos_crease * np.linalg.norm(g[0]):
                    g[0] += n[f]
                    new_tri.append(g[1])
                    break
            else:
                groups[int(v)].append([n[f].copy(), len(out_pts)])
                new_tri.append(len(out_pts))
                out_pts.append([float(c) for c in P[v]])
        out_tris.append(new_tri)
    return out_pts, out_tris


def _merge(parts):
    pts, tris = [], []
    for p, t in parts:
        off = len(pts)
        pts += p
        tris += [[i + off for i in tri] for tri in t]
    return pts, tris


def _axis_arrow(axis, shaft_len, head_len, shaft_half, head_half):
    """Arrow along +X (axis 0) or +Y (axis 1) from the origin."""
    if axis == 0:
        shaft = _box(0.0, -shaft_half, -shaft_half, shaft_len, shaft_half, shaft_half, skip={'+X'})
        base = [[shaft_len, -head_half, -head_half], [shaft_len, head_half, -head_half],
                [shaft_len, head_half, head_half], [shaft_len, -head_half, head_half]]
        tip = [shaft_len + head_len, 0.0, 0.0]
    else:
        shaft = _box(-shaft_half, 0.0, -shaft_half, shaft_half, shaft_len, shaft_half, skip={'+Y'})
        base = [[-head_half, shaft_len, -head_half], [-head_half, shaft_len, head_half],
                [head_half, shaft_len, head_half], [head_half, shaft_len, -head_half]]
        tip = [0.0, shaft_len + head_len, 0.0]
    head = (base + [tip], [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 2, 1], [0, 3, 2]])
    return _merge([shaft, head])


def _stroke(x0, y0, x1, y1, half_width, z0, z1):
    """A straight letter stroke: a prism from (x0, y0) to (x1, y1)."""
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    nx, ny = -dy / length * half_width, dx / length * half_width
    unit, tris = _box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    pts = [[x0 + u * dx + (2 * v - 1) * nx, y0 + u * dy + (2 * v - 1) * ny, z0 + w * (z1 - z0)]
           for u, v, w in unit]
    return pts, tris


def _letter(strokes, half_width, z0, z1):
    return _merge([_stroke(*s, half_width, z0, z1) for s in strokes])


def _ellipsoid(c, radii, n_lat=12, n_lon=24):
    """A closed ellipsoid (UV sphere scaled by `radii` along x, y, z) centred at c."""
    pts = [[c[0], c[1], c[2] - radii[2]]]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat - math.pi / 2
        for k in range(n_lon):
            ph = 2 * math.pi * k / n_lon
            pts.append([c[0] + radii[0] * math.cos(th) * math.cos(ph), c[1] + radii[1] * math.cos(th) * math.sin(ph),
                        c[2] + radii[2] * math.sin(th)])
    pts.append([c[0], c[1], c[2] + radii[2]])
    top = len(pts) - 1
    tris = []
    for k in range(n_lon):
        k2 = (k + 1) % n_lon
        tris.append([0, 1 + k2, 1 + k])
    for i in range(n_lat - 2):
        r0, r1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
        for k in range(n_lon):
            k2 = (k + 1) % n_lon
            tris += [[r0 + k, r0 + k2, r1 + k2], [r0 + k, r1 + k2, r1 + k]]
    last = 1 + (n_lat - 2) * n_lon
    for k in range(n_lon):
        k2 = (k + 1) % n_lon
        tris.append([last + k, last + k2, top])
    return pts, tris


def _frustum_z(cx, cy, z0, z1, r0, r1, n=40, caps=(True, True)):
    """A closed truncated cone (a cylinder if r0 == r1) with a vertical axis at (cx, cy), radius r0 at z0, r1 at z1."""
    pts, tris = [], []
    for k in range(n):
        a = 2 * math.pi * k / n
        pts.append([cx + r0 * math.cos(a), cy + r0 * math.sin(a), z0])
        pts.append([cx + r1 * math.cos(a), cy + r1 * math.sin(a), z1])
    for k in range(n):
        k2 = (k + 1) % n
        a, b, c, d = 2 * k, 2 * k2, 2 * k2 + 1, 2 * k + 1
        tris += [[a, b, c], [a, c, d]]
    if caps[0]:
        base = len(pts)
        pts.append([cx, cy, z0])
        tris += [[base, 2 * ((k + 1) % n), 2 * k] for k in range(n)]
    if caps[1]:
        base = len(pts)
        pts.append([cx, cy, z1])
        tris += [[base, 2 * k + 1, 2 * ((k + 1) % n) + 1] for k in range(n)]
    return pts, tris


def _cylinder_along(p0, p1, r, n=16):
    """A closed cylinder from p0 to p1 (any direction)."""
    axis = np.array(p1, dtype=float) - np.array(p0, dtype=float)
    length = float(np.linalg.norm(axis))
    w = axis / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(w, helper)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    pts, tris = _frustum_z(0.0, 0.0, 0.0, length, r, r, n=n)
    out = [list(np.array(p0) + p[0] * u + p[1] * v + p[2] * w) for p in pts]
    return [[float(c) for c in p] for p in out], tris


class FingerFrame:
    """Where finger k (of n) hangs. Its own frame: x along the finger from its proximal end (down), z out of its
    roof (outwards, away from the axis), the base (the side it curls towards) at z = 0, y across the finger
    (tangential). Its hinges turn about y; a positive angle curls it towards the axis."""

    def __init__(self, k, n, radius, azimuth_deg, z_palm):
        self.k = k
        phi = math.radians(azimuth_deg + 360.0 * k / n)
        self.radial = np.array([math.cos(phi), math.sin(phi), 0.0])
        self.tangent = np.array([-math.sin(phi), math.cos(phi), 0.0])  # z x radial: curls the tip inwards
        self.down = np.array([0.0, 0.0, -1.0])
        self.origin = radius * self.radial + np.array([0.0, 0.0, z_palm])

    def to_world(self, p):
        w = self.origin + p[0] * self.down + p[1] * self.tangent + p[2] * self.radial
        return [float(c) for c in w]


def _spline(ctrl, k):
    """Catmull-Rom through the points `ctrl` [(a, b)], `k` steps per span."""
    c = [ctrl[0]] + list(ctrl) + [ctrl[-1]]
    out = []
    for i in range(1, len(c) - 2):
        p0, p1, p2, p3 = (np.array(c[i + d], float) for d in (-1, 0, 1, 2))
        for step in range(k):
            t = step / k
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    out.append(np.array(ctrl[-1], float))
    return out


def _revolve(ctrl, n, k, radius, height, z0=0.0, shape=None):
    """A closed surface turned about z from the profile `ctrl` [(r, z)] (from the bottom pole to the top one, r = 0
    there; r times `radius`, z times `height`, plus z0), `n` around, `k` rows per span; `shape(r, z, phi)` -> (r,
    x shift) varies it around (lobes) and along (a lean)."""
    prof = _spline(ctrl, k)
    pts, tris = [], []
    rings = []
    for idx, (r, z) in enumerate(prof):
        zz = z0 + z * height
        if idx in (0, len(prof) - 1):
            rr, dx = shape(0.0, z, 0.0) if shape else (0.0, 0.0)
            rings.append([len(pts)])
            pts.append([dx, 0.0, zz])
            continue
        ring = []
        for kk in range(n):
            phi = 2 * math.pi * kk / n
            rr, dx = shape(r, z, phi) if shape else (r, 0.0)
            ring.append(len(pts))
            pts.append([dx + rr * radius * math.cos(phi), rr * radius * math.sin(phi), zz])
        rings.append(ring)
    for a, b in zip(rings[:-1], rings[1:]):
        if len(a) == 1:
            tris += [[a[0], b[(kk + 1) % n], b[kk]] for kk in range(n)]
        elif len(b) == 1:
            tris += [[a[kk], a[(kk + 1) % n], b[0]] for kk in range(n)]
        else:
            for kk in range(n):
                k2 = (kk + 1) % n
                tris += [[a[kk], a[k2], b[k2]], [a[kk], b[k2], b[kk]]]
    return [[float(c) for c in q] for q in pts], tris


# Profiles (r, z), bottom pole to top pole. An apple: widest a little above the middle, a shallow eye underneath and
# a deep stem cavity on top. A pear: a round bottom and a long tapering neck.
APPLE_PROFILE = [(0.0, -0.9), (0.18, -0.97), (0.38, -1.0), (0.6, -0.93), (0.8, -0.72), (0.95, -0.4), (1.0, -0.02),
                 (0.98, 0.32), (0.9, 0.62), (0.74, 0.87), (0.52, 1.0), (0.32, 0.98), (0.16, 0.86), (0.0, 0.72)]
PEAR_PROFILE = [(0.0, 0.03), (0.3, 0.0), (0.62, 0.03), (0.86, 0.11), (0.99, 0.25), (0.97, 0.38), (0.83, 0.5),
                (0.64, 0.6), (0.5, 0.7), (0.43, 0.8), (0.37, 0.89), (0.25, 0.96), (0.0, 0.985)]
PEAR_LEAN = 0.1  # the neck's lean to +x at the top, times the radius


def _apple(r, hz, n=24, k=2):
    """An apple centred on its origin: five faint lobes, strongest towards the bottom."""
    def shape(rho, z, phi):
        return rho * (1 + 0.035 * math.cos(5 * phi) * (max(0.0, -z) + 0.3)), 0.0
    return _revolve(APPLE_PROFILE, n, k, r, hz, shape=shape)


def _pear(r, h, n=24, k=2):
    """A pear standing on its bottom, centred on its bounding box; the neck leans a little."""
    def shape(rho, z, phi):
        return rho, PEAR_LEAN * r * z * z
    return _revolve(PEAR_PROFILE, n, k, r, h, z0=-h / 2, shape=shape)


def _banana_axis(length, bend):
    """The banana's centre line: t (0 stalk .. 1 tip) -> (point, unit tangent) in the xy plane; an arc of chord
    `length` bulging by `bend` towards +y, centred on the origin."""
    rc = (length * length / 4 + bend * bend) / (2 * bend)
    half = math.asin(length / 2 / rc)

    def at(t):
        a = (t - 0.5) * 2 * half
        return (np.array([rc * math.sin(a), rc * math.cos(a) - rc + bend / 2, 0.0]),
                np.array([math.cos(a), -math.sin(a), 0.0]))
    return at


def _banana_radius(t):
    """The banana's radius along it (1 in the middle): a thin stalk end, a blunt tip."""
    smooth = lambda a, b, x: (lambda u: u * u * (3 - 2 * u))(min(1.0, max(0.0, (x - a) / (b - a))))
    if t < 0.5:
        return 0.3 + 0.7 * smooth(0.06, 0.32, t)
    return 0.42 + 0.58 * (1 - smooth(0.72, 0.99, t))


def _tube(axis, radius_at, t0, t1, rings, n, ridges=0):
    """A closed tube along `axis` (t -> point, tangent in xy) from t0 to t1; radius_at(t), `ridges` faint ridges."""
    pts, tris, ring_idx = [], [], []
    up = np.array([0.0, 0.0, 1.0])
    for i in range(rings):
        t = t0 + (t1 - t0) * i / (rings - 1)
        c, tan = axis(t)
        side = np.cross(up, tan)
        rr = radius_at(t)
        ring = []
        for kk in range(n):
            th = 2 * math.pi * kk / n
            rk = rr * (1 + 0.05 * math.cos(ridges * th)) if ridges else rr
            ring.append(len(pts))
            pts.append(list(c + rk * (math.cos(th) * side + math.sin(th) * up)))
        ring_idx.append(ring)
    for a, b in zip(ring_idx[:-1], ring_idx[1:]):
        for kk in range(n):
            k2 = (kk + 1) % n
            tris += [[a[kk], b[kk], b[k2]], [a[kk], b[k2], a[k2]]]
    for ring, t, flip in ((ring_idx[0], t0, True), (ring_idx[-1], t1, False)):
        centre = len(pts)
        pts.append(list(axis(t)[0]))
        for kk in range(n):
            k2 = (kk + 1) % n
            tris.append([centre, ring[k2], ring[kk]] if flip else [centre, ring[kk], ring[k2]])
    return [[float(v) for v in q] for q in pts], tris


def _banana(length, r, bend, rings=26, n=15):
    """A banana lying along x (its stalk at -x), its peel's five ridges; without the stalk and the tip."""
    axis = _banana_axis(length, bend)
    return _tube(axis, lambda t: r * _banana_radius(t), 0.07, 0.985, rings, n, ridges=5)


BANANA_CAPSULES = [0.13, 0.33, 0.5, 0.67, 0.9]  # where along it the physics' capsules meet


def _item_meshes(shape, dims, smooth):
    """An item's surface centred on its own origin (the centre of its bounding box), as [(points, triangles,
    colour key)], `smooth` for the picture, coarse for the collision model (or a list of spheres)."""
    n = 40 if smooth else 10  # (each corner of the coarse base is a contact with the floor)
    if shape == 'sphere':
        r, = dims
        return [(*_ellipsoid((0, 0, 0), (r, r, r), 16 if smooth else 8, 32 if smooth else 12), 'body')]
    if shape == 'egg':
        a, r = dims
        return [(*_ellipsoid((0, 0, 0), (a, r, r), 16 if smooth else 8, 32 if smooth else 12), 'body')]
    if shape == 'cylinder':
        r, h = dims
        return [(*_frustum_z(0, 0, -h / 2, h / 2, r, r, n=n), 'body')]
    if shape == 'cup':
        r0, r1, h = dims
        return [(*_frustum_z(0, 0, -h / 2, h / 2, r0, r1, n=n), 'body')]
    if shape == 'box':
        x, y, z = dims
        return [(*_box(-x / 2, -y / 2, -z / 2, x / 2, y / 2, z / 2), 'body')]
    if shape == 'apple':
        r, hz = dims
        return [(*(_apple(r, hz) if smooth else _ellipsoid((0, 0, 0), (r, r, hz), 8, 12)), 'body')]
    if shape == 'pear':
        r, h = dims
        return [(*(_pear(r, h) if smooth else _pear(r, h, n=12, k=1)), 'body')]
    if shape == 'banana':
        return [(*_banana(*dims), 'body')]
    raise ValueError(shape)


def _item_extent(shape, dims):
    """Half-heights (the centre's height when it stands on the floor) and the inertia per kg [m^2] about x, y, z."""
    if shape == 'sphere':
        r, = dims
        return r, [0.4 * r * r] * 3
    if shape == 'egg':
        a, r = dims
        return r, [0.4 * r * r, 0.2 * (a * a + r * r), 0.2 * (a * a + r * r)]
    if shape in ('cylinder', 'cup'):
        r, h = (dims[0], dims[1]) if shape == 'cylinder' else (0.5 * (dims[0] + dims[1]), dims[2])
        side = (3 * r * r + h * h) / 12.0
        return h / 2, [side, side, 0.5 * r * r]
    if shape == 'apple':
        r, hz = dims
        return hz, [0.2 * (r * r + hz * hz)] * 2 + [0.4 * r * r]
    if shape == 'pear':
        r, h = dims
        return h / 2, [0.2 * (r * r + 0.3 * h * h)] * 2 + [0.3 * r * r]
    if shape == 'banana':
        length, r, bend = dims
        return r, [0.5 * r * r + bend * bend / 8, length * length / 12, length * length / 12]
    x, y, z = dims
    return z / 2, [(y * y + z * z) / 12.0, (x * x + z * z) / 12.0, (x * x + y * y) / 12.0]


# The items' colours, and their details (visual only): [(points, triangles, colour)] in the item's frame (mm).
def _item_details(key, dims):
    mm = MM
    if key == 'apple':
        r, hz = dims
        return [(*_cylinder_along([0, 0, hz * 0.7], [0.08 * r, 0, hz * 1.22], 1.6 * mm, n=10), [0.36, 0.22, 0.1]),
                (*_ellipsoid((0.2 * r, 0, hz * 1.1), (0.2 * r, 0.09 * r, 0.025 * r), 6, 12), [0.25, 0.6, 0.18])]
    if key == 'pear':
        r, h = dims
        top = np.array([PEAR_LEAN * r, 0, h * 0.485])
        return [(*_cylinder_along(list(top - [0, 0, 3 * mm]), list(top + [0.25 * r, 0, 18 * mm]), 1.5 * mm, n=10),
                 [0.36, 0.25, 0.1])]
    if key == 'banana':
        length, r, bend = dims
        axis = _banana_axis(length, bend)
        stalk = _tube(axis, lambda t: r * (0.3 if t > 0.0 else 0.36), -0.02, 0.09, 4, 8)
        tip_c, tip_t = axis(0.985)
        tip = _ellipsoid(list(tip_c + 1.5 * mm * tip_t), (0.3 * r, 0.3 * r, 0.3 * r), 6, 10)
        return [(*stalk, [0.5, 0.45, 0.2]), (*tip, [0.22, 0.16, 0.1])]
    if key == 'tomato':
        r = dims[0]
        return [(*_frustum_z(0, 0, r * 0.93, r * 1.02, 0.42 * r, 0.12 * r, n=10), [0.2, 0.55, 0.15]),
                (*_cylinder_along([0, 0, r * 1.0], [0, 0, r * 1.15], 1.2 * mm, n=8), [0.2, 0.5, 0.15])]
    if key == 'can':
        r, h = dims
        return [(*_frustum_z(0, 0, h / 2 - 0.1 * mm, h / 2 + 0.6 * mm, r * 0.96, r * 0.86, n=40), [0.78, 0.8, 0.82]),
                (*_frustum_z(0, 0, -h / 2 - 0.1 * mm, -h / 2 + 4 * mm, r * 0.86, r * 1.005, n=40), [0.78, 0.8, 0.82]),
                (*_frustum_z(0, 0, -h * 0.15, h * 0.15, r * 1.012, r * 1.012, n=40, caps=(False, False)), [0.95, 0.95, 0.95])]
    if key == 'juice':
        x, y, z = dims
        # The straw in its wrapper, glued diagonally to the large face opposite the label (as sold).
        sx = x / 2 + 2.3 * mm
        return [(*_cylinder_along([sx, y * 0.3, -z * 0.4], [sx, -y * 0.2, z * 0.42], 2.2 * mm, n=10),
                 [0.95, 0.95, 0.95]),
                (*_box(-x / 2 - 0.4 * mm, -y * 0.3, -z * 0.2, -x / 2, y * 0.3, z * 0.2), [0.98, 0.85, 0.2])]
    if key == 'spice':
        r, h = dims
        return [(*_frustum_z(0, 0, h / 2 - 18 * mm, h / 2 + 0.5 * mm, r * 1.04, r * 1.04, n=40), [0.75, 0.12, 0.1]),
                (*_frustum_z(0, 0, -h * 0.25, h * 0.1, r * 1.012, r * 1.012, n=40, caps=(False, False)), [0.95, 0.92, 0.8])]
    if key == 'yogurt':
        r0, r1, h = dims
        return [(*_frustum_z(0, 0, h / 2, h / 2 + 0.8 * mm, r1 * 1.04, r1 * 1.04, n=40), [0.2, 0.42, 0.85])]
    if key == 'tea':
        x, y, z = dims
        return [(*_box(-x / 2 - 0.4 * mm, -y * 0.35, -z * 0.3, -x / 2, y * 0.35, z * 0.3), [0.95, 0.9, 0.7]),
                (*_box(-x * 0.35, -y / 2 - 0.4 * mm, -z * 0.3, x * 0.35, -y / 2, z * 0.3), [0.95, 0.9, 0.7])]
    if key == 'butter':
        x, y, z = dims
        return [(*_box(-x * 0.3, -y / 2 - 0.4 * mm, -z * 0.3, x * 0.3, -y / 2, z * 0.3), [0.2, 0.35, 0.75]),
                (*_box(-x * 0.3, -y * 0.3, z / 2, x * 0.3, y * 0.3, z / 2 + 0.4 * mm), [0.2, 0.35, 0.75])]
    return []


ITEM_COLOURS = {'pear': [0.72, 0.78, 0.24], 'banana': [0.97, 0.83, 0.2], 'apple': [0.8, 0.1, 0.1], 'tomato': [0.92, 0.25, 0.12], 'egg': [0.96, 0.9, 0.78],
                'can': [0.8, 0.08, 0.12], 'juice': [0.98, 0.55, 0.1], 'spice': [0.5, 0.28, 0.12],
                'yogurt': [0.97, 0.97, 0.95], 'tea': [0.2, 0.55, 0.32], 'butter': [0.98, 0.86, 0.35]}
# The browser's surface textures (sofaweb's textures.js) of each product's parts: the item, then its details in
# _item_details' order (the last one for any more).
ITEM_TEXTURES = {'pear': ['skin', 'cardboard'], 'banana': ['peel', 'cardboard', 'cardboard'],
                 'apple': ['skin', 'cardboard', 'skin'], 'tomato': ['skin'], 'egg': ['eggshell'],
                 'can': ['painted', 'brushed', 'brushed', 'paper'], 'juice': ['paper', 'plastic', 'paper'],
                 'spice': ['plastic', 'plastic', 'paper'], 'yogurt': ['plastic', 'foil'],
                 'tea': ['cardboard', 'paper'], 'butter': ['foil', 'paper']}
# The packages' printed labels (made-up words in the browser, one brand per product): part -> 'wrap' (around the
# item's axis), 'plane' (on a flat face) or 'box' (on every face of a box: cartons are printed all over). A detail
# part carrying a label is an overlay: drawn in front of the item.
ITEM_LABELS = {'can': {3: 'wrap'}, 'juice': {0: 'box', 2: 'plane'}, 'spice': {2: 'wrap'}, 'yogurt': {0: 'wrap'},
               'tea': {0: 'box', 1: 'plane', 2: 'plane'}, 'butter': {0: 'box', 1: 'plane', 2: 'plane'}}



def _hexahedron(corners):
    """A closed convex hexahedron from its 8 corners in _box's order (a box deformed, e.g. a tapered tip)."""
    _, tris = _box(0, 0, 0, 1, 1, 1)
    return [list(map(float, c)) for c in corners], tris


def _item_footprint(shape, dims, yaw_deg=0.0):
    """The item's size on the shelf (x, y) [m] when standing upright, turned by `yaw_deg` (0 or 90)."""
    if shape in ('sphere', 'cylinder'):
        fx = fy = 2 * dims[0]
    elif shape == 'egg':
        fx, fy = 2 * dims[0], 2 * dims[1]
    elif shape == 'cup':
        fx = fy = 2 * max(dims[0], dims[1])
    elif shape in ('apple', 'pear'):
        fx = fy = 2 * dims[0]
    elif shape == 'banana':
        fx, fy = dims[0] + 2 * dims[1] * 0.3, dims[2] + 2 * dims[1]
    else:
        fx, fy = dims[0], dims[1]
    return (fy, fx) if abs(yaw_deg) % 180 == 90 else (fx, fy)


def _quat_mul(a, b):
    """Quaternion product a b (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def euler_to_quat(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll) (radians) as a quaternion (w, x, y, z)."""
    qx = [math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]
    qy = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
    qz = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    return _quat_mul(_quat_mul(qz, qy), qx)


def matrix_to_euler(R):
    """Roll, pitch, yaw (radians) of a rotation matrix R = Rz Ry Rx."""
    R = np.asarray(R).reshape(3, 3)
    pitch = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    if abs(R[2, 0]) < 0.999999:
        return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])
    return 0.0, pitch, math.atan2(-R[0, 1], R[1, 1])  # gimbal lock: all the turn as yaw


# ----------------------------------------------------------------------
# Controllers: Python run before every MuJoCo step
# ----------------------------------------------------------------------
class FlangeController:
    """The end-effector's target (a mocap body, the tool point): moves towards the goal -- `targetX`, `targetY`,
    `targetZ` [m, world] and `roll`, `pitch`, `yaw` [rad] (the interface's controls set them) -- with a limited
    `acceleration` and `speed` per axis, and turns along the shortest rotation with a limited `angular_acceleration`
    and `angular_speed`. The weld pulls the end-effector after it."""

    def __init__(self, mocap, start, acceleration=0.8, speed=0.3, angular_acceleration=12.0, angular_speed=2.0):
        self.mocap = mocap
        self.targetX, self.targetY, self.targetZ = (float(v) for v in start)
        self.roll = self.pitch = self.yaw = 0.0
        self.acceleration, self.speed = acceleration, speed
        self.angular_acceleration, self.angular_speed = angular_acceleration, angular_speed
        self.position = [float(v) for v in start]
        self.velocity = [0.0, 0.0, 0.0]
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = 0.0

    def goal_quat(self):
        return euler_to_quat(float(self.roll), float(self.pitch), float(self.yaw))

    def __call__(self, scene, h):
        accel = max(float(self.acceleration), 1e-4)
        top = max(float(self.speed), 1e-4)
        goals = (float(self.targetX), float(self.targetY), float(self.targetZ))
        for a in range(3):
            d = goals[a] - self.position[a]
            wanted = math.copysign(min(top, math.sqrt(2.0 * accel * abs(d))), d)
            dv = max(-accel * h, min(accel * h, wanted - self.velocity[a]))
            self.velocity[a] += dv
            self.position[a] += self.velocity[a] * h
            if abs(d) < 1e-6 and abs(self.velocity[a]) <= 2 * accel * h:  # arrived
                self.position[a], self.velocity[a] = goals[a], 0.0
        # The rotation still to go: goal * conj(current), the shorter way round.
        goal = self.goal_quat()
        rel = _quat_mul(goal, self.quat * np.array([1.0, -1.0, -1.0, -1.0]))
        if rel[0] < 0:
            rel = -rel
        s = float(np.linalg.norm(rel[1:]))
        angle = 2.0 * math.atan2(s, float(rel[0]))
        alpha = max(float(self.angular_acceleration), 1e-3)
        wtop = max(float(self.angular_speed), 1e-3)
        wanted = min(wtop, math.sqrt(2.0 * alpha * angle))
        self.omega += max(-alpha * h, min(alpha * h, wanted - self.omega))
        step = min(angle, max(self.omega, 0.0) * h)
        if angle < 1e-6 or step >= angle:
            self.quat = goal / np.linalg.norm(goal)
            if angle < 1e-6:
                self.omega = 0.0
        elif step > 0:
            axis = rel[1:] / s
            turn = np.array([math.cos(step / 2), *(math.sin(step / 2) * axis)])
            self.quat = _quat_mul(turn, self.quat)
            self.quat /= np.linalg.norm(self.quat)
        if self.mocap is not None:
            scene.data.mocap_pos[self.mocap] = self.position
            scene.data.mocap_quat[self.mocap] = self.quat


class ArmIK:
    """Inverse kinematics of the arm alone (its own small MuJoCo model, the base where it stands in the platform's
    frame): damped least squares on the tool point's position and orientation, from a seed."""

    def __init__(self, model, limits, damping=0.02, tol_pos=5e-4, tol_rot=math.radians(0.3), iterations=60):
        self.model, self.data = model, mujoco.MjData(model)
        self.lo = np.array([lo for lo, _ in limits])
        self.hi = np.array([hi for _, hi in limits])
        self.site = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp'))
        mujoco.mj_kinematics(model, self.data)
        self.base = self.data.xpos[1].copy()  # (body 1: the arm's base)
        R = self.data.xmat[1].reshape(3, 3)
        self.base_yaw = math.atan2(R[1, 0], R[0, 0])  # (it may be turned on the cart)
        self.rest_bearing = 0.0  # the base joint's angle in the rest pose, less the target's bearing
        self.damping, self.tol_pos, self.tol_rot, self.iterations = damping, tol_pos, tol_rot, iterations
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))
        self.target_mat = np.zeros(9)
        # A redundant (7-axis) arm: the spare freedom (the null space) is spent staying near `rest` (its start pose,
        # set by home()), so the elbow doesn't drift towards its limits from one move to the next.
        self.rest = None
        self.rest_gain = 0.15

    def solve(self, pos, quat, seed):
        """Joint angles putting the tool point at `pos` turned by `quat` (w, x, y, z), from `seed`; None if none is
        found within tolerance."""
        m, d = self.model, self.data
        mujoco.mju_quat2Mat(self.target_mat, np.asarray(quat, dtype=np.float64))
        Rt = self.target_mat.reshape(3, 3)
        q = np.clip(np.array(seed, dtype=np.float64), self.lo, self.hi)
        rest = None
        if self.rest is not None and m.nv > 6:
            # The rest pose turned to face the target: the base joint does the turning, not the others.
            rest = self.rest.copy()
            bearing = math.atan2(pos[1] - self.base[1], pos[0] - self.base[0]) - self.base_yaw
            rest[0] = float(np.clip(q[0] + _wrap(bearing + self.rest_bearing - q[0]), self.lo[0], self.hi[0]))
        for _ in range(self.iterations):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.site]
            R = d.site_xmat[self.site].reshape(3, 3)
            e_pos = np.asarray(pos) - p
            e_rot = 0.5 * (np.cross(R[:, 0], Rt[:, 0]) + np.cross(R[:, 1], Rt[:, 1]) + np.cross(R[:, 2], Rt[:, 2]))
            if np.linalg.norm(e_pos) < self.tol_pos and np.linalg.norm(e_rot) < self.tol_rot:
                return q
            mujoco.mj_jacSite(m, d, self.jp, self.jr, self.site)
            J = np.vstack([self.jp, self.jr])
            e = np.concatenate([e_pos, e_rot])
            Jpinv = J.T @ np.linalg.inv(J @ J.T + self.damping ** 2 * np.eye(6))
            dq = Jpinv @ e
            if rest is not None:
                dq += (np.eye(m.nv) - Jpinv @ J) @ (self.rest_gain * (rest - q))
            step = np.max(np.abs(dq))
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, self.lo, self.hi)
        return None

    def home(self, pos, quat, seeds):
        """A first solution for the start pose: from the arm's own seeds facing the target (`seeds(bearing)`: elbow
        up, the wrist turned down), then from random ones."""
        bearing = math.atan2(pos[1] - self.base[1], pos[0] - self.base[0]) - self.base_yaw
        rng = np.random.default_rng(1)
        lo, hi = np.maximum(self.lo, -math.pi), np.minimum(self.hi, math.pi)
        for seed in list(seeds(bearing)) + [rng.uniform(lo, hi) for _ in range(400)]:
            q = self.solve(pos, quat, seed)
            if q is not None:
                # The same pose with each fully turning joint within half a turn of zero (an unwound wrist).
                full = (self.hi - self.lo) > 2 * math.pi + 1e-6
                q[full] = (q[full] + math.pi) % (2 * math.pi) - math.pi
                if self.model.nv > 6:
                    # The rest pose: this one, the joints drawn a little towards the middle of their ranges; its
                    # base joint will face each target as this one faces the start.
                    self.rest = q + 0.3 * (0.5 * (self.lo + self.hi) - q)
                    self.rest_bearing = _wrap(float(q[0]) - bearing)
                    q2 = self.solve(pos, quat, q)
                    q = q2 if q2 is not None else q
                return q
        return None


class ArmController:
    """The arm follows the tool point's target (`motion`, a FlangeController moving smoothly in the platform's
    frame): `rate` times a second it solves the inverse kinematics from the last solution and the joints' position
    actuators move towards it (interpolated in between). A target pose without a solution, or one that would move a
    joint by more than `max_jump` at once, isn't followed: the motion is put back to the last pose that could be
    reached (and stopped), so the gripper goes towards an unreachable goal as far as it can and waits there
    (`reachable` False)."""

    def __init__(self, motion, ik, actuators, q0, rate=100.0, max_jump=math.radians(30)):
        self.motion, self.ik, self.actuators = motion, ik, list(actuators)
        self.rate, self.max_jump = rate, max_jump
        self.q_prev = np.array(q0, dtype=np.float64)
        self.q_next = self.q_prev.copy()
        self.good = (list(motion.position), motion.quat.copy())
        self.reachable = True
        self.elapsed = 0.0
        self.period = 1.0 / max(rate, 1.0)
        # The automatic stocking lets the arm change its configuration: when the goal can't be followed to, it finds
        # a fresh solution for it (`seeds`: the arm's own start guesses) and goes there in joint space.
        self.reconfigure, self.seeds = False, None
        self._reconf = None  # (from, to, elapsed, duration)

    def _start_reconfiguration(self):
        m = self.motion
        goal = np.array([float(m.targetX), float(m.targetY), float(m.targetZ)])
        gq = m.goal_quat()
        rng = np.random.default_rng(int(1000 * abs(goal).sum()) % 2 ** 31)
        lo, hi = np.maximum(self.ik.lo, -math.pi), np.minimum(self.ik.hi, math.pi)
        bearing = math.atan2(goal[1] - self.ik.base[1], goal[0] - self.ik.base[0]) - self.ik.base_yaw
        best = None
        for seed in list(self.seeds(bearing)) + [rng.uniform(lo, hi) for _ in range(40)]:
            q = self.ik.solve(goal, gq, seed)
            if q is not None and (best is None or np.max(np.abs(q - self.q_next)) < np.max(np.abs(best - self.q_next))):
                best = q
        if best is None:
            return False
        span = float(np.max(np.abs(best - self.q_next)))
        self._reconf = (self.q_next.copy(), best, 0.0, max(0.5, span / 0.8), goal, gq)  # (0.8 rad/s)
        return True

    def __call__(self, scene, h):
        self.elapsed += h
        if self._reconf is not None:  # going to a fresh solution in joint space
            q_from, q_to, t, dur, goal, gq = self._reconf
            t += h
            f = min(1.0, t / dur)
            f = f * f * (3 - 2 * f)
            scene.data.ctrl[self.actuators] = q_from + (q_to - q_from) * f
            self._reconf = (q_from, q_to, t, dur, goal, gq)
            self.reachable = True
            if t >= dur:
                self._reconf = None
                self.q_prev = self.q_next = q_to.copy()
                m = self.motion
                m.position, m.quat = [float(c) for c in goal], np.array(gq, float)
                m.velocity, m.omega = [0.0, 0.0, 0.0], 0.0
                self.good = (list(m.position), m.quat.copy())
            return
        if self.elapsed >= self.period - 1e-9:
            self.elapsed = 0.0
            q = self.ik.solve(self.motion.position, self.motion.quat, self.q_next)
            if q is not None and np.max(np.abs(q - self.q_next)) <= self.max_jump:
                self.q_prev, self.q_next = self.q_next, q
                self.good = (list(self.motion.position), self.motion.quat.copy())
                self.reachable = True
            elif self.reconfigure and self.seeds is not None and self._start_reconfiguration():
                m = self.motion
                m.position, m.quat = list(self.good[0]), self.good[1].copy()
                m.velocity, m.omega = [0.0, 0.0, 0.0], 0.0
                return
            else:
                m = self.motion
                m.position, m.quat = list(self.good[0]), self.good[1].copy()
                m.velocity, m.omega = [0.0, 0.0, 0.0], 0.0
                self.q_prev = self.q_next.copy()
                self.reachable = False
        frac = min(1.0, self.elapsed / self.period)
        scene.data.ctrl[self.actuators] = self.q_prev + (self.q_next - self.q_prev) * frac

    def hold(self, scene, qpos):
        """Back on its current solution (Repair): the joints at it, at rest."""
        self._reconf = None
        self.q_prev = self.q_next.copy()
        scene.data.ctrl[self.actuators] = self.q_next
        scene.data.qpos[qpos] = self.q_next


class Repair:
    """Puts the gripper right (the interface's Repair button): every finger hinge straight and at rest, the arm on
    its last solution. For when a hard push has jammed a finger."""

    def __init__(self, hinge_qpos, hinge_dofs, arm, arm_qpos, arm_dofs):
        self.hinge_qpos, self.hinge_dofs = list(hinge_qpos), list(hinge_dofs)
        self.arm, self.arm_qpos, self.arm_dofs = arm, list(arm_qpos), list(arm_dofs)

    def __call__(self, scene):
        d = scene.data
        d.qpos[self.hinge_qpos] = 0.0
        d.qvel[self.hinge_dofs] = 0.0
        self.arm.hold(scene, self.arm_qpos)
        d.qvel[self.arm_dofs] = 0.0
        d.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(scene.model, d)


def pack_crate(contents, cl, cgap):
    """Items for a crate of inside length `cl` (along x): rows along x, 45 mm from the walls, centred; [{'product',
    'centre' (relative to the crate's centre), 'yaw'}]."""
    items = [p for p, n in contents for _ in range(n)]
    rows, row, row_w = [], [], 0.0
    for product in items:
        _, shape, dims, _ = PRODUCTS[product]
        yaw = EGG_YAW if product == 'egg' else 0.0
        fx, fy = _item_footprint(shape, [d * MM for d in dims], yaw)
        if row and row_w + cgap + fx > cl - 90 * MM:  # (45 mm from the walls, room for the fingers)
            rows.append(row)
            row, row_w = [], 0.0
        row.append((product, fx, fy, yaw))
        row_w += (cgap if len(row) > 1 else 0.0) + fx
    if row:
        rows.append(row)
    depths = [max(fy for _, _, fy, _ in r) for r in rows]
    total = sum(depths) + cgap * (len(rows) - 1)
    y = -total / 2
    spots = []
    for r, d in zip(rows, depths):
        width = sum(fx for _, fx, _, _ in r) + cgap * (len(r) - 1)
        x = -width / 2
        for product, fx, fy, yaw in r:
            spots.append({'product': product, 'centre': [x + fx / 2, y + d / 2], 'yaw': yaw})
            x += fx + cgap
        y += d + cgap
    return spots


class PressureRamp:
    """Drives the fingers' pressure towards `targetPressure` [Pa], ramped over `rampTime` (from 0 to the target)
    after `rampStart`, or at once (`instant`), and applies it as the torque `torque_per_pa` x p on every hinge."""

    def __init__(self, dofs, torque_per_pa, target_pressure=0.0, ramp_time=0.5, instant=False, ramp_start=0.0):
        self.dofs, self.torque_per_pa = np.asarray(dofs, dtype=np.int64), torque_per_pa
        self.targetPressure, self.rampTime, self.instant, self.rampStart = target_pressure, ramp_time, instant, ramp_start
        self.current_pressure = 0.0

    def __call__(self, scene, h):
        target = self.targetPressure if scene.data.time >= self.rampStart - 1e-9 else 0.0
        if self.instant:
            self.current_pressure = target
        else:
            ramp_time = max(self.rampTime, 1e-6)
            max_step = max((abs(target) if target != 0 else abs(self.current_pressure)) / ramp_time * h, 1e-9)
            diff = target - self.current_pressure
            self.current_pressure = target if abs(diff) <= max_step else \
                self.current_pressure + (max_step if diff > 0 else -max_step)
        scene.data.qfrc_applied[self.dofs] = self.torque_per_pa * self.current_pressure

    def applied_pressure(self):
        return self.current_pressure


# ----------------------------------------------------------------------
# The robot: arms, grippers and carts to choose from (the parameters 'arm', 'gripper', 'cart')
# ----------------------------------------------------------------------
ARM_CHOICES = [('ur', 'UR-like, 6 axes'), ('kuka', 'KUKA LBR iiwa 14, 7 axes'),
               ('franka', 'Franka Emika Panda, 7 axes')]
GRIPPER_CHOICES = [('pneunet', 'Soft PneuNet fingers (Gripper 5)'), ('parallel', 'Rigid parallel jaws'),
                   ('three_jaw', 'Rigid three-jaw chuck'), ('suction', 'Suction cup')]
CART_CHOICES = [('omni', 'Omniwheels (any direction)'), ('diff', 'Two-wheel differential drive'),
                ('tank', 'Tank tracks (skid steering)'), ('car', 'Car-like (front-wheel steering)')]


def arm_spec(P, pedestal):
    """The chosen arm, described at zero (every joint angle 0) in the platform's orientation: `joints` [(axis, offset
    from the previous joint, collision capsule (from, to) in the link, radius)], `limits`, link `masses`, position
    `gains` (times arm_gain), `inertia` (for the damping), the tool flange (`tool`: its offset in the last link, `dir`:
    where it points), `colours` (links, joint housings), `seeds(bearing)`: IK start guesses facing a target, and a
    `label`. The UR-like arm takes its sizes from the parameters; the KUKA LBR iiwa 14 R820 and the Franka Emika Panda
    have their real ones."""
    name = P['arm']
    if name == 'kuka':
        joints = [((0, 0, 1), (0, 0, pedestal), ((0, 0, 0.0), (0, 0, 0.36)), 0.07),
                  ((0, 1, 0), (0, 0, 0.36), ((0, 0, 0), (0, 0, 0.21)), 0.065),
                  ((0, 0, 1), (0, 0, 0.21), ((0, 0, 0), (0, 0, 0.21)), 0.06),
                  ((0, -1, 0), (0, 0, 0.21), ((0, 0, 0), (0, 0, 0.2)), 0.06),
                  ((0, 0, 1), (0, 0, 0.2), ((0, 0, 0), (0, 0, 0.2)), 0.055),
                  ((0, 1, 0), (0, 0, 0.2), ((0, 0, 0), (0, 0, 0.06)), 0.05),
                  ((0, 0, 1), (0, 0, 0.06), ((0, 0, 0), (0, 0, 0.05)), 0.045)]
        limits = [(-math.radians(d), math.radians(d)) for d in (170, 120, 170, 120, 170, 120, 175)]

        def seeds(b):
            return [[b + db, sh, 0.0, s4 * el, 0.0, s6 * wr, 0.0]
                    for db in (0.0, -0.3, 0.3) for sh, el, wr in ((0.5, 1.5, 1.1), (0.3, 1.9, 0.9), (0.8, 1.2, 1.1))
                    for s4 in (-1, 1) for s6 in (-1, 1)]
        return {'label': 'KUKA LBR iiwa 14 (7 axes, 820 mm reach)', 'joints': joints, 'limits': limits,
                'masses': [4.0, 4.0, 3.2, 2.7, 1.7, 1.8, 0.3], 'gains': [1, 1, 1, 1, 0.1, 0.1, 0.1],
                'inertia': [1.5, 1.5, 0.6, 0.5, 0.06, 0.05, 0.02], 'tool': (0.0, 0.0, 0.066), 'dir': (0, 0, 1),
                'colours': ([0.9, 0.9, 0.88, 1.0], [0.96, 0.45, 0.05, 1.0]), 'seeds': seeds, 'reach': 0.82,
                # At the cart's front, centred: its elbow folds only to 120 degrees, so it can't reach close to its
                # base; from there the whole crate is 0.4-0.8 m away.
                'mount': (0.30, 0.0), 'mount_yaw': math.pi, 'inner': 0.42}  # (turned round: its base joint turns only 340 deg)
    if name == 'franka':
        # Craig's modified DH parameters, at zero: the elbow (joint 4) 82.5 mm forward, the hand 88 mm forward and
        # pointing down.
        joints = [((0, 0, 1), (0, 0, pedestal), ((0, 0, 0), (0, 0, 0.333)), 0.06),
                  ((0, 1, 0), (0, 0, 0.333), ((0, 0, 0), (0, 0, 0.316)), 0.06),
                  ((0, 0, 1), (0, 0, 0.316), ((0, 0, 0), (0.0825, 0, 0)), 0.055),
                  ((0, -1, 0), (0.0825, 0, 0), ((0, 0, 0), (-0.0825, 0, 0.384)), 0.055),
                  ((0, 0, 1), (-0.0825, 0, 0.384), ((0, 0, -0.2), (0, 0, 0)), 0.05),
                  ((0, -1, 0), (0, 0, 0), ((0, 0, 0), (0.088, 0, 0)), 0.05),
                  ((0, 0, -1), (0.088, 0, 0), ((0, 0, 0), (0, 0, -0.05)), 0.045)]
        limits = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698), (-2.8973, 2.8973),
                  (-0.0175, 3.7525), (-2.8973, 2.8973)]

        def seeds(b):  # around its 'ready' pose (0, -45, 0, -135, 0, 90, 45 deg): the hand down in front
            return [[b + db, sh, 0.0, el, 0.0, wr, math.pi / 4]
                    for db in (0.0, -0.3, 0.3) for sh in (-0.8, -0.4, 0.0, 0.4) for el in (-2.4, -1.8, -1.2)
                    for wr in (1.6, 2.2, 2.8)]
        return {'label': 'Franka Emika Panda (7 axes, 855 mm reach)', 'joints': joints, 'limits': limits,
                'masses': [3.0, 2.5, 2.5, 2.5, 1.5, 1.5, 0.6], 'gains': [1, 1, 1, 1, 0.1, 0.1, 0.1],
                'inertia': [1.2, 1.2, 0.5, 0.4, 0.05, 0.04, 0.02], 'tool': (0.0, 0.0, -0.107), 'dir': (0, 0, -1),
                'colours': ([0.95, 0.95, 0.95, 1.0], [0.14, 0.14, 0.15, 1.0]), 'seeds': seeds, 'reach': 0.855,
                'mount': (0.12, 0.12), 'mount_yaw': math.pi, 'inner': 0.3, 'tool_roll': True}  # (nearer the crate, facing it)
    A = {k: max(float(P[f'arm_{k}']), 1.0) * MM for k in ('base_height', 'shoulder_offset', 'upper', 'elbow_offset',
                                                           'fore', 'wrist1', 'wrist2', 'wrist3')}
    joints = [((0, 0, 1), (0, 0, pedestal), ((0, 0, 0), (0, 0, A['base_height'])), 0.06),
              ((0, 1, 0), (0, A['shoulder_offset'], A['base_height']), ((0, 0, 0), (0, 0, A['upper'])), 0.05),
              ((0, 1, 0), (0, -A['elbow_offset'], A['upper']), ((0, 0, 0), (0, 0, A['fore'])), 0.04),
              ((0, 1, 0), (0, 0, A['fore']), ((0, 0, 0), (0, A['wrist1'], 0)), 0.04),
              ((0, 0, 1), (0, A['wrist1'], 0), ((0, 0, 0), (0, 0, A['wrist2'])), 0.04),
              ((0, 1, 0), (0, 0, A['wrist2']), ((0, 0, 0), (0, A['wrist3'], 0)), 0.04)]
    limits = [(-2 * math.pi, 2 * math.pi), (-math.pi, math.pi), (-2.8, 2.8),
              (-2 * math.pi, 2 * math.pi), (-2 * math.pi, 2 * math.pi), (-2 * math.pi, 2 * math.pi)]

    def seeds(b):
        # The base turned to face the target, the shoulder leaning towards it and the elbow up.
        return [[b + math.pi / 2 + db, sh, el, wr, w5, 0.0]
                for db in (0.0, -0.3, 0.3) for sh, el, wr in ((0.5, 1.5, 1.1), (0.3, 1.9, 0.9), (0.8, 1.2, 1.1))
                for w5 in (-math.pi / 2, math.pi / 2)]
    return {'label': f"UR-like (6 axes, {(A['upper'] + A['fore'] + A['wrist2']) / MM:.0f} mm reach)",
            'joints': joints, 'limits': limits, 'masses': [3.7, 8.4, 2.3, 1.2, 1.2, 0.25],
            'gains': [1, 1, 1, 0.1, 0.1, 0.1], 'inertia': [1.5, 1.5, 0.4, 0.05, 0.05, 0.03],
            'tool': (0.0, A['wrist3'], 0.0), 'dir': (0, 1, 0),
            'colours': ([0.82, 0.84, 0.86, 1.0], [0.16, 0.3, 0.55, 1.0]), 'seeds': seeds,
            'reach': A['upper'] + A['fore'] + A['wrist2']}


def _quat_z_to(v):
    """The rotation (w, x, y, z) taking +z to the unit vector v."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    c = float(v[2])
    if c > 1 - 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    if c < -1 + 1e-9:
        return (0.0, 1.0, 0.0, 0.0)
    axis = np.cross([0.0, 0.0, 1.0], v)
    axis /= np.linalg.norm(axis)
    half = math.acos(c) / 2
    return (math.cos(half), *(math.sin(half) * axis))


def _wrap(a, period=2 * math.pi):
    """`a` brought within half a `period` of zero."""
    return (a + period / 2) % period - period / 2


class DriveController:
    """The platform's motion: `forward`, `sideways` [m/s, in its own frame: forward and to the left] and `turn`
    [rad/s] (the interface's drive controls) become a pose integrated with limited accelerations, which the platform's
    stiff position actuators follow (`lead`: their damping's lag, velocity x kv / kp, added in front, so they don't
    trail). The cart (`kind`) sets what it can do: 'omni' moves any way and turns at once; 'diff' (two driven wheels)
    and 'tank' (tracks, turning a little slower) only go along their heading and turn on the spot, the sideways
    command turning them; 'car' steers its front wheels (`turn` / `sideways` set the steering angle, up to
    `steer_limit`) and turns only while it moves (yaw rate = speed x tan(steering) / `wheelbase`). When the platform
    is held up (it ran into a shelf), the integrated pose is kept within a few centimetres of where it is, so it
    doesn't push harder and harder."""

    def __init__(self, actuators, qpos, lead=(0.0, 0.0), speed=0.5, acceleration=0.6, turn_speed=1.0,
                 turn_acceleration=3.0, kind='omni', wheelbase=0.6, track=0.6, steer_limit=0.6, steer_actuators=()):
        self.actuators, self.qpos, self.lead = list(actuators), list(qpos), lead
        self.speed, self.acceleration = speed, acceleration
        self.turn_speed, self.turn_acceleration = turn_speed, turn_acceleration
        self.kind, self.wheelbase, self.track, self.steer_limit = kind, wheelbase, track, steer_limit
        self.steer_actuators = list(steer_actuators)
        self.forward = self.sideways = self.turn = 0.0  # commanded
        self.v = [0.0, 0.0]  # actual, in the platform's frame
        self.w = 0.0
        self.steer = 0.0  # [rad] the car's steering angle
        self.pose = [0.0, 0.0, 0.0]  # target displacement x, y and yaw (the joints' values)

    def __call__(self, scene, h):
        top_v, top_w = max(float(self.speed), 1e-3), max(float(self.turn_speed), 1e-3)
        kind = self.kind
        if kind == 'omni':
            want = [float(self.forward), float(self.sideways)]
            want_w = float(self.turn)
        else:
            want = [float(self.forward), 0.0]
            if kind == 'tank':
                top_w *= 0.75  # skid steering: slower turns
            want_w = float(self.turn) + float(self.sideways) * top_w / top_v
        n = math.hypot(*want)
        if n > top_v:
            want = [c * top_v / n for c in want]
        a, alpha = max(float(self.acceleration), 1e-3) * h, max(float(self.turn_acceleration), 1e-3) * h
        dv = [want[0] - self.v[0], want[1] - self.v[1]]
        n = math.hypot(*dv)
        if n > a:
            dv = [c * a / n for c in dv]
        self.v = [self.v[0] + dv[0], self.v[1] + dv[1]]
        if kind == 'car':
            goal = max(-1.0, min(1.0, want_w / top_w)) * self.steer_limit
            step = 1.5 * h  # [rad/s] steering rate
            self.steer += max(-step, min(step, goal - self.steer))
            self.w = self.v[0] * math.tan(self.steer) / max(self.wheelbase, 1e-3)
        else:
            want_w = max(-top_w, min(top_w, want_w))
            self.w += max(-alpha, min(alpha, want_w - self.w))
        x, y, th = self.pose
        th += self.w * h
        c, s = math.cos(th), math.sin(th)
        vx, vy = c * self.v[0] - s * self.v[1], s * self.v[0] + c * self.v[1]  # in the world
        x += vx * h
        y += vy * h
        # Not too far ahead of the platform itself.
        qx, qy, qth = (float(scene.data.qpos[i]) for i in self.qpos)
        dx, dy = x - qx, y - qy
        d = math.hypot(dx, dy)
        if d > 0.04:
            x, y = qx + dx * 0.04 / d, qy + dy * 0.04 / d
            self.v = [0.5 * self.v[0], 0.5 * self.v[1]]
        if abs(th - qth) > 0.08:
            th = qth + math.copysign(0.08, th - qth)
            self.w *= 0.5
        self.pose = [x, y, th]
        lin, ang = self.lead
        scene.data.ctrl[self.actuators] = [x + lin * vx, y + lin * vy, th + ang * self.w]
        if self.steer_actuators:  # Ackermann: the inner wheel turns more
            t = math.tan(self.steer)
            if abs(t) < 1e-6:
                left = right = 0.0
            else:
                r = self.wheelbase / t  # turning radius (signed, + to the left)
                left = math.atan(self.wheelbase / (r - self.track / 2))
                right = math.atan(self.wheelbase / (r + self.track / 2))
            scene.data.ctrl[self.steer_actuators] = [left, right]


class JawController:
    """Rigid jaws on slide joints (towards the axis): while the grip pressure (the PneuNet's control, 0-100 kPa) is
    on, each jaw is pushed inwards with a force proportional to it (`force` at 100 kPa); while it's off they're
    driven open, to `open_to` [m] from the axis (the automatic stocking opens them just wide enough)."""

    def __init__(self, ramp, dofs, qpos, stroke, force):
        self.ramp, self.dofs, self.qpos = ramp, list(dofs), list(qpos)
        self.stroke, self.force = stroke, force
        self.open_to = stroke

    def __call__(self, scene, h):
        d = scene.data
        p = self.ramp.current_pressure / 100e3
        q = d.qpos[self.qpos]
        if p > 0.02:
            d.qfrc_applied[self.dofs] = self.force * min(p, 1.5)
        else:
            goal = max(0.0, self.stroke - max(min(self.open_to, self.stroke), 0.0))
            f = 3000.0 * (goal - q) - 60.0 * d.qvel[self.dofs]
            d.qfrc_applied[self.dofs] = np.clip(f, -self.force, self.force)

    def opening(self, data):
        """The gap between the jaws' inner faces [m] (twice a jaw's distance from the axis, on average)."""
        return 2.0 * float(np.mean(self.stroke - data.qpos[self.qpos]))


class SuctionController:
    """The suction cup: MuJoCo's adhesion on the cup's contacts, from the grip pressure (the PneuNet's control:
    80 kPa or more is full suction)."""

    def __init__(self, ramp, actuator):
        self.ramp, self.actuator = ramp, actuator

    def __call__(self, scene, h):
        scene.data.ctrl[self.actuator] = max(0.0, min(1.0, self.ramp.current_pressure / 80e3))


class _Abort(Exception):
    pass


class Stocker:
    """The robot stocking shelves by itself. Each request (an item: clicked, or every item in the platform's crate) is
    a task run as a generator, a step at a time between the physics steps: choose an empty place for the product,
    drive to it (along a lane in front of the shelves, the arm beside the place), take the item (from the crate from
    above; from a shelf reaching in under the next shelf; from the floor), put it in its place and come back to the
    carrying pose above the crate. It steers the same things as the interface: the tool point's target (in the
    platform's frame), the grip pressure and the drive commands; anything the user does stops it."""

    GRIP = 80e3  # [Pa] the grip pressure it uses

    def __init__(self, motion, arm, drive, ramp, jaws, plan):
        self.motion, self.arm, self.drive, self.ramp, self.jaws = motion, arm, drive, ramp, jaws
        self.plan = plan
        self.queue, self.task, self.current = [], None, None
        self.done = self.failed = 0
        self.tries = {}  # item -> attempts that failed
        self._speeds = None  # the motion's own speeds, while it moves gently
        self.reserved = set()
        self.holding = False
        self._tick = 0
        self.data = None

    # -- requests ---------------------------------------------------------------
    @property
    def active(self):
        return self.task is not None or bool(self.queue)

    def request_point(self, data, point):
        """The item nearest a clicked point [m, world]: queued (if it isn't in its place already)."""
        items = self.plan['items']
        if not items:
            return
        p = np.asarray(point, float)
        dist = [np.linalg.norm(data.xpos[it['body']] - p) - 0.8 * it['half_h'] for it in items]
        i = int(np.argmin(dist))
        if dist[i] > 0.12:
            print('[auto] no item there')
            return
        state = item_states(data)[i]
        name = _item_name(items[i])
        if state == 'stocked':
            print(f'[auto] the {name} is in its place already')
        elif i in self.queue or i == self.current:
            print(f'[auto] the {name} is on the list already')
        else:
            self.queue.append(i)
            print(f'[auto] stock the {name}' + (f' (after {len(self.queue) - 1} more)' if len(self.queue) > 1 else ''))

    def stock_all(self, data):
        """Every item in the platform's crate, nearest the arm first."""
        states = item_states(data)
        new = [i for i, s in enumerate(states) if s == 'crate' and i not in self.queue and i != self.current]
        self.queue += new
        print(f'[auto] stock everything in the crate: {len(new)} items' if new else '[auto] the crate is empty')

    def cancel(self, why='stopped'):
        if self._speeds is not None:
            m = self.motion
            m.speed, m.acceleration, m.angular_speed, m.angular_acceleration = self._speeds
            self._speeds = None
        if not self.active:
            return
        print(f'[auto] {why}: {len(self.queue) + (self.task is not None)} items left as they are')
        self.queue.clear()
        self.task, self.current = None, None
        self.reserved.clear()
        self._stop_driving()

    # -- the loop ---------------------------------------------------------------------
    def __call__(self, scene, h):
        self.data = scene.data
        self.arm.reconfigure = self.active
        self._tick += 1
        if self._tick % 5:  # every 5 ms
            return
        if self.task is None:
            if not self.queue:
                return
            self.current = self.queue.pop(0)
            self.task = self._stock(self.current)
        try:
            next(self.task)
        except StopIteration:
            self.task, self.current = None, None
            self.reserved.clear()
            self._stop_driving()
        except _Abort as e:
            name = _item_name(self.plan['items'][self.current]) if self.current is not None else 'item'
            tries = self.tries.get(self.current, 0) + 1
            self.tries[self.current] = tries
            if tries < 2 and self.current is not None:
                print(f'[auto] the {name}: {e}; trying again later')
                self.queue.append(self.current)
            else:
                print(f'[auto] gave up on the {name}: {e}')
                self.failed += 1
            self.current = None
            self.reserved.clear()
            self._stop_driving()
            self.task = self._recover()

    # -- helpers -----------------------------------------------------------------------
    def _now(self):
        return float(self.data.time)

    def _pose(self):
        """The platform's x, y [m] and heading [rad] in the world."""
        pid = self.plan['platform_id']
        R = self.data.xmat[pid].reshape(3, 3)
        return float(self.data.xpos[pid][0]), float(self.data.xpos[pid][1]), math.atan2(R[1, 0], R[0, 0])

    def _stop_driving(self):
        self.drive.forward = self.drive.sideways = self.drive.turn = 0.0

    def _wait(self, seconds):
        end = self._now() + seconds
        while self._now() < end:
            yield

    def _tcp_world(self):
        return self.data.xpos[self.plan['flange_id']].copy()

    def _tcp_yaw_world(self):
        R = self.data.xmat[self.plan['flange_id']].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    def _move_local(self, local, yaw_local, timeout=10.0, what='move'):
        """The tool point to `local` (the platform's frame) turned by `yaw_local` (pointing down); until it's there.
        A move round the arm (more than 25 degrees about its base) goes along an arc about the base rather than
        straight, through waypoints: a straight line can pass too close to the base, inside the ring an arm can't reach
        (its elbow folds only so far; near the axis its wrist would have to flip)."""
        start = np.array(self.motion.position, float)
        goal = np.array(local, float)
        base = np.array(self.plan['mount'])
        a, b = start[:2] - base, goal[:2] - base
        ra, rb = np.linalg.norm(a), np.linalg.norm(b)
        if min(ra, rb) > 0.12:
            pa, pb = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
            dphi = _wrap(pb - pa)
            if abs(dphi) > math.radians(25):
                n = max(2, int(math.ceil(abs(dphi) / 0.3)))
                yaw0 = float(self.motion.yaw)
                dyaw = _wrap(yaw_local - yaw0)
                for k in range(1, n):
                    f = k / n
                    phi = pa + f * dphi
                    r = max(ra + f * (rb - ra), self.plan['inner'])
                    way = (base[0] + r * math.cos(phi), base[1] + r * math.sin(phi),
                           start[2] + f * (goal[2] - start[2]))
                    yield from self._go_local(way, yaw0 + f * dyaw, timeout, what, tol=0.03)
        yield from self._go_local(local, yaw_local, timeout, what)

    def _go_local(self, local, yaw_local, timeout, what, tol=None):
        m = self.motion
        m.targetX, m.targetY, m.targetZ = (float(c) for c in local)
        m.roll = m.pitch = 0.0
        m.yaw = float(_wrap(yaw_local))
        start, unreachable, last = self._now(), 0.0, self._now()
        goal = np.array(local, float)
        while True:
            now = self._now()
            if not self.arm.reachable:
                unreachable += now - last
            last = now
            if unreachable > 0.5:
                raise _Abort(f'{what}: out of the arm\'s reach')
            if now - start > timeout:
                raise _Abort(f'{what}: took too long')
            if tol is not None and np.linalg.norm(np.array(m.position) - goal) < tol:
                return  # a waypoint: on to the next without stopping
            arrived = (np.linalg.norm(np.array(m.position) - goal) < 0.002 and m.omega < 1e-3 and
                       np.linalg.norm(m.velocity) < 0.005)
            if arrived:
                # And the gripper itself there (the arm lags its target a little).
                x, y, th = self._pose()
                c, s = math.cos(th), math.sin(th)
                world = np.array([x + c * goal[0] - s * goal[1], y + s * goal[0] + c * goal[1], goal[2]])
                if np.linalg.norm(self._tcp_world() - world) < 0.006 or now - start > timeout - 1.0:
                    return
            yield

    def _reachable(self, world, yaw_world):
        """Whether the arm can put the tool point at `world` turned by `yaw_world` (from its current joints)."""
        x, y, th = self._pose()
        c, s = math.cos(th), math.sin(th)
        dx, dy = world[0] - x, world[1] - y
        local = np.array([c * dx + s * dy, -s * dx + c * dy, world[2]])
        return self.arm.ik.solve(local, euler_to_quat(0.0, 0.0, _wrap(yaw_world - th)), self.arm.q_next) is not None

    def _move(self, world, yaw_world, **kw):
        x, y, th = self._pose()
        c, s = math.cos(th), math.sin(th)
        dx, dy = world[0] - x, world[1] - y
        local = (c * dx + s * dy, -s * dx + c * dy, world[2])
        yield from self._move_local(local, yaw_world - th, **kw)

    def _carry(self):
        """The carrying pose: above the crate, pointing down, turned as it is."""
        yield from self._move_local(self.plan['carry'], self.motion.yaw, what='back to the crate')

    def _grip(self, on, open_to=None):
        if self.jaws is not None and open_to is not None:
            self.jaws.open_to = open_to
        self.ramp.targetPressure = self.GRIP if on else 0.0
        target = self.ramp.targetPressure
        if self.plan['gripper'] == 'suction':  # an item on a suction cup swings: gentler while holding one
            m = self.motion
            if on and self._speeds is None:
                self._speeds = (m.speed, m.acceleration, m.angular_speed, m.angular_acceleration)
                m.speed, m.acceleration = 0.5 * m.speed, 0.3 * m.acceleration
                m.angular_speed, m.angular_acceleration = 0.5 * m.angular_speed, 0.3 * m.angular_acceleration
            elif not on and self._speeds is not None:
                m.speed, m.acceleration, m.angular_speed, m.angular_acceleration = self._speeds
                self._speeds = None
        start = self._now()
        while abs(self.ramp.current_pressure - target) > 500.0 and self._now() - start < 3.0:
            yield
        yield from self._wait(self.plan['settle'])
        self.holding = on

    def _recover(self):
        """After a failure: what it holds back into the crate, the gripper open, in the carrying pose."""
        try:
            if self.holding:
                yield from self._carry()
            yield from self._grip(False, open_to=self.plan['stroke'])
            yield from self._carry()
        except _Abort:
            yield from self._grip(False, open_to=self.plan['stroke'])

    # -- driving -----------------------------------------------------------------------
    def _drive_to(self, gx, gy, gth, timeout=60.0):
        """The platform to (gx, gy) heading gth: omniwheels straight there; the others along the line through the
        goal in its heading (pure pursuit, forwards or backwards), first converging onto it if they're off it."""
        plan, kind = self.plan, self.plan['cart']
        x, y, th = self._pose()
        if math.hypot(gx - x, gy - y) < 0.015 and abs(_wrap(gth - th)) < 0.03:
            return
        start = self._now()
        speed = 0.8 * float(self.drive.speed)
        safe_y = plan['front'] - plan['half_diagonal'] - 0.04  # turning on the spot clears the shelves below this
        d = np.array([math.cos(gth), math.sin(gth)])
        n = np.array([-d[1], d[0]])
        while True:
            if self._now() - start > timeout:
                raise _Abort('driving took too long (blocked?)')
            x, y, th = self._pose()
            p = np.array([x, y])
            rel = p - [gx, gy]
            s_along, e_lat = float(rel @ d), float(rel @ n)
            eth = _wrap(gth - th)
            dist = math.hypot(*rel)
            if kind == 'omni':
                if dist < 0.01 and abs(eth) < 0.02:
                    break
                if abs(eth) > 0.15 and y > safe_y:  # away from the shelves before turning
                    vx, vy = 0.0, -speed
                    w = 0.0
                else:
                    v = min(speed, 1.5 * dist + 0.02)
                    vx, vy = (-rel / max(dist, 1e-9) * v) if dist > 0.003 else (0.0, 0.0)
                    w = max(-1.0, min(1.0, 2.5 * eth)) * float(self.drive.turn_speed)
                c, s = math.cos(th), math.sin(th)
                self.drive.forward, self.drive.sideways, self.drive.turn = c * vx + s * vy, -s * vx + c * vy, w
                yield
                continue
            heading = np.array([math.cos(th), math.sin(th)])
            misaligned = abs(float(heading @ d)) < math.cos(math.radians(25))
            if kind in ('diff', 'tank') and misaligned:
                # Turn on the spot to the lane's direction (forwards or backwards along it), away from the shelves.
                if y > safe_y:
                    self.drive.forward = -speed if heading[1] > 0 else speed
                    self.drive.turn = 0.0
                else:
                    target = gth if float(heading @ d) >= 0 else gth + math.pi
                    self.drive.forward = 0.0
                    self.drive.turn = max(-1.0, min(1.0, 2.0 * _wrap(target - th))) * float(self.drive.turn_speed)
                yield
                continue
            if abs(s_along) < 0.012 and abs(e_lat) < 0.03:
                if kind == 'car' or abs(eth) < 0.03 or abs(_wrap(eth + math.pi)) < 0.03:
                    break
                self.drive.forward = 0.0  # the last few degrees on the spot
                self.drive.turn = max(-1.0, min(1.0, 3.0 * eth)) * float(self.drive.turn_speed)
                yield
                continue
            look = 0.45
            if abs(e_lat) > 0.04:  # converge onto the line, carrying on along it (overshooting if need be)
                way = -math.copysign(1.0, s_along) if abs(s_along) > 0.7 else (1.0 if float(heading @ d) >= 0 else -1.0)
                target = np.array([gx, gy]) + d * (s_along + way * look)
            else:  # along the line to the goal
                s_t = min(s_along + look, 0.0) if s_along < 0 else max(s_along - look, 0.0)
                target = np.array([gx, gy]) + d * s_t
            c, s = math.cos(th), math.sin(th)
            tb = np.array([c * (target[0] - x) + s * (target[1] - y), -s * (target[0] - x) + c * (target[1] - y)])
            ln2 = max(float(tb @ tb), 1e-6)
            kappa = 2.0 * tb[1] / ln2
            v = math.copysign(min(speed, 1.2 * dist + 0.04), tb[0] if abs(tb[0]) > 1e-6 else 1.0)
            if kind == 'car':
                steer = math.atan(kappa * self.drive.wheelbase)
                self.drive.turn = max(-1.0, min(1.0, steer / max(self.drive.steer_limit, 1e-3))) * \
                    float(self.drive.turn_speed)
                self.drive.forward = v * max(0.35, 1.0 - abs(steer) / 1.2)
            else:
                self.drive.forward = v
                self.drive.turn = v * kappa
            self.drive.sideways = 0.0
            yield
        self._stop_driving()
        yield from self._wait(0.4)

    def _goto_lane(self, base_x):
        """Drive so the arm's base is beside x = `base_x`, the platform on the lane in front of the shelves."""
        plan = self.plan
        _, _, th = self._pose()
        gth = 0.0
        if plan['cart'] == 'car' and abs(_wrap(th)) > math.pi / 2:
            gth = math.pi  # a car can't turn round on the spot: it works facing the other way
        mx, my = plan['mount']
        c, s = math.cos(gth), math.sin(gth)
        gx = base_x - (c * mx - s * my)
        lim = plan['floor_x']
        gx = max(-lim, min(lim, gx))
        yield from self._drive_to(gx, plan['lane_y'], gth)

    # -- the task ----------------------------------------------------------------------
    def _grasp_plan(self, i):
        """Where and how to take item i: the grasp point (world x, y), the tool point's height, yaw (world) and how
        far to open the jaws; and the item's top and bottom."""
        plan, it, d = self.plan, self.plan['items'][i], self.data
        c, R = d.xpos[it['body']].copy(), d.xmat[it['body']].reshape(3, 3)
        shape, dims, hh = it['shape'], it['dims'], it['half_h']
        # Round items can be taken however they've rolled; an egg as long as it lies on its side (its long axis, x,
        # level); the rest only standing up.
        if shape == 'egg':
            if abs(R[2, 0]) > 0.4:
                raise _Abort('it stands on its end')
        elif shape not in ('sphere', 'apple') and R[2, 2] < 0.8:
            raise _Abort('it lies on its side')
        yaw_item = math.atan2(R[1, 0], R[0, 0])
        bottom, top = c[2] - hh, c[2] + hh
        H = 2 * hh
        point = c[:2].copy()
        if shape == 'box':
            dx, dy = dims[0], dims[1]
        elif shape == 'egg':
            dx, dy = 2 * dims[0], 2 * dims[1]
        elif shape == 'banana':
            dx, dy = dims[0], 2 * dims[1]
            point += R[:2, :2] @ np.array([0.0, dims[2] / 2])  # the curve's middle
        elif shape == 'cup':
            dx = dy = 2 * max(dims[0], dims[1])
        else:
            dx = dy = 2 * dims[0]
        g = plan['gripper']
        if g == 'parallel':
            yaw, period = (yaw_item if dx <= dy else yaw_item + math.pi / 2), math.pi
            width = min(dx, dy)
            open_to = width / 2 + 0.014
        elif g == 'three_jaw':
            yaw, period = yaw_item, 2 * math.pi / 3
            # (round its middle: a box's corners, an egg's long axis; a banana across its middle)
            open_to = (dims[1] if shape == 'banana' else 0.5 * math.hypot(dx, dy) if shape in ('box', 'egg')
                       else dx / 2) + 0.012
        elif g == 'suction':
            yaw, period, open_to = self._tcp_yaw_world(), 2 * math.pi, None
        else:
            yaw, period, open_to = yaw_item, 2 * math.pi / max(plan['fingers'], 1), None
        here = self._tcp_yaw_world()
        yaw = here + _wrap(yaw - here, period)  # the least turn of the wrist ...
        n_eq = int(round(2 * math.pi / period))
        for k in sorted(range(-n_eq // 2, n_eq // 2 + 1), key=abs):  # ... that the arm can reach
            if self._reachable((*point, top + 0.05), yaw + k * period):
                yaw += k * period
                break
        if open_to is not None and open_to > plan['stroke'] - 0.002:
            raise _Abort(f'too big for the jaws ({2 * open_to * 1000:.0f} mm)')
        self._open_to = open_to
        if g == 'suction':
            z = top - 0.003
        else:
            low = {'pear': 0.08, 'banana': 0.1, 'egg': 0.12}.get(shape, 0.3 if g == 'pneunet' else 0.25)
            z = max(bottom + low * H + plan['reach_below_tcp'], top - plan['palm_above_tcp'] + 0.005)
        return {'point': point, 'z': z, 'yaw': yaw, 'open_to': open_to, 'top': top, 'bottom': bottom, 'H': H,
                'yaw_item': yaw_item}

    def _free_slot(self, i):
        """An empty place for item i's product, the nearest the arm first; None if there's none."""
        plan, d = self.plan, self.data
        product = plan['items'][i]['product']
        pos = d.xpos[[it['body'] for it in plan['items']]]
        x, y, th = self._pose()
        mx, my = plan['mount']
        base_x = x + math.cos(th) * mx - math.sin(th) * my
        best, best_d = None, None
        for k, s in enumerate(plan['slots']):
            if s['product'] != product or k in self.reserved:
                continue
            inside = ((np.abs(pos[:, 0] - s['x']) < s['hx']) & (np.abs(pos[:, 1] - s['y']) < s['hy']) &
                      (pos[:, 2] > s['top']) & (pos[:, 2] < s['top'] + 0.25))
            inside[i] = False
            if inside.any():
                continue
            dist = abs(s['x'] - base_x)
            if best is None or dist < best_d:
                best, best_d = k, dist
        return best

    def _stock(self, i):
        plan = self.plan
        it = plan['items'][i]
        name = _item_name(it)
        state = item_states(self.data)[i]
        if state == 'moving':
            raise _Abort('it isn\'t lying still')
        k = self._free_slot(i)
        if k is None:
            raise _Abort('no empty place for it')
        self.reserved.add(k)
        slot = plan['slots'][k]
        print(f'[auto] the {name} ({state}) -> {slot["unit"]} unit, shelf {slot["level"] + 1}')
        yield from self._grip(False, open_to=plan['stroke'])
        yield from self._carry()
        # ---- take it
        if state in ('shelf', 'stocked'):
            yield from self._goto_lane(float(self.data.xpos[it['body']][0]))
            yield from self._take_from_shelf(i)
        elif state == 'floor':
            yield from self._goto_item(i)
            yield from self._take_from_above(i)
        else:
            yield from self._take_from_above(i)
        yield from self._carry()
        # ---- put it in its place
        yield from self._goto_lane(slot['x'])
        yield from self._put(i, slot)
        yield from self._carry()
        ok = item_states(self.data)[i] == 'stocked'
        self.done += ok
        self.failed += not ok
        print(f'[auto] the {name}: ' + ('in its place' if ok else 'put down, but not quite in its place'))

    def _goto_item(self, i):
        """Drive the arm's base to about 0.45 m from an item on the floor (heading as it is)."""
        plan = self.plan
        x, y, th = self._pose()
        mx, my = plan['mount']
        c, s = math.cos(th), math.sin(th)
        base = np.array([x + c * mx - s * my, y + s * mx + c * my])
        item = self.data.xpos[plan['items'][i]['body']][:2]
        off = base - item
        n = np.linalg.norm(off)
        goal_base = item + (off / n if n > 1e-6 else np.array([0.0, -1.0])) * 0.45
        gx, gy = goal_base[0] - (c * mx - s * my), goal_base[1] - (s * mx + c * my)
        gy = min(gy, plan['lane_y'])
        yield from self._drive_to(gx, gy, th)

    def _lifted(self, i, z0, rise):
        return float(self.data.xpos[self.plan['items'][i]['body']][2]) > z0 + 0.6 * rise

    def _take_from_above(self, i):
        """From the crate (or the floor): over it at the carrying height, down, grip, back up."""
        plan = self.plan
        g = self._grasp_plan(i)
        z0 = float(self.data.xpos[plan['items'][i]['body']][2])
        travel = max(plan['carry'][2], g['top'] + plan['reach_below_tcp'] + 0.06)
        yield from self._grip(False, open_to=g['open_to'] if g['open_to'] else plan['stroke'])
        yield from self._move((*g['point'], travel), g['yaw'], what='over the item')
        yield from self._move((*g['point'], g['z']), g['yaw'], what='down to the item', timeout=3.0)
        yield from self._grip(True)
        yield from self._move((*g['point'], travel), g['yaw'], what='lifting it')
        if not self._lifted(i, z0, travel - g['z']):
            raise _Abort('it slipped out of the gripper')
        self._held = self._hold_state(i)

    def _take_from_shelf(self, i):
        """From a shelf: in front of it, in over the item (under the next shelf), down, grip, up a little, out."""
        plan = self.plan
        g = self._grasp_plan(i)
        z0 = float(self.data.xpos[plan['items'][i]['body']][2])
        y_out = plan['y_out']
        over = g['top'] + plan['reach_below_tcp'] + 0.02
        travel = max(plan['carry'][2], over)
        yield from self._grip(False, open_to=g['open_to'] if g['open_to'] else plan['stroke'])
        yield from self._move((g['point'][0], y_out, travel), g['yaw'], what='in front of the shelf')
        yield from self._move((g['point'][0], y_out, over), g['yaw'], what='down in front of the shelf')
        yield from self._move((*g['point'], over), g['yaw'], what='into the shelf')
        yield from self._move((*g['point'], g['z']), g['yaw'], what='down to the item', timeout=3.0)
        yield from self._grip(True)
        up = g['z'] + plan['lip'] + 0.02
        yield from self._move((*g['point'], up), g['yaw'], what='lifting it')
        if not self._lifted(i, z0, up - g['z']):
            raise _Abort('it slipped out of the gripper')
        yield from self._move((g['point'][0], y_out, up), g['yaw'], what='out of the shelf')
        yield from self._move((g['point'][0], y_out, travel), g['yaw'], what='up')
        self._held = self._hold_state(i)

    def _hold_state(self, i):
        """How the item hangs from the tool point: its bottom below it, its centre's offset (world x, y), its yaw."""
        it, d = self.plan['items'][i], self.data
        c, R = d.xpos[it['body']], d.xmat[it['body']].reshape(3, 3)
        tcp = self._tcp_world()
        return {'below': float(tcp[2] - (c[2] - it['half_h'])), 'offset': (tcp[:2] - c[:2]).copy(),
                'yaw_item': math.atan2(R[1, 0], R[0, 0]), 'yaw_tcp': self._tcp_yaw_world(),
                'H': 2 * it['half_h'], 'open_to': self._open_to}

    def _put(self, i, slot):
        """Into its place: in front of the shelf at a height that clears the lip, in, down onto the shelf, let go,
        up clear of the item, out and up."""
        plan, it = self.plan, self.plan['items'][i]
        h = self._held
        if it['shape'] in ('box', 'egg', 'banana'):
            turn0, period = _wrap(math.radians(slot['yaw']) - h['yaw_item'], math.pi), math.pi
        else:
            turn0, period = 0.0, math.pi / 4  # round: any way round
        z_place = slot['top'] + h['below'] + 0.004
        o = h['offset']

        def target(turn):
            c, s = math.cos(turn), math.sin(turn)
            return slot['x'] + c * o[0] - s * o[1], slot['cy'] + s * o[0] + c * o[1], h['yaw_tcp'] + turn

        # The item may go in either way round (a box: half a turn; round: any): the least turn the arm can reach.
        turns = sorted((turn0 + k * period for k in range(-int(round(2 * math.pi / period)), int(round(2 * math.pi / period)) + 1)
                        if abs(turn0 + k * period) <= math.pi + 1e-9), key=abs)
        tx, ty, yaw = target(turns[0])
        for turn in turns:
            cx_, cy_, cyaw = target(turn)
            if self._reachable((cx_, cy_, z_place), cyaw) and self._reachable((cx_, plan['y_out'], z_place + 0.05), cyaw):
                tx, ty, yaw = cx_, cy_, cyaw
                break
        z_in = slot['top'] + h['below'] + plan['lip'] + 0.02
        z_clear = slot['top'] + h['H'] + plan['reach_below_tcp'] + 0.02
        if slot['ceiling'] is not None:  # under the next shelf: the wrist must fit
            z_clear = min(z_clear, slot['ceiling'] - plan['above_tcp'] - 0.01)
            z_in = min(z_in, slot['ceiling'] - plan['above_tcp'] - 0.01)
        y_out = plan['y_out']
        travel = max(plan['carry'][2], z_in + 0.02, z_clear + 0.02)  # (outside the shelf)
        yield from self._move((tx, y_out, travel), yaw, what='in front of the place')
        yield from self._move((tx, y_out, z_in), yaw, what='down in front of the place')
        yield from self._move((tx, ty, z_in), yaw, what='into the shelf')
        yield from self._move((tx, ty, z_place), yaw, what='down onto the shelf', timeout=3.0)
        yield from self._grip(False, open_to=h['open_to'])  # (jaws: no wider than for taking it)
        yield from self._move((tx, ty, max(z_clear, z_place + 0.02)), yaw, what='up off the item')
        yield from self._move((tx, y_out, max(z_clear, z_place + 0.02)), yaw, what='out of the shelf')
        yield from self._move((tx, y_out, travel), yaw, what='up')


def _item_name(it):
    return PRODUCTS[it['product']][0].lower()


# ----------------------------------------------------------------------
# The store's layout
# ----------------------------------------------------------------------
def store_layout(P):
    """Where everything goes [m]: the shelf units (their boards, panels), the places on the shelves (product, centre,
    filled or not) and the crates with their items' starting positions."""
    front = max(float(P['shelf_distance']), 125.0) * MM
    length = max(float(P['shelf_length']), 400.0) * MM
    depth = max(float(P['shelf_depth']), 120.0) * MM
    deck = max(float(P['deck_height']), 20.0) * MM
    level = max(float(P['level_height']), 150.0) * MM
    board = max(float(P['board_thickness']), 5.0) * MM
    gap = max(float(P['item_gap']), 0.0) * MM
    panel = 15 * MM  # the end panels' thickness (as in createScene)
    # Side by side along x, both facing -y; the left one alone is centred.
    units = ([('left', -(length / 2 + panel)), ('right', length / 2 + panel)] if P['second_unit'] else [('left', 0.0)])
    levels = 1 + max(k[2] for k in SHELF_PLAN)
    margin = 15 * MM
    shelves, places = [], []
    y_front, into = front, 1.0
    for name, cx in units:
        tops = [deck + i * level for i in range(levels)]
        shelves.append({'name': name, 'cx': cx, 'y_front': y_front, 'into': into, 'tops': tops, 'length': length,
                        'depth': depth, 'board': board, 'height': tops[-1] + 0.85 * level})
        for bay in (0, 1):
            x_lo, x_hi = (cx - length / 2 + margin, cx - margin) if bay == 0 else (cx + margin, cx + length / 2 - margin)
            for lv in range(levels):
                plan = SHELF_PLAN.get((name, bay, lv), [])
                widths = []
                for product, n, _ in plan:
                    _, shape, dims, _ = PRODUCTS[product]
                    fx, fy = _item_footprint(shape, [d * MM for d in dims], EGG_YAW if product == 'egg' else 0.0)
                    widths.append((fx + gap) * n)
                spare = max((x_hi - x_lo) - sum(widths), 0.0) / max(len(plan), 1)
                cursor = 0.0  # left to right, along +x
                for (product, n, filled), w in zip(plan, widths):
                    label, shape, dims, _ = PRODUCTS[product]
                    dims_m = [d * MM for d in dims]
                    yaw = EGG_YAW if product == 'egg' else 0.0
                    fx, fy = _item_footprint(shape, dims_m, yaw)
                    start = cursor + spare / 2
                    a0 = start
                    cursor += w + spare
                    y_c = y_front + into * (12 * MM + 0.5 * max(fy, 0.07))
                    for i in range(n):
                        u = (i + 0.5) * (fx + gap)
                        x = x_lo + a0 + u
                        places.append({'unit': name, 'bay': bay, 'level': lv, 'product': product,
                                       'centre': [x, y_c], 'size': (fx, fy), 'yaw': yaw, 'top': tops[lv],
                                       'filled': i < filled})
    # The platform's crate: at the platform's back (its frame: x forward, y left), packed with PLATFORM_CRATE.
    cl, cw = max(float(P['crate_length']), 80.0) * MM, max(float(P['crate_width']), 60.0) * MM
    ch, ct = max(float(P['crate_height']), 20.0) * MM, max(float(P['crate_wall']), 2.0) * MM
    cgap = max(float(P['crate_gap']), 0.0) * MM
    cx = -max(float(P['platform_length']), 100.0) * MM / 2 + 0.03 + ct + cl / 2
    spots = pack_crate(PLATFORM_CRATE, cl, cgap)
    for spot in spots:
        spot['centre'] = [cx + spot['centre'][0], spot['centre'][1]]
    platform_crate = {'centre': [cx, 0.0], 'inner': (cl, cw), 'height': ch, 'wall': ct, 'items': spots}
    crates = []
    width = sum(length + 2 * panel for _ in units)
    return {'shelves': shelves, 'places': places, 'crates': crates, 'platform_crate': platform_crate, 'front': front,
            'length': length, 'width': width, 'depth': depth, 'levels': levels}


# ----------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------
BUILD = {}


def _v(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def _scene_class():
    try:
        from sofaweb.mjscene import MjScene
    except ImportError:  # run from the repository: scenes/_base/sofaweb
        sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..', '..', '_base'))
        from sofaweb.mjscene import MjScene
    return MjScene


def createScene():
    """Builds the MuJoCo model and returns an MjScene (see scenes/_base/sofaweb/mjscene.py)."""
    launch_params_editor()
    P = PARAMS
    MjScene = _scene_class()

    F = max(1, int(P['fingers']))
    N = max(2, int(P['segments']))
    L, W, H = float(P['length']) * MM, float(P['width']) * MM, float(P['height']) * MM
    sep = max(float(P['channel_separation']), 0.0) * MM
    pitch = L + sep
    finger_len = N * L + (N - 1) * sep
    we, ws, wb, wt = (float(P[k]) * MM for k in ('wall_end', 'wall_side', 'wall_bottom', 'wall_top'))
    radius = max(float(P['finger_radius']), 0.0) * MM
    tcp_depth = max(float(P['tcp_depth']), 0.0) * MM
    # The gripper is built in its own frame (as Gripper 5 at rest): the tool point at the origin, the palm above it.
    z_palm = tcp_depth
    extent = radius + H
    g = 9.81 if P['gravity_enabled'] else 0.0
    layout = store_layout(P)

    cav_l, cav_w, cav_h = max(L - 2 * we, 0.0), max(W - 2 * ws, 0.0), max(H - wb - wt, 0.0)
    seg_mass = float(P['density']) * (L * W * H - cav_l * cav_w * cav_h)
    cavity_area = cav_w * cav_h
    cavity_mid = wb + 0.5 * cav_h
    torque_per_pa = float(P['pressure_torque_gain']) * cavity_area * cavity_mid
    tips = bool(P['tips_enabled']) and float(P['tip_length']) > 0
    tip_l = max(float(P['tip_length']), 0.5) * MM
    tip_w = min(max(float(P['tip_width']), 0.2) * MM, W)
    tip_t = min(max(float(P['tip_thickness']), 0.2) * MM, H)

    floor_z = 0.0
    thick = 0.6 * H
    gkind = P['gripper'] if P['gripper'] in dict(GRIPPER_CHOICES) else 'pneunet'
    cart = P['cart'] if P['cart'] in dict(CART_CHOICES) else 'omni'
    frames = ([FingerFrame(k, F, radius, float(P['finger_azimuth']), z_palm) for k in range(F)]
              if gkind == 'pneunet' else [])
    flange_pos = np.zeros(3)  # the gripper's frame: at the tool point
    grip_top = z_palm + thick + 0.058  # the top of its wrist adapter (it sits on the arm's flange)
    open_lim = math.radians(float(P['joint_open_limit']))
    close_lim = 2.0 * math.atan2(sep / 2, H)  # the roofs touch (bent outwards): the gap is `sep` at the base
    m_flange = max(float(P['flange_mass']), 1e-3)
    pad, stem = 0.1 * H, 0.3 * H
    plate_r = extent + pad
    mu = max(float(P['friction']), 0.0)
    k_joint = float(P['joint_stiffness'])
    timestep = max(float(P['timestep']), 1e-5)
    limit_tc = max(float(P['limit_timeconst']), 2 * timestep)

    # The platform and the arm [m].
    PL, PW, PH = (max(float(P[k]), 100.0) * MM for k in ('platform_length', 'platform_width', 'platform_height'))
    start = np.array([float(P['start_x']) * MM, float(P['start_y']) * MM, 0.0])
    mount = np.array([float(P['arm_mount_x']) * MM, float(P['arm_mount_y']) * MM, PH])
    pedestal = 0.05
    spec = arm_spec(P, pedestal)
    n_arm = len(spec['joints'])
    if 'mount' in spec:
        mount = np.array([*spec['mount'], PH])
    mount_yaw = spec.get('mount_yaw', 0.0)  # the arm's base turned on the cart (its zero facing back)

    finger_bits = 'contype="2" conaffinity="3"' if P['finger_collisions'] else 'contype="2" conaffinity="1"'
    other_bits = 'contype="1" conaffinity="3"'
    # The arm's links touch the shelves and the items, not the gripper (contype 4 / conaffinity 1); the platform
    # touches the shelves, the items, the fingers and the arm (contype 1 / conaffinity 6).
    arm_bits = 'contype="4" conaffinity="1"'
    platform_bits = 'contype="1" conaffinity="6"'

    crate = layout['platform_crate']
    gripper_text = (f"{F} soft fingers of {N} segments" + (f", pointy tips {tip_l / MM:.0f} mm" if tips else "")
                    if gkind == 'pneunet' else dict(GRIPPER_CHOICES)[gkind].lower())
    print(f"[model] a {PL / MM:.0f} x {PW / MM:.0f} mm cart ({dict(CART_CHOICES)[cart].lower()}) with a "
          f"{spec['label']} arm; {gripper_text}; {sum(1 for p in layout['places'] if p['filled'])} items on the "
          f"shelves, {len(crate['items'])} in the platform's crate")

    def seg_geom(fr, j, origin, name):
        c = np.array(fr.to_world([j * pitch + L / 2, 0.0, H / 2])) - origin
        return (f'<geom name="{name}" type="box" size="{_v([L / 2, W / 2, H / 2])}" pos="{_v(c)}" '
                f'xyaxes="{_v(fr.down)} {_v(fr.tangent)}" mass="{seg_mass:.9g}" class="finger"/>')

    def tip_corners(fr):
        """The tip's 8 corners (gripper frame, at rest) in _box's order: the last segment's end face, then the point."""
        x0 = (N - 1) * pitch + L
        x1 = x0 + tip_l
        local = [[x0, -W / 2, 0.0], [x1, -tip_w / 2, 0.0], [x1, tip_w / 2, 0.0], [x0, W / 2, 0.0],
                 [x0, -W / 2, H], [x1, -tip_w / 2, tip_t], [x1, tip_w / 2, tip_t], [x0, W / 2, H]]
        return [fr.to_world(p) for p in local]

    tip_volume = tip_l * (W * H + tip_w * tip_t + math.sqrt(W * H * tip_w * tip_t)) / 3.0  # a frustum's
    tip_mass = float(P['density']) * tip_volume

    assets, world, statics = [], [], []

    def static_box(name, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        statics.append((name, lo, hi))
        world.append(f'<geom name="{name}" type="box" size="{_v((hi - lo) / 2)}" pos="{_v((hi + lo) / 2)}" '
                     f'class="static"/>')

    half_floor = max(layout['width'] / 2 + 1.0, layout['front'] + layout['depth'] + 1.5)
    world.append(f'<geom name="floor" type="plane" size="{_v([2 * half_floor, 2 * half_floor, 0.05])}" '
                 f'pos="0 0 {floor_z}" class="static"/>')

    # ---- the shelf units: plinth (the deck), boards, lips, back and end panels
    lip_h = max(float(P['lip_height']), 0.0) * MM
    lip_t = 6 * MM
    panel_t = 15 * MM
    for sh in layout['shelves']:
        n, yf, s, Lx, D, bt, cx = (sh['name'], sh['y_front'], sh['into'], sh['length'], sh['depth'], sh['board'],
                                   sh['cx'])
        y0, y1 = sorted((yf, yf + s * D))
        for i, top in enumerate(sh['tops']):
            static_box(f'{n}_board{i}', [cx - Lx / 2, y0, 0.0 if i == 0 else top - bt], [cx + Lx / 2, y1, top])
            if lip_h > 0:
                ly0, ly1 = sorted((yf, yf + s * lip_t))
                static_box(f'{n}_lip{i}', [cx - Lx / 2, ly0, top], [cx + Lx / 2, ly1, top + lip_h])
        by0, by1 = sorted((yf + s * D, yf + s * (D + panel_t)))
        static_box(f'{n}_back', [cx - Lx / 2 - panel_t, by0, 0.0], [cx + Lx / 2 + panel_t, by1, sh['height']])
        for e, (ex0, ex1) in enumerate(((cx - Lx / 2 - panel_t, cx - Lx / 2), (cx + Lx / 2, cx + Lx / 2 + panel_t))):
            static_box(f'{n}_end{e}', [ex0, y0, 0.0], [ex1, y1, sh['height']])

    # ---- the gripper (Gripper 5's): its bodies in its own frame, the tool point at the origin
    hinge_names, side_names = [], []
    k_side = max(float(P['side_stiffness']), 0.0) * k_joint
    side_lim = math.radians(max(float(P['side_limit']), 0.1))
    gripper = [f'<geom name="plate" type="cylinder" size="{plate_r:.9g} {thick / 2:.9g}" '
               f'pos="0 0 {tcp_depth + thick / 2:.9g}" mass="{m_flange:.9g}" class="finger"/>']
    pivots = {}
    for fr in frames:
        gripper.append(seg_geom(fr, 0, flange_pos, f'f{fr.k + 1}s0'))
        parent = flange_pos
        closing = []
        for j in range(1, N):
            pv = np.array(fr.to_world([j * pitch - sep / 2, 0.0, 0.0]))
            pivots[(fr.k, j)] = pv
            name = f'f{fr.k + 1}j{j}'
            hinge_names.append(name)
            gripper.append(f'<body name="f{fr.k + 1}s{j}" pos="{_v(pv - parent)}">')
            gripper.append(f'<joint name="{name}" type="hinge" axis="{_v(fr.tangent)}" range="{-close_lim:.9g} {open_lim:.9g}" '
                           f'stiffness="{k_joint:.9g}" damping="{float(P["joint_damping"]) * k_joint:.9g}" '
                           f'armature="{float(P["joint_armature"]):.9g}" solreflimit="{limit_tc:.9g} 1" '
                           f'solimplimit="0.95 0.99 0.001"/>')
            if k_side > 0:  # sideways, about the finger's radial direction (the tip moves tangentially)
                side_names.append(f'{name}side')
                gripper.append(f'<joint name="{name}side" type="hinge" axis="{_v(fr.radial)}" '
                               f'range="{-side_lim:.9g} {side_lim:.9g}" stiffness="{k_side:.9g}" '
                               f'damping="{float(P["joint_damping"]) * k_side:.9g}" '
                               f'armature="{float(P["joint_armature"]):.9g}" solreflimit="{limit_tc:.9g} 1" '
                               f'solimplimit="0.95 0.99 0.001"/>')
            gripper.append(seg_geom(fr, j, pv, f'f{fr.k + 1}s{j}'))
            if tips and j == N - 1:
                corners = [list(np.array(p) - pv) for p in tip_corners(fr)]
                assets.append(f'<mesh name="tip{fr.k + 1}" vertex="{_v(np.asarray(corners).ravel())}"/>')
                gripper.append(f'<geom name="f{fr.k + 1}tip" type="mesh" mesh="tip{fr.k + 1}" mass="{tip_mass:.9g}" '
                               f'class="finger"/>')
            closing.append('</body>')
            parent = pv
        gripper += closing

    stroke = max(float(P['jaw_stroke']), 5.0) * MM
    jaw_l, jaw_w, jaw_t = max(float(P['jaw_length']), 10.0) * MM, max(float(P['jaw_width']), 4.0) * MM, 10 * MM
    jaw_names, jaw_dirs = [], []
    if gkind in ('parallel', 'three_jaw'):
        n_jaws = 2 if gkind == 'parallel' else 3
        gripper.append(f'<geom name="jaw_rail" type="box" size="{_v([stroke + jaw_t, 0.016, 0.01])}" '
                       f'pos="0 0 {z_palm - 0.01:.9g}" mass="0.3" class="finger"/>' if n_jaws == 2 else
                       f'<geom name="jaw_rail" type="cylinder" size="{stroke + jaw_t:.9g} 0.01" '
                       f'pos="0 0 {z_palm - 0.01:.9g}" mass="0.3" class="finger"/>')
        for k in range(n_jaws):
            phi = 2 * math.pi * k / n_jaws
            e = np.array([math.cos(phi), math.sin(phi), 0.0])
            t = np.array([-e[1], e[0], 0.0])
            name = f'jaw{k + 1}'
            jaw_names.append(name)
            jaw_dirs.append(e)
            centre = e * (stroke + jaw_t / 2) + np.array([0.0, 0.0, z_palm - 0.02 - jaw_l / 2])
            gripper += [f'<body name="{name}">',
                        f'<joint name="{name}" type="slide" axis="{_v(-e)}" range="0 {stroke - 0.004:.9g}" '
                        f'damping="150" armature="0.02"/>',
                        f'<geom name="{name}" type="box" size="{_v([jaw_t / 2, jaw_w / 2, jaw_l / 2])}" '
                        f'pos="{_v(centre)}" xyaxes="{_v(e)} {_v(t)}" mass="0.06" friction="1.2 0.01 0.0001" '
                        f'class="finger"/>', '</body>']
    elif gkind == 'suction':
        gripper += ['<body name="cup">',
                    f'<geom name="cup_stem" type="cylinder" size="0.009 {(z_palm - 0.012) / 2:.9g}" '
                    f'pos="0 0 {0.012 + (z_palm - 0.012) / 2:.9g}" mass="0.05" class="finger"/>',
                    '<geom name="cup" type="cylinder" size="0.019 0.006" pos="0 0 0.006" mass="0.02" '
                    'margin="0.004" gap="0.004" friction="1.2 0.01 0.0001" class="finger"/>', '</body>']

    # ---- the arm: at zero every link is aligned with the platform, the links pointing up and the tool along +y
    # (the shoulder axis); the gripper sits on the flange, turned so its frame's +z points back into the flange.
    gains = [g_ * float(P['arm_gain']) for g_ in spec['gains']]
    arm_joints = [f'arm_j{i}' for i in range(1, n_arm + 1)]
    limits, link_mass, links = spec['limits'], spec['masses'], spec['joints']
    tool_dir = np.array(spec['dir'], float)
    flange_in_link6 = tuple(np.array(spec['tool']) + tool_dir * grip_top)  # (the last link's)
    # The gripper's +z points back into the flange; on the Panda (turned round on the cart) it's also turned half a
    # turn about the tool axis, so pointing it forward needs no half turn of the last joint (it turns only 332 deg).
    flange_quat = _quat_z_to(-tool_dir)
    if spec.get('tool_roll'):
        flange_quat = tuple(_quat_mul(np.array(flange_quat), np.array([math.cos(mount_yaw / 2), 0, 0,
                                                                         math.sin(mount_yaw / 2)])))

    def arm_bodies(ik):
        """The arm's bodies (from its base on the platform); the gripper on its flange, or (for the IK model) only
        a site at the tool point."""
        out = [f'<body name="arm_base" pos="{_v(mount)}" quat="{_v([math.cos(mount_yaw / 2), 0, 0, math.sin(mount_yaw / 2)])}">']
        if not ik:
            out.append(f'<geom name="arm_pedestal" type="cylinder" size="0.075 {pedestal / 2:.9g}" '
                       f'pos="0 0 {pedestal / 2:.9g}" mass="2" class="platform"/>')
        for i, (axis, offset, (p0, p1), r) in enumerate(links):
            lo, hi = limits[i]
            out.append(f'<body name="arm_l{i + 1}" pos="{_v(offset)}"{"" if ik else " gravcomp=\"1\""}>')
            out.append(f'<joint name="{arm_joints[i]}" type="hinge" axis="{_v(axis)}" range="{lo:.9g} {hi:.9g}" '
                       f'armature="{0.05 if i < 3 else 0.01}" damping="{0.5 if i < 3 else 0.05}"/>')
            if ik:  # (kinematics only, but a moving body needs some inertia)
                out.append('<inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>')
            else:
                out.append(f'<geom name="arm_l{i + 1}" type="capsule" fromto="{_v(p0)} {_v(p1)}" size="{r}" '
                           f'mass="{link_mass[i]}" class="arm"/>')
        if ik:
            out.append(f'<site name="tcp" pos="{_v(flange_in_link6)}" quat="{_v(flange_quat)}"/>')
        else:
            out.append(f'<body name="flange" pos="{_v(flange_in_link6)}" quat="{_v(flange_quat)}">')
            out += gripper
            out.append('</body>')
        out += ['</body>'] * (n_arm + 1)
        return out

    # ---- the platform: a heavy body on x, y and yaw joints, carrying the arm and a crate
    plat = [f'<body name="platform" pos="{_v(start)}" sleep="never">',
            '<joint name="base_x" type="slide" axis="1 0 0"/>', '<joint name="base_y" type="slide" axis="0 1 0"/>',
            '<joint name="base_yaw" type="hinge" axis="0 0 1"/>',
            f'<geom name="deck" type="box" size="{_v([PL / 2, PW / 2, (PH - 0.04) / 2])}" pos="0 0 {(PH + 0.04) / 2:.9g}" '
            f'mass="{max(float(P["platform_mass"]), 1.0):.9g}" class="platform"/>']
    (ccx, ccy), (cl, cw), ch, ct = crate['centre'], crate['inner'], crate['height'], crate['wall']
    x0, x1, y0, y1 = ccx - cl / 2, ccx + cl / 2, ccy - cw / 2, ccy + cw / 2
    crate_walls = [('front', (x0 - ct, y0 - ct), (x1 + ct, y0)), ('back', (x0 - ct, y1), (x1 + ct, y1 + ct)),
                   ('left', (x0 - ct, y0), (x0, y1)), ('right', (x1, y0), (x1 + ct, y1))]
    for name, (a0, b0), (a1, b1) in crate_walls:
        plat.append(f'<geom name="pcrate_{name}" type="box" size="{_v([(a1 - a0) / 2, (b1 - b0) / 2, ch / 2])}" '
                    f'pos="{_v([(a0 + a1) / 2, (b0 + b1) / 2, PH + ch / 2])}" mass="0.5" class="platform"/>')
    track_w, wheel_r = 0.07, 0.09
    if cart == 'tank':
        for side in (-1, 1):
            plat.append(f'<geom name="track{side + 1}" type="box" size="{_v([PL * 0.47, track_w / 2, 0.075])}" '
                        f'pos="{_v([0, side * (PW / 2 + track_w / 2 + 0.005), 0.08])}" mass="5" class="platform"/>')
    car_x, car_y = PL / 2 - 0.17, PW / 2 - 0.02
    if cart == 'car':
        for side, name in ((1, 'steer_l'), (-1, 'steer_r')):
            plat += [f'<body name="{name}" pos="{_v([car_x, side * car_y, wheel_r])}">',
                     f'<joint name="{name}" type="hinge" axis="0 0 1" range="-0.9 0.9" damping="2"/>',
                     '<inertial pos="0 0 0" mass="0.5" diaginertia="0.003 0.003 0.003"/>', '</body>']
    plat += arm_bodies(False)
    plat.append('</body>')
    world += plat

    # ---- the groceries: free bodies, on the shelves (the filled places) and in the platform's crate
    scale = max(float(P['item_mass_scale']), 1e-3)
    rolling = max(float(P['rolling_friction']), 0.0)
    items, counts = [], {}
    mesh_done = set()

    def add_item(product, x, y, z_base, yaw_deg, origin):
        label, shape, dims_mm, mass_g = PRODUCTS[product]
        dims = [d * MM for d in dims_mm]
        counts[product] = counts.get(product, 0) + 1
        key = f'{product}{counts[product]}'
        half_h, _ = _item_extent(shape, dims)
        centre = [x, y, z_base + half_h + 0.2 * MM]
        if shape == 'sphere':
            geom = f'type="sphere" size="{dims[0]:.9g}"'
        elif shape == 'apple':
            geom = f'type="ellipsoid" size="{_v([dims[0], dims[0], dims[1]])}"'
        elif shape == 'egg':
            geom = f'type="ellipsoid" size="{_v([dims[0], dims[1], dims[1]])}"'
        elif shape == 'cylinder':
            geom = f'type="cylinder" size="{dims[0]:.9g} {dims[1] / 2:.9g}"'
        elif shape == 'box':
            geom = f'type="box" size="{_v([d / 2 for d in dims])}"'
        elif shape == 'banana':
            geom = ''  # (capsules, below)
        else:  # the cup and the pear: convex meshes
            if product not in mesh_done:
                pts, _, _ = _item_meshes(shape, dims, smooth=False)[0]
                assets.append(f'<mesh name="{product}_mesh" vertex="{_v(np.asarray(pts).ravel())}"/>')
                mesh_done.add(product)
            geom = f'type="mesh" mesh="{product}_mesh"'
        extra = (f' condim="6" friction="{mu:.9g} {float(P["torsional_friction"]):.9g} {rolling:.9g}"'
                 if product in ROUND else '')
        q = [math.cos(math.radians(yaw_deg) / 2), 0.0, 0.0, math.sin(math.radians(yaw_deg) / 2)]
        mass = mass_g * 1e-3 * scale
        if shape == 'banana':  # capsules along the curve, as thick as it is there
            length, r, bend = dims
            axis = _banana_axis(length, bend)
            spans = list(zip(BANANA_CAPSULES[:-1], BANANA_CAPSULES[1:]))
            geoms = ''.join(
                f'<geom name="item_{key}_{i}" type="capsule" fromto="{_v(list(axis(a)[0]) + list(axis(b)[0]))}" '
                f'size="{r * min(_banana_radius(a), _banana_radius(b)):.9g}" mass="{mass / len(spans):.9g}" '
                f'class="item"/>' for i, (a, b) in enumerate(spans))
        else:
            geoms = f'<geom name="item_{key}" {geom} mass="{mass:.9g}" class="item"{extra}/>'
        world.append(f'<body name="item_{key}" pos="{_v(centre)}" quat="{_v(q)}"><freejoint name="item_{key}"/>'
                     f'{geoms}</body>')
        items.append({'key': key, 'product': product, 'shape': shape, 'dims': dims, 'centre': centre,
                      'half_h': half_h, 'origin': origin})

    for place in layout['places']:
        if place['filled']:
            add_item(place['product'], *place['centre'], place['top'], place['yaw'], 'shelf')
    for spot in crate['items']:  # (the platform starts with yaw 0: its frame is the world's, moved to `start`)
        add_item(spot['product'], start[0] + spot['centre'][0], start[1] + spot['centre'][1], PH, spot['yaw'], 'crate')

    # No contacts within one finger, nor between a finger and the palm (see Gripper 5), nor between the arm's base,
    # shoulder and upper arm and the platform they stand on.
    excludes = []
    for k in range(len(frames)):
        chain = ['flange'] + [f'f{k + 1}s{j}' for j in range(1, N)]
        for a in range(len(chain)):
            for b in range(a + 2, len(chain)):  # (a body and its parent never touch in MuJoCo anyway)
                excludes.append(f'<exclude body1="{chain[a]}" body2="{chain[b]}"/>')
    for link in ('arm_l1', 'arm_l2', 'arm_l3'):  # (and the jaws never touch the flange's own rail)
        excludes.append(f'<exclude body1="platform" body2="{link}"/>')
        excludes.append(f'<exclude body1="arm_base" body2="{link}"/>')

    # The actuators: the platform's three (stiff, critically damped for its mass) and the arm's six.
    m_total = max(float(P['platform_mass']), 1.0) + sum(link_mass) + 5.0
    i_total = m_total * (PL ** 2 + PW ** 2) / 12.0 + 3.0
    kp_lin, kp_yaw = 2e5, 2e4
    actuators = [f'<position name="drive_x" joint="base_x" kp="{kp_lin:.9g}" kv="{2 * math.sqrt(kp_lin * m_total):.9g}"/>',
                 f'<position name="drive_y" joint="base_y" kp="{kp_lin:.9g}" kv="{2 * math.sqrt(kp_lin * m_total):.9g}"/>',
                 f'<position name="drive_yaw" joint="base_yaw" kp="{kp_yaw:.9g}" kv="{2 * math.sqrt(kp_yaw * i_total):.9g}"/>']
    inertia = spec['inertia']
    for i, name in enumerate(arm_joints):
        actuators.append(f'<position name="{name}" joint="{name}" kp="{gains[i]:.9g}" '
                         f'kv="{2 * math.sqrt(gains[i] * inertia[i]):.9g}"/>')
    if cart == 'car':
        actuators += [f'<position name="{n}" joint="{n}" kp="60" kv="4"/>' for n in ('steer_l', 'steer_r')]
    if gkind == 'suction':
        actuators.append(f'<adhesion name="suction" body="cup" ctrlrange="0 1" '
                         f'gain="{max(float(P["suction_force"]), 0.0):.9g}"/>')

    sleep_flag = '<flag sleep="enable"/>' if P['sleep'] else ''
    defaults = f"""<default>
    <geom friction="{mu:.9g} {float(P['torsional_friction']):.9g} 0.0001" condim="4" solref="{max(4 * timestep, 0.002):.9g} 1"/>
    <default class="finger"><geom {finger_bits}/></default>
    <default class="item"><geom {other_bits}/></default>
    <default class="static"><geom {other_bits}/></default>
    <default class="arm"><geom {arm_bits}/></default>
    <default class="platform"><geom {platform_bits}/></default>
  </default>"""
    xml = f"""<mujoco model="gripper7">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep:.9g}" gravity="0 0 {-g:.9g}" cone="elliptic" impratio="{float(P['impratio']):.9g}"
          noslip_iterations="{max(int(P['noslip_iterations']), 0)}" integrator="implicitfast">{sleep_flag}</option>
  {defaults}
  <asset>{''.join(assets)}</asset>
  <worldbody>
    {chr(10).join(world)}
  </worldbody>
  <contact>{''.join(excludes)}</contact>
  <actuator>{''.join(actuators)}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    scene = MjScene(model, dt=float(P['dt']))
    scene.xml = xml
    body = scene.body_id
    flange_id = body('flange')
    platform_id = body('platform')

    # The arm alone, for the inverse kinematics: its base where it stands in the platform's frame.
    ik_xml = f"""<mujoco model="gripper7_ik"><compiler angle="radian" autolimits="true"/>
  <worldbody>{chr(10).join(arm_bodies(True))}</worldbody></mujoco>"""
    ik_model = mujoco.MjModel.from_xml_string(ik_xml)

    def body_of(k, j):
        return flange_id if j == 0 else body(f'f{k + 1}s{j}')

    def origin_of(k, j):
        return flange_pos if j == 0 else pivots[(k, j)]

    # ---- what the browser draws
    scene.background = [0.86, 0.89, 0.93, 1.0]
    scene.ambient = [0.42, 0.42, 0.42, 1.0]
    scene.add_light(direction=[-0.35, -0.6, 1.0])
    scene.add_light(color=[0.4, 0.4, 0.4, 1.0], direction=[0.6, 0.5, 0.4])
    # From the front, a little to the left and above: the robot in front, both units' shelves behind it.
    cam_lookat = [0.0, 0.5 * layout['front'] - 0.05, 0.35]
    cam_dir = (-0.25, -1.0, 0.65)
    norm = math.sqrt(sum(c * c for c in cam_dir))
    cam_distance = 1.15 * max(layout['width'], 1.4)
    cam_position = [l + cam_distance * c / norm for l, c in zip(cam_lookat, cam_dir)]
    scene.set_camera(cam_position, _look_at_quaternion(cam_position, cam_lookat), cam_distance)

    # The floor: grey and white tiles (two models, one per colour).
    tile = 0.3
    nt = int(math.ceil(half_floor / tile))
    tiles = ([], [])
    for i in range(-nt, nt):
        for j in range(-nt, nt):
            pts, tris = _box(i * tile, j * tile, floor_z - 2e-4, (i + 1) * tile, (j + 1) * tile, floor_z - 1e-4,
                             skip=('-Z', '-X', '+X', '-Y', '+Y'))
            tiles[(i + j) % 2].append((pts, tris))
    scene.add_visual('/floor/visu/visual', *_merge(tiles[0]), 0, [0.82, 0.82, 0.8, 1.0], texture='terrazzo')
    scene.add_visual('/floor/tiles/visual', *_merge(tiles[1]), 0, [0.68, 0.69, 0.7, 1.0], texture='terrazzo')

    # The shelves: boards white, plinths and panels grey, lips a price strip.
    boards, panels, plinths, lips = [], [], [], []
    for name, lo, hi in statics:
        part = _box(*lo, *hi)
        if name.startswith(('left_board', 'right_board')):
            (plinths if name.endswith('board0') else boards).append(part)
        elif '_lip' in name:
            lips.append(part)
        elif name.startswith(('left_', 'right_')):
            panels.append(part)
    store = '/world/store'
    if boards:
        scene.add_visual(f'{store}/shelves/boards/visual', *_flat_shaded(*_merge(boards)), 0, [0.93, 0.93, 0.91, 1.0],
                         texture='painted')
    if plinths:
        scene.add_visual(f'{store}/shelves/plinths/visual', *_flat_shaded(*_merge(plinths)), 0, [0.36, 0.38, 0.42, 1.0],
                         texture='brushed')
    if panels:
        scene.add_visual(f'{store}/shelves/panels/visual', *_flat_shaded(*_merge(panels)), 0, [0.55, 0.62, 0.7, 1.0],
                         texture='pegboard')
    if lips:
        scene.add_visual(f'{store}/shelves/lips/visual', *_flat_shaded(*_merge(lips)), 0, [0.25, 0.27, 0.3, 1.0],
                         texture='plastic')
    headers = []
    for sh in layout['shelves']:
        yb = sh['y_front'] + sh['into'] * sh['depth']
        y0h, y1h = sorted((yb, yb + sh['into'] * panel_t))
        headers.append(_box(sh['cx'] - sh['length'] / 2 - panel_t, y0h - 2e-3, sh['height'],
                            sh['cx'] + sh['length'] / 2 + panel_t, y1h + 2e-3, sh['height'] + 0.07))
    scene.add_visual(f'{store}/shelves/headers/visual', *_flat_shaded(*_merge(headers)), 0, [0.85, 0.2, 0.15, 1.0],
                     texture='plastic')

    # Price tags on the lips (one per place, in the product's colour) and the empty places' marks on the boards.
    tags, spots = {}, {}
    tag_h = max(lip_h, 6 * MM)
    for place in layout['places']:
        sh = next(s for s in layout['shelves'] if s['name'] == place['unit'])
        x, y = place['centre']
        fx, fy = place['size']
        yt = sh['y_front'] - sh['into'] * 0.6 * MM
        ya, yb2 = sorted((yt, yt - sh['into'] * 0.4 * MM))
        tags.setdefault(place['product'], []).append(
            _box(x - 0.35 * fx, ya, place['top'] + 0.1 * tag_h, x + 0.35 * fx, yb2, place['top'] + 0.9 * tag_h))
        if not place['filled']:
            spots.setdefault(place['product'], []).append(
                _box(x - fx / 2, y - fy / 2, place['top'] + 0.2 * MM, x + fx / 2, y + fy / 2, place['top'] + 0.8 * MM,
                     skip=('-Z',)))
    for product, parts in tags.items():
        scene.add_visual(f'{store}/tags/{product}/visual', *_flat_shaded(*_merge(parts)), 0,
                         list(ITEM_COLOURS[product]) + [1.0], texture='paper')
    if P['show_spots']:
        for product, parts in spots.items():
            scene.add_visual(f'{store}/spots/{product}/visual', *_flat_shaded(*_merge(parts)), 0,
                             list(ITEM_COLOURS[product]) + [float(P['spot_alpha'])], texture='paper')

    # Coordinate axes on the floor, at the origin.
    arrow_ref = 0.12
    head_len, shaft_half = arrow_ref * 0.2, arrow_ref * 0.015
    az = 1e-4
    for name, axis, color in (('arrow_x', 0, [0.9, 0.1, 0.1, 1.0]), ('arrow_y', 1, [0.1, 0.8, 0.1, 1.0])):
        pts, tris = _axis_arrow(axis, arrow_ref - head_len, head_len, shaft_half, shaft_half * 3)
        scene.add_visual(f'/{name}/visual', *_flat_shaded([[p[0], p[1], p[2] + az] for p in pts], tris), 0, color,
                         texture='painted')

    # The platform (in its own frame): a grey deck with a yellow bumper, black wheels, an arrow on top pointing
    # forward, and the crate.
    robot = '/world/robot'
    deck = [_box(-PL / 2, -PW / 2, 0.04, PL / 2, PW / 2, PH)]
    bumper = [_box(-PL / 2 - 0.012, -PW / 2 - 0.012, 0.07, PL / 2 + 0.012, PW / 2 + 0.012, 0.13)]
    wheels, hubs, steered = [], [], []
    if cart == 'omni':  # four mecanum wheels: rollers at 45 degrees round each rim
        for wx in (-PL / 2 + 0.16, PL / 2 - 0.16):
            for side in (-1, 1):
                wy = side * (PW / 2 - 0.02)
                hubs.append(_cylinder_along([wx, wy - 0.037, 0.075], [wx, wy + 0.037, 0.075], 0.05, n=20))
                slant = 1 if (wx > 0) == (side > 0) else -1
                for k in range(10):
                    a = 2 * math.pi * k / 10
                    c = np.array([wx + 0.062 * math.cos(a), wy, 0.075 + 0.062 * math.sin(a)])
                    tang = np.array([-math.sin(a), 0.0, math.cos(a)])
                    ax = (tang + slant * np.array([0.0, 1.0, 0.0])) / math.sqrt(2)
                    wheels.append(_cylinder_along(list(c - 0.022 * ax), list(c + 0.022 * ax), 0.012, n=8))
    elif cart == 'diff':  # two big driven wheels in the middle, a caster at each end
        for side in (-1, 1):
            wy = side * (PW / 2 - 0.005)
            wheels.append(_cylinder_along([0.0, wy - 0.03, 0.11], [0.0, wy + 0.03, 0.11], 0.11, n=32))
            hubs.append(_cylinder_along([0.0, wy - 0.032, 0.11], [0.0, wy + 0.032, 0.11], 0.045, n=16))
        for cx_ in (-PL / 2 + 0.1, PL / 2 - 0.1):
            wheels.append(_cylinder_along([cx_, -0.015, 0.035], [cx_, 0.015, 0.035], 0.035, n=16))
            hubs.append(_box(cx_ - 0.025, -0.03, 0.03, cx_ + 0.025, 0.03, 0.045))
    elif cart == 'tank':  # a track along each side: two drive sprockets and the belt round them
        for side in (-1, 1):
            wy = side * (PW / 2 + track_w / 2 + 0.005)
            y0_, y1_ = wy - track_w / 2, wy + track_w / 2
            xe = PL * 0.47 - 0.075
            for xc in (-xe, xe):
                wheels.append(_cylinder_along([xc, y0_, 0.08], [xc, y1_, 0.08], 0.075, n=24))
                hubs.append(_cylinder_along([xc, y0_ - 0.002, 0.08], [xc, y1_ + 0.002, 0.08], 0.03, n=12))
            wheels.append(_box(-xe, y0_, 0.005, xe, y1_, 0.155))
    else:  # car: four wheels, the front two steered (on their own bodies)
        for side in (-1, 1):
            wy = side * car_y
            wheels.append(_cylinder_along([-car_x, wy - 0.035, wheel_r], [-car_x, wy + 0.035, wheel_r], wheel_r, n=28))
            hubs.append(_cylinder_along([-car_x, wy - 0.037, wheel_r], [-car_x, wy + 0.037, wheel_r], 0.04, n=16))
        for name in ('steer_l', 'steer_r'):
            steered.append((name, _cylinder_along([0, -0.035, 0], [0, 0.035, 0], wheel_r, n=28),
                            _cylinder_along([0, -0.037, 0], [0, 0.037, 0], 0.04, n=16)))
    arrow = [([[PL / 2 - 0.05, 0, PH + 1e-3], [PL / 2 - 0.17, 0.06, PH + 1e-3], [PL / 2 - 0.17, -0.06, PH + 1e-3]],
              [[0, 1, 2]])]
    crate_parts = [_box(a0, b0, PH, a1, b1, PH + ch, skip=('-Z',)) for _, (a0, b0), (a1, b1) in crate_walls]
    crate_bottom = [_box(x0, y0, PH, x1, y1, PH + 0.3 * MM, skip=('-Z',))]
    for name, parts, colour, texture in (
            ('deck', deck, [0.3, 0.32, 0.36, 1.0], 'treadplate'), ('bumper', bumper, [0.95, 0.75, 0.1, 1.0], 'hazard'),
            ('wheels', wheels, [0.08, 0.08, 0.09, 1.0], 'rubber'), ('arrow', arrow, [0.95, 0.75, 0.1, 1.0], 'painted'),
            ('crate', crate_parts, [0.74, 0.56, 0.36, 1.0], 'cardboard'),
            ('crate_bottom', crate_bottom, [0.6, 0.44, 0.28, 1.0], 'cardboard'),
            ('hubs', hubs, [0.55, 0.57, 0.6, 1.0], 'brushed')):
        if parts:
            scene.add_visual(f'{robot}/platform/{name}/visual', *_smooth_shaded(*_merge(parts)), platform_id, colour,
                             texture=texture)
    for name, tyre, hub in steered:
        scene.add_visual(f'{robot}/platform/{name}/visual', *_smooth_shaded(*tyre), body(name),
                         [0.08, 0.08, 0.09, 1.0], texture='rubber')
        scene.add_visual(f'{robot}/platform/{name}_hub/visual', *_smooth_shaded(*hub), body(name),
                         [0.55, 0.57, 0.6, 1.0], texture='brushed')

    # The arm (each link in its own frame, which at zero is the platform's turned by nothing), in its maker's colours.
    shell_rgba, housing_rgba = spec['colours']
    origin = mount.copy()
    link_origins = []
    for axis, offset, _, _ in links:
        origin = origin + np.array(offset)
        link_origins.append(origin.copy())
    scene.add_visual(f'{robot}/arm/pedestal/visual',
                     *_smooth_shaded(*_frustum_z(mount[0], mount[1], PH, PH + pedestal, 0.075, 0.075, n=32)),
                     platform_id, [0.2, 0.22, 0.26, 1.0], texture='brushed')
    for i, (axis, offset, (p0, p1), r) in enumerate(links):
        o = link_origins[i]
        shell = _cylinder_along(list(np.array(p0)), list(np.array(p1)), r * 0.85, n=24)
        housing_axis = np.array(axis, float)
        # A joint offset sideways along its own axis (the shoulder, the elbow): the housing spans the offset, from
        # the previous link's centre line to this one's, so the links join up as on the real arm.
        along = -float(np.dot(offset, housing_axis)) if abs(housing_axis[2]) < 0.5 else 0.0
        housing = _cylinder_along(list(housing_axis * (min(0.0, along) - r * 1.1)),
                                  list(housing_axis * (max(0.0, along) + r * 1.1)), r * 1.12, n=24)
        bid = body(f'arm_l{i + 1}')
        scene.add_visual(f'{robot}/arm/link{i + 1}/visual', *_smooth_shaded(*shell), bid, shell_rgba,
                         texture='painted')
        scene.add_visual(f'{robot}/arm/joint{i + 1}/visual', *_smooth_shaded(*housing), bid, housing_rgba,
                         texture='painted')
    tool = np.array(spec['tool'])
    flange_disc = _cylinder_along(list(tool - 0.01 * tool_dir), list(tool + 0.004 * tool_dir), 0.045, n=24)
    scene.add_visual(f'{robot}/arm/flange/visual', *_smooth_shaded(*flange_disc), body(f'arm_l{n_arm}'),
                     [0.25, 0.27, 0.3, 1.0], texture='brushed')

    def carried(parts):  # [(points at rest, triangles, body id, body origin)] -> one mesh with a body per point
        pts, tris, bodies = [], [], []
        for p, t, b, o in parts:
            off = len(pts)
            pts += [list(np.array(q) - o) for q in p]
            tris += [[i + off for i in tri] for tri in t]
            bodies += [b] * len(p)
        return pts, tris, bodies

    seg_parts, sheet_parts, mark_parts, tip_parts = [], [], [], []
    for fr in frames:
        for j in range(N):
            xs = j * pitch
            pts, tris = _box(xs, -W / 2, 0.0, xs + L, W / 2, H)
            seg_parts.append(([fr.to_world(p) for p in pts], tris, body_of(fr.k, j), origin_of(fr.k, j)))
            sx0, sx1 = xs - (sep / 2 if j > 0 else 0.0), xs + L + (sep / 2 if j < N - 1 else 0.0)
            sp, st = _box(sx0, -W / 2 * 0.98, -0.4 * MM, sx1, W / 2 * 0.98, 0.0)
            sheet_parts.append(([fr.to_world(p) for p in sp], st, body_of(fr.k, j), origin_of(fr.k, j)))
            if j > 0:
                pv = pivots[(fr.k, j)]
                half_w = W / 2 + 0.6 * MM
                mp, mt = _cylinder_along(list(pv - half_w * fr.tangent), list(pv + half_w * fr.tangent), 1.2 * MM, n=16)
                mark_parts.append((*_smooth_shaded(mp, mt), body_of(fr.k, j), origin_of(fr.k, j)))
        if tips:
            tp, tt = _hexahedron(tip_corners(fr))
            tip_parts.append((tp, tt, body_of(fr.k, N - 1), origin_of(fr.k, N - 1)))
    pts, tris, bodies = carried(seg_parts)
    scene.add_visual('/world/angles/bodies/visu/visual', pts, tris, bodies, [0.15, 0.45, 0.95, float(P['wall_alpha'])],
                     texture='silicone')
    pts, tris, bodies = carried(sheet_parts)
    scene.add_visual('/world/angles/bodies/base_layer/visual', pts, tris, bodies,
                     [0.96, 0.94, 0.86, float(P['wall_alpha'])], texture='fabric')
    if tip_parts:
        fp = []
        for tp, tt, b, o in tip_parts:  # flat-shaded: unshared vertices per face
            p2, t2 = _flat_shaded(tp, tt)
            fp.append((p2, t2, b, o))
        pts, tris, bodies = carried(fp)
        scene.add_visual('/world/angles/bodies/tips/visual', pts, tris, bodies, [0.1, 0.25, 0.6, float(P['wall_alpha'])],
                         texture='silicone')
    if P['show_joint_marks'] and mark_parts:
        pts, tris, bodies = carried(mark_parts)
        scene.add_visual('/world/angles/bodies/joint_marks/visual', pts, tris, bodies,
                         [0.95, 0.6, 0.1, float(P['joint_alpha'])], texture='plastic')
    # The gripper's plate and its wrist adapter, up to the arm's flange.
    plate_pts, plate_tris = _frustum_z(0.0, 0.0, z_palm, z_palm + thick, plate_r, plate_r, n=max(24, 4 * F))
    wrist_pts, wrist_tris = _frustum_z(0.0, 0.0, z_palm + thick, z_palm + thick + 0.05, 1.6 * stem + 4 * MM,
                                       1.6 * stem + 4 * MM, n=24)
    cap_pts, cap_tris = _frustum_z(0.0, 0.0, z_palm + thick + 0.05, z_palm + thick + 0.058, 0.55 * plate_r,
                                   0.55 * plate_r, n=32)
    mount_pts, mount_tris = _smooth_shaded(*_merge([(plate_pts, plate_tris), (wrist_pts, wrist_tris),
                                                  (cap_pts, cap_tris)]))
    scene.add_visual('/world/angles/bodies/trunk/visual', mount_pts, mount_tris, flange_id, [0.4, 0.26, 0.13, 1.0],
                     texture='printed')
    if jaw_names:
        n_jaws = len(jaw_names)
        rail = (_box(-(stroke + jaw_t), -0.016, z_palm - 0.02, stroke + jaw_t, 0.016, z_palm) if n_jaws == 2 else
                _frustum_z(0, 0, z_palm - 0.02, z_palm, stroke + jaw_t, stroke + jaw_t, n=36))
        scene.add_visual('/world/angles/bodies/rail/visual', *_smooth_shaded(*rail), flange_id,
                         [0.3, 0.32, 0.36, float(P['wall_alpha'])], texture='brushed')
        jaw_parts, pad_parts = [], []
        for name, e in zip(jaw_names, jaw_dirs):
            t = np.array([-e[1], e[0], 0.0])
            def block(r0, r1, w, z0, z1, e=e, t=t):
                pts, tris = _box(r0, -w / 2, z0, r1, w / 2, z1)
                return [list(p[0] * e + p[1] * t + np.array([0, 0, p[2]])) for p in pts], tris
            jb = body(name)
            z1 = z_palm - 0.02
            jaw_parts.append((*block(stroke, stroke + jaw_t, jaw_w, z1 - jaw_l, z1), jb, np.zeros(3)))
            pad_parts.append((*block(stroke - 0.5 * MM, stroke, 0.9 * jaw_w, z1 - jaw_l + 0.004, z1 - 0.012), jb,
                              np.zeros(3)))
        for vname, parts, rgba, tex in (('jaws', jaw_parts, [0.72, 0.74, 0.78], 'brushed'),
                                        ('jaw_pads', pad_parts, [0.1, 0.1, 0.11], 'rubber')):
            fp = [(*_flat_shaded(pp, tt), b, o) for pp, tt, b, o in parts]
            pts, tris, bodies = carried(fp)
            scene.add_visual(f'/world/angles/bodies/{vname}/visual', pts, tris, bodies,
                             rgba + [float(P['wall_alpha'])], texture=tex)
    if gkind == 'suction':
        cup_id = body('cup')
        stem_m = _cylinder_along([0, 0, 0.016], [0, 0, z_palm], 0.009, n=16)
        bellows = _merge([_frustum_z(0, 0, 0.012 + 0.008 * k, 0.016 + 0.008 * k, 0.014, 0.011, n=24) for k in range(3)])
        lip = _frustum_z(0, 0, 0.0, 0.012, 0.02, 0.012, n=32)
        scene.add_visual('/world/angles/bodies/cup_stem/visual', *_smooth_shaded(*stem_m), cup_id,
                         [0.72, 0.74, 0.78, float(P['wall_alpha'])], texture='brushed')
        scene.add_visual('/world/angles/bodies/cup/visual', *_smooth_shaded(*_merge([bellows, lip])), cup_id,
                         [0.12, 0.12, 0.13, float(P['wall_alpha'])], texture='rubber')

    for item in items:
        key, product, shape, dims = item['key'], item['product'], item['shape'], item['dims']
        parts = [(p, t, ITEM_COLOURS[product]) for p, t, _ in _item_meshes(shape, dims, smooth=True)] + \
            _item_details(product, dims)
        textures = ITEM_TEXTURES[product]
        for n_part, (vp, vt, colour) in enumerate(parts):
            vp, vt = _smooth_shaded(vp, vt)
            scene.add_visual(f'/world/items/{key}/{"visu" if n_part == 0 else f"detail{n_part}"}/visual', vp, vt,
                             body(f'item_{key}'), list(colour) + [float(P['item_alpha'])],
                             texture=textures[min(n_part, len(textures) - 1)],
                             label={'mode': ITEM_LABELS[product][n_part], 'seed': product, 'overlay': n_part > 0}
                             if n_part in ITEM_LABELS.get(product, {}) else None)

    # ---- controllers
    jnt = lambda name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    act = lambda name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
    hinge_dofs = [int(model.jnt_dofadr[jnt(n)]) for n in hinge_names]
    arm_qpos = [int(model.jnt_qposadr[jnt(n)]) for n in arm_joints]
    arm_dofs = [int(model.jnt_dofadr[jnt(n)]) for n in arm_joints]
    ik = ArmIK(ik_model, limits)
    tcp_start = np.array([c * MM for c in TCP_START])
    q0 = ik.home(tcp_start, euler_to_quat(0.0, 0.0, 0.0), spec['seeds'])
    if q0 is None:
        raise RuntimeError("the arm can't reach its start pose: check the arm's dimensions and TCP_START")
    scene.data.qpos[arm_qpos] = q0
    scene.data.ctrl[[act(n) for n in arm_joints]] = q0
    mujoco.mj_forward(model, scene.data)

    motion = FlangeController(None, tcp_start, acceleration=float(P['flange_acceleration']),
                              speed=float(P['flange_speed']),
                              angular_acceleration=math.radians(float(P['angular_acceleration'])),
                              angular_speed=math.radians(float(P['angular_speed'])))
    arm = ArmController(motion, ik, [act(n) for n in arm_joints], q0, rate=float(P['ik_rate']),
                        max_jump=math.radians(float(P['ik_max_jump'])))
    arm.seeds = spec['seeds']
    drive = DriveController([act('drive_x'), act('drive_y'), act('drive_yaw')],
                            [int(model.jnt_qposadr[jnt(n)]) for n in ('base_x', 'base_y', 'base_yaw')],
                            lead=(2 * math.sqrt(m_total / kp_lin), 2 * math.sqrt(i_total / kp_yaw)), speed=float(P['drive_speed']), acceleration=float(P['drive_acceleration']),
                            turn_speed=math.radians(float(P['turn_speed'])),
                            turn_acceleration=math.radians(float(P['turn_acceleration'])), kind=cart,
                            wheelbase=2 * car_x, track=2 * car_y, steer_limit=math.radians(float(P['steer_limit'])),
                            steer_actuators=[act('steer_l'), act('steer_r')] if cart == 'car' else ())
    pressure = PressureRamp(hinge_dofs, torque_per_pa, 0.0, float(P['ramp_time']), instant=bool(P['instant']),
                            ramp_start=float(P['ramp_start']))
    scene.controllers += [drive, motion, arm, pressure]
    jaws = None
    if jaw_names:
        jaws = JawController(pressure, [int(model.jnt_dofadr[jnt(n)]) for n in jaw_names],
                             [int(model.jnt_qposadr[jnt(n)]) for n in jaw_names], stroke,
                             max(float(P['jaw_force']), 0.0))
        scene.controllers.append(jaws)
    if gkind == 'suction':
        scene.controllers.append(SuctionController(pressure, act('suction')))
    all_hinges = hinge_names + side_names
    repair = Repair([int(model.jnt_qposadr[jnt(n)]) for n in all_hinges],
                    [int(model.jnt_dofadr[jnt(n)]) for n in all_hinges], arm, arm_qpos, arm_dofs)
    scene.objects.update({'Flange': motion, 'Arm': arm, 'Drive': drive, 'PressureRamp': pressure, 'Repair': repair,
                          'Jaws': jaws})

    slots = [{'product': p['product'], 'x': p['centre'][0], 'y': p['centre'][1], 'cy': p['centre'][1],
              'hx': 0.5 * p['size'][0] + 0.5 * float(P['item_gap']) * MM, 'hy': 0.5 * layout['depth'],
              'top': p['top'], 'unit': p['unit'], 'level': p['level'], 'yaw': p['yaw']} for p in layout['places']]
    for sl in slots:  # the place spans the shelf's depth; the next shelf up is its ceiling
        sh = next(u for u in layout['shelves'] if u['name'] == sl['unit'])
        sl['y'] = sh['y_front'] + sh['into'] * 0.5 * layout['depth']
        sl['ceiling'] = (sh['tops'][sl['level'] + 1] - sh['board']) if sl['level'] + 1 < len(sh['tops']) else None

    # ---- the automatic stocking
    half_w = PW / 2 + (track_w + 0.005 if cart == 'tank' else 0.0)
    if gkind == 'pneunet':
        below = max(finger_len + (tip_l if tips else 0.0) - tcp_depth, 0.0)  # the fingertips, hanging straight
        palm = tcp_depth
    elif jaw_names:
        below, palm = jaw_l + 0.02 - tcp_depth, tcp_depth - 0.02
    else:
        below, palm = 0.0, tcp_depth
    stocker_plan = {
        'items': [{**it, 'body': body(f'item_{it["key"]}')} for it in items], 'slots': slots,
        'gripper': gkind, 'fingers': len(frames), 'stroke': stroke, 'cart': cart,
        'platform_id': platform_id,
        'flange_id': flange_id, 'mount': (float(mount[0]), float(mount[1])), 'inner': spec.get('inner', 0.25),
        'front': layout['front'], 'lane_y': layout['front'] - half_w - 0.2,
        'y_out': layout['front'] - max(plate_r, stroke + jaw_t if jaw_names else 0.0, 0.035) - 0.03,
        'half_diagonal': math.hypot(PL / 2, half_w), 'floor_x': half_floor - PL / 2 - 0.2,
        'carry': tuple(float(c) for c in tcp_start), 'lip': lip_h, 'reach_below_tcp': below,
        'palm_above_tcp': palm, 'above_tcp': grip_top + 0.2, 'settle': 0.6 if gkind == 'pneunet' else 0.35,
    }
    stocker = Stocker(motion, arm, drive, pressure, jaws, stocker_plan)
    scene.controllers.append(stocker)
    scene.objects['Stocker'] = stocker
    BUILD.clear()
    BUILD.update({
        'segments': N, 'fingers': F, 'flange_id': flange_id, 'platform_id': platform_id, 'tcp_depth': tcp_depth,
        'hinge_qpos': [int(model.jnt_qposadr[jnt(n)]) for n in hinge_names],
        'side_qpos': [int(model.jnt_qposadr[jnt(n)]) for n in side_names],
        'arm_qpos': arm_qpos,
        'items': [{**it, 'body': body(f'item_{it["key"]}')} for it in items],
        'slots': slots,
        'crates': [{'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'top': PH + ch, 'platform': True}],
        'torque_per_pa': torque_per_pa, 'seg_mass': seg_mass, 'plate_radius': plate_r,
        'finger_length': finger_len + (tip_l if tips else 0.0),
        'gripper': gkind, 'arm': P['arm'], 'cart': cart, 'arm_joints': n_arm,
    })
    return scene


def item_states(data):
    """Where each item (BUILD['items']) is: 'stocked' (standing in a place for its product), 'shelf' (on a shelf,
    elsewhere), 'crate' (in the platform's crate), 'floor', or 'moving' (held, falling, on the platform's deck...)."""
    items, slots, crates = BUILD['items'], BUILD['slots'], BUILD['crates']
    if 'slot_arrays' not in BUILD:
        BUILD['slot_arrays'] = {k: np.array([s[k] for s in slots]) for k in ('x', 'y', 'hx', 'hy', 'top')}
        BUILD['slot_products'] = np.array([s['product'] for s in slots])
        BUILD['item_arrays'] = (np.array([it['body'] for it in items]), np.array([it['half_h'] for it in items]),
                                np.array([it['product'] for it in items]))
    S = BUILD['slot_arrays']
    bodies, half, products = BUILD['item_arrays']
    q = data.xpos[bodies]
    x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3]
    hh = half[:, None]
    in_xy = (np.abs(x - S['x']) <= S['hx']) & (np.abs(y - S['y']) <= S['hy'])
    standing = in_xy & (np.abs(z - S['top'] - hh) < 0.5 * hh + 0.01)
    right = standing & (BUILD['slot_products'][None, :] == products[:, None])
    on_shelf = (np.abs(y - S['y']) <= S['hy']) & (z > S['top']) & (z < S['top'] + 2.5 * hh + 0.02)
    # The crate is on the platform: the items in the platform's frame.
    pid = BUILD['platform_id']
    local = (q - data.xpos[pid]) @ data.xmat[pid].reshape(3, 3)
    states = []
    for i in range(len(items)):
        if right[i].any():
            states.append('stocked')
        elif standing[i].any() or on_shelf[i].any():
            states.append('shelf')
        elif any(c['x0'] < local[i, 0] < c['x1'] and c['y0'] < local[i, 1] < c['y1'] and local[i, 2] < c['top'] + half[i]
                 for c in crates):
            states.append('crate')
        elif z[i, 0] < half[i] + 0.015:
            states.append('floor')
        else:
            states.append('moving')
    return states


def main():
    os.environ.setdefault('GRIPPER7MJ_NO_EDITOR', '1')
    scene = createScene()
    if '--headless' in sys.argv:
        import time
        start = time.time()
        while scene.time < 2.0:
            scene.step(scene.dt.value)
        print(f"2 s simulated in {time.time() - start:.2f} s")
        return
    import time
    import mujoco.viewer
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:  # (the controllers run in scene.step)
        while viewer.is_running():
            start = time.time()
            scene.step(scene.dt.value)
            viewer.sync()
            time.sleep(max(0.0, scene.dt.value - (time.time() - start)))


if __name__ == '__main__':
    main()
