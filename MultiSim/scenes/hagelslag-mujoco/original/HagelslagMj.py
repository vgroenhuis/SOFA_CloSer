# -*- coding: utf-8 -*-
"""
Hagelslag (MuJoCo): Gripper 7's mobile manipulator (a choice of gripper, arm and cart) at a real supermarket shelf:
the two-bay Hagelslag shelf (hagelslag_shelf/, modelled from photos: see its README.md), its 985 packages each a body
of its own with its own collision shape, from placements.json; the shelf itself (back panel, wire shelves, rails,
glass dividers) from shelf.glb's fixture nodes (fx_*), the packages' meshes from its product nodes (p_*), split into
the separate packages. The model's frame (+y up, +z towards the aisle) is turned into MuJoCo's (+z up), the shelf
facing -y as Gripper 7's.

  - Only the front `dynamic_rows` packages of each lane are free bodies (1: the front row, 176 of them); the ones
    behind are fixed (they still collide). All 985 free makes the simulation 2 to 10 times slower than real time
    (a lane the robot touches wakes every package in it).
  - Masses and frictions are placeholders, in one file: hagelslag_physics.json.
  - The cart carries a crate of a few of the shelf's products. No automatic stocking (Gripper 7's): the robot is
    driven by hand.
  - The robot, the gripper and the contacts are Gripper 7's (../../gripper-7-mujoco/original/Gripper7Mj.py).

Units: the scene is SI (metres, kg, s, Pa); params.json takes millimetres for lengths, as labelled.

Run with MuJoCo's viewer:   python HagelslagMj.py       (pip install mujoco)
Headless check:             python HagelslagMj.py --headless
Parameters: params.json next to this script (edit by hand, or with params_editor.py).
"""
import json
import math
import os
import struct
import subprocess
import sys

import numpy as np
import mujoco

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')

# The platform's crate: some of the shelf's products (placements.json's ids) and how many of each, packed in rows
# along the platform.
PLATFORM_CRATE = [('jb_ha_melk', 2), ('dr_ha_puur', 1), ('jb_minis', 1), ('nut_400', 2), ('pen_van', 1),
                  ('tin_rood', 1), ('cs_groen', 1)]

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
    'suction_force': 30.0,       # [N] the simple cup's pull at full suction (80 kPa and more)
    'finray_length': 80.0,       # [mm] Festo Fin Ray finger (DHAS-GF-80 size): below the jaw
    'finray_base': 30.0,         # [mm] its depth at the jaw (the triangle's base)
    'finray_stiffness': 0.12,    # [N m/rad] each flank hinge (the flanks' bending)
    'mia_torque': 1.0,           # [N m] Mia hand: each motor's largest torque at its finger's base joint
    'tactile_skin': True,        # sensing pads on the fingers' gripping faces (Touchlab / TG0 style e-skin)
    'cup_vacuum': 30.0,          # [kPa] a vacuum cup's (GripWiq, SoftGripping) vacuum at full grip (80 kPa)
    'cup_lip_stiffness': 150.0,  # [N/m] each segment of a vacuum cup's silicone lip
    'sg_spacing': 45.0,          # [mm] SoftGripping parallel base: between the two fingers on each side
    'sg_open_vacuum': 20.0,      # [kPa] SoftGripping: the negative pressure that opens the fingers wide
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
    'shelf_distance': 300.0,     # the shelf's front edge (its deck's; y = this)
    'shelf_x': 0.0,              # the shelf's middle, along x
    'dynamic_rows': 1,           # how many packages of each lane, from the front, are free bodies (0: all); the
                                 #   ones behind are fixed (they still collide). Masses and frictions:
                                 #   hagelslag_physics.json
    'crate_length': 440.0,       # the platform's crate, inside, along the platform
    'crate_width': 500.0,        # inside, across it
    'crate_height': 80.0,
    'crate_wall': 6.0,
    'crate_gap': 30.0,           # [mm] between packages in a crate (room for the fingers)
    'item_mass_scale': 1.0,      # times each package's mass (see hagelslag_physics.json)
    # -- contacts --
    'friction': 0.8,             # Coulomb friction coefficient of every contact
    'torsional_friction': 0.005, # [m] resistance to twisting about a contact's normal
    'rolling_friction': 0.002,   # [m] round packages (jars, tins, tubs): resistance to rolling
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
    'show_joint_marks': True,
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
    """Opens params_editor.py in its own process. Skipped when HAGELSLAGMJ_NO_EDITOR is set (headless runs)."""
    if os.environ.get('HAGELSLAGMJ_NO_EDITOR'):
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

    def __init__(self, k, n, radius, azimuth_deg, z_palm, shift=0.0):
        """Evenly round the axis (k of n, from `azimuth_deg`), `shift` [m] sideways (a parallel base's rows)."""
        self.k = k
        phi = math.radians(azimuth_deg + 360.0 * k / n)
        self.radial = np.array([math.cos(phi), math.sin(phi), 0.0])
        self.tangent = np.array([-math.sin(phi), math.cos(phi), 0.0])  # z x radial: curls the tip inwards
        self.down = np.array([0.0, 0.0, -1.0])
        self.origin = radius * self.radial + shift * self.tangent + np.array([0.0, 0.0, z_palm])

    def to_world(self, p):
        w = self.origin + p[0] * self.down + p[1] * self.tangent + p[2] * self.radial
        return [float(c) for c in w]


def _hexahedron(corners):
    """A closed convex hexahedron from its 8 corners in _box's order (a box deformed, e.g. a tapered tip)."""
    _, tris = _box(0, 0, 0, 1, 1, 1)
    return [list(map(float, c)) for c in corners], tris


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


# ----------------------------------------------------------------------
# The Hagelslag shelf: hagelslag_shelf/ (shelf.glb, placements.json; see its README.md), hagelslag_physics.json
# ----------------------------------------------------------------------
SHELF_DIR = os.path.join(_SCRIPT_DIR, 'hagelslag_shelf')
PHYSICS_FILE = os.path.join(_SCRIPT_DIR, 'hagelslag_physics.json')
_GLB_TYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_GLB_WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
_KIND_DEFAULT = {'density': 0.5, 'friction': 0.5, 'shape': 'box'}  # (a kind hagelslag_physics.json doesn't list)


def read_glb(path):
    """A binary glTF file: (its JSON, accessor(i) -> its values as a count x width array, image(i) -> (media type,
    bytes))."""
    with open(path, 'rb') as f:
        data = f.read()
    magic, _, _ = struct.unpack_from('<4sII', data, 0)
    if magic != b'glTF':
        raise ValueError(f"{path} is not a binary glTF file")
    chunks, off = {}, 12
    while off + 8 <= len(data):
        length, kind = struct.unpack_from('<I4s', data, off)
        chunks[kind] = data[off + 8:off + 8 + length]
        off += 8 + length
    gltf = json.loads(chunks[b'JSON'])
    binary = chunks.get(b'BIN\x00', b'')

    def view(i):
        bv = gltf['bufferViews'][i]
        start = bv.get('byteOffset', 0)
        return bv, binary[start:start + bv['byteLength']]

    def accessor(i):
        a = gltf['accessors'][i]
        bv, raw = view(a['bufferView'])
        dtype = np.dtype(_GLB_TYPES[a['componentType']])
        width = _GLB_WIDTH[a['type']]
        stride = bv.get('byteStride') or width * dtype.itemsize
        return np.array(np.ndarray((a['count'], width), dtype, buffer=raw, offset=a.get('byteOffset', 0),
                                   strides=(stride, dtype.itemsize)))

    def image(i):
        im = gltf['images'][i]
        return im.get('mimeType', 'image/jpeg'), bytes(view(im['bufferView'])[1])

    return gltf, accessor, image


def load_physics(path=PHYSICS_FILE):
    """hagelslag_physics.json: the packages' placeholder masses and frictions (see its _about)."""
    with open(path, 'r') as f:
        cfg = json.load(f)
    cfg.setdefault('kinds', {})
    cfg.setdefault('products', {})
    cfg.setdefault('fixtures', {})
    return cfg


def _components(tris, n):
    """A mesh's connected parts: a label per triangle (triangles that share a vertex have the same one)."""
    lab = np.arange(n)
    while True:
        new = lab.copy()
        np.minimum.at(new, tris.ravel(), np.repeat(lab[tris].min(axis=1), 3))
        new = new[new]  # (pointer jumping: converges in a few passes)
        if np.array_equal(new, lab):
            return lab[tris[:, 0]]
        lab = new


def hagelslag_shelf(P):
    """The Hagelslag shelf in the scene's frame [m]. The model's frame (metres, +y up, +z towards the aisle, origin on
    the floor at its back left) is turned a quarter turn about x into MuJoCo's (+z up): the shelf faces -y, its deck's
    front edge at y = shelf_distance and its middle at x = shelf_x.

    Returns {'fixtures': collision boxes [{'name', 'group' (its friction in hagelslag_physics.json), 'lo', 'hi'}],
    'visuals': the fixtures' meshes [{'name', 'points', 'triangles', 'uv', 'image', 'rgba'}], 'packages': one per
    placements.json entry [{'index', 'product', 'name', 'kind', 'row' (0: the front of its lane), 'centre', 'size',
    'half' (its collision shape's half sizes), 'shape', 'mass', 'friction', 'parts' (its meshes, about its centre)}],
    'images': {name: (media type, bytes)}, 'width', 'depth', 'height', 'front', 'levels' (the shelves' tops, bay by
    bay), 'physics'}."""
    gltf, accessor, image = read_glb(os.path.join(SHELF_DIR, 'shelf.glb'))
    with open(os.path.join(SHELF_DIR, 'placements.json'), 'r') as f:
        units = json.load(f)['units']
    physics = load_physics()
    nodes = {n['name']: n for n in gltf['nodes'] if 'mesh' in n}
    for n in nodes.values():
        if any(k in n for k in ('matrix', 'translation', 'rotation', 'scale')):
            raise ValueError(f"shelf.glb: node {n['name']} has a transform (not expected)")
    images, image_of = {}, {}

    def material_look(pr):
        """A primitive's colour (rgba) and image name (or None): its material's base colour and texture."""
        mat = gltf['materials'][pr['material']] if 'material' in pr else {}
        pbr = mat.get('pbrMetallicRoughness', {})
        rgba = list(pbr.get('baseColorFactor', [1.0, 1.0, 1.0, 1.0]))
        if mat.get('alphaMode', 'OPAQUE') != 'BLEND':
            rgba[3] = 1.0
        tex = pbr.get('baseColorTexture')
        name = None
        if tex is not None and 'TEXCOORD_0' in pr['attributes']:
            src = gltf['textures'][tex['index']]['source']
            if src not in image_of:
                mime, raw = image(src)
                name = f"{mat.get('name', 'image')}_{src}.{'png' if mime == 'image/png' else 'jpg'}"
                images[name] = (mime, raw)
                image_of[src] = name
            name = image_of[src]
        return rgba, name

    def primitives(node):  # [(points, triangles, uv or None, rgba, image)] in the model's frame
        out = []
        for pr in gltf['meshes'][nodes[node]['mesh']]['primitives']:
            pts = accessor(pr['attributes']['POSITION']).astype(np.float64)
            tris = accessor(pr['indices']).reshape(-1, 3).astype(np.int64)
            rgba, img = material_look(pr)
            uv = accessor(pr['attributes']['TEXCOORD_0']).astype(np.float64) if img else None
            out.append((pts, tris, uv, rgba, img))
        return out

    deck = primitives('fx_deck')[0][0]
    mid_x, z_front = 0.5 * (deck[:, 0].min() + deck[:, 0].max()), deck[:, 2].max()
    sx = float(P['shelf_x']) * MM
    front = max(float(P['shelf_distance']), 125.0) * MM

    def to_scene(p):  # model (x, y up, z to the aisle) -> scene (x, y away from the aisle, z up)
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        return np.stack([p[:, 0] - mid_x + sx, front + z_front - p[:, 2], p[:, 1]], axis=1)

    # ---- the fixtures: their meshes as they are; for the physics, their boxes (the meshes are boxes, 24 points
    # each), a shelf's wires as one plate, the price tags left out
    visuals, fixtures = [], []
    for node in sorted(nodes):
        if not node.startswith('fx_'):
            continue
        boxes = []
        for k, (pts, tris, uv, rgba, img) in enumerate(primitives(node)):
            q = to_scene(pts)
            visuals.append({'name': node[3:] + (f'_{k}' if k else ''), 'points': q, 'triangles': tris, 'uv': uv,
                            'image': img, 'rgba': rgba})
            if len(q) % 24 == 0:
                boxes += [(q[i:i + 24].min(axis=0), q[i:i + 24].max(axis=0)) for i in range(0, len(q), 24)]
            else:
                boxes.append((q.min(axis=0), q.max(axis=0)))
        if node == 'fx_tags':
            continue
        if node == 'fx_wire':  # by bay (left or right of the middle) and height
            boxes.sort(key=lambda b: (b[0][0] + b[1][0] > 2 * sx, b[0][2]))
            plates = []
            for lo, hi in boxes:
                last = plates[-1] if plates else None
                if last and (last[0][0] + last[1][0] > 2 * sx) == (lo[0] + hi[0] > 2 * sx) and lo[2] - last[0][2] < 0.015:
                    plates[-1] = (np.minimum(last[0], lo), np.maximum(last[1], hi))
                else:
                    plates.append((lo.copy(), hi.copy()))
            boxes = plates
        group = {'fx_wire': 'wire', 'fx_glass': 'glass', 'fx_rail': 'rail'}.get(node, 'panel')
        fixtures += [{'name': f'{node[3:]}{i}', 'group': group, 'lo': lo, 'hi': hi} for i, (lo, hi) in enumerate(boxes)]

    # ---- the packages: placements.json's boxes; each product's merged mesh split into its connected parts, each
    # part given to the nearest package of that product
    clearance = max(float(physics.get('clearance_mm', 1.0)), 0.0) * MM
    lanes = {}
    for i, u in enumerate(units):
        key = (u['bay'], u['shelf'], round(u['center_m'][0], 3), round(u['center_m'][1], 3))
        lanes.setdefault(key, []).append(i)
    row_of = {}
    for members in lanes.values():  # the front of the lane first (the model's +z)
        for r, i in enumerate(sorted(members, key=lambda i: -units[i]['center_m'][2])):
            row_of[i] = r
    parts = {i: [] for i in range(len(units))}
    by_product = {}
    for i, u in enumerate(units):
        by_product.setdefault(u['product'], []).append(i)
    for product, members in by_product.items():
        node = f'p_{product}'
        if node not in nodes:
            print(f"[shelf] no mesh {node} in shelf.glb: its packages are drawn as plain boxes")
            continue
        centres = np.array([units[i]['center_m'] for i in members])
        for pts, tris, uv, rgba, img in primitives(node):
            labels = _components(tris, len(pts))
            for lab in np.unique(labels):
                t = tris[labels == lab]
                used, local_t = np.unique(t, return_inverse=True)
                k = members[int(np.argmin(np.linalg.norm(centres - pts[used].mean(axis=0), axis=1)))]
                parts[k].append((pts[used], local_t.reshape(-1, 3), None if uv is None else uv[used], rgba, img))
    kinds, overrides = physics['kinds'], physics['products']
    packages = []
    for i, u in enumerate(units):
        cfg = {**_KIND_DEFAULT, **kinds.get(u['kind'], {}), **overrides.get(u['product'], {})}
        w, h, d = (float(v) for v in u['size_m'])  # (the model's x, y up, z depth)
        size = np.array([w, d, h])  # the scene's x, y (depth), z (up)
        centre = to_scene(u['center_m'])[0]
        mass = float(cfg['mass']) if 'mass' in cfg else float(cfg['density']) * w * h * d * 1000.0
        shape = 'cylinder' if cfg.get('shape') == 'cylinder' else 'box'
        half = np.maximum(size / 2 - clearance / 2, 0.002)
        if shape == 'cylinder':
            half[0] = half[1] = min(half[0], half[1])
        meshes = []
        for pts, tris, uv, rgba, img in parts[i]:
            meshes.append((to_scene(pts) - centre, tris, uv, rgba, img))
        if not meshes:  # (no mesh found: a plain box)
            bp, bt = _box(*(-size / 2), *(size / 2))
            meshes.append((np.asarray(bp, float), np.asarray(bt), None, [0.7, 0.6, 0.45, 1.0], None))
        packages.append({'index': i, 'product': u['product'], 'name': u['name'], 'kind': u['kind'],
                         'bay': u['bay'], 'shelf': u['shelf'], 'row': row_of[i], 'centre': centre, 'size': size,
                         'half': half, 'shape': shape, 'mass': mass, 'friction': float(cfg['friction']),
                         'parts': meshes})

    # Standing on what's under it (the fixtures, or a package it's stacked on): its collision shape 0.1 mm above the
    # highest top under its footprint, from the lowest package up.
    los = [f['lo'] for f in fixtures]
    his = [f['hi'] for f in fixtures]
    for pk in sorted(packages, key=lambda pk: pk['centre'][2] - pk['size'][2] / 2):
        c, s = pk['centre'], pk['size']
        lo_a, hi_a = np.array(los), np.array(his)
        bottom = c[2] - s[2] / 2
        over = ((np.minimum(hi_a[:, 0], c[0] + s[0] / 2) - np.maximum(lo_a[:, 0], c[0] - s[0] / 2) > 0.002)
                & (np.minimum(hi_a[:, 1], c[1] + s[1] / 2) - np.maximum(lo_a[:, 1], c[1] - s[1] / 2) > 0.002)
                & (hi_a[:, 2] < bottom + 0.01))
        if over.any():
            pk['centre'] = c + np.array([0.0, 0.0, hi_a[over, 2].max() + pk['half'][2] + 1e-4 - c[2]])
        else:
            print(f"[shelf] package {pk['index']} ({pk['name']}) stands on nothing")
        los.append(pk['centre'] - pk['half'])
        his.append(pk['centre'] + pk['half'])
    all_pts = np.concatenate([v['points'] for v in visuals])
    levels = {}
    for f in fixtures:
        if f['group'] == 'wire' or f['name'].startswith(('deck', 'plate')):
            levels.setdefault('right' if f['lo'][0] + f['hi'][0] > 2 * sx else 'left', []).append(float(f['hi'][2]))
    return {'fixtures': fixtures, 'visuals': visuals, 'packages': packages, 'images': images,
            'width': float(all_pts[:, 0].max() - all_pts[:, 0].min()), 'x0': float(all_pts[:, 0].min()),
            'x1': float(all_pts[:, 0].max()), 'depth': float(all_pts[:, 1].max() - front), 'front': front,
            'height': float(all_pts[:, 2].max()), 'levels': {k: sorted(v) for k, v in levels.items()},
            'physics': physics}


def pack_crate(contents, sizes, cl, cgap):
    """Packages for a crate of inside length `cl` (along x): rows along x, 30 mm from the walls, centred; `sizes` the
    products' (x, y) footprints. [{'product', 'centre' (relative to the crate's centre)}]."""
    rows, row, row_w = [], [], 0.0
    for product in [p for p, n in contents for _ in range(n)]:
        fx, fy = sizes[product]
        if row and row_w + cgap + fx > cl - 60 * MM:
            rows.append(row)
            row, row_w = [], 0.0
        row.append((product, fx, fy))
        row_w += (cgap if len(row) > 1 else 0.0) + fx
    if row:
        rows.append(row)
    depths = [max(fy for _, _, fy in r) for r in rows]
    y = -(sum(depths) + cgap * (len(rows) - 1)) / 2
    spots = []
    for r, d in zip(rows, depths):
        x = -(sum(fx for _, fx, _ in r) + cgap * (len(r) - 1)) / 2
        for product, fx, fy in r:
            spots.append({'product': product, 'centre': [x + fx / 2, y + d / 2]})
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
        self.idle = 0.0  # [Pa] the pressure while not gripping (negative: a vacuum opens the fingers wide)

    def goal(self):
        """The pressure it's heading for: the target, or the resting pressure when the target is 0."""
        return self.targetPressure if self.targetPressure > 0 else self.idle

    def __call__(self, scene, h):
        target = self.goal() if scene.data.time >= self.rampStart - 1e-9 else 0.0
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
GRIPPER_CHOICES = [('pneunet', 'Soft PneuNet fingers (Gripper 5)'),
                   ('softgripping_centric', 'SoftGripping SoftGripper, centric base'),
                   ('softgripping_parallel', 'SoftGripping SoftGripper, parallel base'),
                   ('softgripping_suction', 'SoftGripping SoftGripper, centric with suction cup'),
                   ('gripwiq_st', 'GripWiq SofTouch suction cup, ST'),
                   ('gripwiq_cs', 'GripWiq SofTouch suction cup, CS (larger)'),
                   ('festo_finray', 'Festo Fin Ray fingers (DHAS) on parallel jaws'),
                   ('mia', 'Prensilia Mia hand'),
                   ('parallel', 'Rigid parallel jaws'), ('three_jaw', 'Rigid three-jaw chuck'),
                   ('suction', 'Simple suction cup (adhesion)')]
JAW_KINDS = ('parallel', 'three_jaw', 'festo_finray')
SOFT_FINGERS = ('pneunet', 'softgripping_centric', 'softgripping_parallel', 'softgripping_suction')
VACUUM_CUPS = {'gripwiq_st': (0.020, 12), 'gripwiq_cs': (0.030, 16), 'softgripping_suction': (0.016, 10)}  # (r, lips)
CART_CHOICES = [('omni', 'Omniwheels (any direction)'), ('diff', 'Two-wheel differential drive'),
                ('tank', 'Tank tracks (skid steering)'), ('car', 'Car-like (front-wheel steering)')]


def cart_wheels(cart, PL, PW, car_x, car_y, wheel_r, track_w):
    """The cart's wheels: name, its body's parent and position there, `at` (x, y in the platform), radius `r`
    (0: not turned), mecanum `slant` (0: a plain wheel), `well` (half length, depth in from the side, top) for the
    deck's cut-out, and what it looks like (`kind`, `width`, `hub`)."""
    out = []

    def add(name, x, y, z, r, width, kind, slant=0, well=None, hub=0.0, parent='platform', pos=None):
        out.append({'name': name, 'parent': parent, 'pos': pos if pos is not None else [x, y, z], 'at': (x, y),
                    'r': r, 'slant': slant, 'well': well, 'width': width, 'kind': kind, 'hub': hub})
    if cart == 'omni':  # four mecanum wheels in the corners; the lowest rollers' axes point at the platform's middle
        for i, wx in enumerate((-PL / 2 + 0.16, PL / 2 - 0.16)):
            for side in (-1, 1):
                wy = side * (PW / 2 - 0.045)
                add(f'wheel_{"rf"[i]}{"rl"[side > 0]}', wx, wy, 0.075, 0.075, 0.074, 'mecanum',
                    slant=-1 if (wx > 0) == (side > 0) else 1, well=(0.092, 0.092, 0.165), hub=0.05)
    elif cart == 'diff':  # two big driven wheels in the middle
        for side in (-1, 1):
            wy = side * (PW / 2 - 0.04)
            add(f'wheel_{"rl"[side > 0]}', 0.0, wy, 0.11, 0.11, 0.06, 'tyre', well=(0.125, 0.085, 0.235), hub=0.045)
    elif cart == 'tank':  # the drive sprockets at the tracks' ends
        xe = PL * 0.47 - 0.075
        for i, xc in enumerate((-xe, xe)):
            for side in (-1, 1):
                add(f'sprocket_{"rf"[i]}{"rl"[side > 0]}', xc, side * (PW / 2 + track_w / 2 + 0.005), 0.08, 0.075,
                    track_w, 'sprocket', hub=0.03)
    else:  # car: rear wheels fixed, the front ones on the steering bodies
        for side in (-1, 1):
            add(f'wheel_r{"rl"[side > 0]}', -car_x, side * car_y, wheel_r, wheel_r, 0.07, 'tyre',
                well=(0.11, 0.075, wheel_r * 2 + 0.015), hub=0.04)
        for side, steer in ((1, 'steer_l'), (-1, 'steer_r')):
            add(f'wheel_f{"rl"[side > 0]}', car_x, side * car_y, wheel_r, wheel_r, 0.07, 'tyre',
                well=(0.13, 0.12, wheel_r * 2 + 0.015), hub=0.04, parent=steer, pos=[0.0, 0.0, 0.0])
    return out


def wheel_meshes(w):
    """A wheel in its own frame (its centre, the axle along y): (tyre or rollers, hub) parts."""
    hw, r = w['width'] / 2, w['r']
    if w['kind'] == 'mecanum':  # rollers at 45 degrees round the rim, between two side plates
        rollers = []
        for k in range(10):
            a = 2 * math.pi * k / 10
            c = np.array([(r - 0.013) * math.cos(a), 0.0, (r - 0.013) * math.sin(a)])
            tang = np.array([-math.sin(a), 0.0, math.cos(a)])
            ax = (tang + w['slant'] * np.array([0.0, 1.0, 0.0])) / math.sqrt(2)
            rollers.append(_cylinder_along(list(c - 0.022 * ax), list(c + 0.022 * ax), 0.012, n=8))
        hub = [_cylinder_along([0, -hw, 0], [0, -hw + 0.006, 0], w['hub'], n=20),
               _cylinder_along([0, hw - 0.006, 0], [0, hw, 0], w['hub'], n=20),
               _cylinder_along([0, -hw, 0], [0, hw, 0], 0.02, n=12)]
        return rollers, hub
    if w['kind'] == 'sprocket':
        return ([_cylinder_along([0, -hw, 0], [0, hw, 0], r, n=24)],
                [_cylinder_along([0, -hw - 0.002, 0], [0, hw + 0.002, 0], w['hub'], n=12)])
    return ([_cylinder_along([0, -hw, 0], [0, hw, 0], r, n=28)],
            [_cylinder_along([0, -hw - 0.002, 0], [0, hw + 0.002, 0], w['hub'], n=16)])


def _notched_box(x0, y0, z0, x1, y1, z1, wells):
    """A box with wells cut in from both its long sides (y0, y1): `wells` [(xa, xb, depth, top)], each from the
    bottom up to `top` (and x from xa to xb); as boxes: a slab above the wells, bands along x below."""
    if not wells:
        return [_box(x0, y0, z0, x1, y1, z1)]
    top = min(max(t for _, _, _, t in wells), z1)
    parts = [_box(x0, y0, top, x1, y1, z1)] if top < z1 else []
    if top <= z0:
        return parts
    cuts = sorted(wells)
    x = x0
    for xa, xb, depth, _ in cuts + [(x1, x1, 0.0, 0.0)]:
        xa, xb = max(xa, x0), min(xb, x1)
        if xa > x:
            parts.append(_box(x, y0, z0, xa, y1, top))
        if xb > xa and depth > 0:
            parts.append(_box(xa, y0 + depth, z0, xb, y1 - depth, top))
        x = max(x, xb)
    return parts


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
                'meshes': 'kuka.npz', 'mesh_colours': {(0.4, 0.4, 0.4): [0.8, 0.81, 0.82]},  # (KUKA's light grey)
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
                'meshes': 'franka.npz',
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


def arm_meshes(spec):
    """The real arm's link meshes (arm_meshes/, from MuJoCo Menagerie: see its README): {link: [(points in the arm
    base's frame at zero, on its pedestal), triangles, rgba]}, link 0 the base; {} for an arm without them."""
    if not spec.get('meshes'):
        return {}
    with np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arm_meshes', spec['meshes'])) as f:
        index = json.loads(str(f['index']))
        out = {}
        for i, (k, rgba) in enumerate(index):
            rgb = spec.get('mesh_colours', {}).get(tuple(round(c, 3) for c in rgba[:3]), rgba[:3])
            out.setdefault(k, []).append((f[f'p{i}'].astype(float), f[f't{i}'].tolist(), list(rgb) + [1.0]))
    return out


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
                 turn_acceleration=3.0, kind='omni', wheelbase=0.6, track=0.6, steer_limit=0.6, steer_actuators=(),
                 dofs=(), wheels=()):
        self.actuators, self.qpos, self.lead = list(actuators), list(qpos), lead
        self.speed, self.acceleration = speed, acceleration
        self.turn_speed, self.turn_acceleration = turn_speed, turn_acceleration
        self.kind, self.wheelbase, self.track, self.steer_limit = kind, wheelbase, track, steer_limit
        self.steer_actuators = list(steer_actuators)
        # The wheels, only seen turning: [(qpos, dof, x, y, r, slant, steer qpos or None)], `dofs` the platform's.
        self.dofs, self.wheels = list(dofs), list(wheels)
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
        if self.wheels:
            self.spin(scene.data, h)

    def spin(self, d, h):
        """Each wheel turns as it rolls with the platform's actual motion, at its contact point: a plain wheel at the
        speed along its heading; a mecanum wheel so its rim matches the motion along the axis of its lowest roller
        (slant +-1: the roller's axis forward and to the left or right), which makes it roll sideways: going to the
        left, the front-left and back-right wheels turn backwards, the others forwards."""
        qth = float(d.qpos[self.qpos[2]])
        c, s = math.cos(qth), math.sin(qth)
        wx, wy, w = (float(d.qvel[i]) for i in self.dofs)
        vx, vy = c * wx + s * wy, -s * wx + c * wy  # in the platform's frame
        for qpos, dof, x, y, r, slant, steer in self.wheels:
            ux, uy = vx - w * y, vy + w * x  # the velocity at the wheel
            if slant:
                rate = (ux + slant * uy) / r
            else:
                a = float(d.qpos[steer]) if steer is not None else 0.0
                rate = (math.cos(a) * ux + math.sin(a) * uy) / r
            d.qvel[dof] = rate
            d.qpos[qpos] = math.remainder(float(d.qpos[qpos]) + rate * h, 2 * math.pi)


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


class VacuumCup:
    """A vacuum suction cup (GripWiq SofTouch, SoftGripping's): its flexible silicone lip is a ring of segments, each
    on a sprung slide joint so the ring follows a curved or uneven surface. Under vacuum (the grip control: full at
    80 kPa) it pulls the item it touches with the vacuum times the area inside the lip, along the cup's axis (and the
    cup the other way), as far as the lip seals: the share of the lip's segments touching that one item (a tilted or
    small surface leaks)."""

    def __init__(self, ramp, cup_body, lip_geoms, area, vacuum):
        self.ramp, self.cup_body, self.area, self.vacuum = ramp, cup_body, area, vacuum
        self.lip_geoms = set(int(g) for g in lip_geoms)
        self.n = max(len(self.lip_geoms), 1)
        self.seal = 0.0
        self.held = -1
        self.force = 0.0

    def __call__(self, scene, h):
        self._tick = getattr(self, '_tick', -1) + 1
        if self._tick % 4:  # (every 4 ms: the force stays applied in between)
            return
        d, m = scene.data, scene.model
        if self.held >= 0:
            d.xfrc_applied[self.held] = 0.0
        d.xfrc_applied[self.cup_body] = 0.0
        self.held, self.force = -1, 0.0
        if not hasattr(self, '_lips'):
            self._lips = np.array(sorted(self.lip_geoms))
        g1, g2 = d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]
        on1, on2 = np.isin(g1, self._lips), np.isin(g2, self._lips)
        lip = np.concatenate([g1[on1 & ~on2], g2[on2 & ~on1]])
        other = np.concatenate([g2[on1 & ~on2], g1[on2 & ~on1]])
        if not len(lip):
            self.seal = 0.0
            return
        bodies = m.geom_bodyid[other]
        best, count = 0, 0
        for b_ in np.unique(bodies):
            n_ = len(np.unique(lip[bodies == b_]))
            if n_ > count:
                best, count = int(b_), n_
        body = best
        self.seal = count / self.n
        dp = self.vacuum * max(0.0, min(1.0, self.ramp.current_pressure / 80e3))
        if dp <= 0 or body == 0:
            return
        sealed = max(0.0, min(1.0, (self.seal - 0.55) / 0.35))
        f = dp * self.area * sealed * sealed * (3 - 2 * sealed)
        if f <= 0:
            return
        axis = d.xmat[self.cup_body].reshape(3, 3)[:, 2]  # the cup's +z: back into the gripper
        point = d.xpos[self.cup_body]
        F = f * axis
        d.xfrc_applied[body, :3] = F
        d.xfrc_applied[body, 3:] = np.cross(point - d.xipos[body], F)
        d.xfrc_applied[self.cup_body, :3] = -F
        d.xfrc_applied[self.cup_body, 3:] = np.cross(point - d.xipos[self.cup_body], -F)
        self.held, self.force = body, f


class HandController:
    """The Mia hand's three motors from the grip control (0: open, a little past straight; 80 kPa and more:
    closed): position servos whose torque is limited, so a finger stops on the item and the others close on."""

    OPEN, CLOSED = -0.45, 1.45  # [rad] at the fingers' base joints

    def __init__(self, ramp, actuators):
        self.ramp, self.actuators = ramp, list(actuators)

    def __call__(self, scene, h):
        f = max(0.0, min(1.0, self.ramp.current_pressure / 80e3))
        scene.data.ctrl[self.actuators] = self.OPEN + f * (self.CLOSED - self.OPEN)


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
    shelf = hagelslag_shelf(P)
    physics = shelf['physics']

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
    if gkind == 'softgripping_parallel':  # two fingers on each side, facing each other
        spacing = max(float(P['sg_spacing']), 0.0) * MM
        F = 4
        frames = [FingerFrame(k, F, radius, 0.0, z_palm) for k in range(F)]
        for k, (side, shift) in enumerate(((0, -1), (0, 1), (1, -1), (1, 1))):
            frames[k] = FingerFrame(k, 2, radius, 0.0 if side == 0 else 180.0, z_palm, shift * spacing / 2)
            frames[k].k = k
    elif gkind in SOFT_FINGERS:
        frames = [FingerFrame(k, F, radius, float(P['finger_azimuth']), z_palm) for k in range(F)]
    else:
        frames = []
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

    # The platform's crate: at the platform's back (its frame: x forward, y left), packed with PLATFORM_CRATE.
    cl, cw = max(float(P['crate_length']), 80.0) * MM, max(float(P['crate_width']), 60.0) * MM
    ch, ct = max(float(P['crate_height']), 20.0) * MM, max(float(P['crate_wall']), 2.0) * MM
    front_of = {}  # a product's front-row package (the crate's copies look like it)
    for pk in shelf['packages']:
        if pk['product'] not in front_of or pk['row'] < front_of[pk['product']]['row']:
            front_of[pk['product']] = pk
    crate_contents = [(p_, n_) for p_, n_ in PLATFORM_CRATE if p_ in front_of]
    spots = pack_crate(crate_contents, {p_: tuple(front_of[p_]['size'][:2]) for p_, _ in crate_contents}, cl,
                       max(float(P['crate_gap']), 0.0) * MM)
    ccx = -max(float(P['platform_length']), 100.0) * MM / 2 + 0.03 + ct + cl / 2
    for spot in spots:
        spot['centre'] = [ccx + spot['centre'][0], spot['centre'][1]]
    crate = {'centre': [ccx, 0.0], 'inner': (cl, cw), 'height': ch, 'wall': ct, 'items': spots}
    rows = int(P['dynamic_rows'])
    n_free = sum(1 for pk in shelf['packages'] if rows <= 0 or pk['row'] < rows)
    gripper_text = (f"{F} soft fingers of {N} segments" + (f", pointy tips {tip_l / MM:.0f} mm" if tips else "")
                    if gkind == 'pneunet' else dict(GRIPPER_CHOICES)[gkind])
    print(f"[model] a {PL / MM:.0f} x {PW / MM:.0f} mm cart ({dict(CART_CHOICES)[cart].lower()}) with a "
          f"{spec['label']} arm; {gripper_text}; {len(shelf['packages'])} packages on the shelf ({n_free} free), "
          f"{len(crate['items'])} in the platform's crate")

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

    assets, world = [], []
    sx = float(P['shelf_x']) * MM
    half_floor = max(max(abs(shelf['x0']), abs(shelf['x1'])) + 1.0, shelf['front'] + shelf['depth'] + 1.5)
    world.append(f'<geom name="floor" type="plane" size="{_v([2 * half_floor, 2 * half_floor, 0.05])}" '
                 f'pos="0 0 {floor_z}" class="static"/>')

    # ---- the shelf (shelf.glb's fixtures): boxes; a shelf's wires one plate; each part's friction from
    # hagelslag_physics.json
    tors = float(P['torsional_friction'])
    fixture_mu = physics['fixtures']
    for fx in shelf['fixtures']:
        lo, hi = fx['lo'], fx['hi']
        mu_f = max(float(fixture_mu.get(fx['group'], 0.4)), 0.0)
        world.append(f'<geom name="shelf_{fx["name"]}" type="box" size="{_v(np.maximum((hi - lo) / 2, 0.0006))}" '
                     f'pos="{_v((hi + lo) / 2)}" friction="{mu_f:.9g} {tors:.9g} 0.0001" class="static"/>')

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
    equalities, finray = [], {}  # (Fin Ray: per jaw, its bodies' names, points and front-flank geoms)
    if gkind in JAW_KINDS:
        n_jaws = 3 if gkind == 'three_jaw' else 2
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
                        f'damping="150" armature="0.02"/>']
            if gkind != 'festo_finray':
                gripper.append(f'<geom name="{name}" type="box" size="{_v([jaw_t / 2, jaw_w / 2, jaw_l / 2])}" '
                               f'pos="{_v(centre)}" xyaxes="{_v(e)} {_v(t)}" mass="0.06" '
                               f'friction="1.2 0.01 0.0001" class="finger"/>')
            else:
                # A Fin Ray finger hanging from the jaw: the front flank (its gripping face) and the back flank, each
                # a chain of n hinged segments, meeting at the tip; ribs across, pivoted on the front flank and
                # pinned to the back flank. Pressed anywhere on its front, the tip curls round the item.
                n_fr = 6
                Lf = max(float(P['finray_length']), 20.0) * MM
                wb = max(float(P['finray_base']), 5.0) * MM
                k_fr = max(float(P['finray_stiffness']), 1e-4)
                z0 = z_palm - 0.02
                Pf = [e * stroke + np.array([0, 0, z0 - i * Lf / n_fr]) for i in range(n_fr + 1)]
                Qb = [e * (stroke + wb * (1 - i / n_fr)) + np.array([0, 0, z0 - i * Lf / n_fr]) for i in range(n_fr + 1)]
                gripper.append(f'<geom name="{name}" type="box" size="{_v([wb / 2 + 0.004, jaw_w / 2, 0.006])}" '
                               f'pos="{_v(e * (stroke + wb / 2) + np.array([0, 0, z0 + 0.006]))}" '
                               f'xyaxes="{_v(e)} {_v(t)}" mass="0.05" class="finger"/>')
                hinge = (f'type="hinge" axis="{_v(t)}" stiffness="{k_fr:.9g}" damping="{0.02 * k_fr:.9g}" '
                         f'armature="1e-5" range="-1.2 1.2"')
                front_geoms = []
                for chain, pts, tag in (('F', Pf, 'front'), ('B', Qb, 'back')):
                    closing = []
                    for i in range(1, n_fr + 1):
                        rel = pts[i - 1] - (pts[i - 2] if i > 1 else np.zeros(3))
                        seg = pts[i] - pts[i - 1]
                        gname = f'{name}{chain}{i}'
                        gripper.append(f'<body name="{gname}" pos="{_v(rel)}">')
                        gripper.append(f'<joint name="{gname}" {hinge}/>')
                        if tag == 'front':  # the gripping face: 2.5 mm thick, inside the flank line
                            c = 0.5 * seg - e * 0.00125
                            gripper.append(f'<geom name="{gname}" type="box" '
                                           f'size="{_v([0.00125, jaw_w / 2, np.linalg.norm(seg) / 2])}" '
                                           f'pos="{_v(c)}" xyaxes="{_v(e)} {_v(t)}" mass="0.003" '
                                           f'friction="1.3 0.01 0.0001" class="finger"/>')
                            front_geoms.append(gname)
                            if i < n_fr:  # the rib from this segment's end (pivoted) to the back flank
                                rib = Qb[i] - Pf[i]
                                gripper += [f'<body name="{name}R{i}" pos="{_v(seg)}">',
                                            f'<joint name="{name}R{i}" type="hinge" axis="{_v(t)}" damping="0.0005" '
                                            f'armature="1e-6"/>',
                                            f'<geom type="capsule" fromto="0 0 0 {_v(rib)}" size="0.001" mass="0.001" '
                                            f'contype="0" conaffinity="0"/>', '</body>']
                                equalities.append(f'<connect body1="{name}R{i}" body2="{name}B{i + 1}" '
                                                  f'anchor="{_v(rib)}" solref="0.004 1"/>')
                        else:
                            gripper.append(f'<geom type="capsule" fromto="0 0 0 {_v(seg)}" size="0.0012" '
                                           f'mass="0.003" contype="0" conaffinity="0"/>')
                        closing.append('</body>')
                    gripper += closing
                # the tip: the flanks' ends together
                equalities.append(f'<connect body1="{name}F{n_fr}" body2="{name}B{n_fr}" '
                                  f'anchor="{_v(Pf[n_fr] - Pf[n_fr - 1])}" solref="0.004 1"/>')
                finray[name] = {'P': Pf, 'Q': Qb, 'n': n_fr, 'front': front_geoms, 'e': e, 't': t}
            gripper.append('</body>')
    elif gkind == 'suction':
        gripper += ['<body name="cup">',
                    f'<geom name="cup_stem" type="cylinder" size="0.009 {(z_palm - 0.012) / 2:.9g}" '
                    f'pos="0 0 {0.012 + (z_palm - 0.012) / 2:.9g}" mass="0.05" class="finger"/>',
                    '<geom name="cup" type="cylinder" size="0.019 0.006" pos="0 0 0.006" mass="0.02" '
                    'margin="0.004" gap="0.004" friction="1.2 0.01 0.0001" class="finger"/>', '</body>']

    lip_names, cup_r, cup_lift = [], 0.0, 0.0
    if gkind in VACUUM_CUPS:
        cup_r, n_lip = VACUUM_CUPS[gkind]
        lip_h, lip_t = 0.006, 0.0028
        k_lip = max(float(P['cup_lip_stiffness']), 1.0)
        # Between soft fingers the cup is short (25 mm from the palm), its lip above the tool point: the fingers,
        # opened, still reach well below it.
        if gkind in SOFT_FINGERS:
            cup_lift = max(z_palm - 0.025, 0.0)
        hub_h = z_palm - cup_lift - lip_h
        gripper += [f'<body name="vcup" pos="0 0 {cup_lift:.9g}">',
                    f'<geom name="vcup_hub" type="cylinder" size="{0.55 * cup_r:.9g} {hub_h / 2:.9g}" '
                    f'pos="0 0 {lip_h + hub_h / 2:.9g}" mass="0.04" class="finger"/>']
        seg = 2 * math.pi * cup_r / n_lip
        for i in range(n_lip):
            a = 2 * math.pi * i / n_lip
            c = np.array([cup_r * math.cos(a), cup_r * math.sin(a), lip_t])
            t = np.array([-math.sin(a), math.cos(a), 0.0])
            name = f'lip{i + 1}'
            lip_names.append(name)
            gripper += [f'<body name="{name}" pos="{_v(c)}">',
                        f'<joint name="{name}" type="slide" axis="0 0 1" range="0 0.015" stiffness="{k_lip:.9g}" '
                        f'damping="1.5" armature="0.002"/>',
                        # (the segments never touch each other: contype 2 meets only the items and the store)
                        f'<geom name="{name}" type="capsule" '
                        f'fromto="{_v(-(0.5 * seg - lip_t) * t)} {_v((0.5 * seg - lip_t) * t)}" size="{lip_t:.9g}" '
                        f'mass="0.002" friction="1.0 0.01 0.0001" contype="2" conaffinity="1"/>', '</body>']
        gripper.append('</body>')

    # ---- the Prensilia Mia hand, fingers down: the palm upright above the finger roots (its palmar side to -x),
    # index to little finger along y, the thumb opposite them (at -x), fixed in opposition. Three motors (servos
    # with a torque limit): the thumb, the index, and the middle finger with the ring and little fingers coupled to
    # it; in each finger the distal joint follows the proximal one.
    MIA_FINGERS = [('index', 0.03, 0.045, 0.040, 0.0085), ('middle', 0.01, 0.048, 0.043, 0.0085),
                   ('ring', -0.01, 0.045, 0.040, 0.0082), ('little', -0.03, 0.037, 0.032, 0.0075)]
    MIA_THUMB = ('thumb', 0.0, 0.048, 0.040, 0.0095)  # (opposite the middle of the fingers, as long)
    mia = {}
    if gkind == 'mia':
        gripper = [g_ for g_ in gripper if 'name="plate"' not in g_]
        zr = z_palm
        grip_top = zr + 0.09 + 0.025
        gripper += [f'<geom name="mia_palm" type="box" size="0.016 0.043 0.045" pos="0.026 0 {zr + 0.045:.9g}" '
                    f'mass="0.35" class="finger"/>',
                    f'<geom name="mia_wrist" type="cylinder" size="0.024 {0.0125:.9g}" pos="0.026 0 {zr + 0.1025:.9g}" '
                    f'mass="0.1" class="finger"/>',
                    # the thumb's housing (the thenar side), from the palm out to the thumb's root
                    f'<geom name="mia_thenar" type="box" size="0.027 0.016 0.023" pos="-0.017 0 {zr + 0.027:.9g}" '
                    f'mass="0.08" class="finger"/>']
        for name, y, l1, l2, r in MIA_FINGERS + [MIA_THUMB]:
            thumb = name == 'thumb'
            root = np.array([-0.03, y, zr]) if thumb else np.array([0.025, y, zr])
            axis = '0 -1 0' if thumb else '0 1 0'  # (positive: curl towards the other side)
            mia[name] = {'root': root, 'l': (l1, l2), 'r': r, 'thumb': thumb}
            gripper += [f'<body name="mia_{name}1" pos="{_v(root)}">',
                        f'<joint name="mia_{name}_mcp" type="hinge" axis="{axis}" range="-0.6 1.6" damping="0.02" '
                        f'armature="1e-4"/>',
                        f'<geom name="mia_{name}1" type="capsule" fromto="0 0 0 0 0 {-l1:.9g}" size="{r:.9g}" '
                        f'mass="0.015" friction="1.2 0.01 0.0001" class="finger"/>',
                        f'<body name="mia_{name}2" pos="0 0 {-l1:.9g}">',
                        f'<joint name="mia_{name}_pip" type="hinge" axis="{axis}" range="-0.2 1.7" damping="0.01" '
                        f'armature="5e-5"/>',
                        f'<geom name="mia_{name}2" type="capsule" fromto="0 0 0 0 0 {-l2:.9g}" size="{0.92 * r:.9g}" '
                        f'mass="0.01" friction="1.2 0.01 0.0001" class="finger"/>', '</body>', '</body>']
            equalities.append(f'<joint joint1="mia_{name}_pip" joint2="mia_{name}_mcp" polycoef="0 1 0 0 0" '
                              f'solref="0.005 1"/>')
        for name in ('ring', 'little'):
            equalities.append(f'<joint joint1="mia_{name}_mcp" joint2="mia_middle_mcp" polycoef="0 1 0 0 0" '
                              f'solref="0.005 1"/>')

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
    wheel_set = cart_wheels(cart, PL, PW, car_x, car_y, wheel_r, track_w)

    def spinning(w):  # a wheel's body, its axle along y: only seen turning (the platform rides on its joints)
        return [f'<body name="{w["name"]}" pos="{_v(w["pos"])}">',
                f'<joint name="{w["name"]}" type="hinge" axis="0 1 0"/>',
                '<inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>', '</body>']
    if cart == 'car':
        for side, name in ((1, 'steer_l'), (-1, 'steer_r')):
            plat += [f'<body name="{name}" pos="{_v([car_x, side * car_y, wheel_r])}">',
                     f'<joint name="{name}" type="hinge" axis="0 0 1" range="-0.9 0.9" damping="2"/>',
                     '<inertial pos="0 0 0" mass="0.5" diaginertia="0.003 0.003 0.003"/>']
            plat += spinning(next(w for w in wheel_set if w['parent'] == name))
            plat.append('</body>')
    for w in wheel_set:
        if w['parent'] == 'platform':
            plat += spinning(w)
    plat += arm_bodies(False)
    plat.append('</body>')
    world += plat

    # ---- the packages: the front `dynamic_rows` of each lane free bodies (asleep until touched), the ones behind
    # fixed geoms; and the crate's, free. Their shapes, masses and frictions: hagelslag_shelf / hagelslag_physics.json.
    scale = max(float(P['item_mass_scale']), 1e-3)
    rolling = max(float(P['rolling_friction']), 0.0)
    items = []

    def package_geom(pk, name):
        h = pk['half']
        if pk['shape'] == 'cylinder':  # (round: it can roll, so rolling friction and condim 6)
            return (f'name="{name}" type="cylinder" size="{h[0]:.9g} {h[2]:.9g}" condim="6" '
                    f'friction="{pk["friction"]:.9g} {tors:.9g} {rolling:.9g}"')
        return f'name="{name}" type="box" size="{_v(h)}" friction="{pk["friction"]:.9g} {tors:.9g} 0.0001"'

    asleep = ' sleep="init"' if P['sleep'] else ''
    for pk in shelf['packages']:
        key = f'pkg{pk["index"]}'
        if rows <= 0 or pk['row'] < rows:
            world.append(f'<body name="{key}" pos="{_v(pk["centre"])}"{asleep}><freejoint name="{key}"/>'
                         f'<geom {package_geom(pk, key)} mass="{pk["mass"] * scale:.9g}" class="item"/></body>')
            items.append({'key': key, 'pk': pk, 'centre': np.array(pk['centre']), 'origin': 'shelf'})
        else:
            world.append(f'<geom {package_geom(pk, key)} pos="{_v(pk["centre"])}" class="static"/>')
    for k, spot in enumerate(crate['items']):  # (the platform starts with yaw 0: its frame is the world's, moved)
        pk = front_of[spot['product']]
        key = f'crate{k + 1}'
        centre = [start[0] + spot['centre'][0], start[1] + spot['centre'][1], PH + pk['half'][2] + 0.3 * MM]
        world.append(f'<body name="{key}" pos="{_v(centre)}"><freejoint name="{key}"/>'
                     f'<geom {package_geom(pk, key)} mass="{pk["mass"] * scale:.9g}" class="item"/></body>')
        items.append({'key': key, 'pk': pk, 'centre': np.array(centre), 'origin': 'crate'})

    # No contacts within one finger, nor between a finger and the palm (see Gripper 5), nor between the arm's base,
    # shoulder and upper arm and the platform they stand on.
    excludes = []
    for k in range(len(frames)):
        chain = ['flange'] + [f'f{k + 1}s{j}' for j in range(1, N)]
        for a in range(len(chain)):
            for b in range(a + 2, len(chain)):  # (a body and its parent never touch in MuJoCo anyway)
                excludes.append(f'<exclude body1="{chain[a]}" body2="{chain[b]}"/>')
    for name, fr_ in finray.items():
        parts = [name] + [f'{name}F{i}' for i in range(1, fr_['n'] + 1)]
        for a in range(len(parts)):
            for b in range(a + 1, len(parts)):
                excludes.append(f'<exclude body1="{parts[a]}" body2="{parts[b]}"/>')
        for part in parts[1:]:
            excludes.append(f'<exclude body1="flange" body2="{part}"/>')
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
    if gkind == 'mia':
        t_max = max(float(P['mia_torque']), 0.01)
        actuators += [f'<position name="mia_{n}" joint="mia_{n}_mcp" kp="3" kv="0.06" '
                      f'forcerange="{-t_max:.9g} {t_max:.9g}"/>' for n in ('thumb', 'index', 'middle')]
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
    xml = f"""<mujoco model="hagelslag">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep:.9g}" gravity="0 0 {-g:.9g}" cone="elliptic" impratio="{float(P['impratio']):.9g}"
          noslip_iterations="{max(int(P['noslip_iterations']), 0)}" integrator="implicitfast">{sleep_flag}</option>
  {defaults}
  <asset>{''.join(assets)}</asset>
  <worldbody>
    {chr(10).join(world)}
  </worldbody>
  <contact>{''.join(excludes)}</contact>
  <equality>{''.join(equalities)}</equality>
  <actuator>{''.join(actuators)}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    scene = MjScene(model, dt=float(P['dt']))
    scene.xml = xml
    body = scene.body_id
    flange_id = body('flange')
    platform_id = body('platform')

    # The arm alone, for the inverse kinematics: its base where it stands in the platform's frame.
    ik_xml = f"""<mujoco model="hagelslag_ik"><compiler angle="radian" autolimits="true"/>
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
    # From the aisle, a little to the left and above: the robot in front of the shelf.
    cam_lookat = [sx, 0.5 * shelf['front'] - 0.05, 0.75]
    cam_dir = (-0.25, -1.0, 0.4)
    norm = math.sqrt(sum(c * c for c in cam_dir))
    cam_distance = 1.25 * max(shelf['width'], 1.4)
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

    # The shelf: shelf.glb's fixtures as they are (the price tags with their image), and the fixed packages, a
    # mesh per product (on the world: sent once).
    scene.images.update(shelf['images'])
    store = '/world/store'
    for v in shelf['visuals']:  # (the glass dividers and the rails see-through of their own: not the Shelf opacity's)
        part = 'shelf' if v['rgba'][3] >= 0.999 else 'glass'
        scene.add_visual(f'{store}/{part}/{v["name"]}/visual', v['points'], v['triangles'], 0, v['rgba'],
                         image=v['image'], uv=v['uv'])

    def package_visuals(path, entries):
        """[(package, body id, centre)] -> one visual per material (a product's photo), each point on its
        package's body: its position in that body's frame (the world's: its place)."""
        groups = {}
        for pk, b, centre in entries:
            for pts, tris, uv, rgba, img in pk['parts']:
                g_ = groups.setdefault(img, {'pts': [], 'tris': [], 'uv': [], 'bodies': [], 'rgba': rgba})
                off = sum(len(q) for q in g_['pts'])
                g_['pts'].append(pts + (centre if b == 0 else 0.0))
                g_['tris'].append(tris + off)
                g_['uv'].append(uv if uv is not None else np.zeros((len(pts), 2)))
                g_['bodies'] += [b] * len(pts)
        for k, (img, g_) in enumerate(groups.items()):
            scene.add_visual(f'{path}{"" if k == 0 else f"_{k}"}/visual', np.concatenate(g_['pts']),
                             np.concatenate(g_['tris']), g_['bodies'], list(g_['rgba'][:3]) + [float(P['item_alpha'])],
                             image=img, uv=np.concatenate(g_['uv']) if img else None)

    fixed = {}
    for pk in shelf['packages']:
        if not (rows <= 0 or pk['row'] < rows):
            fixed.setdefault(pk['product'], []).append((pk, 0, pk['centre']))
    for product, entries in fixed.items():
        package_visuals(f'{store}/packages/{product}', entries)

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
    # The wheels sit in wells, open to the side: the deck and the bumper are cut away round them.
    wells = [(w['at'][0] - w['well'][0], w['at'][0] + w['well'][0], w['well'][1], w['well'][2])
             for w in wheel_set if w['well'] and w['at'][1] > 0]
    deck = _notched_box(-PL / 2, -PW / 2, 0.04, PL / 2, PW / 2, PH, wells)
    bumper = _notched_box(-PL / 2 - 0.012, -PW / 2 - 0.012, 0.07, PL / 2 + 0.012, PW / 2 + 0.012, 0.13,
                          # (2 mm past the deck's faces, so the two don't fight where they'd coincide)
                          [(a - 0.002, b + 0.002, d + 0.014, top) for a, b, d, top in wells])
    wheels, hubs = [], []
    if cart == 'diff':  # a small caster at each end, low under the deck
        for cx_ in (-PL / 2 + 0.1, PL / 2 - 0.1):
            wheels.append(_cylinder_along([cx_, -0.012, 0.02], [cx_, 0.012, 0.02], 0.02, n=16))
            for fy in (-0.017, 0.013):
                hubs.append(_box(cx_ - 0.012, fy, 0.012, cx_ + 0.012, fy + 0.004, 0.04))
    elif cart == 'tank':  # the belt round each track's two sprockets
        for side in (-1, 1):
            wy = side * (PW / 2 + track_w / 2 + 0.005)
            xe = PL * 0.47 - 0.075
            wheels.append(_box(-xe, wy - track_w / 2, 0.005, xe, wy + track_w / 2, 0.155))
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
    for w in wheel_set:  # each on its own body, in its frame (the axle along y)
        tyre, hub = wheel_meshes(w)
        scene.add_visual(f'{robot}/platform/{w["name"]}/visual', *_smooth_shaded(*_merge(tyre)), body(w['name']),
                         [0.08, 0.08, 0.09, 1.0], texture='rubber')
        scene.add_visual(f'{robot}/platform/{w["name"]}_hub/visual', *_smooth_shaded(*_merge(hub)), body(w['name']),
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
    meshes = arm_meshes(spec)
    for i, (axis, offset, (p0, p1), r) in enumerate(links):
        if meshes:
            break
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
    # The real arm's links (its makers' CAD, simplified), each in its link's frame: at zero that's the arm base's,
    # moved to the link's joint.
    for k, groups in sorted(meshes.items()):
        at = sum((np.array(o_, float) for _, o_, _, _ in links[:k]), np.zeros(3))
        at[2] -= pedestal  # (the meshes' base is on top of the pedestal; arm_base is at its foot)
        for j, (pts, tris, rgba) in enumerate(groups):
            scene.add_visual(f'{robot}/arm/link{k}_{j}/visual', *_smooth_shaded([list(q - at) for q in pts], tris),
                             body(f'arm_l{k}') if k else body('arm_base'), rgba, texture='painted')
    if not meshes:
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
    soft_rgb = [0.93, 0.93, 0.9] if gkind.startswith('softgripping') else [0.15, 0.45, 0.95]
    scene.add_visual('/world/angles/bodies/visu/visual', pts, tris, bodies, soft_rgb + [float(P['wall_alpha'])],
                     texture='silicone')
    pts, tris, bodies = carried(sheet_parts)
    scene.add_visual('/world/angles/bodies/base_layer/visual', pts, tris, bodies,
                     [0.96, 0.94, 0.86, float(P['wall_alpha'])], texture='fabric')
    # A tactile skin (Touchlab / TG0 style): a thin sensing pad on each finger segment's gripping face (its base
    # layer), its own model so the viewer can colour it by the force on that segment (the probes touch_<pad>).
    skin_pads, skin_names = [], {}
    if P['tactile_skin']:
        for fr in frames:
            for j in range(N):
                xs = j * pitch
                pp, pt = _box(xs + 0.08 * L, -0.4 * W, -1.0 * MM, xs + 0.92 * L, 0.4 * W, -0.45 * MM)
                o = origin_of(fr.k, j)
                pad = f'f{fr.k + 1}s{j + 1}'
                local = [list(np.array(fr.to_world(q)) - o) for q in pp]
                skin_pads.append((pad, local, pt, body_of(fr.k, j)))
                skin_names[pad] = [f'f{fr.k + 1}s{j}'] + ([f'f{fr.k + 1}tip'] if tips and j == N - 1 else [])
    if tip_parts:
        fp = []
        for tp, tt, b, o in tip_parts:  # flat-shaded: unshared vertices per face
            p2, t2 = _flat_shaded(tp, tt)
            fp.append((p2, t2, b, o))
        pts, tris, bodies = carried(fp)
        tip_rgb = [0.8, 0.8, 0.77] if gkind.startswith('softgripping') else [0.1, 0.25, 0.6]
        scene.add_visual('/world/angles/bodies/tips/visual', pts, tris, bodies, tip_rgb + [float(P['wall_alpha'])],
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
    if not mia:
        scene.add_visual('/world/angles/bodies/trunk/visual', mount_pts, mount_tris, flange_id, [0.4, 0.26, 0.13, 1.0],
                         texture='printed')
    else:  # the hand: palm and wrist, rounded phalanges, a skin pad on each phalanx's palmar side
        zr = z_palm
        palm_v = _merge([_box(0.010, -0.043, zr, 0.042, 0.043, zr + 0.09),
                         _cylinder_along([0.026, 0, zr + 0.09], [0.026, 0, zr + 0.115], 0.024, n=28),
                         # the thumb's housing: out from the palm to the thumb's root, round over its joint
                         _box(-0.03, -0.016, zr + 0.004, 0.0105, 0.016, zr + 0.05),
                         _box(-0.044, -0.016, zr + 0.017, -0.0295, 0.016, zr + 0.05),
                         _cylinder_along([-0.03, -0.016, zr + 0.017], [-0.03, 0.016, zr + 0.017], 0.014, n=24)])
        scene.add_visual('/world/angles/bodies/trunk/visual', *_smooth_shaded(*palm_v), flange_id,
                         [0.86, 0.87, 0.89, 1.0], texture='plastic')
        shells, knuckles = [], []
        for name, f_ in mia.items():
            r = f_['r']
            for seg, l in ((1, f_['l'][0]), (2, f_['l'][1])):
                rr = r if seg == 1 else 0.92 * r
                b_ = body(f'mia_{name}{seg}')
                shells.append((*_smooth_shaded(*_cylinder_along([0, 0, 0], [0, 0, -l], rr, n=20)), b_, np.zeros(3)))
                knuckles.append((*_smooth_shaded(*_ellipsoid((0, 0, 0), (1.05 * rr, 1.05 * rr, 1.05 * rr), 6, 14)), b_,
                                 np.zeros(3)))
                if seg == 2:
                    knuckles.append((*_smooth_shaded(*_ellipsoid((0, 0, -l), (rr, rr, rr), 6, 14)), b_, np.zeros(3)))
                if P['tactile_skin']:
                    side = 1.0 if f_['thumb'] else -1.0  # the palmar side faces the other side's fingers
                    pad_x0, pad_x1 = sorted((side * (rr - 0.0012), side * (rr + 0.0006)))
                    pp, tt = _box(pad_x0, -0.75 * rr, -0.88 * l, pad_x1, 0.75 * rr, -0.12 * l)
                    skin_pads.append((f'{name}{seg}', pp, tt, b_))
                    skin_names[f'{name}{seg}'] = [f'mia_{name}{seg}']
        for vname, parts, rgba in (('mia_fingers', shells, [0.86, 0.87, 0.89]),
                                   ('mia_joints', knuckles, [0.25, 0.26, 0.28])):
            pts, tris, bodies = carried(parts)
            scene.add_visual(f'/world/angles/bodies/{vname}/visual', pts, tris, bodies,
                             rgba + [float(P['wall_alpha'])], texture='plastic')
    if jaw_names:
        n_jaws = len(jaw_names)
        rail = (_box(-(stroke + jaw_t), -0.016, z_palm - 0.02, stroke + jaw_t, 0.016, z_palm) if n_jaws == 2 else
                _frustum_z(0, 0, z_palm - 0.02, z_palm, stroke + jaw_t, stroke + jaw_t, n=36))
        scene.add_visual('/world/angles/bodies/rail/visual', *_smooth_shaded(*rail), flange_id,
                         [0.3, 0.32, 0.36, float(P['wall_alpha'])], texture='brushed')
        jaw_parts, pad_parts, finray_parts = [], [], []
        for name, e in zip(jaw_names, jaw_dirs):
            t = np.array([-e[1], e[0], 0.0])
            def block(r0, r1, w, z0, z1, e=e, t=t):
                pts, tris = _box(r0, -w / 2, z0, r1, w / 2, z1)
                return [list(p[0] * e + p[1] * t + np.array([0, 0, p[2]])) for p in pts], tris
            jb = body(name)
            z1 = z_palm - 0.02
            if name in finray:
                f_ = finray[name]
                wb = max(float(P['finray_base']), 5.0) * MM
                jaw_parts.append((*block(stroke - 0.004, stroke + wb + 0.004, jaw_w, z1, z1 + 0.012), jb, np.zeros(3)))
                for i in range(1, f_['n'] + 1):
                    for chain, pts, thick_ in (('F', f_['P'], 0.0025), ('B', f_['Q'], 0.002)):
                        a, b_ = pts[i - 1], pts[i]
                        r0 = float(np.dot(a, e)) - (thick_ if chain == 'F' else 0.0)
                        pp, tt = _box(0, -jaw_w / 2, 0, 1, jaw_w / 2, 1)
                        # a slab from a to b, `thick_` deep (along e), jaw_w wide
                        quad = [a - (thick_ if chain == 'F' else 0) * e, b_ - (thick_ if chain == 'F' else 0) * e,
                                b_ + (0 if chain == 'F' else thick_) * e, a + (0 if chain == 'F' else thick_) * e]
                        pts8 = []
                        for q in pp:
                            base_ = quad[0] if q[0] == 0 and q[2] == 0 else quad[1] if q[0] == 0 else \
                                quad[3] if q[2] == 0 else quad[2]
                            pts8.append(list(base_ + q[1] * t - pts[i - 1]))
                        finray_parts.append((pts8, tt, body(f'{name}{chain}{i}'), np.zeros(3), chain))
                    if i < f_['n']:
                        rib_a, rib_b = np.zeros(3), f_['Q'][i] - f_['P'][i]
                        rp, rt = _box(0, -0.42 * jaw_w, -0.0008, 1, 0.42 * jaw_w, 0.0008)
                        pts8 = [list(rib_a + q[0] * rib_b + q[1] * t + np.array([0, 0, q[2]])) for q in rp]
                        finray_parts.append((pts8, rt, body(f'{name}R{i}'), np.zeros(3), 'R'))
                for i in range(1, f_['n'] + 1):  # a pad on each front segment
                    a, b_ = f_['P'][i - 1], f_['P'][i]
                    quad_pts = []
                    for q in _box(0, -0.4 * jaw_w, 0, 1, 0.4 * jaw_w, 1)[0]:
                        z_ = a + (0.1 + 0.8 * q[0]) * (b_ - a)
                        quad_pts.append(list(z_ - (0.0025 + (0.0006 if q[2] == 0 else 0.0001)) * e + q[1] * t - a))
                    pad_parts.append((quad_pts, _box(0, 0, 0, 1, 1, 1)[1], body(f'{name}F{i}'), np.zeros(3)))
            else:
                jaw_parts.append((*block(stroke, stroke + jaw_t, jaw_w, z1 - jaw_l, z1), jb, np.zeros(3)))
                pad_parts.append((*block(stroke - 0.5 * MM, stroke, 0.9 * jaw_w, z1 - jaw_l + 0.004, z1 - 0.012), jb,
                                  np.zeros(3)))
        if P['tactile_skin']:  # each jaw's pad (a Fin Ray finger's: each front segment's) is a skin of its own
            if finray:
                for (pp, tt, b, o), k in zip(pad_parts, range(len(pad_parts))):
                    name, i = jaw_names[k // 6], k % 6 + 1
                    skin_pads.append((f'{name}s{i}', pp, tt, b))
                    skin_names[f'{name}s{i}'] = [f'{name}F{i}']
            else:
                for (pp, tt, b, o), name in zip(pad_parts, jaw_names):
                    skin_pads.append((name, pp, tt, b))
                    skin_names[name] = [name]
            pad_parts = []
        for chain, rgba in (('F', [0.12, 0.42, 0.8]), ('B', [0.12, 0.42, 0.8]), ('R', [0.55, 0.72, 0.92])):
            parts = [(*_flat_shaded(pp, tt), b, o) for pp, tt, b, o, c in finray_parts if c == chain]
            if parts:
                pts, tris, bodies = carried(parts)
                scene.add_visual(f'/world/angles/bodies/finray_{chain}/visual', pts, tris, bodies,
                                 rgba + [float(P['wall_alpha'])], texture='plastic')
        for vname, parts, rgba, tex in (('jaws', jaw_parts, [0.72, 0.74, 0.78], 'brushed'),
                                        ('jaw_pads', pad_parts, [0.1, 0.1, 0.11], 'rubber')):
            if not parts:
                continue
            fp = [(*_flat_shaded(pp, tt), b, o) for pp, tt, b, o in parts]
            pts, tris, bodies = carried(fp)
            scene.add_visual(f'/world/angles/bodies/{vname}/visual', pts, tris, bodies,
                             rgba + [float(P['wall_alpha'])], texture=tex)
    for pad, pp, tt, b in skin_pads:
        scene.add_visual(f'/world/angles/bodies/skin/{pad}/visual', *_flat_shaded(pp, tt), b,
                         [0.22, 0.24, 0.27, float(P['wall_alpha'])], texture='plastic')
    if lip_names:
        hub = _frustum_z(0, 0, 0.006, z_palm - cup_lift, 0.75 * cup_r, 0.4 * cup_r, n=32)
        scene.add_visual('/world/angles/bodies/cup_stem/visual', *_smooth_shaded(*hub), body('vcup'),
                         [0.94, 0.94, 0.9, float(P['wall_alpha'])], texture='silicone')
        petals = []
        n_lip = len(lip_names)
        for i, name in enumerate(lip_names):
            a0, a1 = 2 * math.pi * (i - 0.5) / n_lip, 2 * math.pi * (i + 0.5) / n_lip
            pts, tris = [], []
            for kk in range(5):  # a curved strip of the lip: flared out at the bottom
                a = a0 + (a1 - a0) * kk / 4
                for r_, z_ in ((cup_r + 0.003, -0.0026), (cup_r - 0.001, 0.0), (0.8 * cup_r, 0.006)):
                    pts.append([r_ * math.cos(a), r_ * math.sin(a), z_])
            for kk in range(4):
                for j in range(2):
                    q0, q1 = 3 * kk + j, 3 * (kk + 1) + j
                    tris += [[q0, q1, q1 + 1], [q0, q1 + 1, q0 + 1]]
            c = np.array([cup_r * math.cos(2 * math.pi * i / n_lip), cup_r * math.sin(2 * math.pi * i / n_lip), 0.0028])
            petals.append(([list(np.array(p_) - c) for p_ in pts], tris, body(name), np.zeros(3)))
        pts, tris, bodies = carried([(*_smooth_shaded(pp, tt), b, o) for pp, tt, b, o in petals])
        scene.add_visual('/world/angles/bodies/cup/visual', pts, tris, bodies, [0.9, 0.93, 0.96, float(P['wall_alpha'])],
                         texture='silicone')
    if gkind == 'suction':
        cup_id = body('cup')
        stem_m = _cylinder_along([0, 0, 0.016], [0, 0, z_palm], 0.009, n=16)
        bellows = _merge([_frustum_z(0, 0, 0.012 + 0.008 * k, 0.016 + 0.008 * k, 0.014, 0.011, n=24) for k in range(3)])
        lip = _frustum_z(0, 0, 0.0, 0.012, 0.02, 0.012, n=32)
        scene.add_visual('/world/angles/bodies/cup_stem/visual', *_smooth_shaded(*stem_m), cup_id,
                         [0.72, 0.74, 0.78, float(P['wall_alpha'])], texture='brushed')
        scene.add_visual('/world/angles/bodies/cup/visual', *_smooth_shaded(*_merge([bellows, lip])), cup_id,
                         [0.12, 0.12, 0.13, float(P['wall_alpha'])], texture='rubber')

    # The free packages (on the shelf's front rows, in the crate): a mesh per product, each point on its package.
    free = {}
    for it in items:
        free.setdefault(it['pk']['product'], []).append((it['pk'], body(it['key']), it['centre']))
    for product, entries in free.items():
        package_visuals(f'/world/items/{product}', entries)

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
                            steer_actuators=[act('steer_l'), act('steer_r')] if cart == 'car' else (),
                            dofs=[int(model.jnt_dofadr[jnt(n)]) for n in ('base_x', 'base_y', 'base_yaw')],
                            wheels=[(int(model.jnt_qposadr[jnt(w['name'])]), int(model.jnt_dofadr[jnt(w['name'])]),
                                     *w['at'], w['r'], w['slant'],
                                     int(model.jnt_qposadr[jnt(w['parent'])]) if w['parent'] != 'platform' else None)
                                    for w in wheel_set if w['r']])
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
    if mia:
        hand = HandController(pressure, [act(f'mia_{n}') for n in ('thumb', 'index', 'middle')])
        scene.controllers.append(hand)
        data_ctrl = scene.data.ctrl
        data_ctrl[hand.actuators] = HandController.OPEN
    vacuum = None
    if lip_names:
        lip_geoms = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)) for n in lip_names]
        vacuum = VacuumCup(pressure, body('vcup'), lip_geoms, math.pi * (cup_r - 0.002) ** 2,
                           max(float(P['cup_vacuum']), 0.0) * 1e3)
        scene.controllers.append(vacuum)
        scene.objects['Vacuum'] = vacuum
    if gkind.startswith('softgripping'):
        pressure.idle = -max(float(P['sg_open_vacuum']), 0.0) * 1e3
    all_hinges = hinge_names + side_names
    repair = Repair([int(model.jnt_qposadr[jnt(n)]) for n in all_hinges],
                    [int(model.jnt_dofadr[jnt(n)]) for n in all_hinges], arm, arm_qpos, arm_dofs)
    scene.objects.update({'Flange': motion, 'Arm': arm, 'Drive': drive, 'PressureRamp': pressure, 'Repair': repair,
                          'Jaws': jaws})

    BUILD.clear()
    BUILD.update({
        'segments': N, 'fingers': F, 'flange_id': flange_id, 'platform_id': platform_id, 'tcp_depth': tcp_depth,
        'hinge_qpos': [int(model.jnt_qposadr[jnt(n)]) for n in hinge_names],
        'side_qpos': [int(model.jnt_qposadr[jnt(n)]) for n in side_names],
        'arm_qpos': arm_qpos,
        'items': [{'key': it['key'], 'product': it['pk']['product'], 'origin': it['origin'], 'start': it['centre'],
                   'half_h': float(it['pk']['half'][2]), 'body': body(it['key'])} for it in items],
        'crates': [{'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'top': PH + ch, 'platform': True}],
        'shelf_levels': shelf['levels'], 'packages': len(shelf['packages']), 'free_packages': n_free,
        'torque_per_pa': torque_per_pa, 'seg_mass': seg_mass, 'plate_radius': plate_r,
        'finger_length': finger_len + (tip_l if tips else 0.0),
        'gripper': gkind, 'arm': P['arm'], 'cart': cart, 'arm_joints': n_arm,
        # the tactile skin: geom id -> its pad, and the pads
        'skin_geoms': {int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)): pad
                       for pad, geoms in skin_names.items() for g in geoms},
        'skin_pads': list(skin_names),
    })
    return scene


def package_states(data):
    """Where each free package (BUILD['items']) is: 'crate' (in the platform's crate), 'floor', 'moved' (more than
    2 cm from where it started, elsewhere) or 'place' (where it started)."""
    items, crates = BUILD['items'], BUILD['crates']
    if 'item_arrays' not in BUILD:
        BUILD['item_arrays'] = (np.array([it['body'] for it in items]), np.array([it['half_h'] for it in items]),
                                np.array([it['start'] for it in items]))
    bodies, half, starts = BUILD['item_arrays']
    q = data.xpos[bodies]
    pid = BUILD['platform_id']
    local = (q - data.xpos[pid]) @ data.xmat[pid].reshape(3, 3)
    moved = np.linalg.norm(q - starts, axis=1) > 0.02
    states = []
    for i in range(len(items)):
        if any(c['x0'] < local[i, 0] < c['x1'] and c['y0'] < local[i, 1] < c['y1'] and local[i, 2] < c['top'] + half[i]
               for c in crates):
            states.append('crate')
        elif q[i, 2] < half[i] + 0.015:
            states.append('floor')
        else:
            states.append('moved' if moved[i] else 'place')
    return states


def main():
    os.environ.setdefault('HAGELSLAGMJ_NO_EDITOR', '1')
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
