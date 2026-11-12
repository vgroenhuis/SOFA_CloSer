# -*- coding: utf-8 -*-
"""
Gripper 8: Grip test line (MuJoCo). Every gripper of Gripper 7 / Hagelslag (GRIPPER_CHOICES) in a row along x, each
over its own copy of the same grocery item, and an automatic test that runs them all at once:

  down to the item, grip, up, a full turn (360 degrees) about the x axis (does it hold the item upside down and on
  its side?), one metre forward along y, down to the floor, let go, up.

The item is a choice (the parameter 'item'): Gripper 7's groceries, the Hagelslag shelf's packages (their meshes and
photographed fronts from hagelslag_shelf/shelf.glb, masses and frictions from hagelslag_physics.json), or one of two
deformable ones:
  - a bag of chips, in four sizes (150 g: its size from the parameters; 40 g, 25 g and a 15 g mini bag): a closed
    bag of air and chips that keeps its volume. A MuJoCo flex (a rounded, pillow-shaped grid of nodes split into
    tetrahedra) with nearly incompressible elasticity (Poisson's ratio near 0.5): squeezed in one place it bulges
    out elsewhere. The smaller ones fit between the jaws (grasped across), the big one is pinched on top;
  - a bag of potatoes: round potatoes (spheres to the physics) in a net: slack strings (MuJoCo tendons with a
    largest length) between neighbouring potatoes and to the bag's clip; the potatoes shift inside it.

Each gripper hangs from a mocap body (it's moved, not driven: an infinitely strong, exact robot) and has its own grip
controllers (Gripper 7's): pressure for the soft fingers, force for the jaws, vacuum for the cups, servos for the
Mia hand. The grippers' code is Gripper 7's (../../gripper-7-mujoco/original/Gripper7Mj.py, via HagelslagMj.py);
adding a gripper there means adding it to build_gripper here too, and the line grows by itself.

Units: the scene is SI (metres, kg, s, Pa); params.json takes millimetres for lengths, as labelled.

Run with MuJoCo's viewer:   python Gripper8Mj.py       (pip install mujoco)
Headless check:             python Gripper8Mj.py --headless
Parameters: params.json next to this script (edit by hand, or with params_editor.py).
"""
import itertools
import json
import math
import os
import re
import struct
import subprocess
import sys

import numpy as np
import mujoco

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')
SHELF_DIR = os.path.join(_SCRIPT_DIR, 'hagelslag_shelf')
PHYSICS_FILE = os.path.join(_SCRIPT_DIR, 'hagelslag_physics.json')
PLATES_DIR = os.path.join(_SCRIPT_DIR, 'name_plates')

_DEFAULT_PARAMS = {
    # -- the test --
    'item': 'g7_egg',            # the grocery item (ITEM_CHOICES): 'chips' (150 g, sized below), 'chips_40',
                                 # 'chips_25', 'chips_15', 'potatoes', 'g7_<product>', 'hs_<product>'
    'spacing': 400.0,            # [mm] between the grippers, along x
    'lift_height': 450.0,        # [mm] the tool point's height while it carries and turns the item
    'travel': 1000.0,            # [mm] how far forward (+y) it carries the item
    'move_speed': 200.0,         # [mm/s] the moves' average speed (smooth starts and stops: the top speed 1.9 times)
    'turn_speed': 60.0,          # [deg/s] the turn about x, on average
    'grip_pressure': 80.0,       # [kPa] the grip (soft fingers: pressure; jaws: 80 = 80 % of Jaw force at 100;
                                 #   cups: full vacuum from 80; the Mia hand: closed from 80)
    'grip_wait': 1.0,            # [s] after the grip has ramped up, before lifting
    'release_wait': 1.5,         # [s] after letting go, before rising
    # -- the deformable items --
    'chips_length': 220.0,       # [mm] the 150 g bag of chips ('chips'), lying flat: along x
    'chips_width': 150.0,        # [mm] across
    'chips_thickness': 90.0,     # [mm] in the middle (thinner towards the edges)
    'chips_mass': 150.0,         # [g] the bag and the chips
    'chips_roundness': 0.6,      # every bag of chips: its long sides' thickness, of the middle's (0.3: flat, 0.9: a
                                 # sausage); the sealed ends are always flat (0.3)
    'chips_stiffness': 100.0,    # [kPa] its Young's modulus (the air inside and the foil: how hard it is to squash)
    'chips_poisson': 0.47,       # its Poisson's ratio: 0.5 keeps the volume exactly (above 0.48 gets stiff to solve)
    'potato_count': 6,           # potatoes in the bag (rows of three, two rows, then a layer above)
    'potato_radius': 28.0,       # [mm] each one (a sphere)
    'potato_mass': 100.0,        # [g] each one
    'net_slack': 15.0,           # [%] how much longer the net's strings are than where the potatoes start
    'item_mass_scale': 1.0,      # times each item's mass
    # -- rigid jaws and the suction cups --
    'jaw_stroke': 55.0,          # [mm] each jaw's inner face from the axis, fully open
    'jaw_length': 70.0,          # [mm] below the palm
    'jaw_width': 22.0,           # [mm]
    'jaw_force': 40.0,           # [N] each jaw's grip at 100 kPa
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
    # -- the soft fingers (PneuNet) --
    'fingers': 4,                # how many, evenly spaced around the end-effector's axis
    'finger_radius': 45.0,       # [mm] from the axis to each finger's inner face (its base layer)
    'finger_azimuth': 0.0,       # [deg] where finger 1 hangs (0: +x, 90: +y)
    'segments': 6,               # rigid segments (chambers) in each finger
    'length': 8.0,               # [mm] one segment, along the finger
    'width': 20.0,               # [mm] across it
    'height': 16.0,              # [mm] from the base layer to the roof
    'wall_end': 1.5,             # [mm] end walls (they set the cavity's cross-section and the mass)
    'wall_side': 1.5,
    'wall_bottom': 3.0,
    'wall_top': 1.5,
    'channel_separation': 2.0,   # [mm] gap between neighbouring segments, at the base
    'tips_enabled': True,
    'tip_length': 14.0,          # [mm] beyond the last segment
    'tip_width': 3.0,            # [mm] the point, across the finger
    'tip_thickness': 2.0,        # [mm] the point, from the finger's inner face
    'joint_stiffness': 0.34,     # [N m/rad] torsional stiffness of each hinge
    'joint_damping': 0.01,       # [s] stiffness-proportional damping of the joints
    'joint_open_limit': 90.0,    # [deg] largest bend of a hinge (a hard stop beyond it)
    'side_stiffness': 5.0,       # sideways bending of each joint: its stiffness, times joint_stiffness
    'side_limit': 20.0,          # [deg] largest sideways bend of a joint
    'joint_armature': 2e-6,      # [kg m^2] added inertia of each hinge (MuJoCo: keeps the light segments stable)
    'limit_timeconst': 0.002,    # [s] the hinges' limits (MuJoCo solref)
    'pressure_torque_gain': 1.0, # 1: the torque of the pressure on the cavity's cross-section
    'tcp_depth': 50.0,           # [mm] the tool point below the palm (the point that moves and turns)
    'flange_mass': 0.5,          # [kg] the gripper's plate and wrist adapter
    # -- contacts --
    'friction': 0.8,             # Coulomb friction coefficient of every contact
    'torsional_friction': 0.005, # [m] resistance to twisting about a contact's normal
    'rolling_friction': 0.002,   # [m] round items: resistance to rolling
    'impratio': 10.0,            # MuJoCo: ratio of frictional to normal constraint impedance (higher: less slip)
    'noslip_iterations': 10,     # MuJoCo's no-slip solver passes: without them a held item creeps out (~1 mm/s)
    'finger_collisions': True,   # the fingers touch each other
    # -- material --
    'density': 1000.0,           # [kg/m^3] of the silicone (the segment's mass: the walls' volume)
    'gravity_enabled': True,
    # -- the grip's ramp --
    'ramp_start': 0.0,           # [s] the grip can't start before this
    'ramp_time': 0.4,            # [s] from 0 to the grip
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
    """Opens params_editor.py in its own process. Skipped when GRIPPER8MJ_NO_EDITOR is set (headless runs)."""
    if os.environ.get('GRIPPER8MJ_NO_EDITOR'):
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
    80 kPa or more is full suction). On a body in `porous` (the potatoes: the cup would sit on the bag's net) it
    leaks: no suction. MuJoCo's adhesion doesn't act on a flex (the bag of chips): on one, the cup pulls its nodes
    nearest the cup with the same force (`force` at full suction) itself."""

    def __init__(self, ramp, actuator, cup_geom=-1, cup_body=-1, force=0.0):
        self.ramp, self.actuator, self.cup_geom, self.cup_body, self.force = ramp, actuator, cup_geom, cup_body, force
        self.porous = set()
        self.held = []

    def __call__(self, scene, h):
        d, m = scene.data, scene.model
        f = max(0.0, min(1.0, self.ramp.current_pressure / 80e3))
        for b in self.held:
            d.xfrc_applied[b] = 0.0
        if self.held:
            d.xfrc_applied[self.cup_body] = 0.0
        self.held = []
        if f > 0:
            g = d.contact.geom[:d.ncon]
            mine = (g[:, 0] == self.cup_geom) | (g[:, 1] == self.cup_geom)
            other = g[mine].ravel()
            other = other[(other >= 0) & (other != self.cup_geom)]
            if self.porous and any(int(b) in self.porous for b in m.geom_bodyid[other]):
                f = 0.0
            flexes = d.contact.flex[:d.ncon][mine].ravel()
            flexes = flexes[flexes >= 0]
            if f > 0 and len(flexes):
                fid = int(flexes[0])
                adr, num = int(m.flex_vertadr[fid]), int(m.flex_vertnum[fid])
                nodes = np.unique(m.flex_vertbodyid[adr:adr + num])
                point = d.xpos[self.cup_body]
                near = nodes[np.argsort(np.linalg.norm(d.xpos[nodes] - point, axis=1))[:4]]
                F = f * self.force * d.xmat[self.cup_body].reshape(3, 3)[:, 2]
                for b in near:
                    d.xfrc_applied[b, :3] = F / len(near)
                d.xfrc_applied[self.cup_body, :3] = -F
                self.held = [int(b) for b in near]
        d.ctrl[self.actuator] = f


class VacuumCup:
    """A vacuum suction cup (GripWiq SofTouch, SoftGripping's): its flexible silicone lip is a ring of segments, each
    on a sprung slide joint so the ring follows a curved or uneven surface. Under vacuum (the grip control: full at
    80 kPa) it pulls the item it touches with the vacuum times the area inside the lip, along the cup's axis (and the
    cup the other way), as far as the lip seals: the share of the lip's segments touching that one item (a tilted or
    small surface leaks).

    A deformable item (a MuJoCo flex: the bag of chips) counts as one item: the pull is shared by its nodes under
    the cup. Bodies in `porous` (the potatoes: the cup would sit on the bag's net) seal but are never pulled."""

    def __init__(self, ramp, cup_body, lip_geoms, area, vacuum, porous=()):
        self.ramp, self.cup_body, self.area, self.vacuum = ramp, cup_body, area, vacuum
        self.lip_geoms = set(int(g) for g in lip_geoms)
        self.n = max(len(self.lip_geoms), 1)
        self.porous = set(int(b) for b in porous)
        self.seal = 0.0
        self.held = []
        self.force = 0.0
        self._latched = None
        self.reach = 0.0  # [m] how far a latched bag's nearest node may be from the cup (set by the scene)

    def __call__(self, scene, h):
        self._tick = getattr(self, '_tick', -1) + 1
        if self._tick % 4:  # (every 4 ms: the force stays applied in between)
            return
        d, m = scene.data, scene.model
        for b in self.held:
            d.xfrc_applied[b] = 0.0
        d.xfrc_applied[self.cup_body] = 0.0
        self.held, self.force = [], 0.0
        if not hasattr(self, '_lips'):
            self._lips = np.array(sorted(self.lip_geoms))
        n = d.ncon
        if not n:
            self.seal = 0.0
            if self._still_held(d, m):
                self._pull(d, m, self._latched, self.n)
            return
        geoms, flexes = d.contact.geom[:n], d.contact.flex[:n]
        on1, on2 = np.isin(geoms[:, 0], self._lips), np.isin(geoms[:, 1], self._lips)
        rows1, rows2 = np.nonzero(on1 & ~on2)[0], np.nonzero(on2 & ~on1)[0]
        lip = np.concatenate([geoms[rows1, 0], geoms[rows2, 1]])
        if not len(lip):
            self.seal = 0.0
            if not self._still_held(d, m):
                return
            best, count = self._latched, self.n
        else:
            best, count = self._touching(d, m, n, geoms, flexes, rows1, rows2, lip)
        self._pull(d, m, best, count)

    def _still_held(self, d, m):
        """A bag sucked onto the cup (latched) stays on it while its nearest node is within the cup's reach, even
        when its coarse surface swings off the lip for a moment."""
        fid = getattr(self, '_latched', None)
        if fid is None:
            return False
        f_ = -1 - fid
        adr, num = int(m.flex_vertadr[f_]), int(m.flex_vertnum[f_])
        nodes = m.flex_vertbodyid[adr:adr + num]
        near = float(np.min(np.linalg.norm(d.xpos[nodes] - d.xpos[self.cup_body], axis=1)))
        if near > self.reach:
            self._latched = None
            return False
        self.seal = 1.0
        return True

    def _touching(self, d, m, n, geoms, flexes, rows1, rows2, lip):
        # What each touching lip segment touches: a body (a geom's), or a flex (-1 - its id)
        og = np.concatenate([geoms[rows1, 1], geoms[rows2, 0]])
        of = np.concatenate([flexes[rows1, 1], flexes[rows2, 0]])
        keys = np.where(og >= 0, m.geom_bodyid[np.maximum(og, 0)], -1 - of)
        best, count = 0, 0
        for k in np.unique(keys):
            n_ = len(np.unique(lip[keys == k]))
            if n_ > count:
                best, count = int(k), n_
        return best, count

    def _pull(self, d, m, best, count):
        self.seal = count / self.n
        # A bag's foil (a flex) sucked onto the cup stays sealed while it touches the lip: the coarse bag can't
        # follow the lip all round once the vacuum pulls its top into the cup.
        if best < 0:
            if self.seal >= 0.9:
                self._latched = best
            elif getattr(self, '_latched', None) == best and (count > 0 or self._still_held(d, m)):
                self.seal = 1.0
        else:
            self._latched = None
        dp = self.vacuum * max(0.0, min(1.0, self.ramp.current_pressure / 80e3))
        if dp <= 0 or best == 0 or best in self.porous:
            return
        sealed = max(0.0, min(1.0, (self.seal - 0.55) / 0.35))
        f = dp * self.area * sealed * sealed * (3 - 2 * sealed)
        if f <= 0:
            return
        axis = d.xmat[self.cup_body].reshape(3, 3)[:, 2]  # the cup's +z: back into the gripper
        point = d.xpos[self.cup_body]
        F = f * axis
        if best > 0:
            d.xfrc_applied[best, :3] = F
            d.xfrc_applied[best, 3:] = np.cross(point - d.xipos[best], F)
            self.held = [best]
        else:  # a flex: its four nodes nearest the cup share the pull
            fid = -1 - best
            adr, num = int(m.flex_vertadr[fid]), int(m.flex_vertnum[fid])
            nodes = np.unique(m.flex_vertbodyid[adr:adr + num])
            off = np.linalg.norm(d.xpos[nodes] - point, axis=1)  # (the nearest: on the side the cup is on)
            near = nodes[np.argsort(off)[:4]]
            for b in near:
                d.xfrc_applied[b, :3] = F / len(near)
            self.held = [int(b) for b in near]
        d.xfrc_applied[self.cup_body, :3] = -F
        d.xfrc_applied[self.cup_body, 3:] = np.cross(point - d.xipos[self.cup_body], -F)
        self.force = f


class HandController:
    """The Mia hand's three motors from the grip control (0: open, a little past straight; 80 kPa and more:
    closed): position servos whose torque is limited, so a finger stops on the item and the others close on."""

    OPEN, CLOSED = -0.45, 1.45  # [rad] at the fingers' base joints

    def __init__(self, ramp, actuators):
        self.ramp, self.actuators = ramp, list(actuators)

    def __call__(self, scene, h):
        f = max(0.0, min(1.0, self.ramp.current_pressure / 80e3))
        scene.data.ctrl[self.actuators] = self.OPEN + f * (self.CLOSED - self.OPEN)


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


def _v(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def _scene_class():
    try:
        from sofaweb.mjscene import MjScene
    except ImportError:  # run from the repository: scenes/_base/sofaweb
        sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..', '..', '_base'))
        from sofaweb.mjscene import MjScene
    return MjScene


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


def build_gripper(P, gkind):
    """One gripper (GRIPPER_CHOICES' `gkind`) in its own frame, the tool point at the origin, the fingers pointing
    down (-z); its root body 'flange' (a free body, welded to a mocap target that the test moves). Returns its MJCF pieces (names
    unprefixed: see _prefixed) and two functions to call once the model is built: visuals(scene, body, path) draws
    it, controllers(scene, model, jnt, act, body, geom) makes its grip controllers. `body`, `jnt`, `act`, `geom`: name -> id."""
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

    cav_l, cav_w, cav_h = max(L - 2 * we, 0.0), max(W - 2 * ws, 0.0), max(H - wb - wt, 0.0)
    seg_mass = float(P['density']) * (L * W * H - cav_l * cav_w * cav_h)
    cavity_area = cav_w * cav_h
    cavity_mid = wb + 0.5 * cav_h
    torque_per_pa = float(P['pressure_torque_gain']) * cavity_area * cavity_mid
    tips = bool(P['tips_enabled']) and float(P['tip_length']) > 0
    tip_l = max(float(P['tip_length']), 0.5) * MM
    tip_w = min(max(float(P['tip_width']), 0.2) * MM, W)
    tip_t = min(max(float(P['tip_thickness']), 0.2) * MM, H)

    thick = 0.6 * H
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

    assets = []
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
    actuators = []
    if gkind == 'mia':
        t_max = max(float(P['mia_torque']), 0.01)
        actuators += [f'<position name="mia_{n}" joint="mia_{n}_mcp" kp="3" kv="0.06" '
                      f'forcerange="{-t_max:.9g} {t_max:.9g}"/>' for n in ('thumb', 'index', 'middle')]
    if gkind == 'suction':
        actuators.append(f'<adhesion name="suction" body="cup" ctrlrange="0 1" '
                         f'gain="{max(float(P["suction_force"]), 0.0):.9g}"/>')

    def visuals(scene, body, path):
        flange_id = body('flange')

        def body_of(k, j):
            return flange_id if j == 0 else body(f'f{k + 1}s{j}')

        def origin_of(k, j):
            return flange_pos if j == 0 else pivots[(k, j)]

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
        scene.add_visual(f'{path}/bodies/visu/visual', pts, tris, bodies, soft_rgb + [float(P['wall_alpha'])],
                         texture='silicone')
        pts, tris, bodies = carried(sheet_parts)
        scene.add_visual(f'{path}/bodies/base_layer/visual', pts, tris, bodies,
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
            scene.add_visual(f'{path}/bodies/tips/visual', pts, tris, bodies, tip_rgb + [float(P['wall_alpha'])],
                             texture='silicone')
        if P['show_joint_marks'] and mark_parts:
            pts, tris, bodies = carried(mark_parts)
            scene.add_visual(f'{path}/bodies/joint_marks/visual', pts, tris, bodies,
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
            scene.add_visual(f'{path}/bodies/trunk/visual', mount_pts, mount_tris, flange_id, [0.4, 0.26, 0.13, 1.0],
                             texture='printed')
        else:  # the hand: palm and wrist, rounded phalanges, a skin pad on each phalanx's palmar side
            zr = z_palm
            palm_v = _merge([_box(0.010, -0.043, zr, 0.042, 0.043, zr + 0.09),
                             _cylinder_along([0.026, 0, zr + 0.09], [0.026, 0, zr + 0.115], 0.024, n=28),
                             # the thumb's housing: out from the palm to the thumb's root, round over its joint
                             _box(-0.03, -0.016, zr + 0.004, 0.0105, 0.016, zr + 0.05),
                             _box(-0.044, -0.016, zr + 0.017, -0.0295, 0.016, zr + 0.05),
                             _cylinder_along([-0.03, -0.016, zr + 0.017], [-0.03, 0.016, zr + 0.017], 0.014, n=24)])
            scene.add_visual(f'{path}/bodies/trunk/visual', *_smooth_shaded(*palm_v), flange_id,
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
                scene.add_visual(f'{path}/bodies/{vname}/visual', pts, tris, bodies,
                                 rgba + [float(P['wall_alpha'])], texture='plastic')
        if jaw_names:
            n_jaws = len(jaw_names)
            rail = (_box(-(stroke + jaw_t), -0.016, z_palm - 0.02, stroke + jaw_t, 0.016, z_palm) if n_jaws == 2 else
                    _frustum_z(0, 0, z_palm - 0.02, z_palm, stroke + jaw_t, stroke + jaw_t, n=36))
            scene.add_visual(f'{path}/bodies/rail/visual', *_smooth_shaded(*rail), flange_id,
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
                    scene.add_visual(f'{path}/bodies/finray_{chain}/visual', pts, tris, bodies,
                                     rgba + [float(P['wall_alpha'])], texture='plastic')
            for vname, parts, rgba, tex in (('jaws', jaw_parts, [0.72, 0.74, 0.78], 'brushed'),
                                            ('jaw_pads', pad_parts, [0.1, 0.1, 0.11], 'rubber')):
                if not parts:
                    continue
                fp = [(*_flat_shaded(pp, tt), b, o) for pp, tt, b, o in parts]
                pts, tris, bodies = carried(fp)
                scene.add_visual(f'{path}/bodies/{vname}/visual', pts, tris, bodies,
                                 rgba + [float(P['wall_alpha'])], texture=tex)
        for pad, pp, tt, b in skin_pads:
            scene.add_visual(f'{path}/bodies/skin/{gkind}.{pad}/visual', *_flat_shaded(pp, tt), b,
                             [0.22, 0.24, 0.27, float(P['wall_alpha'])], texture='plastic')
        if lip_names:
            hub = _frustum_z(0, 0, 0.006, z_palm - cup_lift, 0.75 * cup_r, 0.4 * cup_r, n=32)
            scene.add_visual(f'{path}/bodies/cup_stem/visual', *_smooth_shaded(*hub), body('vcup'),
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
            scene.add_visual(f'{path}/bodies/cup/visual', pts, tris, bodies, [0.9, 0.93, 0.96, float(P['wall_alpha'])],
                             texture='silicone')
        if gkind == 'suction':
            cup_id = body('cup')
            stem_m = _cylinder_along([0, 0, 0.016], [0, 0, z_palm], 0.009, n=16)
            bellows = _merge([_frustum_z(0, 0, 0.012 + 0.008 * k, 0.016 + 0.008 * k, 0.014, 0.011, n=24) for k in range(3)])
            lip = _frustum_z(0, 0, 0.0, 0.012, 0.02, 0.012, n=32)
            scene.add_visual(f'{path}/bodies/cup_stem/visual', *_smooth_shaded(*stem_m), cup_id,
                             [0.72, 0.74, 0.78, float(P['wall_alpha'])], texture='brushed')
            scene.add_visual(f'{path}/bodies/cup/visual', *_smooth_shaded(*_merge([bellows, lip])), cup_id,
                             [0.12, 0.12, 0.13, float(P['wall_alpha'])], texture='rubber')
        return skin_names

    def controllers(scene, model, jnt, act, body, geom):
        hinge_dofs = [int(model.jnt_dofadr[jnt(n)]) for n in hinge_names]
        pressure = PressureRamp(hinge_dofs, torque_per_pa, 0.0, float(P['ramp_time']), instant=bool(P['instant']),
                                ramp_start=float(P['ramp_start']))
        scene.controllers.append(pressure)
        jaws = suction = None
        if jaw_names:
            jaws = JawController(pressure, [int(model.jnt_dofadr[jnt(n)]) for n in jaw_names],
                                 [int(model.jnt_qposadr[jnt(n)]) for n in jaw_names], stroke,
                                 max(float(P['jaw_force']), 0.0))
            scene.controllers.append(jaws)
        if gkind == 'suction':
            suction = SuctionController(pressure, act('suction'), geom('cup'), body('cup'),
                                        max(float(P['suction_force']), 0.0))
            scene.controllers.append(suction)
        if mia:
            hand = HandController(pressure, [act(f'mia_{n}') for n in ('thumb', 'index', 'middle')])
            scene.controllers.append(hand)
            data_ctrl = scene.data.ctrl
            data_ctrl[hand.actuators] = HandController.OPEN
        vacuum = None
        if lip_names:
            lip_geoms = [geom(n) for n in lip_names]
            vacuum = VacuumCup(pressure, body('vcup'), lip_geoms, math.pi * (cup_r - 0.002) ** 2,
                               max(float(P['cup_vacuum']), 0.0) * 1e3)
            scene.controllers.append(vacuum)
        if gkind.startswith('softgripping'):
            pressure.idle = -max(float(P['sg_open_vacuum']), 0.0) * 1e3
        return {'pressure': pressure, 'jaws': jaws, 'vacuum': vacuum, 'suction': suction}

    if gkind in SOFT_FINGERS:  # how far below the tool point it reaches, and the palm above it
        below = max(finger_len + (tip_l if tips else 0.0) - tcp_depth, 0.0)
        palm = tcp_depth
    elif finray:
        below, palm = max(float(P['finray_length']), 20.0) * MM + 0.02 - tcp_depth, tcp_depth - 0.02
    elif mia:
        below, palm = 0.085 - tcp_depth + 0.004, tcp_depth
    elif jaw_names:
        below, palm = jaw_l + 0.02 - tcp_depth, tcp_depth - 0.02
    else:
        below, palm = 0.0, tcp_depth
    return {'kind': gkind, 'bodies': gripper, 'assets': assets, 'equalities': equalities, 'excludes': excludes,
            'actuators': actuators, 'visuals': visuals, 'controllers': controllers, 'grip_top': grip_top,
            'hinges': hinge_names + side_names, 'stroke': stroke, 'jaws': bool(jaw_names), 'cup_r': cup_r,
            'cup_lift': cup_lift, 'vacuum': gkind == 'suction' or gkind in VACUUM_CUPS, 'fingers': len(frames),
            'reach_below_tcp': below, 'palm_above_tcp': palm, 'plate_r': plate_r, 'finray': bool(finray),
            'mia': bool(mia), 'segments': N}


# ----------------------------------------------------------------------
# The items to choose from
# ----------------------------------------------------------------------
def _hagelslag_units():
    try:
        with open(os.path.join(SHELF_DIR, 'placements.json'), 'r') as f:
            return json.load(f)['units']
    except (OSError, ValueError, KeyError):
        return []


def item_choices():
    """[(value, text)]: the deformable items, Gripper 7's groceries, the Hagelslag shelf's products."""
    out = [('chips', 'Bag of chips 150 g (deformable, size as set)')]
    out += [(k, f'{v[0][0].upper()}{v[0][1:]} (deformable)') for k, v in CHIPS_SIZES.items()]
    out += [('potatoes', 'Bag of potatoes (deformable)')]
    out += [(f'g7_{k}', f'Gripper 7: {v[0]}') for k, v in PRODUCTS.items()]
    seen = set()
    for u in _hagelslag_units():
        if u['product'] not in seen:
            seen.add(u['product'])
            out.append((f"hs_{u['product']}", f"Hagelslag: {u['name']}"))
    return out


def hagelslag_product(product):
    """One product of the Hagelslag shelf (hagelslag_shelf/, hagelslag_physics.json): its front package's meshes
    about its centre, in the scene's frame (x, y = the shelf's depth, z up), with its collision shape, mass and
    friction (as HagelslagMj.hagelslag_shelf)."""
    units = [u for u in _hagelslag_units() if u['product'] == product]
    if not units:
        raise ValueError(f"no product {product!r} in hagelslag_shelf/placements.json")
    gltf, accessor, image = read_glb(os.path.join(SHELF_DIR, 'shelf.glb'))
    physics = load_physics()
    nodes = {n['name']: n for n in gltf['nodes'] if 'mesh' in n}
    centres = np.array([u['center_m'] for u in units])
    front = int(np.argmax(centres[:, 2]))  # (the model's +z: towards the aisle)
    u = units[front]
    images, parts = {}, []
    for pr in gltf['meshes'][nodes[f'p_{product}']['mesh']]['primitives'] if f'p_{product}' in nodes else []:
        pts = accessor(pr['attributes']['POSITION']).astype(np.float64)
        tris = accessor(pr['indices']).reshape(-1, 3).astype(np.int64)
        mat = gltf['materials'][pr['material']] if 'material' in pr else {}
        pbr = mat.get('pbrMetallicRoughness', {})
        rgba = list(pbr.get('baseColorFactor', [1.0, 1.0, 1.0, 1.0]))
        rgba[3] = 1.0
        img, uv = None, None
        tex = pbr.get('baseColorTexture')
        if tex is not None and 'TEXCOORD_0' in pr['attributes']:
            src = gltf['textures'][tex['index']]['source']
            mime, raw = image(src)
            img = f"{mat.get('name', 'image')}_{src}.{'png' if mime == 'image/png' else 'jpg'}"
            images[img] = (mime, raw)
            uv = accessor(pr['attributes']['TEXCOORD_0']).astype(np.float64)
        labels = _components(tris, len(pts))
        for lab in np.unique(labels):  # the parts nearest the front package's centre are its own
            t = tris[labels == lab]
            used, local_t = np.unique(t, return_inverse=True)
            if int(np.argmin(np.linalg.norm(centres - pts[used].mean(axis=0), axis=1))) != front:
                continue
            q = pts[used] - np.array(u['center_m'])
            parts.append((np.stack([q[:, 0], -q[:, 2], q[:, 1]], axis=1), local_t.reshape(-1, 3),
                          None if uv is None else uv[used], rgba, img))
    cfg = {**_KIND_DEFAULT, **physics['kinds'].get(u['kind'], {}), **physics['products'].get(product, {})}
    w, h, d = (float(v) for v in u['size_m'])
    size = np.array([w, d, h])
    clearance = max(float(physics.get('clearance_mm', 1.0)), 0.0) * MM
    half = np.maximum(size / 2 - clearance / 2, 0.002)
    shape = 'cylinder' if cfg.get('shape') == 'cylinder' else 'box'
    if shape == 'cylinder':
        half[0] = half[1] = min(half[0], half[1])
    if not parts:
        bp, bt = _box(*(-size / 2), *(size / 2))
        parts.append((np.asarray(bp, float), np.asarray(bt), None, [0.7, 0.6, 0.45, 1.0], None))
    mass = float(cfg['mass']) if 'mass' in cfg else float(cfg['density']) * w * h * d * 1000.0
    return {'name': u['name'], 'shape': shape, 'half': half, 'size': size, 'mass': mass,
            'friction': float(cfg['friction']), 'parts': parts, 'images': images}


def item_spec(P, key):
    """The chosen item: {'key', 'kind' ('chips', 'potatoes', 'g7', 'hs'), 'label', 'shape' and 'dims' (as Gripper 7's
    PRODUCTS, metres: what the grasp is planned for), 'half_h' (its centre's height on the floor), 'size' (its
    footprint and height), and per kind its details}."""
    scale = max(float(P['item_mass_scale']), 1e-3)
    if key == 'chips' or key in CHIPS_SIZES:
        if key == 'chips':
            label = 'bag of chips (150 g)'
            L, W, T = (max(float(P[k]), 10.0) * MM for k in ('chips_length', 'chips_width', 'chips_thickness'))
            mass = max(float(P['chips_mass']), 1.0) * 1e-3
        else:
            label, L, W, T, mass = CHIPS_SIZES[key]
            L, W, T, mass = L * MM, W * MM, T * MM, mass * 1e-3
        rim = min(max(float(P['chips_roundness']), 0.2), 0.95)
        return {'key': key, 'kind': 'chips', 'label': label, 'shape': 'box', 'dims': [L, W, T],
                'half_h': T / 2, 'size': [L, W, T], 'mass': mass * scale, 'rim': rim}
    if key == 'potatoes':
        n = min(max(int(P['potato_count']), 1), 12)
        r = max(float(P['potato_radius']), 10.0) * MM
        centres = []
        for k in range(n):  # rows of three along x, two rows, then a layer above (in the dips)
            layer, rest = divmod(k, 6)
            row, col = divmod(rest, 3)
            centres.append([(col - 1) * (2 * r + 0.002) + layer * r, (row - 0.5) * (2 * r + 0.002),
                            r + layer * 1.55 * r])
        centres = np.array(centres)
        centres[:, :2] -= centres[:, :2].mean(axis=0)
        lo, hi = centres.min(axis=0) - r, centres.max(axis=0) + r
        return {'key': key, 'kind': 'potatoes', 'label': 'bag of potatoes', 'shape': 'box',
                'dims': list(hi - lo), 'half_h': float(hi[2] / 2), 'size': list(hi - lo), 'r': r,
                'centres': centres, 'mass': max(float(P['potato_mass']), 1.0) * 1e-3 * scale}
    if key.startswith('hs_'):
        pk = hagelslag_product(key[3:])
        h = pk['half']
        dims = [h[0], 2 * h[2]] if pk['shape'] == 'cylinder' else list(2 * h)
        return {'key': key, 'kind': 'hs', 'label': pk['name'], 'shape': pk['shape'], 'dims': dims,
                'half_h': float(h[2]), 'size': list(2 * h), 'pk': pk, 'mass': pk['mass'] * scale}
    product = key[3:] if key.startswith('g7_') else key
    if product not in PRODUCTS:
        print(f"[model] unknown item {key!r}: the bag of chips instead")
        return item_spec(P, 'chips')
    label, shape, dims_mm, mass_g = PRODUCTS[product]
    dims = [d * MM for d in dims_mm]
    half_h, _ = _item_extent(shape, dims)
    if shape == 'box':
        fx, fy = dims[0], dims[1]
    elif shape == 'egg':
        fx, fy = 2 * dims[0], 2 * dims[1]
    elif shape == 'banana':
        fx, fy = dims[0], dims[2] + 2 * dims[1]
    elif shape == 'cup':
        fx = fy = 2 * max(dims[0], dims[1])
    else:
        fx = fy = 2 * dims[0]
    return {'key': key, 'kind': 'g7', 'label': label.lower(), 'product': product, 'shape': shape, 'dims': dims,
            'half_h': half_h, 'size': [fx, fy, 2 * half_h], 'mass': mass_g * 1e-3 * scale}


def grasp_plan(g, it):
    """How gripper `g` (build_gripper's) takes item `it` lying on the floor, from above: the tool point's height
    'z', the gripper's turn 'yaw' about z (the jaws squeeze along the gripper's x), 'open_to' (the jaws: from the
    axis), the grasp point's offset from the item's centre (x, y) and whether the gripper can do it at all (a note
    if it likely can't: it still tries)."""
    shape, dims, hh = it['shape'], it['dims'], it['half_h']
    H, top, bottom = 2 * hh, 2 * hh, 0.0
    off = np.zeros(2)
    if shape == 'box':
        dx, dy = dims[0], dims[1]
    elif shape == 'egg':
        dx, dy = 2 * dims[0], 2 * dims[1]
    elif shape == 'banana':
        dx, dy = dims[0], 2 * dims[1]
        off = np.array([0.0, dims[2] / 2])  # the curve's middle
    elif shape == 'cup':
        dx = dy = 2 * max(dims[0], dims[1])
    else:
        dx = dy = 2 * dims[0]
    kind, note = g['kind'], ''
    yaw, open_to = 0.0, None
    if kind in ('parallel', 'festo_finray', 'mia', 'softgripping_parallel'):  # (they squeeze along their x)
        yaw = 0.0 if dx <= dy else math.pi / 2
        width = min(dx, dy)
        if g['jaws']:
            open_to = width / 2 + (0.008 if it['kind'] == 'chips' else 0.014)  # (a bag gives)
    elif kind == 'three_jaw':
        open_to = (dims[1] if shape == 'banana' else 0.5 * math.hypot(dx, dy) if shape in ('box', 'egg')
                   else dx / 2) + 0.012
    elif g['fingers'] and it['kind'] == 'chips' and min(dx, dy) / 2 + 0.008 <= g['stroke'] - 0.002:
        yaw = 0.0 if g['fingers'] == 2 else math.pi / 4 if dx > 1.3 * dy else 0.0  # (a narrow bag: wrapped round)
    elif g['fingers'] and it['kind'] in ('chips', 'potatoes'):
        yaw = math.pi / max(g['fingers'], 1) if g['fingers'] > 2 else 0.0  # (the fingers on the diagonals of a bag)
    if open_to is not None and open_to > g['stroke'] - 0.002:
        note = f"too big for the jaws ({2 * open_to * 1000:.0f} mm across)"
        open_to = g['stroke']
    if g['vacuum']:
        cup_r = g['cup_r'] if g['cup_r'] > 0 else 0.019
        curv = {'sphere': lambda: dims[0], 'apple': lambda: dims[0] ** 2 / dims[1], 'egg': lambda: dims[1],
                'banana': lambda: dims[1], 'pear': lambda: 0.25 * dims[0]}.get(shape, lambda: None)()
        if it['kind'] == 'potatoes':
            curv = it['r']
            note = 'a cup on a net bag leaks'
        flat = {'cylinder': lambda: dims[0], 'cup': lambda: dims[1],
                'box': lambda: 0.5 * min(dx, dy)}.get(shape, lambda: None)()
        sag = 0.0
        if flat is not None and flat < 0.95 * cup_r:
            note = f"its top ({2 * flat * 1000:.0f} mm) is smaller than the cup ({2 * cup_r * 1000:.0f} mm)"
        if curv is not None:
            if curv <= cup_r:
                note = note or f"too small or round for the cup ({2 * curv * 1000:.0f} mm across)"
            else:
                sag = curv - math.sqrt(curv * curv - cup_r * cup_r)
        z = top - min(sag + 0.003, 0.0145) - g['cup_lift']
        if it['kind'] == 'chips':
            z -= 0.01  # (a soft bag: pressed in a little more)
    else:
        low = {'pear': 0.08, 'banana': 0.1, 'egg': 0.12}.get(shape, 0.3 if g['fingers'] else 0.25)
        if it['kind'] == 'potatoes':
            low = 0.15
        z = max(bottom + low * H + g['reach_below_tcp'], top - g['palm_above_tcp'] + 0.005)
        fits = min(dx, dy) / 2 + 0.008 <= g['stroke'] - 0.002 and not (kind == 'three_jaw' and note)
        if it['kind'] == 'chips' and fits:  # a bag that fits between the jaws or fingers: grasped across it
            z = max(bottom + 0.3 * H + g['reach_below_tcp'], top - g['palm_above_tcp'] + 0.005)
        elif it['kind'] == 'chips':  # a bag lying flat, wider than the grippers: they close on its top, 15 mm deep
            z = max(top - 0.015 + g['reach_below_tcp'], top - g['palm_above_tcp'] + 0.005)
    return {'z': float(z), 'yaw': yaw, 'open_to': open_to, 'offset': off, 'note': note}


# ----------------------------------------------------------------------
# The deformable items
# ----------------------------------------------------------------------
CHIPS_NODES = (4, 3)  # the bag of chips' grid of nodes along x and y (and two layers: its top and its bottom)
CHIPS_JAW_FORCE = 0.2  # the jaws' force on a bag of chips, of their usual (40 N at full pressure: crushes it)
# The smaller bags of chips: item key -> (label, length, width, thickness [mm] filled, lying flat; mass [g])
CHIPS_SIZES = {'chips_40': ('bag of chips (40 g)', 165.0, 105.0, 65.0, 44.0),
               'chips_25': ('bag of chips (25 g)', 130.0, 85.0, 55.0, 28.0),
               'chips_15': ('mini bag of chips (15 g)', 105.0, 70.0, 45.0, 17.0)}


def chips_nodes(L, W, T, nx=CHIPS_NODES[0], ny=CHIPS_NODES[1], rim=0.6, seal=0.3):
    """The bag's nodes at rest: a pillow T thick in the middle, `rim` x T at its long sides and `seal` x T at its
    (sealed) ends; node (i, j, k) is (i * ny + j) * 2 + k (k 0: the bottom, 1: the top)."""
    pts = []
    for i in range(nx):
        x = -L / 2 + L * i / (nx - 1)
        for j in range(ny):
            y = -W / 2 + W * j / (ny - 1)
            f = (1 - (1 - seal) * (2 * x / L) ** 2) * (1 - (1 - rim) * (2 * y / W) ** 2)
            h = T / 2 * max(f, seal * rim)
            pts += [[x, y, -h], [x, y, h]]
    return np.array(pts)


def grid_tets(nx, ny, nz, pts):
    """A grid of nodes (index (i * ny + j) * nz + k) split into tetrahedra: six per cell (Kuhn's, along its
    diagonal), each with a positive volume."""
    idx = lambda i, j, k: (i * ny + j) * nz + k
    out = []
    for i, j, k in itertools.product(range(nx - 1), range(ny - 1), range(nz - 1)):
        c = [idx(i + a, j + b, k + e) for a in (0, 1) for b in (0, 1) for e in (0, 1)]
        for path in itertools.permutations((4, 2, 1)):
            seq, cur = [c[0]], 0
            for s in path:
                cur += s
                seq.append(c[cur])
            a, b, e, f = (pts[q] for q in seq)
            if np.dot(np.cross(b - a, e - a), f - a) < 0:
                seq = [seq[0], seq[2], seq[1], seq[3]]
            out.append(seq)
    return np.array(out)


def _spline_weights(u, n):
    """1-D weights over n nodes for the parameter u (0 .. n-1): a Catmull-Rom spline through them (end nodes
    mirrored), straight on beyond the ends."""
    w = np.zeros(n)
    if u <= 0:
        w[0], w[1] = 1 - u, u
        return w
    if u >= n - 1:
        t = u - (n - 1)
        w[n - 1], w[n - 2] = 1 + t, -t
        return w
    i = min(int(math.floor(u)), n - 2)
    t = u - i
    basis = 0.5 * np.array([-t ** 3 + 2 * t ** 2 - t, 3 * t ** 3 - 5 * t ** 2 + 2, -3 * t ** 3 + 4 * t ** 2 + t,
                            t ** 3 - t ** 2])
    for b, k in zip(basis, (i - 1, i, i + 1, i + 2)):
        if k < 0:  # (a mirrored node: 2 P0 - P1)
            w[0] += 2 * b
            w[1] -= b
        elif k > n - 1:
            w[n - 1] += 2 * b
            w[n - 2] -= b
        else:
            w[k] += b
    return w


def chips_surface(nx, ny, nu=22, nv=17, seal=0.3, side=0.0):
    """The bag's surface as weights on its nodes: the top sheet and the bottom sheet, which meet at the rim; flat
    sealed strips (`seal` of a cell) at the ends along x; round long sides, bulging `side` (of a cell) out beyond the
    side nodes. Returns (weights_top, weights_bottom, triangles) with the weights (points x nodes) and one sheet's
    triangles."""
    us = np.linspace(-seal, nx - 1 + seal, nu)
    vs = np.linspace(-side, ny - 1 + side, nv)
    tops, bottoms = [], []
    for u in us:
        wu = _spline_weights(u, nx)
        # how far in from the rim: 0 on the rim and the seals, 1 a third of a cell inside
        g_u = math.sin(0.5 * math.pi * min(max(min(u, nx - 1 - u) / 0.35, 0.0), 1.0))
        for v in vs:
            wv = _spline_weights(v, ny)
            if side > 0:  # round: a quarter ellipse from the bulge's edge in to the side nodes
                s = min(max((min(v, ny - 1 - v) + side) / side, 0.0), 1.0)
                g_v = math.sqrt(max(1 - (1 - s) ** 2, 0.0))
            else:
                g_v = math.sin(0.5 * math.pi * min(max(min(v, ny - 1 - v) / 0.35, 0.0), 1.0))
            g = min(g_u, g_v)
            plane = np.outer(wu, wv).ravel()
            for sheet, k in ((tops, 1), (bottoms, 0)):
                wk = 0.5 + (0.5 if k else -0.5) * g  # the top's share
                w = np.zeros(nx * ny * 2)
                w[0::2] = plane * (1 - wk)
                w[1::2] = plane * wk
                sheet.append(w)
    tris = []
    for a in range(nu - 1):
        for b in range(nv - 1):
            p0, p1, p2, p3 = a * nv + b, (a + 1) * nv + b, (a + 1) * nv + b + 1, a * nv + b + 1
            tris += [[p0, p1, p2], [p0, p2, p3]]
    return np.array(tops), np.array(bottoms), np.array(tris)


def _tube_mesh(a, b, r, n=4):
    """A thin tube from a to b (world): points and triangles."""
    d = b - a
    ln = np.linalg.norm(d)
    d = d / ln if ln > 1e-9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    ring = [r * (math.cos(2 * math.pi * k / n) * e1 + math.sin(2 * math.pi * k / n) * e2) for k in range(n)]
    return [a + q for q in ring] + [b + q for q in ring]


def _tube_tris(n=4):
    tris = []
    for k in range(n):
        k1 = (k + 1) % n
        tris += [[k, k1, n + k1], [k, n + k1, n + k]]
    return tris


def _bumpy_sphere(r, seed, n_lat=10, n_lon=18):
    """A potato's look: a sphere with a few gentle bumps (within 4 % of r; the physics is the sphere)."""
    pts, tris = _ellipsoid((0, 0, 0), (r, r, r), n_lat, n_lon)
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(4, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    out = []
    for p in pts:
        q = np.array(p) / r
        bump = sum(0.02 * math.exp(-4 * (1 - float(q @ dd))) for dd in dirs) - 0.02
        out.append(list(np.array(p) * (1 + bump)))
    return out, tris


# ----------------------------------------------------------------------
# The test
# ----------------------------------------------------------------------
def _minjerk(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


class GripTest:
    """The test, the same for every gripper at the same time: a timetable of phases. Each gripper's mocap body
    follows its own heights (its grasp height) on the common timetable; the grip goes on and off for all at once.
    Each gripper's item is watched: 'dropped' when it's more than 3 cm (a bag: 8 cm) further from the tool point
    than after the lift, for a quarter of a second,
    'placed' when it lies at its goal (within 5 cm) after the release."""

    def __init__(self, rigs, P, item_label, deformable=False):
        self.rigs = rigs
        self.item_label = item_label
        # how far the item's centre may move from where it hung after the lift (a bag's centre moves as it bends)
        self.tolerance = 0.08 if deformable else 0.03
        self._away = [None] * len(rigs)
        z_lift = max(float(P['lift_height']), 100.0) * MM
        self.z_lift = z_lift
        self.travel = float(P['travel']) * MM
        v = max(float(P['move_speed']), 10.0) * MM
        drop = max(z_lift - r['z_grasp'] for r in rigs)
        t_v = max(drop / v, 0.8)
        t_turn = 360.0 / max(float(P['turn_speed']), 5.0)
        self.grip = max(float(P['grip_pressure']), 0.0) * 1e3
        grip_t = max(float(P['ramp_time']), 0.0) + max(float(P['grip_wait']), 0.0)
        self.phases = [('settle', 0.6), ('down', t_v), ('grip', grip_t), ('lift', t_v), ('turn', t_turn),
                       ('hold', 0.5), ('carry', max(abs(self.travel) / v, 0.8)), ('lower', t_v),
                       ('release', max(float(P['ramp_time']), 0.0) + max(float(P['release_wait']), 0.0)),
                       ('rise', t_v), ('done', float('inf'))]
        self.starts = np.cumsum([0.0] + [d for _, d in self.phases[:-1]])
        self.phase, self.phase_index, self.roll = 'settle', 0, 0.0
        self.status = [0] * len(rigs)  # 0 waiting, 1 holding, 2 placed at its goal, -1 dropped, -2 not placed
        self.where = [''] * len(rigs)
        self._ref = [None] * len(rigs)
        self._last = -1
        self._reported = False

    @property
    def duration(self):
        return float(self.starts[-1])

    def _pose(self, rig, t):
        """Its tool point (world) and turn (roll about x) at time t."""
        name, k = self.phase, self.phase_index
        u = _minjerk((t - self.starts[k]) / self.phases[k][1]) if math.isfinite(self.phases[k][1]) else 1.0
        zl, zg, zp = self.z_lift, rig['z_grasp'], rig['z_grasp'] + 0.004
        y, roll = 0.0, 0.0
        if name in ('settle',):
            z = zl
        elif name == 'down':
            z = zl + (zg - zl) * u
        elif name == 'grip':
            z = zg
        elif name == 'lift':
            z = zg + (zl - zg) * u
        elif name == 'turn':
            z, roll = zl, 2 * math.pi * u
        elif name == 'hold':
            z = zl
        elif name == 'carry':
            z, y = zl, self.travel * u
        elif name == 'lower':
            z, y = zl + (zp - zl) * u, self.travel
        elif name == 'release':
            z, y = zp, self.travel
        else:  # rise, done
            z, y = zp + (zl - zp) * (u if name == 'rise' else 1.0), self.travel
        return np.array([rig['x'], rig['y'] + y, z]), roll

    def __call__(self, scene, h):
        d = scene.data
        t = float(d.time)
        k = int(np.searchsorted(self.starts, t, side='right') - 1)
        k = min(max(k, 0), len(self.phases) - 1)
        self.phase_index, self.phase = k, self.phases[k][0]
        if k != self._last:
            self._enter(d, k)
        for i, rig in enumerate(self.rigs):
            p, roll = self._pose(rig, t)
            yaw = rig['yaw']
            d.mocap_pos[rig['mocap']] = p
            d.mocap_quat[rig['mocap']] = _quat_mul(np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]),
                                                   np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))
            if i == 0:
                self.roll = math.degrees(roll)
        self._watch(d, t)

    def _enter(self, d, k):
        self._last = k
        name = self.phase
        if name == 'settle':
            for rig in self.rigs:
                if rig['jaws'] is not None and rig['open_to'] is not None:
                    rig['jaws'].open_to = rig['open_to']
        if name == 'grip':
            for rig in self.rigs:
                rig['ramp'].targetPressure = self.grip
        if name == 'lift':
            for i, rig in enumerate(self.rigs):
                self.status[i] = 1
        if name == 'release':
            for rig in self.rigs:
                rig['ramp'].targetPressure = 0.0
        if math.isfinite(self.phases[k][1]):
            print(f"[test] {name} ({self.phases[k][1]:.1f} s)")

    def _watch(self, d, t):
        name = self.phase
        for i, rig in enumerate(self.rigs):
            at = rig['item_pos'](d)
            tcp = d.mocap_pos[rig['mocap']]
            if name in ('turn', 'hold', 'carry', 'lower') and self.status[i] == 1:
                dist = float(np.linalg.norm(at - tcp))
                if self._ref[i] is None:
                    self._ref[i] = dist
                elif dist <= self._ref[i] + self.tolerance:
                    self._away[i] = None
                elif self._away[i] is None:
                    self._away[i] = t
                elif t - self._away[i] > 0.25:  # (a quarter of a second out of place: gone)
                    self.status[i] = -1
                    self.where[i] = (f"in the turn at {self.roll:.0f} deg" if name == 'turn' else
                                     'while carrying it' if name in ('hold', 'carry') else 'on the way down')
                    print(f"[test] {rig['label']}: dropped the {self.item_label} {self.where[i]}")
            elif name == 'lift' and self.status[i] == 1:
                u = (t - self.starts[self.phase_index]) / self.phases[self.phase_index][1]
                if u > 0.7 and at[2] < rig['item_z0'] + 0.3 * (self.z_lift - rig['z_grasp']):
                    self.status[i] = -1
                    self.where[i] = 'when lifting it (it never came up)'
                    print(f"[test] {rig['label']}: didn't lift the {self.item_label}")
            elif name == 'done' and self.status[i] == 1:
                goal = np.array([rig['x'] + rig['item_dx'], rig['y'] + self.travel + rig['item_dy']])
                ok = np.linalg.norm(at[:2] - goal) < 0.05 and at[2] < rig['item_z0'] + 0.03
                self.status[i] = 2 if ok else -2
                if not ok:
                    self.where[i] = ("carried it, but couldn't let go (it came back up with the gripper)"
                                     if at[2] > rig['item_z0'] + 0.1 else 'carried it, but not put down at its goal')
                print(f"[test] {rig['label']}: " + (f'put the {self.item_label} down at its goal' if ok else
                                                    self.where[i]))
        if name == 'done' and not self._reported:
            self._reported = True
            print(f"[test] done: {self.summary()}")

    def summary(self):
        placed = sum(s == 2 for s in self.status)
        return f"{placed} of {len(self.rigs)} grippers carried the {self.item_label} through the test"


# ----------------------------------------------------------------------
# The scene
# ----------------------------------------------------------------------
BUILD = {}
ONLY = None  # (for tests: a set of GRIPPER_CHOICES keys, the line made of only those)
_REF_ATTRS = re.compile(r'\b(name|body1|body2|joint|joint1|joint2|body|mesh|site|tendon)="([^"]*)"')


def _prefixed(text, pre):
    """MJCF with every name (and every reference to one) given the prefix `pre`: one gripper's copy."""
    return _REF_ATTRS.sub(lambda m_: f'{m_.group(1)}="{pre}{m_.group(2)}"', text)


def _q_yaw(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def createScene():
    """Builds the MuJoCo model and returns an MjScene (see scenes/_base/sofaweb/mjscene.py)."""
    launch_params_editor()
    P = PARAMS
    MjScene = _scene_class()
    kinds = [k for k, _ in GRIPPER_CHOICES if ONLY is None or k in ONLY]
    labels = dict(GRIPPER_CHOICES)
    n = len(kinds)
    spacing = max(float(P['spacing']), 150.0) * MM
    z_lift = max(float(P['lift_height']), 100.0) * MM
    g = 9.81 if P['gravity_enabled'] else 0.0
    timestep = max(float(P['timestep']), 1e-5)
    mu = max(float(P['friction']), 0.0)
    tors = float(P['torsional_friction'])
    rolling = max(float(P['rolling_friction']), 0.0)
    it = item_spec(P, str(P['item']))
    grippers = [build_gripper(P, k) for k in kinds]
    plans = [grasp_plan(gr, it) for gr in grippers]
    xs = [(i - (n - 1) / 2) * spacing for i in range(n)]
    half_x = xs[-1] + spacing
    travel = float(P['travel']) * MM
    print(f"[model] {n} grippers in a row, {spacing / MM:.0f} mm apart, each over a {it['label']}")
    for k, pl in zip(kinds, plans):
        if pl['note']:
            print(f"[model] {labels[k]}: {pl['note']}")

    assets, world, equalities, excludes, actuators, tendons = [], [], [], [], [], []
    floor_half = [half_x + 0.6, max(travel, 0.0) + 1.0]
    world.append(f'<geom name="floor" type="plane" size="{_v([floor_half[0], floor_half[1] + 0.6, 0.05])}" '
                 f'pos="0 {max(travel, 0.0) / 2:.9g} 0" class="static"/>')

    # ---- the grippers: each on a mocap body at its tool point, over its item, turned for its grasp
    for i, (k, gr, pl) in enumerate(zip(kinds, grippers, plans)):
        pre = f'g{i + 1}_'
        body_xml = '\n'.join(gr['bodies'])
        # (A free body welded to a mocap target, as Gripper 5's: a mocap body itself has no velocity in the
        # dynamics, so the friction between its fingers and an item wouldn't carry the item along.)
        at = f'pos="{_v([xs[i], 0.0, z_lift])}" quat="{_v(_q_yaw(pl["yaw"]))}"'
        world.append(_prefixed(f'<body name="target" mocap="true" {at}/>', pre))
        world.append(_prefixed(f'<body name="flange" {at} gravcomp="1"><freejoint name="flange"/>\n{body_xml}\n'
                               f'</body>', pre))
        equalities.append(_prefixed(f'<weld body1="flange" body2="target" solref="{max(0.01, 4 * timestep):.9g} 1"/>',
                                    pre))
        assets += [_prefixed(a, pre) for a in gr['assets']]
        equalities += [_prefixed(e, pre) for e in gr['equalities']]
        excludes += [_prefixed(e, pre) for e in gr['excludes']]
        actuators += [_prefixed(a, pre) for a in gr['actuators']]

    # ---- the items: one under each gripper (its grasp point under the tool point)
    scale = max(float(P['item_mass_scale']), 1e-3)
    item_bodies = []  # per gripper: the item's body ids are known only after compiling; names for now
    chips = None
    if it['kind'] == 'chips':
        L, W, T = it['dims']
        nx, ny = CHIPS_NODES
        nodes0 = chips_nodes(L, W, T, nx, ny, rim=it['rim'])
        elements = grid_tets(nx, ny, 2, nodes0)
        chips = {'nodes': nodes0, 'elements': elements}
        young = max(float(P['chips_stiffness']), 0.1) * 1e3
        poisson = min(max(float(P['chips_poisson']), 0.0), 0.49)
    for i, pl in enumerate(plans):
        pre = f'g{i + 1}_'
        c = np.array([xs[i], 0.0]) - pl['offset']
        if it['kind'] == 'chips':
            world.append(f'<flexcomp name="{pre}chips" type="direct" dim="3" point="{_v(nodes0.ravel())}" '
                         f'element="{" ".join(str(int(q)) for q in elements.ravel())}" '
                         f'pos="{_v([c[0], c[1], T / 2 + 0.001])}" radius="0.001" mass="{it["mass"]:.9g}" '
                         f'rgba="0.9 0.3 0.1 0">'
                         f'<elasticity young="{young:.9g}" poisson="{poisson:.9g}" damping="0.001"/>'
                         f'<contact selfcollide="none" condim="4" contype="1" conaffinity="3" '
                         f'friction="{0.6:.9g} {tors:.9g} 0.0001" priority="1" solref="{max(2 * timestep, 0.001):.9g} 1" '
                         f'solimp="0.95 0.99 0.001"/>'  # (the bag's own: its nodes are light, so a softer
                         f'</flexcomp>')                 # contact lets the jaws sink in deep)
            item_bodies.append({'flex': f'{pre}chips', 'centre': [c[0], c[1], T / 2]})
        elif it['kind'] == 'potatoes':
            r, centres = it['r'], it['centres']
            names = []
            for j, p in enumerate(centres):
                name = f'{pre}potato{j + 1}'
                names.append(name)
                world.append(f'<body name="{name}" pos="{_v([c[0] + p[0], c[1] + p[1], p[2] + 0.0005])}">'
                             f'<freejoint name="{name}"/><geom name="{name}" type="sphere" size="{r:.9g}" '
                             f'mass="{it["mass"]:.9g}" condim="6" friction="{0.6:.9g} {tors:.9g} {rolling:.9g}" '
                             f'class="item"/><site name="{name}" pos="0 0 0"/></body>')
            # the clip that closes the net: beyond the potatoes at +x, at their middle's height
            clip = np.array([centres[:, 0].max() + r + 0.025, 0.0, r])
            world.append(f'<body name="{pre}clip" pos="{_v([c[0] + clip[0], c[1] + clip[1], clip[2]])}">'
                         f'<freejoint name="{pre}clip"/><geom name="{pre}clip" type="box" size="0.012 0.018 0.006" '
                         f'mass="0.004" class="item"/><site name="{pre}clip" pos="0 0 0"/></body>')
            slack = 1.0 + max(float(P['net_slack']), 0.0) / 100.0
            strings = []
            pts_all = list(centres) + [clip]
            for a in range(len(centres)):  # neighbours, and each potato to the clip
                for b in list(range(a + 1, len(centres))) + [len(centres)]:
                    dist = float(np.linalg.norm(pts_all[a] - pts_all[b]))
                    if b < len(centres) and dist > 2.6 * r:
                        continue
                    sa = names[a]
                    sb = names[b] if b < len(centres) else f'{pre}clip'
                    tendons.append(f'<spatial name="{pre}net{a + 1}_{b + 1}" limited="true" '
                                   f'range="0 {dist * slack:.9g}" solreflimit="0.01 1" width="0.001">'
                                   f'<site site="{sa}"/><site site="{sb}"/></spatial>')
                    strings.append((a, b))
            item_bodies.append({'bodies': names, 'clip': f'{pre}clip', 'strings': strings,
                                'centre': [c[0], c[1], it['half_h']]})
        elif it['kind'] == 'hs':
            pk = it['pk']
            h = pk['half']
            geom = (f'type="cylinder" size="{h[0]:.9g} {h[2]:.9g}" condim="6" '
                    f'friction="{pk["friction"]:.9g} {tors:.9g} {rolling:.9g}"' if pk['shape'] == 'cylinder' else
                    f'type="box" size="{_v(h)}" friction="{pk["friction"]:.9g} {tors:.9g} 0.0001"')
            name = f'{pre}item'
            world.append(f'<body name="{name}" pos="{_v([c[0], c[1], h[2] + 0.0003])}"><freejoint name="{name}"/>'
                         f'<geom name="{name}" {geom} mass="{it["mass"]:.9g}" class="item"/></body>')
            item_bodies.append({'body': name, 'centre': [c[0], c[1], h[2]]})
        else:
            product, shape, dims = it['product'], it['shape'], it['dims']
            name = f'{pre}item'
            if shape == 'sphere':
                geom = f'type="sphere" size="{dims[0]:.9g}"'
            elif shape == 'apple':
                geom = f'type="ellipsoid" size="{_v([dims[0], dims[0], dims[1]])}"'
            elif shape == 'egg':
                geom = f'type="ellipsoid" size="{_v([dims[0], dims[1], dims[1]])}"'
            elif shape == 'cylinder':
                geom = f'type="cylinder" size="{dims[0]:.9g} {dims[1] / 2:.9g}"'
            elif shape == 'box':
                geom = f'type="box" size="{_v([d_ / 2 for d_ in dims])}"'
            elif shape == 'banana':
                geom = ''
            else:  # the cup and the pear: convex meshes (one asset for all)
                if i == 0:
                    pts, _, _ = _item_meshes(shape, dims, smooth=False)[0]
                    assets.append(f'<mesh name="item_mesh" vertex="{_v(np.asarray(pts).ravel())}"/>')
                geom = 'type="mesh" mesh="item_mesh"'
            extra = (f' condim="6" friction="{mu:.9g} {tors:.9g} {rolling:.9g}"' if product in ROUND else '')
            if shape == 'banana':
                length, r, bend = dims
                axis = _banana_axis(length, bend)
                spans = list(zip(BANANA_CAPSULES[:-1], BANANA_CAPSULES[1:]))
                geoms = ''.join(
                    f'<geom name="{name}_{j}" type="capsule" fromto="{_v(list(axis(a)[0]) + list(axis(b)[0]))}" '
                    f'size="{r * min(_banana_radius(a), _banana_radius(b)):.9g}" mass="{it["mass"] / len(spans):.9g}" '
                    f'class="item"/>' for j, (a, b) in enumerate(spans))
            else:
                geoms = f'<geom name="{name}" {geom} mass="{it["mass"]:.9g}" class="item"{extra}/>'
            world.append(f'<body name="{name}" pos="{_v([c[0], c[1], it["half_h"] + 0.0002])}">'
                         f'<freejoint name="{name}"/>{geoms}</body>')
            item_bodies.append({'body': name, 'centre': [c[0], c[1], it['half_h']]})

    finger_bits = 'contype="2" conaffinity="3"' if P['finger_collisions'] else 'contype="2" conaffinity="1"'
    other_bits = 'contype="1" conaffinity="3"'
    # Flex elasticity (the bag of chips) needs MuJoCo's 'discrete' integrator; the rest is as Gripper 7's.
    # (and it can't run MuJoCo's no-slip passes with a flex: they're off then)
    integrator = 'discrete' if chips is not None else 'implicitfast'
    noslip = 0 if chips is not None else max(int(P['noslip_iterations']), 0)
    xml = f"""<mujoco model="gripper8">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep:.9g}" gravity="0 0 {-g:.9g}" cone="elliptic" impratio="{float(P['impratio']):.9g}"
          noslip_iterations="{noslip}" integrator="{integrator}"/>
  <default>
    <geom friction="{mu:.9g} {tors:.9g} 0.0001" condim="4" solref="{max(4 * timestep, 0.002):.9g} 1"/>
    <default class="finger"><geom {finger_bits}/></default>
    <default class="item"><geom {other_bits}/></default>
    <default class="static"><geom {other_bits}/></default>
  </default>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    {chr(10).join(world)}
  </worldbody>
  <contact>{''.join(excludes)}</contact>
  <equality>{''.join(equalities)}</equality>
  <tendon>{''.join(tendons)}</tendon>
  <actuator>{''.join(actuators)}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    scene = MjScene(model, dt=float(P['dt']))
    scene.xml = xml
    body = scene.body_id
    name2id = lambda kind: (lambda name: int(mujoco.mj_name2id(model, kind, name)))
    jnt_id, act_id, geom_id = name2id(mujoco.mjtObj.mjOBJ_JOINT), name2id(mujoco.mjtObj.mjOBJ_ACTUATOR), \
        name2id(mujoco.mjtObj.mjOBJ_GEOM)

    # ---- what the browser draws
    scene.background = [0.86, 0.89, 0.93, 1.0]
    scene.ambient = [0.42, 0.42, 0.42, 1.0]
    scene.add_light(direction=[-0.35, -0.6, 1.0])
    scene.add_light(color=[0.4, 0.4, 0.4, 1.0], direction=[0.6, 0.5, 0.4])
    # From the front (-y) and above: the whole line, and where the items go.
    cam_lookat = [0.0, 0.3 * travel, 0.15]
    cam_dir = (0.0, -1.0, 0.6)
    norm = math.sqrt(sum(c_ * c_ for c_ in cam_dir))
    cam_distance = 0.68 * (2 * half_x) + 0.4
    cam_position = [l + cam_distance * c_ / norm for l, c_ in zip(cam_lookat, cam_dir)]
    scene.set_camera(cam_position, _look_at_quaternion(cam_position, cam_lookat), cam_distance)

    # The floor: grey and white tiles; a start and a goal mark under each gripper; its name on a plate in front.
    tile = 0.3
    tiles = ([], [])
    ny0, ny1 = int(math.floor(-0.6 / tile)), int(math.ceil((max(travel, 0.0) + 0.6) / tile))
    nx_ = int(math.ceil((half_x + 0.6) / tile))
    for a in range(-nx_, nx_):
        for b in range(ny0, ny1):
            pts, tris = _box(a * tile, b * tile, -2e-4, (a + 1) * tile, (b + 1) * tile, -1e-4,
                             skip=('-Z', '-X', '+X', '-Y', '+Y'))
            tiles[(a + b) % 2].append((pts, tris))
    scene.add_visual('/floor/visu/visual', *_merge(tiles[0]), 0, [0.82, 0.82, 0.8, 1.0], texture='terrazzo')
    scene.add_visual('/floor/tiles/visual', *_merge(tiles[1]), 0, [0.68, 0.69, 0.7, 1.0], texture='terrazzo')
    marks = []
    for x in xs:
        for y in (0.0, travel):
            for a, b in ((0.0, 1.0), (math.pi / 2, 1.0)):  # a cross
                ca, sa = math.cos(a), math.sin(a)
                corners = [(-0.06, -0.003), (0.06, -0.003), (0.06, 0.003), (-0.06, 0.003)]
                q = [[x + ca * u - sa * v, y + sa * u + ca * v, 1e-4] for u, v in corners]
                marks.append((q, [[0, 1, 2], [0, 2, 3]]))
    scene.add_visual('/floor/marks/visual', *_merge(marks), 0, [0.95, 0.75, 0.1, 1.0], texture='painted')
    for i, k in enumerate(kinds):
        plate = os.path.join(PLATES_DIR, f'{k}.png')
        if not os.path.isfile(plate):
            continue
        with open(plate, 'rb') as f:
            scene.images[f'plate_{k}.png'] = ('image/png', f.read())
        w_, h_ = min(0.9 * spacing, 0.36), 0.09
        y_ = -0.2 - max(it['size'][1] / 2, 0.05)
        pts = [[xs[i] - w_ / 2, y_ - h_ / 2, 2e-4], [xs[i] + w_ / 2, y_ - h_ / 2, 2e-4],
               [xs[i] + w_ / 2, y_ + h_ / 2, 2e-4], [xs[i] - w_ / 2, y_ + h_ / 2, 2e-4]]
        scene.add_visual(f'/floor/plates/{k}/visual', pts, [[0, 1, 2], [0, 2, 3]], 0, [1.0, 1.0, 1.0, 1.0],
                         image=f'plate_{k}.png', uv=[[0, 1], [1, 1], [1, 0], [0, 0]])

    # The grippers (each with its own models, under /world/grippers/<kind>/bodies/...)
    skin_geoms, skin_pads = {}, []
    for i, (k, gr) in enumerate(zip(kinds, grippers)):
        pre = f'g{i + 1}_'
        pads = gr['visuals'](scene, lambda name, pre=pre: body(pre + name), f'/world/grippers/{k}')
        for pad, geoms in pads.items():
            key = f'{k}.{pad}'
            skin_pads.append(key)
            for gname in geoms:
                skin_geoms[geom_id(pre + gname)] = key

    # The items
    alpha = float(P['item_alpha'])
    item_pos = []
    if it['kind'] == 'chips':
        nx, ny = CHIPS_NODES
        # (round sides: bulging out a third of the side's half-thickness beyond the side nodes)
        L, W, T = it['dims']
        w_top, w_bot, tris = chips_surface(nx, ny, side=0.35 * it['rim'] * T / 2 / (W / (ny - 1)))
        rest_top, rest_bot = w_top @ chips['nodes'], w_bot @ chips['nodes']
        bot_tris = tris[:, ::-1]
        for i, ib in enumerate(item_bodies):
            f_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, ib['flex']))
            adr, num = int(model.flex_vertadr[f_id]), int(model.flex_vertnum[f_id])
            sl = slice(adr, adr + num)
            path = f'/world/items/bag{i + 1}'
            scene.add_dynamic_visual(f'{path}/top/visual', lambda d, w=w_top, s=sl: w @ d.flexvert_xpos[s],
                                     rest_top, tris, [0.86, 0.18, 0.12, alpha], texture='foil',
                                     label={'mode': 'plane', 'seed': 'chips', 'overlay': False})
            scene.add_dynamic_visual(f'{path}/bottom/visual', lambda d, w=w_bot, s=sl: w @ d.flexvert_xpos[s],
                                     rest_bot, bot_tris, [0.8, 0.16, 0.1, alpha], texture='foil')
            item_pos.append(lambda d, s=sl: d.flexvert_xpos[s].mean(axis=0))
    elif it['kind'] == 'potatoes':
        r = it['r']
        tube_t = _tube_tris()
        for i, ib in enumerate(item_bodies):
            ids = [body(nm) for nm in ib['bodies']]
            pts_all, tris_all, owners = [], [], []
            for j, b in enumerate(ids):
                pp, tt = _bumpy_sphere(r, seed=j)
                off = len(pts_all)
                pts_all += pp
                tris_all += [[a + off for a in t_] for t_ in tt]
                owners += [b] * len(pp)
            path = f'/world/items/bag{i + 1}'
            scene.add_visual(f'{path}/potatoes/visual', *_smooth_shaded(pts_all, tris_all)[:2], owners,
                             [0.72, 0.55, 0.32, alpha], texture='skin')
            clip_id = body(ib['clip'])
            cp, ct = _box(-0.012, -0.018, -0.006, 0.012, 0.018, 0.006)
            scene.add_visual(f'{path}/clip/visual', *_flat_shaded(cp, ct), clip_id, [0.95, 0.8, 0.1, alpha],
                             texture='plastic')
            ends = ids + [clip_id]
            strings = ib['strings']

            def net(d, ends=ends, strings=strings, n_p=len(ids)):
                out = []
                for a, b in strings:
                    pa, pb = d.xpos[ends[a]], d.xpos[ends[b]]
                    u = pb - pa
                    ln = float(np.linalg.norm(u))
                    u = u / ln if ln > 1e-9 else np.array([0.0, 0.0, 1.0])
                    # over the potatoes' surfaces (a little outside), not through them
                    lift = np.array([0.0, 0.0, 0.0])
                    out += _tube_mesh(pa + r * 1.02 * u + lift, pb - (r * 1.02 if b < n_p else 0.0) * u + lift,
                                      0.0012)
                return np.array(out)
            rest = net(scene.data)
            n_t = len(strings)
            scene.add_dynamic_visual(f'{path}/net/visual', net, rest,
                                     [[a + 8 * s for a in t_] for s in range(n_t) for t_ in tube_t],
                                     [0.85, 0.35, 0.1, alpha], texture='fabric')
            item_pos.append(lambda d, ids=ids: d.xpos[ids].mean(axis=0))
    else:
        if it['kind'] == 'hs':
            scene.images.update(it['pk']['images'])
        for i, ib in enumerate(item_bodies):
            b = body(ib['body'])
            path = f'/world/items/item{i + 1}'
            if it['kind'] == 'hs':
                for j, (pts, tris, uv, rgba, img) in enumerate(it['pk']['parts']):
                    scene.add_visual(f'{path}{"" if j == 0 else f"_{j}"}/visual', pts, tris, b,
                                     list(rgba[:3]) + [alpha], image=img, uv=uv if img else None)
            else:
                product, shape, dims = it['product'], it['shape'], it['dims']
                parts = [(p, t, ITEM_COLOURS[product]) for p, t, _ in _item_meshes(shape, dims, smooth=True)] + \
                    _item_details(product, dims)
                textures = ITEM_TEXTURES[product]
                for n_part, (vp, vt, colour) in enumerate(parts):
                    vp, vt = _smooth_shaded(vp, vt)
                    scene.add_visual(f'{path}/{"visu" if n_part == 0 else f"detail{n_part}"}/visual', vp, vt, b,
                                     list(colour) + [alpha], texture=textures[min(n_part, len(textures) - 1)],
                                     label={'mode': ITEM_LABELS[product][n_part], 'seed': product,
                                            'overlay': n_part > 0}
                                     if n_part in ITEM_LABELS.get(product, {}) else None)
            item_pos.append(lambda d, b=b: d.xipos[b].copy())

    # Coordinate axes on the floor, at the origin.
    arrow_ref = 0.12
    head_len, shaft_half = arrow_ref * 0.2, arrow_ref * 0.015
    for name, axis, color in (('arrow_x', 0, [0.9, 0.1, 0.1, 1.0]), ('arrow_y', 1, [0.1, 0.8, 0.1, 1.0])):
        pts, tris = _axis_arrow(axis, arrow_ref - head_len, head_len, shaft_half, shaft_half * 3)
        scene.add_visual(f'/{name}/visual', *_flat_shaded([[p[0] + xs[0] - 0.3, p[1] - 0.3, p[2] + 1e-4]
                                                           for p in pts], tris), 0, color, texture='painted')

    # ---- controllers: each gripper's grip, then the test
    rigs, porous = [], set()
    if it['kind'] == 'potatoes':
        porous = {body(nm) for ib in item_bodies for nm in ib['bodies'] + [ib['clip']]}
    mujoco.mj_forward(model, scene.data)
    for i, (k, gr, pl, ib) in enumerate(zip(kinds, grippers, plans, item_bodies)):
        pre = f'g{i + 1}_'
        ctl = gr['controllers'](scene, model, lambda nm, pre=pre: jnt_id(pre + nm),
                                lambda nm, pre=pre: act_id(pre + nm), lambda nm, pre=pre: body(pre + nm),
                                lambda nm, pre=pre: geom_id(pre + nm))
        for cup in (ctl['vacuum'], ctl['suction']):
            if cup is not None:
                cup.porous = porous
        if ctl['jaws'] is not None and chips is not None:  # (a bag of chips: gently)
            ctl['jaws'].force *= CHIPS_JAW_FORCE
        if ctl['vacuum'] is not None and chips is not None:  # (half a cell's diagonal, and some)
            nx, ny = CHIPS_NODES
            ctl['vacuum'].reach = 0.5 * math.hypot(it['dims'][0] / (nx - 1), it['dims'][1] / (ny - 1)) + 0.03
        mocap = int(model.body_mocapid[body(pre + 'target')])
        z0 = float(item_pos[i](scene.data)[2])
        rigs.append({'kind': k, 'label': labels[k], 'mocap': mocap, 'x': xs[i], 'y': 0.0, 'yaw': pl['yaw'],
                     'z_grasp': pl['z'], 'open_to': pl['open_to'], 'ramp': ctl['pressure'], 'jaws': ctl['jaws'],
                     'vacuum': ctl['vacuum'], 'item_pos': item_pos[i], 'item_z0': z0,
                     'item_dx': ib['centre'][0] - xs[i], 'item_dy': ib['centre'][1], 'note': pl['note']})
    test = GripTest(rigs, P, it['label'], deformable=it['kind'] in ('chips', 'potatoes'))
    scene.controllers.insert(0, test)
    scene.objects['Test'] = test

    BUILD.clear()
    BUILD.update({'kinds': kinds, 'labels': labels, 'rigs': rigs, 'item': it['key'], 'item_label': it['label'],
                  'skin_geoms': skin_geoms, 'skin_pads': skin_pads, 'notes': {k: pl['note'] for k, pl in
                                                                              zip(kinds, plans)}})
    return scene


def main():
    os.environ.setdefault('GRIPPER8MJ_NO_EDITOR', '1')
    scene = createScene()
    if '--headless' in sys.argv:
        import time
        test = scene.objects['Test']
        end = float(sys.argv[sys.argv.index('--seconds') + 1]) if '--seconds' in sys.argv else test.duration + 0.5
        start = time.time()
        while scene.data.time < end:
            scene.step(scene.dt.value)
        print(f"{scene.data.time:.1f} s simulated in {time.time() - start:.1f} s")
        for rig, s, w in zip(test.rigs, test.status, test.where):
            print(f"  {rig['label']:<50} {dict([(2, 'OK'), (1, 'holding'), (0, '-')]).get(s, 'FAILED')} {w}")
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
