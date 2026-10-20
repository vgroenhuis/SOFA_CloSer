# -*- coding: utf-8 -*-
"""
Gripper 4 (MuJoCo): the scene of ../../gripper-4/original/Gripper4.py -- rigid-segment fingers on a gripper that
moves in X, Y and Z over grocery items, with a large open box to put them in -- simulated with MuJoCo instead of
SOFA, to compare the two. The geometry, the items, the parameters and the controls are the same; the physics is
MuJoCo's own:

  - Each finger is a chain of `segments` rigid boxes (one per PneuNet chamber) joined by hinges at the base of the
    gaps between them (MuJoCo hinge joints, the bodies nested: a chain of generalised coordinates, as the
    ArticulatedSystemMapping in the SOFA scene). On each hinge: a torsional spring (`joint_stiffness`) with damping
    `joint_damping` x stiffness, hard limits (the roofs touching, `joint_open_limit`), and the pressure as a torque
    p A z (`pressure_torque_gain`) applied to the joint. The first segment is part of the end-effector.
  - The end-effector (`flange_mass`, its weight compensated) slides in x, y and z on three position actuators (the
    wrist: `wrist_stiffness`, critically damped) whose targets move smoothly towards the sliders' goals
    (`flange_acceleration`, `flange_speed`).
  - Contacts are MuJoCo's soft constraints with an elliptic friction cone (`friction`; `impratio` makes them stick
    better, for grasping), between the fingers, the plate, the items, the floor and the box. The items are MuJoCo's
    own shapes (sphere, ellipsoid for the egg, cylinder, box) and a convex mesh for the yogurt cup. Unlike the SOFA
    scene, the fingers can touch each other too (`finger_collisions`).
  - Each frame step `dt` is many MuJoCo steps of `timestep`.

Units: the scene is SI (metres, kg, s, Pa); params.json takes millimetres for lengths, grams for masses, as labelled.

Run with MuJoCo's viewer:   python Gripper4Mj.py       (pip install mujoco)
Headless check:             python Gripper4Mj.py --headless
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

# The grocery items: (key, label, shape, dimensions [mm], mass [g], where it starts [x, y mm]). Shapes:
#   'sphere' (r), 'egg' (half-length along x, radius), 'cylinder' (r, height), 'cup' (bottom r, top r, height),
#   'box' (x, y, z sizes).
ITEMS = [
    ('apple',     'Apple',          'sphere',   (36.0,),              170.0, (0.0, 0.0)),
    ('tomato',    'Tomato',         'sphere',   (29.0,),              100.0, (-140.0, 0.0)),
    ('egg',       'Egg',            'egg',      (29.0, 22.0),          60.0, (140.0, 0.0)),
    ('can',       'Soda can',       'cylinder', (33.0, 115.0),        380.0, (-140.0, 140.0)),
    ('juice',     'Juice box',      'box',      (40.0, 63.0, 103.0),  220.0, (0.0, 140.0)),
    ('spice',     'Spice jar',      'cylinder', (22.0, 95.0),         120.0, (140.0, 140.0)),
    ('yogurt',    'Yogurt cup',     'cup',      (27.0, 36.0, 72.0),   150.0, (-140.0, -140.0)),
    ('tea',       'Tea box',        'box',      (65.0, 65.0, 75.0),    90.0, (0.0, -140.0)),
    ('butter',    'Butter',         'box',      (110.0, 62.0, 40.0),  250.0, (140.0, -140.0)),
]

_DEFAULT_PARAMS = {
    # -- the fingers --
    'fingers': 4,                # how many, evenly spaced around the end-effector's vertical axis
    'finger_radius': 45.0,       # [mm] from the axis to each finger's inner face (its base layer)
    'finger_azimuth': 45.0,      # [deg] where finger 1 hangs (0: +x, 90: +y, the back in the default view)
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
    'floor_size': 600.0,         # [mm] the floor (it stops the items and the fingers)
    # -- joints --
    'joint_stiffness': 0.34,     # [N m/rad] torsional stiffness of each hinge
    'joint_damping': 0.01,       # [s] stiffness-proportional damping of the joints
    'joint_open_limit': 90.0,    # [deg] largest bend of a hinge (a hard stop beyond it)
    'joint_armature': 2e-6,      # [kg m^2] added inertia of each hinge (MuJoCo: keeps the light segments stable)
    'pressure_torque_gain': 1.0, # 1: the torque of the pressure on the cavity's cross-section (see the docstring)
    'flange_acceleration': 0.5,       # [m/s2] largest acceleration of the end-effector
    'flange_speed': 0.25,             # [m/s] its largest speed
    'flange_mass': 0.5,               # [kg] the end-effector (it follows the target on the wrist spring)
    'wrist_stiffness': 3000.0,        # [N/m] the spring between the target and the end-effector (critically damped)
    # -- the items on the floor, and the contacts --
    'item_apple': True, 'item_tomato': True, 'item_egg': True, 'item_can': True, 'item_juice': True,
    'item_spice': True, 'item_yogurt': True, 'item_tea': True, 'item_butter': True,
    'item_mass_scale': 1.0,      # times each item's mass (see ITEMS)
    'item_spacing': 1.0,         # times the distances between the items (see ITEMS)
    'item_yaw': 45.0,            # [deg] the items are turned by this about the vertical (to face the fingers)
    'friction': 0.6,             # Coulomb friction coefficient of every contact
    'torsional_friction': 0.005, # [m] resistance to twisting about a contact's normal
    'impratio': 10.0,            # MuJoCo: ratio of frictional to normal constraint impedance (higher: less slip)
    'finger_collisions': True,   # the fingers touch each other (in the SOFA scene they don't)
    'item_alpha': 1.0,
    # -- a large open box fixed to the floor, to put items in --
    'bin_enabled': True,
    'bin_x': 0.0, 'bin_y': 290.0,   # [mm] its centre
    'bin_length': 220.0,            # [mm] inside, along x
    'bin_width': 130.0,             # [mm] inside, along y
    'bin_height': 70.0,             # [mm] its walls
    'bin_wall': 8.0,                # [mm] their thickness
    'bin_alpha': 1.0,
    # -- material --
    'density': 1000.0,           # [kg/m^3] of the silicone (the segment's mass: the walls' volume)
    'gravity_enabled': True,
    # -- pressure --
    'ramp_start': 0.0,           # [s] when the pressure starts to rise
    'ramp_time': 0.5,            # [s]
    'dt': 0.02,                  # [s] one frame step (many MuJoCo steps)
    'timestep': 0.001,           # [s] MuJoCo's time step
    'instant': False,
    # -- visuals --
    'wall_alpha': 1.0,
    'joint_alpha': 1.0,
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
    """Opens params_editor.py in its own process. Skipped when GRIPPER4MJ_NO_EDITOR is set (headless runs)."""
    if os.environ.get('GRIPPER4MJ_NO_EDITOR'):
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
FLANGE_START = (0.0, 0.0, 190.0)   # the flange's x, y and the height of its underside above the floor [mm] at the start

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
    x, y, z = dims
    return z / 2, [(y * y + z * z) / 12.0, (x * x + z * z) / 12.0, (x * x + y * y) / 12.0]


# The items' colours, and their details (visual only): [(points, triangles, colour)] in the item's frame (mm).
def _item_details(key, dims):
    mm = MM
    if key == 'apple':
        r = dims[0]
        return [(*_cylinder_along([0, 0, r * 0.85], [0.02 * r, 0, r * 1.18], 1.6 * mm, n=10), [0.36, 0.22, 0.1]),
                (*_ellipsoid((0.16 * r, 0, r * 1.06), (0.2 * r, 0.09 * r, 0.025 * r), 6, 12), [0.25, 0.6, 0.18])]
    if key == 'tomato':
        r = dims[0]
        return [(*_frustum_z(0, 0, r * 0.93, r * 1.02, 0.42 * r, 0.12 * r, n=10), [0.2, 0.55, 0.15]),
                (*_cylinder_along([0, 0, r * 1.0], [0, 0, r * 1.15], 1.2 * mm, n=8), [0.2, 0.5, 0.15])]
    if key == 'can':
        r, h = dims
        return [(*_frustum_z(0, 0, h / 2 - 0.1 * mm, h / 2 + 0.6 * mm, r * 0.96, r * 0.86, n=40), [0.78, 0.8, 0.82]),
                (*_frustum_z(0, 0, -h / 2 - 0.1 * mm, -h / 2 + 4 * mm, r * 0.86, r * 1.005, n=40), [0.78, 0.8, 0.82]),
                (*_frustum_z(0, 0, -h * 0.15, h * 0.15, r * 1.004, r * 1.004, n=40, caps=(False, False)), [0.95, 0.95, 0.95])]
    if key == 'juice':
        x, y, z = dims
        return [(*_cylinder_along([x * 0.25, y * 0.25, z / 2], [x * 0.25, y * 0.25, z / 2 + 30 * mm], 2.2 * mm, n=10),
                 [0.95, 0.95, 0.95]),
                (*_box(-x / 2 - 0.2 * mm, -y * 0.3, -z * 0.2, -x / 2, y * 0.3, z * 0.2), [0.98, 0.85, 0.2])]
    if key == 'spice':
        r, h = dims
        return [(*_frustum_z(0, 0, h / 2 - 18 * mm, h / 2 + 0.5 * mm, r * 1.04, r * 1.04, n=40), [0.75, 0.12, 0.1]),
                (*_frustum_z(0, 0, -h * 0.25, h * 0.1, r * 1.004, r * 1.004, n=40, caps=(False, False)), [0.95, 0.92, 0.8])]
    if key == 'yogurt':
        r0, r1, h = dims
        return [(*_frustum_z(0, 0, h / 2, h / 2 + 0.8 * mm, r1 * 1.04, r1 * 1.04, n=40), [0.2, 0.42, 0.85])]
    if key == 'tea':
        x, y, z = dims
        return [(*_box(-x / 2 - 0.2 * mm, -y * 0.35, -z * 0.3, -x / 2, y * 0.35, z * 0.3), [0.95, 0.9, 0.7]),
                (*_box(-x * 0.35, -y / 2 - 0.2 * mm, -z * 0.3, x * 0.35, -y / 2, z * 0.3), [0.95, 0.9, 0.7])]
    if key == 'butter':
        x, y, z = dims
        return [(*_box(-x * 0.3, -y / 2 - 0.2 * mm, -z * 0.3, x * 0.3, -y / 2, z * 0.3), [0.2, 0.35, 0.75]),
                (*_box(-x * 0.3, -y * 0.3, z / 2, x * 0.3, y * 0.3, z / 2 + 0.2 * mm), [0.2, 0.35, 0.75])]
    return []


ITEM_COLOURS = {'apple': [0.8, 0.1, 0.1], 'tomato': [0.92, 0.25, 0.12], 'egg': [0.96, 0.9, 0.78],
                'can': [0.8, 0.08, 0.12], 'juice': [0.98, 0.55, 0.1], 'spice': [0.5, 0.28, 0.12],
                'yogurt': [0.97, 0.97, 0.95], 'tea': [0.2, 0.55, 0.32], 'butter': [0.98, 0.86, 0.35]}


# ----------------------------------------------------------------------
# Controllers: Python run before every MuJoCo step
# ----------------------------------------------------------------------
class FlangeController:
    """The end-effector's target: moves towards a goal (`targetX`, `targetY`, `targetZ`, metres from where it starts;
    the interface's sliders set them) with a limited `acceleration` and `speed`. The wrist actuators pull the
    end-effector towards it, ahead by the damping's drag (velocity x damping / stiffness), as in the SOFA scene."""

    def __init__(self, actuators, lead, acceleration=0.5, speed=0.25):
        self.actuators, self.lead = actuators, lead
        self.targetX = self.targetY = self.targetZ = 0.0
        self.acceleration, self.speed = acceleration, speed
        self.offset = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]

    def __call__(self, scene, h):
        accel = max(float(self.acceleration), 1e-4)
        top = max(float(self.speed), 1e-4)
        goals = (float(self.targetX), float(self.targetY), float(self.targetZ))
        for a in range(3):
            d = goals[a] - self.offset[a]
            wanted = math.copysign(min(top, math.sqrt(2.0 * accel * abs(d))), d)
            dv = max(-accel * h, min(accel * h, wanted - self.velocity[a]))
            self.velocity[a] += dv
            self.offset[a] += self.velocity[a] * h
            if abs(d) < 1e-6 and abs(self.velocity[a]) <= 2 * accel * h:  # arrived
                self.offset[a], self.velocity[a] = goals[a], 0.0
            scene.data.ctrl[self.actuators[a]] = self.offset[a] + self.lead * self.velocity[a]


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
    z_palm = FLANGE_START[2] * MM
    extent = radius + H
    g = 9.81 if P['gravity_enabled'] else 0.0

    cav_l, cav_w, cav_h = max(L - 2 * we, 0.0), max(W - 2 * ws, 0.0), max(H - wb - wt, 0.0)
    seg_mass = float(P['density']) * (L * W * H - cav_l * cav_w * cav_h)
    cavity_area = cav_w * cav_h
    cavity_mid = wb + 0.5 * cav_h
    torque_per_pa = float(P['pressure_torque_gain']) * cavity_area * cavity_mid
    print(f"[model] {F} fingers of {N} rigid segments, {seg_mass * 1e3:.2f} g each; pressure torque "
          f"{torque_per_pa * 5e4:.3f} N m per joint at 50 kPa; joint stiffness {P['joint_stiffness']} N m/rad")

    floor_z = 0.0
    thick = 0.6 * H
    R_gauge = 0.6 * H
    z_top = z_palm + thick + 1.12 * R_gauge + 0.6 * H + R_gauge
    spacing = max(float(P['item_spacing']), 0.1)
    half = max(P['floor_size'] * MM, 3 * extent) / 2
    bin_box = None
    if P['bin_enabled']:
        bl, bw = max(float(P['bin_length']), 10.0) * MM / 2, max(float(P['bin_width']), 10.0) * MM / 2
        bx, by = float(P['bin_x']) * MM, float(P['bin_y']) * MM
        bt = max(float(P['bin_wall']), 1.0) * MM
        bin_box = (bx - bl, by - bw, bx + bl, by + bw, max(float(P['bin_height']), 5.0) * MM, bt)
        half = max(half, abs(bx) + bl + bt + 30 * MM, abs(by) + bw + bt + 30 * MM)

    frames = [FingerFrame(k, F, radius, float(P['finger_azimuth']), z_palm) for k in range(F)]
    flange_pos = np.array([0.0, 0.0, z_palm + thick / 2])
    open_lim = math.radians(float(P['joint_open_limit']))
    close_lim = 2.0 * math.atan2(sep / 2, H)  # the roofs touch (bent outwards): the gap is `sep` at the base
    m_flange = max(float(P['flange_mass']), 1e-3)
    k_wrist = max(float(P['wrist_stiffness']), 1.0)
    c_wrist = 2.0 * math.sqrt(k_wrist * m_flange)
    pad, stem = 0.1 * H, 0.3 * H
    plate_r = extent + pad
    mu = max(float(P['friction']), 0.0)
    k_joint = float(P['joint_stiffness'])

    # Collision filters: the fingers (and the plate) are contype 2 / conaffinity 1, everything else 1 / 3, so the
    # fingers touch the items, the floor and the box; with `finger_collisions` they touch each other too.
    finger_bits = 'contype="2" conaffinity="3"' if P['finger_collisions'] else 'contype="2" conaffinity="1"'
    other_bits = 'contype="1" conaffinity="3"'

    def seg_geom(fr, j, origin, name):
        c = np.array(fr.to_world([j * pitch + L / 2, 0.0, H / 2])) - origin
        return (f'<geom name="{name}" type="box" size="{_v([L / 2, W / 2, H / 2])}" pos="{_v(c)}" '
                f'xyaxes="{_v(fr.down)} {_v(fr.tangent)}" mass="{seg_mass:.9g}" class="finger"/>')

    assets, world, actuators = [], [], []
    world.append(f'<geom name="floor" type="plane" size="{_v([3.5 * half, 3.5 * half, 0.05])}" pos="0 0 {floor_z}" '
                 f'class="static"/>')
    if bin_box:
        x0, y0, x1, y1, bh, bt = bin_box
        for name, (a0, b0, a1, b1) in (('front', (x0 - bt, y0 - bt, x1 + bt, y0)), ('back', (x0 - bt, y1, x1 + bt, y1 + bt)),
                                       ('left', (x0 - bt, y0, x0, y1)), ('right', (x1, y0, x1 + bt, y1))):
            world.append(f'<geom name="bin_{name}" type="box" size="{_v([(a1 - a0) / 2, (b1 - b0) / 2, bh / 2])}" '
                         f'pos="{_v([(a0 + a1) / 2, (b0 + b1) / 2, floor_z + bh / 2])}" class="static"/>')

    # The end-effector and the fingers: nested bodies, each segment's origin at its hinge (at rest every body is
    # aligned with the world).
    hinge_names = []
    flange = [f'<body name="flange" pos="{_v(flange_pos)}" gravcomp="1">',
              '<joint name="flange_x" type="slide" axis="1 0 0"/>', '<joint name="flange_y" type="slide" axis="0 1 0"/>',
              '<joint name="flange_z" type="slide" axis="0 0 1"/>',
              f'<geom name="plate" type="cylinder" size="{plate_r:.9g} {thick / 2:.9g}" mass="{m_flange:.9g}" class="finger"/>']
    pivots = {}
    for fr in frames:
        flange.append(seg_geom(fr, 0, flange_pos, f'f{fr.k + 1}s0'))
        parent = flange_pos
        closing = []
        for j in range(1, N):
            pv = np.array(fr.to_world([j * pitch - sep / 2, 0.0, 0.0]))
            pivots[(fr.k, j)] = pv
            name = f'f{fr.k + 1}j{j}'
            hinge_names.append(name)
            flange.append(f'<body name="f{fr.k + 1}s{j}" pos="{_v(pv - parent)}">')
            flange.append(f'<joint name="{name}" type="hinge" axis="{_v(fr.tangent)}" range="{-close_lim:.9g} {open_lim:.9g}" '
                          f'stiffness="{k_joint:.9g}" damping="{float(P["joint_damping"]) * k_joint:.9g}" '
                          f'armature="{float(P["joint_armature"]):.9g}"/>')
            flange.append(seg_geom(fr, j, pv, f'f{fr.k + 1}s{j}'))
            closing.append('</body>')
            parent = pv
        flange += closing
    flange.append('</body>')
    world += flange
    for axis in 'xyz':
        actuators.append(f'<position name="wrist_{axis}" joint="flange_{axis}" kp="{k_wrist:.9g}" kv="{c_wrist:.9g}" '
                         f'ctrlrange="-1 1"/>')

    # The grocery items: free bodies.
    scale = max(float(P['item_mass_scale']), 1e-3)
    yaw = math.radians(float(P['item_yaw']))
    quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    items = []
    for key, label, shape, dims_mm, mass_g, (x_mm, y_mm) in ITEMS:
        if not P.get(f'item_{key}', True):
            continue
        dims = [d * MM for d in dims_mm]
        mass = mass_g * 1e-3 * scale
        half_h, _ = _item_extent(shape, dims)
        centre = [x_mm * MM * spacing, y_mm * MM * spacing, floor_z + half_h + 0.2 * MM]
        if shape == 'sphere':
            geom = f'type="sphere" size="{dims[0]:.9g}"'
        elif shape == 'egg':
            geom = f'type="ellipsoid" size="{_v([dims[0], dims[1], dims[1]])}"'
        elif shape == 'cylinder':
            geom = f'type="cylinder" size="{dims[0]:.9g} {dims[1] / 2:.9g}"'
        elif shape == 'box':
            geom = f'type="box" size="{_v([d / 2 for d in dims])}"'
        else:  # the cup: a convex mesh
            pts, _, _ = _item_meshes(shape, dims, smooth=False)[0]
            assets.append(f'<mesh name="{key}_mesh" vertex="{_v(np.asarray(pts).ravel())}"/>')
            geom = f'type="mesh" mesh="{key}_mesh"'
        items.append((key, label, shape, dims, mass, centre))
        world.append(f'<body name="item_{key}" pos="{_v(centre)}" quat="{_v(quat)}"><freejoint name="item_{key}"/>'
                     f'<geom name="item_{key}" {geom} mass="{mass:.9g}" class="item"/></body>')

    timestep = max(float(P['timestep']), 1e-5)
    xml = f"""<mujoco model="gripper4">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep:.9g}" gravity="0 0 {-g:.9g}" cone="elliptic" impratio="{float(P['impratio']):.9g}"
          integrator="implicitfast"/>
  <default>
    <geom friction="{mu:.9g} {float(P['torsional_friction']):.9g} 0.0001" condim="4" solref="{max(4 * timestep, 0.002):.9g} 1"/>
    <default class="finger"><geom {finger_bits}/></default>
    <default class="item"><geom {other_bits}/></default>
    <default class="static"><geom {other_bits}/></default>
  </default>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    {chr(10).join(world)}
  </worldbody>
  <actuator>{''.join(actuators)}</actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    scene = MjScene(model, dt=float(P['dt']))
    scene.xml = xml
    body = scene.body_id
    flange_id = body('flange')

    def body_of(k, j):
        return flange_id if j == 0 else body(f'f{k + 1}s{j}')

    def origin_of(k, j):
        return flange_pos if j == 0 else pivots[(k, j)]

    # ---- what the browser draws (as the SOFA scene's visual models, with the same names)
    scene.background = [0.78, 0.87, 0.96, 1.0]
    scene.ambient = [0.35, 0.35, 0.35, 1.0]
    scene.add_light(direction=[-0.5, -0.15, 1.0])
    scene.add_light(color=[0.35, 0.35, 0.35, 1.0], direction=[0.5, 0.2, 0.3])
    cam_lookat = [0.0, 0.3 * (bin_box[1] + bin_box[3]) / 2 if bin_box else 0.0, 0.35 * z_top]
    cam_dir = (0.25, -1.0, 0.75)
    norm = math.sqrt(sum(c * c for c in cam_dir))
    cam_distance = max(1.25 * z_top, 1.05 * (140 * MM * spacing + 60 * MM)) / math.tan(math.radians(22.5))
    cam_position = [l + cam_distance * c / norm for l, c in zip(cam_lookat, cam_dir)]
    scene.set_camera(cam_position, _look_at_quaternion(cam_position, cam_lookat), cam_distance)

    floor_pts = [[-half, -half, floor_z - 1e-4], [half, -half, floor_z - 1e-4], [half, half, floor_z - 1e-4],
                 [-half, half, floor_z - 1e-4]]
    scene.add_visual('/floor/visu/visual', floor_pts, [[0, 1, 2], [0, 2, 3]], 0, [0.08, 0.28, 0.1, 1.0])
    if bin_box:
        x0, y0, x1, y1, bh, bt = bin_box
        walls = [_box(x0 - bt, y0 - bt, floor_z, x1 + bt, y0, floor_z + bh, skip=('-Z',)),
                 _box(x0 - bt, y1, floor_z, x1 + bt, y1 + bt, floor_z + bh, skip=('-Z',)),
                 _box(x0 - bt, y0, floor_z, x0, y1, floor_z + bh, skip=('-Z', '-Y', '+Y')),
                 _box(x1, y0, floor_z, x1 + bt, y1, floor_z + bh, skip=('-Z', '-Y', '+Y'))]
        alpha = float(P['bin_alpha'])
        scene.add_visual('/bin/visu/visual', *_flat_shaded(*_merge(walls)), 0, [0.74, 0.56, 0.36, alpha])
        scene.add_visual('/bin/bottom/visual',
                         *_flat_shaded(*_box(x0, y0, floor_z - 1e-4, x1, y1, floor_z + 0.3 * MM, skip=('-Z',))), 0,
                         [0.62, 0.45, 0.28, alpha])

    arrow_ref = 0.6 * max(finger_len, W, H)
    head_len, shaft_half = arrow_ref * 0.2, arrow_ref * 0.015
    total = max(arrow_ref, extent + 0.3 * arrow_ref + head_len)
    az = 1e-4
    for name, axis, color in (('arrow_x', 0, [0.9, 0.1, 0.1, 1.0]), ('arrow_y', 1, [0.1, 0.8, 0.1, 1.0])):
        pts, tris = _axis_arrow(axis, total - head_len, head_len, shaft_half, shaft_half * 3)
        scene.add_visual(f'/{name}/visual', *_flat_shaded([[p[0], p[1], p[2] + az] for p in pts], tris), 0, color)
    lh, lgap = arrow_ref * 0.09, arrow_ref * 0.08
    for name, strokes, color in (
            ('label_x', [(total + lgap, -lh, total + lgap + 2 * lh, lh), (total + lgap, lh, total + lgap + 2 * lh, -lh)],
             [0.9, 0.1, 0.1, 1.0]),
            ('label_y', [(-lh, total + lgap + 2 * lh, 0.0, total + lgap + lh), (lh, total + lgap + 2 * lh, 0.0, total + lgap + lh),
                         (0.0, total + lgap + lh, 0.0, total + lgap)], [0.1, 0.8, 0.1, 1.0])):
        scene.add_visual(f'/{name}/visual', *_flat_shaded(*_letter(strokes, lh * 0.125, az, az + shaft_half * 2)), 0, color)

    def carried(parts):  # [(world points at rest, triangles, body id, body origin)] -> one mesh with a body per point
        pts, tris, bodies = [], [], []
        for p, t, b, o in parts:
            off = len(pts)
            pts += [list(np.array(q) - o) for q in p]
            tris += [[i + off for i in tri] for tri in t]
            bodies += [b] * len(p)
        return pts, tris, bodies

    seg_parts, sheet_parts, mark_parts = [], [], []
    for fr in frames:
        for j in range(N):
            x0 = j * pitch
            pts, tris = _box(x0, -W / 2, 0.0, x0 + L, W / 2, H)
            seg_parts.append(([fr.to_world(p) for p in pts], tris, body_of(fr.k, j), origin_of(fr.k, j)))
            sx0, sx1 = x0 - (sep / 2 if j > 0 else 0.0), x0 + L + (sep / 2 if j < N - 1 else 0.0)
            sp, st = _box(sx0, -W / 2 * 0.98, -0.4 * MM, sx1, W / 2 * 0.98, 0.0)
            sheet_parts.append(([fr.to_world(p) for p in sp], st, body_of(fr.k, j), origin_of(fr.k, j)))
            if j > 0:
                pv = pivots[(fr.k, j)]
                half_w = W / 2 + 0.6 * MM
                mp, mt = _cylinder_along(list(pv - half_w * fr.tangent), list(pv + half_w * fr.tangent), 1.2 * MM, n=16)
                mark_parts.append((mp, mt, body_of(fr.k, j), origin_of(fr.k, j)))
    pts, tris, bodies = carried(seg_parts)
    scene.add_visual('/world/angles/bodies/visu/visual', pts, tris, bodies, [0.15, 0.45, 0.95, float(P['wall_alpha'])])
    pts, tris, bodies = carried(sheet_parts)
    scene.add_visual('/world/angles/bodies/base_layer/visual', pts, tris, bodies,
                     [0.96, 0.94, 0.86, float(P['wall_alpha'])])
    if P['show_joint_marks'] and mark_parts:
        pts, tris, bodies = carried(mark_parts)
        scene.add_visual('/world/angles/bodies/joint_marks/visual', pts, tris, bodies,
                         [0.95, 0.6, 0.1, float(P['joint_alpha'])])
    plate_pts, plate_tris = _frustum_z(0.0, 0.0, z_palm, z_palm + thick, plate_r, plate_r, n=max(24, 4 * F))
    mount_pts, mount_tris = _flat_shaded(*_merge([(plate_pts, plate_tris),
                                                  _box(-stem, -stem, z_palm + thick, stem, stem, z_top + 0.2 * H)]))
    scene.add_visual('/world/angles/bodies/trunk/visual', [list(np.array(p) - flange_pos) for p in mount_pts], mount_tris,
                     flange_id, [0.4, 0.26, 0.13, 1.0])

    for key, label, shape, dims, mass, centre in items:
        parts = [(p, t, ITEM_COLOURS[key]) for p, t, _ in _item_meshes(shape, dims, smooth=True)] + _item_details(key, dims)
        for n_part, (vp, vt, colour) in enumerate(parts):
            vp, vt = (vp, vt) if shape in ('sphere', 'egg', 'cylinder', 'cup') and n_part == 0 else _flat_shaded(vp, vt)
            scene.add_visual(f'/world/items/{key}/{"visu" if n_part == 0 else f"detail{n_part}"}/visual', vp, vt,
                             body(f'item_{key}'), list(colour) + [float(P['item_alpha'])])

    # ---- controllers
    jnt = lambda name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    act = lambda name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
    hinge_dofs = [int(model.jnt_dofadr[jnt(n)]) for n in hinge_names]
    flange_ctl = FlangeController([act(f'wrist_{a}') for a in 'xyz'], c_wrist / k_wrist,
                                  acceleration=float(P['flange_acceleration']), speed=float(P['flange_speed']))
    pressure = PressureRamp(hinge_dofs, torque_per_pa, 0.0, float(P['ramp_time']), instant=bool(P['instant']),
                            ramp_start=float(P['ramp_start']))
    scene.controllers += [flange_ctl, pressure]
    scene.objects.update({'Flange': flange_ctl, 'PressureRamp': pressure})

    BUILD.clear()
    BUILD.update({
        'segments': N, 'fingers': F, 'flange_id': flange_id, 'flange_rest0': list(flange_pos),
        'flange_rest': [(min(p[q] for p in mount_pts) + max(p[q] for p in mount_pts)) / 2 for q in range(3)],
        'hinge_qpos': [int(model.jnt_qposadr[jnt(n)]) for n in hinge_names],
        'items': [{'key': key, 'label': label, 'rest': centre, 'mass': mass, 'body': body(f'item_{key}')}
                  for key, label, shape, dims, mass, centre in items],
        'cavity_mid': cavity_mid, 'torque_per_pa': torque_per_pa, 'close_limit': close_lim, 'seg_mass': seg_mass,
        'plate_radius': plate_r, 'stem': stem, 'bin': list(bin_box) if bin_box else None,
    })
    return scene


def main():
    os.environ.setdefault('GRIPPER4MJ_NO_EDITOR', '1')
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
