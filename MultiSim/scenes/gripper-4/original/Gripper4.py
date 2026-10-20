# -*- coding: utf-8 -*-
"""
Gripper 4: the rigid-segment fingers of Gripper 3 on a gripper that moves in X, Y and Z over a floor with
grocery items and boxes to pick up -- an apple, a tomato, an egg, a soda can, a juice box, a yogurt cup, a
spice jar, a tea box and a block of butter. The number of fingers is a parameter: they hang from the end-
effector evenly spaced around its vertical axis, their base layers facing the middle, and curl inwards.

Fingers. A PneuNet finger of `segments` chambers is `segments` rigid bodies, one per chamber, in a row along
the finger with a `channel_separation` gap between neighbours; the first is welded to the end-effector. The
chambers are only joined by the thin base layer, so each gap is a hinge at the base, its axis across the finger
(horizontal, tangential to the circle of fingers). Unlike Gripper 3 (where the joints are springs between
free bodies), the hinge angles themselves are the fingers' degrees of freedom: an ArticulatedSystemMapping
(SOFA's ArticulatedSystemPlugin) places the segments from the angles and the end-effector's pose, so the
fingers can hang at any angle around the end-effector and their joints never come apart. On each angle:
  - a torsional spring of `joint_stiffness` [N m/rad] (the silicone walls and base resisting the bend), with
    stiffness-proportional damping `joint_damping` [s];
  - hard stops (`joint_stop_stiffness`): closing (the roofs of two neighbouring chambers touching, the gap
    being `channel_separation` wide at the base and 0 at the roof) and over-opening (`joint_open_limit`);
  - the pressure: p times the cavity's cross-section is a force on an end wall at the cavity's mid height above
    the base, a torque p A z (times `pressure_torque_gain`) on the hinge, as in Gripper 3;
  - an inertia: that of the segments beyond it about the hinge (averaged over the hinges). The segments'
    weight acts on them (mapped), so the fingers sag under gravity; moving the end-effector does not swing them.

Contact. The fingers' segments (boxes: their faces and corners), the end-effector's plate, the items and the floor
are SOFA collision models (MinProximityIntersection); contacts are Lagrangian constraints with Coulomb friction
(`friction`, the same for every pair), solved each step by a Gauss-Seidel constraint solver
(FreeMotionAnimationLoop), so items can be held by friction as well as by the fingers curling under them, and they
stand, slide, roll and tip over on the floor and against each other. The fingers don't touch each other
(meeting head-on, two fingers' faces would coincide: a contact without a direction); instead each finger has an
invisible guard, a vertical plane near the axis that it can't cross, placed where the neighbours' widths would meet,
so fingers closing on nothing stop short of each other. The spheres (apple, tomato; the egg is
three) are exact; the other items are coarse meshes (a 10-sided cylinder or cup). An item falling fast enough to
pass through the floor in one step is put back on it (FloorGuard).

The end-effector (`flange_mass`) hangs on a stiff, critically damped spring (`wrist_stiffness`) from a target that
accelerates smoothly towards the x / y / z set by the sliders (`flange_acceleration`, `flange_speed`), as in Grippers
2 and 3. The spring lets it give way when it is pushed onto the floor or an item (a few millimetres for a few
newtons), and it moves with a velocity, so that friction carries what it holds along.

Units: the scene is SI (metres, kg, s, Pa); params.json takes millimetres for lengths, as labelled.


Run with the GUI:   runSofa -l SofaPython3 -g glfw Gripper4.py
Headless:           python Gripper4.py
Parameters: params.json next to this script (edit by hand, or with params_editor.py).
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import Sofa
import Sofa.Core

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
    'joint_stop_stiffness': 5.0, # [N m/rad] the hard stops
    'pressure_torque_gain': 1.0, # 1: the torque of the pressure on the cavity's cross-section (see the docstring)
    'flange_acceleration': 0.06,      # [m/s2] largest acceleration of the end-effector
    'flange_speed': 0.15,             # [m/s] its largest speed
    'flange_mass': 0.5,               # [kg] the end-effector (it follows the target on the wrist spring)
    'wrist_stiffness': 3000.0,        # [N/m] the spring between the target and the end-effector (critically damped)
    # -- the items on the floor, and the contacts --
    'item_apple': True, 'item_tomato': True, 'item_egg': True, 'item_can': True, 'item_juice': True,
    'item_spice': True, 'item_yogurt': True, 'item_tea': True, 'item_butter': True,
    'item_mass_scale': 1.0,      # times each item's mass (see ITEMS)
    'item_spacing': 1.0,         # times the distances between the items (see ITEMS)
    'item_yaw': 45.0,            # [deg] the items are turned by this about the vertical (to face the fingers)
    'friction': 0.6,             # Coulomb friction coefficient of every contact
    'contact_distance': 0.5,     # [mm] contacts are kept this far apart
    'alarm_distance': 5.0,       # [mm] pairs closer than this are watched for contact
    'item_alpha': 1.0,
    # -- material --
    'density': 1000.0,           # [kg/m^3] of the silicone (the segment's mass: the walls' volume)
    'gravity_enabled': True,
    # -- pressure --
    'ramp_start': 0.0,           # [s] when the pressure starts to rise
    'ramp_time': 0.5,            # [s]
    'dt': 0.02,                  # [s] simulation time step
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
    """Opens params_editor.py in its own process (not in runSofa's embedded
    interpreter). Skipped when GRIPPER4_NO_EDITOR is set (headless runs)."""
    if os.environ.get('GRIPPER4_NO_EDITOR'):
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


# The fingers' collision models: their faces and corners (their edges too would add many edge-edge contacts per item held,
# several times slower, for little difference: an item's own corners and edges still touch the fingers' faces)
FINGER_MODELS = ('TriangleCollisionModel', 'PointCollisionModel')
FINGERS_GROUP, ITEMS_GROUP = 100, 200  # collision groups (see createScene); item i has ITEMS_GROUP + i
IDENTITY = [0.0, 0.0, 0.0, 1.0]  # every body's orientation at rest (the geometry is given in world coordinates and mapped)


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


class FlangeController(Sofa.Core.Controller):
    """The end-effector's target: moves towards a goal (`targetX`, `targetY`, `targetZ`, metres from where it
    starts; the interface's sliders set them) with a limited acceleration and speed. The end-effector follows it on
    the wrist spring (the target is the spring's rest position), carrying the fingers."""

    def __init__(self, root, flange_dofs, rest, lead, acceleration=0.06, speed=0.15, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.root, self.dofs, self.rest, self.lead = root, flange_dofs, np.array(rest, dtype=float), lead
        for axis in 'XYZ':
            self.addData(name=f'target{axis}', type='float', value=0.0,
                         help=f'Flange {axis.lower()} relative to the start [m]. Live-editable.')
        self.addData(name='acceleration', type='float', value=acceleration,
                     help='Largest acceleration of the flange [m/s2]. Live-editable.')
        self.addData(name='speed', type='float', value=speed, help='Largest speed of the flange [m/s]. Live-editable.')
        self.offset = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]

    def onAnimateBeginEvent(self, event):
        dt = float(self.root.dt.value)
        accel = max(float(self.acceleration.value), 1e-4)
        top = max(float(self.speed.value), 1e-4)
        goals = (float(self.targetX.value), float(self.targetY.value), float(self.targetZ.value))
        for a in range(3):
            d = goals[a] - self.offset[a]
            # The speed we may have here and still be able to stop at the goal (braking at `accel`), at most `top`:
            wanted = math.copysign(min(top, math.sqrt(2.0 * accel * abs(d))), d)
            dv = max(-accel * dt, min(accel * dt, wanted - self.velocity[a]))
            self.velocity[a] += dv
            self.offset[a] += self.velocity[a] * dt
            if abs(d) < 1e-6 and abs(self.velocity[a]) <= 2 * accel * dt:  # arrived
                self.offset[a], self.velocity[a] = goals[a], 0.0
        # the wrist spring's anchor, ahead by the damping's drag (velocity x damping / stiffness)
        with self.dofs.rest_position.writeable() as pos:
            for a in range(3):
                pos[0][a] = self.rest[a] + self.offset[a] + self.lead * self.velocity[a]


class FloorGuard(Sofa.Core.Controller):
    """Catches an item that has gone through the floor: one falling fast enough to pass the contact's watch distance
    in one step (dropped from 10 cm it falls about 3 cm per step) is not seen by the floor's contact. After each step,
    an item whose lowest point is under the floor is lifted back onto it, its vertical velocity zeroed."""

    def __init__(self, items, floor_z, contact_distance, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.items, self.floor_z, self.gap = items, floor_z, contact_distance  # [(rigid dofs, collision dofs, radii)]
        self.caught = 0

    def onAnimateEndEvent(self, event):
        for body, points, radii in self.items:
            lowest = float(np.min(np.asarray(points.position.value)[:, 2] - radii))
            if lowest < self.floor_z - 0.5 * self.gap:
                dz = self.floor_z + self.gap - lowest
                with body.position.writeable() as pos:
                    pos[0][2] += dz
                with points.position.writeable() as pos:  # (the next collision detection reads these before they're mapped)
                    pos[:, 2] += dz
                with body.velocity.writeable() as vel:
                    vel[0][2] = 0.0  # (not bouncing off: the contact that pushed it out would launch it)
                self.caught += 1


class PressureRamp(Sofa.Core.Controller):
    """Drives the fingers' pressure towards a live-editable target, ramped or instantly (`targetPressure`,
    `rampTime`, `rampStart`, `instant` are Data fields), and applies it as the torque on every hinge angle
    (`torque_per_pa` x p: the pressure acting on the cavity's cross-section at the cavity's mid height)."""

    def __init__(self, root, force_field, count, torque_per_pa, target_pressure, ramp_time,
                 instant=False, ramp_start=0.0, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.root, self.cff, self.count, self.torque_per_pa = root, force_field, count, torque_per_pa
        self.addData(name='targetPressure', type='float', value=target_pressure,
                     help='Target gauge pressure in the fingers [Pa]. Live-editable.')
        self.addData(name='rampStart', type='float', value=ramp_start,
                     help='Time [s] at which the pressure starts to rise (0 before it). Live-editable.')
        self.addData(name='rampTime', type='float', value=ramp_time,
                     help='Time [s] to go from 0 to targetPressure when not instant.')
        self.addData(name='instant', type='bool', value=instant,
                     help='Apply targetPressure immediately instead of ramping towards it.')
        self.current_pressure = 0.0
        self.time = 0.0

    def onAnimateBeginEvent(self, event):
        dt = self.root.dt.value
        self.time += dt
        target = self.targetPressure.value if self.time >= self.rampStart.value - 1e-9 else 0.0
        if self.instant.value:
            self.current_pressure = target
        else:
            ramp_time = max(self.rampTime.value, 1e-6)
            max_step = max((abs(target) if target != 0 else abs(self.current_pressure)) / ramp_time * dt, 1e-9)
            diff = target - self.current_pressure
            self.current_pressure = target if abs(diff) <= max_step else \
                self.current_pressure + (max_step if diff > 0 else -max_step)
        tau = self.torque_per_pa * self.current_pressure
        self.cff.forces.value = [[tau]] * self.count

    def applied_pressure(self):
        """The pressure acting in the fingers [Pa]."""
        return self.current_pressure


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
# Scene
# ----------------------------------------------------------------------
# Filled in by createScene for the sofaweb probes (and anyone inspecting a built scene): geometry in SOFA units.
BUILD = {}


def createScene(rootNode):
    launch_params_editor()
    P = PARAMS

    plugins = [
        "ArticulatedSystemPlugin",
        "Sofa.Component.AnimationLoop", "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection", "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Collision.Response.Contact", "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Solver", "Sofa.Component.Constraint.Projective",
        "Sofa.Component.LinearSolver.Direct", "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.Mapping.Linear", "Sofa.Component.Mapping.NonLinear", "Sofa.Component.Mass",
        "Sofa.Component.MechanicalLoad", "Sofa.Component.ODESolver.Backward", "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer", "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Setting", "Sofa.Component.Visual", "Sofa.GL.Component.Rendering3D", "Sofa.GL.Component.Shader",
    ]
    for p in plugins:
        rootNode.addObject('RequiredPlugin', name='req_' + p.replace('.', '_'), pluginName=p)

    F = max(1, int(P['fingers']))
    N = max(2, int(P['segments']))
    L, W, H = float(P['length']) * MM, float(P['width']) * MM, float(P['height']) * MM
    sep = max(float(P['channel_separation']), 0.0) * MM
    pitch = L + sep
    finger_len = N * L + (N - 1) * sep
    we, ws, wb, wt = (float(P[k]) * MM for k in ('wall_end', 'wall_side', 'wall_bottom', 'wall_top'))
    radius = max(float(P['finger_radius']), 0.0) * MM
    z_palm = FLANGE_START[2] * MM
    extent = radius + H  # the fingers' outer faces from the axis
    g = 9.81 if P['gravity_enabled'] else 0.0

    # Mass: the walls' volume (the box minus the cavity).
    cav_l, cav_w, cav_h = max(L - 2 * we, 0.0), max(W - 2 * ws, 0.0), max(H - wb - wt, 0.0)
    seg_volume = L * W * H - cav_l * cav_w * cav_h
    seg_mass = float(P['density']) * seg_volume
    seg_inertia = seg_mass * (L * L + H * H) / 12.0
    # Pressure: p x the cavity's cross-section is a force on an end wall, at the cavity's mid height above the base.
    cavity_area = cav_w * cav_h
    cavity_mid = wb + 0.5 * cav_h
    torque_per_pa = float(P['pressure_torque_gain']) * cavity_area * cavity_mid
    # Each hinge's inertia: the segments beyond it about the hinge (finger straight), averaged over the hinges.
    hinge_inertia = []
    for j in range(1, N):
        pivot = j * pitch - sep / 2
        hinge_inertia.append(sum(seg_mass * ((i * pitch + L / 2 - pivot) ** 2 + (H / 2) ** 2) + seg_inertia
                                 for i in range(j, N)))
    angle_inertia = sum(hinge_inertia) / len(hinge_inertia)
    print(f"[model] {F} fingers of {N} rigid segments, {seg_mass * 1e3:.2f} g each; pressure torque "
          f"{torque_per_pa * 5e4:.3f} N m per joint at 50 kPa; joint stiffness {P['joint_stiffness']} N m/rad")

    rootNode.addObject('BackgroundSetting', color=[0.78, 0.87, 0.96, 1.0], image="")
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                                   'hideBoundingCollisionModels hideForceFields hideInteractionForceFields '
                                                   'hideWireframe')

    floor_z = 0.0
    thick = 0.6 * H                     # the end-effector's plate
    R_gauge = 0.6 * H                   # the pressure gauge's dial radius (see sofaweb_scene._gauge)
    z_top = z_palm + thick + 1.12 * R_gauge + 0.6 * H + R_gauge  # the top of the gauge
    spacing = max(float(P['item_spacing']), 0.1)
    half = max(P['floor_size'] * MM, 3 * extent) / 2

    cam_lookat = [0.0, 0.0, 0.35 * z_top]
    cam_dir = (0.25, -1.0, 0.75)
    norm = math.sqrt(sum(c * c for c in cam_dir))
    cam_distance = max(1.25 * z_top, 1.05 * (140 * MM * spacing + 60 * MM)) / math.tan(math.radians(22.5))
    cam_position = [l + cam_distance * c / norm for l, c in zip(cam_lookat, cam_dir)]
    rootNode.addObject('InteractiveCamera', name='camera', position=cam_position, distance=cam_distance,
                       orientation=_look_at_quaternion(cam_position, cam_lookat), activated=True)
    rootNode.addObject('LightManager', ambient=[0.35, 0.35, 0.35, 1.0])
    rootNode.addObject('DirectionalLight', direction=[-0.5, -0.15, 1.0])
    rootNode.addObject('DirectionalLight', color=[0.35, 0.35, 0.35, 1.0], direction=[0.5, 0.2, 0.3])

    # ---- the floor: drawn, and a collision model that stops everything
    floor = rootNode.addChild('floor')
    floor_pts = [[-half, -half, floor_z], [half, -half, floor_z], [half, half, floor_z], [-half, half, floor_z]]
    # The collision floor is a single triangle around the drawn one: an item sitting on an edge between two triangles
    # can fall through (it is on neither), so there is none.
    tri_pts = [[-3.5 * half, -1.5 * half, floor_z], [3.5 * half, -1.5 * half, floor_z], [0.0, 4.0 * half, floor_z]]
    floor.addObject('MeshTopology', name='topo', triangles=[[0, 1, 2]], position=tri_pts)
    floor.addObject('MechanicalObject', name='dofs', template='Vec3', position=tri_pts)
    floor.addObject('TriangleCollisionModel', moving=False, simulated=False)
    floor_visual = floor.addChild('visu')
    floor_visual.addObject('MeshTopology', name='topo', triangles=[[0, 1, 2], [0, 2, 3]],
                           position=[[p[0], p[1], p[2] - 1e-4] for p in floor_pts])
    floor_visual.addObject('OglModel', name='visual', src='@topo', color=[0.08, 0.28, 0.1, 1.0])

    # Axes: purely visual.
    arrow_ref = 0.6 * max(finger_len, W, H)
    head_len, shaft_half = arrow_ref * 0.2, arrow_ref * 0.015
    totals = {0: max(arrow_ref, extent + 0.3 * arrow_ref + head_len), 1: max(arrow_ref, extent + 0.3 * arrow_ref + head_len)}
    az = 1e-4
    for name, axis, color in (('arrow_x', 0, [0.9, 0.1, 0.1, 1.0]), ('arrow_y', 1, [0.1, 0.8, 0.1, 1.0])):
        pts, tris = _axis_arrow(axis, totals[axis] - head_len, head_len, shaft_half, shaft_half * 3)
        pts, tris = _flat_shaded([[p[0], p[1], p[2] + az] for p in pts], tris)
        node = rootNode.addChild(name)
        node.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        node.addObject('OglModel', name='visual', src='@topo', color=color)
    lh = arrow_ref * 0.09
    lgap = arrow_ref * 0.08
    for name, strokes, color in (
        ('label_x', [(totals[0] + lgap, -lh, totals[0] + lgap + 2 * lh, lh),
                     (totals[0] + lgap, lh, totals[0] + lgap + 2 * lh, -lh)], [0.9, 0.1, 0.1, 1.0]),
        ('label_y', [(-lh, totals[1] + lgap + 2 * lh, 0.0, totals[1] + lgap + lh),
                     (lh, totals[1] + lgap + 2 * lh, 0.0, totals[1] + lgap + lh),
                     (0.0, totals[1] + lgap + lh, 0.0, totals[1] + lgap)], [0.1, 0.8, 0.1, 1.0])):
        pts, tris = _flat_shaded(*_letter(strokes, lh * 0.125, az, az + shaft_half * 2))
        node = rootNode.addChild(name)
        node.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        node.addObject('OglModel', name='visual', src='@topo', color=color)

    rootNode.gravity = [0.0, 0.0, -g]
    rootNode.dt = DT
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('BlockGaussSeidelConstraintSolver', name='constraintSolver', maxIterations=40, tolerance=1e-5)
    rootNode.addObject('CollisionPipeline', name='pipeline')
    rootNode.addObject('BruteForceBroadPhase', name='broadPhase')
    rootNode.addObject('BVHNarrowPhase', name='narrowPhase')
    # (LocalMinDistance loses contacts on a mesh's edges and corners: an item on the floor falls through)
    rootNode.addObject('MinProximityIntersection', name='proximity', alarmDistance=float(P['alarm_distance']) * MM,
                       contactDistance=float(P['contact_distance']) * MM, usePointPoint=False, useSurfaceNormals=True,
                       useLinePoint=True)
    rootNode.addObject('CollisionResponse', name='response', response='FrictionContactConstraint',
                       responseParams=f"mu={max(float(P['friction']), 0.0)}")

    world = rootNode.addChild('world')

    # ---- the fingers: the hinge angles are the degrees of freedom (one solver); the end-effector (flange) is moved by
    # the FlangeController; the segments are placed from both by the ArticulatedSystemMapping.
    frames = [FingerFrame(k, F, radius, float(P['finger_azimuth']), z_palm) for k in range(F)]
    n_angles = F * (N - 1)
    angles = world.addChild('angles')
    angles.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=float(P['joint_damping']), rayleighMass=0.0)
    angles.addObject('SparseLDLSolver', name='linearSolver', template='CompressedRowSparseMatrixd')
    angles.addObject('MechanicalObject', name='dofs', template='Vec1', position=[[0.0]] * n_angles,
                     rest_position=[[0.0]] * n_angles)
    angles.addObject('UniformMass', name='mass', vertexMass=angle_inertia)
    angles.addObject('RestShapeSpringsForceField', name='hinges', stiffness=float(P['joint_stiffness']),
                     points=list(range(n_angles)))
    open_lim = math.radians(float(P['joint_open_limit']))
    close_lim = 2.0 * math.atan2(sep / 2, H)  # the roofs touch (bent outwards): the gap is `sep` at the base, 0 at the roof
    kr_stop = float(P['joint_stop_stiffness'])
    angles.addObject('PlaneForceField', name='stopClosed', normal=[1.0], d=-close_lim, stiffness=kr_stop, damping=0.0)
    angles.addObject('PlaneForceField', name='stopOpen', normal=[-1.0], d=-open_lim, stiffness=kr_stop, damping=0.0)
    pressure_cff = angles.addObject('ConstantForceField', name='pressure', indices=list(range(n_angles)),
                                    forces=[[0.0]] * n_angles, showArrowSize=0.0)
    angles.addObject('LinearSolverConstraintCorrection', name='constraintCorrection')
    angles.addObject('ArticulatedHierarchyContainer', name='articulations')

    flange_pos = [0.0, 0.0, z_palm + thick / 2]
    # The end-effector is a mass on a stiff spring (the wrist) that follows the slider target: it moves with a velocity
    # (so the friction of what it holds is carried along) and gives way when pushed onto the floor or an item.
    flange = world.addChild('flange')
    flange.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=0.0, rayleighMass=0.0)
    flange.addObject('CGLinearSolver', name='linearSolver', iterations=25, tolerance=1e-12, threshold=1e-18)
    flange_dofs = flange.addObject('MechanicalObject', name='dofs', template='Rigid3', position=[flange_pos + IDENTITY],
                                   rest_position=[flange_pos + IDENTITY])
    m_flange = max(float(P['flange_mass']), 1e-3)
    k_wrist = max(float(P['wrist_stiffness']), 1.0)
    c_wrist = 2.0 * math.sqrt(k_wrist * m_flange)  # critically damped
    flange.addObject('UniformMass', name='mass', vertexMass=[m_flange, 1.0, 1e-3, 0, 0, 0, 1e-3, 0, 0, 0, 1e-3])
    flange.addObject('PartialFixedProjectiveConstraint', name='noTurning', indices=[0], fixedDirections=[0, 0, 0, 1, 1, 1])
    flange.addObject('RestShapeSpringsForceField', name='wrist', stiffness=k_wrist, angularStiffness=1.0, points=[0])
    flange.addObject('UniformVelocityDampingForceField', name='wristDamping', template='Rigid3', dampingCoefficient=c_wrist,
                     implicit=True)
    flange.addObject('ConstantForceField', name='holdUp', indices=[0], forces=[[0.0, 0.0, m_flange * g, 0.0, 0.0, 0.0]],
                     showArrowSize=0.0)  # (its own weight: carried by the arm, not the wrist spring)
    flange.addObject('UncoupledConstraintCorrection', name='constraintCorrection')

    positions = [flange_pos + IDENTITY]
    body_of = {}
    for fr in frames:
        body_of[(fr.k, 0)] = 0  # the first segment is welded to the end-effector
        for j in range(1, N):
            body_of[(fr.k, j)] = len(positions)
            positions.append(fr.to_world([j * pitch + L / 2, 0.0, H / 2]) + IDENTITY)

    def centre_of(fr, j):
        return flange_pos if j == 0 else fr.to_world([j * pitch + L / 2, 0.0, H / 2])

    centres = angles.addChild('articulationCenters')
    pivots = []  # for the visual marks: (finger, joint, world pivot)
    for fr in frames:
        for j in range(1, N):
            pv = fr.to_world([j * pitch - sep / 2, 0.0, 0.0])  # at the base, mid-gap
            pivots.append((fr.k, j, pv))
            parent, child = centre_of(fr, j - 1), centre_of(fr, j)
            node = centres.addChild(f'finger{fr.k + 1}_joint{j}')
            node.addObject('ArticulationCenter', parentIndex=body_of[(fr.k, j - 1)], childIndex=body_of[(fr.k, j)],
                           posOnParent=[pv[q] - parent[q] for q in range(3)],
                           posOnChild=[pv[q] - child[q] for q in range(3)], articulationProcess=0)
            # (the center finds its articulations in its child nodes)
            node.addChild('articulation').addObject('Articulation', translation=False, rotation=True, rotationAxis=list(fr.tangent),
                           articulationIndex=fr.k * (N - 1) + (j - 1))

    bodies = angles.addChild('bodies')
    flange.addChild(bodies)
    body_dofs = bodies.addObject('MechanicalObject', name='dofs', template='Rigid3', position=positions)
    bodies.addObject('ArticulatedSystemMapping', name='mapping', input1=angles.dofs.getLinkPath(),
                     input2=flange_dofs.getLinkPath(), output=body_dofs.getLinkPath())
    # the segments' weight (the first ones are carried by the end-effector)
    moving = list(range(1, len(positions)))
    bodies.addObject('ConstantForceField', name='weight', indices=moving,
                     forces=[[0.0, 0.0, -seg_mass * g, 0.0, 0.0, 0.0]] * len(moving), showArrowSize=0.0)

    # ---- collision models: each finger (its segments' boxes), the plate, and a guard per finger. Collision groups (two
    # models sharing one don't touch): each finger its own (k + 1) and FINGERS_GROUP (the fingers don't touch each other:
    # meeting head-on, two fingers' faces coincide, and a contact at zero distance has no direction); each item its own.
    for fr in frames:
        pts, tris, idx = [], [], []
        for j in range(N):
            bp, bt = _box(j * pitch, -W / 2, 0.0, j * pitch + L, W / 2, H)
            off = len(pts)
            pts += [fr.to_world(p) for p in bp]
            tris += [[i + off for i in t] for t in bt]
            idx += [body_of[(fr.k, j)]] * len(bp)
        node = bodies.addChild(f'finger{fr.k + 1}')
        node.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        node.addObject('MechanicalObject', name='dofs', template='Vec3', position=pts)
        for model in FINGER_MODELS:
            node.addObject(model, group=[fr.k + 1, FINGERS_GROUP])
        node.addObject('RigidMapping', name='mapping', globalToLocalCoords=True, rigidIndexPerPoint=idx)
    if F >= 2:
        # The guards: a vertical plane in front of each finger, near the axis, that only it touches (it is in every other
        # finger's group and the items'), so the fingers closing on nothing stop short of each other instead of
        # passing through: at (W/2) / tan(180 deg / F) from the axis, where the neighbours' widths would meet.
        d_guard = 0.5 * W / math.tan(math.pi / F) if F > 2 else 0.0
        reach = extent + W
        for fr in frames:
            corners = []
            for t, z in ((-reach, z_palm), (reach, z_palm), (reach, z_palm - finger_len - 2 * H), (-reach, z_palm - finger_len - 2 * H)):
                c = d_guard * fr.radial + t * fr.tangent
                corners.append([float(c[0]), float(c[1]), z])
            a, b, c = (np.array(corners[i]) for i in range(3))
            tris = [[0, 1, 2], [0, 2, 3]] if np.dot(np.cross(b - a, c - a), fr.radial) > 0 else [[0, 2, 1], [0, 3, 2]]
            node = bodies.addChild(f'guard{fr.k + 1}')
            node.addObject('MeshTopology', name='topo', position=corners, triangles=tris)
            node.addObject('MechanicalObject', name='dofs', template='Vec3', position=corners)
            node.addObject('TriangleCollisionModel', group=[j + 1 for j in range(F) if j != fr.k]
                           + [ITEMS_GROUP + i for i in range(len(ITEMS))])
            node.addObject('RigidMapping', name='mapping', globalToLocalCoords=True, rigidIndexPerPoint=[0] * 4)

    pad, stem = 0.1 * H, 0.3 * H
    plate_r = extent + pad
    plate_pts, plate_tris = _frustum_z(0.0, 0.0, z_palm, z_palm + thick, plate_r, plate_r, n=max(24, 4 * F))
    node = bodies.addChild('plateCollision')
    coarse = _frustum_z(0.0, 0.0, z_palm, z_palm + thick, plate_r, plate_r, n=12)
    node.addObject('MeshTopology', name='topo', position=coarse[0], triangles=coarse[1])
    node.addObject('MechanicalObject', name='dofs', template='Vec3', position=coarse[0])
    for model in ('TriangleCollisionModel', 'LineCollisionModel', 'PointCollisionModel'):
        node.addObject(model, group=list(range(1, F + 1)) + [FINGERS_GROUP])  # (not the fingers, nor the guards)
    node.addObject('RigidMapping', name='mapping', globalToLocalCoords=True, rigidIndexPerPoint=[0] * len(coarse[0]))

    # ---- visuals: the segments (boxes), the base layer (a thin strip under each), the hinges' marks, the end-effector
    seg_parts, seg_index = [], []
    sheet_parts, sheet_index = [], []
    mark_parts, mark_index = [], []
    for fr in frames:
        for j in range(N):
            x0 = j * pitch
            pts, tris = _box(x0, -W / 2, 0.0, x0 + L, W / 2, H)
            seg_parts.append(([fr.to_world(p) for p in pts], tris))
            seg_index.append((body_of[(fr.k, j)], len(pts)))
            sx0, sx1 = x0 - (sep / 2 if j > 0 else 0.0), x0 + L + (sep / 2 if j < N - 1 else 0.0)
            sp, st = _box(sx0, -W / 2 * 0.98, -0.4 * MM, sx1, W / 2 * 0.98, 0.0)
            sheet_parts.append(([fr.to_world(p) for p in sp], st))
            sheet_index.append((body_of[(fr.k, j)], len(sp)))
    for fk, j, pv in pivots:  # a small cylinder along the hinge axis, on the distal segment's body
        fr = frames[fk]
        half_w = W / 2 + 0.6 * MM
        mp, mt = _cylinder_along(list(np.array(pv) - half_w * fr.tangent), list(np.array(pv) + half_w * fr.tangent),
                                 1.2 * MM, n=16)
        mark_parts.append((mp, mt))
        mark_index.append((body_of[(fk, j)], len(mp)))

    def add_visual(name, parts, index, color, alpha_key=None):
        pts, tris = _merge(parts)
        per_point = []
        for item in index:
            per_point += [item[0]] * item[1]
        node = bodies.addChild(name)
        node.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        node.addObject('OglModel', name='visual', src='@topo', color=color + [float(P[alpha_key]) if alpha_key else 1.0])
        node.addObject('RigidMapping', name='mapping', globalToLocalCoords=True, input='@..', output='@visual',
                       rigidIndexPerPoint=per_point)

    add_visual('visu', seg_parts, seg_index, [0.15, 0.45, 0.95], 'wall_alpha')
    add_visual('base_layer', sheet_parts, sheet_index, [0.96, 0.94, 0.86], 'wall_alpha')
    if P['show_joint_marks'] and mark_parts:
        add_visual('joint_marks', mark_parts, mark_index, [0.95, 0.6, 0.1], 'joint_alpha')

    # the end-effector: a round plate with an arm up, drawn on the flange body
    mount_parts = [(plate_pts, plate_tris), _box(-stem, -stem, z_palm + thick, stem, stem, z_top + 0.2 * H)]
    mount_pts, mount_tris = _flat_shaded(*_merge(mount_parts))
    mount = bodies.addChild('trunk')
    mount.addObject('MeshTopology', name='topo', position=mount_pts, triangles=mount_tris)
    mount.addObject('OglModel', name='visual', src='@topo', color=[0.4, 0.26, 0.13, 1.0])
    mount.addObject('RigidMapping', name='mapping', globalToLocalCoords=True, input='@..', output='@visual',
                    rigidIndexPerPoint=[0] * len(mount_pts))

    pressure_ramp = PressureRamp(rootNode, pressure_cff, n_angles, torque_per_pa, 0.0, P['ramp_time'],
                                 instant=P['instant'], ramp_start=P['ramp_start'], name='PressureRamp')
    rootNode.addObject(pressure_ramp)

    BUILD.clear()
    BUILD.update({'segments': N, 'fingers': F, 'items': [], 'body_of': dict(body_of), 'flange_rest0': list(flange_pos),
                  'flange_rest': [(min(p[q] for p in mount_pts) + max(p[q] for p in mount_pts)) / 2 for q in range(3)],
                  'cavity_mid': cavity_mid, 'cavity_area': cavity_area, 'torque_per_pa': torque_per_pa,
                  'close_limit': close_lim, 'seg_mass': seg_mass, 'plate_radius': plate_r, 'stem': stem})

    # ---- the grocery items: free rigid bodies on the floor
    items = world.addChild('items')
    scale = max(float(P['item_mass_scale']), 1e-3)
    yaw = math.radians(float(P['item_yaw']))
    guarded = []
    q_yaw = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
    for key, label, shape, dims_mm, mass_g, (x_mm, y_mm) in ITEMS:
        if not P.get(f'item_{key}', True):
            continue
        dims = [d * MM for d in dims_mm]
        item_group = ITEMS_GROUP + len(guarded)  # its own (the items touch each other), one the guards are in
        mass = mass_g * 1e-3 * scale
        half_h, inertia = _item_extent(shape, dims)
        # (a hair above the floor: a contact that starts at exactly zero distance has no normal, and lets go)
        centre = [x_mm * MM * spacing, y_mm * MM * spacing, floor_z + half_h + float(P['contact_distance']) * MM]
        c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)

        def place(p, centre=centre, c_yaw=c_yaw, s_yaw=s_yaw):  # item frame -> world (turned by item_yaw about z)
            return [centre[0] + c_yaw * p[0] - s_yaw * p[1], centre[1] + s_yaw * p[0] + c_yaw * p[1], centre[2] + p[2]]
        node = items.addChild(key)
        node.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=0.0, rayleighMass=0.0)
        node.addObject('CGLinearSolver', name='linearSolver', iterations=25, tolerance=1e-12, threshold=1e-18)
        dofs = node.addObject('MechanicalObject', name='dofs', template='Rigid3', position=[centre + q_yaw])
        node.addObject('UniformMass', name='mass', vertexMass=[mass, 1.0, mass * inertia[0], 0, 0, 0, mass * inertia[1], 0, 0,
                                                               0, mass * inertia[2]])
        node.addObject('UncoupledConstraintCorrection', name='constraintCorrection')
        col = node.addChild('collision')
        if shape in ('sphere', 'egg'):
            if shape == 'sphere':
                spheres = [([0.0, 0.0, 0.0], dims[0])]
            else:  # an egg lying along x: three spheres
                a, r = dims
                off = 0.42 * a
                rr = r * math.sqrt(1.0 - (off / a) ** 2) * 0.92
                spheres = [([-off, 0.0, 0.0], rr), ([0.0, 0.0, 0.0], r), ([off, 0.0, 0.0], rr)]
            col.addObject('MechanicalObject', name='dofs', template='Vec3',
                          position=[place(c) for c, _ in spheres])
            col.addObject('SphereCollisionModel', listRadius=[r for _, r in spheres], group=[item_group])
            radii = [r for _, r in spheres]
        else:
            cp, ct, _ = _item_meshes(shape, dims, smooth=False)[0]
            cp = [place(p) for p in cp]
            col.addObject('MeshTopology', name='topo', position=cp, triangles=ct)
            col.addObject('MechanicalObject', name='dofs', template='Vec3', position=cp)
            for model in ('TriangleCollisionModel', 'LineCollisionModel', 'PointCollisionModel'):
                col.addObject(model, group=[item_group])
            radii = [0.0] * len(cp)
        col.addObject('RigidMapping', name='mapping', globalToLocalCoords=True)
        guarded.append((dofs, col.getObject('dofs'), np.array(radii)))
        parts = [(p, t, ITEM_COLOURS[key]) for p, t, _ in _item_meshes(shape, dims, smooth=True)] + _item_details(key, dims)
        for n_part, (vp, vt, colour) in enumerate(parts):
            vp, vt = (vp, vt) if shape in ('sphere', 'egg', 'cylinder', 'cup') and n_part == 0 else _flat_shaded(vp, vt)
            visu = node.addChild('visu' if n_part == 0 else f'detail{n_part}')
            visu.addObject('MeshTopology', name='topo', position=[place(p) for p in vp],
                           triangles=vt)
            visu.addObject('OglModel', name='visual', src='@topo', color=list(colour) + [float(P['item_alpha'])])
            visu.addObject('RigidMapping', name='mapping', input='@../dofs', output='@visual', globalToLocalCoords=True)
        BUILD['items'].append({'key': key, 'label': label, 'rest': centre, 'mass': mass})

    rootNode.addObject(FloorGuard(guarded, floor_z, float(P['contact_distance']) * MM, name='FloorGuard'))
    rootNode.addObject(FlangeController(rootNode, flange_dofs, flange_pos, c_wrist / k_wrist,
                                        acceleration=float(P['flange_acceleration']), speed=float(P['flange_speed']),
                                        name='Flange'))
    return rootNode


def main():
    import Sofa.Simulation
    os.environ.setdefault('GRIPPER4_NO_EDITOR', '1')
    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)
    for _ in range(int((PARAMS['ramp_start'] + PARAMS['ramp_time'] + 2.0) / DT)):
        Sofa.Simulation.animate(root, DT)
    Sofa.Simulation.unload(root)


if __name__ == '__main__':
    main()
