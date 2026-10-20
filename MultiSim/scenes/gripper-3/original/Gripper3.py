# -*- coding: utf-8 -*-
"""
Gripper 3: the gripper of Gripper 2 -- two fingers hanging from a movable end-effector, loose cylinders on
the floor to pick up -- with the fingers modelled as chains of rigid segments and compliant joints instead of
a finite-element mesh. A PneuNet finger of `segments` chambers becomes `segments` rigid bodies, one per
chamber, in a row along the finger with a `channel_separation` gap between neighbours. The chambers are
only joined by the thin base layer (the strain-limiting layer), so each gap is a hinge at the base with
its axis along Y. The first segment is welded to the end-effector.

Joints. Each hinge is a SOFA JointSpringForceField between two frames at the hinge point, one fixed to
each neighbouring segment:
  - the frames are held together by a stiff spring (a pin: `pin_stiffness`);
  - the rotation about Y is compliant: a torsional spring of `joint_stiffness` [N m/rad] with damping
    stiffness-proportional damping `joint_damping` [s] (the solver's, implicit; the joint spring's own velocity damping
    is explicit and unstable for bodies this light); this is the stiffness of the silicone walls and of the base against
    bending the finger;
  - beyond the hinge's limits the rotation is stopped hard (`joint_stop_stiffness`): closing (the
    roofs of two neighbouring chambers touching, the gap being `channel_separation` wide at the base
    and widening to the roof) and over-opening (`joint_open_limit`).

Pressure. Inflating a chamber pushes its end walls apart; in this reduced model that is a torque on the
joints around it: the pressure p times the cross-section of the cavity (A = (width - 2 side walls) x
(height - bottom - roof)) is a force on the wall, acting at the cavity's mid height above the hinge (the base),
so the torque is p A z, equal and opposite on the two segments of a joint (an internal couple: it bends the
finger and pushes on nothing else), times `pressure_torque_gain`. The torque grows linearly with the pressure, and the
bend with it (bend per joint = torque / `joint_stiffness`); the defaults are fitted to Gripper 2's finite-element
fingers (see the notes in sofaweb_scene.py). There is no ballooning, no stretching of the base and no
nonlinearity of the silicone.

Everything else is as in Gripper 2: the flange accelerates smoothly to the target x / z set by the sliders,
the cylinders (rigid bodies on the floor, axis along Y, no friction) are touched by points on the fingers'
base faces (penalty contact on each point's distance to the cylinder's centreline), the floor and the plane
between the fingers (x = the flange's x) stop the segments' corners. Because the model is planar and has no mesh,
a step takes a few milliseconds: it runs in real time. The cylinders are full cylinders here (not
half models: the fingers are not half models either).

Units: the scene is SI (metres, kg, s, Pa); params.json takes millimetres for lengths, as labelled.

Run with the GUI:   runSofa -l SofaPython3 -g glfw Gripper3.py
Headless:           python Gripper3.py
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

_DEFAULT_PARAMS = {
    # -- geometry [mm] --
    'segments': 5,               # rigid segments (chambers) in each finger
    'length': 8.0,               # one segment, along X (the finger's axis)
    'width': 20.0,               # along Y
    'height': 16.0,              # along Z (up)
    'wall_end': 1.5,             # end walls (they set the cavity's cross-section and the mass)
    'wall_side': 1.5,
    'wall_bottom': 3.0,
    'wall_top': 1.5,
    'channel_separation': 2.0,   # gap between neighbouring segments, at the base
    'floor_size': 320.0,         # green floor, purely visual
    # -- joints --
    'joint_stiffness': 0.34,     # [N m/rad] torsional stiffness of each hinge
    'joint_damping': 0.01,       # [s] stiffness-proportional damping of the joints and contacts (implicit; the joints' own
                                 # velocity damping is explicit, unstable for bodies this light)
    'joint_open_limit': 90.0,    # [deg] largest bend of a hinge (a hard stop beyond it)
    'pin_stiffness': 2.0e4,      # [N/m] holds the two halves of a hinge together
    'joint_stop_stiffness': 5.0, # [N m/rad] the hard stops, and the weld to the end-effector
    'pressure_torque_gain': 1.0, # 1: the torque of the pressure on the cavity's cross-section (see the docstring)
    'flange_acceleration': 0.06,      # [m/s2] largest acceleration of the flange
    'flange_speed': 0.15,             # [m/s] its largest speed
    'finger_gap': 28.0,               # [mm] between the two fingers' inner faces (their base layers)
    # -- the loose cylinders on the floor (rigid, axis along Y), and the contact with them
    'cylinder_radii': [12.0, 9.0, 6.0],   # [mm]
    'cylinder_x': [-60.0, 0.0, 60.0],     # [mm] where they start (on the floor)
    'cylinder_length': 30.0,          # [mm] along Y
    'cylinder_density': 600.0,        # [kg/m^3]
    'cylinder_damping': 0.2,          # [N s/m] velocity damping of the cylinders (rolling and sliding resistance)
    'contact_stiffness': 1000.0,      # [N/m] per contact point (segments - cylinders, floor, plane between the fingers)
    'contact_whole_skin': False,      # points all over the segments' surface can touch a cylinder (False: the base face)
    'cylinder_alpha': 1.0,
    # -- material --
    'density': 1000.0,           # [kg/m^3] of the silicone (the segment's mass: the walls' volume)
    'gravity_enabled': True,
    # -- pressure --
    'ramp_start': 0.0,           # [s] when the pressure starts to rise
    'ramp_time': 0.5,            # [s]
    'dt': 0.01,                  # [s] simulation time step
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
    interpreter). Skipped when GRIPPER3_NO_EDITOR is set (headless runs)."""
    if os.environ.get('GRIPPER3_NO_EDITOR'):
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
ROD_DAMPING = 5.0    # [N s/m] per point of the contacts
FLANGE_START = (0.0, 80.0)   # the flange's x and the height of its underside above the floor [mm] at the start


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


def _cylinder_y(cx, cz, radius, y0, y1, n=48):
    """A closed cylinder with its axis along Y at (cx, cz), from y0 to y1."""
    pts, tris = [], []
    ring = [(cx + radius * math.cos(2 * math.pi * k / n), cz + radius * math.sin(2 * math.pi * k / n)) for k in range(n)]
    for k in range(n):  # the side: two triangles per segment, outward normals
        k2 = (k + 1) % n
        a, b = (ring[k][0], y0, ring[k][1]), (ring[k2][0], y0, ring[k2][1])
        c, d = (ring[k2][0], y1, ring[k2][1]), (ring[k][0], y1, ring[k][1])
        base = len(pts)
        pts += [list(a), list(b), list(c), list(d)]
        tris += [[base, base + 2, base + 1], [base, base + 3, base + 2]]
    for y, up in ((y0, False), (y1, True)):  # the caps
        base = len(pts)
        pts.append([cx, y, cz])
        pts += [[x, y, z] for x, z in ring]
        for k in range(n):
            a, b = base + 1 + k, base + 1 + (k + 1) % n
            tris.append([base, a, b] if up else [base, b, a])
    return pts, tris


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


def _rotate(q, v):
    """Quaternion [x, y, z, w] applied to vector v."""
    ux, uy, uz, w = q
    cx, cy, cz = uy * v[2] - uz * v[1], uz * v[0] - ux * v[2], ux * v[1] - uy * v[0]
    return [v[0] + 2 * (w * cx + uy * cz - uz * cy),
            v[1] + 2 * (w * cy + uz * cx - ux * cz),
            v[2] + 2 * (w * cz + ux * cy - uy * cx)]


def _qmul(a, b):
    """Quaternion product a * b ([x, y, z, w])."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


_S = math.sqrt(0.5)
Q_FINGER = [[0.0, _S, 0.0, _S],                       # finger 1 (+x side): local x -> -z (hangs down), local z -> +x
            _qmul([0.0, 0.0, 1.0, 0.0], [0.0, _S, 0.0, _S])]  # finger 2: the same turned 180 degrees about the vertical
SIDES = (1.0, -1.0)
IDENTITY = [0.0, 0.0, 0.0, 1.0]  # the bodies' and joint frames' orientation at rest (the JointSpringForceField misbehaves with
                                  # turned frames; the geometry is given in world coordinates and mapped, so it needn't be turned)


class FingerFrame:
    """Where a finger hangs. Its own frame: x along the finger from its proximal end, z up out of its roof,
    the base (the side it curls towards) at z = 0, y the width. Finger 2 is finger 1 turned 180 degrees about
    the vertical axis, so both curl towards the middle."""

    def __init__(self, k, gap, z_palm):
        self.k, self.side, self.q = k, SIDES[k], Q_FINGER[k]
        self.origin = [self.side * gap / 2, 0.0, z_palm]

    def to_world(self, p):
        r = _rotate(self.q, p)
        return [r[0] + self.origin[0], r[1] + self.origin[1], r[2] + self.origin[2]]


class FlangeController(Sofa.Core.Controller):
    """The end-effector: moves towards a target position (`targetX`, `targetZ`, metres from where it starts; the
    interface's sliders set them) with a limited acceleration and speed, carrying the fingers (their first segments
    are welded to it) and the plane between the fingers."""

    def __init__(self, root, body_dofs, plate_rest, midplanes, acceleration=0.06, speed=0.15, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.root, self.dofs, self.rest, self.midplanes = root, body_dofs, np.array(plate_rest, dtype=float), midplanes
        self.addData(name='targetX', type='float', value=0.0, help='Flange x relative to the start [m]. Live-editable.')
        self.addData(name='targetZ', type='float', value=0.0, help='Flange z relative to the start [m]. Live-editable.')
        self.addData(name='acceleration', type='float', value=acceleration,
                     help='Largest acceleration of the flange [m/s2]. Live-editable.')
        self.addData(name='speed', type='float', value=speed, help='Largest speed of the flange [m/s]. Live-editable.')
        self.offset = [0.0, 0.0]
        self.velocity = [0.0, 0.0]

    def onAnimateBeginEvent(self, event):
        dt = float(self.root.dt.value)
        accel = max(float(self.acceleration.value), 1e-4)
        top = max(float(self.speed.value), 1e-4)
        goals = (float(self.targetX.value), float(self.targetZ.value))
        for a in range(2):
            d = goals[a] - self.offset[a]
            # The speed we may have here and still be able to stop at the goal (braking at `accel`), at most `top`:
            wanted = math.copysign(min(top, math.sqrt(2.0 * accel * abs(d))), d)
            dv = max(-accel * dt, min(accel * dt, wanted - self.velocity[a]))
            self.velocity[a] += dv
            self.offset[a] += self.velocity[a] * dt
            if abs(d) < 1e-6 and abs(self.velocity[a]) <= 2 * accel * dt:  # arrived
                self.offset[a], self.velocity[a] = goals[a], 0.0
        with self.dofs.position.writeable() as pos:
            pos[0][0] = self.rest[0] + self.offset[0]
            pos[0][2] = self.rest[2] + self.offset[1]
        for plane, side in self.midplanes:  # the plane x = flange x: the finger on +x keeps x >= it, the other x <= it
            plane.d.value = float(side) * self.offset[0]


class PressureRamp(Sofa.Core.Controller):
    """Drives one finger's pressure towards a live-editable target, ramped or instantly (`targetPressure`,
    `rampTime`, `rampStart`, `instant` are Data fields), and applies it as the torque on the finger's joints:
    each joint gets +-`torque_per_pa` x p, a couple on the two segments around it (the pressure acting on the
    cavity's cross-section at the cavity's mid height above the hinge)."""

    def __init__(self, root, force_field, bodies, sign, torque_per_pa, target_pressure, ramp_time,
                 instant=False, ramp_start=0.0, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.root, self.cff, self.bodies, self.sign, self.torque_per_pa = root, force_field, bodies, sign, torque_per_pa
        self.addData(name='targetPressure', type='float', value=target_pressure,
                     help='Target gauge pressure in the finger [Pa]. Live-editable.')
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
        tau = self.sign * self.torque_per_pa * self.current_pressure  # about the world y axis, on the distal segment of a joint
        n = len(self.bodies)
        forces = []
        for j in range(n):
            t = (tau if j > 0 else 0.0) - (tau if j < n - 1 else 0.0)  # +tau from the joint before, -tau from the one after
            forces.append([0.0, 0.0, 0.0, 0.0, t, 0.0])
        self.cff.forces.value = forces

    def applied_pressure(self):
        """The pressure acting in the finger [Pa]."""
        return self.current_pressure


# ----------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------
# Filled in by createScene for the sofaweb probes (and anyone inspecting a built scene): geometry in SOFA units.
BUILD = {}


def createScene(rootNode):
    launch_params_editor()
    P = PARAMS

    plugins = [
        "Sofa.Component.AnimationLoop", "Sofa.Component.Constraint.Projective", "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.Mapping.Linear", "Sofa.Component.Mapping.NonLinear", "Sofa.Component.Mass",
        "Sofa.Component.MechanicalLoad", "Sofa.Component.ODESolver.Backward", "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer", "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Setting", "Sofa.Component.Visual", "Sofa.GL.Component.Rendering3D", "Sofa.GL.Component.Shader",
    ]
    for p in plugins:
        rootNode.addObject('RequiredPlugin', name='req_' + p.replace('.', '_'), pluginName=p)

    N = max(1, int(P['segments']))
    L, W, H = float(P['length']) * MM, float(P['width']) * MM, float(P['height']) * MM
    sep = max(float(P['channel_separation']), 0.0) * MM
    pitch = L + sep
    finger_len = N * L + (N - 1) * sep
    we, ws, wb, wt = (float(P[k]) * MM for k in ('wall_end', 'wall_side', 'wall_bottom', 'wall_top'))
    gap = max(float(P['finger_gap']), 0.0) * MM
    z_palm = FLANGE_START[1] * MM
    extent_x = gap / 2 + H

    # Mass: the walls' volume (the box minus the cavity), a box's moment of inertia about y for the planar motion.
    cav_l, cav_w, cav_h = max(L - 2 * we, 0.0), max(W - 2 * ws, 0.0), max(H - wb - wt, 0.0)
    seg_volume = L * W * H - cav_l * cav_w * cav_h
    seg_mass = float(P['density']) * seg_volume
    seg_inertia = seg_mass * (L * L + H * H) / 12.0
    # Pressure: p x the cavity's cross-section is a force on an end wall, at the cavity's mid height above the base.
    cavity_area = cav_w * cav_h
    cavity_mid = wb + 0.5 * cav_h
    torque_per_pa = float(P['pressure_torque_gain']) * cavity_area * cavity_mid
    print(f"[model] 2 fingers of {N} rigid segments, {seg_mass * 1e3:.2f} g each; pressure torque {torque_per_pa * 5e4:.3f} N m "
          f"per joint at 50 kPa; joint stiffness {P['joint_stiffness']} N m/rad")

    rootNode.addObject('BackgroundSetting', color=[0.78, 0.87, 0.96, 1.0], image="")
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                                   'hideBoundingCollisionModels hideForceFields hideInteractionForceFields '
                                                   'hideWireframe')

    floor_z = 0.0
    thick = 0.6 * H                     # the end-effector's plate
    R_gauge = 0.6 * H                   # the pressure gauges' dial radius (see sofaweb_scene._gauge)
    z_top = z_palm + thick + 1.12 * R_gauge + 0.6 * H + R_gauge  # the top of the gauges
    size = max(z_top, 2 * extent_x)

    cam_lookat = [0.0, 0.0, 0.5 * z_top]
    cam_dir = (1.0, -4.2, 1.2)
    norm = math.sqrt(sum(c * c for c in cam_dir))
    cam_distance = 0.7 * z_top / math.tan(math.radians(22.5))
    cam_position = [l + cam_distance * c / norm for l, c in zip(cam_lookat, cam_dir)]
    rootNode.addObject('InteractiveCamera', name='camera', position=cam_position, distance=cam_distance,
                       orientation=_look_at_quaternion(cam_position, cam_lookat), activated=True)
    rootNode.addObject('LightManager', ambient=[0.35, 0.35, 0.35, 1.0])
    rootNode.addObject('DirectionalLight', direction=[-0.5, -0.15, 1.0])
    rootNode.addObject('DirectionalLight', color=[0.35, 0.35, 0.35, 1.0], direction=[0.5, 0.2, 0.3])

    # Floor and axes: purely visual.
    half = max(P['floor_size'] * MM, 3 * extent_x) / 2
    floor = rootNode.addChild('floor')
    floor.addObject('MeshTopology', name='topo', triangles=[[0, 1, 2], [0, 2, 3]],
                    position=[[-half, -half, floor_z - 1e-4], [half, -half, floor_z - 1e-4],
                              [half, half, floor_z - 1e-4], [-half, half, floor_z - 1e-4]])
    floor.addObject('OglModel', name='visual', src='@topo', color=[0.08, 0.28, 0.1, 1.0])

    arrow_ref = 0.6 * max(finger_len, W, H)
    head_len, shaft_half = arrow_ref * 0.2, arrow_ref * 0.015
    totals = {0: max(arrow_ref, extent_x + 0.3 * arrow_ref + head_len),
              1: max(arrow_ref, 1.6 * (W / 2) + head_len)}
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

    rootNode.gravity = [0.0, 0.0, -9.81] if P['gravity_enabled'] else [0.0, 0.0, 0.0]
    rootNode.dt = DT
    rootNode.addObject('DefaultAnimationLoop')

    # One solver for everything that moves. The joint springs have no stiffness matrix, so the linear system is solved
    # matrix-free (conjugate gradient; the system is tiny).
    world = rootNode.addChild('world')
    world.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=float(P['joint_damping']), rayleighMass=0.0)
    world.addObject('CGLinearSolver', name='linearSolver', template='GraphScattered', iterations=300,
                    tolerance=1e-12, threshold=1e-30)

    # ---- the bodies: 0 = the end-effector's flange (moved by the FlangeController), then each finger's segments ----
    frames = [FingerFrame(k, gap, z_palm) for k in range(2)]
    flange_pos = [0.0, 0.0, z_palm + thick / 2]
    positions = [flange_pos + [0.0, 0.0, 0.0, 1.0]]
    body_of = {}
    for fr in frames:
        for j in range(N):
            c = fr.to_world([j * pitch + L / 2, 0.0, H / 2])
            body_of[(fr.k, j)] = len(positions)
            positions.append(c + IDENTITY)
    bodies = world.addChild('bodies')
    body_dofs = bodies.addObject('MechanicalObject', name='dofs', template='Rigid3', position=positions)
    # planar motion only: x, z and the rotation about y (angular directions are world axes)
    bodies.addObject('PartialFixedProjectiveConstraint', name='planar', indices=list(range(1, len(positions))),
                     fixedDirections=[0, 1, 0, 1, 0, 1])
    bodies.addObject('FixedProjectiveConstraint', name='flangeFixed', indices=[0])
    bodies.addObject('UniformMass', name='mass', vertexMass=[seg_mass, 1.0, seg_inertia, 0, 0, 0, seg_inertia, 0, 0, 0,
                                                              seg_inertia])

    # ---- the joints: a pair of frames at each hinge (rigid on either side), a JointSpringForceField between them
    kt_pin = float(P['pin_stiffness'])
    kr_stop = float(P['joint_stop_stiffness'])
    kr = float(P['joint_stiffness'])
    kd = 0.0  # (the joints' own damping is explicit: unstable for a body this light)
    open_lim = math.radians(float(P['joint_open_limit']))
    close_lim = 2.0 * math.atan2(sep / 2, H)  # the roofs touch: the gap is `sep` wide at the base and 0 at the roof
    joint_frames = []  # [pose, body]
    springs = []
    pivots = []        # for the visual marks: (finger, world pivot)
    for fr in frames:
        # the weld to the flange
        a = len(joint_frames)
        joint_frames.append(fr.to_world([0.0, 0.0, 0.0]) + IDENTITY)
        joint_frames.append(fr.to_world([0.0, 0.0, 0.0]) + IDENTITY)
        springs.append(f"BEGIN_SPRING {a} {a + 1} FREE_AXIS 0 0 0 0 0 0 KS_T 0 {kt_pin} KS_R 0 {kr_stop} KD {kd} "
                       f"R_LIM_X 0 0 R_LIM_Y 0 0 R_LIM_Z 0 0 END_SPRING\n")
        for j in range(N - 1):
            a = len(joint_frames)
            pv = fr.to_world([j * pitch + L + sep / 2, 0.0, 0.0])
            pivots.append((fr.k, pv))
            joint_frames.append(pv + IDENTITY)
            joint_frames.append(pv + IDENTITY)
            # rotation about y_local: positive curls the finger towards its base (closes the roofs' gap when negative)
            lo, hi = (-close_lim, open_lim) if fr.side > 0 else (-open_lim, close_lim)  # finger 2 curls about -y
            springs.append(f"BEGIN_SPRING {a} {a + 1} FREE_AXIS 0 0 0 0 1 0 KS_T 0 {kt_pin} KS_R {kr} {kr_stop} KD {kd} "
                           f"R_LIM_X 0 0 R_LIM_Y {lo} {hi} R_LIM_Z 0 0 END_SPRING\n")
    index_per_frame = []
    for fr in frames:
        index_per_frame += [0, body_of[(fr.k, 0)]]
        for j in range(N - 1):
            index_per_frame += [body_of[(fr.k, j)], body_of[(fr.k, j + 1)]]
    joints = bodies.addChild('joints')
    joints.addObject('MechanicalObject', name='frames', template='Rigid3', position=joint_frames)
    joints.addObject('RigidMapping', name='mapping', globalToLocalCoords=True, input='@..', output='@.', rigidIndexPerPoint=index_per_frame)
    joints.addObject('JointSpringForceField', name='springs', template='Rigid3', spring=''.join(springs))

    # ---- contacts of the fingers: points on the segments (y = 0; the model is planar)
    hull_pts, hull_idx = [], []      # the corners: floor and the plane between the fingers
    touch_pts, touch_idx = [], []    # the points that can touch a cylinder
    for fr in frames:
        for j in range(N):
            for x in (0.0, L):
                for z in (0.0, H):
                    hull_pts.append(fr.to_world([j * pitch + x, 0.0, z]))
                    hull_idx.append(body_of[(fr.k, j)])
            if P['contact_whole_skin']:
                local = [(x, z) for x in (0.0, L / 2, L) for z in (0.0, H / 2, H) if not (x == L / 2 and z == H / 2)]
            else:
                local = [(0.0, 0.0), (L / 2, 0.0), (L, 0.0)]
            for x, z in local:
                touch_pts.append(fr.to_world([j * pitch + x, 0.0, z]))
                touch_idx.append(body_of[(fr.k, j)])
    hull = bodies.addChild('hull')
    hull.addObject('MechanicalObject', name='dofs', template='Vec3', position=hull_pts)
    hull.addObject('RigidMapping', name='mapping', globalToLocalCoords=True, input='@..', output='@.', rigidIndexPerPoint=hull_idx)
    hull.addObject('PlaneForceField', name='floor', normal=[0.0, 0.0, 1.0], d=floor_z,
                   stiffness=float(P['contact_stiffness']), damping=ROD_DAMPING)
    midplanes = []
    for fr in frames:
        # the plane between the fingers (x = the flange's x), which a finger can't cross
        pts = [i for i, b in enumerate(hull_idx) if b in [body_of[(fr.k, j)] for j in range(N)]]
        mid = hull.addChild(f'midPlane{fr.k + 1}')
        mid.addObject('MechanicalObject', name='mstate', position=[hull_pts[i] for i in pts])
        mid.addObject('SubsetMapping', input='@..', output='@mstate', indices=pts, handleTopologyChange=False)
        plane = mid.addObject('PlaneForceField', name='midPlane', normal=[fr.side, 0.0, 0.0], d=0.0,
                              stiffness=float(P['contact_stiffness']), damping=ROD_DAMPING)
        midplanes.append((plane, fr.side))
    touch = bodies.addChild('touch')
    touch_dofs = touch.addObject('MechanicalObject', name='dofs', template='Vec3', position=touch_pts)
    touch.addObject('RigidMapping', name='mapping', globalToLocalCoords=True, input='@..', output='@.', rigidIndexPerPoint=touch_idx)

    # ---- visuals: the segments (boxes), the base layer (a thin strip under each), the hinges' marks, the end-effector
    seg_parts, seg_index = [], []
    sheet_parts, sheet_index = [], []
    mark_parts, mark_index = [], []
    for fr in frames:
        for j in range(N):
            x0 = j * pitch
            corners = _box(x0, -W / 2, 0.0, x0 + L, W / 2, H)
            pts, tris = corners
            seg_parts.append(([fr.to_world(p) for p in pts], tris))
            seg_index.append((body_of[(fr.k, j)], len(pts)))
            sx0, sx1 = x0 - (sep / 2 if j > 0 else 0.0), x0 + L + (sep / 2 if j < N - 1 else 0.0)
            sp, st = _box(sx0, -W / 2 * 0.98, -0.4 * MM, sx1, W / 2 * 0.98, 0.0)
            sheet_parts.append(([fr.to_world(p) for p in sp], st))
            sheet_index.append((body_of[(fr.k, j)], len(sp)))
    for fk, pv in pivots:  # a small cylinder along y at each hinge, on the distal segment's body
        j = len([m for m in mark_index if m[2] == fk])
        r = 1.2 * MM
        mp, mt = _cylinder_y(pv[0], pv[2], r, -W / 2 - 0.6 * MM, W / 2 + 0.6 * MM, n=16)
        mark_parts.append((mp, mt))
        mark_index.append((body_of[(fk, j + 1)], len(mp), fk))

    def add_visual(name, parts, index, color, alpha_key=None, cull=None):
        pts, tris = _merge(parts)
        per_point = []
        for item in index:
            per_point += [item[0]] * item[1]
        node = bodies.addChild(name)
        node.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        kwargs = {} if cull is None else {'cullFace': cull}
        node.addObject('OglModel', name='visual', src='@topo', color=color + [float(P[alpha_key]) if alpha_key else 1.0],
                       **kwargs)
        node.addObject('RigidMapping', name='mapping', globalToLocalCoords=True, input='@..', output='@visual', rigidIndexPerPoint=per_point)

    add_visual('visu', seg_parts, seg_index, [0.15, 0.45, 0.95], 'wall_alpha')
    add_visual('base_layer', sheet_parts, sheet_index, [0.96, 0.94, 0.86], 'wall_alpha')
    if P['show_joint_marks']:
        add_visual('joint_marks', mark_parts, mark_index, [0.95, 0.6, 0.1], 'joint_alpha')

    # the end-effector: a plate with an arm up, drawn on the flange body
    pad, stem = 0.1 * H, 0.3 * H
    mount_parts = [
        _box(-extent_x - pad, -W / 2 - pad, z_palm, extent_x + pad, W / 2 + pad, z_palm + thick),
        _box(-stem, -stem, z_palm + thick, stem, stem, z_top + 0.2 * H),
    ]
    plate_pts, plate_tris = _flat_shaded(*_merge(mount_parts))
    mount = bodies.addChild('trunk')
    mount.addObject('MeshTopology', name='topo', position=plate_pts, triangles=plate_tris)
    mount.addObject('OglModel', name='visual', src='@topo', color=[0.4, 0.26, 0.13, 1.0])
    mount.addObject('RigidMapping', name='mapping', globalToLocalCoords=True, input='@..', output='@visual', rigidIndexPerPoint=[0] * len(plate_pts))

    BUILD.clear()
    BUILD.update({'segments': N, 'fingers': [], 'cylinders': [], 'body_of': dict(body_of),
                  'flange_rest': [(min(p[q] for p in plate_pts) + max(p[q] for p in plate_pts)) / 2 for q in range(3)],
                  'cavity_mid': cavity_mid, 'cavity_area': cavity_area, 'torque_per_pa': torque_per_pa,
                  'close_limit': close_lim, 'seg_mass': seg_mass})

    # ---- the pressure of each finger: a torque couple on the neighbouring segments of each joint
    for fr in frames:
        idx = [body_of[(fr.k, j)] for j in range(N)]
        cff = bodies.addObject('ConstantForceField', name=f'torque{fr.k + 1}', indices=idx,
                               forces=[[0.0] * 6 for _ in idx], showArrowSize=0.0)
        # about the world y axis: finger 1's local y is +y, finger 2's is -y (it is turned about the vertical axis)
        rootNode.addObject(PressureRamp(rootNode, cff, idx, fr.side, torque_per_pa, 0.0, P['ramp_time'],
                                        instant=P['instant'], ramp_start=P['ramp_start'], name=f'PressureRamp{fr.k + 1}'))
        BUILD['fingers'].append({'side': fr.side, 'gap': gap, 'z_palm': z_palm, 'bodies': idx})

    # ---- loose cylinders on the floor: rigid bodies that can slide in x and z and roll about y (their axis stays
    # parallel to Y). Contact with a finger is a penalty on the distance of each of the finger's touch points to the
    # cylinder's centreline, acting on both bodies (no friction). Cylinders keep apart by the same kind of penalty
    # between their centres, and rest on the floor by one on the height of their centre.
    n_cyl = min(len(P['cylinder_radii']), len(P['cylinder_x']))
    cyl_len = max(float(P['cylinder_length']), 1.0) * MM
    palette = [[0.86, 0.55, 0.2], [0.3, 0.72, 0.52], [0.75, 0.42, 0.78], [0.9, 0.8, 0.3], [0.45, 0.62, 0.9], [0.9, 0.45, 0.45]]
    centres = []
    n_touch = len(touch_pts)
    for j in range(n_cyl):
        r = max(float(P['cylinder_radii'][j]), 0.5) * MM
        cx = float(P['cylinder_x'][j]) * MM
        mass = float(P['cylinder_density']) * math.pi * r * r * cyl_len
        inertia = 0.5 * mass * r * r
        # A light cylinder needs a softer contact to stay stable at this time step: the stiffness (per contact point)
        # is capped at about 3e5 N/m per kg of its mass.
        k_cyl = min(float(P['contact_stiffness']), 3.0e5 * mass)
        cyl = world.addChild(f'cylinder{j + 1}')
        cyl_dofs = cyl.addObject('MechanicalObject', name='dofs', template='Rigid3', position=[[cx, 0.0, r, 0.0, 0.0, 0.0, 1.0]])
        cyl.addObject('PartialFixedProjectiveConstraint', indices=[0], fixedDirections=[0, 1, 0, 1, 0, 1])
        cyl.addObject('UniformMass', name='mass', vertexMass=[mass, 1.0, inertia, 0, 0, 0, inertia, 0, 0, 0, inertia])
        # (applied explicitly: a coefficient c needs c dt / mass well below 1 or a light cylinder blows up, so it is capped)
        cyl.addObject('UniformVelocityDampingForceField', template='Rigid3d',
                      dampingCoefficient=min(float(P['cylinder_damping']), 0.3 * mass / DT))
        centre = cyl.addChild('centre')
        centre_dofs = centre.addObject('MechanicalObject', name='dofs', template='Vec3', position=[[cx, 0.0, r]])
        centre.addObject('RigidMapping', input='@../dofs', output='@dofs', globalToLocalCoords=True)
        centre.addObject('PlaneForceField', name='floor', normal=[0.0, 0.0, 1.0], d=r,
                         stiffness=k_cyl, damping=ROD_DAMPING)
        centres.append((centre, centre_dofs, r, k_cyl))
        vp, vt = _cylinder_y(cx, r, r, -cyl_len / 2, cyl_len / 2)
        visu = cyl.addChild('visu')
        visu.addObject('MeshTopology', name='topo', position=vp, triangles=vt)
        visu.addObject('OglModel', name='visual', src='@topo', color=palette[j % len(palette)] + [P['cylinder_alpha']])
        visu.addObject('RigidMapping', input='@../dofs', output='@visual', globalToLocalCoords=True)
        # contact with the fingers: one point on the axis per touch point (at y = 0 like the touch points)
        axis = cyl.addChild('axisPoints')
        axis_dofs = axis.addObject('MechanicalObject', name='dofs', template='Vec3', position=[[cx, 0.0, r]] * n_touch)
        axis.addObject('RigidMapping', input='@../dofs', output='@dofs', globalToLocalCoords=True,
                       rigidIndexPerPoint=[0] * n_touch)
        contact = touch.addChild(f'cylinderContact{j + 1}')
        contact.addObject('MechanicalObject', name='dofs', template='Vec3')
        contact.addObject('SubsetMultiMapping', name='mapping', input=[touch_dofs.getLinkPath(), axis_dofs.getLinkPath()],
                          output='@dofs', indexPairs=[v for q in range(n_touch) for v in (0, q)] + [v for q in range(n_touch) for v in (1, q)])
        axis.addChild(contact)
        dist = contact.addChild('distance')
        dist.addObject('MeshTopology', name='topo', edges=[[q, n_touch + q] for q in range(n_touch)])
        dist.addObject('MechanicalObject', name='dofs', template='Vec1')
        dist.addObject('DistanceMapping', name='mapping', input='@../dofs', output='@dofs', topology='@topo',
                       computeDistance=True, geometricStiffness='None')
        dist.addObject('PlaneForceField', name='contact', normal=[1.0], d=r, stiffness=k_cyl, damping=ROD_DAMPING)
        BUILD['cylinders'].append({'x': cx, 'r': r})
    for a in range(n_cyl):
        for b in range(a + 1, n_cyl):  # two cylinders can't overlap
            ca, da, ra, ka = centres[a]
            cb, db, rb, kb = centres[b]
            pair = ca.addChild(f'pair_{a + 1}_{b + 1}')
            pair.addObject('MechanicalObject', name='dofs', template='Vec3')
            pair.addObject('SubsetMultiMapping', name='mapping', input=[da.getLinkPath(), db.getLinkPath()],
                           output='@dofs', indexPairs=[0, 0, 1, 0])
            cb.addChild(pair)
            gap_node = pair.addChild('distance')
            gap_node.addObject('MeshTopology', name='topo', edges=[[0, 1]])
            gap_node.addObject('MechanicalObject', name='dofs', template='Vec1')
            gap_node.addObject('DistanceMapping', name='mapping', input='@../dofs', output='@dofs', topology='@topo',
                               computeDistance=True, geometricStiffness='None')
            gap_node.addObject('PlaneForceField', name='apart', normal=[1.0], d=ra + rb,
                               stiffness=min(ka, kb), damping=ROD_DAMPING)

    rootNode.addObject(FlangeController(rootNode, body_dofs, flange_pos, midplanes,
                                        acceleration=float(P['flange_acceleration']), speed=float(P['flange_speed']),
                                        name='Flange'))
    return rootNode


def main():
    import Sofa.Simulation
    os.environ.setdefault('GRIPPER3_NO_EDITOR', '1')
    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)
    for _ in range(int((PARAMS['ramp_start'] + PARAMS['ramp_time'] + 2.0) / DT)):
        Sofa.Simulation.animate(root, DT)
    Sofa.Simulation.unload(root)


if __name__ == '__main__':
    main()
