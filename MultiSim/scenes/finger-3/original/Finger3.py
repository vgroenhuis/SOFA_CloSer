# -*- coding: utf-8 -*-
"""
Finger 3: one chamber of a single-channel PneuNet soft finger, inflated
with air. Successor of ../../better-finger/original/BetterFinger.py.

What changed compared to Better finger:

1. Geometry in millimetres. Every feature (box, each wall, channel, duct,
   trunk, symmetry buffer) is given in mm, plus a standard element size.
   Each feature is cut into the whole number of equal elements closest to
   that size (an 18 mm wall with 10 mm elements: 2 x 9 mm), see
   finger3_mesh.py. Walls can now be several elements thick, which a
   linear-tetrahedron model needs to bend realistically.

2. The far-end symmetry plane lets its nodes slide. The far end of the
   channel duct is the midplane between this chamber and its (mirrored)
   neighbour. A symmetry plane only fixes the displacement normal to it:
   the nodes on it stay on the plane but are free to move within it. Better
   finger glued them rigidly to the plane's frame (RigidMapping), which
   also froze the duct's cross-section there: the channel couldn't widen
   at the far end under pressure, and the plane's tilt was underestimated.
   Here each far-end node is held on the plane by the signed volume of a
   tetrahedron made of the node and three points of the plane (zero volume
   = on the plane; VolumeMapping plus a stiff spring), which leaves its two
   in-plane directions free and passes the reaction force on to the plane.
   The old rigid behaviour is still available ("Far-end nodes slide in the
   plane" off) for comparison.

3. Contact with the neighbouring chamber pushes back. The guard that
   stops the end wall from bulging through the far-end plane (where in
   reality it would hit the neighbour's mirrored wall) used to be a
   PlaneForceField moved along by a controller: it pushed the wall back but
   never pushed the plane, so wall-to-wall contact couldn't bend the finger.
   It now uses the same signed volumes, one-sided, so contact acts on both
   the wall and the plane.

4. Optional, closer to a real PneuNet (both off by default, to keep the
   comparison with Better finger direct):
   - a strain-limiting layer (paper, a membrane on the bottom face) instead
     of clamping the bottom to a rigid trunk; the chamber's base is then
     free to bend, held only by its symmetry planes;
   - neo-Hookean hyperelastic silicone instead of linear (corotational)
     elasticity, for large strains.

Units: the scene is SI (metres, kg, s, Pa); params.json takes millimetres
for lengths, as labelled.

Run with the GUI:   runSofa -l SofaPython3 -g glfw Finger3.py
Headless:           python Finger3.py
Parameters: params.json next to this script (edit by hand, or with
params_editor.py, which shows the resulting element count).
"""
import itertools
import json
import math
import os
import subprocess
import sys

import Sofa
import Sofa.Core

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import importlib  # noqa: E402
import finger3_mesh  # noqa: E402
importlib.reload(finger3_mesh)  # the GUI's Reload re-imports this file; pick up mesher edits too

PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')

_DEFAULT_PARAMS = {
    # -- geometry [mm] --
    'element_size': 8.0,         # standard element edge; each feature is split into
                                 # the nearest whole number of equal elements
    'min_wall_layers': 2,        # ... but every wall into at least this many layers
    'length': 32.0,              # chamber, outer extent along X (the finger's axis)
    'width': 160.0,              # along Y
    'height': 96.0,              # along Z (up)
    'wall_end': 8.0,             # end walls (perpendicular to X), facing the neighbours
    'wall_side': 8.0,            # side walls (perpendicular to Y)
    'wall_bottom': 8.0,          # bottom layer
    'wall_top': 8.0,             # roof
    'channel_height': 8.0,       # connecting channel, from the cavity floor up; 0 = closed
    'channel_width': 0.0,        # 0 = the full cavity width (as in Better finger)
    'channel_wall': 8.0,         # duct roof (and, for a narrower channel, its sides)
    'channel_separation': 8.0,   # gap between this chamber and the next, i.e. the
                                 # duct length between them; the model ends halfway
    'trunk_length': 16.0,        # clamped footprint of the base (centred)
    'trunk_width': 160.0,
    'floor_size': 400.0,         # green floor, purely visual
    'symmetry_buffer': 0.0,      # extra FEM material past an exploited midplane (unloaded:
                                 # its cavity walls get no pressure, its far edge is free;
                                 # kept for comparison with Better finger, 0 is exact)
    # -- boundary conditions --
    'symmetry_x': True,
    'symmetry_y': True,
    'prevent_symmetry_crossing': True,
    'channel_end_constraint': True,   # far-end plane: may only shift along X and tilt about Y
    'channel_end_sliding': True,      # ... and its nodes slide within it (False: glued, as Better finger)
    'prevent_channel_end_crossing': True,  # end wall can't pass the far-end plane (contact)
    'clamp_to_trunk': True,           # bottom clamped to a rigid trunk over its footprint
    'strain_limiting_layer': False,   # paper membrane on the bottom face
    'strain_limiting_stiffness': 650000.0,  # [N/m] membrane E*t: paper 6.5 GPa x 0.1 mm
    # -- material --
    'hyperelastic': False,       # neo-Hookean instead of linear corotational
    'young_modulus': 1.0e6,      # [Pa]
    'poisson_ratio': 0.45,
    'density': 1000.0,           # [kg/m^3]
    'gravity_enabled': False,
    # -- pressure --
    'target_pressure': 30000.0,  # [Pa] (30 kPa: a typical PneuNet pressure)
    'ramp_time': 0.2,            # [s]
    'instant': False,
    # -- visuals --
    'cutaway': False,
    'interior_cutaway': False,
    'wall_alpha': 1.0,
    'interior_alpha': 1.0,
    'channel_end_alpha': 0.5,
    'mirror_visual_x': False,
    'mirror_visual_y': False,
    'show_pressure_overlay': False,
    'show_fem_elements': False,
    'theme': 'System',
}

# Hard limits, so a typo in params.json can't make a mesh nobody can run.
MAX_ELEMENTS = 200000


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
    interpreter). Skipped when FINGER3_NO_EDITOR is set (headless runs)."""
    if os.environ.get('FINGER3_NO_EDITOR'):
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
DT = 0.01
# Far-end plane coupling: stiffness per node normal to the plane [N/m], for
# both the sliding constraint (bilateral) and the wall contact (one-sided).
PLANE_STIFFNESS = 1e9
PLANE_DAMPING = 1e4  # [N s/m] per node, for the one-sided contact (keeps it from chattering)
# Size of the triangle spanning each end plane (for the signed volumes).
PLANE_SPAN = 0.02


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


def _flat_rect_grid(y0, y1, z0, z1, cell):
    """Flat grid at local x = 0 (the end-plane indicator)."""
    ny_ = max(1, round((y1 - y0) / cell))
    nz_ = max(1, round((z1 - z0) / cell))
    points = [[0.0, y0 + (y1 - y0) * j / ny_, z0 + (z1 - z0) * k / nz_]
              for j in range(ny_ + 1) for k in range(nz_ + 1)]
    tris = []
    for j in range(ny_):
        for k in range(nz_):
            a, b = j * (nz_ + 1) + k, (j + 1) * (nz_ + 1) + k
            tris += [[a, b, b + 1], [a, b + 1, a + 1]]
    return points, tris


def _mesh_volume(positions, tetrahedra):
    total = 0.0
    for t in tetrahedra:
        p0, p1, p2, p3 = (positions[i] for i in t)
        a = [p1[k] - p0[k] for k in range(3)]
        b = [p2[k] - p0[k] for k in range(3)]
        c = [p3[k] - p0[k] for k in range(3)]
        total += abs(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
                     a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
    return total


def _signed_volume(a, b, c, d):
    u = [b[k] - a[k] for k in range(3)]
    v = [c[k] - a[k] for k in range(3)]
    w = [d[k] - a[k] for k in range(3)]
    return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
            u[2] * (v[0] * w[1] - v[1] * w[0])) / 6.0


def _rotate(q, v):
    """Quaternion [x, y, z, w] applied to vector v."""
    ux, uy, uz, w = q
    cx, cy, cz = uy * v[2] - uz * v[1], uz * v[0] - ux * v[2], ux * v[1] - uy * v[0]
    return [v[0] + 2 * (w * cx + uy * cz - uz * cy),
            v[1] + 2 * (w * cy + uz * cx - ux * cz),
            v[2] + 2 * (w * cz + ux * cy - uy * cx)]


# ----------------------------------------------------------------------
# Controllers
# ----------------------------------------------------------------------
class MirrorController(Sofa.Core.Controller):
    """Copies a visual surface, reflected across symmetry planes, every
    step (the desktop GUI's mirror_visual_x/y; the web viewer mirrors in the
    browser instead)."""

    def __init__(self, source, target, axes, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.source, self.target, self.axes = source, target, axes

    def onAnimateEndEvent(self, event):
        out = []
        for p in self.source.position.value:
            q = list(p)
            for axis in self.axes:
                q[axis] = -q[axis]
            out.append(q)
        self.target.position.value = out


class FramePlaneController(Sofa.Core.Controller):
    """Places the far-end plane indicator on the frame that holds the plane
    (rest-pose points carried along by the frame's motion)."""

    def __init__(self, frame_dofs, index, target, rest_points, rest_origin, offset, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.frame_dofs, self.index, self.target = frame_dofs, index, target
        self.local = [[p[k] - rest_origin[k] for k in range(3)] for p in rest_points]
        self.offset = offset

    def onAnimateEndEvent(self, event):
        pose = [float(c) for c in self.frame_dofs.position.value[self.index]]
        q = pose[3:7]
        normal = _rotate(q, [1.0, 0.0, 0.0])
        out = []
        for v in self.local:
            r = _rotate(q, v)
            out.append([pose[k] + r[k] + self.offset * normal[k] for k in range(3)])
        self.target.position.value = out


class NodePlaneController(Sofa.Core.Controller):
    """Indicator for a far-end plane that no frame holds (end constraint
    off): fitted through three of its nodes."""

    def __init__(self, dofs, anchors, target, rest_points, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.dofs, self.anchors, self.target = dofs, anchors, target
        pos = self.dofs.position.value
        a, b, c = (list(pos[i]) for i in anchors)
        self.basis0 = self._basis(a, b, c)
        o, u, v, n = self.basis0
        self.local = [([p[k] - o[k] for k in range(3)]) for p in rest_points]
        self.local = [(sum(d[k] * u[k] for k in range(3)), sum(d[k] * v[k] for k in range(3))) for d in self.local]

    @staticmethod
    def _basis(a, b, c):
        o = [(a[k] + b[k] + c[k]) / 3 for k in range(3)]
        e1 = [b[k] - a[k] for k in range(3)]
        e2 = [c[k] - a[k] for k in range(3)]
        n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
        nn = math.sqrt(sum(x * x for x in n))
        n = [x / nn for x in n]
        y = [0.0, 1.0, 0.0]
        d = sum(y[k] * n[k] for k in range(3))
        u = [y[k] - d * n[k] for k in range(3)]
        un = math.sqrt(sum(x * x for x in u))
        u = [x / un for x in u]
        v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]]
        return o, u, v, n

    def onAnimateEndEvent(self, event):
        pos = self.dofs.position.value
        o, u, v, n = self._basis(*(list(pos[i]) for i in self.anchors))
        self.target.position.value = [[o[k] + a * u[k] + b * v[k] for k in range(3)] for a, b in self.local]


class PressureRamp(Sofa.Core.Controller):
    """Drives the cavity pressure towards a live-editable target, ramped or
    instantly (`targetPressure`, `rampTime`, `instant` are Data fields, so
    they can be edited in the GUI while running).

    SurfacePressureConstraint is solved as a Lagrangian constraint by the
    FreeMotionAnimationLoop, and what it imposes is an impulse per time
    step, not a force: a value v acts as a pressure v / dt. (Checked: value
    300 at dt = 0.01 s deforms the chamber exactly like a true 30 kPa from
    SurfacePressureForceField, and halving dt doubled the deformation.) So
    the pressure in Pa is multiplied by dt here, and `applied_pressure`
    divides it back out."""

    def __init__(self, root, pressure_constraint, target_pressure, ramp_time, instant=False, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.root = root
        self.pressure = pressure_constraint
        self.addData(name='targetPressure', type='float', value=target_pressure,
                     help='Target gauge pressure in the cavity [Pa]. Live-editable.')
        self.addData(name='rampTime', type='float', value=ramp_time,
                     help='Time [s] to go from 0 to targetPressure when not instant.')
        self.addData(name='instant', type='bool', value=instant,
                     help='Apply targetPressure immediately instead of ramping towards it.')
        self.current_pressure = 0.0
        self.time = 0.0
        self.next_print = 0.0

    def onAnimateBeginEvent(self, event):
        dt = self.root.dt.value
        self.time += dt
        target = self.targetPressure.value
        if self.instant.value:
            self.current_pressure = target
        else:
            ramp_time = max(self.rampTime.value, 1e-6)
            max_step = max((abs(target) if target != 0 else abs(self.current_pressure)) / ramp_time * dt, 1e-9)
            diff = target - self.current_pressure
            self.current_pressure = target if abs(diff) <= max_step else \
                self.current_pressure + (max_step if diff > 0 else -max_step)
        self.pressure.value.value = [self.current_pressure * dt]

    def applied_pressure(self):
        """The pressure acting on the cavity walls [Pa]."""
        return float(self.pressure.pressure.value) / self.root.dt.value

    def onAnimateEndEvent(self, event):
        if self.time + 1e-9 < self.next_print:
            return
        self.next_print += 0.2
        vol = self.pressure.cavityVolume.value
        vol0 = self.pressure.initialCavityVolume.value
        print(f"t={self.time:5.2f}s  target={self.targetPressure.value / 1000.0:6.2f} kPa  "
              f"P_applied={self.applied_pressure() / 1000.0:6.2f} kPa  "
              f"cavity volume={vol * 1e6:8.2f} cm^3  (volume {100 * (vol / vol0 - 1):+5.1f}%)")


# ----------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------
def createScene(rootNode):
    launch_params_editor()
    P = PARAMS

    plugins = [
        "Sofa.Component.AnimationLoop", "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Solver", "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Engine.Select", "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Mapping.Linear", "Sofa.Component.Mapping.NonLinear", "Sofa.Component.Mass",
        "Sofa.Component.MechanicalLoad", "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.SolidMechanics.FEM.Elastic", "Sofa.Component.SolidMechanics.FEM.HyperElastic",
        "Sofa.Component.SolidMechanics.Spring", "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Constant", "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Setting", "Sofa.Component.Visual", "Sofa.GL.Component.Rendering3D",
        "Sofa.GL.Component.Shader", "SoftRobots",
    ]
    for p in plugins:
        rootNode.addObject('RequiredPlugin', name='req_' + p.replace('.', '_'), pluginName=p)

    # ---- mesh -----------------------------------------------------------
    n_tets, n_nodes, geo = finger3_mesh.count_elements(P)
    if n_tets > MAX_ELEMENTS:
        raise ValueError(f"{n_tets} elements (more than {MAX_ELEMENTS}): use a larger element size")
    base_mm = max(4.0, min(geo.h, 10.0))  # the chamber stands this far above the floor (trunk)
    cutaway = {(0, -1, 0)} if P['cutaway'] else set()
    inner_cutaway = {(0, -1, 0)} if P['interior_cutaway'] else set()
    mesh = finger3_mesh.build(P, z_offset_mm=base_mm, outer_skip_normals=cutaway,
                              inner_skip_normals=inner_cutaway)
    g = mesh['geometry']
    positions, tetrahedra = mesh['positions'], mesh['tetrahedra']
    print(f"[mesh] {len(tetrahedra)} tetrahedra, {len(positions)} nodes; {g.summary()}")

    L, W, H = g.L * MM, g.W * MM, g.H * MM
    BASE = base_mm * MM
    sym_x, sym_y = g.symmetry_x, g.symmetry_y
    total_mass = P['density'] * _mesh_volume(positions, tetrahedra)

    rootNode.addObject('BackgroundSetting', color=[0.78, 0.87, 0.96, 1.0], image="")
    rootNode.addObject('VisualStyle', displayFlags=(
        'showVisualModels hideBehaviorModels hideCollisionModels hideBoundingCollisionModels '
        + ('showForceFields ' if P['show_fem_elements'] else 'hideForceFields ')
        + ('showInteractionForceFields ' if P['show_pressure_overlay'] else 'hideInteractionForceFields ')
        + 'hideWireframe'))

    # Camera (Z-up world, so the orientation is set explicitly).
    size = max(L, W, H)
    cam_lookat = [0.0, 0.0, BASE + H * 0.5]
    cam_dir = (4.5, -4.5, 3.25)
    norm = math.sqrt(sum(c * c for c in cam_dir))
    cam_distance = 2.741 * size
    cam_position = [l + cam_distance * c / norm for l, c in zip(cam_lookat, cam_dir)]
    rootNode.addObject('InteractiveCamera', name='camera', position=cam_position, distance=cam_distance,
                       orientation=_look_at_quaternion(cam_position, cam_lookat), activated=True)
    rootNode.addObject('LightManager', ambient=[0.35, 0.35, 0.35, 1.0])
    rootNode.addObject('DirectionalLight', direction=[-0.5, -0.15, 1.0])
    rootNode.addObject('DirectionalLight', color=[0.35, 0.35, 0.35, 1.0], direction=[0.5, 0.2, 0.3])

    # Floor, trunk, axes: purely visual.
    half = P['floor_size'] * MM / 2
    floor = rootNode.addChild('floor')
    floor.addObject('MeshTopology', name='topo', triangles=[[0, 1, 2], [0, 2, 3]],
                    position=[[-half, -half, -1e-4], [half, -half, -1e-4], [half, half, -1e-4], [-half, half, -1e-4]])
    floor.addObject('OglModel', name='visual', src='@topo', color=[0.08, 0.28, 0.1, 1.0])

    if P['clamp_to_trunk']:
        tx0, tx1 = (c * MM for c in g.trunk_x)
        ty0, ty1 = (c * MM for c in g.trunk_y)
        pts, tris = _flat_shaded(*_box(tx0, ty0, 0.0, tx1, ty1, BASE - 1e-4))
        trunk = rootNode.addChild('trunk')
        trunk.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        trunk.addObject('OglModel', name='visual', src='@topo', color=[0.4, 0.26, 0.13, 1.0])

    arrow_ref = 0.6 * size
    head_len, shaft_half = arrow_ref * 0.2, arrow_ref * 0.015
    totals = {0: max(arrow_ref, 1.6 * (L / 2 + g.half_sep * MM) + head_len),
              1: max(arrow_ref, 1.6 * (W / 2) + head_len)}
    for name, axis, color in (('arrow_x', 0, [0.9, 0.1, 0.1, 1.0]), ('arrow_y', 1, [0.1, 0.8, 0.1, 1.0])):
        pts, tris = _axis_arrow(axis, totals[axis] - head_len, head_len, shaft_half, shaft_half * 3)
        pts, tris = _flat_shaded([[p[0], p[1], p[2] + 1e-4] for p in pts], tris)
        node = rootNode.addChild(name)
        node.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        node.addObject('OglModel', name='visual', src='@topo', color=color)
    lh = arrow_ref * 0.09
    gap = arrow_ref * 0.08
    for name, strokes, color in (
        ('label_x', [(totals[0] + gap, -lh, totals[0] + gap + 2 * lh, lh),
                     (totals[0] + gap, lh, totals[0] + gap + 2 * lh, -lh)], [0.9, 0.1, 0.1, 1.0]),
        ('label_y', [(-lh, totals[1] + gap + 2 * lh, 0.0, totals[1] + gap + lh),
                     (lh, totals[1] + gap + 2 * lh, 0.0, totals[1] + gap + lh),
                     (0.0, totals[1] + gap + lh, 0.0, totals[1] + gap)], [0.1, 0.8, 0.1, 1.0])):
        pts, tris = _flat_shaded(*_letter(strokes, lh * 0.125, 1e-4, 1e-4 + shaft_half * 2))
        node = rootNode.addChild(name)
        node.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        node.addObject('OglModel', name='visual', src='@topo', color=color)

    rootNode.gravity = [0.0, 0.0, -9.81] if P['gravity_enabled'] else [0.0, 0.0, 0.0]
    rootNode.dt = DT
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('BlockGaussSeidelConstraintSolver', maxIterations=500, tolerance=1e-8)

    # ---- far-end planes ---------------------------------------------------
    # One Rigid3 frame per far-end plane (the midplane to the neighbouring
    # chamber, at the duct's end), free only to shift along X and to rotate
    # about Y. Its origin is the plane's rest centre; its local X is the
    # plane's normal.
    eps = 1e-7
    far_sets = []  # (side, node indices, far x [m])
    if g.has_duct:
        far_sets.append(('pos', mesh['far_nodes'], g.far_x * MM))
        if mesh['far_nodes_neg']:
            far_sets.append(('neg', mesh['far_nodes_neg'], -g.far_x * MM))
    use_frames = bool(far_sets) and P['channel_end_constraint']
    sliding = use_frames and P['channel_end_sliding']

    finger = rootNode.addChild('finger')
    # Light stiffness damping: the quasi-static end state is the same, but
    # with 0.1 (Better finger's value) a soft mode -- the full-width
    # channel's end-wall flap -- took ~7 s to creep to it instead of ~1.5 s.
    finger.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=0.01, rayleighMass=0.1)
    finger.addObject('SparseLDLSolver', name='linearSolver', template='CompressedRowSparseMatrixd')

    frames = []
    for side, ring, x in far_sets:
        c = [sum(positions[i][k] for i in ring) / len(ring) for k in range(3)]
        frames.append([c[0], c[1], c[2], 0.0, 0.0, 0.0, 1.0])

    rigid_dofs = plane_dofs = None
    if use_frames:
        rigid = finger.addChild('rigidEnds')
        rigid_dofs = rigid.addObject('MechanicalObject', name='dofs', template='Rigid3', position=frames)
        # fixedDirections [tx, ty, tz, rx, ry, rz]: 1 = held.
        rigid.addObject('PartialFixedProjectiveConstraint', name='endPlaneMotion',
                        indices=list(range(len(frames))), fixedDirections=[0, 1, 1, 1, 0, 1])
        rigid.addObject('UniformMass', totalMass=1e-4 * len(frames))
        # Three points spanning each plane: frame origin, +Y and +Z in the plane.
        planes = rigid.addChild('planePoints')
        world = []
        for fr in frames:
            world += [[fr[0], fr[1], fr[2]], [fr[0], fr[1] + PLANE_SPAN, fr[2]], [fr[0], fr[1], fr[2] + PLANE_SPAN]]
        plane_dofs = planes.addObject('MechanicalObject', name='dofs', template='Vec3', position=world)
        planes.addObject('RigidMapping', name='mapping', input='@../dofs', output='@dofs', globalToLocalCoords=True,
                         rigidIndexPerPoint=[f for f in range(len(frames)) for _ in range(3)])

    # ---- the silicone body ----------------------------------------------
    if use_frames and not sliding:
        # Better finger's glued far ends: ring nodes rigidly mapped from the
        # frame; the cube re-assembles all nodes with a SubsetMultiMapping.
        frame_of = {}
        for f, (side, ring, x) in enumerate(far_sets):
            for i in ring:
                frame_of[i] = f
        ring_nodes = sorted(frame_of)
        free_nodes = [i for i in range(len(positions)) if i not in frame_of]
        ring_node = rigid.addChild('ringNodes')
        ring_dofs = ring_node.addObject('MechanicalObject', name='dofs', template='Vec3',
                                        position=[positions[i] for i in ring_nodes])
        ring_node.addObject('RigidMapping', name='mapping', input='@../dofs', output='@dofs',
                            globalToLocalCoords=True, rigidIndexPerPoint=[frame_of[i] for i in ring_nodes])
        free = finger.addChild('freeNodes')
        free_dofs = free.addObject('MechanicalObject', name='dofs', template='Vec3',
                                   position=[positions[i] for i in free_nodes])
        cube = free.addChild('cube')
        ring_node.addChild(cube)
        projective_node = free
    else:
        cube = finger.addChild('cube')
        projective_node = cube

    cube.addObject('TetrahedronSetTopologyContainer', name='container', position=positions, tetrahedra=tetrahedra)
    cube.addObject('TetrahedronSetTopologyModifier')
    cube_dofs = cube.addObject('MechanicalObject', name='dofs', template='Vec3', position=positions)
    if use_frames and not sliding:
        free_index = {n: k for k, n in enumerate(free_nodes)}
        ring_index = {n: k for k, n in enumerate(ring_nodes)}
        pairs = []
        for i in range(len(positions)):
            pairs += [0, free_index[i]] if i in free_index else [1, ring_index[i]]
        cube.addObject('SubsetMultiMapping', name='mapping', input=[free_dofs.getLinkPath(), ring_dofs.getLinkPath()],
                       output='@dofs', indexPairs=pairs)
    cube.addObject('UniformMass', totalMass=total_mass)

    E, nu = P['young_modulus'], P['poisson_ratio']
    if P['hyperelastic']:
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        cube.addObject('TetrahedronHyperelasticityFEMForceField', name='FEM', materialName='NeoHookean',
                       ParameterSet=f"{mu} {lam}")
    else:
        cube.addObject('FastTetrahedralCorotationalForceField', name='FEM', method='large',
                       youngModulus=E, poissonRatio=nu)

    # ---- base -----------------------------------------------------------
    sym_eps = 1e-6
    big = 10 * size
    if P['clamp_to_trunk']:
        tx0, tx1 = (c * MM for c in g.trunk_x)
        ty0, ty1 = (c * MM for c in g.trunk_y)
        cube.addObject('BoxROI', name='baseROI', drawBoxes=False, position='@dofs.rest_position',
                       box=[tx0 - sym_eps, ty0 - sym_eps, BASE - sym_eps, tx1 + sym_eps, ty1 + sym_eps, BASE + sym_eps])
        cube.addObject('RestShapeSpringsForceField', points='@baseROI.indices', stiffness=1e12)
    else:
        # Free base: only rigid-body motion is removed, with conditions that
        # hold anyway by symmetry -- the X and Y midplanes stay put in their
        # normal direction (below), and the bottom edge at x = 0 keeps its
        # height.
        projective_node.addObject('BoxROI', name='zHoldROI', drawBoxes=False, position='@dofs.rest_position',
                                  box=[-sym_eps, -big, BASE - sym_eps, sym_eps, big, BASE + sym_eps])
        projective_node.addObject('PartialFixedProjectiveConstraint', name='zHold',
                                  indices='@zHoldROI.indices', fixedDirections=[0, 0, 1])

    if P['strain_limiting_layer'] and mesh['bottom']:
        # Paper glued to the bottom face: a membrane, stiff in its plane,
        # with no bending stiffness of its own. SOFA's triangle FEM has no
        # thickness parameter (its stiffness is per unit thickness), so its
        # Young's modulus is the membrane stiffness E*t [N/m].
        used = sorted({i for t in mesh['bottom'] for i in t})
        local = {n: k for k, n in enumerate(used)}
        paper = cube.addChild('strainLimitingLayer')
        paper.addObject('TriangleSetTopologyContainer', name='topo', position=[positions[i] for i in used],
                        triangles=[[local[i] for i in t] for t in mesh['bottom']])
        paper.addObject('MechanicalObject', name='dofs', template='Vec3')
        paper.addObject('TriangularFEMForceFieldOptim', name='membrane',
                        youngModulus=P['strain_limiting_stiffness'], poissonRatio=0.2)
        paper.addObject('SubsetMapping', name='mapping', input='@../dofs', output='@dofs', indices=used)

    # ---- symmetry midplanes -----------------------------------------------
    def plane_guard(name, box, normal):
        # One-sided stop at a fixed midplane for every node whose rest
        # position is in `box` (so the buffer past the cut isn't affected).
        cube.addObject('BoxROI', name=name + 'ROI', box=box, drawBoxes=False, position='@dofs.rest_position')
        node = cube.addChild(name)
        node.addObject('MechanicalObject', name='mstate')
        node.addObject('SubsetMapping', input='@..', output='@mstate', indices='@../' + name + 'ROI.indices')
        node.addObject('PlaneForceField', name=name + 'Plane', normal=normal, d=0.0, stiffness=1e9, damping=1e4)

    for axis, exploited in ((0, sym_x), (1, sym_y)):
        lo = [-big] * 3
        hi = [big] * 3
        cut_lo, cut_hi = list(lo), list(hi)
        cut_lo[axis], cut_hi[axis] = -sym_eps, sym_eps
        name = 'symX' if axis == 0 else 'symY'
        fixed = [1 if a == axis else 0 for a in range(3)]
        if exploited or not P['clamp_to_trunk']:
            projective_node.addObject('BoxROI', name=name + 'ROI', box=cut_lo + cut_hi, drawBoxes=False,
                                      position='@dofs.rest_position')
            projective_node.addObject('PartialFixedProjectiveConstraint', name=name,
                                      indices='@' + name + 'ROI.indices', fixedDirections=fixed)
        if P['prevent_symmetry_crossing']:
            pos_lo = list(lo)
            pos_lo[axis] = -sym_eps
            n_pos = [1.0 if a == axis else 0.0 for a in range(3)]
            plane_guard(name + 'Pos', pos_lo + hi, n_pos)
            if not exploited:
                neg_hi = list(hi)
                neg_hi[axis] = sym_eps
                plane_guard(name + 'Neg', lo + neg_hi, [-c for c in n_pos])

    # ---- far-end coupling: sliding constraint and wall contact ------------
    # A node p is on the plane through the frame's three plane points
    # (a, b, c) when the tetrahedron (p, a, b, c) has zero volume; its signed
    # volume is (triangle area / 3) x (p's signed distance to the plane).
    # VolumeMapping turns those tetrahedra into scalars; a spring holds the
    # sliding ones at 0 (bilateral) and a one-sided PlaneForceField keeps the
    # contact ones positive. Both act on the node AND, through RigidMapping,
    # on the frame -- the plane feels what holds it.
    area3 = PLANE_SPAN * PLANE_SPAN / 2 / 3
    k_volume = PLANE_STIFFNESS / area3 ** 2
    coupling = []   # (kind, node, frame)
    if use_frames:
        for f, (side, ring, x) in enumerate(far_sets):
            if sliding:
                coupling += [('slide', i, f) for i in ring]
            if P['prevent_channel_end_crossing']:
                # End-wall nodes that could reach the plane: the end wall's
                # whole thickness, all of it except the duct itself.
                wall_lo, wall_hi = (g.L / 2 - g.wx) * MM, g.L / 2 * MM
                for i, p in enumerate(positions):
                    xx = p[0] if side == 'pos' else -p[0]
                    if wall_lo - eps <= xx <= wall_hi + eps:
                        coupling.append(('contact', i, f))
    if coupling:
        nodes = sorted({i for _, i, _ in coupling})
        slot = {n: k for k, n in enumerate(nodes)}
        n_nodes = len(nodes)
        pairs = []
        for n in nodes:
            pairs += [0, n]
        for k in range(3 * len(frames)):
            pairs += [1, k]
        couple = cube.addChild('endPlaneCoupling')
        couple.addObject('MechanicalObject', name='dofs', template='Vec3')
        couple.addObject('SubsetMultiMapping', name='mapping', input=[cube_dofs.getLinkPath(), plane_dofs.getLinkPath()],
                         output='@dofs', indexPairs=pairs)
        planes.addChild(couple)
        plane_pts_rest = [list(plane_dofs.position.value[k]) for k in range(3 * len(frames))]

        def tets_for(kind):
            out = []
            for kk, i, f in coupling:
                if kk != kind:
                    continue
                a, b, c = n_nodes + 3 * f, n_nodes + 3 * f + 1, n_nodes + 3 * f + 2
                tet = [slot[i], a, b, c]
                pa, pb, pc = plane_pts_rest[3 * f], plane_pts_rest[3 * f + 1], plane_pts_rest[3 * f + 2]
                if kind == 'contact' and _signed_volume(positions[i], pa, pb, pc) < 0:
                    tet = [slot[i], a, c, b]  # positive on the chamber's side
                out.append(tet)
            return out

        for kind in ('slide', 'contact'):
            tets = tets_for(kind)
            if not tets:
                continue
            node = couple.addChild(kind)
            node.addObject('MeshTopology', name='topo', tetrahedra=tets)
            node.addObject('MechanicalObject', name='dofs', template='Vec1')
            node.addObject('VolumeMapping', name='mapping', input='@../dofs', output='@dofs', topology='@topo')
            if kind == 'slide':
                node.addObject('RestShapeSpringsForceField', name='onPlane', stiffness=k_volume)
            else:
                node.addObject('PlaneForceField', name='noCrossing', normal=[1.0], d=0.0,
                               stiffness=k_volume, damping=PLANE_DAMPING / area3 ** 2)

    finger.addObject('GenericConstraintCorrection', name='constraintCorrection')

    # ---- cavity (pressure) --------------------------------------------------
    cavity_pts, cavity_tris = mesh['cavity']
    cavity = cube.addChild('cavity')
    cavity.addObject('MeshTopology', name='topo', position=cavity_pts, triangles=cavity_tris)
    cavity.addObject('MechanicalObject', name='cavityDofs', src='@topo')
    pressure = cavity.addObject('SurfacePressureConstraint', name='pressure', triangles='@topo.triangles',
                                valueType='pressure', value=0)
    cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # ---- visuals --------------------------------------------------------------
    flat_outer = _flat_shaded(*mesh['outer'])
    visu = cube.addChild('visu')
    visu.addObject('MeshTopology', name='topo', position=flat_outer[0], triangles=flat_outer[1])
    visu_ogl = visu.addObject('OglModel', name='visual', src='@topo', color=[0.15, 0.45, 0.95, P['wall_alpha']])
    visu.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    flat_inner = _flat_shaded(*mesh['inner'])
    interior = cube.addChild('interior_visu')
    interior.addObject('MeshTopology', name='topo', position=flat_inner[0], triangles=flat_inner[1])
    interior_ogl = interior.addObject('OglModel', name='visual', src='@topo',
                                      color=[0.95, 0.6, 0.1, P['interior_alpha']])
    interior.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # Far-end plane indicator: a flat grid one element beyond the duct on
    # each side, up to above the chamber's roof; carried by the frame.
    cell = g.h * MM
    y_lo = (-g.buffer_y * MM if sym_y else -W / 2) - cell
    for f, (side, ring, x) in enumerate(far_sets):
        pts, tris = _flat_rect_grid(y_lo, W / 2 + cell, BASE - cell, BASE + H + cell, cell)
        pts = [[x, p[1], p[2]] for p in pts]
        node = rootNode.addChild('channelEndPlane' + side.capitalize())
        node.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        mstate = node.addObject('MechanicalObject', name='mstate', src='@topo')
        node.addObject('OglModel', name='visual', src='@topo', color=[0.85, 0.1, 0.85, P['channel_end_alpha']],
                       cullFace=0)
        node.addObject('IdentityMapping', input='@mstate', output='@visual')
        outward = 1.0 if side == 'pos' else -1.0
        if use_frames:
            node.addObject(FramePlaneController(rigid_dofs, f, mstate, pts, frames[f][:3], outward * 1e-4,
                                                name='planeCtrl'))
        else:
            ys = [positions[i][1] + positions[i][2] for i in ring]
            anchors = (ring[ys.index(min(ys))], ring[ys.index(max(ys))],
                       min(ring, key=lambda i: positions[i][1] - positions[i][2]))
            node.addObject(NodePlaneController(cube_dofs, anchors, mstate, pts, name='planeCtrl'))

    # Desktop-GUI mirroring across the exploited midplanes.
    axes = [a for a, on in ((0, sym_x and P['mirror_visual_x']), (1, sym_y and P['mirror_visual_y'])) if on]
    for r in range(1, len(axes) + 1):
        for combo in itertools.combinations(axes, r):
            suffix = ''.join('XYZ'[a] for a in combo)
            for name, (pts, tris), ogl, color in (
                    ('visu_mirror_', flat_outer, visu_ogl, [0.15, 0.45, 0.95, P['wall_alpha']]),
                    ('interior_visu_mirror_', flat_inner, interior_ogl, [0.95, 0.6, 0.1, P['interior_alpha']])):
                m_pts = [[-q[a] if a in combo else q[a] for a in range(3)] for q in pts]
                m_tris = [[t[0], t[2], t[1]] for t in tris] if len(combo) % 2 else tris
                node = rootNode.addChild(name + suffix)
                node.addObject('MeshTopology', name='topo', position=m_pts, triangles=m_tris)
                mstate = node.addObject('MechanicalObject', name='mstate', src='@topo')
                node.addObject('OglModel', name='visual', src='@topo', color=color)
                node.addObject('IdentityMapping', input='@mstate', output='@visual')
                node.addObject(MirrorController(ogl, mstate, combo, name='mirrorCtrl'))

    rootNode.addObject(PressureRamp(rootNode, pressure, P['target_pressure'], P['ramp_time'],
                                    instant=P['instant'], name='PressureRamp'))
    return rootNode


def main():
    import Sofa.Simulation
    os.environ.setdefault('FINGER3_NO_EDITOR', '1')
    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)
    for _ in range(int((PARAMS['ramp_time'] + 2.0) / DT)):
        Sofa.Simulation.animate(root, DT)
    Sofa.Simulation.unload(root)


if __name__ == '__main__':
    main()
