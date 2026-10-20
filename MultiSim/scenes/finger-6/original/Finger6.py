# -*- coding: utf-8 -*-
"""
Finger 6: Finger on a rod. A copy of Finger 5: Full finger
(../../finger-5/original/Finger5.py) with a rigid round rod below the finger,
which the finger curls onto, and a simplified, fast contact model for it (see
"Rod contact" below). Finger 5 itself is a copy of Finger 4 with strain-limiting
layer (../../finger-4/original/Finger4.py) that simulates the whole finger:
`segments` chambers in a row along X, joined by their connecting channel,
all pressurised together. Only the Y midplane (the finger's own plane of
symmetry, through its length) is used as a symmetry plane; the X midplane
and the far-end plane are not, so every chamber can deform in its own way
(the first one is held by the mount, the last one has a free end).

Every chamber is meshed exactly as Finger 4's single chamber (the same
grid, mirrored about the chamber's centre), and the duct between two
chambers is the same duct, now in one piece.

Mount: the proximal end face (x = -length / 2) is clamped, as a finger
held in a fixture; optionally (or instead) the base of the first chamber
on a rigid trunk, as in Finger 4. The finger lies horizontally and curls
downwards (towards its base) when inflated, so its base is held high above the
floor (z = 0, the world origin), on a trunk standing on the floor (`base_height`).

Rod contact. The rod is a fixed, rigid cylinder with its axis along Y, at
(`rod_x`, `rod_z`) in the XZ plane (world mm, z = 0 is the floor) with radius `rod_radius`. Collisions with it use
a cheap penalty model: the nodes that can touch the rod -- by default the
nodes of the finger's underside (the base, under the strain-limiting layer),
optionally every node of the outer skin -- each get their distance to the
rod's centreline (the point of the axis at the node's own y: the rod is
straight, so that is exact) from a DistanceFromTargetMapping; a one-sided
spring (PlaneForceField on that distance, offset by the radius, with a little
damping) pushes a node back out when it is closer than the radius. As the rod
is rigid and fixed, the penalty acts on the node only (no counter-force
needed), and the model costs next to nothing: every node has its own
independent 1-dimensional constraint, the matrix gets no extra coupling, and
a node's contact doesn't depend on mesh size beyond the node spacing. It has
no friction, and a node slips between two contact nodes only as far as the
element spacing allows (the underside's nodes are 1.5-2.5 mm apart, so
a rod of 10 mm radius is sampled by several nodes). The stiffness per node
(`rod_contact_stiffness`) makes penetration small compared with the elements.

Collision planes. In Finger 4 the far-end plane was the symmetry plane to
the mirrored neighbouring chamber; its end wall touching that plane was
the contact between the two chambers' end walls. Here the plane is kept
between every pair of neighbouring chambers, but it is no longer a
symmetry plane: it is a cheap collision model between the two end walls
facing each other. Each plane is a rigid frame that may shift along X and
Z and tilt about Y (Y symmetry), with
  - the duct's nodes on the plane (its cross-section halfway between the
    chambers) holding it in place, with springs of `plane_hold_stiffness`
    per node normal to the plane: the plane follows the duct's mean
    cross-section without forcing it to stay flat (a stiff value makes it
    a flat section, as Finger 4's symmetry plane);
  - both chambers' end walls kept on their own side of it (one-sided,
    signed volumes as in Finger 4), so the walls push on the plane and,
    through it, on each other -- without a node-to-surface contact search.

Everything below describes Finger 4, which this script is otherwise based on
(Finger 4's far-end plane options are replaced by the collision planes
above; the X symmetry, symmetry buffer and glued far end are gone).

Finger 4 is Finger 3 with a strain-limiting layer under the base, on by
default.

The strain-limiting layer is what makes a PneuNet bend instead of stretch:
an inextensible sheet (in the original PneuNets a piece of paper) in the
base. Default: paper, 0.1 mm thick, E = 6.5 GPa, Poisson's ratio 0.2,
750 kg/m^3 -- the values the Abaqus PneuNet models use (Soft Robotics
Toolkit tutorial; Mosadegh et al.). It is glued to the bottom face of the
base and modelled as a membrane: stiff in its own plane (membrane stiffness
E x t), with no bending stiffness of its own (a 0.1 mm sheet's E t^3 / 12 is
negligible), so it lets the base bend but not stretch. By default it is
tension-only: a sheet this thin wrinkles rather than carrying compression
(tension-only bars along the bottom face's grid edges). Optionally it is a
linear membrane that is just as stiff in compression. SOFA's triangle FEM
takes its Young's modulus per unit thickness, so E x t [N/m] is passed
(checked with a strip in tension). A sheet this thin can't be meshed with
volume elements next to 4-8 mm ones.

From Finger 3: geometry in millimetres. Every feature (box, each wall,
channel, duct, trunk) is given in mm, plus a standard element size. Each
feature is cut into the whole number of equal elements closest to that size
(an 18 mm wall with 10 mm elements: 2 x 9 mm), see finger6_mesh.py. Walls
can be several elements thick, which a linear-tetrahedron model needs to
bend realistically. Optional: neo-Hookean hyperelastic silicone instead of
linear (corotational) elasticity, for large strains.

Units: the scene is SI (metres, kg, s, Pa); params.json takes millimetres
for lengths, as labelled.

Run with the GUI:   runSofa -l SofaPython3 -g glfw Finger6.py
Headless:           python Finger6.py
Parameters: params.json next to this script (edit by hand, or with
params_editor.py, which shows the resulting element count).
"""
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
import finger6_mesh  # noqa: E402
importlib.reload(finger6_mesh)  # the GUI's Reload re-imports this file; pick up mesher edits too

PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')

_DEFAULT_PARAMS = {
    # -- geometry [mm] --
    'segments': 7,               # chambers in the finger
    'element_size': 2.0,         # standard element edge; each feature is split into
                                 # the nearest whole number of equal elements
    'min_wall_layers': 1,        # ... but every wall into at least this many layers (1: single-wall
                                 # chambers, fast; 2 bends more realistically but takes 3-4x as long)
    'length': 8.0,              # one chamber, outer extent along X (the finger's axis)
    'width': 20.0,              # along Y
    'height': 16.0,              # along Z (up)
    'wall_end': 1.5,             # end walls (perpendicular to X), facing the neighbours
    'wall_side': 1.5,            # side walls (perpendicular to Y)
    'wall_bottom': 3.0,          # bottom layer
    'wall_top': 1.5,             # roof
    'channel_height': 2.0,       # connecting channel, from the cavity floor up
    'channel_width': 2.0,        # 0 = the full cavity width
    'channel_wall': 1.5,         # duct roof thickness (the duct spans the full width)
    'channel_separation': 2.0,   # gap between neighbouring chambers, i.e. the duct length
    'trunk_length': 8.0,        # clamped footprint under the first chamber (centred)
    'trunk_width': 20.0,
    'floor_size': 120.0,         # green floor, purely visual
    'base_height': 60.0,         # [mm] the finger's base above the floor (z = 0): high enough that the curling finger clears it
    # -- boundary conditions --
    'clamp_proximal_end': True,       # the first chamber's outer end face is held (the mount)
    'clamp_to_trunk': False,          # the first chamber's base clamped to a rigid trunk
    'symmetry_y': True,
    'prevent_symmetry_crossing': True,
    'collision_planes': True,         # a collision plane between each pair of neighbouring chambers
    # -- rigid rod below the finger: axis along Y, centre at (x, z) [mm], and its contact model --
    'rod_enabled': True,
    'rod_x': 45.0,                    # [mm] along the finger (world X)
    'rod_z': 41.0,                    # [mm] height of the axis above the floor (world Z)
    'rod_radius': 10.0,               # [mm]
    'rod_contact_stiffness': 1.0e5,   # [N/m] per contact node
    'rod_contact_whole_skin': False,  # every node of the outer skin can touch it (False: the underside only)
    'contact_outer_face_only': True,  # ... checked for the end walls' outer faces only (faster)
    'plane_hold_stiffness': 1.0e4,    # [N/m] per duct node, holding each plane to the duct
    'strain_limiting_layer': True,    # inextensible sheet under the base (see the docstring)
    'strain_layer_thickness': 0.1,    # [mm] paper
    'strain_layer_young_modulus': 6.5e9,   # [Pa] paper
    'strain_layer_poisson_ratio': 0.2,
    'strain_layer_density': 750.0,    # [kg/m^3] paper
    'strain_layer_tension_only': True,  # wrinkles rather than carrying compression, like paper
    # -- material --
    'hyperelastic': False,       # neo-Hookean instead of linear corotational
    'young_modulus': 1.0e6,      # [Pa]
    'poisson_ratio': 0.45,
    'density': 1000.0,           # [kg/m^3]
    'gravity_enabled': False,
    # -- pressure --
    'target_pressure': 50000.0,  # [Pa] (30 kPa: a typical PneuNet pressure)
    'ramp_start': 0.0,           # [s] when the pressure starts to rise (e.g. after gravity has settled)
    'ramp_time': 0.5,            # [s]
    'dt': 0.01,                  # [s] simulation time step
    'instant': False,
    # -- visuals --
    'cutaway': False,
    'interior_cutaway': False,
    'wall_alpha': 1.0,
    'interior_alpha': 1.0,
    'channel_end_alpha': 0.5,    # the collision planes
    'strain_layer_alpha': 1.0,
    'rod_alpha': 1.0,
    'mirror_visual_y': True,
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
    interpreter). Skipped when FINGER6_NO_EDITOR is set (headless runs)."""
    if os.environ.get('FINGER6_NO_EDITOR'):
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
# Collision planes: contact stiffness per wall node normal to the plane
# [N/m] (one-sided), and its damping.
PLANE_STIFFNESS = 1e9
PLANE_DAMPING = 1e4  # [N s/m] per node, for the one-sided contact (keeps it from chattering)
ROD_DAMPING = 5.0    # [N s/m] per node of the rod contact
# Size of the triangle spanning each plane (for the signed volumes).
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


def _cross3(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


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


class PlaneIndicatorController(Sofa.Core.Controller):
    """Places a collision plane's indicator grid on the plane: oriented by
    the frame that holds the plane, centred on the duct's cross-section
    (the frame is free to slide within its own plane, where nothing holds
    it, so its origin needn't follow the finger)."""

    def __init__(self, frame_dofs, index, node_dofs, ring, target, rest_points, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.frame_dofs, self.index, self.node_dofs, self.ring, self.target = frame_dofs, index, node_dofs, ring, target
        pos = self.node_dofs.position.value
        c0 = [sum(float(pos[i][k]) for i in ring) / len(ring) for k in range(3)]
        self.local = [[p[k] - c0[k] for k in range(3)] for p in rest_points]

    def onAnimateEndEvent(self, event):
        pose = [float(c) for c in self.frame_dofs.position.value[self.index]]
        q = pose[3:7]
        normal = _rotate(q, [1.0, 0.0, 0.0])
        pos = self.node_dofs.position.value
        c = [sum(float(pos[i][k]) for i in self.ring) / len(self.ring) for k in range(3)]
        off = sum((c[k] - pose[k]) * normal[k] for k in range(3))
        origin = [c[k] - off * normal[k] for k in range(3)]  # the centroid, projected onto the plane
        out = []
        for v in self.local:
            r = _rotate(q, v)
            out.append([origin[k] + r[k] for k in range(3)])
        self.target.position.value = out


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

    def __init__(self, root, pressure_constraint, target_pressure, ramp_time, instant=False, ramp_start=0.0, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.root = root
        self.pressure = pressure_constraint
        self.addData(name='targetPressure', type='float', value=target_pressure,
                     help='Target gauge pressure in the cavity [Pa]. Live-editable.')
        self.addData(name='rampStart', type='float', value=ramp_start,
                     help='Time [s] at which the pressure starts to rise (0 before it); lets gravity and other '
                          'loads settle first. Live-editable.')
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
        # No pressure before the ramp starts (if it's moved later while running,
        # the pressure comes back down the same way).
        target = self.targetPressure.value if self.time >= self.rampStart.value - 1e-9 else 0.0
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
# Filled in by createScene for the sofaweb probes (and anyone inspecting a
# built scene): node sets and geometry in SOFA units.
BUILD = {}


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
    n_tets, n_nodes, geo = finger6_mesh.count_elements(P)
    if n_tets > MAX_ELEMENTS:
        raise ValueError(f"{n_tets} elements (more than {MAX_ELEMENTS}): use a larger element size or fewer chambers")
    base_mm = max(float(P['base_height']), 0.0)  # the finger's base is this far above the floor (z = 0)
    cutaway = {(0, -1, 0)} if P['cutaway'] else set()
    inner_cutaway = {(0, -1, 0)} if P['interior_cutaway'] else set()
    mesh = finger6_mesh.build(P, z_offset_mm=base_mm, outer_skip_normals=cutaway,
                              inner_skip_normals=inner_cutaway)
    g = mesh['geometry']
    positions, tetrahedra = mesh['positions'], mesh['tetrahedra']
    print(f"[mesh] {len(tetrahedra)} tetrahedra, {len(positions)} nodes; {g.summary()}")

    L, W, H = g.L * MM, g.W * MM, g.H * MM
    X0, X1 = g.x_min * MM, g.x_max * MM  # proximal and distal end faces
    finger_len = X1 - X0
    BASE = base_mm * MM
    sym_y = g.symmetry_y
    clamp_trunk = bool(P['clamp_to_trunk'])
    clamp_end = bool(P['clamp_proximal_end'])
    if not clamp_trunk and not clamp_end:
        print("[mount] neither the proximal end nor the trunk is clamped: clamping the proximal end")
        clamp_end = True
    total_mass = P['density'] * _mesh_volume(positions, tetrahedra)
    layer_on = bool(P['strain_limiting_layer']) and bool(mesh['bottom'])
    layer_t = max(0.0, float(P['strain_layer_thickness'])) * MM
    sheet_gap = max(layer_t, 0.4 * MM)  # where the sheet is drawn, below the base
    if layer_on:
        # The sheet's own mass (tiny), spread over the body with the rest.
        bottom_area = sum(0.5 * math.sqrt(sum(c * c for c in _cross3(
            [positions[t[1]][k] - positions[t[0]][k] for k in range(3)],
            [positions[t[2]][k] - positions[t[0]][k] for k in range(3)]))) for t in mesh['bottom'])
        total_mass += P['strain_layer_density'] * bottom_area * layer_t

    rootNode.addObject('BackgroundSetting', color=[0.78, 0.87, 0.96, 1.0], image="")
    rootNode.addObject('VisualStyle', displayFlags=(
        'showVisualModels hideBehaviorModels hideCollisionModels hideBoundingCollisionModels '
        + ('showForceFields ' if P['show_fem_elements'] else 'hideForceFields ')
        + ('showInteractionForceFields ' if P['show_pressure_overlay'] else 'hideInteractionForceFields ')
        + 'hideWireframe'))

    # The floor is the plane z = 0, the world's origin. The finger curls downwards (towards its
    # base), so its base is held high above it ('base_height'; a finger bent into an arc of angle
    # a drops by at most (1 - cos a) / a ~ 0.73 of its length), on a trunk standing on the floor.
    floor_z = 0.0
    rod_on = bool(P['rod_enabled']) and P['rod_radius'] > 0
    rod_cx, rod_cz, rod_r = P['rod_x'] * MM, P['rod_z'] * MM, max(P['rod_radius'], 0.0) * MM

    # Camera (Z-up world, so the orientation is set explicitly): from the
    # front-right (the mount behind it), onto the finger and the space it curls into.
    size = max(finger_len, W, H)
    cam_lookat = [(X0 + X1) / 2, 0.0, BASE + H * 0.5 - 0.3 * finger_len]  # the floor, the curled finger and the gauge above
    cam_dir = (2.6, -4.2, 1.6)
    norm = math.sqrt(sum(c * c for c in cam_dir))
    cam_distance = 2.6 * size
    cam_position = [l + cam_distance * c / norm for l, c in zip(cam_lookat, cam_dir)]
    rootNode.addObject('InteractiveCamera', name='camera', position=cam_position, distance=cam_distance,
                       orientation=_look_at_quaternion(cam_position, cam_lookat), activated=True)
    rootNode.addObject('LightManager', ambient=[0.35, 0.35, 0.35, 1.0])
    rootNode.addObject('DirectionalLight', direction=[-0.5, -0.15, 1.0])
    rootNode.addObject('DirectionalLight', color=[0.35, 0.35, 0.35, 1.0], direction=[0.5, 0.2, 0.3])

    # Floor, mount, axes: purely visual.
    half = max(P['floor_size'] * MM, 1.5 * finger_len) / 2
    fx = (X0 + X1) / 2
    floor = rootNode.addChild('floor')
    floor.addObject('MeshTopology', name='topo', triangles=[[0, 1, 2], [0, 2, 3]],
                    position=[[fx - half, -half, floor_z - 1e-4], [fx + half, -half, floor_z - 1e-4],
                              [fx + half, half, floor_z - 1e-4], [fx - half, half, floor_z - 1e-4]])
    floor.addObject('OglModel', name='visual', src='@topo', color=[0.08, 0.28, 0.1, 1.0])

    mount_parts = []
    pad = 0.1 * H
    if clamp_end:
        # The fixture holding the proximal end face (drawn across the full
        # width: only the finger itself is mirrored in the web viewer).
        mount_parts.append(_box(X0 - max(0.25 * H, 4 * MM), -W / 2 - pad, floor_z,
                                X0 - 1e-4, W / 2 + pad, BASE + H + pad))
    if clamp_trunk:
        tx0, tx1 = (c * MM for c in g.trunk_x)
        ty1 = g.trunk_y[1] * MM
        ty0 = -ty1 if sym_y else g.trunk_y[0] * MM
        top = BASE - (sheet_gap if layer_on else 0.0) - 1e-4  # under the sheet, if there is one
        mount_parts.append(_box(tx0, ty0, floor_z, tx1, ty1, top))
    pts, tris = _flat_shaded(*_merge(mount_parts))
    mount = rootNode.addChild('trunk')
    mount.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
    mount.addObject('OglModel', name='visual', src='@topo', color=[0.4, 0.26, 0.13, 1.0])

    arrow_ref = 0.6 * max(L, W, H)
    head_len, shaft_half = arrow_ref * 0.2, arrow_ref * 0.015
    totals = {0: max(arrow_ref, X1 + 0.3 * arrow_ref + head_len),
              1: max(arrow_ref, 1.6 * (W / 2) + head_len)}
    az = 1e-4  # the arrows start at the true origin, on the floor; the viewer's ruler too
    for name, axis, color in (('arrow_x', 0, [0.9, 0.1, 0.1, 1.0]), ('arrow_y', 1, [0.1, 0.8, 0.1, 1.0])):
        pts, tris = _axis_arrow(axis, totals[axis] - head_len, head_len, shaft_half, shaft_half * 3)
        pts, tris = _flat_shaded([[p[0], p[1], p[2] + az] for p in pts], tris)
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
        pts, tris = _flat_shaded(*_letter(strokes, lh * 0.125, az, az + shaft_half * 2))
        node = rootNode.addChild(name)
        node.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        node.addObject('OglModel', name='visual', src='@topo', color=color)

    rootNode.gravity = [0.0, 0.0, -9.81] if P['gravity_enabled'] else [0.0, 0.0, 0.0]
    rootNode.dt = DT
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('BlockGaussSeidelConstraintSolver', maxIterations=500, tolerance=1e-8)

    finger = rootNode.addChild('finger')
    # Light stiffness damping (Finger 2 with channel uses 0.1): the end state
    # is the same, and the run settles faster.
    finger.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=0.01, rayleighMass=0.1)
    finger.addObject('SparseLDLSolver', name='linearSolver', template='CompressedRowSparseMatrixd')

    # ---- collision planes ---------------------------------------------------
    # One Rigid3 frame per plane between neighbouring chambers, at the rest
    # centre of the duct's cross-section there; its local X is the plane's
    # normal. With Y symmetry it moves in the XZ plane only (shift along X
    # and Z, tilt about Y). Nothing holds it within its own plane (the
    # couplings below act along its normal), where its small mass leaves it
    # in place.
    rings = mesh['plane_nodes'] if P['collision_planes'] else []
    frames = []
    for ring in rings:
        c = [sum(positions[i][k] for i in ring) / len(ring) for k in range(3)]
        frames.append([c[0], c[1], c[2], 0.0, 0.0, 0.0, 1.0])

    rigid_dofs = plane_dofs = planes = None
    if frames:
        rigid = finger.addChild('collisionPlanes')
        rigid_dofs = rigid.addObject('MechanicalObject', name='dofs', template='Rigid3', position=frames)
        if sym_y:
            # fixedDirections [tx, ty, tz, rx, ry, rz]: 1 = held.
            rigid.addObject('PartialFixedProjectiveConstraint', name='planeMotion',
                            indices=list(range(len(frames))), fixedDirections=[0, 1, 0, 1, 0, 1])
        frame_mass = 1e-4
        rigid.addObject('UniformMass', totalMass=frame_mass * len(frames))
        if P['gravity_enabled']:
            # The frames follow the duct, they don't hang from it: without this,
            # gravity makes them fall freely, and a tilted plane that has moved far
            # along its own direction no longer passes through the duct, so its
            # tilt (the bend) is forced to zero. Their weight is cancelled.
            weight = [0.0, 0.0, -frame_mass * float(rootNode.gravity.value[2]), 0.0, 0.0, 0.0]
            rigid.addObject('ConstantForceField', name='noWeight', template='Rigid3d',
                            forces=[weight for _ in frames])
        # Three points spanning each plane: frame origin, +Y and +Z in the plane.
        planes = rigid.addChild('planePoints')
        world = []
        for fr in frames:
            world += [[fr[0], fr[1], fr[2]], [fr[0], fr[1] + PLANE_SPAN, fr[2]], [fr[0], fr[1], fr[2] + PLANE_SPAN]]
        plane_dofs = planes.addObject('MechanicalObject', name='dofs', template='Vec3', position=world)
        planes.addObject('RigidMapping', name='mapping', input='@../dofs', output='@dofs', globalToLocalCoords=True,
                         rigidIndexPerPoint=[f for f in range(len(frames)) for _ in range(3)])

    # ---- the silicone body ----------------------------------------------
    cube = finger.addChild('cube')
    cube.addObject('TetrahedronSetTopologyContainer', name='container', position=positions, tetrahedra=tetrahedra)
    cube.addObject('TetrahedronSetTopologyModifier')
    cube_dofs = cube.addObject('MechanicalObject', name='dofs', template='Vec3', position=positions)
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

    # ---- mount -----------------------------------------------------------
    sym_eps = 1e-6
    big = 10 * size
    if clamp_end:
        cube.addObject('BoxROI', name='mountROI', drawBoxes=False, position='@dofs.rest_position',
                       box=[X0 - sym_eps, -big, -big, X0 + sym_eps, big, big])
        cube.addObject('RestShapeSpringsForceField', name='mount', points='@mountROI.indices', stiffness=1e12)
    if clamp_trunk:
        tx0, tx1 = (c * MM for c in g.trunk_x)
        ty0, ty1 = (c * MM for c in g.trunk_y)
        cube.addObject('BoxROI', name='baseROI', drawBoxes=False, position='@dofs.rest_position',
                       box=[tx0 - sym_eps, ty0 - sym_eps, BASE - sym_eps, tx1 + sym_eps, ty1 + sym_eps, BASE + sym_eps])
        cube.addObject('RestShapeSpringsForceField', name='trunkClamp', points='@baseROI.indices', stiffness=1e12)

    if layer_on:
        # The strain-limiting sheet glued to the bottom face: a membrane, stiff
        # in its plane, with no bending stiffness of its own. SOFA's triangle
        # FEM has no thickness parameter (its stiffness is per unit
        # thickness), so its Young's modulus is the membrane stiffness E*t.
        used = sorted({i for t in mesh['bottom'] for i in t})
        local = {n: k for k, n in enumerate(used)}
        paper = cube.addChild('strainLimitingLayer')
        paper.addObject('TriangleSetTopologyContainer', name='topo', position=[positions[i] for i in used],
                        triangles=[[local[i] for i in t] for t in mesh['bottom']])
        paper.addObject('MechanicalObject', name='dofs', template='Vec3')
        paper.addObject('SubsetMapping', name='mapping', input='@../dofs', output='@dofs', indices=used)
        Et = P['strain_layer_young_modulus'] * layer_t
        if P['strain_layer_tension_only']:
            # Paper wrinkles instead of carrying compression. Tension-only bars
            # on the bottom face's grid edges along X and Y: a bar of length a
            # with tributary width b gets stiffness E t b / a, so the sheet's
            # membrane stiffness along X and Y is E t -- in tension only
            # (DistanceMapping gives each bar's extension, a one-sided
            # PlaneForceField resists only positive extension).
            trib = {}
            for k in range(0, len(mesh['bottom']), 2):  # the bottom face's quads, 2 triangles each
                ids = sorted(set(mesh['bottom'][k]) | set(mesh['bottom'][k + 1]))
                xs_ = sorted({positions[i][0] for i in ids})
                ys_ = sorted({positions[i][1] for i in ids})
                a_, b_ = xs_[-1] - xs_[0], ys_[-1] - ys_[0]
                for i in ids:
                    for j in ids:
                        if i >= j:
                            continue
                        dx = abs(positions[i][0] - positions[j][0])
                        dy = abs(positions[i][1] - positions[j][1])
                        if dy < 1e-12 < dx:
                            trib[(i, j)] = trib.get((i, j), 0.0) + b_ / 2
                        elif dx < 1e-12 < dy:
                            trib[(i, j)] = trib.get((i, j), 0.0) + a_ / 2
            groups = {}  # one PlaneForceField per stiffness value
            for (i, j), w in trib.items():
                k_bar = Et * w / math.dist(positions[i], positions[j])
                groups.setdefault(round(k_bar, 6), []).append([local[i], local[j]])
            for g_i, (k_bar, edges) in enumerate(sorted(groups.items())):
                bars = paper.addChild(f'bars{g_i}')
                bars.addObject('EdgeSetTopologyContainer', name='edges', edges=edges,
                               position=[positions[i] for i in used])
                bars.addObject('MechanicalObject', name='dofs', template='Vec1')
                bars.addObject('DistanceMapping', name='mapping', input='@../dofs', output='@dofs', topology='@edges')
                bars.addObject('PlaneForceField', name='tensionOnly', normal=[-1.0], d=0.0, stiffness=k_bar, damping=0.0)
        else:
            # A linear membrane: also stiff in compression (a sheet bonded so
            # well it can't wrinkle), with Poisson's ratio and in-plane shear.
            paper.addObject('TriangularFEMForceFieldOptim', name='membrane', youngModulus=Et,
                            poissonRatio=P['strain_layer_poisson_ratio'])

    # ---- Y symmetry midplane ----------------------------------------------
    if sym_y:
        cube.addObject('BoxROI', name='symYROI', box=[-big, -sym_eps, -big, big, sym_eps, big], drawBoxes=False,
                       position='@dofs.rest_position')
        cube.addObject('PartialFixedProjectiveConstraint', name='symY', indices='@symYROI.indices',
                       fixedDirections=[0, 1, 0])
        if P['prevent_symmetry_crossing']:
            # One-sided stop at the midplane for every node (none may pass it).
            guard = cube.addChild('symYPos')
            guard.addObject('MechanicalObject', name='mstate')
            guard.addObject('IdentityMapping', input='@..', output='@mstate')
            guard.addObject('PlaneForceField', name='symYPosPlane', normal=[0.0, 1.0, 0.0], d=0.0,
                            stiffness=1e9, damping=1e4)

    # ---- collision planes: holding and wall contact ---------------------------
    # A node p is on the plane through the frame's three plane points
    # (a, b, c) when the tetrahedron (p, a, b, c) has zero volume; its signed
    # volume is (triangle area / 3) x (p's signed distance to the plane).
    # VolumeMapping turns those tetrahedra into scalars. Springs pull the
    # duct's nodes on the plane towards it ('hold', bilateral, soft), and a
    # one-sided PlaneForceField keeps both chambers' end walls on their own
    # side ('contact', stiff). Both act on the node AND, through
    # RigidMapping, on the frame -- the plane feels what holds it.
    area3 = PLANE_SPAN * PLANE_SPAN / 2 / 3
    coupling = []   # (kind, node, frame)
    for f, ring in enumerate(rings):
        coupling += [('hold', i, f) for i in ring]
        # End-wall nodes that could reach the plane, of the chamber before (f)
        # and after (f + 1) it: the end wall's whole thickness -- or only its
        # outer face (above the duct, which is solid below it): the nodes
        # that touch first. Fewer coupled nodes make the solve cheaper (each
        # one couples to the plane's frame).
        outer_only = bool(P['contact_outer_face_only'])
        for i, p in enumerate(positions):
            k, xl = g.local(p[0] / MM)
            if outer_only:
                on_face = (k == f and abs(xl - g.L / 2) <= 1e-6) or (k == f + 1 and abs(xl + g.L / 2) <= 1e-6)
                hit = on_face and p[2] / MM - base_mm >= g.duct_roof - 1e-6
            else:
                hit = (k == f and g.L / 2 - g.wx - 1e-6 <= xl <= g.L / 2 + 1e-6) or \
                    (k == f + 1 and -g.L / 2 - 1e-6 <= xl <= -g.L / 2 + g.wx + 1e-6)
            if hit:
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
        couple = cube.addChild('planeCoupling')
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
                    tet = [slot[i], a, c, b]  # positive on the wall's own side
                out.append(tet)
            return out

        for kind in ('hold', 'contact'):
            tets = tets_for(kind)
            if not tets:
                continue
            node = couple.addChild(kind)
            node.addObject('MeshTopology', name='topo', tetrahedra=tets)
            node.addObject('MechanicalObject', name='dofs', template='Vec1')
            # No geometric stiffness (see Finger 4: with stiff couplings it made
            # the implicit solve alternate).
            node.addObject('VolumeMapping', name='mapping', input='@../dofs', output='@dofs', topology='@topo',
                           geometricStiffness='None')
            if kind == 'hold':
                node.addObject('RestShapeSpringsForceField', name='onPlane',
                               stiffness=float(P['plane_hold_stiffness']) / area3 ** 2)
            else:
                node.addObject('PlaneForceField', name='noCrossing', normal=[1.0], d=0.0,
                               stiffness=PLANE_STIFFNESS / area3 ** 2, damping=PLANE_DAMPING / area3 ** 2)

    finger.addObject('GenericConstraintCorrection', name='constraintCorrection')

    # ---- rigid rod and its contact model ---------------------------------------
    rod_nodes = []
    if rod_on:
        if P['rod_contact_whole_skin']:
            # Every node of the outer skin (its points are FEM nodes).
            lookup = {tuple(round(c, 9) for c in p): i for i, p in enumerate(positions)}
            rod_nodes = sorted({lookup[tuple(round(c, 9) for c in p)] for p in mesh['outer'][0]
                                if tuple(round(c, 9) for c in p) in lookup})
        else:
            rod_nodes = sorted({i for t in mesh['bottom'] for i in t})  # the underside
        # Distance to the rod's centreline: the target is the axis point at the node's own y.
        contact = cube.addChild('rodContact')
        contact.addObject('MechanicalObject', name='dist', template='Vec1')
        contact.addObject('DistanceFromTargetMapping', name='mapping', input='@../dofs', output='@dist',
                          indices=rod_nodes, targetPositions=[[rod_cx, positions[i][1], rod_cz] for i in rod_nodes])
        contact.addObject('PlaneForceField', name='rod', normal=[1.0], d=rod_r,
                          stiffness=float(P['rod_contact_stiffness']), damping=ROD_DAMPING)
        half_len = W / 2 + max(W / 2, 0.15 * finger_len)
        rod_pts, rod_tris = _cylinder_y(rod_cx, rod_cz, rod_r, -half_len, half_len)
        rod = rootNode.addChild('rod')
        rod.addObject('MeshTopology', name='topo', position=rod_pts, triangles=rod_tris)
        rod.addObject('OglModel', name='visual', src='@topo', color=[0.62, 0.64, 0.7, P['rod_alpha']])

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

    if layer_on:
        # The sheet, drawn its own thickness below the base -- but at least
        # 0.4 mm, or the base's bottom face hides it in the depth buffer (only
        # the drawing; the mechanics use the real thickness) -- facing down,
        # following the base's bottom face.
        sheet_pts = [[positions[i][0], positions[i][1], positions[i][2] - sheet_gap]
                     for t in mesh['bottom'] for i in t]
        sheet_tris = [[3 * k, 3 * k + 1, 3 * k + 2] for k in range(len(mesh['bottom']))]  # facing down
        sheet = cube.addChild('strain_layer_visu')
        sheet.addObject('MeshTopology', name='topo', position=sheet_pts, triangles=sheet_tris)
        sheet.addObject('OglModel', name='visual', src='@topo', color=[0.96, 0.94, 0.86, P['strain_layer_alpha']],
                        cullFace=0)
        sheet.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    flat_inner = _flat_shaded(*mesh['inner'])
    interior = cube.addChild('interior_visu')
    interior.addObject('MeshTopology', name='topo', position=flat_inner[0], triangles=flat_inner[1])
    interior_ogl = interior.addObject('OglModel', name='visual', src='@topo',
                                      color=[0.95, 0.6, 0.1, P['interior_alpha']])
    interior.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # Collision plane indicators: a flat grid one element beyond the finger
    # on each side (full width), from below the base to above the roof.
    cell = g.h * MM
    for f, ring in enumerate(rings):
        pts, tris = _flat_rect_grid(-W / 2 - cell, W / 2 + cell, BASE - cell, BASE + H + cell, cell)
        pts = [[g.plane_x[f] * MM, p[1], p[2]] for p in pts]
        node = rootNode.addChild(f'collisionPlane{f + 1}')
        node.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        mstate = node.addObject('MechanicalObject', name='mstate', src='@topo')
        node.addObject('OglModel', name='visual', src='@topo', color=[0.85, 0.1, 0.85, P['channel_end_alpha']],
                       cullFace=0)
        node.addObject('IdentityMapping', input='@mstate', output='@visual')
        node.addObject(PlaneIndicatorController(rigid_dofs, f, cube_dofs, ring, mstate, pts, name='planeCtrl'))

    # Desktop-GUI mirroring across the Y midplane.
    if sym_y and P['mirror_visual_y']:
        for name, (pts, tris), ogl, color in (
                ('visu_mirror_Y', flat_outer, visu_ogl, [0.15, 0.45, 0.95, P['wall_alpha']]),
                ('interior_visu_mirror_Y', flat_inner, interior_ogl, [0.95, 0.6, 0.1, P['interior_alpha']])):
            m_pts = [[q[0], -q[1], q[2]] for q in pts]
            m_tris = [[t[0], t[2], t[1]] for t in tris]
            node = rootNode.addChild(name)
            node.addObject('MeshTopology', name='topo', position=m_pts, triangles=m_tris)
            mstate = node.addObject('MechanicalObject', name='mstate', src='@topo')
            node.addObject('OglModel', name='visual', src='@topo', color=color)
            node.addObject('IdentityMapping', input='@mstate', output='@visual')
            node.addObject(MirrorController(ogl, mstate, (1,), name='mirrorCtrl'))

    # For telemetry: the end faces' nodes and the duct cross-sections.
    BUILD.clear()
    BUILD.update({
        'segments': g.segments,
        'proximal_face': [i for i, p in enumerate(positions) if abs(p[0] - X0) < 1e-9],
        'distal_face': [i for i, p in enumerate(positions) if abs(p[0] - X1) < 1e-9],
        'rings': [list(r) for r in mesh['plane_nodes']],
        'planes_on': bool(frames),
        'rod_nodes': rod_nodes,
        'rod': (rod_cx, rod_cz, rod_r) if rod_on else None,
    })

    rootNode.addObject(PressureRamp(rootNode, pressure, P['target_pressure'], P['ramp_time'],
                                    instant=P['instant'], ramp_start=P['ramp_start'], name='PressureRamp'))
    return rootNode


def main():
    import Sofa.Simulation
    os.environ.setdefault('FINGER6_NO_EDITOR', '1')
    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)
    for _ in range(int((PARAMS['ramp_start'] + PARAMS['ramp_time'] + 2.0) / DT)):
        Sofa.Simulation.animate(root, DT)
    Sofa.Simulation.unload(root)


if __name__ == '__main__':
    main()
