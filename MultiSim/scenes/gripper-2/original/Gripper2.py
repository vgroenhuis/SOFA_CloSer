# -*- coding: utf-8 -*-
"""
Gripper 1: two Finger 6 fingers hanging vertically from a fixed end-effector, facing each
other, with a rigid rod (the object to grasp) between them. A copy of Finger 6: Finger on a
rod (../../finger-6/original/Finger6.py) -- Finger 5's human-scale PneuNet finger of
`segments` chambers, its collision planes between the chambers, and the simplified contact
model for the rod -- twice.

Layout (world, mm; z = 0 is the floor, the origin): the end-effector's underside is at
`effector_height`; each finger's proximal end face is clamped to it and the finger hangs
down from it, its strain-limiting layer (the base, the side it curls towards) facing the
other finger, the two inner faces `finger_gap` apart. The rod (axis along Y) is at (`rod_x`,
`rod_z`) with `rod_radius`, between the fingers. The fingers are the same mesh as in Finger 6,
each placed by a rigid transform (the second is the first's mirror image, with the
orientation of its elements and surfaces flipped), so they behave exactly like Finger 6's
finger, curling towards each other.

Pressure: `pressures` is a list in kPa: one value for both fingers, or two (the first for the
finger on the +x side -- the right one in the default view -- the second for the other).
Each finger has its own pressure controller; each is a separate FEM body with its own cavity.

There is no contact between the two fingers (only with the rod): without the rod, or with
too small a gap, they pass through each other.

Everything else is as in Finger 6, whose documentation follows.

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
(an 18 mm wall with 10 mm elements: 2 x 9 mm), see gripper2_mesh.py. Walls
can be several elements thick, which a linear-tetrahedron model needs to
bend realistically. Optional: neo-Hookean hyperelastic silicone instead of
linear (corotational) elasticity, for large strains.

Units: the scene is SI (metres, kg, s, Pa); params.json takes millimetres
for lengths, as labelled.

Run with the GUI:   runSofa -l SofaPython3 -g glfw Gripper2.py
Headless:           python Gripper2.py
Parameters: params.json next to this script (edit by hand, or with
params_editor.py, which shows the resulting element count).
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
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import importlib  # noqa: E402
import gripper2_mesh  # noqa: E402
importlib.reload(gripper2_mesh)  # the GUI's Reload re-imports this file; pick up mesher edits too

PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')

_DEFAULT_PARAMS = {
    # -- geometry [mm] --
    'segments': 5,               # chambers in the finger
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
    'floor_size': 320.0,         # green floor, purely visual
    # -- boundary conditions --
    'symmetry_y': True,
    'prevent_symmetry_crossing': True,
    'flange_acceleration': 0.06,      # [m/s2] largest acceleration of the flange (higher: the inflated fingers swing and can fold)
    'flange_speed': 0.15,             # [m/s] its largest speed
    'finger_gap': 28.0,               # [mm] between the two fingers' inner faces (their strain-limiting layers)
    # -- the loose cylinders on the floor (rigid, axis along Y), and the contact with them
    'cylinder_radii': [12.0, 9.0, 6.0],   # [mm]
    'cylinder_x': [-60.0, 0.0, 60.0],     # [mm] where they start (on the floor)
    'cylinder_length': 30.0,          # [mm] along Y
    'cylinder_density': 600.0,        # [kg/m^3]
    'cylinder_damping': 0.2,          # [N s/m] velocity damping of the cylinders (rolling and sliding resistance); capped for light ones
    'contact_stiffness': 1000.0,       # [N/m] per contact node (fingers - cylinders, floor, plane between the fingers); soft: stiff contacts make the light cylinders unstable
    'contact_whole_skin': False,      # every node of the fingers' outer skin can touch a cylinder (False: the underside only)
    'cylinder_alpha': 1.0,
    'collision_planes': True,         # a collision plane between each pair of neighbouring chambers
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
    'gravity_enabled': True,
    # -- pressure --
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
    interpreter). Skipped when GRIPPER2_NO_EDITOR is set (headless runs)."""
    if os.environ.get('GRIPPER2_NO_EDITOR'):
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
# Stiffness-proportional damping of the solver [s]: damps how the fingers bend and swing (their own motion), not
# their motion with the flange (unlike the mass-proportional one).
RAYLEIGH_STIFFNESS = 0.01
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


def _qmul(a, b):
    """Quaternion product a * b ([x, y, z, w])."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


# The collision plane frames start rotated by 90 degrees about y, which takes the frame's x axis (the
# plane's normal) to -z: the fingers hang along -z.
Q0 = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
Q0_CONJ = [-Q0[0], -Q0[1], -Q0[2], Q0[3]]


class FingerPose:
    """Where a finger hangs. Its mesh is in its own frame (x along the finger, z up out of its roof, the base --
    the side it curls towards -- at z = 0, y the finger's width); in the world the proximal end is at the top
    (z = tz + x_proximal), the finger hangs along -z, and its base faces the middle: the finger with side = +1
    stands on the +x side (world x = local z + gap / 2), the one with side = -1 is its mirror image."""

    def __init__(self, side, gap, tz):
        self.side, self.gap, self.tz = side, gap, tz

    def to_world(self, p):
        return [self.side * (p[2] + self.gap / 2), p[1], self.tz - p[0]]

    def to_local(self, w):
        return [self.tz - w[2], w[1], self.side * w[0] - self.gap / 2]


class PlaneIndicatorController(Sofa.Core.Controller):
    """Places a collision plane's indicator grid on the plane: turned like the frame that holds the plane
    (relative to its rest pose `q0`), centred on the duct's cross-section (the frame is free to slide within
    its own plane, where nothing holds it, so its origin needn't follow the finger)."""

    def __init__(self, frame_dofs, index, node_dofs, ring, target, rest_points, q0, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.frame_dofs, self.index, self.node_dofs, self.ring, self.target = frame_dofs, index, node_dofs, ring, target
        pos = self.node_dofs.position.value
        c0 = [sum(float(pos[i][k]) for i in ring) / len(ring) for k in range(3)]
        self.local = [[p[k] - c0[k] for k in range(3)] for p in rest_points]
        self.q0_conj = [-q0[0], -q0[1], -q0[2], q0[3]]
        self.n0 = _rotate(q0, [1.0, 0.0, 0.0])  # the plane's normal at rest

    def onAnimateEndEvent(self, event):
        pose = [float(c) for c in self.frame_dofs.position.value[self.index]]
        q = _qmul(pose[3:7], self.q0_conj)  # the turn since rest
        normal = _rotate(q, self.n0)
        pos = self.node_dofs.position.value
        c = [sum(float(pos[i][k]) for i in self.ring) / len(self.ring) for k in range(3)]
        off = sum((c[k] - pose[k]) * normal[k] for k in range(3))
        origin = [c[k] - off * normal[k] for k in range(3)]  # the centroid, projected onto the plane
        out = []
        for v in self.local:
            r = _rotate(q, v)
            out.append([origin[k] + r[k] for k in range(3)])
        self.target.position.value = out


# Where the end-effector (the flange) starts [mm]: x, and z of the plate's underside, where the fingers are clamped.
FLANGE_START = (0.0, 80.0)


class FlangeController(Sofa.Core.Controller):
    """The end-effector: moves towards a target position (`targetX`, `targetZ`, metres from where it starts; the
    interface's sliders set them) with a limited acceleration and speed, dragging the fingers' clamped nodes along
    (their rest positions, which a stiff spring holds them to), the plate that is drawn, and the plane between the
    fingers. The acceleration is gentle on purpose: an inflated finger swings from its clamp like a pendulum with a
    period of about a second, so a sudden start (0.3 m/s in one step is 30 m/s2) folded it. Measured on the crude
    model at 60-100 kPa, the tip's swing relative to its root is 12-17 mm at 0.5 m/s2, 7-8 mm at 0.12 m/s2 and 2-4 mm
    at 0.06 m/s2 (peak speed 0.15 m/s)."""


    def __init__(self, root, ext_states, plate_mstate, plate_rest, midplanes, acceleration=0.06, speed=0.15, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.root, self.ext_states, self.plate, self.plate_rest, self.midplanes = root, ext_states, plate_mstate, plate_rest, midplanes
        self.addData(name='targetX', type='float', value=0.0, help='Flange x relative to the start [m]. Live-editable.')
        self.addData(name='targetZ', type='float', value=0.0, help='Flange z relative to the start [m]. Live-editable.')
        self.addData(name='acceleration', type='float', value=acceleration,
                     help='Largest acceleration of the flange [m/s2]. Live-editable: the faster, the more the fingers swing.')
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
        off = np.array([self.offset[0], 0.0, self.offset[1]])
        for dofs, rest in self.ext_states:
            dofs.position.value = rest + off
        self.plate.position.value = self.plate_rest + off
        for plane, side in self.midplanes:  # the plane x = flange x: the finger on +x keeps x >= it, the other x <= it
            plane.d.value = float(side) * self.offset[0]


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
        print(f"[{self.getName()}] t={self.time:5.2f}s  target={self.targetPressure.value / 1000.0:6.2f} kPa  "
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

    # ---- the finger's mesh (local frame: X along the finger, Z up out of its roof, base at z = 0) ----
    mesh_params = dict(P, trunk_length=0.0, trunk_width=0.0, clamp_to_trunk=False)  # no trunk: it hangs from the end-effector
    n_tets, n_nodes, geo = gripper2_mesh.count_elements(mesh_params)
    if 2 * n_tets > MAX_ELEMENTS:
        raise ValueError(f"{2 * n_tets} elements (more than {MAX_ELEMENTS}): use a larger element size or fewer chambers")
    cutaway = {(0, -1, 0)} if P['cutaway'] else set()
    inner_cutaway = {(0, -1, 0)} if P['interior_cutaway'] else set()
    mesh = gripper2_mesh.build(mesh_params, z_offset_mm=0.0, outer_skip_normals=cutaway,
                               inner_skip_normals=inner_cutaway)
    g = mesh['geometry']
    lpos, ltets = mesh['positions'], mesh['tetrahedra']  # local
    print(f"[mesh] 2 fingers of {len(ltets)} tetrahedra, {len(lpos)} nodes each; {g.summary()}")

    L, W, H = g.L * MM, g.W * MM, g.H * MM
    X0, X1 = g.x_min * MM, g.x_max * MM  # the proximal and distal end faces, local x
    finger_len = X1 - X0
    sym_y = g.symmetry_y
    gap = max(float(P['finger_gap']), 0.0) * MM
    z_palm = FLANGE_START[1] * MM  # the end-effector's underside at the start, where the fingers are clamped
    tz = z_palm + X0                            # local x = X0 (the proximal face) lands at z_palm
    extent_x = gap / 2 + H                      # a finger's outer face
    layer_on = bool(P['strain_limiting_layer']) and bool(mesh['bottom'])
    layer_t = max(0.0, float(P['strain_layer_thickness'])) * MM
    sheet_gap = max(layer_t, 0.4 * MM)  # where the sheet is drawn, below the base
    rootNode.addObject('BackgroundSetting', color=[0.78, 0.87, 0.96, 1.0], image="")
    rootNode.addObject('VisualStyle', displayFlags=(
        'showVisualModels hideBehaviorModels hideCollisionModels hideBoundingCollisionModels '
        + ('showForceFields ' if P['show_fem_elements'] else 'hideForceFields ')
        + ('showInteractionForceFields ' if P['show_pressure_overlay'] else 'hideInteractionForceFields ')
        + 'hideWireframe'))

    # The floor is the plane z = 0, the world's origin; the gripper hangs high above it.
    floor_z = 0.0
    thick = 0.6 * H                     # the end-effector's plate
    R_gauge = 0.6 * H                   # the pressure gauges' dial radius (see sofaweb_scene._gauge)
    z_top = z_palm + thick + 1.12 * R_gauge + 0.6 * H + R_gauge  # the top of the gauges
    size = max(z_top, 2 * extent_x)

    # Camera (Z-up world, so the orientation is set explicitly): from the front, a little to the side and above.
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

    # Floor, end-effector, axes: purely visual.
    half = max(P['floor_size'] * MM, 3 * extent_x) / 2
    floor = rootNode.addChild('floor')
    floor.addObject('MeshTopology', name='topo', triangles=[[0, 1, 2], [0, 2, 3]],
                    position=[[-half, -half, floor_z - 1e-4], [half, -half, floor_z - 1e-4],
                              [half, half, floor_z - 1e-4], [-half, half, floor_z - 1e-4]])
    floor.addObject('OglModel', name='visual', src='@topo', color=[0.08, 0.28, 0.1, 1.0])

    pad = 0.1 * H
    stem = 0.3 * H
    mount_parts = [
        _box(-extent_x - pad, -W / 2 - pad, z_palm, extent_x + pad, W / 2 + pad, z_palm + thick),  # the plate
        _box(-stem, -stem, z_palm + thick, stem, stem, z_top + 0.2 * H),                            # the arm up
    ]
    plate_pts, plate_tris = _flat_shaded(*_merge(mount_parts))
    mount = rootNode.addChild('trunk')
    mount.addObject('MeshTopology', name='topo', position=plate_pts, triangles=plate_tris)
    plate_mstate = mount.addObject('MechanicalObject', name='mstate', src='@topo')  # moved by the FlangeController
    mount.addObject('OglModel', name='visual', src='@topo', color=[0.4, 0.26, 0.13, 1.0])
    mount.addObject('IdentityMapping', input='@mstate', output='@visual')

    arrow_ref = 0.6 * max(finger_len, W, H)
    head_len, shaft_half = arrow_ref * 0.2, arrow_ref * 0.015
    totals = {0: max(arrow_ref, extent_x + 0.3 * arrow_ref + head_len),
              1: max(arrow_ref, 1.6 * (W / 2) + head_len)}
    az = 1e-4  # the arrows start at the true origin, on the floor; the viewer's ruler too
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
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('BlockGaussSeidelConstraintSolver', maxIterations=500, tolerance=1e-8)

    # One solver for everything that moves: the fingers' contact with the loose cylinders couples them.
    world = rootNode.addChild('world')
    world.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=RAYLEIGH_STIFFNESS, rayleighMass=0.1)
    world.addObject('SparseLDLSolver', name='linearSolver', template='CompressedRowSparseMatrixd')

    big = 10 * size
    sym_eps = 1e-6
    area3 = PLANE_SPAN * PLANE_SPAN / 2 / 3
    cell = g.h * MM
    BUILD.clear()
    BUILD.update({'segments': g.segments, 'fingers': [], 'cylinders': [], 'flange_rest': [(min(p[q] for p in plate_pts) + max(p[q] for p in plate_pts)) / 2 for q in range(3)]})
    fingers_cube = []  # (the cube node, its dofs, the contact nodes, their rest positions) per finger
    midplanes = []     # (the PlaneForceField, its sign): the plane between the fingers follows the flange
    ext_states = []    # (the clamped nodes' target positions, as a numpy array)

    # ---- one finger: the same body as Finger 6's, placed by a rigid transform ------------------
    def build_finger(k, pose):
        flip = pose.side < 0  # the mirror image: element and surface orientations are flipped
        positions = [pose.to_world(p) for p in lpos]
        tets = [[t[0], t[1], t[3], t[2]] for t in ltets] if flip else ltets

        def faces(tris):
            return [[t[0], t[2], t[1]] for t in tris] if flip else tris

        total_mass = P['density'] * _mesh_volume(lpos, ltets)
        if layer_on:
            bottom_area = sum(0.5 * math.sqrt(sum(c * c for c in _cross3(
                [lpos[t[1]][q] - lpos[t[0]][q] for q in range(3)],
                [lpos[t[2]][q] - lpos[t[0]][q] for q in range(3)]))) for t in mesh['bottom'])
            total_mass += P['strain_layer_density'] * bottom_area * layer_t

        finger = world.addChild(f'finger{k + 1}')

        # -- collision planes (frames as in Finger 6; the finger hangs along -z, so each plane's normal is
        # world z: the frames start rotated by Q0, which takes the frame's x axis to -z)
        rings = mesh['plane_nodes'] if P['collision_planes'] else []
        frames = []
        for ring in rings:
            c = [sum(positions[i][q] for i in ring) / len(ring) for q in range(3)]
            frames.append([c[0], c[1], c[2]] + Q0)
        rigid_dofs = plane_dofs = planes = None
        if frames:
            rigid = finger.addChild('collisionPlanes')
            rigid_dofs = rigid.addObject('MechanicalObject', name='dofs', template='Rigid3', position=frames)
            if sym_y:
                # [tx, ty, tz, rx, ry, rz]: 1 = held: free along x and z, and a tilt about y, in world axes.
                rigid.addObject('PartialFixedProjectiveConstraint', name='planeMotion',
                                indices=list(range(len(frames))), fixedDirections=[0, 1, 0, 1, 0, 1])
            frame_mass = 1e-4
            rigid.addObject('UniformMass', totalMass=frame_mass * len(frames))
            if P['gravity_enabled']:
                weight = [0.0, 0.0, -frame_mass * float(rootNode.gravity.value[2]), 0.0, 0.0, 0.0]
                rigid.addObject('ConstantForceField', name='noWeight', template='Rigid3d',
                                forces=[weight for _ in frames])
            planes = rigid.addChild('planePoints')
            plane_world = []
            for fr in frames:  # the frame's origin, and a step along y and along x (both in the plane)
                plane_world += [[fr[0], fr[1], fr[2]], [fr[0], fr[1] + PLANE_SPAN, fr[2]], [fr[0] + PLANE_SPAN, fr[1], fr[2]]]
            plane_dofs = planes.addObject('MechanicalObject', name='dofs', template='Vec3', position=plane_world)
            planes.addObject('RigidMapping', name='mapping', input='@../dofs', output='@dofs', globalToLocalCoords=True,
                             rigidIndexPerPoint=[f for f in range(len(frames)) for _ in range(3)])

        # -- the silicone body
        cube = finger.addChild('cube')
        cube.addObject('TetrahedronSetTopologyContainer', name='container', position=positions, tetrahedra=tets)
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

        # -- clamped to the end-effector: the proximal end face
        mount_idx = [i for i, p in enumerate(lpos) if abs(p[0] - X0) < 1e-9]
        target = rootNode.addChild(f'flangeTarget{k + 1}')  # outside the solver: where the clamp holds the nodes
        target_dofs = target.addObject('MechanicalObject', name='dofs', template='Vec3', position=positions)
        cube.addObject('RestShapeSpringsForceField', name='mount', points=mount_idx, stiffness=1e12,
                       external_rest_shape=f'@/flangeTarget{k + 1}/dofs', external_points=mount_idx)
        ext_states.append((target_dofs, np.array(positions, dtype=float)))

        if layer_on:
            used = sorted({i for t in mesh['bottom'] for i in t})
            local = {n: q for q, n in enumerate(used)}
            paper = cube.addChild('strainLimitingLayer')
            paper.addObject('TriangleSetTopologyContainer', name='topo', position=[positions[i] for i in used],
                            triangles=faces([[local[i] for i in t] for t in mesh['bottom']]))
            paper.addObject('MechanicalObject', name='dofs', template='Vec3')
            paper.addObject('SubsetMapping', name='mapping', input='@../dofs', output='@dofs', indices=used)
            Et = P['strain_layer_young_modulus'] * layer_t
            if P['strain_layer_tension_only']:
                trib = {}
                for q in range(0, len(mesh['bottom']), 2):  # the bottom face's quads, 2 triangles each
                    ids = sorted(set(mesh['bottom'][q]) | set(mesh['bottom'][q + 1]))
                    xs_ = sorted({lpos[i][0] for i in ids})
                    ys_ = sorted({lpos[i][1] for i in ids})
                    a_, b_ = xs_[-1] - xs_[0], ys_[-1] - ys_[0]
                    for i in ids:
                        for j in ids:
                            if i >= j:
                                continue
                            dx = abs(lpos[i][0] - lpos[j][0])
                            dy = abs(lpos[i][1] - lpos[j][1])
                            if dy < 1e-12 < dx:
                                trib[(i, j)] = trib.get((i, j), 0.0) + b_ / 2
                            elif dx < 1e-12 < dy:
                                trib[(i, j)] = trib.get((i, j), 0.0) + a_ / 2
                groups = {}  # one PlaneForceField per stiffness value
                for (i, j), w in trib.items():
                    k_bar = Et * w / math.dist(lpos[i], lpos[j])
                    groups.setdefault(round(k_bar, 6), []).append([local[i], local[j]])
                for g_i, (k_bar, edges) in enumerate(sorted(groups.items())):
                    bars = paper.addChild(f'bars{g_i}')
                    bars.addObject('EdgeSetTopologyContainer', name='edges', edges=edges,
                                   position=[positions[i] for i in used])
                    bars.addObject('MechanicalObject', name='dofs', template='Vec1')
                    bars.addObject('DistanceMapping', name='mapping', input='@../dofs', output='@dofs', topology='@edges')
                    bars.addObject('PlaneForceField', name='tensionOnly', normal=[-1.0], d=0.0, stiffness=k_bar,
                                   damping=0.0)
            else:
                paper.addObject('TriangularFEMForceFieldOptim', name='membrane', youngModulus=Et,
                                poissonRatio=P['strain_layer_poisson_ratio'])

        # -- Y symmetry midplane (y is the same in both frames)
        if sym_y:
            cube.addObject('BoxROI', name='symYROI', box=[-big, -sym_eps, -big, big, sym_eps, big], drawBoxes=False,
                           position='@dofs.rest_position')
            cube.addObject('PartialFixedProjectiveConstraint', name='symY', indices='@symYROI.indices',
                           fixedDirections=[0, 1, 0])
            if P['prevent_symmetry_crossing']:
                guard = cube.addChild('symYPos')
                guard.addObject('MechanicalObject', name='mstate')
                guard.addObject('IdentityMapping', input='@..', output='@mstate')
                guard.addObject('PlaneForceField', name='symYPosPlane', normal=[0.0, 1.0, 0.0], d=0.0,
                                stiffness=1e9, damping=1e4)

        # -- collision planes: holding and wall contact (as in Finger 6)
        coupling = []   # (kind, node, frame)
        for f, ring in enumerate(rings):
            coupling += [('hold', i, f) for i in ring]
            outer_only = bool(P['contact_outer_face_only'])
            for i, p in enumerate(lpos):
                kk, xl = g.local(p[0] / MM)
                if outer_only:
                    on_face = (kk == f and abs(xl - g.L / 2) <= 1e-6) or (kk == f + 1 and abs(xl + g.L / 2) <= 1e-6)
                    hit = on_face and p[2] / MM >= g.duct_roof - 1e-6
                else:
                    hit = (kk == f and g.L / 2 - g.wx - 1e-6 <= xl <= g.L / 2 + 1e-6) or \
                        (kk == f + 1 and -g.L / 2 - 1e-6 <= xl <= -g.L / 2 + g.wx + 1e-6)
                if hit:
                    coupling.append(('contact', i, f))
        if coupling:
            nodes = sorted({i for _, i, _ in coupling})
            slot = {n: q for q, n in enumerate(nodes)}
            n_nodes_c = len(nodes)
            pairs = []
            for n in nodes:
                pairs += [0, n]
            for q in range(3 * len(frames)):
                pairs += [1, q]
            couple = cube.addChild('planeCoupling')
            couple.addObject('MechanicalObject', name='dofs', template='Vec3')
            couple.addObject('SubsetMultiMapping', name='mapping',
                             input=[cube_dofs.getLinkPath(), plane_dofs.getLinkPath()], output='@dofs', indexPairs=pairs)
            planes.addChild(couple)
            plane_pts_rest = [list(plane_dofs.position.value[q]) for q in range(3 * len(frames))]

            def tets_for(kind):
                out = []
                for kk, i, f in coupling:
                    if kk != kind:
                        continue
                    a, b, c = n_nodes_c + 3 * f, n_nodes_c + 3 * f + 1, n_nodes_c + 3 * f + 2
                    tet = [slot[i], a, b, c]
                    pa, pb, pc = plane_pts_rest[3 * f], plane_pts_rest[3 * f + 1], plane_pts_rest[3 * f + 2]
                    if kind == 'contact' and _signed_volume(positions[i], pa, pb, pc) < 0:
                        tet = [slot[i], a, c, b]  # positive on the wall's own side
                    out.append(tet)
                return out

            for kind in ('hold', 'contact'):
                tets_k = tets_for(kind)
                if not tets_k:
                    continue
                node = couple.addChild(kind)
                node.addObject('MeshTopology', name='topo', tetrahedra=tets_k)
                node.addObject('MechanicalObject', name='dofs', template='Vec1')
                node.addObject('VolumeMapping', name='mapping', input='@../dofs', output='@dofs', topology='@topo',
                               geometricStiffness='None')
                if kind == 'hold':
                    node.addObject('RestShapeSpringsForceField', name='onPlane',
                                   stiffness=float(P['plane_hold_stiffness']) / area3 ** 2)
                else:
                    node.addObject('PlaneForceField', name='noCrossing', normal=[1.0], d=0.0,
                                   stiffness=PLANE_STIFFNESS / area3 ** 2, damping=PLANE_DAMPING / area3 ** 2)

        # -- guards: the floor (no node below z = 0) and the plane between the fingers (the flange's x), which the
        # fingers can't cross (there is no contact between them otherwise)
        floor_guard = cube.addChild('floorGuard')
        floor_guard.addObject('MechanicalObject', name='mstate')
        floor_guard.addObject('IdentityMapping', input='@..', output='@mstate')
        floor_guard.addObject('PlaneForceField', name='floor', normal=[0.0, 0.0, 1.0], d=0.0,
                              stiffness=float(P['contact_stiffness']), damping=ROD_DAMPING)
        mid_guard = cube.addChild('midPlane')
        mid_guard.addObject('MechanicalObject', name='mstate')
        mid_guard.addObject('IdentityMapping', input='@..', output='@mstate')
        mid_plane = mid_guard.addObject('PlaneForceField', name='midPlane', normal=[float(pose.side), 0.0, 0.0], d=0.0,
                                        stiffness=float(P['contact_stiffness']), damping=ROD_DAMPING)
        midplanes.append((mid_plane, pose.side))

        # -- the nodes that can touch a cylinder: the underside, or the whole outer skin
        if P['contact_whole_skin']:
            lookup = {tuple(round(q, 9) for q in p): i for i, p in enumerate(lpos)}
            contact_nodes = sorted({lookup[tuple(round(q, 9) for q in p)] for p in mesh['outer'][0]
                                    if tuple(round(q, 9) for q in p) in lookup})
        else:
            contact_nodes = sorted({i for t in mesh['bottom'] for i in t})
        fingers_cube.append((finger, cube, cube_dofs, contact_nodes, positions))

        # -- cavity (pressure)
        cavity_pts, cavity_tris = mesh['cavity']
        cavity = cube.addChild('cavity')
        cavity.addObject('MeshTopology', name='topo', position=[pose.to_world(p) for p in cavity_pts],
                         triangles=faces(cavity_tris))
        cavity.addObject('MechanicalObject', name='cavityDofs', src='@topo')
        pressure = cavity.addObject('SurfacePressureConstraint', name='pressure', triangles='@topo.triangles',
                                    valueType='pressure', value=0)
        cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

        # -- visuals
        flat_outer = _flat_shaded(*mesh['outer'])
        visu = cube.addChild('visu')
        visu.addObject('MeshTopology', name='topo', position=[pose.to_world(p) for p in flat_outer[0]],
                       triangles=faces(flat_outer[1]))
        visu.addObject('OglModel', name='visual', src='@topo', color=[0.15, 0.45, 0.95, P['wall_alpha']])
        visu.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

        if layer_on:
            sheet_pts = [pose.to_world([lpos[i][0], lpos[i][1], lpos[i][2] - sheet_gap])
                         for t in mesh['bottom'] for i in t]
            sheet_tris = faces([[3 * q, 3 * q + 1, 3 * q + 2] for q in range(len(mesh['bottom']))])
            sheet = cube.addChild('strain_layer_visu')
            sheet.addObject('MeshTopology', name='topo', position=sheet_pts, triangles=sheet_tris)
            sheet.addObject('OglModel', name='visual', src='@topo',
                            color=[0.96, 0.94, 0.86, P['strain_layer_alpha']], cullFace=0)
            sheet.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

        flat_inner = _flat_shaded(*mesh['inner'])
        interior = cube.addChild('interior_visu')
        interior.addObject('MeshTopology', name='topo', position=[pose.to_world(p) for p in flat_inner[0]],
                           triangles=faces(flat_inner[1]))
        interior.addObject('OglModel', name='visual', src='@topo', color=[0.95, 0.6, 0.1, P['interior_alpha']])
        interior.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

        # -- collision plane indicators: a flat grid one element beyond the finger on each side
        for f, ring in enumerate(rings):
            gp, gt = _flat_rect_grid(-W / 2 - cell, W / 2 + cell, -cell, H + cell, cell)
            pts = [pose.to_world([g.plane_x[f] * MM, p[1], p[2]]) for p in gp]
            node = rootNode.addChild(f'collisionPlane_{k + 1}_{f + 1}')
            node.addObject('MeshTopology', name='topo', position=pts, triangles=gt)
            mstate = node.addObject('MechanicalObject', name='mstate', src='@topo')
            node.addObject('OglModel', name='visual', src='@topo', color=[0.85, 0.1, 0.85, P['channel_end_alpha']],
                           cullFace=0)
            node.addObject('IdentityMapping', input='@mstate', output='@visual')
            node.addObject(PlaneIndicatorController(rigid_dofs, f, cube_dofs, ring, mstate, pts, Q0, name='planeCtrl'))

        rootNode.addObject(PressureRamp(rootNode, pressure, 0.0, P['ramp_time'],
                                        instant=P['instant'], ramp_start=P['ramp_start'], name=f'PressureRamp{k + 1}'))
        BUILD['fingers'].append({
            'proximal_face': mount_idx,
            'distal_face': [i for i, p in enumerate(lpos) if abs(p[0] - X1) < 1e-9],
            'rings': [list(r) for r in mesh['plane_nodes']],
            'planes_on': bool(frames),
            'side': pose.side, 'gap': gap, 'tz': tz,
        })

    build_finger(0, FingerPose(+1, gap, tz))
    build_finger(1, FingerPose(-1, gap, tz))

    # ---- loose cylinders on the floor: rigid bodies that can slide in x and z and roll about y (their axis stays
    # parallel to Y). The model is the +y half of the world (the fingers are half models), so a cylinder is its
    # half too: half the mass. Contact with a finger is a penalty on the distance of each of the finger's contact
    # nodes to the cylinder's centreline (as in Finger 6), acting on both bodies: the cylinder is pushed back by
    # the finger and the finger by the cylinder (no friction). Cylinders keep apart by the same kind of penalty
    # between their centres, and rest on the floor by one on the height of their centre.
    n_cyl = min(len(P['cylinder_radii']), len(P['cylinder_x']))
    cyl_len = max(float(P['cylinder_length']), 1.0) * MM
    palette = [[0.86, 0.55, 0.2], [0.3, 0.72, 0.52], [0.75, 0.42, 0.78], [0.9, 0.8, 0.3], [0.45, 0.62, 0.9], [0.9, 0.45, 0.45]]
    centres = []
    for j in range(n_cyl):
        r = max(float(P['cylinder_radii'][j]), 0.5) * MM
        cx = float(P['cylinder_x'][j]) * MM
        mass = 0.5 * float(P['cylinder_density']) * math.pi * r * r * cyl_len
        inertia = 0.5 * mass * r * r
        # A light cylinder needs a softer contact to stay stable at this time step: the stiffness (per contact node)
        # is capped at about 3e5 N/m per kg of its (half) mass.
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
        # the visual: the full cylinder (length cyl_len), moving with the body
        vp, vt = _cylinder_y(cx, r, r, -cyl_len / 2, cyl_len / 2)
        vp = [[p[0], p[1], p[2]] for p in vp]
        visu = cyl.addChild('visu')
        visu.addObject('MeshTopology', name='topo', position=vp, triangles=vt)
        visu.addObject('OglModel', name='visual', src='@topo', color=palette[j % len(palette)] + [P['cylinder_alpha']])
        visu.addObject('RigidMapping', input='@../dofs', output='@visual', globalToLocalCoords=True)
        # contact with the fingers: one point on the axis per contact node, at that node's y
        all_nodes = [(k, i) for k, (_, _, _, nodes_k, _) in enumerate(fingers_cube) for i in nodes_k]
        axis = cyl.addChild('axisPoints')
        axis_dofs = axis.addObject('MechanicalObject', name='dofs', template='Vec3',
                                   position=[[cx, fingers_cube[k][4][i][1], r] for k, i in all_nodes])
        axis.addObject('RigidMapping', input='@../dofs', output='@dofs', globalToLocalCoords=True)
        n_pts = len(all_nodes)
        pairs = []
        for k, i in all_nodes:
            pairs += [k, i]
        for q in range(n_pts):
            pairs += [2, q]
        contact = fingers_cube[0][1].addChild(f'cylinderContact{j + 1}')
        contact.addObject('MechanicalObject', name='dofs', template='Vec3')
        contact.addObject('SubsetMultiMapping', name='mapping', input=[fingers_cube[0][2].getLinkPath(),
                          fingers_cube[1][2].getLinkPath(), axis_dofs.getLinkPath()], output='@dofs', indexPairs=pairs)
        fingers_cube[1][1].addChild(contact)
        axis.addChild(contact)
        dist = contact.addChild('distance')
        dist.addObject('MeshTopology', name='topo', edges=[[q, n_pts + q] for q in range(n_pts)])
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

    world.addObject('GenericConstraintCorrection', name='constraintCorrection')
    rootNode.addObject(FlangeController(rootNode, ext_states, plate_mstate, np.array(plate_pts, dtype=float), midplanes,
                                        acceleration=float(P['flange_acceleration']), speed=float(P['flange_speed']),
                                        name='Flange'))
    return rootNode


def main():
    import Sofa.Simulation
    os.environ.setdefault('GRIPPER2_NO_EDITOR', '1')
    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)
    for _ in range(int((PARAMS['ramp_start'] + PARAMS['ramp_time'] + 2.0) / DT)):
        Sofa.Simulation.animate(root, DT)
    Sofa.Simulation.unload(root)


if __name__ == '__main__':
    main()
