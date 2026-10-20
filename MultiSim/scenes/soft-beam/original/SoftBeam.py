"""Soft cantilever beam: a silicone beam clamped at one end, loaded at its
free end while it runs.

The beam is a grid of hexahedra split into tetrahedra (corotational linear
FEM, so large deflections are fine), clamped over its end face at x = 0. A
TipLoad controller spreads a force over the nodes of the free end face
(x = length). Its Data fields can be changed while the simulation runs (in
runSofa: select the controller in the scene graph; in the web viewer: the
Controls panel):

  verticalForce [N]  a steady load along Z (up is positive)
  pushing            while on, an extra sideways force (sideForce [N],
                     along -Y: towards the default camera)
  flick()            starts a short upward pulse (flickForce [N] for
                     flickDuration [s]): the beam rings, then settles

An arrow at the tip shows the total force on it, pointing the way it
pushes. Gravity is on (rootNode.gravity; the web viewer can switch it off).

For comparison, Euler-Bernoulli beam theory gives the static tip deflection
F L^3 / (3 E I) for a tip force F and q L^4 / (8 E I) for the beam's own
weight q per length (I = width x height^3 / 12). Linear tetrahedra are too
stiff in bending when there are few of them through the height: with the
default 5 mm elements (two layers) the beam deflects about 56 % of the
theory's value, with 2.5 mm about 76 % (but far slower than real time).

Units: the scene is SI (metres, kg, s, N, Pa); params.json takes
millimetres for lengths, as labelled.

Run with the GUI:   runSofa -l SofaPython3 SoftBeam.py
Parameters: params.json next to this script.
"""

import json
import math
import os

import Sofa
import Sofa.Core

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')

_DEFAULT_PARAMS = {
    'length': 100.0,              # [mm] along X
    'width': 20.0,                # [mm] along Y
    'height': 10.0,               # [mm] along Z
    'element_size': 5.0,          # [mm] target edge of the hexahedra (each split into 6 tetrahedra)
    'young_modulus': 1.0e6,       # [Pa] soft silicone
    'poisson_ratio': 0.45,
    'density': 1100.0,            # [kg/m^3]
    'rayleigh_stiffness': 0.001,  # [s]
    'rayleigh_mass': 0.0,         # [1/s]
    'side_force': 0.3,            # [N] while pushing
    'flick_force': 0.3,           # [N]
    'flick_duration': 0.02,       # [s]
    'arrow_scale': 100.0,         # [mm per N] length of the force arrow
    'dt': 0.01,                   # [s] simulation time step
    'beam_alpha': 1.0,            # opacity of the beam's surface (0-1)
    'clamp_alpha': 1.0,           # opacity of the clamp block
    'arrow_alpha': 1.0,           # opacity of the force arrow
}

# Hard limit, so a typo in params.json can't make a mesh nobody can run.
MAX_ELEMENTS = 60000
MM = 1e-3


def load_params(path=PARAMS_FILE):
    params = dict(_DEFAULT_PARAMS)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                loaded = json.load(f)
            params.update({k: v for k, v in loaded.items() if k in params})
        except (OSError, ValueError) as e:
            print(f"[params] could not read {path} ({e}); using defaults")
    return params


PARAMS = load_params()
DT = PARAMS['dt']


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def _divisions(length, size):
    return max(1, int(round(length / size)))


def beam_mesh(L, W, H, size):
    """Nodes and tetrahedra of an L x W x H box (x from 0 to L, centred on
    y = 0, bottom at z = 0), split into hexahedra of about `size`, each cut
    into 6 tetrahedra around a main diagonal. Neighbouring cells are mirror
    images of each other, so they share their face diagonals (the mesh is
    conforming) and the whole mesh is mirror-symmetric -- a plain repeated
    split would make the beam twist sideways under a vertical load.

    Nodes are numbered cross-section by cross-section along X: the
    stiffness matrix then has a narrow band, which keeps the direct solver
    fast (numbered along X first, it was ~50 times slower)."""
    nx, ny, nz = _divisions(L, size), _divisions(W, size), _divisions(H, size)
    # Even counts across, for the symmetry; at least two layers through
    # the height, or it can hardly bend.
    ny = max(2, ny + ny % 2)
    nz = max(2, nz + nz % 2)

    def index(i, j, k):
        return k + (nz + 1) * (j + (ny + 1) * i)

    positions = [[L * i / nx, W * (j / ny - 0.5), H * k / nz]
                 for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)]
    # Corner c = (dx, dy, dz) bits; each path 0 -> 7 along the three axes
    # in some order is one tetrahedron.
    paths = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    tetrahedra = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                flip = (i % 2) | ((j % 2) << 1) | ((k % 2) << 2)  # mirror odd cells
                corner = [index(i + ((c ^ flip) & 1), j + (((c ^ flip) >> 1) & 1), k + (((c ^ flip) >> 2) & 1))
                          for c in range(8)]
                for path in paths:
                    tet = [corner[c] for c in path]
                    if _tet_volume(*(positions[t] for t in tet)) < 0:
                        tet[2], tet[3] = tet[3], tet[2]
                    tetrahedra.append(tet)
    return positions, tetrahedra, (nx, ny, nz)


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _tet_volume(a, b, c, d):
    return _dot(_sub(b, a), _cross(_sub(c, a), _sub(d, a))) / 6.0


def boundary_triangles(positions, tetrahedra):
    """The faces that belong to one tetrahedron only, wound outward."""
    count = {}
    for tet in tetrahedra:
        for skip in range(4):
            face = [tet[m] for m in range(4) if m != skip]
            key = tuple(sorted(face))
            entry = count.setdefault(key, [0, face, tet[skip]])
            entry[0] += 1
    triangles = []
    for n, face, opposite in count.values():
        if n != 1:
            continue
        a, b, c = (positions[i] for i in face)
        normal = _cross(_sub(b, a), _sub(c, a))
        if _dot(normal, _sub(positions[opposite], a)) > 0:
            face = [face[0], face[2], face[1]]
        triangles.append(face)
    return triangles


def _flat_shaded(points, tris):
    """Unshared vertices per triangle, so normals come out flat."""
    flat_points, flat_tris = [], []
    for tri in tris:
        base = len(flat_points)
        flat_points += [points[i] for i in tri]
        flat_tris.append([base, base + 1, base + 2])
    return flat_points, flat_tris


def _box(x0, y0, z0, x1, y1, z1):
    """Closed axis-aligned box, outward triangles."""
    pts = [[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = []
    for a, b, c, d in quads:
        tris += [[a, b, c], [a, c, d]]
    return pts, tris


def _arrow_mesh(sides=8):
    """Unit arrow along +X from x = 0 (tail) to x = 1 (tip), radius 1 for
    the head; the controller scales and places it. Returns (points, tris,
    head_start_index) -- points before head_start_index belong to the shaft."""
    shaft_r, head_len = 0.35, 0.3
    pts, tris = [], []
    ring = [(math.cos(2 * math.pi * s / sides), math.sin(2 * math.pi * s / sides)) for s in range(sides)]
    # shaft: two rings
    for x in (0.0, 1.0 - head_len):
        pts += [[x, shaft_r * c, shaft_r * s] for c, s in ring]
    for s in range(sides):
        t = (s + 1) % sides
        tris += [[s, sides + t, t], [s, sides + s, sides + t]]
    tail = len(pts)
    pts.append([0.0, 0.0, 0.0])
    for s in range(sides):
        tris.append([tail, (s + 1) % sides, s])
    # head: a cone
    head = len(pts)
    pts += [[1.0 - head_len, c, s] for c, s in ring]
    tip = len(pts)
    pts.append([1.0, 0.0, 0.0])
    centre = len(pts)
    pts.append([1.0 - head_len, 0.0, 0.0])
    for s in range(sides):
        t = (s + 1) % sides
        tris += [[head + s, head + t, tip], [centre, head + t, head + s]]
    return _flat_shaded(pts, tris), head_len


def _look_at_quaternion(eye, target, world_up=(0.0, 0.0, 1.0)):
    """Camera orientation [x, y, z, w] looking from `eye` at `target`
    (OpenGL convention: local -Z forward, +Y up), as SOFA's camera uses."""
    def normalize(v):
        n = math.sqrt(sum(c * c for c in v))
        return [v[0] / n, v[1] / n, v[2] / n]

    forward = normalize(_sub(target, eye))
    right = normalize(_cross(forward, world_up))
    up = _cross(right, forward)
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


# ----------------------------------------------------------------------
# Controller: the load on the free end, and the arrow that shows it
# ----------------------------------------------------------------------
class TipLoad(Sofa.Core.Controller):
    def __init__(self, root, dofs, force_field, tip_nodes, arrow_dofs, arrow_points, arrow_scale,
                 side_force, flick_force, flick_duration, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.root, self.dofs, self.force_field = root, dofs, force_field
        self.tip_nodes = tip_nodes
        self.arrow_dofs, self.arrow_points, self.arrow_scale = arrow_dofs, arrow_points, arrow_scale
        self.addData(name='verticalForce', type='double', value=0.0, help='Steady tip load along Z [N]',
                     group='Load')
        self.addData(name='pushing', type='bool', value=False, help='Push sideways (-Y) while on', group='Load')
        self.addData(name='sideForce', type='double', value=side_force, help='Sideways push [N]', group='Load')
        self.addData(name='flickForce', type='double', value=flick_force, help='Upward pulse [N]', group='Load')
        self.addData(name='flickDuration', type='double', value=flick_duration, help='Pulse length [s]',
                     group='Load')
        self.flick_until = -1.0
        self.applied = [0.0, 0.0, 0.0]
        self._apply()

    def flick(self):
        """Starts the upward pulse (from the next step on)."""
        self.flick_until = float(self.root.time.value) + float(self.flickDuration.value)

    def total_force(self):
        t = float(self.root.time.value)
        fy = -float(self.sideForce.value) if self.pushing.value else 0.0
        fz = float(self.verticalForce.value) + (float(self.flickForce.value) if t < self.flick_until - 1e-9 else 0.0)
        return [0.0, fy, fz]

    def tip_centre(self):
        pos = self.dofs.position.value
        n = len(self.tip_nodes)
        return [sum(float(pos[i][k]) for i in self.tip_nodes) / n for k in range(3)]

    def _apply(self):
        force = self.total_force()
        self.applied = force
        share = [f / len(self.tip_nodes) for f in force]
        self.force_field.forces.value = [share] * len(self.tip_nodes)
        self._place_arrow(force)

    def _place_arrow(self, force):
        """The arrow ends at the tip and points along the force (it pushes);
        its length shows the size. Without a force it shrinks to a 0.5 mm
        stub inside the beam's end (not to a point: the viewer would take
        all its vertices for one)."""
        tip = self.tip_centre()
        size = math.sqrt(_dot(force, force))
        if size < 1e-9:
            length, d = 0.5 * MM, [1.0, 0.0, 0.0]
        else:
            length, d = size * self.arrow_scale, [f / size for f in force]
        # Any two directions across the arrow.
        helper = [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0]
        u = _cross(d, helper)
        nu = math.sqrt(_dot(u, u))
        u = [c / nu for c in u]
        v = _cross(d, u)
        radius = min(0.12 * length, 3.0 * MM)
        start = [tip[k] - d[k] * length for k in range(3)]
        self.arrow_dofs.position.value = [
            [start[k] + d[k] * p[0] * length + (u[k] * p[1] + v[k] * p[2]) * radius for k in range(3)]
            for p in self.arrow_points
        ]

    def onAnimateBeginEvent(self, event):
        self._apply()

    def onAnimateEndEvent(self, event):
        self._place_arrow(self.applied)


# ----------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------
def createScene(rootNode):
    P = PARAMS
    plugins = [
        "Sofa.Component.Constraint.Projective", "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Mapping.Linear", "Sofa.Component.Mass", "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward", "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.StateContainer", "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Setting", "Sofa.Component.Visual", "Sofa.GL.Component.Rendering3D",
        "Sofa.GL.Component.Shader",
    ]
    for p in plugins:
        rootNode.addObject('RequiredPlugin', name='req_' + p.replace('.', '_'), pluginName=p)

    L, W, H = P['length'] * MM, P['width'] * MM, P['height'] * MM
    size = max(0.5, float(P['element_size'])) * MM
    positions, tetrahedra, (nx, ny, nz) = beam_mesh(L, W, H, size)
    if len(tetrahedra) > MAX_ELEMENTS:
        raise ValueError(f"{len(tetrahedra)} elements (more than {MAX_ELEMENTS}): use a larger element size")
    print(f"[mesh] {nx} x {ny} x {nz} cells, {len(tetrahedra)} tetrahedra, {len(positions)} nodes")
    eps = 1e-9
    clamped = [i for i, p in enumerate(positions) if p[0] < eps]
    tip_nodes = [i for i, p in enumerate(positions) if p[0] > L - eps]

    rootNode.gravity = [0.0, 0.0, -9.81]
    rootNode.dt = DT
    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('BackgroundSetting', color=[0.78, 0.87, 0.96, 1.0], image="")
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideForceFields '
                                                   'hideCollisionModels hideWireframe')

    # Camera: from the front, from above (so the sideways push shows too)
    # and a little from the free end.
    cam_lookat = [0.55 * L, 0.0, -0.1 * L]
    cam_dir = (0.8, -1.0, 1.0)
    norm = math.sqrt(sum(c * c for c in cam_dir))
    cam_distance = 1.7 * L
    cam_position = [l + cam_distance * c / norm for l, c in zip(cam_lookat, cam_dir)]
    rootNode.addObject('InteractiveCamera', name='camera', position=cam_position, distance=cam_distance,
                       orientation=_look_at_quaternion(cam_position, cam_lookat), activated=True)
    rootNode.addObject('LightManager', ambient=[0.35, 0.35, 0.35, 1.0])
    rootNode.addObject('DirectionalLight', direction=[-0.5, -0.6, 1.0])
    rootNode.addObject('DirectionalLight', color=[0.35, 0.35, 0.35, 1.0], direction=[0.5, 0.4, 0.3])

    # ---- the beam -------------------------------------------------------
    beam = rootNode.addChild('beam')
    beam.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=P['rayleigh_stiffness'],
                   rayleighMass=P['rayleigh_mass'])
    beam.addObject('SparseLDLSolver', name='linearSolver', template='CompressedRowSparseMatrixMat3x3d')
    beam.addObject('TetrahedronSetTopologyContainer', name='topo', position=positions, tetrahedra=tetrahedra)
    dofs = beam.addObject('MechanicalObject', name='dofs', template='Vec3', position=positions)
    beam.addObject('MeshMatrixMass', name='mass', massDensity=P['density'], lumping=True)
    # Corotational linear elasticity; this variant assembles about twice as
    # fast as TetrahedronFEMForceField (method 'large'), same results.
    beam.addObject('FastTetrahedralCorotationalForceField', name='fem', method='qr',
                   youngModulus=P['young_modulus'], poissonRatio=P['poisson_ratio'])
    beam.addObject('FixedProjectiveConstraint', name='clamp', indices=clamped)
    load = beam.addObject('ConstantForceField', name='tipLoad', indices=tip_nodes,
                          forces=[[0.0, 0.0, 0.0]] * len(tip_nodes))

    surface = boundary_triangles(positions, tetrahedra)
    flat = _flat_shaded(positions, surface)
    visu = beam.addChild('visu')
    visu.addObject('MeshTopology', name='topo', position=flat[0], triangles=flat[1])
    visu.addObject('OglModel', name='visual', src='@topo', color=[0.95, 0.55, 0.15, P['beam_alpha']])
    visu.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # ---- purely visual: the clamp block and the force arrow ---------------
    pts, tris = _flat_shaded(*_box(-0.12 * L, -0.9 * W - 0.01 * L, -0.4 * H - 0.05 * L,
                                   -1e-4, 0.9 * W + 0.01 * L, 1.4 * H + 0.05 * L))
    clamp = rootNode.addChild('clampBlock')
    clamp.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
    clamp.addObject('OglModel', name='visual', src='@topo', color=[0.45, 0.47, 0.52, P['clamp_alpha']])

    (arrow_pts, arrow_tris), _ = _arrow_mesh()
    arrow = rootNode.addChild('forceArrow')
    arrow.addObject('MeshTopology', name='topo', position=arrow_pts, triangles=arrow_tris)
    arrow_dofs = arrow.addObject('MechanicalObject', name='dofs', template='Vec3', position=arrow_pts)
    arrow.addObject('OglModel', name='visual', src='@topo', color=[0.85, 0.12, 0.12, P['arrow_alpha']])
    arrow.addObject('IdentityMapping', input='@dofs', output='@visual')

    rootNode.addObject(TipLoad(rootNode, dofs, load, tip_nodes, arrow_dofs, arrow_pts, P['arrow_scale'] * MM,
                               P['side_force'], P['flick_force'], P['flick_duration'], name='tipLoadCtrl'))
    return rootNode


def main():
    """Headless: a tip load of 0.1 N, then report the static deflection."""
    import Sofa.Simulation
    root = Sofa.Core.Node('root')
    createScene(root)
    Sofa.Simulation.init(root)
    ctrl = root.getObject('tipLoadCtrl')
    ctrl.verticalForce.value = -0.1
    z0 = ctrl.tip_centre()[2]
    for _ in range(int(3.0 / DT)):
        Sofa.Simulation.animate(root, DT)
    print(f"tip deflection after 3 s: {(ctrl.tip_centre()[2] - z0) * 1000:.2f} mm")


if __name__ == '__main__':
    main()
