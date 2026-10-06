# -*- coding: utf-8 -*-
"""
Hollow silicone-rubber cube inflated with air (FEM pressure-cavity test).

This is a minimal test bed for pneumatic soft-robotic actuators (e.g.
silicone pneumatic fingers): a hollow box shell is meshed procedurally
(no external mesh files), clamped on part of its bottom face to a rigid
"trunk" raising it one block above the floor (see ATTACH_*/BASE_OFFSET),
and its internal cavity is inflated by ramping up an air pressure via
the SoftRobots SurfacePressureConstraint.

Units are SI throughout: metres, kilograms, seconds, pascals, newtons.

Run with the GUI -- SofaPython3 must be loaded explicitly (-l), and the
GUI backend must be given explicitly too (-g), otherwise the plugin
autoload / default-GUI resolution doesn't pick either up reliably:
    runSofa.exe -l SofaPython3 -g imgui HollowCubeInflation.py   (default:
        docked Scene Graph/Viewport/Log panels, but its background is a
        branded backdrop baked into its own rendering code -- neither the
        scene's BackgroundSetting component nor GUIManager.set
        BackgroundImage() can override it, confirmed by testing both)
    runSofa.exe -l SofaPython3 -g glfw  HollowCubeInflation.py   (older,
        plainer single-viewport GUI, but it DOES respect BackgroundSetting
        -- use this one if you want the plain light-blue background)
See run_imgui.bat / run_glfw.bat for one-click shortcuts to each.

Run headless (prints cavity pressure/volume to the console):
    python HollowCubeInflation.py
(requires a Python whose ABI matches the SOFA build, see README.md)

Parameters (block size, box dimensions and wall thickness in blocks,
material, target pressure, ramp time, instant mode) live in params.json
next to this script, created
with defaults on first run. Edit it by hand, or with the standalone
editor (edit_params.bat / `python params_editor.py`) -- an independent
Tkinter window, no SOFA dependency, doesn't touch a running session
directly. After Save: geometry/material changes need the SOFA GUI's
Reload button; target pressure/ramp time/instant are ALSO live-editable
while the sim is running, directly in the Scene Graph panel (select the
"PressureRamp" node), no reload needed for those.
"""
import itertools
import json
import math
import os
import subprocess

import Sofa
import Sofa.Core


def _look_at_quaternion(eye, target, world_up=(0.0, 0.0, 1.0)):
    """Camera orientation quaternion [x, y, z, w] for a camera at `eye`
    looking towards `target`, with `world_up` as the world's vertical
    axis. Assumes the standard OpenGL camera convention (local -Z is the
    view direction, local +Y is up, local +X is right) that SOFA's
    BaseCamera also follows.
    """
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    def normalize(v):
        n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        return (v[0] / n, v[1] / n, v[2] / n)

    forward = normalize(sub(target, eye))
    right = normalize(cross(forward, world_up))
    up = cross(right, forward)

    # Camera-to-world rotation matrix, columns = local axes (right, up,
    # -forward) expressed in world space -- matches sofa::type::Quat's
    # own quaternion-to-matrix convention (see Quat.h, toMatrix()).
    m00, m01, m02 = right[0], up[0], -forward[0]
    m10, m11, m12 = right[1], up[1], -forward[1]
    m20, m21, m22 = right[2], up[2], -forward[2]

    trace = m00 + m11 + m22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return [qx, qy, qz, qw]


# ----------------------------------------------------------------------
# Parameters -- loaded from params.json next to this script, so they can
# be edited without touching code (by hand, or with params_editor.py).
# The file is created with these defaults on first run if missing.
#
# The box is built from cube-shaped "blocks" of a single elementary size:
# `block_size` sets that size, `length_blocks`/`width_blocks`/
# `height_blocks` set the outer box's extent in X/Y/Z as a count of those
# blocks, and `wall_length_blocks`/`wall_width_blocks`/`wall_height_blocks`
# set the shell's thickness -- in blocks -- on the pair of faces
# perpendicular to each axis (e.g. `wall_height_blocks` is how many
# blocks thick the top and bottom shell is). `trunk_length_blocks`/
# `trunk_width_blocks` set the footprint (centered, in blocks along X/Y)
# of the rigid "trunk" attaching the box to fixed world -- see
# _resolve_trunk_axis. These only take effect on scene (re)build -- after
# editing them, use the GUI's Reload button, there is no way to resize a
# mesh that's already been built and is mid-simulation.
# `target_pressure`/`ramp_time`/`instant` seed the PressureRamp
# controller's initial values, but are ALSO exposed as live-editable
# Data fields on that controller once the scene is running (select the
# "PressureRamp" node in the Scene Graph panel) -- editing params.json
# again would only apply on the next reload, not retroactively.
# ----------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')

_DEFAULT_PARAMS = {
    'block_size': 10.0,          # elementary block edge length [mm]
    'length_blocks': 6,          # box extent along X, in blocks
    'width_blocks': 10,          # box extent along Y, in blocks
    'height_blocks': 8,          # box extent along Z (vertical), in blocks
    'wall_length_blocks': 1,     # shell thickness on the +-X faces, in blocks
                                  # (>=1; >=2 needed for a believable bending
                                  # response -- see build_hollow_cube_mesh)
    'wall_width_blocks': 1,      # shell thickness on the +-Y faces, in blocks
    'wall_height_blocks': 1,     # shell thickness on the +-Z (top/bottom)
                                  # faces, in blocks
    'trunk_length_blocks': 6,    # footprint of the rigid "trunk" attaching
    'trunk_width_blocks': 10,    # the box to fixed world, in blocks along
                                  # X/Y -- always centered; snapped to the
                                  # same odd/even parity as length_blocks/
                                  # width_blocks respectively so it stays
                                  # exactly grid-aligned (see
                                  # _resolve_trunk_axis). Defaults to the
                                  # whole footprint, matching length_blocks/
                                  # width_blocks above.
    'floor_size_blocks': 30,     # edge length of the square green ground
                                  # plane, in blocks (purely visual, centered
                                  # at the world origin).
    'symmetry_x': False,         # only build the +X half of the box (cut
    'symmetry_y': False,         # at the midplane) / +Y half / bottom Z
    'symmetry_z': False,         # half respectively, halving FEM element
                                  # count per axis enabled, with a
                                  # PartialFixedProjectiveConstraint at
                                  # each cut plane holding its points to
                                  # zero displacement in the plane's
                                  # normal direction -- see createScene
                                  # and the block-count parity note on
                                  # _resolve_geometry. X/Y are valid
                                  # symmetries of this model (the trunk
                                  # is centered on both, and gravity/
                                  # pressure don't break either); Z is
                                  # NOT -- the box is clamped only at its
                                  # bottom, not top and bottom, so
                                  # symmetry_z models a different,
                                  # unphysical problem rather than a
                                  # reduction of the real one.
    'symmetry_buffer_blocks': 0,  # simulate this many extra blocks of
                                  # ordinary FEM material past each active
                                  # symmetry plane (0 = cut exactly at the
                                  # plane, the original behaviour). Gives
                                  # the symmetry-constrained row a natural
                                  # elastic neighbour on both sides
                                  # instead of being rigid on one side and
                                  # free on the other, which otherwise
                                  # shows up as a visible kink right at
                                  # the cut -- costs a few extra elements
                                  # per axis enabled, in exchange for a
                                  # smoother transition there. Only
                                  # affects the FEM shell mesh, not the
                                  # visual surfaces or the cavity (both
                                  # still cut exactly at the plane) or
                                  # where the symmetry constraint itself
                                  # sits (still the true plane, unmoved).
    'prevent_symmetry_crossing': True,  # add a one-sided PlaneForceField
                                  # at each X/Y/Z symmetry midplane, so
                                  # points elsewhere in the mesh (not just
                                  # the ones exactly on the plane at rest,
                                  # which the constraint above already
                                  # holds) get pushed back if they deform
                                  # past it -- without this, e.g. negative
                                  # pressure can fold the mesh through the
                                  # plane (a self-collision in reality).
                                  # Independent of symmetry_x/y/z: those
                                  # only control whether the mesh itself
                                  # is halved to save FEM elements -- the
                                  # midplanes exist geometrically either
                                  # way, so this guards all three whether
                                  # or not their element count is halved
                                  # (guarding both sides of the full mesh
                                  # when it isn't).
    'mirror_visual_x': True,     # live-mirror the outer-skin/interior-
    'mirror_visual_y': True,     # volume visuals across the X/Y/Z
    'mirror_visual_z': True,     # symmetry plane respectively (and every
                                  # combination of the enabled ones, for
                                  # more than one), so the render shows
                                  # the complete deformed shape rather
                                  # than just the simulated fraction --
                                  # see MirrorController. Each only has an
                                  # effect if its own symmetry_* flag is
                                  # also on; independently disabling one
                                  # here leaves that plane un-mirrored
                                  # (open/cut) even if the others are
                                  # fully reconstructed. Purely visual,
                                  # doesn't touch the physics.
    'young_modulus': 1.0e6,      # [Pa]  ~soft silicone rubber (Shore ~30-40A)
    'poisson_ratio': 0.45,       # near-incompressible rubber
    'density': 1100.0,           # [kg/m^3] silicone rubber
    'target_pressure': 1000.0,   # [Pa] gauge pressure the cavity ramps/jumps to
    'ramp_time': 2.0,            # [s] time to reach target_pressure if not instant
    'instant': False,            # apply target_pressure immediately instead
                                  # of ramping towards it
    'gravity_enabled': True,     # apply the -Z 9.81 m/s^2 gravity below;
                                  # off gives free-floating [0,0,0] gravity
                                  # (useful to isolate inflation-only
                                  # behaviour from the RestShapeSprings
                                  # base clamp fighting sag)
    'cutaway': True,             # omit CUTAWAY_FACE from the outer skin so
                                  # you can see into the cavity
    'interior_cutaway': True,    # same, but for the interior-volume visual
                                  # surface (same face as `cutaway`). Purely
                                  # visual -- the closed surface used for
                                  # the actual pressure/volume physics is
                                  # unaffected either way.
    'wall_alpha': 0.4,           # opacity of the blue outer-shell visual,
                                  # 0 (invisible) .. 1 (opaque)
    'interior_alpha': 0.5,       # opacity of the orange interior-volume
                                  # visual, 0 (invisible) .. 1 (opaque)
    'show_pressure_overlay': True,  # SurfacePressureConstraint's own solid
                                     # red(inflating)/green(idle) cavity-
                                     # surface indicator. No alpha control
                                     # is possible for this one -- its draw
                                     # code takes raw RGB with no alpha
                                     # parameter at all (SurfacePressureModel
                                     # ::draw(), confirmed by reading the
                                     # source) -- on/off is all there is.
    'show_fem_elements': False,  # VisualStyle's showForceFields flag --
                                  # draws the FEM shell's own tetrahedra
                                  # (wireframe-ish, hard-opaque regardless
                                  # of wall_alpha, see the note in
                                  # createScene). Set here rather than
                                  # toggled by hand in the SOFA GUI's
                                  # display-flags menu because Reload
                                  # resets those flags back to whatever
                                  # this scene sets on build.
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
    """Windows-only PID liveness check via `tasklist` (stdlib-only, no
    psutil dependency)."""
    try:
        output = subprocess.check_output(
            ['tasklist', '/FI', f'PID eq {pid}'],
            stderr=subprocess.DEVNULL, text=True)
        return str(pid) in output
    except (OSError, subprocess.SubprocessError):
        return False


def launch_params_editor():
    """Opens params_editor.py in its own process (a plain `python`, not
    runSofa's embedded interpreter -- inside that, sys.executable points
    at runSofa.exe itself, which would try to load the editor script as
    a SOFA scene rather than run it). Non-blocking.

    Guarded against spawning duplicate windows via a PID file on disk
    rather than an in-memory flag: the SOFA GUI's Reload button re-
    imports this whole module from scratch (so that code edits actually
    take effect), which would reset any plain module-level variable back
    to its initial value on every reload, defeating an in-memory guard.

    Set the HOLLOWCUBE_NO_EDITOR environment variable to skip this (used
    for headless/automated runs so they don't pop up a stray window).
    """
    if os.environ.get('HOLLOWCUBE_NO_EDITOR'):
        return

    if os.path.exists(_EDITOR_PID_FILE):
        try:
            with open(_EDITOR_PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
        except (OSError, ValueError):
            old_pid = None
        if old_pid is not None and _pid_is_running(old_pid):
            return  # editor from a previous (re)load is still open

    editor_path = os.path.join(_SCRIPT_DIR, 'params_editor.py')
    try:
        # stdio must be detached: inheriting runSofa's handles means
        # anything reading its stdout (a pipe, a log redirect, ...)
        # would block until this GUI window is closed, since it'd hold
        # that pipe open too.
        process = subprocess.Popen(
            ['python', editor_path],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True)
        with open(_EDITOR_PID_FILE, 'w') as f:
            f.write(str(process.pid))
    except OSError as e:
        print(f"[params] could not launch params_editor.py: {e}")


def _resolve_trunk_axis(divs, trunk_blocks):
    """Snaps a requested trunk footprint (in blocks, along one axis) to a
    value that's centered and grid-aligned: since the box's own centre
    sits exactly on a grid point when `divs` is even, and exactly midway
    between two grid points when `divs` is odd, a symmetric trunk span
    only lands on grid points if its own block count has the same parity
    as `divs`. Clamped to [1 or 2, divs] (never bigger than the box
    itself), rounding up to the nearest valid value when the request is
    in between.
    """
    trunk = max(1, int(round(trunk_blocks)))
    trunk = min(trunk, divs)
    if trunk % 2 != divs % 2:
        trunk = trunk + 1 if trunk < divs else trunk - 1
    return trunk


def _resolve_geometry(block_size, length_blocks, width_blocks, height_blocks,
                       wall_length_blocks, wall_width_blocks, wall_height_blocks,
                       trunk_length_blocks, trunk_width_blocks,
                       symmetry_x=False, symmetry_y=False, symmetry_z=False):
    """Snaps the block counts to an achievable structured-grid geometry
    (each axis needs at least one interior/cavity cell) instead of
    requiring the caller to get it right -- params.json is meant to be
    hand-edited (or edited via params_editor.py), so this has to tolerate
    arbitrary reasonable input rather than asserting.

    An axis with symmetry enabled is additionally rounded up to an even
    block count, so its midplane -- where the model gets cut in half --
    always falls exactly on a grid boundary instead of through the
    middle of a block.
    """
    cell = block_size * 1e-3  # mm -> m

    def resolve_axis(extent_blocks, wall_blocks, symmetric):
        divs = max(1, int(round(extent_blocks)))
        wall = max(1, int(round(wall_blocks)))
        min_divs = 2 * wall + 1  # need at least 1 interior (cavity) cell
        divs = max(divs, min_divs)
        if symmetric:
            divs += divs % 2  # round up to even
        return divs, wall

    divs_x, wall_x = resolve_axis(length_blocks, wall_length_blocks, symmetry_x)
    divs_y, wall_y = resolve_axis(width_blocks, wall_width_blocks, symmetry_y)
    divs_z, wall_z = resolve_axis(height_blocks, wall_height_blocks, symmetry_z)
    trunk_divs_x = _resolve_trunk_axis(divs_x, trunk_length_blocks)
    trunk_divs_y = _resolve_trunk_axis(divs_y, trunk_width_blocks)
    return cell, divs_x, divs_y, divs_z, wall_x, wall_y, wall_z, trunk_divs_x, trunk_divs_y


SYMMETRY_X = PARAMS['symmetry_x']
SYMMETRY_Y = PARAMS['symmetry_y']
SYMMETRY_Z = PARAMS['symmetry_z']
SYMMETRY_BUFFER_BLOCKS = PARAMS['symmetry_buffer_blocks']
PREVENT_SYMMETRY_CROSSING = PARAMS['prevent_symmetry_crossing']
MIRROR_VISUAL_X = PARAMS['mirror_visual_x']
MIRROR_VISUAL_Y = PARAMS['mirror_visual_y']
MIRROR_VISUAL_Z = PARAMS['mirror_visual_z']

CELL, DIVS_X, DIVS_Y, DIVS_Z, WALL_X, WALL_Y, WALL_Z, TRUNK_DIVS_X, TRUNK_DIVS_Y = _resolve_geometry(
    PARAMS['block_size'], PARAMS['length_blocks'], PARAMS['width_blocks'], PARAMS['height_blocks'],
    PARAMS['wall_length_blocks'], PARAMS['wall_width_blocks'], PARAMS['wall_height_blocks'],
    PARAMS['trunk_length_blocks'], PARAMS['trunk_width_blocks'],
    SYMMETRY_X, SYMMETRY_Y, SYMMETRY_Z)

LENGTH = DIVS_X * CELL  # [m] outer box extent along X
WIDTH = DIVS_Y * CELL   # [m] outer box extent along Y
HEIGHT = DIVS_Z * CELL  # [m] outer box extent along Z (vertical)

FLOOR_SIZE = PARAMS['floor_size_blocks'] * CELL  # [m] edge length of the
                                                  # square green floor

BASE_OFFSET = CELL  # [m] the hollow box is raised this far above the floor
                     # (one elementary block), leaving room for the "trunk"
                     # visual below it -- see ATTACH_* and createScene().

# Footprint (in the box's local X/Y, before centring) of the part of the
# bottom face attached to fixed world, visualized by the trunk between the
# floor and the box -- TRUNK_DIVS_X/TRUNK_DIVS_Y blocks, centered. The
# BoxROI clamp and the trunk mesh both derive from these same bounds, so
# they can't drift out of sync. Clipped to X>=0 / Y>=0 when the
# corresponding symmetry flag is on, matching the box's own build_hollow_
# cube_mesh clipping below.
ATTACH_XMIN, ATTACH_XMAX = -TRUNK_DIVS_X * CELL / 2, TRUNK_DIVS_X * CELL / 2
ATTACH_YMIN, ATTACH_YMAX = -TRUNK_DIVS_Y * CELL / 2, TRUNK_DIVS_Y * CELL / 2
if SYMMETRY_X:
    ATTACH_XMIN = 0.0
if SYMMETRY_Y:
    ATTACH_YMIN = 0.0

YOUNG_MODULUS = PARAMS['young_modulus']
POISSON_RATIO = PARAMS['poisson_ratio']
DENSITY = PARAMS['density']

TARGET_PRESSURE = PARAMS['target_pressure']
RAMP_TIME = PARAMS['ramp_time']
INSTANT = PARAMS['instant']

GRAVITY = [0.0, 0.0, -9.81] if PARAMS['gravity_enabled'] else [0.0, 0.0, 0.0]
DT = 0.01

CAMERA_DISTANCE_FACTOR = 2.741  # distance from the box's centre to the camera,
                                 # as a multiple of its largest dimension
CAMERA_DISTANCE = CAMERA_DISTANCE_FACTOR * max(LENGTH, WIDTH, HEIGHT)  # [m]

CUTAWAY_FACE = '-Y'       # which outer/interior face the cutout window opens
                           # in, when `cutaway`/`interior_cutaway` is true in
                           # params.json ('-Y' faces the camera). Names:
                           # '+X','-X','+Y','-Y','+Z','-Z'.
CUTAWAY_FACES = {CUTAWAY_FACE} if PARAMS['cutaway'] else set()
INTERIOR_CUTAWAY_FACES = {CUTAWAY_FACE} if PARAMS['interior_cutaway'] else set()

WALL_ALPHA = PARAMS['wall_alpha']
INTERIOR_ALPHA = PARAMS['interior_alpha']
SHOW_PRESSURE_OVERLAY = PARAMS['show_pressure_overlay']
SHOW_FEM_ELEMENTS = PARAMS['show_fem_elements']


# ----------------------------------------------------------------------
# Procedural mesh generation
# ----------------------------------------------------------------------
# Each face maps its two tangential (a, b) loop indices to a (i, j, k)
# grid coordinate, plus which of divs_x/divs_y/divs_z each of a/b ranges
# over -- e.g. '+X'/'-X' are spanned by Y and Z, so a loop 0..divs_y and
# b loops 0..divs_z.
_BOX_FACES = {
    '+X': (lambda a, b, dx, dy, dz: (dx, a, b), False, 'y', 'z'),
    '-X': (lambda a, b, dx, dy, dz: (0, a, b), True, 'y', 'z'),
    '+Y': (lambda a, b, dx, dy, dz: (a, dy, b), True, 'x', 'z'),
    '-Y': (lambda a, b, dx, dy, dz: (a, 0, b), False, 'x', 'z'),
    '+Z': (lambda a, b, dx, dy, dz: (a, b, dz), False, 'x', 'y'),
    '-Z': (lambda a, b, dx, dy, dz: (a, b, 0), True, 'x', 'y'),
}


def _box_surface(xmin, ymin, zmin, xmax, ymax, zmax, divs_x, divs_y, divs_z, skip_faces=()):
    """Axis-aligned box surface mesh with outward normals, `divs_x`/
    `divs_y`/`divs_z` quads per edge along X/Y/Z respectively (so all-1
    gives the original 8-vertex, 12-triangle box). Points are shared
    across adjacent faces (no seam duplicates), and each face's own grid
    points are generated in the same left-to-right, bottom-to-top order
    the shell mesh uses, so with matching divs they land exactly on the
    shell's inner boundary nodes instead of just the 8 corners.

    `skip_faces`: names from {'+X','-X','+Y','-Y','+Z','-Z'} to omit --
    e.g. for a cutaway view that lets you see inside the box.
    """
    point_index = {}
    points = []

    def get_point(i, j, k):
        key = (i, j, k)
        idx = point_index.get(key)
        if idx is None:
            idx = len(points)
            point_index[key] = idx
            points.append([
                xmin + (xmax - xmin) * i / divs_x,
                ymin + (ymax - ymin) * j / divs_y,
                zmin + (zmax - zmin) * k / divs_z,
            ])
        return idx

    tris = []
    axis_divs = {'x': divs_x, 'y': divs_y, 'z': divs_z}

    def add_face(ijk_of, flip, divs_a, divs_b):
        for a in range(divs_a):
            for b in range(divs_b):
                p00 = get_point(*ijk_of(a, b, divs_x, divs_y, divs_z))
                p10 = get_point(*ijk_of(a + 1, b, divs_x, divs_y, divs_z))
                p11 = get_point(*ijk_of(a + 1, b + 1, divs_x, divs_y, divs_z))
                p01 = get_point(*ijk_of(a, b + 1, divs_x, divs_y, divs_z))
                if not flip:
                    tris.append([p00, p10, p11])
                    tris.append([p00, p11, p01])
                else:
                    tris.append([p00, p01, p11])
                    tris.append([p00, p11, p10])

    for name, (ijk_of, flip, axis_a, axis_b) in _BOX_FACES.items():
        if name not in skip_faces:
            add_face(ijk_of, flip, axis_divs[axis_a], axis_divs[axis_b])

    return points, tris


def _fan_cap_face(face_name, xmin, ymin, zmin, xmax, ymax, zmax, divs_x, divs_y, divs_z):
    """Closes one `_BOX_FACES` face with a fan triangulation of only its
    own boundary loop (at the face's full a/b resolution, i.e. the same
    resolution `_box_surface` would use there) -- no interior points are
    invented. Every returned point duplicates a point on the loop that
    `_box_surface` already places on one of the box's OTHER (adjacent,
    still-built) faces, so -- unlike a point genuinely inside the face --
    it coincides exactly with real material there; see
    `_capped_box_surface`'s docstring for why that matters.
    """
    ijk_of, flip, axis_a, axis_b = _BOX_FACES[face_name]
    axis_divs = {'x': divs_x, 'y': divs_y, 'z': divs_z}
    divs_a, divs_b = axis_divs[axis_a], axis_divs[axis_b]

    def world(i, j, k):
        return [xmin + (xmax - xmin) * i / divs_x,
                ymin + (ymax - ymin) * j / divs_y,
                zmin + (zmax - zmin) * k / divs_z]

    # Walk the perimeter once (each corner visited exactly once): bottom
    # edge left-to-right, right edge bottom-to-top, top edge right-to-
    # left, left edge top-to-bottom.
    loop_ab = ([(a, 0) for a in range(divs_a)] +
               [(divs_a, b) for b in range(divs_b)] +
               [(a, divs_b) for a in range(divs_a, 0, -1)] +
               [(0, b) for b in range(divs_b, 0, -1)])
    if flip:
        loop_ab.reverse()  # match _box_surface's own winding for this face

    points = [world(*ijk_of(a, b, divs_x, divs_y, divs_z)) for a, b in loop_ab]
    tris = [[0, i, i + 1] for i in range(1, len(points) - 1)]
    return points, tris


def _capped_box_surface(xmin, ymin, zmin, xmax, ymax, zmax, divs_x, divs_y, divs_z,
                         coarse_faces=(), skip_faces=()):
    """Like `_box_surface`, but any face named in `coarse_faces` is
    closed with `_fan_cap_face` (using only its own boundary-loop
    points) instead of being subdivided into a full interior grid.

    This exists for a symmetry cut face on the cavity surface: points on
    its boundary loop coincide exactly with real shell vertices (mapped
    correctly by BarycentricMapping and held on the plane by the
    symmetry boundary condition), but a fully subdivided cap's *interior*
    points generally don't -- there's no shell material there, only the
    hollow interior -- so BarycentricMapping has nothing valid to
    interpolate them from and they drift off the cut plane as the
    simulation runs. Fan-capping from the boundary loop alone sidesteps
    the problem entirely: the enclosed-volume computation only depends
    on a flat cap's boundary shape, not on how (or whether) its interior
    is subdivided, so nothing is lost by dropping those unmappable
    interior points.
    """
    fine_skip = set(coarse_faces) | set(skip_faces)
    points, tris = _box_surface(xmin, ymin, zmin, xmax, ymax, zmax,
                                 divs_x, divs_y, divs_z, skip_faces=fine_skip)
    for face_name in coarse_faces:
        cap_points, cap_tris = _fan_cap_face(face_name, xmin, ymin, zmin, xmax, ymax, zmax,
                                              divs_x, divs_y, divs_z)
        offset = len(points)
        points = points + cap_points
        tris = tris + [[i + offset for i in t] for t in cap_tris]
    return points, tris


def _mirror_surface(points, tris, axes, mirror_values):
    """Rest-pose mirror of a (points, tris) surface across one or more
    axis-aligned planes (`axes[i]`'s coordinate reflected about
    `mirror_values[i]`) -- the initial shape for a mirror_visual copy
    (see createScene): there's no real shell material on the mirrored
    side to map its LIVE position from (same reason the cavity's
    symmetry cap needed _capped_box_surface), so its per-frame position
    is instead driven directly by MirrorController, reflecting the real
    surface's current position the same way each step. Triangle winding
    is reversed for an odd number of axes and kept as-is for an even
    number, so outward normals stay correct after reflection.
    """
    mirrored = []
    for p in points:
        q = list(p)
        for axis, value in zip(axes, mirror_values):
            q[axis] = 2 * value - q[axis]
        mirrored.append(q)
    if len(axes) % 2 == 1:
        tris = [[t[0], t[2], t[1]] for t in tris]
    else:
        tris = [list(t) for t in tris]
    return mirrored, tris


def _build_axis_arrow(axis, shaft_len, head_len, shaft_half, head_half):
    """Builds a simple arrow mesh (a box shaft plus a 4-sided pyramid
    head) starting at the origin and pointing along the given world axis
    -- for the static X/Y direction indicators added in createScene().
    `axis`: 0 for +X, 1 for +Y. The shaft's own end cap (facing the tip)
    is left out -- since `head_half` > `shaft_half`, the pyramid's base
    already fully covers that opening, and including both would leave two
    coincident, overlapping quads there (z-fighting, as elsewhere in this
    file).
    """
    if axis == 0:
        pts, tris = _box_surface(0.0, -shaft_half, -shaft_half,
                                  shaft_len, shaft_half, shaft_half, 1, 1, 1,
                                  skip_faces={'+X'})
        base = [[shaft_len, -head_half, -head_half],
                [shaft_len,  head_half, -head_half],
                [shaft_len,  head_half,  head_half],
                [shaft_len, -head_half,  head_half]]
        tip = [shaft_len + head_len, 0.0, 0.0]
    elif axis == 1:
        pts, tris = _box_surface(-shaft_half, 0.0, -shaft_half,
                                  shaft_half, shaft_len, shaft_half, 1, 1, 1,
                                  skip_faces={'+Y'})
        base = [[-head_half, shaft_len, -head_half],
                [ head_half, shaft_len, -head_half],
                [ head_half, shaft_len,  head_half],
                [-head_half, shaft_len,  head_half]]
        tip = [0.0, shaft_len + head_len, 0.0]
    else:
        raise ValueError("axis must be 0 (X) or 1 (Y)")

    base_start = len(pts)
    pts = pts + base + [tip]
    b0, b1, b2, b3, t = (base_start, base_start + 1, base_start + 2,
                          base_start + 3, base_start + 4)
    tris = tris + [
        [b0, b1, t], [b1, b2, t], [b2, b3, t], [b3, b0, t],  # 4 pyramid sides
        [b0, b2, b1], [b0, b3, b2],                          # base cap
    ]
    return pts, tris


def _diagonal_stroke_box(x0, y0, x1, y1, half_width, z0, z1):
    """Builds a rectangular prism (8 corners, 12 triangles) for a single
    straight stroke of a letter glyph: runs from (x0, y0) to (x1, y1) in
    the world XY plane with the given half-width perpendicular to that
    run (in-plane), extruded vertically from z0 to z1. Needed because the
    X/Y axis-label letters below have diagonal strokes, which
    `_box_surface` (axis-aligned only) can't build directly.

    Reuses `_box_surface`'s own canonical unit box (correct winding
    already) and remaps its points into the stroke's own (along the run,
    across the run, up) frame -- that frame has the same handedness as
    world XYZ by construction (the "across" direction is the "along"
    direction rotated +90 degrees in-plane), so the remapped winding
    stays outward-correct too.
    """
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length  # unit vector along the run
    nx, ny = -uy, ux                   # unit vector across the run (in-plane)

    unit_pts, tris = _box_surface(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1, 1)
    pts = []
    for u, v, w in unit_pts:
        along, side = u, 2 * v - 1
        pts.append([
            x0 + along * dx + side * half_width * nx,
            y0 + along * dy + side * half_width * ny,
            z0 + w * (z1 - z0),
        ])
    return pts, tris


def _build_letter_glyph(strokes, stroke_half, z0, z1):
    """Builds a simple flat 3D "glyph" out of straight strokes (each its
    own closed prism from `_diagonal_stroke_box`, not welded to the
    others -- fine for a purely visual, non-simulated shape, same
    approach as the arrow shaft/head above). `strokes`: list of
    (x0, y0, x1, y1) stroke centerlines in world XY. Used for the big
    X/Y axis-label letters in createScene().
    """
    pts, tris = [], []
    for x0, y0, x1, y1 in strokes:
        stroke_pts, stroke_tris = _diagonal_stroke_box(x0, y0, x1, y1, stroke_half, z0, z1)
        offset = len(pts)
        pts += stroke_pts
        tris += [[i + offset for i in t] for t in stroke_tris]
    return pts, tris


def _letter_x_strokes(cx, cy, half_size):
    """Two crossing diagonal strokes forming a letter X, centered at
    (cx, cy) in world XY, spanning +-half_size."""
    return [
        (cx - half_size, cy - half_size, cx + half_size, cy + half_size),
        (cx - half_size, cy + half_size, cx + half_size, cy - half_size),
    ]


def _letter_y_strokes(cx, cy, half_size):
    """Two upper arms meeting at the centre plus a stem down to the
    bottom, forming a letter Y, centered at (cx, cy) in world XY,
    spanning +-half_size."""
    return [
        (cx - half_size, cy + half_size, cx, cy),
        (cx + half_size, cy + half_size, cx, cy),
        (cx, cy, cx, cy - half_size),
    ]


# Kuhn (body-diagonal) decomposition of a hexahedron into 6 tetrahedra,
# expressed as local corner indices 0..7 in the order:
#   0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0) 4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
# Every hexahedron in the structured grid uses this same local-corner
# template (never a reflected/rotated one), which guarantees the diagonal
# chosen on any face shared by two neighbouring hexahedra agrees from both
# sides -- so the resulting tet mesh is conforming (no cracks/overlaps).
_KUHN_TETS = [
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
]


def _symmetric_axis_bounds(lo, hi, divs, symmetric, keep_lower_half):
    """Clips (lo, hi) and its cell count `divs` to one half about their
    own midpoint when `symmetric` is set -- `keep_lower_half` selects
    whether the kept half is [lo, mid] or [mid, hi]. `divs` must be even
    when `symmetric` (guaranteed by _resolve_geometry's rounding) so the
    cut lands exactly on a grid boundary rather than through a block.
    Used to clip both the shell mesh's own per-axis cell range and the
    cavity/outer-skin box bounds passed to _box_surface, at each active
    symmetry plane, in build_hollow_cube_mesh below.
    """
    if not symmetric:
        return lo, hi, divs
    mid = (lo + hi) / 2.0
    half_divs = divs // 2
    return (lo, mid, half_divs) if keep_lower_half else (mid, hi, half_divs)


def build_hollow_cube_mesh(divs_x=DIVS_X, divs_y=DIVS_Y, divs_z=DIVS_Z,
                            wall_x=WALL_X, wall_y=WALL_Y, wall_z=WALL_Z, cell=CELL,
                            z_offset=0.0,
                            symmetry_x=False, symmetry_y=False, symmetry_z=False,
                            symmetry_buffer_blocks=0,
                            outer_skip_faces=(), inner_skip_faces=()):
    """Builds a tetrahedral shell mesh of a hollow box out of cube-shaped
    blocks of edge length `cell`, plus the triangulated inner-cavity
    surface used for the pressure load, plus the outer skin and
    interior-volume surfaces used for visualization.

    The box spans `divs_x`/`divs_y`/`divs_z` blocks along X/Y/Z, with its
    bottom face at world Z=`z_offset` (0 sits it directly on the floor).
    The shell is `wall_x` blocks thick on the +-X faces, `wall_y` on +-Y,
    `wall_z` on +-Z (top/bottom); the interior
    (divs_x - 2*wall_x) x (divs_y - 2*wall_y) x (divs_z - 2*wall_z) blocks
    are left unmeshed and form the air cavity. Each shell grid cell is
    split into 6 tetrahedra (Kuhn triangulation). Grid points that aren't
    a corner of any shell cell (the free-floating interior of the cavity)
    are dropped so the FEM mesh has no unused DOFs.

    `symmetry_x`/`symmetry_y`/`symmetry_z`: when set, only the +X/+Y/
    bottom-Z half of the box is built (cut at its own midplane -- `divs_x`
    etc. must be even, see _resolve_geometry), roughly halving the element
    count per axis enabled. The cavity/outer-skin/interior-visual boxes
    are clipped to the same half, so the cavity surface stays closed
    (needed for SurfacePressureConstraint's volume computation) with a
    real face exactly at the cut instead of a gap. The caller is
    responsible for adding the actual symmetry boundary condition (fixing
    displacement normal to each cut plane) -- this function only builds
    the reduced geometry.

    `symmetry_buffer_blocks`: extends the *shell FEM mesh only* (not the
    cavity/outer/inner visual surfaces, which stay cut exactly at the
    plane) this many extra blocks past each active symmetry plane, as
    ordinary, freely-deforming material -- no special treatment, since
    the wall pattern is already mirror-symmetric about the cut by
    construction. This gives the symmetry-constrained row a natural
    elastic neighbour on both sides (matching what it would see in the
    unreduced model) instead of being rigid on one side and free on the
    other, which otherwise shows up as a visible kink right at the cut.
    The buffer's own far edge is left unconstrained/free; that's a much
    smaller, more localized artifact that decays with distance from the
    plane. 0 (default) reproduces the original exact-cut behaviour.

    `outer_skip_faces`/`inner_skip_faces`: passed straight to `_box_surface`
    for the outer skin / interior-volume visual surfaces respectively --
    names from {'+X','-X','+Y','-Y','+Z','-Z'} to leave out, e.g. for a
    cutaway view into the cavity. Purely visual: `cavity_pts`/`cavity_tris`
    (used for the actual pressure load) are always the complete, closed
    surface regardless of `inner_skip_faces` -- an open cavity mesh would
    make the enclosed-volume computation meaningless.

    Returns:
        positions   : list of [x, y, z] shell mesh points
        tetrahedra  : list of 4-index tuples (compact indices into positions)
        cavity_pts  : list of [x, y, z] cavity surface vertices (closed, for
                      SurfacePressureConstraint)
        cavity_tris : list of 3-index tuples (outward-normal triangles)
        outer_pts   : list of [x, y, z] outer skin visual surface vertices
        outer_tris  : list of 3-index tuples (outward-normal triangles)
        inner_pts   : list of [x, y, z] interior-volume visual surface
                      vertices (same geometry as cavity_pts/cavity_tris,
                      but may have `inner_skip_faces` cut away)
        inner_tris  : list of 3-index tuples (outward-normal triangles)
    """
    length, width, height = divs_x * cell, divs_y * cell, divs_z * cell
    nx, ny, nz = divs_x + 1, divs_y + 1, divs_z + 1

    def pidx(i, j, k):
        return (i * ny + j) * nz + k

    raw_positions = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                raw_positions.append([-length / 2 + i * cell,
                                       -width / 2 + j * cell,
                                       z_offset + k * cell])

    # Symmetry keeps the +X half, the +Y half, and the bottom (trunk-
    # attached) Z half respectively, each extended by symmetry_buffer_
    # blocks extra cells past the true midplane -- unreferenced points on
    # the discarded side are dropped below, same as the deep cavity
    # interior.
    buf = symmetry_buffer_blocks
    i_lo, i_hi = (max(0, divs_x // 2 - buf), divs_x) if symmetry_x else (0, divs_x)
    j_lo, j_hi = (max(0, divs_y // 2 - buf), divs_y) if symmetry_y else (0, divs_y)
    k_lo, k_hi = (0, min(divs_z, divs_z // 2 + buf)) if symmetry_z else (0, divs_z)

    raw_tetrahedra = []
    for i in range(i_lo, i_hi):
        for j in range(j_lo, j_hi):
            for k in range(k_lo, k_hi):
                is_shell = (i < wall_x or i >= divs_x - wall_x or
                            j < wall_y or j >= divs_y - wall_y or
                            k < wall_z or k >= divs_z - wall_z)
                if not is_shell:
                    continue
                corners = [
                    pidx(i,     j,     k),
                    pidx(i + 1, j,     k),
                    pidx(i + 1, j + 1, k),
                    pidx(i,     j + 1, k),
                    pidx(i,     j,     k + 1),
                    pidx(i + 1, j,     k + 1),
                    pidx(i + 1, j + 1, k + 1),
                    pidx(i,     j + 1, k + 1),
                ]
                for a, b, c, d in _KUHN_TETS:
                    raw_tetrahedra.append([corners[a], corners[b], corners[c], corners[d]])

    # Drop unreferenced (deep-interior, or cut away by symmetry) points
    # and remap indices compactly.
    used = sorted({idx for tet in raw_tetrahedra for idx in tet})
    remap = {old: new for new, old in enumerate(used)}
    positions = [raw_positions[old] for old in used]
    tetrahedra = [[remap[idx] for idx in tet] for tet in raw_tetrahedra]

    # Inner cavity surface: `wall_x`/`wall_y`/`wall_z` cells in from the
    # shell on each pair of faces, subdivided at the shell's own cell
    # resolution (cavity_divs_* quads per face) so its vertices coincide
    # exactly with the shell's inner boundary grid points -- both for an
    # exact BarycentricMapping and so the pressure load is spread across
    # the whole inner wall instead of concentrated at the 8 box corners.
    # Clipped to the built half at each active symmetry plane -- no wall
    # thickness subtracted there, since it's an internal cut, not a real
    # exterior wall.
    wx, wy, wz = wall_x * cell, wall_y * cell, wall_z * cell
    xmin, xmax, cavity_divs_x = _symmetric_axis_bounds(
        -length / 2 + wx, length / 2 - wx, divs_x - 2 * wall_x, symmetry_x, keep_lower_half=False)
    ymin, ymax, cavity_divs_y = _symmetric_axis_bounds(
        -width / 2 + wy, width / 2 - wy, divs_y - 2 * wall_y, symmetry_y, keep_lower_half=False)
    zmin, zmax, cavity_divs_z = _symmetric_axis_bounds(
        z_offset + wz, z_offset + height - wz, divs_z - 2 * wall_z, symmetry_z, keep_lower_half=True)

    # Face(s) coinciding with an active symmetry cut plane -- '-X'/'-Y'
    # since those axes keep their +-half (so the cut sits at the box's
    # own min bound), '+Z' since Z keeps its bottom half (cut at the max
    # bound). Used below to coarsen the cavity cap (see
    # _capped_box_surface) and to leave the cut face out of the purely
    # visual outer-skin/interior-volume surfaces entirely (like an
    # automatic cutaway -- they don't need to be closed, so there's no
    # reason to give them the same unmappable-interior-points problem).
    symmetry_cut_faces = set()
    if symmetry_x:
        symmetry_cut_faces.add('-X')
    if symmetry_y:
        symmetry_cut_faces.add('-Y')
    if symmetry_z:
        symmetry_cut_faces.add('+Z')

    cavity_pts, cavity_tris = _capped_box_surface(xmin, ymin, zmin, xmax, ymax, zmax,
                                                   cavity_divs_x, cavity_divs_y, cavity_divs_z,
                                                   coarse_faces=symmetry_cut_faces)

    # Interior-volume visual surface: same box as the cavity above (for
    # the physics), but a separate mesh so it can have a cutaway window
    # without touching the closed surface SurfacePressureConstraint needs.
    # Symmetry cut faces are left open (skipped), not capped -- purely
    # visual, so there's no need for it to stay closed there; capping it
    # would just add a redundant flat patch sitting right where a
    # mirror_visual copy belongs (z-fighting with it) when mirroring is
    # on, for no benefit when it's off either.
    inner_pts, inner_tris = _box_surface(xmin, ymin, zmin, xmax, ymax, zmax,
                                          cavity_divs_x, cavity_divs_y, cavity_divs_z,
                                          skip_faces=set(inner_skip_faces) | symmetry_cut_faces)

    # Outer skin surface, at the shell's own full grid resolution so its
    # vertices coincide with the shell's actual outer boundary nodes
    # (exact BarycentricMapping, same reasoning as the cavity surface).
    # Clipped and left open the same way as the interior-volume surface
    # above.
    outer_xmin, outer_xmax, outer_divs_x = _symmetric_axis_bounds(
        -length / 2, length / 2, divs_x, symmetry_x, keep_lower_half=False)
    outer_ymin, outer_ymax, outer_divs_y = _symmetric_axis_bounds(
        -width / 2, width / 2, divs_y, symmetry_y, keep_lower_half=False)
    outer_zmin, outer_zmax, outer_divs_z = _symmetric_axis_bounds(
        z_offset, z_offset + height, divs_z, symmetry_z, keep_lower_half=True)
    outer_pts, outer_tris = _box_surface(outer_xmin, outer_ymin, outer_zmin,
                                          outer_xmax, outer_ymax, outer_zmax,
                                          outer_divs_x, outer_divs_y, outer_divs_z,
                                          skip_faces=set(outer_skip_faces) | symmetry_cut_faces)

    return positions, tetrahedra, cavity_pts, cavity_tris, outer_pts, outer_tris, inner_pts, inner_tris


def _mesh_volume(positions, tetrahedra):
    """Total volume of a tetrahedral mesh, by summing each tet's
    (unsigned) volume. Used for the shell's mass instead of an analytic
    formula based on nominal wall/cavity dimensions, since that formula
    would need a special case for every symmetry/symmetry_buffer_blocks
    combination to stay exact -- summing the actually-built tets is
    always correct regardless.
    """
    total = 0.0
    for t in tetrahedra:
        p0, p1, p2, p3 = (positions[i] for i in t)
        ax, ay, az = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
        bx, by, bz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
        cx, cy, cz = p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]
        cross_x = by * cz - bz * cy
        cross_y = bz * cx - bx * cz
        cross_z = bx * cy - by * cx
        total += abs(ax * cross_x + ay * cross_y + az * cross_z) / 6.0
    return total


# ----------------------------------------------------------------------
# Controller: prints the live camera state whenever it changes, so it's
# possible to tell from the console log whether something in the GUI
# (e.g. an auto-fit-to-scene-bounds on load) is overriding the explicit
# position/orientation/distance set in createScene().
# ----------------------------------------------------------------------
class CameraPrinter(Sofa.Core.Controller):
    def __init__(self, camera, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.camera = camera
        self.last = None

    def onAnimateBeginEvent(self, event):
        pos = tuple(round(float(c), 5) for c in self.camera.position.value)
        dist = round(float(self.camera.distance.value), 5)
        state = (pos, dist)
        if state != self.last:
            self.last = state
            print(f"[camera] position={list(pos)}  distance={dist}")


# ----------------------------------------------------------------------
# Controller: live-mirrors one visual surface's vertex positions across
# one or more symmetry planes each step, so mirror_visual (see
# createScene) shows the complete deformed shape even though only a
# fraction of it is actually simulated. There's no BarycentricMapping
# equivalent for this -- the mirrored side has no real shell material to
# interpolate from -- so this just copies+reflects the real (already
# correctly mapped) surface's current position directly.
# ----------------------------------------------------------------------
class MirrorController(Sofa.Core.Controller):
    def __init__(self, source, target, axes, mirror_values, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.source = source
        self.target = target
        self.axes = axes
        self.mirror_values = mirror_values

    def onAnimateEndEvent(self, event):
        mirrored = []
        for p in self.source.position.value:
            q = list(p)
            for axis, value in zip(self.axes, self.mirror_values):
                q[axis] = 2 * value - q[axis]
            mirrored.append(q)
        self.target.position.value = mirrored


# ----------------------------------------------------------------------
# Controller: drives the cavity pressure towards a live-editable target,
# either instantly or ramped, and logs volume growth. `targetPressure`,
# `rampTime` and `instant` are exposed as SOFA Data fields, so they show
# up -- and can be edited live, mid-simulation -- in the GUI's Scene
# Graph / properties panel (select the "PressureRamp" node).
# ----------------------------------------------------------------------
class PressureRamp(Sofa.Core.Controller):
    def __init__(self, root, pressure_constraint, target_pressure, ramp_time, instant=False, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.root = root
        self.pressure = pressure_constraint
        self.addData(name='targetPressure', type='float', value=target_pressure,
                     help='Target gauge pressure in the cavity [Pa]. Live-editable.')
        self.addData(name='rampTime', type='float', value=ramp_time,
                     help='Time [s] to go from 0 to targetPressure when not instant. '
                          'Also limits how fast a live change in targetPressure is followed.')
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
            max_step = (abs(target) if target != 0 else self.pressure.pressure.value) / ramp_time * dt
            max_step = max(max_step, 1e-9)
            diff = target - self.current_pressure
            if abs(diff) <= max_step:
                self.current_pressure = target
            else:
                self.current_pressure += max_step if diff > 0 else -max_step

        self.pressure.value.value = [self.current_pressure]

    def onAnimateEndEvent(self, event):
        if self.time + 1e-9 < self.next_print:
            return
        self.next_print += 0.2
        vol = self.pressure.cavityVolume.value
        vol0 = self.pressure.initialCavityVolume.value
        applied = self.pressure.pressure.value
        growth = (vol / vol0 - 1.0) * 100.0
        mode = 'instant' if self.instant.value else 'ramp'
        print(f"t={self.time:5.2f}s  [{mode}] target={self.targetPressure.value/1000.0:6.2f} kPa  "
              f"P_applied={applied/1000.0:6.2f} kPa  "
              f"cavity volume={vol*1e6:8.2f} cm^3  (volume +{growth:5.1f}%)")


# ----------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------
def createScene(rootNode):
    launch_params_editor()

    plugins = [
        "Sofa.Component.AnimationLoop",                # FreeMotionAnimationLoop
        "Sofa.Component.Constraint.Lagrangian.Correction",  # LinearSolverConstraintCorrection
        "Sofa.Component.Constraint.Lagrangian.Solver",      # BlockGaussSeidelConstraintSolver
        "Sofa.Component.Constraint.Projective",        # PartialFixedProjectiveConstraint (symmetry BCs)
        "Sofa.Component.Engine.Select",                # BoxROI
        "Sofa.Component.LinearSolver.Direct",          # EigenSimplicialLDLT
        "Sofa.Component.Mapping.Linear",               # BarycentricMapping
        "Sofa.Component.Mass",                         # UniformMass
        "Sofa.Component.MechanicalLoad",               # PlaneForceField (symmetry no-crossing)
        "Sofa.Component.ODESolver.Backward",           # EulerImplicitSolver
        "Sofa.Component.SolidMechanics.FEM.Elastic",   # FastTetrahedralCorotationalForceField
        "Sofa.Component.SolidMechanics.Spring",        # RestShapeSpringsForceField
        "Sofa.Component.StateContainer",               # MechanicalObject
        "Sofa.Component.Topology.Container.Constant",  # MeshTopology (cavity, outer skin)
        "Sofa.Component.Topology.Container.Dynamic",   # TetrahedronSetTopologyContainer
        "Sofa.Component.Setting",                      # BackgroundSetting
        "Sofa.Component.Visual",                       # VisualStyle
        "Sofa.GL.Component.Rendering3D",                # OglModel
        "Sofa.GL.Component.Shader",                     # LightManager, DirectionalLight
        "SoftRobots",                                   # SurfacePressureConstraint
    ]
    for p in plugins:
        rootNode.addObject('RequiredPlugin', name='req_' + p.replace('.', '_'), pluginName=p)

    # Plain background instead of the GUI's default SOFA-branded backdrop.
    # `image` must be cleared explicitly -- it defaults to SOFA's own
    # branding texture and takes priority over `color` when set. Note:
    # the ImGui GUI's branded backdrop is baked into its own rendering
    # code and ignores this component entirely (confirmed: GUIManager's
    # setBackgroundImage() reaches it with no error but has no visible
    # effect either) -- use `-g glfw` for a GUI that respects this.
    rootNode.addObject('BackgroundSetting', color=[0.78, 0.87, 0.96, 1.0], image="")

    interaction_flag = 'showInteractionForceFields' if SHOW_PRESSURE_OVERLAY else 'hideInteractionForceFields'
    force_fields_flag = 'showForceFields' if SHOW_FEM_ELEMENTS else 'hideForceFields'
    rootNode.addObject('VisualStyle',
                        displayFlags='showVisualModels hideBehaviorModels '
                                     'hideCollisionModels hideBoundingCollisionModels '
                                     f'{force_fields_flag} {interaction_flag} hideWireframe')

    # Pull the default camera back, up and to the side instead of the
    # tight auto-fit view (scaled off the box's largest dimension so it
    # stays well-framed if the geometry changes). SOFA's camera derives
    # orientation from position/lookAt assuming a Y-up world, but this
    # scene is Z-up (gravity = -Z), so we compute and set the orientation
    # quaternion explicitly with world-up = +Z to get a proper
    # top-down-tilted view instead of the mismatched (upside-down-looking)
    # default.
    cam_lookat = [0.0, 0.0, BASE_OFFSET + HEIGHT * 0.5]
    cam_dir = (4.5, -4.5, 3.25)  # back / to the side / up, un-normalized
    cam_dir_norm = math.sqrt(sum(c * c for c in cam_dir))
    cam_distance = CAMERA_DISTANCE
    cam_position = [l + cam_distance * c / cam_dir_norm for l, c in zip(cam_lookat, cam_dir)]
    camera = rootNode.addObject('InteractiveCamera', name='camera',
                                 position=cam_position,
                                 distance=cam_distance,
                                 orientation=_look_at_quaternion(cam_position, cam_lookat, world_up=(0.0, 0.0, 1.0)),
                                 activated=True)
    rootNode.addObject(CameraPrinter(camera, name='CameraPrinter'))

    # Explicit lighting from above (matching the Z-up "up" direction),
    # instead of relying on the GUI's default light placement.
    # `direction` turned out to mean "vector pointing towards the light",
    # not "the direction its rays travel" -- a positive Z is what puts
    # the light in the sky (confirmed: the -1.0 we had before was
    # lighting surfaces from underneath instead of from above).
    rootNode.addObject('LightManager', ambient=[0.35, 0.35, 0.35, 1.0])
    rootNode.addObject('DirectionalLight', direction=[-0.3, -0.3, 1.0])

    # Static ground plane (purely visual -- not simulated). Offset 0.1mm
    # below z=0 so it isn't exactly coplanar with the trunk's bottom face
    # -- coplanar triangles z-fight (flicker) since neither reliably wins
    # depth testing.
    floor_half_x = FLOOR_SIZE / 2
    floor_half_y = FLOOR_SIZE / 2
    floor_z = -1e-4
    floor_pts = [
        [-floor_half_x, -floor_half_y, floor_z],
        [floor_half_x, -floor_half_y, floor_z],
        [floor_half_x, floor_half_y, floor_z],
        [-floor_half_x, floor_half_y, floor_z],
    ]
    floor_tris = [[0, 1, 2], [0, 2, 3]]
    floor = rootNode.addChild('floor')
    floor.addObject('MeshTopology', name='topo', position=floor_pts, triangles=floor_tris)
    floor.addObject('OglModel', name='visual', src='@topo', color=[0.08, 0.28, 0.1, 1.0])

    # Static "trunk" (purely visual -- not simulated), a rigid brown block
    # spanning the floor-to-box gap created by BASE_OFFSET, filling exactly
    # the ATTACH_* footprint -- i.e. showing which part of the bottom face
    # is actually clamped to fixed world by the BoxROI/RestShapeSprings
    # below, rather than implying the whole base is attached when it isn't.
    # Its top face is nudged 0.1mm below the cube's actual bottom (at
    # BASE_OFFSET) for the same z-fighting reason as the floor's offset
    # above -- exactly coplanar triangles flicker since neither reliably
    # wins depth testing.
    trunk_pts, trunk_tris = _box_surface(ATTACH_XMIN, ATTACH_YMIN, 0.0,
                                          ATTACH_XMAX, ATTACH_YMAX, BASE_OFFSET - 1e-4,
                                          1, 1, 1)
    trunk = rootNode.addChild('trunk')
    trunk.addObject('MeshTopology', name='topo', position=trunk_pts, triangles=trunk_tris)
    trunk.addObject('OglModel', name='visual', src='@topo', color=[0.4, 0.26, 0.13, 1.0])

    # Static X (red) / Y (green) direction arrows from the world origin,
    # lying flat on the floor (purely visual -- not simulated). Each
    # arrow's tip clears its OWN axis's box half-extent by a comfortable
    # margin, rather than both sharing one length derived from the box's
    # single largest dimension -- otherwise whichever axis happens to be
    # the box's longest ends up with its own arrow buried under the box
    # footprint instead of visibly poking out past it (tip only ~20%
    # beyond the edge). Thickness/head size still come from the shared
    # reference scale so both arrows look consistently proportioned.
    # Raised 0.1mm above the floor for the same z-fighting reason as the
    # trunk/floor offsets above.
    arrow_ref = 0.6 * max(LENGTH, WIDTH, HEIGHT)
    arrow_head_len = arrow_ref * 0.2
    arrow_shaft_half = arrow_ref * 0.015
    arrow_head_half = arrow_shaft_half * 3.0
    arrow_z = 1e-4
    arrow_clearance = 1.6  # tip extends this many times the box's own half-extent
    arrow_totals = {
        0: max(arrow_ref, arrow_clearance * (LENGTH / 2) + arrow_head_len),
        1: max(arrow_ref, arrow_clearance * (WIDTH / 2) + arrow_head_len),
    }

    for name, axis, color in (('arrow_x', 0, [0.9, 0.1, 0.1, 1.0]),
                               ('arrow_y', 1, [0.1, 0.8, 0.1, 1.0])):
        arrow_shaft_len = arrow_totals[axis] - arrow_head_len
        pts, tris = _build_axis_arrow(axis, arrow_shaft_len, arrow_head_len,
                                       arrow_shaft_half, arrow_head_half)
        pts = [[p[0], p[1], p[2] + arrow_z] for p in pts]
        arrow = rootNode.addChild(name)
        arrow.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        arrow.addObject('OglModel', name='visual', src='@topo', color=color)

    # Big X (red) / Y (green) letters just past each arrow's tip,
    # continuing along the arrow's own line -- same flat-on-the-floor,
    # purely-visual treatment as the arrows.
    letter_half = arrow_ref * 0.09
    letter_gap = arrow_ref * 0.08
    letter_stroke_half = letter_half * 0.125
    letter_z0, letter_z1 = arrow_z, arrow_z + arrow_shaft_half * 2

    for name, strokes, color in (
        ('label_x', _letter_x_strokes(arrow_totals[0] + letter_gap + letter_half, 0.0, letter_half),
         [0.9, 0.1, 0.1, 1.0]),
        ('label_y', _letter_y_strokes(0.0, arrow_totals[1] + letter_gap + letter_half, letter_half),
         [0.1, 0.8, 0.1, 1.0]),
    ):
        pts, tris = _build_letter_glyph(strokes, letter_stroke_half, letter_z0, letter_z1)
        label = rootNode.addChild(name)
        label.addObject('MeshTopology', name='topo', position=pts, triangles=tris)
        label.addObject('OglModel', name='visual', src='@topo', color=color)

    rootNode.gravity = GRAVITY
    rootNode.dt = DT

    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('BlockGaussSeidelConstraintSolver', maxIterations=500, tolerance=1e-8)

    # ---- build the mesh -------------------------------------------------
    positions, tetrahedra, cavity_pts, cavity_tris, outer_pts, outer_tris, inner_pts, inner_tris = \
        build_hollow_cube_mesh(z_offset=BASE_OFFSET,
                                symmetry_x=SYMMETRY_X, symmetry_y=SYMMETRY_Y, symmetry_z=SYMMETRY_Z,
                                symmetry_buffer_blocks=SYMMETRY_BUFFER_BLOCKS,
                                outer_skip_faces=CUTAWAY_FACES, inner_skip_faces=INTERIOR_CUTAWAY_FACES)
    total_mass = DENSITY * _mesh_volume(positions, tetrahedra)

    # ---- cube (silicone shell) ------------------------------------------
    cube = rootNode.addChild('cube')
    cube.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=0.1, rayleighMass=0.1)
    cube.addObject('EigenSimplicialLDLT', template='CompressedRowSparseMatrixMat3x3')

    cube.addObject('TetrahedronSetTopologyContainer', name='container',
                    position=positions, tetrahedra=tetrahedra)
    cube.addObject('TetrahedronSetTopologyModifier')

    cube.addObject('MechanicalObject', name='dofs', template='Vec3')
    cube.addObject('UniformMass', totalMass=total_mass)
    cube.addObject('FastTetrahedralCorotationalForceField', name='FEM', method='large',
                    youngModulus=YOUNG_MODULUS, poissonRatio=POISSON_RATIO)
    # Note: this force field's built-in "show elements" draw path (the
    # showForceFields display flag) is hard-opaque -- its drawColor1-4
    # ignore the alpha channel entirely (no GL_BLEND anywhere in that
    # code path). For an actually-transparent view of the mesh, that
    # flag is off and the visu/ OglModel below (which does support
    # blending) is tinted blue with the wireframe shown on top instead.

    # Clamp the ATTACH_* footprint of the bottom face to a rigid base (like
    # a pneumatic actuator mounted on a fixed manifold) -- see the trunk
    # visual above, which shows this same region.
    cube.addObject('BoxROI', name='baseROI',
                    box=[ATTACH_XMIN - 1e-4, ATTACH_YMIN - 1e-4, BASE_OFFSET - 1e-4,
                         ATTACH_XMAX + 1e-4, ATTACH_YMAX + 1e-4, BASE_OFFSET + 1e-4],
                    drawBoxes=True, position='@dofs.rest_position')
    cube.addObject('RestShapeSpringsForceField', points='@baseROI.indices', stiffness=1e12)

    # Symmetry boundary conditions: points on each active cut plane are
    # locked to zero displacement in that plane's normal direction only
    # (PartialFixedProjectiveConstraint), so they stay on the plane while
    # still sliding freely within it -- this is what makes the reduced
    # model physically equivalent to simulating the full mirrored one.
    #
    # CAVEAT: only valid for a plane the loaded, constrained problem is
    # actually symmetric about. X and Y qualify here -- the trunk/base
    # clamp (ATTACH_*) is centered on both, and gravity/pressure don't
    # break either. Z does NOT: the box is clamped only at its bottom,
    # not top and bottom, so symmetry_z models a different (and wrong)
    # problem rather than a reduction of this one -- included for
    # completeness/experimentation, not because it's correct here.
    sym_eps = 1e-4
    sym_mid_z = BASE_OFFSET + HEIGHT / 2

    def _add_plane_guard(name, box, plane_normal, plane_d):
        # One-sided (bilateral=False, the default): only pushes back
        # points that have crossed to the wrong side of `plane_normal`/
        # `plane_d`, restricted to whichever nodes' REST position falls
        # in `box` (via a SubsetMapping) -- so it applies to EVERY such
        # point in the mesh, not just the ones that started exactly on
        # the plane, catching points elsewhere that fold across it under
        # large deformation (e.g. a collapse under negative pressure).
        # Stiff enough to act as a hard stop; the direct linear solver
        # (EigenSimplicialLDLT) handles that without instability, same
        # reasoning as the 1e12 base-clamp spring above.
        #
        # The rest-position `box` restriction (rather than applying to
        # the whole cube) matters for two distinct reasons depending on
        # the caller: when this axis's symmetry IS exploited and
        # symmetry_buffer_blocks > 0, the buffer's own nodes sit on the
        # "wrong" side of the plane AT REST BY DESIGN, so an unrestricted
        # guard would immediately treat them as "already crossed" and
        # slam them with the full 1e9 stiffness fighting their own
        # legitimate rest position (the original blow-up this avoids);
        # when symmetry is NOT exploited, `box` instead just picks out
        # "this side only" so the two guards (see below) don't fight
        # each other across the plane.
        cube.addObject('BoxROI', name=name + 'ROI', box=box, drawBoxes=False,
                        position='@dofs.rest_position')
        guard = cube.addChild(name)
        guard.addObject('MechanicalObject', name='mstate')
        guard.addObject('SubsetMapping', input='@..', output='@mstate',
                         indices='@../' + name + 'ROI.indices')
        guard.addObject('PlaneForceField', name=name + 'Plane',
                         normal=plane_normal, d=plane_d,
                         stiffness=1e9, damping=1e4, bilateral=False)

    # `pos_*`/`neg_*` describe the plane guard on each side of the axis's
    # midplane -- `pos` is also the "kept" (built) half when that axis's
    # symmetry IS exploited. Every axis's midplane physically exists
    # regardless of whether it's being exploited to halve the FEM element
    # count, so prevent_symmetry_crossing guards it either way: with one
    # guard restricted to the built half when exploited (see
    # _add_plane_guard above for why), or with both when not (nothing to
    # restrict to -- the full mesh already has material on both sides).
    for exploited, name, fixed_directions, cut_box, mirrored, \
            pos_box, pos_normal, pos_d, neg_box, neg_normal, neg_d in (
        (SYMMETRY_X, 'symX', [1, 0, 0],
         [-sym_eps, -WIDTH, BASE_OFFSET - HEIGHT, sym_eps, WIDTH, BASE_OFFSET + 2 * HEIGHT],
         MIRROR_VISUAL_X,
         [-sym_eps, -WIDTH, BASE_OFFSET - HEIGHT, LENGTH, WIDTH, BASE_OFFSET + 2 * HEIGHT], [1, 0, 0], 0.0,
         [-LENGTH, -WIDTH, BASE_OFFSET - HEIGHT, sym_eps, WIDTH, BASE_OFFSET + 2 * HEIGHT], [-1, 0, 0], 0.0),
        (SYMMETRY_Y, 'symY', [0, 1, 0],
         [-LENGTH, -sym_eps, BASE_OFFSET - HEIGHT, LENGTH, sym_eps, BASE_OFFSET + 2 * HEIGHT],
         MIRROR_VISUAL_Y,
         [-LENGTH, -sym_eps, BASE_OFFSET - HEIGHT, LENGTH, WIDTH, BASE_OFFSET + 2 * HEIGHT], [0, 1, 0], 0.0,
         [-LENGTH, -WIDTH, BASE_OFFSET - HEIGHT, LENGTH, sym_eps, BASE_OFFSET + 2 * HEIGHT], [0, -1, 0], 0.0),
        (SYMMETRY_Z, 'symZ', [0, 0, 1],
         [-LENGTH, -WIDTH, sym_mid_z - sym_eps, LENGTH, WIDTH, sym_mid_z + sym_eps],
         MIRROR_VISUAL_Z,
         [-LENGTH, -WIDTH, BASE_OFFSET - HEIGHT, LENGTH, WIDTH, sym_mid_z + sym_eps], [0, 0, -1], -sym_mid_z,
         [-LENGTH, -WIDTH, sym_mid_z - sym_eps, LENGTH, WIDTH, BASE_OFFSET + 2 * HEIGHT], [0, 0, 1], sym_mid_z),
    ):
        if exploited:
            # Skip the debug box outline once this axis's own
            # mirror_visual_* is filling in the rest of the shape --
            # redundant clutter once the render already shows the
            # complete model there; still useful as a visual cue for
            # where the cut is when that plane is left un-mirrored.
            cube.addObject('BoxROI', name=name + 'ROI', box=cut_box,
                            drawBoxes=not mirrored, position='@dofs.rest_position')
            cube.addObject('PartialFixedProjectiveConstraint', name=name,
                            indices='@' + name + 'ROI.indices', fixedDirections=fixed_directions)
            if PREVENT_SYMMETRY_CROSSING:
                _add_plane_guard(name + 'Kept', pos_box, pos_normal, pos_d)
        elif PREVENT_SYMMETRY_CROSSING:
            _add_plane_guard(name + 'Pos', pos_box, pos_normal, pos_d)
            _add_plane_guard(name + 'Neg', neg_box, neg_normal, neg_d)

    cube.addObject('LinearSolverConstraintCorrection')

    # ---- cavity (air chamber) -------------------------------------------
    cavity = cube.addChild('cavity')
    cavity.addObject('MeshTopology', name='topo', position=cavity_pts, triangles=cavity_tris)
    cavity.addObject('MechanicalObject', name='cavityDofs', src='@topo')
    pressure = cavity.addObject('SurfacePressureConstraint', name='pressure',
                                 triangles='@topo.triangles',
                                 valueType='pressure', value=0)
    cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # ---- visualization -----------------------------------------------
    # Procedurally-generated outer skin (not derived from the tetrahedra
    # via Tetra2Triangle) so CUTAWAY_FACES can simply omit a face for a
    # look-inside view -- extracting an arbitrary boundary subset from
    # the tet mesh directly would need per-tet orientation handling this
    # sidesteps entirely. Mapped onto the shell the same way the cavity
    # surface is (BarycentricMapping; vertices coincide with the shell's
    # actual outer-boundary grid nodes, so the mapping is exact).
    visu = cube.addChild('visu')
    # No wireframe override here: `showWireframe` switches OpenGL to
    # line-only rendering (GL_LINE) for the whole subtree, replacing the
    # filled surface rather than overlaying it -- it looked like a
    # translucent solid at low alpha only because many overlapping
    # semi-transparent lines blend together, but at alpha=1 that breaks
    # down into a sparse, mostly-see-through line grid. Plain filled
    # rendering (inherited from the root VisualStyle's hideWireframe)
    # respects wall_alpha correctly across its whole 0-1 range.
    visu.addObject('MeshTopology', name='topo', position=outer_pts, triangles=outer_tris)
    visu_ogl = visu.addObject('OglModel', name='visual', src='@topo', color=[0.15, 0.45, 0.95, WALL_ALPHA])
    visu.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # Interior-volume visual surface: same geometry as the (physics-only,
    # never rendered) cavity surface above, but as its own mesh so it can
    # have its own cutaway window without touching the closed surface the
    # pressure/volume computation needs. Same BarycentricMapping pattern
    # as the outer skin, just onto the shell's inner-boundary nodes.
    interior_visu = cube.addChild('interior_visu')
    interior_visu.addObject('MeshTopology', name='topo', position=inner_pts, triangles=inner_tris)
    interior_visu_ogl = interior_visu.addObject('OglModel', name='visual', src='@topo',
                                                 color=[0.95, 0.6, 0.1, INTERIOR_ALPHA])
    interior_visu.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # mirror_visual_x/y/z: complete the visual model by live-mirroring
    # the outer-skin/interior-volume surfaces across each enabled
    # symmetry plane -- and every combination of them, for more than one
    # enabled axis (e.g. X+Y needs an X-only, a Y-only, AND an
    # XY-diagonal copy to fill in all 4 quadrants). Each axis only
    # participates if BOTH its symmetry_* flag AND its own
    # mirror_visual_* are on, so a plane can be left un-mirrored (open)
    # even while the others are fully reconstructed. Purely visual --
    # there's no equivalent physics mesh on the mirrored side, so this
    # can't use BarycentricMapping (same reason the cavity's own
    # symmetry cap needed _capped_box_surface instead); MirrorController
    # just copies+reflects the real surface's already-computed position
    # each step.
    sym_axes = []
    if SYMMETRY_X and MIRROR_VISUAL_X:
        sym_axes.append((0, 0.0))
    if SYMMETRY_Y and MIRROR_VISUAL_Y:
        sym_axes.append((1, 0.0))
    if SYMMETRY_Z and MIRROR_VISUAL_Z:
        sym_axes.append((2, sym_mid_z))

    if sym_axes:
        for r in range(1, len(sym_axes) + 1):
            for combo in itertools.combinations(sym_axes, r):
                axes = tuple(a for a, _ in combo)
                values = tuple(v for _, v in combo)
                suffix = ''.join('XYZ'[a] for a in axes)

                # A plain write into the OglModel's own position doesn't
                # reliably reach the renderer (it isn't driven by a real
                # SOFA Mapping, so nothing signals the draw pipeline that
                # it changed) -- same pattern as every other visual piece
                # in this scene instead: a MechanicalObject holds the
                # actual (controller-driven) position, and IdentityMapping
                # propagates it to the OglModel each step.
                m_outer_pts, m_outer_tris = _mirror_surface(outer_pts, outer_tris, axes, values)
                mirror_outer = rootNode.addChild('visu_mirror_' + suffix)
                mirror_outer.addObject('MeshTopology', name='topo', position=m_outer_pts, triangles=m_outer_tris)
                mirror_outer_mstate = mirror_outer.addObject('MechanicalObject', name='mstate', src='@topo')
                mirror_outer.addObject('OglModel', name='visual', src='@topo',
                                        color=[0.15, 0.45, 0.95, WALL_ALPHA])
                mirror_outer.addObject('IdentityMapping', input='@mstate', output='@visual')
                mirror_outer.addObject(MirrorController(visu_ogl, mirror_outer_mstate, axes, values,
                                                          name='mirrorCtrl'))

                m_inner_pts, m_inner_tris = _mirror_surface(inner_pts, inner_tris, axes, values)
                mirror_inner = rootNode.addChild('interior_visu_mirror_' + suffix)
                mirror_inner.addObject('MeshTopology', name='topo', position=m_inner_pts, triangles=m_inner_tris)
                mirror_inner_mstate = mirror_inner.addObject('MechanicalObject', name='mstate', src='@topo')
                mirror_inner.addObject('OglModel', name='visual', src='@topo',
                                        color=[0.95, 0.6, 0.1, INTERIOR_ALPHA])
                mirror_inner.addObject('IdentityMapping', input='@mstate', output='@visual')
                mirror_inner.addObject(MirrorController(interior_visu_ogl, mirror_inner_mstate, axes, values,
                                                          name='mirrorCtrl'))

    rootNode.addObject(PressureRamp(rootNode, pressure, TARGET_PRESSURE, RAMP_TIME,
                                     instant=INSTANT, name='PressureRamp'))

    return rootNode


# ----------------------------------------------------------------------
# Headless standalone run (python HollowCubeInflation.py)
# ----------------------------------------------------------------------
def main():
    import Sofa.Simulation

    # Headless entry point -- the SOFA GUI (run_imgui.bat/run_glfw.bat)
    # never calls main() at all, it loads this file and calls
    # createScene() directly, so this only ever suppresses the editor for
    # a plain `python HollowCubeInflation.py` run (setdefault so an
    # operator who explicitly wants it anyway can still unset/override
    # this first).
    os.environ.setdefault('HOLLOWCUBE_NO_EDITOR', '1')

    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)

    duration = RAMP_TIME + 2.0
    steps = int(duration / DT)
    for _ in range(steps):
        Sofa.Simulation.animate(root, DT)

    Sofa.Simulation.unload(root)


if __name__ == '__main__':
    main()
