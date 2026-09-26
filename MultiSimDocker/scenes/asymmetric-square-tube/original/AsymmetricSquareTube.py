# -*- coding: utf-8 -*-
"""
AsymmetricSquareTube: a real FEM simulation (same philosophy as
../SqueezeSquare2D's own) of a silicone-rubber square cross-section with
a rectangular gap punched through it off-center -- a square tube with an
asymmetric wall thickness on each side -- rendered as a genuine 3D object
and driven through a params.json-backed parameter editor. World
convention matches every sibling project: X horizontal, Z up (gravity =
-Z), Y depth.

This project generalizes ../SqueezeSquare2D in four ways:

1. Parametrized outer and cavity geometry. outer_width/outer_height (so
   the tube's own cross-section need not be square) and divs_x/divs_z
   (independent cell counts per axis, so cells need not be square either)
   set the outer grid; cavity_width/cavity_height (independent, so the
   hole need not be square) and cavity_center_x/cavity_center_z (so it
   need not be centered) set the hole -- all params.json fields, see
   build_block_mesh. Unlike ../SqueezeSquare2D's own compile-time assert,
   these come from a live-editable file, so a cavity that doesn't land
   exactly on a grid line is snapped to the nearest one and clamped to
   leave at least one solid cell of wall on every side, with a console
   note if that changed what was asked for, rather than crashing the
   scene over an off-by-a-fraction-of-a-millimeter entry.

2. The top row's own "moving platen" constraint (see ../SqueezeSquare2D's
   own docstring for why this is a direct per-step projection controller
   rather than any of PneuNetFinger's plane-tracking controllers, or a
   RigidMapping-coupled rigid body) now has ONE MORE degree of freedom:
   instead of forcing every top node to the SAME Z (a rigid plate that
   can only translate), TopPlateController fits the best (least-squares)
   straight LINE z = a*x + b through the top row's own post-solve
   positions each step and projects every node onto that line -- exactly
   the same idea as before, generalized from a degree-0 (constant) fit to
   a degree-1 (affine) one, so the plate can also tilt. This isn't just
   added generality for its own sake: an off-center cavity leaves genuinely
   different wall thickness on each side, so the two sides of the top row
   are no longer equally stiff -- forcing it to stay perfectly flat (as
   ../SqueezeSquare2D's own symmetric cavity made harmless) would be
   physically wrong here, fighting the real asymmetry instead of letting
   the platen tilt the way an unequally-supported rigid plate actually
   would.

3. Proper 3D rendering. The underlying mesh is still, physically, a
   single element-thick plane-strain slab exactly like ../SqueezeSquare2D
   (a real 2D cross-section, only ever needing one layer of elements in
   Y -- see its own docstring for the technique) -- but that's a
   modelling choice, not a reason to render it as a flat picture. `depth`
   is now a generous, independently-set visual/physical Y-extent (not
   tied to the cross-section's own cell size), the floor and platen are
   real 3D boxes (not flat 2D bars), the block's own visual skin covers
   all 6 faces of the (now hollow, tube-shaped) solid -- front, back, the
   outer perimeter, and the cavity's own inner perimeter -- and the
   camera sits at a genuine 3/4 perspective angle instead of dead-on
   orthographic. The block's own fill color, transparency and edge-line
   color are all editable (block_fill_color, block_alpha, block_show_edges,
   block_edge_color) -- the edge overlay is its own separate OglModel
   drawing the skin's own edges, not SOFA's global wireframe flag, which
   would replace the filled surface rather than draw lines on top of it
   (see createScene()'s "block edge overlay" section for why). show_platen
   toggles the grey pressing-plate visual (its own MechanicalObject still
   gets driven every step regardless -- only its OglModel is skipped). The
   light sources (ambient_color, key_light_direction/color, fill_light_
   direction/color) are editable too -- see createScene()'s own lighting
   comments; these colors are RGB only (no alpha, unlike block_fill_color's
   own alpha -- see ambient_color's own _DEFAULT_PARAMS comment for why).
   key_light_rotate_enabled/key_light_rotate_period sweep the key light's
   own direction around the world Z axis (LightRotationController) as a
   build/inspection aid: a moving light passes over every outward-facing
   direction once per period, so a face with a flipped normal stays dark
   even as the light sweeps right over where it should light up -- a much
   more direct visual check of the skin's own winding than a static light
   gives (added after an earlier revision's outer-wall/cavity-wall/floor/
   platen normals were found reversed and fixed -- see _side_skin's own
   docstring).
   Every multi-component property here (a color or a direction) is a
   plain JSON array, edited in params_editor.py as one comma-separated
   field rather than one field per component.

4. A params.json-backed parameter editor (params_editor.py), same
   standalone-Tkinter-app pattern as ../DoublePendulum's own: every
   geometry/material/load/flag constant below is loaded from params.json
   (created with defaults on first run), auto-launched non-blocking from
   createScene() (guarded against duplicate windows via a PID file, same
   as ../DoublePendulum), and skippable via the ASYMMETRICSQUARETUBE_NO_EDITOR
   environment variable for headless/automated runs.

Material: silicone rubber, same default values as ../SqueezeSquare2D's
own (Young's modulus 1 MPa, Poisson's ratio 0.45, density 1100 kg/m^3),
now editable.

Boundary conditions (same as ../SqueezeSquare2D unless noted):
  - Bottom row (Z=0): vertical position pinned at 0, free to slide in X.
  - Top row: constrained to the best-fit tilting line described in (2)
    above, each node still free to slide in X independently. A uniform
    top_pressure acts on it, ramped in from top_ramp_start over
    top_ramp_time.
  - The cavity's own 4 walls carry a uniform cavity_pressure, ramped in
    from cavity_ramp_start over cavity_ramp_time (defaults: held at 0
    until t=1s so the block first settles under gravity and the top
    load alone, then ramped to 10 bar over the following 5s) -- same
    per-node tributary-area recipe as ../SqueezeSquare2D's own, with
    corner nodes receiving both adjacent walls' contributions.

Run headless (prints per-step platen height/tilt/cavity size to the console):
    run_headless.bat
Run with the GUI:
    run_imgui.bat   (docked Scene Graph/Viewport/Log panels)
    run_glfw.bat    (plain single viewport, respects BackgroundSetting)
Edit parameters (also auto-launched by createScene() itself):
    edit_params.bat
"""
import json
import math
import os
import shutil
import subprocess
import Sofa
import Sofa.Core


def _look_at_quaternion(eye, target, world_up=(0.0, 0.0, 1.0)):
    """Camera orientation quaternion [x, y, z, w] for a camera at `eye`
    looking towards `target`, with `world_up` as the world's vertical
    axis. Assumes the standard OpenGL camera convention (local -Z is the
    view direction, local +Y is up, local +X is right) that SOFA's
    BaseCamera also follows. (Same helper as the sibling projects' own
    -- copied rather than imported, so this project has no dependency
    on those.)
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
# Parameters -- loaded from params.json next to this script, editable via
# params_editor.py. The file is created with these defaults on first run
# if missing. Everything here only takes effect on the next scene
# (re)build -- after editing, use the SOFA GUI's Reload button, or just
# re-run for a headless session.
# ----------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')

_DEFAULT_PARAMS = {
    'outer_width': 0.1,             # [m] outer cross-section extent along X
    'outer_height': 0.1,            # [m] outer cross-section extent along Z -- independent
                                     # of outer_width, so the tube need not be square
    'divs_x': 10,                   # cells along X
    'divs_z': 10,                   # cells along Z -- independent of divs_x, so cells need
                                     # not be square either (outer_width/divs_x can differ
                                     # from outer_height/divs_z)
    'depth': 0.03,                  # [m] the single element layer's own Y-extent -- a
                                     # free visual/physical parameter (plane strain makes
                                     # the physics scale exactly with it, see the module
                                     # docstring), picked bigger than the cell size purely
                                     # so the tube reads clearly as 3D from an angle.
    'young_modulus': 1.0e6,         # [Pa] silicone rubber
    'poisson_ratio': 0.45,
    'density': 1100.0,              # [kg/m^3]
    'cavity_width': 0.02,           # [m]
    'cavity_height': 0.02,          # [m]
    # Off-center -> asymmetric wall thickness (with the defaults above:
    # 50/30 mm left/right, 30/50 mm bottom/top). Chosen to land exactly on
    # grid lines (multiples of outer_width/divs_x = outer_height/divs_z =
    # 0.01 m here) rather than a half-cell tie, so the default geometry
    # needs no snapping at all -- see the CAVITY_I_LO etc. derivation
    # below for what happens when a (typically hand-edited) value doesn't.
    'cavity_center_x': 0.06,        # [m]
    'cavity_center_z': 0.04,        # [m]
    'top_pressure': 1.0e5,          # [Pa] 1 bar, downward, on the top row
    'top_ramp_start': 0.0,          # [s]
    'top_ramp_time': 0.5,           # [s]
    'cavity_pressure': 1.0e6,       # [Pa] 10 bar, outward, on the cavity's own walls
    'cavity_ramp_start': 1.0,       # [s] held at 0 until this, so the block first settles
    'cavity_ramp_time': 5.0,        # [s] 0 -> cavity_pressure, linear, from ramp_start
    'gravity': 9.81,                # [m/s^2]
    'gravity_enabled': True,
    'dt': 0.01,                     # [s]
    'run_duration': 9.0,            # headless run length, main() only [s]
    'rayleigh_stiffness': 0.1,
    'rayleigh_mass': 0.1,
    'console_log_enabled': True,    # print per-step platen/cavity diagnostics
    # ---- block visual properties (the FEM rubber block itself -- not
    # the floor/platen, which stay fixed) -- see createScene()'s "visual
    # skin" section for how block_show_edges is actually rendered (a
    # separate line overlay, since SOFA's own global wireframe toggle
    # REPLACES the filled surface rather than drawing lines on top of it).
    # Multi-component properties (colors, directions) are plain JSON
    # arrays here, edited in params_editor.py as one comma-separated
    # field ("0.85, 0.55, 0.65") rather than one field per component.
    'block_fill_color': [0.85, 0.55, 0.65],   # [r, g, b]
    'block_alpha': 0.95,            # 0 = fully transparent, 1 = fully opaque
    'block_show_edges': True,
    'block_edge_color': [0.25, 0.12, 0.18],   # [r, g, b]
    'show_platen': True,   # the grey rigid plate visual on top -- its own
                            # MechanicalObject still gets driven every step
                            # regardless (TopPlateController's tilt-fit
                            # enforcement doesn't depend on it being drawn),
                            # this only skips adding its OglModel/IdentityMapping.
    # ---- light sources -- see createScene()'s own lighting comments for
    # why each one's direction sign pattern lights the faces it does.
    # RGB only, same as block_fill_color/block_edge_color above -- unlike
    # a material, a light's own alpha has no visual meaning in this
    # renderer (confirmed directly: SOFA's LightManager.ambient and
    # DirectionalLight.color are still a strongly-typed Vec4/RGBA
    # underneath, so it's fixed at 1.0 when actually building each
    # object below rather than exposed here as an editable value that
    # would do nothing).
    'ambient_color': [0.65, 0.65, 0.65],       # [r, g, b] -- LightManager
    'key_light_direction': [0.3, -0.6, 0.7],   # [x, y, z] -- lights +X, -Y, +Z
    'key_light_color': [1.0, 1.0, 1.0],        # [r, g, b]
    'fill_light_direction': [-0.3, 0.6, 0.0],  # [x, y, z] -- lights -X, +Y
    'fill_light_color': [0.55, 0.55, 0.55],    # [r, g, b] -- dimmer than the key light
    # Sweeps the key light's direction around the world Z axis, a full
    # revolution every key_light_rotate_period seconds -- a moving light
    # sweeps across every outward-facing direction over one period, so a
    # face with a flipped normal stays dark even as the light passes
    # right over where it should light up, which is a much more direct
    # visual check of the skin's own winding than a static light gives
    # (this is literally how the front/back-vs-sides normal bug that
    # motivated this option got first noticed). Off by default since it's
    # a build/inspection aid, not part of the normal look of the scene.
    'key_light_rotate_enabled': False,
    'key_light_rotate_period': 8.0,   # [s] time for one full 360 degree sweep
    'show_fem_elements': False,  # VisualStyle's showForceFields flag -- draws the
                                  # FEM mesh's own tetrahedra (wireframe-ish, hard-
                                  # opaque regardless of block_alpha), same flag
                                  # ../PneuNetFinger's own params.json exposes. Set
                                  # here rather than toggled by hand in the SOFA
                                  # GUI's display-flags menu because Reload resets
                                  # those flags back to whatever this scene sets.
    'theme': 'System',   # 'Light' / 'Dark' / 'System' -- UI-only preference,
                          # stored here for convenience but never read by
                          # AsymmetricSquareTube.py itself.
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
    """Opens params_editor.py in its own process -- see
    ../DoublePendulum/DoublePendulum.py's own launch_params_editor for
    the full reasoning (identical here): a plain `python`, not runSofa's
    embedded interpreter; guarded against duplicate windows via a PID
    file on disk (survives the SOFA GUI's Reload re-importing this
    module); skippable via ASYMMETRICSQUARETUBE_NO_EDITOR for headless/
    automated runs.
    """
    if os.environ.get('ASYMMETRICSQUARETUBE_NO_EDITOR'):
        return

    if os.path.exists(_EDITOR_PID_FILE):
        try:
            with open(_EDITOR_PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
        except (OSError, ValueError):
            old_pid = None
        if old_pid is not None and _pid_is_running(old_pid):
            return  # editor from a previous (re)load is still open

    python_exe = shutil.which('python')
    if python_exe is None:
        print("[params] no 'python' found on PATH -- skipping the parameter editor.")
        return

    editor_path = os.path.join(_SCRIPT_DIR, 'params_editor.py')
    try:
        process = subprocess.Popen(
            [python_exe, editor_path],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True)
        with open(_EDITOR_PID_FILE, 'w') as f:
            f.write(str(process.pid))
    except OSError as e:
        print(f"[params] could not launch params_editor.py: {e}")


# ----------------------------------------------------------------------
# Derived constants
# ----------------------------------------------------------------------
WIDTH = PARAMS['outer_width']
HEIGHT = PARAMS['outer_height']
DIVS_X = PARAMS['divs_x']
DIVS_Z = PARAMS['divs_z']
CELL_X = WIDTH / DIVS_X
CELL_Z = HEIGHT / DIVS_Z
DEPTH = PARAMS['depth']
NX = DIVS_X + 1
NZ = DIVS_Z + 1
NY = 2

BASE_Z = 0.0

YOUNG_MODULUS = PARAMS['young_modulus']
POISSON_RATIO = PARAMS['poisson_ratio']
DENSITY = PARAMS['density']

PRESSURE = PARAMS['top_pressure']
TOP_RAMP_START = PARAMS['top_ramp_start']
TOP_RAMP_TIME = PARAMS['top_ramp_time']
CAVITY_PRESSURE = PARAMS['cavity_pressure']
CAVITY_RAMP_START = PARAMS['cavity_ramp_start']
CAVITY_RAMP_TIME = PARAMS['cavity_ramp_time']

GRAVITY_MAGNITUDE = PARAMS['gravity'] if PARAMS['gravity_enabled'] else 0.0
GRAVITY = [0.0, 0.0, -GRAVITY_MAGNITUDE]
DT = PARAMS['dt']
RUN_DURATION = PARAMS['run_duration']

RAYLEIGH_STIFFNESS = PARAMS['rayleigh_stiffness']
RAYLEIGH_MASS = PARAMS['rayleigh_mass']
CONSOLE_LOG_ENABLED = PARAMS['console_log_enabled']

# Cavity extent in node-index space. Snapped to the nearest grid line and
# clamped to leave at least one solid cell of wall on every side (see the
# module docstring) rather than asserting exact alignment -- these come
# from a live-editable file, not compile-time constants. X uses CELL_X/
# DIVS_X, Z uses CELL_Z/DIVS_Z -- independent since outer_width/divs_x
# need not match outer_height/divs_z.
_raw_i_lo = (PARAMS['cavity_center_x'] - PARAMS['cavity_width'] / 2.0) / CELL_X
_raw_i_hi = (PARAMS['cavity_center_x'] + PARAMS['cavity_width'] / 2.0) / CELL_X
_raw_k_lo = (PARAMS['cavity_center_z'] - PARAMS['cavity_height'] / 2.0) / CELL_Z
_raw_k_hi = (PARAMS['cavity_center_z'] + PARAMS['cavity_height'] / 2.0) / CELL_Z

CAVITY_I_LO = min(max(round(_raw_i_lo), 1), DIVS_X - 2)
CAVITY_I_HI = min(max(round(_raw_i_hi), CAVITY_I_LO + 1), DIVS_X - 1)
CAVITY_K_LO = min(max(round(_raw_k_lo), 1), DIVS_Z - 2)
CAVITY_K_HI = min(max(round(_raw_k_hi), CAVITY_K_LO + 1), DIVS_Z - 1)

if (abs(_raw_i_lo - CAVITY_I_LO) > 0.01 or abs(_raw_i_hi - CAVITY_I_HI) > 0.01 or
        abs(_raw_k_lo - CAVITY_K_LO) > 0.01 or abs(_raw_k_hi - CAVITY_K_HI) > 0.01):
    print(f"[params] cavity geometry snapped/clamped to the grid: "
          f"x in [{CAVITY_I_LO * CELL_X:.4f}, {CAVITY_I_HI * CELL_X:.4f}], "
          f"z in [{CAVITY_K_LO * CELL_Z:.4f}, {CAVITY_K_HI * CELL_Z:.4f}] m "
          f"(requested x in [{PARAMS['cavity_center_x'] - PARAMS['cavity_width'] / 2.0:.4f}, "
          f"{PARAMS['cavity_center_x'] + PARAMS['cavity_width'] / 2.0:.4f}], "
          f"z in [{PARAMS['cavity_center_z'] - PARAMS['cavity_height'] / 2.0:.4f}, "
          f"{PARAMS['cavity_center_z'] + PARAMS['cavity_height'] / 2.0:.4f}] m)")

# ---- visuals ---------------------------------------------------------
SKY_COLOR = [0.78, 0.87, 0.96, 1.0]
RUBBER_COLOR = list(PARAMS['block_fill_color']) + [PARAMS['block_alpha']]
BLOCK_SHOW_EDGES = PARAMS['block_show_edges']
BLOCK_EDGE_COLOR = list(PARAMS['block_edge_color']) + [1.0]
SHOW_PLATEN = PARAMS['show_platen']
SHOW_FEM_ELEMENTS = PARAMS['show_fem_elements']

# LightManager.ambient and DirectionalLight.color are a strongly-typed
# Vec4/RGBA underneath (verified directly -- passing only 3 components
# triggers a SOFA warning and falls back to a default), so alpha=1.0 is
# appended here rather than stored/edited as a param -- see
# ambient_color's own _DEFAULT_PARAMS comment for why.
AMBIENT_COLOR = list(PARAMS['ambient_color']) + [1.0]
KEY_LIGHT_DIRECTION = PARAMS['key_light_direction']
KEY_LIGHT_COLOR = list(PARAMS['key_light_color']) + [1.0]
FILL_LIGHT_DIRECTION = PARAMS['fill_light_direction']
FILL_LIGHT_COLOR = list(PARAMS['fill_light_color']) + [1.0]
KEY_LIGHT_ROTATE_ENABLED = PARAMS['key_light_rotate_enabled']
KEY_LIGHT_ROTATE_PERIOD = PARAMS['key_light_rotate_period']

FLOOR_MARGIN = 0.03            # [m] how far the floor slab extends past the block, each side
FLOOR_THICKNESS = 0.006
FLOOR_COLOR = [0.1, 0.55, 0.15, 1.0]
# The block's own bottom row is pinned exactly at BASE_Z (see the
# PartialFixedProjectiveConstraint below), so a floor top surface placed
# at that same BASE_Z would be perfectly coplanar with it -- fine when
# the block is opaque, but a visible Z-fighting flicker once block_alpha
# makes it transparent enough to actually see the floor through it.
# Dropping the floor's own top surface a hair below BASE_Z removes the
# exact coincidence; small enough (well under the block's own 10 mm
# cells) to read as sitting flush, not as a visible gap.
FLOOR_Z_GAP = 0.0003            # [m]

PLATEN_OVERHANG = 0.02         # [m] each side, so it visibly reads as a separate plate
PLATEN_HALF_THICKNESS = 0.003
PLATEN_COLOR = [0.35, 0.35, 0.4, 1.0]

# A genuine 3/4 perspective view (not dead-on orthographic, unlike
# ../SqueezeSquare2D -- see the module docstring's item 3), aimed at the
# block's own center, offset by CAMERA_DISTANCE at CAMERA_AZIMUTH_DEG
# (rotation about the vertical Z axis, 0 = looking along +Y) and
# CAMERA_ELEVATION_DEG (tilt up from the horizontal).
CAMERA_LOOKAT = [WIDTH / 2.0, DEPTH / 2.0, HEIGHT / 2.0]
CAMERA_DISTANCE = max(WIDTH, HEIGHT, DEPTH) * 4.0
CAMERA_AZIMUTH_DEG = -35.0
CAMERA_ELEVATION_DEG = 24.0
CAMERA_FOV = 35.0


def _camera_eye():
    az = math.radians(CAMERA_AZIMUTH_DEG)
    el = math.radians(CAMERA_ELEVATION_DEG)
    return [
        CAMERA_LOOKAT[0] + CAMERA_DISTANCE * math.cos(el) * math.sin(az),
        CAMERA_LOOKAT[1] - CAMERA_DISTANCE * math.cos(el) * math.cos(az),
        CAMERA_LOOKAT[2] + CAMERA_DISTANCE * math.sin(el),
    ]


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def _thin_box(x0, z0, x1, z1, half_thickness, y0, y1):
    """A solid box following the segment (x0,z0)-(x1,z1) (offset
    perpendicular to it by +-half_thickness, same technique as the
    sibling projects' own 2D _thin_bar), extruded from y0 to y1 -- used
    for the platen, which needs to follow its own current tilt.
    """
    dx, dz = x1 - x0, z1 - z0
    length = math.hypot(dx, dz)
    ux, uz = dx / length, dz / length
    nx, nz = -uz, ux
    quad = [
        (x0 - half_thickness * nx, z0 - half_thickness * nz),
        (x1 - half_thickness * nx, z1 - half_thickness * nz),
        (x1 + half_thickness * nx, z1 + half_thickness * nz),
        (x0 + half_thickness * nx, z0 + half_thickness * nz),
    ]
    front = [[px, y0, pz] for px, pz in quad]
    back = [[px, y1, pz] for px, pz in quad]
    points = front + back
    # The two end caps (y=y0, y=y1) are correctly wound as written; the 4
    # side walls the loop below builds were not (verified with a per-
    # triangle normal-direction check against the analytically-known
    # expected direction: all 8 side triangles came out reversed, both
    # caps were already correct) -- [a, b, b+4]/[a, b+4, a+4] fixed to
    # [a, b+4, b]/[a, a+4, b+4].
    tris = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]
    for i in range(4):
        a, b = i, (i + 1) % 4
        tris.append([a, b + 4, b])
        tris.append([a, a + 4, b + 4])
    return points, tris


def _aabb_box(xmin, ymin, zmin, xmax, ymax, zmax):
    """A plain axis-aligned solid box -- used for the (always flat)
    floor slab. Every face's winding below was verified against its own
    analytically-known outward direction (all 12 triangles came out
    reversed before this fix -- e.g. [0, 1, 2] at z=zmin pointed +Z
    instead of -Z).
    """
    p = [
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ]
    tris = [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 5, 4], [0, 1, 5], [3, 6, 2], [3, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 6, 5], [1, 2, 6],
    ]
    return p, tris


def _perimeter_loop(i_lo, i_hi, k_lo, k_hi):
    """Ordered (i, k) grid-index pairs walking once around the boundary
    of the axis-aligned rectangle [i_lo, i_hi] x [k_lo, k_hi] at unit
    node spacing (closed -- the first point isn't repeated at the end).
    Used for both the block's own outer perimeter and the cavity's own
    inner one, to build each one's side-wall skin.
    """
    pts = []
    for i in range(i_lo, i_hi):
        pts.append((i, k_lo))
    for k in range(k_lo, k_hi):
        pts.append((i_hi, k))
    for i in range(i_hi, i_lo, -1):
        pts.append((i, k_hi))
    for k in range(k_hi, k_lo, -1):
        pts.append((i_lo, k))
    return pts


def _side_skin(loop, outward):
    """Quad-strip skin connecting the y=0 and y=DEPTH copies of a closed
    (i, k) loop (from _perimeter_loop) at rest -- `outward` picks the
    winding so the strip's own normals point away from the material
    (True for the block's outer perimeter, False for the cavity's own
    inner one, whose "outward" is into the hole).

    _perimeter_loop always walks its rectangle the same rotational sense
    regardless of which one it's tracing, but "away from material" is the
    OPPOSITE geometric direction for the two callers -- the outer loop
    encloses material (its interior IS the block), the cavity loop
    encloses empty space (its interior is the hole, material surrounds
    it) -- so the two cases genuinely need opposite winding, not the same
    one. (Caught by a normal-direction check against the analytically-
    known expected direction on every triangle of both surfaces: every
    single one, on both, came out reversed before this fix.)
    """
    n = len(loop)
    points = ([[i * CELL_X, 0.0, k * CELL_Z] for i, k in loop] +
              [[i * CELL_X, DEPTH, k * CELL_Z] for i, k in loop])
    tris = []
    for idx in range(n):
        a, b = idx, (idx + 1) % n
        a2, b2 = a + n, b + n
        if outward:
            tris.append([a, b2, b])
            tris.append([a, a2, b2])
        else:
            tris.append([a, b, b2])
            tris.append([a, b2, a2])
    return points, tris


def _face_skin(y):
    """Flat quad grid over the DIVS_X x DIVS_Z cell footprint at world
    Y=y, skipping cavity cells -- the block's own front (y=0) or back
    (y=DEPTH) face, with the hole showing through.
    """
    pts = [[i * CELL_X, y, k * CELL_Z] for k in range(NZ) for i in range(NX)]
    tris = []
    for k in range(DIVS_Z):
        for i in range(DIVS_X):
            if _is_cavity_cell(i, k):
                continue
            a = k * NX + i
            b = k * NX + (i + 1)
            c = (k + 1) * NX + (i + 1)
            d = (k + 1) * NX + i
            tris.append([a, b, c] if y == 0.0 else [a, c, b])
            tris.append([a, c, d] if y == 0.0 else [a, d, c])
    return pts, tris


def _merge(parts):
    points, tris = [], []
    for pts, ts in parts:
        offset = len(points)
        points += pts
        tris += [[i + offset for i in t] for t in ts]
    return points, tris


def _flat_shaded(points, tris):
    """Duplicates each triangle's own 3 vertices so none are shared with
    a neighboring triangle -- same technique and reasoning as
    ../PneuNetFinger/PneuNetFinger.py's own _flat_shaded (copied rather
    than imported, so this project has no dependency on that one).
    SOFA's OglModel always computes smooth, per-vertex-averaged normals
    with no flat-shading override (checked directly in PneuNetFinger's
    own build: neither OglModel nor its VisualModelImpl base declares
    any such Data field in this SOFA version) -- but with no shared
    vertices left to average across, each vertex's "averaged" normal
    degenerates to exactly its one triangle's own flat face normal,
    giving the same visual result. Applied to every rendered solid here
    (the block's own skin, the floor, the platen) so their edges/corners
    read as crisp facets rather than smoothed over -- geometrically a
    no-op (every duplicated vertex sits at the exact same position as
    before), so it doesn't disturb the block skin's own BarycentricMapping
    or the platen's per-step rebuild, and the block's separate edge
    overlay (drawn from the UNDUPLICATED vertices) still lines up with it
    exactly.
    """
    flat_points = []
    flat_tris = []
    for tri in tris:
        base = len(flat_points)
        for i in tri:
            flat_points.append(points[i])
        flat_tris.append([base, base + 1, base + 2])
    return flat_points, flat_tris


# ----------------------------------------------------------------------
# Mesh construction: a structured NX x NZ x NY grid of nodes, each cube
# cell split into 6 tetrahedra via the standard Kuhn (body-diagonal)
# decomposition -- same technique as ../SqueezeSquare2D's own (see its
# docstring for the full reasoning), generalized here to an
# off-center, independently-sized cavity.
# ----------------------------------------------------------------------
_KUHN_LOCAL_OFFSETS = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]  # (di, dk, dj) -- i.e. (X, Z, Y) offsets from a cell's own (i, k, j) corner
_KUHN_TETS = [
    (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
]


def _node_id(i, k, j):
    return j * (NX * NZ) + k * NX + i


def _is_cavity_cell(i, k):
    return CAVITY_I_LO <= i < CAVITY_I_HI and CAVITY_K_LO <= k < CAVITY_K_HI


def build_block_mesh():
    """Builds the block's tetrahedral mesh with the (possibly
    off-center, possibly non-square) cavity punched through its center,
    compacting/remapping every index so the returned mesh has no unused
    DOFs -- same approach as ../SqueezeSquare2D's own build_block_mesh.
    """
    raw_positions = [None] * (NX * NZ * NY)
    for j in range(NY):
        for k in range(NZ):
            for i in range(NX):
                raw_positions[_node_id(i, k, j)] = [i * CELL_X, j * DEPTH, k * CELL_Z]

    raw_tetrahedra = []
    used = set()
    for k in range(DIVS_Z):
        for i in range(DIVS_X):
            if _is_cavity_cell(i, k):
                continue
            corners = [_node_id(i + di, k + dk, dj) for di, dk, dj in _KUHN_LOCAL_OFFSETS]
            used.update(corners)
            for tet in _KUHN_TETS:
                raw_tetrahedra.append([corners[c] for c in tet])

    remap = {old: new for new, old in enumerate(sorted(used))}
    positions = [raw_positions[old] for old in sorted(used)]
    tetrahedra = [[remap[c] for c in tet] for tet in raw_tetrahedra]

    bottom_indices = [remap[_node_id(i, 0, j)] for j in range(NY) for i in range(NX)]
    top_indices = [remap[_node_id(i, DIVS_Z, j)] for j in range(NY) for i in range(NX)]
    cavity_indices, cavity_forces = _build_cavity_wall_forces(remap)

    k_mid = (CAVITY_K_LO + CAVITY_K_HI) // 2
    i_mid = (CAVITY_I_LO + CAVITY_I_HI) // 2
    cavity_probes = {
        'left': remap[_node_id(CAVITY_I_LO, k_mid, 0)],
        'right': remap[_node_id(CAVITY_I_HI, k_mid, 0)],
        'bottom': remap[_node_id(i_mid, CAVITY_K_LO, 0)],
        'top': remap[_node_id(i_mid, CAVITY_K_HI, 0)],
    }

    return positions, tetrahedra, bottom_indices, top_indices, cavity_indices, cavity_forces, cavity_probes


def _tributary_dx(i):
    return CELL_X if 0 < i < DIVS_X else CELL_X / 2.0


def _tributary_along(idx, lo, hi, cell):
    return cell if lo < idx < hi else cell / 2.0


def _build_cavity_wall_forces(remap):
    """Per-node outward force from CAVITY_PRESSURE on the cavity's own 4
    walls -- same tributary-area recipe as ../SqueezeSquare2D's own (see
    its docstring's "The cavity" section), unchanged by the cavity now
    being off-center/non-square since it works from CAVITY_I_LO/HI and
    CAVITY_K_LO/HI directly rather than assuming a centered square.
    """
    accum = {}

    def add(i, k, j, fx, fy, fz):
        nid = remap[_node_id(i, k, j)]
        f = accum.setdefault(nid, [0.0, 0.0, 0.0])
        f[0] += fx
        f[1] += fy
        f[2] += fz

    for k in range(CAVITY_K_LO, CAVITY_K_HI + 1):
        dz = _tributary_along(k, CAVITY_K_LO, CAVITY_K_HI, CELL_Z)
        for j in range(NY):
            area = dz * (DEPTH / 2.0)
            add(CAVITY_I_LO, k, j, -CAVITY_PRESSURE * area, 0.0, 0.0)
            add(CAVITY_I_HI, k, j, CAVITY_PRESSURE * area, 0.0, 0.0)

    for i in range(CAVITY_I_LO, CAVITY_I_HI + 1):
        dx = _tributary_along(i, CAVITY_I_LO, CAVITY_I_HI, CELL_X)
        for j in range(NY):
            area = dx * (DEPTH / 2.0)
            add(i, CAVITY_K_LO, j, 0.0, 0.0, -CAVITY_PRESSURE * area)
            add(i, CAVITY_K_HI, j, 0.0, 0.0, CAVITY_PRESSURE * area)

    indices = sorted(accum.keys())
    forces = [accum[idx] for idx in indices]
    return indices, forces


def build_top_pressure_forces(pressure):
    forces = []
    for j in range(NY):
        for i in range(NX):
            area = _tributary_dx(i) * (DEPTH / 2.0)
            forces.append([0.0, 0.0, -pressure * area])
    return forces


# ----------------------------------------------------------------------
# Controller: sweeps the key light's direction around the world Z axis --
# a build/inspection aid (key_light_rotate_enabled), see its own
# _DEFAULT_PARAMS comment for why a moving light is a more direct check
# of the skin's own winding than a static one.
# ----------------------------------------------------------------------
class LightRotationController(Sofa.Core.Controller):
    def __init__(self, light, base_direction, period, dt, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.light = light
        self.base_x, self.base_y, self.base_z = base_direction
        self.period = period
        self.dt = dt
        self.sim_time = 0.0

    def onAnimateBeginEvent(self, event):
        angle = 2.0 * math.pi * (self.sim_time / self.period)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x = self.base_x * cos_a - self.base_y * sin_a
        y = self.base_x * sin_a + self.base_y * cos_a
        self.light.direction.value = [x, y, self.base_z]
        self.sim_time += self.dt


# ----------------------------------------------------------------------
# Controller: the tiltable "moving platen" -- see module docstring item
# 2 for the full reasoning. Also owns the pressure ramps and the console
# diagnostic log, since both naturally piggyback on the same per-step
# bookkeeping.
# ----------------------------------------------------------------------
class TopPlateController(Sofa.Core.Controller):
    def __init__(self, dofs, top_indices, ramped_force_fields, dt,
                 platen_mstate, platen_half_span, cavity_probes, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.dofs = dofs
        self.top_indices = top_indices
        # list of (ConstantForceField, base_forces, ramp_start, ramp_time)
        # 4-tuples -- see ../SqueezeSquare2D's own TopPlaneController.
        self.ramped_force_fields = ramped_force_fields
        self.dt = dt
        self.platen_mstate = platen_mstate
        self.platen_half_span = platen_half_span
        self.cavity_probes = cavity_probes
        self.sim_time = 0.0

    def onAnimateBeginEvent(self, event):
        for ff, base_forces, ramp_start, ramp_time in self.ramped_force_fields:
            elapsed = self.sim_time - ramp_start
            factor = min(max(elapsed, 0.0) / ramp_time, 1.0) if ramp_time > 0 else (1.0 if elapsed >= 0 else 0.0)
            ff.forces.value = [[fx * factor, fy * factor, fz * factor] for fx, fy, fz in base_forces]

    def onAnimateEndEvent(self, event):
        self.sim_time += self.dt

        pos = [list(p) for p in self.dofs.position.value]
        vel = [list(v) for v in self.dofs.velocity.value]

        # Best-fit line z = a*x + b through the top row's own current
        # (post-solve, still-independent) positions -- ordinary least
        # squares, mass-weighted but all top-row masses are equal
        # (UniformMass over a uniform mesh) so that reduces to the plain
        # formula below. x is centered on its own mean first, the
        # standard trick that decouples the slope (a) and intercept (b)
        # normal equations so each can be solved directly rather than as
        # a coupled 2x2 system. Velocity gets the exact same fit (of vz
        # against the same x, da/dt and db/dt), so the projected motion
        # stays consistent with the projected position from one step to
        # the next.
        n = len(self.top_indices)
        xs = [pos[i][0] for i in self.top_indices]
        zs = [pos[i][2] for i in self.top_indices]
        vzs = [vel[i][2] for i in self.top_indices]
        x_bar = sum(xs) / n
        z_bar = sum(zs) / n
        vz_bar = sum(vzs) / n
        sxx = sum((x - x_bar) ** 2 for x in xs)
        a = sum((x - x_bar) * (z - z_bar) for x, z in zip(xs, zs)) / sxx
        b = z_bar - a * x_bar
        da_dt = sum((x - x_bar) * (vz - vz_bar) for x, vz in zip(xs, vzs)) / sxx
        db_dt = vz_bar - da_dt * x_bar

        max_spread = 0.0
        for idx, i in enumerate(self.top_indices):
            x = xs[idx]
            z_fit = a * x + b
            max_spread = max(max_spread, abs(zs[idx] - z_fit))
            pos[i][2] = z_fit
            vel[i][2] = da_dt * x + db_dt
        self.dofs.position.value = pos
        self.dofs.velocity.value = vel

        cx = WIDTH / 2.0
        z_left = a * (cx - self.platen_half_span) + b
        z_right = a * (cx + self.platen_half_span) + b
        # _flat_shaded here only re-duplicates the (moved) points -- the
        # topology (triangle indices) was already fixed to this same
        # shape at scene-build time and never changes, only positions do.
        platen_pts, _ = _flat_shaded(*_thin_box(
            cx - self.platen_half_span, z_left, cx + self.platen_half_span, z_right,
            PLATEN_HALF_THICKNESS, -PLATEN_OVERHANG, DEPTH + PLATEN_OVERHANG))
        self.platen_mstate.position.value = platen_pts

        cavity_width = pos[self.cavity_probes['right']][0] - pos[self.cavity_probes['left']][0]
        cavity_height = pos[self.cavity_probes['top']][2] - pos[self.cavity_probes['bottom']][2]
        tilt_deg = math.degrees(math.atan(a))
        platen_center_z = a * cx + b

        if CONSOLE_LOG_ENABLED:
            print(f"t={self.sim_time:6.3f} platenZ={platen_center_z:8.5f} tilt={tilt_deg:7.3f}deg "
                  f"cavityW={cavity_width:7.5f} cavityH={cavity_height:7.5f} spread={max_spread:.1e}")


def createScene(rootNode):
    launch_params_editor()

    plugins = [
        "Sofa.Component.AnimationLoop",                # DefaultAnimationLoop
        "Sofa.Component.Constraint.Projective",         # PartialFixedProjectiveConstraint
        "Sofa.Component.LinearSolver.Direct",           # EigenSimplicialLDLT
        "Sofa.Component.Mapping.Linear",                # BarycentricMapping, IdentityMapping
        "Sofa.Component.Mass",                          # UniformMass
        "Sofa.Component.MechanicalLoad",                # ConstantForceField
        "Sofa.Component.ODESolver.Backward",             # EulerImplicitSolver
        "Sofa.Component.SolidMechanics.FEM.Elastic",     # FastTetrahedralCorotationalForceField
        "Sofa.Component.Setting",                        # BackgroundSetting
        "Sofa.Component.StateContainer",                 # MechanicalObject
        "Sofa.Component.Topology.Container.Constant",    # MeshTopology (floor, platen, visu skins)
        "Sofa.Component.Topology.Container.Dynamic",     # TetrahedronSetTopologyContainer
        "Sofa.Component.Visual",                         # VisualStyle
        "Sofa.GL.Component.Rendering3D",                  # OglModel
        "Sofa.GL.Component.Shader",                       # LightManager, DirectionalLight
    ]
    for p in plugins:
        rootNode.addObject('RequiredPlugin', name='req_' + p.replace('.', '_'), pluginName=p)

    rootNode.gravity = GRAVITY
    rootNode.dt = DT

    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('BackgroundSetting', color=SKY_COLOR)

    # show_fem_elements toggles the raw FEM tetrahedra debug rendering
    # (hard-opaque, ignores block_alpha) on top of the normal skin/edge
    # visuals -- same flag and reasoning as ../PneuNetFinger's own.
    force_fields_flag = 'showForceFields' if SHOW_FEM_ELEMENTS else 'hideForceFields'
    rootNode.addObject('VisualStyle',
                        displayFlags='showVisualModels hideBehaviorModels '
                                     'hideCollisionModels hideBoundingCollisionModels '
                                     f'{force_fields_flag} hideWireframe')

    eye = _camera_eye()
    rootNode.addObject('InteractiveCamera', name='camera',
                        position=eye,
                        distance=CAMERA_DISTANCE,
                        fieldOfView=CAMERA_FOV,
                        orientation=_look_at_quaternion(eye, CAMERA_LOOKAT, world_up=(0.0, 0.0, 1.0)),
                        projectionType='Perspective',
                        activated=True)

    rootNode.addObject('LightManager', ambient=AMBIENT_COLOR)
    # Key light: direction is "which way the light travels", and a face is
    # lit exactly when its own outward normal has a positive dot product
    # with it -- so each axis's SIGN picks which of that axis's two faces
    # catches it (a positive x lights +X, a negative y lights -Y, etc).
    # The default [0.3, -0.6, 0.7] lights +X, -Y and +Z -- the z=+0.7 is
    # what makes the upward-facing faces the BRIGHT ones (this is the
    # strong light), like an overhead sun.
    # Named explicitly -- two DirectionalLight objects left unnamed both
    # default to the same auto-assigned name, which SOFA warns about as a
    # duplicate within the node.
    key_light = rootNode.addObject('DirectionalLight', name='keyLight',
                                    direction=KEY_LIGHT_DIRECTION, color=KEY_LIGHT_COLOR)
    # Fill light: by default lights -X and +Y (the two faces the key
    # light leaves dark), with z=0 so it deliberately does NOT reach -Z --
    # nothing needs the downward-facing faces lit, and giving that light
    # any z-component (either sign) would put brightness back on -Z or
    # take it off +Z. Dimmer than the key light by default so the block
    # still reads as having one dominant light direction rather than
    # looking flatly lit.
    rootNode.addObject('DirectionalLight', name='fillLight',
                        direction=FILL_LIGHT_DIRECTION, color=FILL_LIGHT_COLOR)

    if KEY_LIGHT_ROTATE_ENABLED:
        rootNode.addObject(LightRotationController(
            key_light, KEY_LIGHT_DIRECTION, KEY_LIGHT_ROTATE_PERIOD, DT, name='lightRotationCtrl'))

    # ---- floor (purely decorative, a real 3D slab) ----------------------
    # Top surface sits FLOOR_Z_GAP below BASE_Z, not exactly at it -- see
    # FLOOR_Z_GAP's own comment (avoids Z-fighting against the block's
    # bottom row, which IS pinned exactly at BASE_Z).
    floor_pts, floor_tris = _flat_shaded(*_aabb_box(
        -FLOOR_MARGIN, -FLOOR_MARGIN, BASE_Z - FLOOR_THICKNESS - FLOOR_Z_GAP,
        WIDTH + FLOOR_MARGIN, DEPTH + FLOOR_MARGIN, BASE_Z - FLOOR_Z_GAP))
    floor = rootNode.addChild('floor')
    floor.addObject('MeshTopology', name='topo', position=floor_pts, triangles=floor_tris)
    floor.addObject('OglModel', name='visual', src='@topo', color=FLOOR_COLOR)

    # ---- FEM block -------------------------------------------------------
    (positions, tetrahedra, bottom_indices, top_indices,
     cavity_indices, cavity_forces, cavity_probes) = build_block_mesh()
    cavity_area = ((CAVITY_I_HI - CAVITY_I_LO) * CELL_X) * ((CAVITY_K_HI - CAVITY_K_LO) * CELL_Z)
    total_mass = DENSITY * (WIDTH * HEIGHT * DEPTH - cavity_area * DEPTH)

    cube = rootNode.addChild('cube')
    cube.addObject('EulerImplicitSolver', name='odesolver',
                    rayleighStiffness=RAYLEIGH_STIFFNESS, rayleighMass=RAYLEIGH_MASS)
    cube.addObject('EigenSimplicialLDLT', template='CompressedRowSparseMatrixMat3x3')

    cube.addObject('TetrahedronSetTopologyContainer', name='container',
                    position=positions, tetrahedra=tetrahedra)
    cube.addObject('TetrahedronSetTopologyModifier')

    cube_dofs = cube.addObject('MechanicalObject', name='dofs', template='Vec3')
    cube.addObject('UniformMass', totalMass=total_mass)
    cube.addObject('FastTetrahedralCorotationalForceField', name='FEM', method='large',
                    youngModulus=YOUNG_MODULUS, poissonRatio=POISSON_RATIO)

    # ---- boundary conditions ---------------------------------------------
    cube.addObject('PartialFixedProjectiveConstraint', name='planeStrain',
                    indices=list(range(len(positions))), fixedDirections=[0, 1, 0])
    cube.addObject('PartialFixedProjectiveConstraint', name='bottomZ',
                    indices=bottom_indices, fixedDirections=[0, 1, 1])
    # Top row's Z is intentionally left unconstrained here -- TopPlateController
    # (added below) is what actually enforces its "shared tilting line" height.

    # ---- pressure loads: top row (downward) and cavity walls (outward) --
    top_forces = build_top_pressure_forces(PRESSURE)
    top_pressure_ff = cube.addObject('ConstantForceField', name='topPressure',
                                      indices=top_indices, forces=[[0.0, 0.0, 0.0]] * len(top_indices))
    cavity_pressure_ff = cube.addObject('ConstantForceField', name='cavityPressure',
                                         indices=cavity_indices, forces=[[0.0, 0.0, 0.0]] * len(cavity_indices))

    # ---- visual skin: all 6 faces of the (hollow) block -- front, back,
    # outer perimeter, and the cavity's own inner perimeter -- so it
    # reads as a genuine 3D tube rather than a flat picture, mapped onto
    # the FEM mesh via BarycentricMapping (skin vertices don't need to
    # coincide 1:1 with FEM node indices), same pattern as
    # ../PneuNetFinger and ../SqueezeSquare2D's own outer-skin visuals.
    front_pts, front_tris = _face_skin(0.0)
    back_pts, back_tris = _face_skin(DEPTH)
    outer_loop = _perimeter_loop(0, DIVS_X, 0, DIVS_Z)
    outer_pts, outer_tris = _side_skin(outer_loop, outward=True)
    cavity_loop = _perimeter_loop(CAVITY_I_LO, CAVITY_I_HI, CAVITY_K_LO, CAVITY_K_HI)
    cavity_pts, cavity_tris = _side_skin(cavity_loop, outward=False)
    skin_pts, skin_tris = _merge([
        (front_pts, front_tris), (back_pts, back_tris), (outer_pts, outer_tris), (cavity_pts, cavity_tris),
    ])
    # Flat-shaded (see _flat_shaded) for the FILL mesh only -- the edge
    # overlay below dedupes edges from the original (undupped) skin_tris,
    # since duplicated vertices would just make every edge appear twice.
    flat_skin_pts, flat_skin_tris = _flat_shaded(skin_pts, skin_tris)
    visu = cube.addChild('visu')
    visu.addObject('MeshTopology', name='topo', position=flat_skin_pts, triangles=flat_skin_tris)
    visu.addObject('OglModel', name='visual', src='@topo', color=RUBBER_COLOR)
    visu.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # ---- block edge overlay (optional) -----------------------------------
    # A genuine "filled surface + line outline" look needs a SEPARATE
    # OglModel drawing just the skin's own edges: VisualStyle's global
    # showWireframe/hideWireframe flag switches the WHOLE scene between
    # filled and line-only rendering, replacing the fill rather than
    # overlaying lines on it (confirmed by ../PneuNetFinger's own
    # comment on exactly this). Built as its own sibling node under
    # `cube` (not nested inside `visu`) so its BarycentricMapping's
    # implicit ancestor search for an input MechanicalObject only has to
    # climb one level, same as `visu`'s own.
    if BLOCK_SHOW_EDGES:
        edge_set = set()
        for tri in skin_tris:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge_set.add((a, b) if a < b else (b, a))
        edges = [list(e) for e in sorted(edge_set)]
        visu_edges = cube.addChild('visuEdges')
        visu_edges.addObject('MeshTopology', name='topo', position=skin_pts, edges=edges)
        # keepLines defaults to False -- an OglModel with only `edges` (no
        # triangles/quads of its own) otherwise silently drops them and
        # renders nothing, which is why block_edge_color appeared to have
        # no effect regardless of block_show_edges (verified directly:
        # the Data field's own edges.value held the right data, but
        # nothing was drawn without this). lineWidth bumped up from its
        # own default of 1.0 so the color is actually easy to see, not
        # just technically present.
        visu_edges.addObject('OglModel', name='visual', src='@topo', color=BLOCK_EDGE_COLOR,
                              keepLines=True, lineWidth=2.0)
        visu_edges.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # ---- platen (purely visual -- position/tilt driven every step by
    # TopPlateController, from the same line-fit it enforces on the mesh) --
    platen_half_span = WIDTH / 2.0 + PLATEN_OVERHANG
    platen_pts, platen_tris = _flat_shaded(*_thin_box(
        WIDTH / 2.0 - platen_half_span, HEIGHT, WIDTH / 2.0 + platen_half_span, HEIGHT,
        PLATEN_HALF_THICKNESS, -PLATEN_OVERHANG, DEPTH + PLATEN_OVERHANG))
    platen = rootNode.addChild('platen')
    platen.addObject('MeshTopology', name='topo', position=platen_pts, triangles=platen_tris)
    platen_mstate = platen.addObject('MechanicalObject', name='dofs', template='Vec3', src='@topo')
    # OglModel lives in its own child node rather than alongside `dofs`
    # (same reasoning as the block's own 'visu' child of 'cube'): a Node
    # can only hold one BaseState, and OglModel counts as one in this
    # SOFA version, so putting it in the same node as the MechanicalObject
    # triggers a "duplicate BaseState" warning (harmless in practice --
    # the sibling Pendulum/DoublePendulum/RodWithBalls projects all use
    # the same-node form and still render correctly -- but avoidable).
    if SHOW_PLATEN:
        platen_visu = platen.addChild('visu')
        platen_visu.addObject('OglModel', name='visual', src='@../topo', color=PLATEN_COLOR)
        platen_visu.addObject('IdentityMapping', input='@../dofs', output='@visual')

    rootNode.addObject(TopPlateController(
        cube_dofs, top_indices,
        [(top_pressure_ff, top_forces, TOP_RAMP_START, TOP_RAMP_TIME),
         (cavity_pressure_ff, cavity_forces, CAVITY_RAMP_START, CAVITY_RAMP_TIME)],
        DT, platen_mstate, platen_half_span, cavity_probes, name='topPlateCtrl'))


def main():
    import Sofa.Simulation

    # Headless entry point -- the SOFA GUI (run_imgui.bat/run_glfw.bat)
    # never calls main() at all, it loads this file and calls
    # createScene() directly, so this only ever suppresses the editor for
    # a plain `python AsymmetricSquareTube.py` run (setdefault so an
    # operator who explicitly wants it anyway can still unset/override
    # this first).
    os.environ.setdefault('ASYMMETRICSQUARETUBE_NO_EDITOR', '1')

    root = Sofa.Core.Node('root')
    createScene(root)
    Sofa.Simulation.init(root)

    steps = int(round(RUN_DURATION / DT))
    for _ in range(steps):
        Sofa.Simulation.animate(root, DT)

    Sofa.Simulation.unload(root)


if __name__ == '__main__':
    main()
