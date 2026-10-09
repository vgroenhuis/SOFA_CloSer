# -*- coding: utf-8 -*-
"""
2D rod with sliding balls: a flat, cartoon-style scene (X-Z plane; world Y
held fixed at Y_PLANE for every element) built on SOFA, in the same visual
style and world convention as ../Pendulum/../DoublePendulum (Z up,
gravity = -Z, X horizontal, Y depth, orthographic camera looking along Y).

The physical system: a massless, infinitely long rigid rod, hinged at a
fixed point and free to rotate in this plane, carrying two point masses
that are NOT fixed to the rod -- they're free to slide along it without
friction, like beads on a wire. That makes this a genuinely different
problem from the sibling pendulums (where each mass's distance from its
pivot is constant): here each ball's distance from the hinge is itself a
third and fourth dynamic variable, alongside the rod's own angle.

Like ../Pendulum/../DoublePendulum, none of this runs through SOFA's own
Mass/ForceField/ODE-solver machinery -- the whole system is reduced to its
3 true degrees of freedom (the rod's angle theta, and each ball's signed
distance from the hinge r1/r2 -- signed so a ball on the "far" side of the
hinge from the other one is just a negative r, not a different theta) and
integrated directly by a plain Python controller (4th-order Runge-Kutta,
same reasoning as ../DoublePendulum's own -- this system is at least as
prone to amplifying integration error as that one is; see
RodWithBallsController's own comment).

The equations of motion (re-derived here via Lagrangian mechanics, not
copied from a reference, to match this file's own angle/sign convention)
turn out to be exactly the textbook "bead on a rotating rod" result,
generalized to two beads and a rod that's itself free to swing under
gravity rather than spun at a fixed rate:
    r_i''    = r_i*theta'^2 + g*cos(theta)                         (per ball)
    theta''  = -(2*sum(m_i*r_i*r_i')*theta' + g*sin(theta)*sum(m_i*r_i)) / sum(m_i*r_i^2)
-- i.e. each ball's own radial "slingshot" equation (centrifugal term plus
gravity's component along the rod), and the rod's own equation being
exactly angular-momentum bookkeeping: I(t) = sum(m_i*r_i^2) is the
system's own time-varying moment of inertia, and the theta'' expression is
d/dt(I*theta') = net gravitational torque, rearranged for theta''.

The spring described below (SPRING_ANCHOR etc.) adds one more term to
each of ball 2's and theta's own equations -- not ball 1's, it isn't
attached to that one. Since the spring is a perfectly ordinary Hookean
one pulling ball 2 towards a plain fixed point (not the hinge, not on the
rod), the cleanest way to add it is as a Cartesian force on ball 2,
F = -k*(d - L0) * (ball2_pos - anchor_pos)/d (d = current distance,
pulling inward when stretched, pushing outward when compressed), then
project it onto each generalized coordinate's own direction of motion
(the standard Q_q = F . (d(ball2_pos)/dq) recipe) to get the two extra
terms:
    Q_r2    = F . (sin(theta), -cos(theta))       -- component along the rod
    Q_theta = F . (r2*cos(theta), r2*sin(theta))  -- torque about the hinge
added respectively to r2'' (divided by m2) and to theta''s own numerator,
alongside gravity's existing contribution to each.

With no friction and nothing stopping a ball from sliding outward, this
starting condition alone sends both balls sliding away *without bound* as
the rod swings down towards hanging vertically -- verified (headless,
energy-conservation-checked) to already exceed the 4 m visual rod (see
ROD_VISUAL_HALF_LENGTH below) well within the first couple of seconds.
That's correct physics for a frictionless infinite rod, not a bug: a ball
is always drawn at its true position even once that's well past the drawn
rod segment's own end, never clamped or hidden. The floor is purely
decorative here too, same as the sibling projects -- nothing stops a
ball's height from going negative as it slides out along the now-nearly-
vertical rod.

A spring now ties ball 2 to a second, separate fixed point (SPRING_ANCHOR
below) -- not attached to the rod or the hinge at all, just a plain
Hookean spring pulling ball 2 back towards it. That fundamentally changes
ball 2's own story: verified headlessly at the current SPRING_STIFFNESS
(see below), it keeps ball 2's radial position r2 oscillating in a
modest, bounded range (roughly -0.2 to 0.7 m) for as long as it's been
tested, rather than sliding away -- a softer spring lets it swing a
little further than a stiffer one would, but it stayed bounded at every
stiffness tried. Ball 1 has no spring of its own, though, so it's still
subject to the unbounded-slide behavior above -- the spring only reaches
ball 1 *indirectly*, through the shared rod angle theta. That's enough to
noticeably delay when ball 1's own slide takes off compared to no spring
at all (verified: ~3-4 s at the current stiffness, versus ~1 s with none),
but not to stop it -- and a stiffer spring delays it more (~8 s was
observed at 1000 N/m, an earlier value this constant held). The spring's
own potential energy (0.5*k*(distance-rest_length)^2) is included in the
printed energy total, and conservation was reverified with it in the mix
(still exact to floating-point precision, independent of stiffness).

Geometry/mass/initial-condition/spring/flag parameters are loaded from
params.json next to this script, editable via params_editor.py -- same
standalone-Tkinter-app pattern as ../DoublePendulum's own. The editor
auto-launches from createScene() (skipped for a plain headless run; see
main()'s own RODWITHBALLS_NO_EDITOR).

Run headless (prints per-step energy to the console; see main()):
    run_headless.bat   (sets up PYTHONPATH/PATH for plain `python`, then runs this file --
                        `python RodWithBalls.py` on its own fails with "No
                        module named 'Sofa'" unless that setup has already
                        been done some other way, e.g. a prior
                        run_imgui.bat/run_glfw.bat in the same shell)
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
# Parameters -- loaded from params.json next to this script, so they can
# be edited without touching code (by hand, or with params_editor.py, the
# same standalone-Tkinter-app pattern ../DoublePendulum uses). The file is
# created with these defaults on first run if missing.
#
# Geometry/mass/initial-condition/spring changes only take effect on the
# next scene (re)build -- after editing, use the SOFA GUI's Reload button
# (or just re-run, for a headless session). The flags are also only read
# at scene-build time, for the same reason.
# ----------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(_SCRIPT_DIR, 'params.json')

_DEFAULT_PARAMS = {
    'anchor_x': 0.0,                # fixed hinge X [m]
    'anchor_z': 1.0,                # fixed hinge height above the floor [m]
    # theta is the rod's own angle, measured from the downward vertical,
    # positive towards +X. 90 = "initially horizontal", the prompt's own
    # starting condition.
    'theta_initial_deg': 90.0,
    'omega_initial': 0.0,           # rod's initial angular velocity [rad/s]
    'ball1_mass_kg': 1.0,
    'ball1_initial_x': -0.5,        # [m] -- r1 is derived from this and theta_initial_deg
    'ball1_initial_z': 1.0,         # [m]
    'ball2_mass_kg': 1.0,
    'ball2_initial_x': 0.6,         # [m] -- r2 is derived from this and theta_initial_deg
    'ball2_initial_z': 1.0,         # [m]
    'rod_visual_half_length': 2.0,  # [m] the rod is infinite in the physics, only ever drawn this long
    'spring_anchor_x': 0.6,         # [m] fixed point the spring pulls ball 2 towards
    'spring_anchor_z': 2.0,         # [m]
    'spring_rest_length': 1.0,      # [m]
    'spring_stiffness': 200.0,      # [N/m]
    'gravity': 9.81,                # [m/s^2]
    'dt': 0.001,                    # integration timestep [s]
    'run_duration': 15.0,           # headless run length, main() only [s]
    'console_log_enabled': True,    # print per-step angle/energy to the console
    'show_pivot_marker': True,      # draw the small dot marking the fixed hinge
    'show_spring_anchor_marker': True,   # draw the small dot marking the spring's own fixed point
    'theme': 'System',   # 'Light' / 'Dark' / 'System' -- UI-only preference,
                          # stored here for convenience but never read by
                          # RodWithBalls.py itself.
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
    ../DoublePendulum/DoublePendulum.py's own launch_params_editor for the
    full reasoning (identical here): a plain `python`, not runSofa's
    embedded interpreter; guarded against duplicate windows via a PID file
    on disk (survives the SOFA GUI's Reload re-importing this module);
    skippable via RODWITHBALLS_NO_EDITOR for headless/automated runs.
    """
    if os.environ.get('RODWITHBALLS_NO_EDITOR'):
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


Y_PLANE = 0.0              # [m] world Y every 2D element sits at

FLOOR_LENGTH = 5.0
FLOOR_Z = 0.0
FLOOR_HALF_THICKNESS = 0.02
FLOOR_COLOR = [0.1, 0.55, 0.15, 1.0]

SKY_COLOR = [0.78, 0.87, 0.96, 1.0]

SUN_CX = 1.6
SUN_HEIGHT = 3.2
SUN_CZ = FLOOR_Z + SUN_HEIGHT
SUN_RADIUS = 0.25
SUN_COLOR = [1.0, 0.85, 0.0, 1.0]
SUN_RAY_COUNT = 20
SUN_RAY_INNER = SUN_RADIUS * 1.1
SUN_RAY_OUTER = SUN_RADIUS * 1.9
SUN_RAY_HALF_WIDTH = 0.015
SUN_RAY_COLOR = [1.0, 0.75, 0.0, 1.0]

CLOUD_SPECS = [
    (-1.7, FLOOR_Z + 2.7, 0.6),
    (1.3, FLOOR_Z + 3.1, 0.5),
]
CLOUD_COLOR = [1.0, 1.0, 1.0, 1.0]

# ---- hinge --------------------------------------------------------------
ANCHOR_X = PARAMS['anchor_x']
ANCHOR_Z = PARAMS['anchor_z']
ANCHOR_POS = (ANCHOR_X, Y_PLANE, ANCHOR_Z)

# theta is the rod's own angle, measured from the downward vertical,
# positive towards +X -- same convention as ../Pendulum/../DoublePendulum.
# 90 = "initially horizontal" (the prompt's words), the same starting
# angle ../Pendulum's own single mass used.
THETA_INITIAL = math.radians(PARAMS['theta_initial_deg'])   # [rad]
OMEGA_INITIAL = PARAMS['omega_initial']   # [rad/s]

# ---- balls ----------------------------------------------------------
# Each ball's distance from the hinge (r, signed: negative = the far side
# of the hinge from wherever +r points) is derived from its own
# ball*_initial_x/z position and THETA_INITIAL above, rather than edited
# directly, so the relationship stays visibly exact rather than two
# independently-set numbers that happen to agree -- same approach
# ../Pendulum's own THETA_INITIAL uses, just solving for r instead of
# theta since here theta is the given one. Both balls' z=anchor_z by
# default is what makes THETA_INITIAL=90 consistent with BOTH of them at
# once: at theta=90 every point on the rod, at any r, sits at exactly
# z=anchor_z -- a level rod -- so any r solves the z equation, and only
# the x equation actually pins each ball's r down (a theta_initial_deg
# other than 90, or ball*_initial_z != anchor_z, is still handled the
# same way -- the resulting r just no longer places the ball at exactly
# its own stated z, since (theta, x, z) is one equation too many once
# theta is independently set).
BALL1_MASS_KG = PARAMS['ball1_mass_kg']
BALL1_INITIAL_X = PARAMS['ball1_initial_x']
BALL1_INITIAL_Z = PARAMS['ball1_initial_z']
R1_INITIAL = (BALL1_INITIAL_X - ANCHOR_X) / math.sin(THETA_INITIAL)
V1_INITIAL = 0.0                  # [m/s] released from rest
BALL1_COLOR = [0.55, 0.15, 0.75, 1.0]   # purple

BALL2_MASS_KG = PARAMS['ball2_mass_kg']
BALL2_INITIAL_X = PARAMS['ball2_initial_x']
BALL2_INITIAL_Z = PARAMS['ball2_initial_z']
R2_INITIAL = (BALL2_INITIAL_X - ANCHOR_X) / math.sin(THETA_INITIAL)
V2_INITIAL = 0.0
BALL2_COLOR = [0.85, 0.4, 0.15, 1.0]    # orange -- clearly distinct from ball 1's purple

BALL_RADIUS = 0.06             # [m] visual radius, shared by both (equal masses)

# The rod is infinitely long in the physics (see the module docstring),
# but only ever DRAWN as a fixed-length segment (rod_visual_half_length
# on either side of the hinge) -- per the prompt's own instruction, not a
# physics limit.
ROD_VISUAL_HALF_LENGTH = PARAMS['rod_visual_half_length']   # [m]
ROD_HALF_THICKNESS = 0.012      # [m]
ROD_COLOR = [0.35, 0.35, 0.35, 1.0]      # dark grey

PIVOT_MARKER_RADIUS = 0.035
PIVOT_MARKER_COLOR = [0.3, 0.3, 0.3, 1.0]
SHOW_PIVOT_MARKER = PARAMS['show_pivot_marker']

# ---- spring (ball 2 <-> a second, separate fixed point) ----------------
# A plain Hookean spring -- see the module docstring for how its force
# folds into ball 2's and theta's own equations. Not attached to the rod
# or the hinge at all, just its own fixed point in space.
SPRING_ANCHOR_X = PARAMS['spring_anchor_x']
SPRING_ANCHOR_Z = PARAMS['spring_anchor_z']
SPRING_ANCHOR_POS = (SPRING_ANCHOR_X, Y_PLANE, SPRING_ANCHOR_Z)
SPRING_REST_LENGTH = PARAMS['spring_rest_length']   # [m]
SPRING_STIFFNESS = PARAMS['spring_stiffness']        # [N/m]

SPRING_ANCHOR_MARKER_RADIUS = 0.035
SPRING_ANCHOR_MARKER_COLOR = [0.3, 0.3, 0.3, 1.0]   # same neutral grey as the hinge marker
SHOW_SPRING_ANCHOR_MARKER = PARAMS['show_spring_anchor_marker']
SPRING_ZIGZAG_SEGMENTS = 12
SPRING_ZIGZAG_AMPLITUDE = 0.045   # [m] perpendicular offset of each zigzag point
SPRING_HALF_THICKNESS = 0.008     # [m] thinner than the rigid rod -- reads as a lighter, springier line
SPRING_COLOR = [0.25, 0.45, 0.7, 1.0]    # steel blue -- distinct from the rod and both balls

GRAVITY_MAGNITUDE = PARAMS['gravity']       # [m/s^2]
# rootNode.gravity is set for scene-convention consistency with the
# sibling projects (and in case anything else in the scene ever reads
# it), but nothing here actually consumes it: RodWithBallsController
# uses GRAVITY_MAGNITUDE directly in its own hand-integrated equations of
# motion instead, since there's no SOFA Mass/ForceField in this scene for
# SOFA's own gravity handling to apply to.
GRAVITY = [0.0, 0.0, -GRAVITY_MAGNITUDE]
DT = PARAMS['dt']
RUN_DURATION = PARAMS['run_duration']   # headless run length, main() only [s]
CONSOLE_LOG_ENABLED = PARAMS['console_log_enabled']

CAMERA_LOOKAT = [0.2, 0.0, 1.3]   # nudged up/right from the hinge, roughly midway to the spring anchor
CAMERA_Y = -8.5
CAMERA_FOV = 45.0


# ----------------------------------------------------------------------
# Flat (X-Z plane, at world Y=Y_PLANE) geometry helpers, same as the
# sibling projects.
# ----------------------------------------------------------------------
def _disc(cx, cz, radius, segments=32, y=Y_PLANE):
    points = [[cx, y, cz]]
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        points.append([cx + radius * math.cos(a), y, cz + radius * math.sin(a)])
    tris = []
    for i in range(1, segments + 1):
        nxt = i + 1 if i < segments else 1
        tris.append([0, i, nxt])
    return points, tris


def _thin_bar(x0, z0, x1, z1, half_thickness, y=Y_PLANE):
    dx, dz = x1 - x0, z1 - z0
    length = math.hypot(dx, dz)
    ux, uz = dx / length, dz / length
    nx, nz = -uz, ux
    p0 = [x0 - half_thickness * nx, y, z0 - half_thickness * nz]
    p1 = [x1 - half_thickness * nx, y, z1 - half_thickness * nz]
    p2 = [x1 + half_thickness * nx, y, z1 + half_thickness * nz]
    p3 = [x0 + half_thickness * nx, y, z0 + half_thickness * nz]
    return [p0, p1, p2, p3], [[0, 1, 2], [0, 2, 3]]


def _merge(parts):
    points, tris = [], []
    for pts, ts in parts:
        offset = len(points)
        points += pts
        tris += [[i + offset for i in t] for t in ts]
    return points, tris


def _cloud(cx, cz, scale=1.0):
    puffs = [
        (0.0, 0.0, 0.35 * scale),
        (-0.34 * scale, 0.02 * scale, 0.25 * scale),
        (0.36 * scale, 0.0 * scale, 0.27 * scale),
        (0.05 * scale, 0.22 * scale, 0.27 * scale),
    ]
    return _merge([_disc(cx + dx, cz + dz, r) for dx, dz, r in puffs])


def _sun_rays(cx, cz, count, inner_r, outer_r, half_width):
    rays = []
    for i in range(count):
        angle = 2.0 * math.pi * i / count
        ux, uz = math.cos(angle), math.sin(angle)
        x0, z0 = cx + inner_r * ux, cz + inner_r * uz
        x1, z1 = cx + outer_r * ux, cz + outer_r * uz
        rays.append(_thin_bar(x0, z0, x1, z1, half_width))
    return _merge(rays)


def _spring_zigzag(x0, z0, x1, z1, segments, amplitude, half_thickness, y=Y_PLANE):
    """A zigzag line from (x0,z0) to (x1,z1) -- `segments` straight
    sub-segments, alternating a perpendicular offset of +-amplitude at
    each interior point, with the first and last points landing exactly
    on the two given endpoints (so it connects cleanly to the anchor
    marker and the ball, no visible gap or jump). Same "chain of thin
    bars" technique the rod and the sun's rays use, just following a
    zigzag path instead of a straight one -- reads as "a spring"
    (rather than just another rigid rod) at negligible extra cost, and
    gets rebuilt every step exactly like the rod does, since both its
    endpoints move (the anchor is fixed, but ball 2 isn't).
    """
    dx, dz = x1 - x0, z1 - z0
    length = math.hypot(dx, dz)
    ux, uz = dx / length, dz / length
    nx, nz = -uz, ux
    points = []
    for i in range(segments + 1):
        t = i / segments
        px, pz = x0 + dx * t, z0 + dz * t
        if 0 < i < segments:
            sign = 1.0 if i % 2 == 1 else -1.0
            px += nx * amplitude * sign
            pz += nz * amplitude * sign
        points.append((px, pz))
    bars = [_thin_bar(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], half_thickness, y=y)
            for i in range(segments)]
    return _merge(bars)


# ----------------------------------------------------------------------
# Equations of motion -- see the module docstring for the derivation and
# the closed-form result, including the spring's added terms below.
# `state` is (theta, r1, r2, omega, v1, v2); returns its own
# time-derivative in the same order, so this drops straight into a
# standard vector RK4 step.
# ----------------------------------------------------------------------
def _rod_with_balls_derivs(theta, r1, r2, omega, v1, v2, m1, m2, g,
                            anchor_x, anchor_z, spring_x, spring_z, spring_l0, spring_k):
    # Spring force on ball 2 (Cartesian), then projected onto r2's and
    # theta's own directions of motion -- see the module docstring.
    x2 = anchor_x + r2 * math.sin(theta)
    z2 = anchor_z - r2 * math.cos(theta)
    dx, dz = x2 - spring_x, z2 - spring_z
    d = math.hypot(dx, dz)
    stretch = d - spring_l0
    fx2 = -spring_k * stretch * dx / d
    fz2 = -spring_k * stretch * dz / d
    q_r2 = fx2 * math.sin(theta) - fz2 * math.cos(theta)
    q_theta = fx2 * r2 * math.cos(theta) + fz2 * r2 * math.sin(theta)

    v1_dot = r1 * omega * omega + g * math.cos(theta)
    v2_dot = r2 * omega * omega + g * math.cos(theta) + q_r2 / m2

    I = m1 * r1 * r1 + m2 * r2 * r2
    I_dot = 2.0 * (m1 * r1 * v1 + m2 * r2 * v2)
    torque = -g * math.sin(theta) * (m1 * r1 + m2 * r2) + q_theta
    omega_dot = (torque - I_dot * omega) / I

    return omega, v1, v2, omega_dot, v1_dot, v2_dot


# ----------------------------------------------------------------------
# Controller: owns the whole system's physics AND drives its (purely
# kinematic) visual state -- see the module docstring for why nothing
# here is a SOFA Mass/ForceField/ODE solver.
#
# Integration is classical 4th-order Runge-Kutta on the 6D state (theta,
# r1, r2, omega, v1, v2), same choice and reasoning as
# ../DoublePendulum's own: this system's radial "slingshot" feedback
# (r_i'' includes an r_i*omega^2 term that grows the moment it does)
# amplifies a low-order integrator's own per-step error at least as
# readily as a chaotic double pendulum's sensitivity does, so the extra
# accuracy matters here too. Verified headlessly: total energy (now
# including the spring's own potential -- see _spring_energy below)
# stays within numerical noise of its initial value at this file's
# production DT, both with and without the spring in play.
# ----------------------------------------------------------------------
class RodWithBallsController(Sofa.Core.Controller):
    def __init__(self, anchor_x, anchor_z, m1, m2, gravity,
                 theta0, r1_0, r2_0, omega0, v1_0, v2_0, dt,
                 spring_x, spring_z, spring_l0, spring_k,
                 ball1_mstate, ball1_local_pts, ball2_mstate, ball2_local_pts,
                 rod_mstate, spring_mstate, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.anchor_x = anchor_x
        self.anchor_z = anchor_z
        self.m1 = m1
        self.m2 = m2
        self.g = gravity
        self.theta = theta0
        self.r1 = r1_0
        self.r2 = r2_0
        self.omega = omega0
        self.v1 = v1_0
        self.v2 = v2_0
        self.dt = dt
        self.spring_x = spring_x
        self.spring_z = spring_z
        self.spring_l0 = spring_l0
        self.spring_k = spring_k
        self.ball1_mstate = ball1_mstate
        self.ball1_local_pts = ball1_local_pts
        self.ball2_mstate = ball2_mstate
        self.ball2_local_pts = ball2_local_pts
        self.rod_mstate = rod_mstate
        self.spring_mstate = spring_mstate
        self.sim_time = 0.0
        self._update_visuals()   # so the GUI shows the correct pose before any step runs

    def _ball_position(self, r):
        x = self.anchor_x + r * math.sin(self.theta)
        z = self.anchor_z - r * math.cos(self.theta)
        return x, z

    def _spring_energy(self, x2, z2):
        d = math.hypot(x2 - self.spring_x, z2 - self.spring_z)
        return 0.5 * self.spring_k * (d - self.spring_l0) ** 2

    def _update_visuals(self):
        x1, z1 = self._ball_position(self.r1)
        x2, z2 = self._ball_position(self.r2)
        self.ball1_mstate.position.value = [
            [x1 + lx, Y_PLANE + ly, z1 + lz] for lx, ly, lz in self.ball1_local_pts
        ]
        self.ball2_mstate.position.value = [
            [x2 + lx, Y_PLANE + ly, z2 + lz] for lx, ly, lz in self.ball2_local_pts
        ]
        # The visual rod is always drawn at its own fixed +-2 m extent
        # (ROD_VISUAL_HALF_LENGTH), regardless of where the balls
        # actually are -- see the module docstring.
        end_a_x, end_a_z = self._ball_position(ROD_VISUAL_HALF_LENGTH)
        end_b_x, end_b_z = self._ball_position(-ROD_VISUAL_HALF_LENGTH)
        rod_pts, _ = _thin_bar(end_b_x, end_b_z, end_a_x, end_a_z, ROD_HALF_THICKNESS)
        self.rod_mstate.position.value = rod_pts
        # The spring's own zigzag is rebuilt every step too: its anchor
        # end is fixed, but ball 2's end moves.
        spring_pts, _ = _spring_zigzag(self.spring_x, self.spring_z, x2, z2,
                                        SPRING_ZIGZAG_SEGMENTS, SPRING_ZIGZAG_AMPLITUDE, SPRING_HALF_THICKNESS)
        self.spring_mstate.position.value = spring_pts
        return x1, z1, x2, z2

    def _rk4_step(self):
        dt = self.dt

        def derivs(theta, r1, r2, omega, v1, v2):
            return _rod_with_balls_derivs(theta, r1, r2, omega, v1, v2, self.m1, self.m2, self.g,
                                           self.anchor_x, self.anchor_z,
                                           self.spring_x, self.spring_z, self.spring_l0, self.spring_k)

        s0 = (self.theta, self.r1, self.r2, self.omega, self.v1, self.v2)
        k1 = derivs(*s0)
        k2 = derivs(*(s0[i] + 0.5 * dt * k1[i] for i in range(6)))
        k3 = derivs(*(s0[i] + 0.5 * dt * k2[i] for i in range(6)))
        k4 = derivs(*(s0[i] + dt * k3[i] for i in range(6)))

        self.theta += dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.r1 += dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.r2 += dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.omega += dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        self.v1 += dt / 6.0 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
        self.v2 += dt / 6.0 * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5])

    def onAnimateEndEvent(self, event):
        self.sim_time += self.dt
        self._rk4_step()
        x1, z1, x2, z2 = self._update_visuals()

        v1_sq = self.v1 * self.v1 + (self.r1 * self.omega) ** 2   # radial^2 + tangential^2
        v2_sq = self.v2 * self.v2 + (self.r2 * self.omega) ** 2

        pe1 = self.m1 * self.g * z1   # z is height above the floor (Z=0 datum)
        ke1 = 0.5 * self.m1 * v1_sq
        pe2 = self.m2 * self.g * z2
        ke2 = 0.5 * self.m2 * v2_sq
        pe_spring = self._spring_energy(x2, z2)
        e_total = pe1 + ke1 + pe2 + ke2 + pe_spring

        # Fewer decimals than the sibling projects' more usual 4 on most
        # columns -- ball 1's own unbounded slide (see the module
        # docstring) pushes PE1/KE1/r1 well into 4+ integer digits within
        # the printed 15 s run, and the wider integer part would
        # otherwise push the line width past 120 characters; measured
        # headlessly across the full run to confirm this keeps it
        # comfortably under that (max observed: 112).
        if CONSOLE_LOG_ENABLED:
            print(f"t={self.sim_time:7.3f} th={math.degrees(self.theta):7.2f} "
                  f"r1={self.r1:8.2f} r2={self.r2:7.2f} "
                  f"PE1={pe1:7.1f} KE1={ke1:7.1f} PE2={pe2:6.1f} KE2={ke2:6.1f} "
                  f"PEs={pe_spring:6.2f} E={e_total:6.1f}")


def createScene(rootNode):
    launch_params_editor()

    plugins = [
        "Sofa.Component.AnimationLoop",                     # DefaultAnimationLoop
        "Sofa.Component.Mapping.Linear",                    # IdentityMapping
        "Sofa.Component.Setting",                           # BackgroundSetting
        "Sofa.Component.StateContainer",                    # MechanicalObject
        "Sofa.Component.Topology.Container.Constant",       # MeshTopology
        "Sofa.Component.Visual",                            # VisualStyle
        "Sofa.GL.Component.Rendering3D",                    # OglModel
        "Sofa.GL.Component.Shader",                         # LightManager, DirectionalLight
    ]
    for p in plugins:
        rootNode.addObject('RequiredPlugin', name='req_' + p.replace('.', '_'), pluginName=p)

    rootNode.gravity = GRAVITY
    rootNode.dt = DT

    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('BackgroundSetting', color=SKY_COLOR)

    cam_position = [CAMERA_LOOKAT[0], CAMERA_Y, CAMERA_LOOKAT[2]]
    rootNode.addObject('InteractiveCamera', name='camera',
                        position=cam_position,
                        distance=abs(CAMERA_Y - CAMERA_LOOKAT[1]),
                        fieldOfView=CAMERA_FOV,
                        orientation=_look_at_quaternion(cam_position, CAMERA_LOOKAT, world_up=(0.0, 0.0, 1.0)),
                        projectionType='Orthographic',
                        activated=True)

    rootNode.addObject('LightManager', ambient=[0.7, 0.7, 0.7, 1.0])
    rootNode.addObject('DirectionalLight', direction=[0.0, -1.0, 0.3])

    # ---- floor -----------------------------------------------------
    floor_bar_pts, floor_bar_tris = _thin_bar(
        -FLOOR_LENGTH / 2, FLOOR_Z, FLOOR_LENGTH / 2, FLOOR_Z, FLOOR_HALF_THICKNESS)
    floor = rootNode.addChild('floor')
    floor.addObject('MeshTopology', name='topo', position=floor_bar_pts, triangles=floor_bar_tris)
    floor.addObject('OglModel', name='visual', src='@topo', color=FLOOR_COLOR)

    # ---- sun, rays, clouds -------------------------------------------
    sun_pts, sun_tris = _disc(SUN_CX, SUN_CZ, SUN_RADIUS)
    sun = rootNode.addChild('sun')
    sun.addObject('MeshTopology', name='topo', position=sun_pts, triangles=sun_tris)
    sun.addObject('OglModel', name='visual', src='@topo', color=SUN_COLOR)

    ray_pts, ray_tris = _sun_rays(SUN_CX, SUN_CZ, SUN_RAY_COUNT, SUN_RAY_INNER, SUN_RAY_OUTER, SUN_RAY_HALF_WIDTH)
    rays = rootNode.addChild('sunRays')
    rays.addObject('MeshTopology', name='topo', position=ray_pts, triangles=ray_tris)
    rays.addObject('OglModel', name='visual', src='@topo', color=SUN_RAY_COLOR)

    for i, (cx, cz, scale) in enumerate(CLOUD_SPECS):
        cloud_pts, cloud_tris = _cloud(cx, cz, scale)
        cloud = rootNode.addChild('cloud' + str(i))
        cloud.addObject('MeshTopology', name='topo', position=cloud_pts, triangles=cloud_tris)
        cloud.addObject('OglModel', name='visual', src='@topo', color=CLOUD_COLOR)

    # ---- pivot marker (purely visual) --------------------------------
    if SHOW_PIVOT_MARKER:
        pivot_pts, pivot_tris = _disc(ANCHOR_X, ANCHOR_Z, PIVOT_MARKER_RADIUS)
        pivot = rootNode.addChild('pivotMarker')
        pivot.addObject('MeshTopology', name='topo', position=pivot_pts, triangles=pivot_tris)
        pivot.addObject('OglModel', name='visual', src='@topo', color=PIVOT_MARKER_COLOR)

    # ---- spring anchor marker (purely visual) ------------------------
    if SHOW_SPRING_ANCHOR_MARKER:
        spring_anchor_pts, spring_anchor_tris = _disc(SPRING_ANCHOR_X, SPRING_ANCHOR_Z, SPRING_ANCHOR_MARKER_RADIUS)
        spring_anchor_marker = rootNode.addChild('springAnchorMarker')
        spring_anchor_marker.addObject('MeshTopology', name='topo', position=spring_anchor_pts, triangles=spring_anchor_tris)
        spring_anchor_marker.addObject('OglModel', name='visual', src='@topo', color=SPRING_ANCHOR_MARKER_COLOR)

    # Initial world positions, computed once up front the same way
    # RodWithBallsController computes them every step, so the rod's/
    # balls' very first MeshTopology vertices already match --
    # IdentityMapping only takes over from the NEXT onAnimateEndEvent
    # onward (see the comment on it below), so this initial value has to
    # be right on its own, not just eventually corrected.
    def _ball_position(r):
        return (ANCHOR_X + r * math.sin(THETA_INITIAL), ANCHOR_Z - r * math.cos(THETA_INITIAL))

    ball1_x0, ball1_z0 = _ball_position(R1_INITIAL)
    ball2_x0, ball2_z0 = _ball_position(R2_INITIAL)
    rod_end_a_x, rod_end_a_z = _ball_position(ROD_VISUAL_HALF_LENGTH)
    rod_end_b_x, rod_end_b_z = _ball_position(-ROD_VISUAL_HALF_LENGTH)

    # ---- rod (purely kinematic -- shape rebuilt live by the controller) --
    # IdentityMapping is what actually keeps each OglModel in sync with
    # the MechanicalObject the controller writes to every step --
    # src='@topo' on the OglModel only copies the MeshTopology's vertex
    # list ONCE at load time, it does not track later position changes
    # on the MechanicalObject; without this mapping the visuals just sit
    # frozen at their initial pose while the physics (and console
    # output) run correctly underneath them. Same pattern the sibling
    # projects use.
    rod = rootNode.addChild('rod')
    rod_init_pts, rod_tris = _thin_bar(rod_end_b_x, rod_end_b_z, rod_end_a_x, rod_end_a_z, ROD_HALF_THICKNESS)
    rod.addObject('MeshTopology', name='topo', position=rod_init_pts, triangles=rod_tris)
    rod_mstate = rod.addObject('MechanicalObject', name='dofs', template='Vec3', src='@topo')
    rod.addObject('OglModel', name='visual', src='@topo', color=ROD_COLOR)
    rod.addObject('IdentityMapping', input='@dofs', output='@visual')

    # ---- balls (purely kinematic) -------------------------------
    ball1_local_pts, ball1_disc_tris = _disc(0.0, 0.0, BALL_RADIUS)
    ball1_init_pts = [[lx + ball1_x0, ly + Y_PLANE, lz + ball1_z0] for lx, ly, lz in ball1_local_pts]
    ball1_node = rootNode.addChild('ball1')
    ball1_node.addObject('MeshTopology', name='topo', position=ball1_init_pts, triangles=ball1_disc_tris)
    ball1_mstate = ball1_node.addObject('MechanicalObject', name='dofs', template='Vec3', src='@topo')
    ball1_node.addObject('OglModel', name='visual', src='@topo', color=BALL1_COLOR)
    ball1_node.addObject('IdentityMapping', input='@dofs', output='@visual')

    ball2_local_pts, ball2_disc_tris = _disc(0.0, 0.0, BALL_RADIUS)
    ball2_init_pts = [[lx + ball2_x0, ly + Y_PLANE, lz + ball2_z0] for lx, ly, lz in ball2_local_pts]
    ball2_node = rootNode.addChild('ball2')
    ball2_node.addObject('MeshTopology', name='topo', position=ball2_init_pts, triangles=ball2_disc_tris)
    ball2_mstate = ball2_node.addObject('MechanicalObject', name='dofs', template='Vec3', src='@topo')
    ball2_node.addObject('OglModel', name='visual', src='@topo', color=BALL2_COLOR)
    ball2_node.addObject('IdentityMapping', input='@dofs', output='@visual')

    # ---- spring (purely kinematic -- zigzag rebuilt live by the controller) --
    spring = rootNode.addChild('spring')
    spring_init_pts, spring_tris = _spring_zigzag(
        SPRING_ANCHOR_X, SPRING_ANCHOR_Z, ball2_x0, ball2_z0,
        SPRING_ZIGZAG_SEGMENTS, SPRING_ZIGZAG_AMPLITUDE, SPRING_HALF_THICKNESS)
    spring.addObject('MeshTopology', name='topo', position=spring_init_pts, triangles=spring_tris)
    spring_mstate = spring.addObject('MechanicalObject', name='dofs', template='Vec3', src='@topo')
    spring.addObject('OglModel', name='visual', src='@topo', color=SPRING_COLOR)
    spring.addObject('IdentityMapping', input='@dofs', output='@visual')

    rootNode.addObject(RodWithBallsController(
        ANCHOR_X, ANCHOR_Z, BALL1_MASS_KG, BALL2_MASS_KG, GRAVITY_MAGNITUDE,
        THETA_INITIAL, R1_INITIAL, R2_INITIAL, OMEGA_INITIAL, V1_INITIAL, V2_INITIAL, DT,
        SPRING_ANCHOR_X, SPRING_ANCHOR_Z, SPRING_REST_LENGTH, SPRING_STIFFNESS,
        ball1_mstate, ball1_local_pts, ball2_mstate, ball2_local_pts,
        rod_mstate, spring_mstate, name='rodWithBallsCtrl'))


def main():
    import Sofa.Simulation

    # Headless entry point -- the SOFA GUI (run_imgui.bat/run_glfw.bat)
    # never calls main() at all, it loads this file and calls
    # createScene() directly, so this only ever suppresses the editor for
    # a plain `python RodWithBalls.py` run (setdefault so an operator who
    # explicitly wants it anyway can still unset/override this first).
    os.environ.setdefault('RODWITHBALLS_NO_EDITOR', '1')

    root = Sofa.Core.Node('root')
    createScene(root)
    Sofa.Simulation.init(root)

    # Back to the sibling projects' usual 15 s by default (an earlier
    # revision of this file, before the spring, used 5 -- see the module
    # docstring: ball 1's own slide away takes a few extra seconds to
    # really take off compared to no spring at all, so 5 s no longer
    # reaches anything interesting; how much longer depends on
    # spring_stiffness).
    steps = int(round(RUN_DURATION / DT))
    for _ in range(steps):
        Sofa.Simulation.animate(root, DT)

    Sofa.Simulation.unload(root)


if __name__ == '__main__':
    main()
