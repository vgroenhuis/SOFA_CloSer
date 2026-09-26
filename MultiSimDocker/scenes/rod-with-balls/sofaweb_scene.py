"""RodWithBalls, run unmodified through sofaweb (see scenes/_base)."""

import math

from sofaweb.config import SceneConfig


def probes(root, module):
    c = root.getObject("rodWithBallsCtrl")
    x1, z1 = c._ball_position(c.r1)
    x2, z2 = c._ball_position(c.r2)
    # Same energy bookkeeping as the controller's own console log.
    ke = 0.5 * c.m1 * (c.v1 ** 2 + (c.r1 * c.omega) ** 2) + 0.5 * c.m2 * (c.v2 ** 2 + (c.r2 * c.omega) ** 2)
    pe = c.m1 * c.g * z1 + c.m2 * c.g * z2
    spring = c._spring_energy(x2, z2)
    total = pe + ke + spring
    if not hasattr(c, "_sofaweb_e0"):
        c._sofaweb_e0 = total
    drift = (total - c._sofaweb_e0) / abs(c._sofaweb_e0) * 100 if c._sofaweb_e0 else 0.0
    return {
        "theta": math.degrees(c.theta),
        "r1": c.r1,
        "r2": c.r2,
        "PE": pe,
        "KE": ke,
        "spring": spring,
        "E": total,
        "drift": drift,
    }


# Ball 1 has no spring, so it eventually slides off along the infinite rod
# without bound (correct physics, see the original's docstring). Once it's
# well out of view there's nothing left to watch, so the run ends there.
OUT_OF_VIEW_M = 8.0


def end_condition(root, module):
    c = root.getObject("rodWithBallsCtrl")
    if max(abs(c.r1), abs(c.r2)) > OUT_OF_VIEW_M:
        return f"a ball slid more than {OUT_OF_VIEW_M:g} m along the rod"
    return None


CONFIG = SceneConfig(
    title="Rod with sliding balls",
    script="original/RodWithBalls.py",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    env={"RODWITHBALLS_NO_EDITOR": "1"},
    forced_params={"console_log_enabled": False},
    hidden_params={"theme"},
    labels={"run_duration": "Run duration (then restart)"},
    param_limits={
        "anchor_x": (-3.0, 3.0),
        "anchor_z": (-1.0, 4.0),
        "theta_initial_deg": (-180.0, 180.0),
        "omega_initial": (-20.0, 20.0),
        "ball1_mass_kg": (0.01, 100.0),
        "ball2_mass_kg": (0.01, 100.0),
        "ball1_initial_x": (-4.0, 4.0),
        "ball1_initial_z": (-2.0, 5.0),
        "ball2_initial_x": (-4.0, 4.0),
        "ball2_initial_z": (-2.0, 5.0),
        "rod_visual_half_length": (0.1, 10.0),
        "spring_anchor_x": (-4.0, 4.0),
        "spring_anchor_z": (-2.0, 5.0),
        "spring_rest_length": (0.0, 5.0),
        "spring_stiffness": (0.0, 100000.0),
        "gravity": (0.0, 30.0),
        "dt": (0.0001, 0.005),
        "run_duration": (1.0, 600.0),
    },
    sections=[
        ("Rod", ["anchor_x", "anchor_z", "theta_initial_deg", "omega_initial", "rod_visual_half_length"]),
        ("Balls", ["ball1_mass_kg", "ball1_initial_x", "ball1_initial_z", "ball2_mass_kg", "ball2_initial_x", "ball2_initial_z"]),
        ("Spring (ball 2 to a fixed point)", ["spring_anchor_x", "spring_anchor_z", "spring_rest_length", "spring_stiffness"]),
        ("Simulation", ["gravity", "dt", "run_duration"]),
        ("Display", ["show_pivot_marker", "show_spring_anchor_marker"]),
    ],
    probes=probes,
    charts=[
        {"title": "Rod angle", "unit": "deg", "series": [{"key": "theta", "label": "theta", "color": "#8f97a3"}]},
        {
            "title": "Ball distance along the rod",
            "unit": "m",
            "series": [
                {"key": "r1", "label": "ball 1", "color": "#8c26bf"},
                {"key": "r2", "label": "ball 2", "color": "#d96626"},
            ],
        },
        {
            "title": "Energy",
            "unit": "J",
            "series": [
                {"key": "PE", "label": "potential", "color": "#4a9be8"},
                {"key": "KE", "label": "kinetic", "color": "#e8933c"},
                {"key": "spring", "label": "spring", "color": "#c77dff"},
                {"key": "E", "label": "total", "color": "#6bcf7a"},
            ],
        },
    ],
    readouts=[
        {"key": "theta", "label": "rod angle", "digits": 1, "unit": "deg"},
        {"key": "r1", "label": "ball 1 (r1)", "digits": 3, "unit": "m"},
        {"key": "r2", "label": "ball 2 (r2)", "digits": 3, "unit": "m"},
        {"key": "E", "label": "total energy", "digits": 3, "unit": "J"},
        {"key": "drift", "label": "energy drift", "digits": 4, "unit": "%"},
    ],
    stop_at=lambda p: p["run_duration"],
    end_condition=end_condition,
    auto_restart=True,
    view="2d",
    about=[
        "A massless, infinitely long rigid rod, hinged at a fixed point, carries two beads that slide along it "
        "without friction. Ball 2 is also tied to a separate fixed point by a spring. The state is the rod angle "
        "and each ball's distance from the hinge, integrated with Runge-Kutta 4 by a SOFA controller.",
        "Ball 1 has no spring of its own: as the rod swings down it slides away without bound. That's correct "
        "physics for a frictionless infinite rod. The run restarts once a ball is far out of view, or after "
        "the run duration.",
    ],
    origin="Original scene: verilogscripts/SOFA/RodWithBalls/RodWithBalls.py.",
)
