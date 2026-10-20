"""Soft cantilever beam (SoftBeam.py), run through sofaweb (see scenes/_base).
The Controls panel acts on the running beam."""

from sofaweb.config import SceneConfig


def _ctrl(root):
    return root.getObject("tipLoadCtrl")


def _rest_tip(c):
    """The free end's centre at rest (once per build)."""
    if not hasattr(c, "_sofaweb_rest_tip"):
        rest = c.dofs.rest_position.value
        n = len(c.tip_nodes)
        c._sofaweb_rest_tip = [sum(float(rest[i][k]) for i in c.tip_nodes) / n for k in range(3)]
    return c._sofaweb_rest_tip


def probes(root, module):
    c = _ctrl(root)
    tip, rest = c.tip_centre(), _rest_tip(c)
    fx, fy, fz = c.applied
    return {
        "dz": (tip[2] - rest[2]) * 1000.0,
        "dy": (tip[1] - rest[1]) * 1000.0,
        "fz": fz,
        "fy": fy,
    }


def _set_vertical(root, value):
    _ctrl(root).verticalForce.value = float(value)


def _set_pushing(root, on):
    _ctrl(root).pushing.value = bool(on)


def _flick(root):
    _ctrl(root).flick()


def _set_gravity(root, on):
    root.gravity.value = [0.0, 0.0, -9.81 if on else 0.0]


CONFIG = SceneConfig(
    title="Soft cantilever beam",
    script="original/SoftBeam.py",
    params_file="original/params.json",
    editor_file="original/params_editor.py",
    param_limits={
        "length": (10.0, 500.0),
        "width": (2.0, 200.0),
        "height": (2.0, 200.0),
        "element_size": (1.0, 50.0),
        "young_modulus": (1e4, 1e9),
        "poisson_ratio": (0.0, 0.49),
        "density": (10.0, 20000.0),
        "rayleigh_stiffness": (0.0, 1.0),
        "rayleigh_mass": (0.0, 100.0),
        "side_force": (0.0, 10.0),
        "flick_force": (0.0, 10.0),
        "flick_duration": (0.001, 1.0),
        "arrow_scale": (1.0, 2000.0),
        "dt": (0.0005, 0.02),
    },
    sections=[
        ("Geometry (mm)", ["length", "width", "height", "element_size"]),
        ("Material", ["young_modulus", "poisson_ratio", "density", "rayleigh_stiffness", "rayleigh_mass"]),
        ("Push and flick", ["side_force", "flick_force", "flick_duration", "arrow_scale"]),
        ("Simulation", ["dt"]),
    ],
    # Per viewer, in the browser: hide the beam's surface (or make it see-through)
    # to look at the FEM elements (Display > Overlays).
    display=[
        {"key": "beam_alpha", "label": "Beam surface opacity", "type": "opacity", "models": r"(^|/)beam/visu/visual$"},
        {"key": "clamp_alpha", "label": "Clamp block opacity", "type": "opacity", "models": r"^/clampBlock/visual$"},
        {"key": "arrow_alpha", "label": "Force arrow opacity", "type": "opacity", "models": r"^/forceArrow/visual$"},
    ],
    controls=[
        {"key": "tip_force", "label": "Tip force (up +)", "type": "slider", "min": -0.5, "max": 0.5, "step": 0.005,
         "default": 0.0, "unit": "N", "digits": 3, "apply": _set_vertical,
         "description": "A steady vertical force on the free end: drag it and watch the beam follow"},
        {"key": "push", "label": "Hold to push sideways", "type": "hold", "apply": _set_pushing,
         "description": "While held down, the tip is pushed sideways, towards you (Push force in the parameters)"},
        {"key": "flick", "label": "Flick the tip", "type": "button", "apply": _flick,
         "description": "A short upward kick (Flick force for Flick duration): the beam rings, then settles"},
        {"key": "gravity", "label": "Gravity", "type": "toggle", "default": True, "apply": _set_gravity},
    ],
    probes=probes,
    charts=[
        {
            "title": "Tip displacement",
            "unit": "mm",
            "series": [
                {"key": "dz", "label": "vertical", "color": "#e8933c"},
                {"key": "dy", "label": "sideways", "color": "#2a78d6"},
            ],
        },
        {
            "title": "Force on the tip",
            "unit": "N",
            "series": [
                {"key": "fz", "label": "vertical", "color": "#e8933c"},
                {"key": "fy", "label": "sideways", "color": "#2a78d6"},
            ],
        },
    ],
    readouts=[
        {"key": "dz", "label": "tip, vertical", "digits": 2, "unit": "mm"},
        {"key": "dy", "label": "tip, sideways", "digits": 2, "unit": "mm"},
        {"key": "fz", "label": "force, vertical", "digits": 3, "unit": "N"},
        {"key": "fy", "label": "force, sideways", "digits": 3, "unit": "N"},
    ],
    # Runs until paused (or the simulation is released).
    stop_at=None,
    about=[
        "A soft silicone beam (100 x 20 x 10 mm, E = 1 MPa) clamped at its left end, simulated with corotational "
        "linear tetrahedra under gravity. Use the Controls panel while it runs: the slider sets a steady vertical "
        "force on the free end, the push button pushes the tip sideways for as long as you hold it, and Flick "
        "gives it a short upward kick. The red arrow shows the force on the tip.",
        "The beam swings at its natural frequency (about 6 Hz with the defaults) and settles, damped by the "
        "Rayleigh damping and by the implicit time integration itself (a smaller time step damps less, but "
        "costs more). It is four times stiffer sideways than vertically (it's twice as wide as it is high, and "
        "bending stiffness goes with the thickness cubed), so the same force bends it less sideways.",
        "Beam theory gives a static tip deflection of F L^3 / (3 E I) for a tip force F (I = width x height^3 / "
        "12). Linear tetrahedra are too stiff in bending with few of them through the height: with the default "
        "5 mm elements (two layers) the beam bends about 56 % as far as beam theory says, with 2.5 mm about "
        "76 % -- but then it runs far slower than real time.",
    ],
    origin="Scene script: scenes/soft-beam/original/SoftBeam.py",
)
