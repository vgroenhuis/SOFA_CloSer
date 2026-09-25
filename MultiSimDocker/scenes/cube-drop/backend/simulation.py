"""SOFA scene: a rigid cube falling onto a penalty-based floor plane.

No GUI/OpenGL components are used here -- the scene is stepped headlessly
and the cube's transform is streamed to the browser, which does its own
rendering with three.js.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Callable

import Sofa

# The physics integrates at a fine timestep for accuracy, but the browser
# only needs a fraction of those steps -- broadcasting every physics step
# would be both unnecessary and wasteful of bandwidth. STEPS_PER_BROADCAST
# fine steps are integrated back-to-back each "tick," and only the state
# after that batch is sent out and checked for settling.
PHYSICS_DT = 0.001
BROADCAST_DT = 0.02
STEPS_PER_BROADCAST = round(BROADCAST_DT / PHYSICS_DT)

START_POSITION = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0]
CUBE_HALF_SIZE = 0.5
CUBE_SIDE = 2.0 * CUBE_HALF_SIZE
CUBE_MASS = 1.0
GRAVITY = 9.81
# Moment of inertia of a solid cube about a centroidal axis: m*a^2/6. Since
# this is the same on all three axes, it's isotropic -- world-frame and
# body-frame angular velocity give the same kinetic energy, so no rotation
# tracking is needed to compute it below.
CUBE_INERTIA = CUBE_MASS * CUBE_SIDE**2 / 6.0
MAX_ANGULAR_VELOCITY = 3.0  # rad/s, randomized per axis on every drop

# Floor contact: damping is 0 so the vertical bounce itself is elastic
# (energy loss on landing comes only from RAYLEIGH_MASS/friction below, not
# from the contact spring). Stiffness is deliberately soft: EulerImplicitSolver
# is unconditionally stable but *numerically* dissipative for stiff forces at
# a given timestep, so a very stiff floor (e.g. thousands) ate almost all the
# impact energy in implicit-integration damping alone, even with damping=0 --
# softening it (and accepting a bit more visible penetration on impact) is
# what actually lets the cube rebound.
FLOOR_STIFFNESS = 2000.0
FLOOR_DAMPING = 0.0

# Solver (Rayleigh) damping: a velocity-proportional drag applied to the
# whole body at all times, not just on contact -- this is what eventually
# bleeds off the spin and settles the cube. rayleighStiffness in particular
# scales with the *stiffness* it's damping, so even a small coefficient
# becomes a large effective damping force during floor contact (where
# FLOOR_STIFFNESS is large) and can kill the bounce almost entirely; it's
# kept small so the floor stays mostly elastic, and rayleighMass alone (which
# doesn't scale with contact stiffness) does most of the settling.
RAYLEIGH_MASS = 0.15
RAYLEIGH_STIFFNESS = 0.0025

# Contact friction: PlaneForceField only pushes back along the floor normal,
# so a corner resting on the floor can slide/spin forever with nothing to
# resist it. This approximates friction by directly damping the *lateral*
# velocity (horizontal slide + spin about the vertical axis) whenever a
# corner is touching, each step, while leaving the vertical velocity and the
# tipping rotations (which drive the bounce) untouched.
CONTACT_HEIGHT = 0.1
LATERAL_FRICTION_DAMPING = 0.8  # per-step multiplier while in contact

# The tuned starting point for the sliders in the "physics parameters" panel
# -- also what the "Default" button restores.
DEFAULT_PARAMS: dict[str, float] = {
    "floorStiffness": FLOOR_STIFFNESS,
    "floorDamping": FLOOR_DAMPING,
    "rayleighMass": RAYLEIGH_MASS,
    "rayleighStiffness": RAYLEIGH_STIFFNESS,
    "lateralFrictionDamping": LATERAL_FRICTION_DAMPING,
}

SETTLE_LINEAR_VELOCITY = 0.05
SETTLE_ANGULAR_VELOCITY = 0.1
SETTLE_STEPS_REQUIRED = 50  # ~1s of near-zero velocity, checked once per broadcast tick
SETTLE_PAUSE_SECONDS = 1.5

# Column order for both the per-substep download log and the CSV it renders as.
LOG_COLUMNS = (
    "t", "x", "y", "z", "qx", "qy", "qz", "qw",
    "height", "kineticEnergy", "potentialEnergy", "elasticEnergy", "dissipatedEnergy",
)

_REQUIRED_PLUGINS = " ".join(
    [
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Mass",
        "Sofa.Component.Mapping.NonLinear",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.AnimationLoop",
    ]
)


def _cube_corners(half: float) -> list[list[float]]:
    signs = (-1.0, 1.0)
    return [[sx * half, sy * half, sz * half] for sx in signs for sy in signs for sz in signs]


def _random_angular_velocity() -> list[float]:
    """A random spin (zero linear velocity, random angular velocity per axis).

    Almost never lands flat on a face on its own -- typically an edge (or
    corner) touches down first and the PlaneForceField contact + gravity
    settle it from there.
    """
    spin = [random.uniform(-MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY) for _ in range(3)]
    return [0.0, 0.0, 0.0, *spin]


def build_scene(root: Sofa.Core.Node) -> Sofa.Core.Node:
    root.addObject("RequiredPlugin", pluginName=_REQUIRED_PLUGINS)
    root.gravity = [0.0, -GRAVITY, 0.0]
    root.dt = PHYSICS_DT
    root.addObject("DefaultAnimationLoop")

    cube = root.addChild("Cube")
    cube.addObject("EulerImplicitSolver", rayleighStiffness=RAYLEIGH_STIFFNESS, rayleighMass=RAYLEIGH_MASS)
    cube.addObject("CGLinearSolver", iterations=25, tolerance=1e-5, threshold=1e-5)
    rigid_dof = cube.addObject(
        "MechanicalObject",
        name="rigidDOF",
        template="Rigid3d",
        position=[START_POSITION],
        showObject=False,
    )
    # Explicit inertia (rather than the component's identity-matrix default)
    # so the rotational dynamics -- and the kinetic-energy readout below --
    # match a real solid cube of this size and mass.
    cube.addObject(
        "UniformMass",
        vertexMass=[
            CUBE_MASS,
            CUBE_SIDE**3,
            [CUBE_INERTIA, 0.0, 0.0, 0.0, CUBE_INERTIA, 0.0, 0.0, 0.0, CUBE_INERTIA],
        ],
    )

    corners = cube.addChild("Corners")
    corner_dof = corners.addObject(
        "MechanicalObject",
        name="cornerDOF",
        template="Vec3d",
        position=_cube_corners(CUBE_HALF_SIZE),
    )
    corners.addObject(
        "RigidMapping",
        input=rigid_dof.getLinkPath(),
        output=corner_dof.getLinkPath(),
    )
    corners.addObject(
        "PlaneForceField",
        normal=[0.0, 1.0, 0.0],
        d=0.0,
        stiffness=FLOOR_STIFFNESS,
        damping=FLOOR_DAMPING,
        showPlane=False,
    )

    return root


class SimulationRunner:
    """Steps a SOFA scene in a background thread and broadcasts each frame."""

    def __init__(self, on_frame: Callable[[dict], None]) -> None:
        self._on_frame = on_frame
        self._root = Sofa.Core.Node("root")
        build_scene(self._root)
        Sofa.Simulation.init(self._root)
        self._mstate = self._root.Cube.rigidDOF
        self._corners = self._root.Cube.Corners.cornerDOF
        self._euler_solver = self._root.Cube.EulerImplicitSolver
        self._plane_force_field = self._root.Cube.Corners.PlaneForceField
        self._lateral_friction_damping = LATERAL_FRICTION_DAMPING
        self._apply_random_spin()
        self._sim_time = 0.0
        self._initial_energy = sum(self._mechanical_energy())
        self._paused = False
        self._auto_reset_enabled = True
        # Full-resolution (PHYSICS_DT) log since the last reset, for the
        # "download data" button -- the browser itself only ever sees the
        # throttled BROADCAST_DT frames, to keep the websocket cheap.
        self._log: list[tuple[float, ...]] = []

        self._reset_requested = threading.Event()
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_requested.set()
        self._thread.join(timeout=2.0)

    def reset(self) -> None:
        self._reset_requested.set()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def set_auto_reset(self, enabled: bool) -> None:
        self._auto_reset_enabled = enabled

    def get_info(self) -> dict:
        """Static-ish simulation configuration for the "Info" panel."""
        return {
            "physicsDt": PHYSICS_DT,
            "broadcastDt": BROADCAST_DT,
            "stepsPerBroadcast": STEPS_PER_BROADCAST,
            "gravity": GRAVITY,
            "cubeMass": CUBE_MASS,
            "cubeSide": CUBE_SIDE,
            "cubeInertia": CUBE_INERTIA,
            "startHeight": START_POSITION[1],
            "maxAngularVelocity": MAX_ANGULAR_VELOCITY,
            "contactHeight": CONTACT_HEIGHT,
            "settleLinearVelocity": SETTLE_LINEAR_VELOCITY,
            "settleAngularVelocity": SETTLE_ANGULAR_VELOCITY,
            "settleStepsRequired": SETTLE_STEPS_REQUIRED,
            "settlePauseSeconds": SETTLE_PAUSE_SECONDS,
            "params": self.get_params(),
        }

    def get_params(self) -> dict[str, float]:
        return {
            "floorStiffness": float(self._plane_force_field.stiffness.value),
            "floorDamping": float(self._plane_force_field.damping.value),
            "rayleighMass": float(self._euler_solver.rayleighMass.value),
            "rayleighStiffness": float(self._euler_solver.rayleighStiffness.value),
            "lateralFrictionDamping": self._lateral_friction_damping,
        }

    def set_params(self, **params: float) -> None:
        if "floorStiffness" in params:
            self._plane_force_field.stiffness.value = float(params["floorStiffness"])
        if "floorDamping" in params:
            self._plane_force_field.damping.value = float(params["floorDamping"])
        if "rayleighMass" in params:
            self._euler_solver.rayleighMass.value = float(params["rayleighMass"])
        if "rayleighStiffness" in params:
            self._euler_solver.rayleighStiffness.value = float(params["rayleighStiffness"])
        if "lateralFrictionDamping" in params:
            self._lateral_friction_damping = float(params["lateralFrictionDamping"])

    def reset_params(self) -> dict[str, float]:
        self.set_params(**DEFAULT_PARAMS)
        return self.get_params()

    def _apply_random_spin(self) -> None:
        self._mstate.velocity.value = [_random_angular_velocity()]

    def _apply_contact_friction(self) -> None:
        corner_heights = [float(p[1]) for p in self._corners.position.value]
        if min(corner_heights) > CONTACT_HEIGHT:
            return
        damping = self._lateral_friction_damping
        v = [float(c) for c in self._mstate.velocity.value[0]]
        v[0] *= damping  # horizontal slide (x)
        v[2] *= damping  # horizontal slide (z)
        v[4] *= damping  # spin about the vertical (y) axis
        self._mstate.velocity.value = [v]

    def _mechanical_energy(self) -> tuple[float, float]:
        """Returns (kinetic, potential) energy of the cube, in joules."""
        position = self._mstate.position.value[0]
        velocity = self._mstate.velocity.value[0]
        linear_ke = 0.5 * CUBE_MASS * sum(float(c) * float(c) for c in velocity[0:3])
        angular_ke = 0.5 * CUBE_INERTIA * sum(float(c) * float(c) for c in velocity[3:6])
        potential = CUBE_MASS * GRAVITY * float(position[1])
        return linear_ke + angular_ke, potential

    def _elastic_energy(self) -> float:
        """Energy currently stored in the floor's contact spring (PlaneForceField).

        The floor is the plane y=0 with normal (0, 1, 0), so a corner at
        y < 0 is penetrating it by -y; each penetrating corner acts as an
        independent spring of stiffness FLOOR_STIFFNESS.
        """
        energy = 0.0
        for corner in self._corners.position.value:
            penetration = -float(corner[1])
            if penetration > 0.0:
                energy += 0.5 * FLOOR_STIFFNESS * penetration * penetration
        return energy

    def _sample_row(self) -> tuple[float, ...]:
        """One LOG_COLUMNS-shaped row for the current instant."""
        position = [float(v) for v in self._mstate.position.value[0]]
        kinetic, potential = self._mechanical_energy()
        elastic = self._elastic_energy()
        dissipated = max(0.0, self._initial_energy - (kinetic + potential + elastic))
        return (
            self._sim_time,
            position[0], position[1], position[2],
            position[3], position[4], position[5], position[6],
            position[1],
            kinetic, potential, elastic, dissipated,
        )

    def _broadcast_frame(self, reset: bool) -> None:
        row = self._sample_row()
        self._on_frame(
            {
                "t": round(row[0], 4),
                "position": list(row[1:4]),
                "quaternion": list(row[4:8]),
                "height": row[8],
                "kineticEnergy": row[9],
                "potentialEnergy": row[10],
                "elasticEnergy": row[11],
                "dissipatedEnergy": row[12],
                "reset": reset,
                "paused": self._paused,
            }
        )

    def get_log_csv(self) -> str:
        """CSV of every physics substep (PHYSICS_DT resolution) since the last reset."""
        lines = [",".join(LOG_COLUMNS)]
        lines.extend(",".join(f"{value:.6f}" for value in row) for row in self._log)
        return "\n".join(lines) + "\n"

    def _run(self) -> None:
        settled_steps = 0
        pause_announced = False
        while not self._stop_requested.is_set():
            loop_start = time.monotonic()

            if self._reset_requested.is_set():
                Sofa.Simulation.reset(self._root)
                self._apply_random_spin()
                self._sim_time = 0.0
                self._initial_energy = sum(self._mechanical_energy())
                self._reset_requested.clear()
                settled_steps = 0
                self._log = [self._sample_row()]
                # Broadcast immediately so a reset is visible right away even
                # while paused, when the normal per-step broadcast below is
                # skipped.
                self._broadcast_frame(reset=True)

            if self._paused:
                # One frame on entering pause, so every browser learns the
                # new state (no frames are sent while paused).
                if not pause_announced:
                    self._broadcast_frame(reset=False)
                    pause_announced = True
                time.sleep(0.05)
                continue
            pause_announced = False

            for _ in range(STEPS_PER_BROADCAST):
                Sofa.Simulation.animate(self._root, PHYSICS_DT)
                self._sim_time += PHYSICS_DT
                self._log.append(self._sample_row())
            self._apply_contact_friction()

            velocity = [float(v) for v in self._mstate.velocity.value[0]]
            linear_speed = sum(v * v for v in velocity[:3]) ** 0.5
            angular_speed = sum(v * v for v in velocity[3:6]) ** 0.5

            self._broadcast_frame(reset=False)

            if linear_speed < SETTLE_LINEAR_VELOCITY and angular_speed < SETTLE_ANGULAR_VELOCITY:
                settled_steps += 1
            else:
                settled_steps = 0

            if settled_steps >= SETTLE_STEPS_REQUIRED:
                settled_steps = 0
                if self._auto_reset_enabled:
                    time.sleep(SETTLE_PAUSE_SECONDS)
                    self._reset_requested.set()

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, BROADCAST_DT - elapsed))
