"""SOFA scene: a 2D double pendulum (X-Z plane, Z up, gravity -Z).

Ported from verilogscripts/SOFA/DoublePendulum/DoublePendulum.py for the
web framework. The physics is unchanged: as in the original, the state is
the two rod angles theta1, theta2 (each measured from ITS OWN downward
vertical -- rod 1 from the fixed anchor, rod 2 from mass 1) and their rates,
integrated with classical RK4 by a Sofa.Core.Controller in
onAnimateEndEvent. Mass positions are an exact trigonometric function of
the angles, so the rods can never stretch.

What changed for running headless in a container: no OglModel / camera /
lights (the browser draws the scene with three.js instead), no params.json
editor or matplotlib live-plot subprocess (replaced by the page's own
parameter panel and charts). The masses' positions are still written into
MechanicalObjects every step, so the SOFA scene graph carries the same
state it did in the original.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable

import Sofa
import Sofa.Core

PHYSICS_DT = 0.001
BROADCAST_DT = 0.02
STEPS_PER_BROADCAST = round(BROADCAST_DT / PHYSICS_DT)

# Full-resolution CSV log is kept for this many seconds of simulated time
# (the pendulum never settles, so unlike the cube scene it can't just log
# "since the last reset" without growing forever).
LOG_SECONDS = 120.0

ANCHOR_X = 0.0

# Defaults from the original params.json. All of these are initial
# conditions or geometry, so -- as in the original, where they only took
# effect on a scene reload -- changing them restarts the swing.
DEFAULT_PARAMS: dict[str, float] = {
    "l1": 0.5,                   # rod 1 length [m]
    "m1_kg": 1.0,                # mass 1 [kg]
    "l2": 0.4,                   # rod 2 length [m]
    "m2_kg": 0.5,                # mass 2 [kg]
    "anchor_z": 1.0,             # anchor height above the floor [m]
    "gravity": 9.81,             # [m/s^2]
    "theta1_initial_deg": 90.0,  # rod 1 from the downward vertical, + towards +X
    "omega1_initial": 0.0,       # [rad/s]
    "theta2_initial_deg": 180.0, # rod 2, same convention, measured at mass 1
    "omega2_initial": 0.0,       # [rad/s]
}

# (min, max) accepted by set_params -- keeps the anchor above the floor-ish
# and the masses/lengths strictly positive (the EOM divide by them).
PARAM_LIMITS: dict[str, tuple[float, float]] = {
    "l1": (0.05, 1.0),
    "m1_kg": (0.05, 10.0),
    "l2": (0.05, 1.0),
    "m2_kg": (0.05, 10.0),
    "anchor_z": (0.2, 2.5),
    "gravity": (0.0, 30.0),
    "theta1_initial_deg": (-180.0, 180.0),
    "omega1_initial": (-20.0, 20.0),
    "theta2_initial_deg": (-180.0, 180.0),
    "omega2_initial": (-20.0, 20.0),
}

LOG_COLUMNS = (
    "t", "theta1", "theta2", "omega1", "omega2",
    "x1", "z1", "x2", "z2", "PE1", "KE1", "PE2", "KE2", "E",
)


def double_pendulum_accel(theta1, theta2, omega1, omega2, L1, L2, m1, m2, g):
    """Angular accelerations from the Euler-Lagrange equations (two point
    masses on massless rigid rods), solved by Cramer's rule. Unchanged from
    the original scene."""
    delta = theta1 - theta2
    cos_d = math.cos(delta)
    sin_d = math.sin(delta)

    A = (m1 + m2) * L1
    B = m2 * L2 * cos_d
    C = -m2 * L2 * omega2 * omega2 * sin_d - (m1 + m2) * g * math.sin(theta1)

    D = L1 * cos_d
    E = L2
    F = L1 * omega1 * omega1 * sin_d - g * math.sin(theta2)

    det = A * E - B * D  # = L1*L2*(m1 + m2*sin_d^2) -- never zero for m1, L1, L2 > 0
    alpha1 = (C * E - B * F) / det
    alpha2 = (A * F - C * D) / det
    return alpha1, alpha2


def _wrap_deg(radians: float) -> float:
    """(-180, 180] for display -- the physics state itself is never wrapped."""
    return ((math.degrees(radians) + 180.0) % 360.0) - 180.0


class DoublePendulumController(Sofa.Core.Controller):
    """Owns the pendulum's physics: RK4 on (theta1, theta2, omega1, omega2)
    every animation step, then writes both masses' positions into their
    MechanicalObjects."""

    def __init__(self, params: dict[str, float], mass1_mstate, mass2_mstate, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.mass1_mstate = mass1_mstate
        self.mass2_mstate = mass2_mstate
        self.configure(params)

    def configure(self, params: dict[str, float]) -> None:
        self.L1 = params["l1"]
        self.L2 = params["l2"]
        self.m1 = params["m1_kg"]
        self.m2 = params["m2_kg"]
        self.g = params["gravity"]
        self.anchor_z = params["anchor_z"]
        self.theta1 = math.radians(params["theta1_initial_deg"])
        self.theta2 = math.radians(params["theta2_initial_deg"])
        self.omega1 = params["omega1_initial"]
        self.omega2 = params["omega2_initial"]
        self.sim_time = 0.0
        self._write_positions()

    def positions(self) -> tuple[float, float, float, float]:
        x1 = ANCHOR_X + self.L1 * math.sin(self.theta1)
        z1 = self.anchor_z - self.L1 * math.cos(self.theta1)
        x2 = x1 + self.L2 * math.sin(self.theta2)
        z2 = z1 - self.L2 * math.cos(self.theta2)
        return x1, z1, x2, z2

    def energies(self) -> tuple[float, float, float, float]:
        """(PE1, KE1, PE2, KE2), potential measured from the floor (z=0)."""
        _, z1, _, z2 = self.positions()
        v1_sq = (self.L1 * self.omega1) ** 2
        v2_sq = (
            (self.L1 * self.omega1) ** 2
            + (self.L2 * self.omega2) ** 2
            + 2 * self.L1 * self.L2 * self.omega1 * self.omega2 * math.cos(self.theta1 - self.theta2)
        )
        return (
            self.m1 * self.g * z1,
            0.5 * self.m1 * v1_sq,
            self.m2 * self.g * z2,
            0.5 * self.m2 * v2_sq,
        )

    def _write_positions(self) -> None:
        x1, z1, x2, z2 = self.positions()
        self.mass1_mstate.position.value = [[x1, 0.0, z1]]
        self.mass2_mstate.position.value = [[x2, 0.0, z2]]

    def _rk4_step(self, dt: float) -> None:
        def derivs(theta1, theta2, omega1, omega2):
            alpha1, alpha2 = double_pendulum_accel(
                theta1, theta2, omega1, omega2, self.L1, self.L2, self.m1, self.m2, self.g
            )
            return omega1, omega2, alpha1, alpha2

        s0 = (self.theta1, self.theta2, self.omega1, self.omega2)
        k1 = derivs(*s0)
        k2 = derivs(*(s0[i] + 0.5 * dt * k1[i] for i in range(4)))
        k3 = derivs(*(s0[i] + 0.5 * dt * k2[i] for i in range(4)))
        k4 = derivs(*(s0[i] + dt * k3[i] for i in range(4)))

        self.theta1 += dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.theta2 += dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.omega1 += dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.omega2 += dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

    def onAnimateEndEvent(self, event) -> None:
        self.sim_time += PHYSICS_DT
        self._rk4_step(PHYSICS_DT)
        self._write_positions()


def build_scene(root: Sofa.Core.Node, params: dict[str, float] | None = None) -> DoublePendulumController:
    params = dict(DEFAULT_PARAMS if params is None else params)
    root.addObject(
        "RequiredPlugin",
        pluginName="Sofa.Component.AnimationLoop Sofa.Component.StateContainer",
    )
    # As in the original: set for scene-convention consistency only -- no
    # SOFA Mass/ForceField consumes it, the controller integrates gravity
    # itself.
    root.gravity = [0.0, 0.0, -params["gravity"]]
    root.dt = PHYSICS_DT
    root.addObject("DefaultAnimationLoop")

    mass1 = root.addChild("mass1")
    mass1_mstate = mass1.addObject("MechanicalObject", name="dofs", template="Vec3d", position=[[0.0, 0.0, 0.0]])
    mass2 = root.addChild("mass2")
    mass2_mstate = mass2.addObject("MechanicalObject", name="dofs", template="Vec3d", position=[[0.0, 0.0, 0.0]])

    return root.addObject(DoublePendulumController(params, mass1_mstate, mass2_mstate, name="doublePendulumCtrl"))


class SimulationRunner:
    """Steps the SOFA scene in real time in a background thread and
    broadcasts a throttled frame every BROADCAST_DT."""

    def __init__(self, on_frame: Callable[[dict], None]) -> None:
        self._on_frame = on_frame
        self._params = dict(DEFAULT_PARAMS)
        self._root = Sofa.Core.Node("root")
        self._ctrl = build_scene(self._root, self._params)
        Sofa.Simulation.init(self._root)
        self._initial_energy = sum(self._ctrl.energies())
        self._paused = False
        # Guards _log (and the scene) against get_log_csv copying it from a
        # request thread while the sim thread appends.
        self._lock = threading.Lock()
        self._log: deque[tuple[float, ...]] = deque(maxlen=int(LOG_SECONDS / PHYSICS_DT))
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

    def get_params(self) -> dict[str, float]:
        return dict(self._params)

    def set_params(self, **changes: float) -> dict[str, float]:
        """Validates, stores and applies new parameters. Every parameter is
        geometry or an initial condition, so applying restarts the swing."""
        for key, value in changes.items():
            if key not in PARAM_LIMITS:
                continue
            lo, hi = PARAM_LIMITS[key]
            self._params[key] = min(hi, max(lo, float(value)))
        self.reset()
        return self.get_params()

    def reset_params(self) -> dict[str, float]:
        self._params = dict(DEFAULT_PARAMS)
        self.reset()
        return self.get_params()

    def get_info(self) -> dict:
        return {
            "physicsDt": PHYSICS_DT,
            "broadcastDt": BROADCAST_DT,
            "stepsPerBroadcast": STEPS_PER_BROADCAST,
            "integrator": "classical 4th-order Runge-Kutta (RK4) on (theta1, theta2, omega1, omega2)",
            "logSeconds": LOG_SECONDS,
            "params": self.get_params(),
            "paramLimits": PARAM_LIMITS,
        }

    def _sample_row(self) -> tuple[float, ...]:
        c = self._ctrl
        x1, z1, x2, z2 = c.positions()
        pe1, ke1, pe2, ke2 = c.energies()
        return (c.sim_time, c.theta1, c.theta2, c.omega1, c.omega2, x1, z1, x2, z2, pe1, ke1, pe2, ke2, pe1 + ke1 + pe2 + ke2)

    def _broadcast_frame(self, reset: bool) -> None:
        row = self._sample_row()
        (t, th1, th2, om1, om2, x1, z1, x2, z2, pe1, ke1, pe2, ke2, e) = row
        drift = (e - self._initial_energy) / abs(self._initial_energy) if self._initial_energy else 0.0
        self._on_frame(
            {
                "t": round(t, 4),
                "theta1": _wrap_deg(th1),
                "theta2": _wrap_deg(th2),
                "omega1": om1,
                "omega2": om2,
                "mass1": [x1, z1],
                "mass2": [x2, z2],
                "anchor": [ANCHOR_X, self._ctrl.anchor_z],
                "l1": self._ctrl.L1,
                "l2": self._ctrl.L2,
                "m1": self._ctrl.m1,
                "m2": self._ctrl.m2,
                "PE1": pe1, "KE1": ke1, "PE2": pe2, "KE2": ke2,
                "E": e,
                "energyDrift": drift,
                "reset": reset,
                "paused": self._paused,
            }
        )

    def get_log_csv(self) -> str:
        with self._lock:
            rows = list(self._log)
        lines = [",".join(LOG_COLUMNS)]
        lines.extend(",".join(f"{value:.6f}" for value in row) for row in rows)
        return "\n".join(lines) + "\n"

    def _run(self) -> None:
        # First frame, so a browser connecting before the first tick (or
        # while paused) sees the starting pose.
        self._log.append(self._sample_row())
        next_tick = time.monotonic()
        pause_announced = False
        while not self._stop_requested.is_set():
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                with self._lock:
                    Sofa.Simulation.reset(self._root)
                    self._root.gravity = [0.0, 0.0, -self._params["gravity"]]
                    self._ctrl.configure(self._params)
                    self._initial_energy = sum(self._ctrl.energies())
                    self._log.clear()
                    self._log.append(self._sample_row())
                self._broadcast_frame(reset=True)

            if self._paused:
                # One frame on entering pause, so every browser learns the
                # new state (no frames are sent while paused).
                if not pause_announced:
                    self._broadcast_frame(reset=False)
                    pause_announced = True
                time.sleep(0.05)
                next_tick = time.monotonic()
                continue
            pause_announced = False

            with self._lock:
                for _ in range(STEPS_PER_BROADCAST):
                    Sofa.Simulation.animate(self._root, PHYSICS_DT)
                    self._log.append(self._sample_row())
            self._broadcast_frame(reset=False)

            # Fixed-rate schedule (rather than "sleep BROADCAST_DT minus this
            # tick's work") so simulated time tracks wall time without drift.
            next_tick += BROADCAST_DT
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # fell behind; don't try to catch up in a burst
