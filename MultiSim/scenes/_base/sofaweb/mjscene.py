"""A MuJoCo scene for sofaweb (SceneConfig(engine="mujoco")).

A MuJoCo scene script's createScene() builds a mujoco.MjModel / MjData and
returns an MjScene: the model and data, what the browser draws (meshes
attached to bodies, given in the body's frame at rest), the lights, the
camera and the background, and controllers that run before every physics
step (the scene's own Python: targets, pressure ramps, ...).

One sofaweb step (`dt`, the step the browser's time and charts follow) is
several MuJoCo steps (model.opt.timestep): contacts and stiff joints want a
much smaller step than the 20-40 ms between frames.

Visual meshes look like SOFA visual models to the runner: an object with
`position.value` (world positions, n x 3), `triangles.value`, `quads.value`,
`edges.value` and getPathName(), so the browser, the recordings and the
scene configs (display regexes, gauge "follow") treat them alike.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import mujoco
import numpy as np


class Value:
    """A SOFA-Data-like holder: `.value`."""

    def __init__(self, value: Any) -> None:
        self.value = value


class MjVisual:
    """A mesh drawn in the browser, each vertex carried by a body (its
    position in that body's frame)."""

    def __init__(self, scene: "MjScene", path: str, points, triangles, bodies) -> None:
        self._scene = scene
        self._path = path
        self.local = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        body = np.asarray(bodies, dtype=np.int64).reshape(-1)
        self.body = np.full(len(self.local), int(body[0])) if body.size == 1 else body
        self.triangles = Value(np.asarray(triangles, dtype=np.int64).reshape(-1, 3))
        self.quads = Value(np.zeros((0, 4), dtype=np.int64))
        self.edges = Value(np.zeros((0, 2), dtype=np.int64))
        # The last positions and the poses of its bodies they came from: a mesh on bodies that haven't moved (the
        # world, a sleeping item) isn't computed again.
        self._bodies = np.unique(self.body)
        self._pose: Optional[np.ndarray] = None
        self._cached: Optional[Value] = None

    @property
    def position(self) -> Value:
        data = self._scene.data
        pose = np.concatenate([data.xpos[self._bodies].ravel(), data.xmat[self._bodies].ravel()])
        if self._cached is not None and np.array_equal(pose, self._pose):
            return self._cached
        pos = data.xpos[self.body]
        rot = data.xmat[self.body].reshape(-1, 3, 3)
        self._pose, self._cached = pose, Value(pos + np.einsum("nij,nj->ni", rot, self.local))
        return self._cached

    def getPathName(self) -> str:
        return self._path


class VisualRecord:
    def __init__(self, obj: MjVisual, color: list[float], line_width: float = 1.0) -> None:
        self.obj, self.color, self.line_width = obj, color, line_width


class MjScene:
    """A built MuJoCo scene: see the module docstring. `objects` holds the
    scene's named Python objects (controllers, ...), for its sofaweb config's
    callbacks (getObject(name))."""

    def __init__(self, model: mujoco.MjModel, data: Optional[mujoco.MjData] = None, dt: float = 0.02) -> None:
        self.model = model
        self.data = data if data is not None else mujoco.MjData(model)
        self.dt = Value(float(dt))
        self.visuals: list[VisualRecord] = []
        self.lights: list[dict] = []
        self.ambient: list[float] = [0.3, 0.3, 0.3, 1.0]
        self.background: list[float] = [0.0, 0.0, 0.0, 1.0]
        self.camera: Optional[dict] = None
        self.objects: dict[str, Any] = {}
        self.controllers: list[Callable[["MjScene", float], None]] = []
        mujoco.mj_forward(self.model, self.data)

    # -- building -------------------------------------------------------------

    def body_id(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))

    def add_visual(self, path: str, points, triangles, bodies=0, color=(1.0, 1.0, 1.0, 1.0)) -> MjVisual:
        """A mesh carried by `bodies` (one body id, or one per point; 0 is the world), `points` in the body's frame."""
        visual = MjVisual(self, path, points, triangles, bodies)
        rgba = list(color) + [1.0] * (4 - len(color))
        self.visuals.append(VisualRecord(visual, [float(c) for c in rgba[:4]]))
        return visual

    def add_light(self, kind: str = "DirectionalLight", color=(1.0, 1.0, 1.0, 1.0), direction=None, position=None,
                  name: str = "") -> None:
        light: dict = {"kind": kind, "color": [float(c) for c in color]}
        if direction is not None:
            light["direction"] = [float(v) for v in direction]
        if position is not None:
            light["position"] = [float(v) for v in position]
        if name:
            light["name"] = name
        self.lights.append(light)

    def set_camera(self, position, orientation, distance: float, fov: float = 45.0,
                   projection: str = "Perspective") -> None:
        """As SOFA's InteractiveCamera: orientation a quaternion (x, y, z, w) of a camera looking along its -z."""
        self.camera = {"position": [float(v) for v in position], "orientation": [float(v) for v in orientation],
                       "fov": float(fov), "distance": float(distance), "projection": projection}

    def getObject(self, name: str) -> Any:
        return self.objects.get(name)

    # -- running ------------------------------------------------------------------

    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self, dt: float) -> None:
        h = float(self.model.opt.timestep)
        for _ in range(max(1, int(round(dt / h)))):
            for controller in self.controllers:
                controller(self, h)
            mujoco.mj_step(self.model, self.data)
