"""The physics engines a scene can run on (SceneConfig.engine).

The runner talks to the simulation only through an engine: build the scene
from the script's module, step it, read its time step, background, camera
and lights, and unload it. A built scene is a "root" (SOFA's root node, or a
MuJoCo scene object) and a "capture" (what the browser draws: `visuals`, a
list of records with `obj`, `color` and `line_width`, where `obj` has
`position.value` (n x 3), `triangles.value`, `quads.value`, `edges.value`
and `getPathName()`; plus the lights). The scene's own callbacks (probes,
controls, gauge, ...) receive the root and the module, and know which kind it
is.
"""

from __future__ import annotations

from typing import Any, Optional


class SofaEngine:
    """SOFA: createScene(root) on a fresh root node, with the GL components
    swapped out (capture.Capture)."""

    name = "sofa"

    def __init__(self) -> None:
        import Sofa.Core
        import Sofa.Simulation

        self._core, self._simulation = Sofa.Core, Sofa.Simulation

    def build(self, module) -> tuple[Any, Any]:
        from .capture import Capture

        root = self._core.Node("root")
        with Capture() as capture:
            module.createScene(root)
        self._simulation.init(root)
        return root, capture

    def step(self, root, dt: float) -> None:
        self._simulation.animate(root, dt)

    @staticmethod
    def dt(root) -> float:
        return float(root.dt.value)

    def unload(self, root) -> None:
        self._simulation.unload(root)

    @staticmethod
    def background(root) -> list[float]:
        for obj in root.objects:
            if obj.getClassName() == "BackgroundSetting":
                return [float(c) for c in obj.color.value]
        return [0.0, 0.0, 0.0, 1.0]

    @staticmethod
    def camera(root) -> Optional[dict]:
        for obj in root.objects:
            if obj.getClassName() in ("InteractiveCamera", "Camera", "RecordedCamera"):
                return {
                    "position": [float(v) for v in obj.position.value],
                    "orientation": [float(v) for v in obj.orientation.value],
                    "fov": float(obj.fieldOfView.value),
                    "distance": float(obj.distance.value),
                    "projection": str(obj.projectionType.value),
                }
        return None

    @staticmethod
    def ambient(capture) -> list[float]:
        from .capture import ambient_color

        return ambient_color(capture)

    @staticmethod
    def lights(capture) -> list[dict]:
        from .capture import light_list

        return light_list(capture)


class MujocoEngine:
    """MuJoCo: the script's createScene() returns a scene object (see
    sofaweb.mjscene.MjScene) that is both the root and the capture."""

    name = "mujoco"

    def build(self, module) -> tuple[Any, Any]:
        scene = module.createScene()
        return scene, scene

    @staticmethod
    def step(root, dt: float) -> None:
        root.step(dt)

    @staticmethod
    def dt(root) -> float:
        return float(root.dt.value)

    @staticmethod
    def unload(root) -> None:
        pass

    @staticmethod
    def background(root) -> list[float]:
        return list(root.background)

    @staticmethod
    def camera(root) -> Optional[dict]:
        return root.camera

    @staticmethod
    def ambient(capture) -> list[float]:
        return list(capture.ambient)

    @staticmethod
    def lights(capture) -> list[dict]:
        return [dict(light) for light in capture.lights]


ENGINES = {"sofa": SofaEngine, "mujoco": MujocoEngine}


def make_engine(name: str):
    try:
        return ENGINES[name]()
    except KeyError:
        raise ValueError(f"unknown engine {name!r} (one of {', '.join(ENGINES)})") from None
