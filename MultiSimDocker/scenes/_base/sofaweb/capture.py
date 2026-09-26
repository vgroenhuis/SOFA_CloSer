"""Lets an unmodified SOFA scene script build its scene headlessly.

While a scene's createScene() runs, Sofa.Core.Node.addObject is patched so
that everything needing an OpenGL context is swapped out as it is created:

* OglModel -> VisualModelImpl, the GL-free base class of OglModel. Mappings
  into it (IdentityMapping, BarycentricMapping, ...) and controllers that
  read or write its `position` keep working unchanged. The model's color
  and line settings are recorded so the browser can draw it.
* LightManager / DirectionalLight (and other Ogl* helpers) -> a stub object
  that accepts any Data access; their settings are recorded for the
  browser's own lights.
* RequiredPlugin entries for Sofa.GL.* are dropped.

The camera (InteractiveCamera) and BackgroundSetting are GL-free and stay
real SOFA components; their values are read back after init.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import Sofa.Core

log = logging.getLogger(__name__)

# OglModel arguments that only mean something to the GL renderer.
_GL_ONLY_KWARGS = {
    "lineWidth", "pointSize", "blendEquation", "sfactor", "dfactor", "premultipliedAlpha",
    "alphaBlend", "depthTest", "cullFace", "writeZTransparent", "primitiveType", "isEnabled",
    "enable", "lineSmooth", "pointSmooth",
}
_STUBBED_TYPES = {"LightManager", "DirectionalLight", "SpotLight", "PositionalLight", "OglShadowShader"}


class _StubData:
    def __init__(self, value: Any = None) -> None:
        self.value = value

    def getValueString(self) -> str:
        return str(self.value)


class Stub:
    """Stands in for a GL-only component: any attribute is a Data-like
    object with a settable `.value`, so scene code that tweaks e.g. a
    light's direction every step keeps running."""

    def __init__(self, type_name: str, **data: Any) -> None:
        object.__setattr__(self, "_type", type_name)
        object.__setattr__(self, "_data", {k: _StubData(v) for k, v in data.items()})

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return self._data.setdefault(name, _StubData())

    def getName(self) -> str:
        value = self._data.get("name")
        return value.value if value and value.value else self._type

    def getClassName(self) -> str:
        return self._type

    def getPathName(self) -> str:
        return self.getName()


def _rgba(value: Any, default=(1.0, 1.0, 1.0, 1.0)) -> list[float]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        try:
            value = [float(p) for p in parts]
        except ValueError:
            return list(default)  # named colors aren't used by these scenes
    value = [float(v) for v in value]
    if len(value) == 3:
        value.append(1.0)
    return value[:4] if len(value) >= 4 else list(default)


@dataclass
class VisualRecord:
    obj: Any
    color: list[float]
    line_width: float
    order: int


@dataclass
class LightRecord:
    kind: str
    stub: Stub
    order: int


@dataclass
class Capture:
    visuals: list[VisualRecord] = field(default_factory=list)
    lights: list[LightRecord] = field(default_factory=list)
    ambient: Optional[Stub] = None

    def _add_object(self, original, node, type_or_obj, *args, **kwargs):
        if not isinstance(type_or_obj, str):
            return original(node, type_or_obj, *args, **kwargs)
        type_name = type_or_obj

        if type_name == "OglModel":
            color = _rgba(kwargs.get("color"))
            line_width = float(kwargs.get("lineWidth", 1.0))
            kept = {k: v for k, v in kwargs.items() if k not in _GL_ONLY_KWARGS}
            obj = original(node, "VisualModelImpl", *args, **kept)
            self.visuals.append(VisualRecord(obj, color, line_width, len(self.visuals)))
            return obj

        if type_name in _STUBBED_TYPES or (type_name.startswith("Ogl") and type_name != "OglModel"):
            stub = Stub(type_name, **kwargs)
            if type_name == "LightManager":
                self.ambient = stub
            elif type_name.endswith("Light"):
                self.lights.append(LightRecord(type_name, stub, len(self.lights)))
            return stub

        if type_name == "RequiredPlugin":
            names = kwargs.get("pluginName", kwargs.get("name", ""))
            if isinstance(names, str):
                names = names.split()
            kept_names = [n for n in names if not n.startswith("Sofa.GL")]
            if not kept_names:
                return Stub("RequiredPlugin", **kwargs)
            kwargs = {**kwargs, "pluginName": " ".join(kept_names)}
            return original(node, type_name, *args, **kwargs)

        return original(node, type_name, *args, **kwargs)

    def __enter__(self) -> "Capture":
        original = Sofa.Core.Node.addObject
        self._original = original
        capture = self

        def patched(node, type_or_obj, *args, **kwargs):
            return capture._add_object(original, node, type_or_obj, *args, **kwargs)

        Sofa.Core.Node.addObject = patched
        return self

    def __exit__(self, *exc) -> None:
        Sofa.Core.Node.addObject = self._original


def ambient_color(capture: Capture) -> list[float]:
    if capture.ambient is None:
        return [0.0, 0.0, 0.0, 1.0]
    return _rgba(capture.ambient.ambient.value, default=(0.0, 0.0, 0.0, 1.0))


def light_list(capture: Capture) -> list[dict]:
    lights = []
    for record in capture.lights:
        stub = record.stub
        entry = {"kind": record.kind, "color": _rgba(stub.color.value)}
        if stub.direction.value is not None:
            entry["direction"] = [float(v) for v in stub.direction.value]
        if stub.position.value is not None:
            entry["position"] = [float(v) for v in stub.position.value]
        lights.append(entry)
    return lights
