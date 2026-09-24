"""Scene catalog: every `scenes/<id>/scene.json` is one launchable scene.

The catalog is re-read on every request, so dropping a new scene folder in
(and building its image) makes it available without restarting anything.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config

log = logging.getLogger(__name__)

_SCENE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


@dataclass
class Scene:
    id: str
    title: str
    description: str
    image: str
    port: int
    cpus: float
    memory: str
    thumbnail: Optional[Path]
    buildable: bool  # has a Dockerfile the admin page can build

    def public(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "thumbnail": f"/scenes/{self.id}/thumbnail" if self.thumbnail else None,
        }


def _load_scene(scene_dir: Path) -> Optional[Scene]:
    meta_file = scene_dir / "scene.json"
    if not meta_file.is_file() or not _SCENE_ID.match(scene_dir.name):
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Skipping scene %s: invalid scene.json (%s)", scene_dir.name, exc)
        return None
    thumbnail = scene_dir / meta["thumbnail"] if meta.get("thumbnail") else None
    if thumbnail is not None and not thumbnail.resolve().is_relative_to(scene_dir.resolve()):
        thumbnail = None
    return Scene(
        id=scene_dir.name,
        title=str(meta.get("title", scene_dir.name)),
        description=str(meta.get("description", "")),
        image=str(meta.get("image") or f"{config.IMAGE_PREFIX}{scene_dir.name}"),
        port=int(meta.get("port", 8000)),
        cpus=float(meta.get("cpus", config.SIM_CPUS)),
        memory=str(meta.get("memory", config.SIM_MEMORY)),
        thumbnail=thumbnail if thumbnail is not None and thumbnail.is_file() else None,
        buildable=(scene_dir / "Dockerfile").is_file(),
    )


def list_scenes() -> list[Scene]:
    if not config.SCENES_DIR.is_dir():
        return []
    scenes = (_load_scene(d) for d in sorted(config.SCENES_DIR.iterdir()) if d.is_dir())
    return [s for s in scenes if s is not None]


def get_scene(scene_id: str) -> Optional[Scene]:
    if not _SCENE_ID.match(scene_id):
        return None
    scene_dir = config.SCENES_DIR / scene_id
    return _load_scene(scene_dir) if scene_dir.is_dir() else None
