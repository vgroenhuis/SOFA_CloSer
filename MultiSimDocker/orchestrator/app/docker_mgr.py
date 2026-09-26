"""Thin wrapper around the Docker SDK for simulation container lifecycle.

All methods are blocking; the async callers run them via asyncio.to_thread.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import docker
from docker.errors import APIError, ImageNotFound, NotFound

from . import config
from .catalog import Scene

log = logging.getLogger(__name__)


class DockerManager:
    def __init__(self) -> None:
        self._client = docker.from_env()
        # scene_id -> {"state": "building" | "done" | "failed", "detail": str, "ts": float}
        self.builds: dict[str, dict] = {}
        self._builds_lock = threading.Lock()

    def ensure_network(self) -> None:
        try:
            self._client.networks.get(config.DOCKER_NETWORK)
        except NotFound:
            log.info("Creating docker network %s", config.DOCKER_NETWORK)
            self._client.networks.create(config.DOCKER_NETWORK, driver="bridge")

    def image_exists(self, image: str) -> bool:
        try:
            self._client.images.get(image)
            return True
        except ImageNotFound:
            return False

    def start(self, sim_id: str, container_name: str, scene: Scene) -> None:
        self._client.containers.run(
            scene.image,
            name=container_name,
            detach=True,
            network=config.DOCKER_NETWORK,
            labels={
                config.MANAGED_LABEL: "true",
                "msd.sim_id": sim_id,
                "msd.scene_id": scene.id,
            },
            environment={"MSD_SIM_ID": sim_id},
            nano_cpus=int(scene.cpus * 1e9),
            mem_limit=scene.memory,
            pids_limit=1024,
            restart_policy={"Name": "no"},
        )

    def remove(self, container_name: str) -> None:
        try:
            self._client.containers.get(container_name).remove(force=True)
        except NotFound:
            pass
        except APIError as exc:
            # A removal already in progress (e.g. reaper and user racing).
            log.warning("Removing %s failed: %s", container_name, exc)

    def restart(self, container_name: str) -> None:
        self._client.containers.get(container_name).restart(timeout=5)

    def managed_containers(self) -> dict[str, str]:
        """container name -> docker status ('running', 'exited', ...)."""
        containers = self._client.containers.list(all=True, filters={"label": f"{config.MANAGED_LABEL}=true"})
        return {c.name: c.status for c in containers}

    def logs(self, container_name: str, tail: int = 300) -> str:
        try:
            raw = self._client.containers.get(container_name).logs(tail=tail, timestamps=True)
        except NotFound:
            return "(container not found)"
        return raw.decode("utf-8", errors="replace")

    def stats(self, container_name: str) -> dict:
        """One-shot CPU/memory snapshot for the admin page."""
        try:
            s = self._client.containers.get(container_name).stats(stream=False)
        except NotFound:
            return {}
        try:
            cpu_delta = s["cpu_stats"]["cpu_usage"]["total_usage"] - s["precpu_stats"]["cpu_usage"]["total_usage"]
            sys_delta = s["cpu_stats"]["system_cpu_usage"] - s["precpu_stats"].get("system_cpu_usage", 0)
            ncpu = s["cpu_stats"].get("online_cpus") or 1
            cpu = (cpu_delta / sys_delta) * ncpu * 100.0 if sys_delta > 0 else 0.0
        except (KeyError, TypeError):
            cpu = None
        mem = s.get("memory_stats", {})
        return {"cpuPercent": cpu, "memoryBytes": mem.get("usage"), "memoryLimitBytes": mem.get("limit")}

    # -- image builds (triggered from the admin page) -------------------

    def build_image(self, scene: Scene, scene_dir: Path) -> None:
        with self._builds_lock:
            if self.builds.get(scene.id, {}).get("state") == "building":
                return
            self.builds[scene.id] = {"state": "building", "detail": "", "ts": time.time()}
        try:
            _, output = self._client.images.build(path=str(scene_dir), tag=scene.image, rm=True)
            tail = [line.get("stream", "").rstrip() for line in output if line.get("stream")]
            self.builds[scene.id] = {"state": "done", "detail": "\n".join(tail[-15:]), "ts": time.time()}
        except Exception as exc:  # BuildError, APIError, ...
            detail = str(exc)
            build_log = getattr(exc, "build_log", None)
            if build_log:
                lines = [line.get("stream", "") or line.get("error", "") for line in build_log]
                detail += "\n" + "".join(lines)[-3000:]
            self.builds[scene.id] = {"state": "failed", "detail": detail, "ts": time.time()}
