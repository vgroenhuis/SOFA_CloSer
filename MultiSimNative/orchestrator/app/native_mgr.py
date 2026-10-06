"""Simulation lifecycle as plain host processes (no Docker).

Each simulation is one child process of a scene's web app, listening on a
free 127.0.0.1 port. It is started in its own session, wrapped in a
transient systemd user scope when available (`systemd-run --user --scope`),
which gives it real cgroup CPU and memory caps.

The method names mirror what the orchestrator needs from a runtime:
start / remove / restart / managed_containers / logs / stats / build_image.
A "container name" is just the simulation's handle (`msd-sim-<id>`).

State survives an orchestrator restart: each simulation writes
`<data>/run/<name>.json` (pid, port, start time) and is re-adopted on
startup if that process is still the same one.

All methods are blocking; the async callers run them via asyncio.to_thread.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import psutil

from . import config
from .catalog import Scene

log = logging.getLogger(__name__)

_MEM_SUFFIX = {"k": "K", "m": "M", "g": "G", "t": "T"}


def _systemd_memory(value: str) -> str:
    """'2g' -> '2G' (systemd wants upper-case suffixes)."""
    value = value.strip()
    return value[:-1] + _MEM_SUFFIX[value[-1].lower()] if value and value[-1].lower() in _MEM_SUFFIX else value


def _memory_bytes(value: str) -> Optional[int]:
    value = value.strip().lower()
    mult = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}
    try:
        return int(float(value[:-1]) * mult[value[-1]]) if value[-1] in mult else int(value)
    except (ValueError, IndexError):
        return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _can_use_scope() -> bool:
    """True if `systemd-run --user --scope` works with resource limits."""
    if config.USE_CGROUPS == "off" or shutil.which("systemd-run") is None:
        return False
    try:
        r = subprocess.run(
            ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=512M", "-p", "CPUQuota=100%", "true"],
            capture_output=True,
            timeout=10,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class NativeManager:
    def __init__(self) -> None:
        # scene_id -> {"state": "building" | "done" | "failed", "detail": str, "ts": float}
        self.builds: dict[str, dict] = {}
        self._builds_lock = threading.Lock()
        self._lock = threading.Lock()
        # name -> {"pid", "port", "created", "sim_id", "scene_id", "defaults", "proc" (Popen or None)}
        self._procs: dict[str, dict] = {}
        self._run_dir = config.DATA_DIR / "run"
        self._log_dir = config.DATA_DIR / "logs"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self.use_scope = _can_use_scope()
        log.info("Resource limits via systemd scopes: %s", "on" if self.use_scope else "off (no caps!)")
        self._adopt()

    # -- runtime environment -----------------------------------------------

    def _runtime_ok(self) -> bool:
        return (config.SOFA_ROOT / "plugins" / "SofaPython3").is_dir() and Path(config.PYTHON).exists()

    def _env(self) -> dict:
        env = dict(os.environ)
        sofa = config.SOFA_ROOT
        env["PYTHONPATH"] = os.pathsep.join(
            [str(sofa / "plugins/SofaPython3/lib/python3/site-packages"), str(config.SOFAWEB_DIR)]
        )
        libs = [str(sofa / "lib"), str(sofa / "plugins/SofaPython3/lib")]
        if config.PYTHON_LIBDIR:
            libs.append(str(config.PYTHON_LIBDIR))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libs)
        env["SOFA_ROOT"] = str(sofa)
        return env

    @staticmethod
    def _scene_command(scene_dir: Path, port: int) -> list[str]:
        """What to run for a scene, from scene.json's optional "command"
        (with {python} and {port} placeholders) or by convention."""
        meta = {}
        try:
            meta = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        template = meta.get("command")
        if not template:
            if (scene_dir / "sofaweb_scene.py").is_file():
                template = ["{python}", "-m", "sofaweb", "serve", "sofaweb_scene.py", "--host", "127.0.0.1", "--port", "{port}"]
            else:
                template = ["{python}", "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "{port}"]
        return [str(part).format(python=config.PYTHON, port=port) for part in template]

    # -- readiness ("image exists") ---------------------------------------------

    def image_exists(self, scene: Scene) -> bool:
        """A scene is launchable when the SOFA runtime is installed and the
        scene folder has something to run."""
        scene_dir = config.SCENES_DIR / scene.id
        return self._runtime_ok() and ((scene_dir / "sofaweb_scene.py").is_file() or (scene_dir / "backend").is_dir() or (scene_dir / "scene.json").is_file())

    def ensure_network(self) -> None:
        """Nothing to do; kept so the orchestrator's startup is runtime-agnostic."""

    def address(self, name: str) -> Optional[str]:
        info = self._procs.get(name)
        return f"127.0.0.1:{info['port']}" if info else None

    # -- lifecycle ------------------------------------------------------------

    def start(self, sim_id: str, name: str, scene: Scene, defaults: Optional[dict] = None) -> None:
        scene_dir = config.SCENES_DIR / scene.id
        port = _free_port()
        env = self._env()
        env["MSD_SIM_ID"] = sim_id
        if defaults:
            env["MSD_SCENE_DEFAULTS"] = json.dumps(defaults)
        cmd = self._scene_command(scene_dir, port)
        if self.use_scope:
            cmd = [
                "systemd-run", "--user", "--scope", "-q",
                f"--unit={name}",
                "-p", f"CPUQuota={max(int(scene.cpus * 100), 1)}%",
                "-p", f"MemoryMax={_systemd_memory(scene.memory)}",
                "-p", "TasksMax=1024",
                "--", *cmd,
            ]
        logfile = open(self._log_dir / f"{name}.log", "ab")
        logfile.write(f"--- starting {scene.id} on port {port}: {' '.join(cmd)}\n".encode())
        logfile.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=scene_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        logfile.close()
        created = psutil.Process(proc.pid).create_time()
        info = {
            "pid": proc.pid, "port": port, "created": created,
            "sim_id": sim_id, "scene_id": scene.id, "defaults": defaults, "proc": proc,
        }
        with self._lock:
            self._procs[name] = info
        self._save(name, info)

    def _save(self, name: str, info: dict) -> None:
        data = {k: v for k, v in info.items() if k not in ("proc", "defaults")}
        (self._run_dir / f"{name}.json").write_text(json.dumps(data))

    def _adopt(self) -> None:
        """Re-attach to simulations started by a previous orchestrator run."""
        for f in self._run_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                p = psutil.Process(data["pid"])
                if abs(p.create_time() - data["created"]) < 1.0 and p.status() != psutil.STATUS_ZOMBIE:
                    self._procs[f.stem] = {**data, "defaults": None, "proc": None}
                    log.info("Adopted running simulation %s (pid %s)", f.stem, data["pid"])
                    continue
            except (OSError, ValueError, KeyError, psutil.Error):
                pass
            f.unlink(missing_ok=True)

    def _alive(self, name: str) -> bool:
        info = self._procs.get(name)
        if not info:
            return False
        proc = info.get("proc")
        if proc is not None:
            return proc.poll() is None
        try:
            p = psutil.Process(info["pid"])
            return abs(p.create_time() - info["created"]) < 1.0 and p.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    def remove(self, name: str) -> None:
        with self._lock:
            info = self._procs.pop(name, None)
        (self._run_dir / f"{name}.json").unlink(missing_ok=True)
        if not info:
            return
        try:
            root = psutil.Process(info["pid"])
            if abs(root.create_time() - info["created"]) < 1.0:
                tree = [root, *root.children(recursive=True)]
                for p in tree:
                    try:
                        p.send_signal(signal.SIGTERM)
                    except psutil.Error:
                        pass
                _, alive = psutil.wait_procs(tree, timeout=3)
                for p in alive:
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
        except psutil.Error:
            pass
        proc = info.get("proc")
        if proc is not None:
            try:
                proc.wait(timeout=5)  # reap the zombie
            except subprocess.TimeoutExpired:
                pass
        if self.use_scope:
            subprocess.run(["systemctl", "--user", "stop", f"{name}.scope"], capture_output=True, timeout=10)

    def restart(self, name: str) -> None:
        info = self._procs.get(name)
        if not info:
            raise KeyError(name)
        from . import catalog

        scene = catalog.get_scene(info["scene_id"])
        if scene is None:
            raise KeyError(info["scene_id"])
        sim_id, defaults = info["sim_id"], info.get("defaults")
        self.remove(name)
        self.start(sim_id, name, scene, defaults)

    def managed_containers(self) -> dict[str, str]:
        """handle -> 'running' | 'exited'."""
        with self._lock:
            names = list(self._procs)
        return {n: ("running" if self._alive(n) else "exited") for n in names}

    def logs(self, name: str, tail: int = 300) -> str:
        path = self._log_dir / f"{name}.log"
        if not path.is_file():
            return "(no log found)"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:]) + "\n"

    def stats(self, name: str) -> dict:
        """CPU / memory snapshot of the simulation's process tree."""
        info = self._procs.get(name)
        if not info or not self._alive(name):
            return {}
        try:
            root = psutil.Process(info["pid"])
            tree = [root, *root.children(recursive=True)]
            for p in tree:
                p.cpu_percent(None)
            time.sleep(0.3)
            cpu = sum(p.cpu_percent(None) for p in tree)
            mem = sum(p.memory_info().rss for p in tree)
        except psutil.Error:
            return {}
        scene = None
        try:
            from . import catalog

            scene = catalog.get_scene(info["scene_id"])
        except Exception:
            pass
        return {
            "cpuPercent": cpu,
            "memoryBytes": mem,
            "memoryLimitBytes": _memory_bytes(scene.memory) if scene else None,
        }

    # -- "build" (a scene self-check, triggered from the admin page) ---------------

    def build_image(self, scene: Scene, scene_dir: Path) -> None:
        with self._builds_lock:
            if self.builds.get(scene.id, {}).get("state") == "building":
                return
            self.builds[scene.id] = {"state": "building", "detail": "", "ts": time.time()}
        try:
            if (scene_dir / "sofaweb_scene.py").is_file():
                cmd = [config.PYTHON, "-m", "sofaweb", "check", "sofaweb_scene.py", "--seconds", "0.2"]
            else:
                check = scene_dir / "check.py"
                if not check.is_file():
                    raise RuntimeError("scene has no sofaweb_scene.py or check.py to verify")
                cmd = [config.PYTHON, str(check)]
            r = subprocess.run(cmd, cwd=scene_dir, env=self._env(), capture_output=True, text=True, timeout=600)
            tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
            state = "done" if r.returncode == 0 else "failed"
            self.builds[scene.id] = {"state": state, "detail": "\n".join(tail), "ts": time.time()}
        except Exception as exc:
            self.builds[scene.id] = {"state": "failed", "detail": str(exc), "ts": time.time()}
