"""FastAPI app around a Runner, following the MultiSim scene contract:
relative URLs, GET is read-only, every change is a POST."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import SceneConfig
from .runner import Runner

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
CLIENT_QUEUE_FRAMES = 6


class ParamsBody(BaseModel):
    changes: dict[str, Any]


class PresetBody(BaseModel):
    name: str


class ImportBody(BaseModel):
    scene: str
    name: str


class ControlBody(BaseModel):
    key: Optional[str] = None
    value: Any = None
    # Several sliders at once (the manipulator): {key: value}.
    values: Optional[dict[str, Any]] = None


class Client:
    """One browser connection. Frames are dropped (not queued up) when the
    browser can't keep up; setup/status messages are never dropped."""

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self.queue: asyncio.Queue = asyncio.Queue()
        self.frames_queued = 0

    def offer(self, message: Any) -> None:
        if isinstance(message, bytes):
            if self.frames_queued >= CLIENT_QUEUE_FRAMES:
                return
            self.frames_queued += 1
        self.queue.put_nowait(message)

    async def pump(self) -> None:
        while True:
            message = await self.queue.get()
            if isinstance(message, bytes):
                self.frames_queued -= 1
                await self.ws.send_bytes(message)
            else:
                await self.ws.send_text(message)


def load_config(path: Path) -> SceneConfig:
    spec = importlib.util.spec_from_file_location("sofaweb_scene_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def create_app(config_path: Path) -> FastAPI:
    config = load_config(config_path)
    clients: set[Client] = set()
    loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

    def deliver(message: Any) -> None:
        loop = loop_holder.get("loop")
        if loop is None:
            return
        loop.call_soon_threadsafe(lambda: [c.offer(message) for c in list(clients)])

    runner = Runner(config, config_path.parent, deliver)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop_holder["loop"] = asyncio.get_running_loop()
        runner.start()
        yield
        await asyncio.to_thread(runner.stop)

    app = FastAPI(title=config.title, lifespan=lifespan, docs_url=None, redoc_url=None)

    async def command(name: str, **kwargs) -> Any:
        return await asyncio.wrap_future(runner.request(name, **kwargs))

    @app.websocket("/ws/sim")
    async def ws_sim(websocket: WebSocket) -> None:
        await websocket.accept()
        client = Client(websocket)
        for message in runner.snapshot():
            client.offer(message)
        clients.add(client)
        pump = asyncio.create_task(client.pump())
        try:
            while True:
                await websocket.receive_text()  # nothing is expected from the browser
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(client)
            pump.cancel()

    @app.get("/api/scene")
    async def scene() -> dict:
        return {
            "title": config.title,
            "about": config.about,
            "origin": config.origin,
            "charts": config.charts,
            "readouts": config.readouts,
            "spec": runner.spec,
            # Initial values come from params.json ...
            "display": [{**d, "value": runner.defaults.get(d["key"])} for d in config.display],
            # ... unless an admin set the Display / Lighting panels' state
            # as the scene's default (see Runner.apply_saved_defaults).
            "displayDefaults": {
                **(runner.saved_defaults.get("display") or {}),
                **(runner.saved_defaults.get("lighting") or {}),
            },
            "view": config.view,
            "presets": runner.presets(),
            "imports": runner.import_sources(),
            "hasGauge": config.gauge is not None,
            "controls": [{k: v for k, v in c.items() if k != "apply"} for c in config.controls],
            "manipulator": config.manipulator,
        }

    @app.post("/api/defaults")
    async def set_defaults(body: dict[str, Any]) -> dict:
        # From the orchestrator only (its proxy refuses this path to browsers).
        runner.apply_saved_defaults(body)
        runner.request("presets_changed")  # keeps or forgets recorded preset runs
        return {"ok": True}

    @app.get("/api/info")
    async def info() -> dict:
        return runner.info()

    @app.get("/api/params")
    async def get_params() -> dict:
        visible = runner.visible_keys()
        return {k: v for k, v in runner.params.items() if k in visible}

    @app.post("/api/params")
    async def set_params(body: ParamsBody) -> dict:
        changes = runner.coerce(body.changes)
        params = await command("params", changes=changes)
        visible = runner.visible_keys()
        return {k: v for k, v in params.items() if k in visible}

    @app.post("/api/params/default")
    async def default_params() -> dict:
        defaults = {k: v for k, v in runner.defaults.items() if k in runner.visible_keys()}
        params = await command("params", changes=defaults, rebuild=True)
        return {k: v for k, v in params.items() if k in runner.visible_keys()}

    @app.post("/api/params/preset")
    async def preset_params(body: PresetBody):
        params = runner.preset_params(body.name)
        if params is None:
            return JSONResponse({"detail": {"message": f"No preset named {body.name!r}."}}, status_code=404)
        params = await command("params", changes=params, rebuild=True)
        return {k: v for k, v in params.items() if k in runner.visible_keys()}

    @app.post("/api/params/import")
    async def import_params(body: ImportBody):
        """Another scene's parameters preset, converted to this scene's
        parameters (see SceneConfig.import_presets) and applied, with notes on
        what couldn't be carried over directly."""
        try:
            result = await asyncio.to_thread(runner.import_preset, body.scene, body.name)
        except KeyError as exc:
            return JSONResponse({"detail": {"message": str(exc.args[0])}}, status_code=404)
        params = await command("params", changes=result["params"], rebuild=True)
        visible = runner.visible_keys()
        return {**result, "params": {k: v for k, v in params.items() if k in visible}}

    @app.post("/api/control")
    async def control(body: ControlBody):
        # A slider moved, a button pressed or let go (owner only: the
        # orchestrator's proxy passes POSTs from the owner alone).
        try:
            if body.values is not None:
                return await command("controls", values=body.values)
            return await command("control", key=body.key, value=body.value)
        except (ValueError, RuntimeError) as exc:
            return JSONResponse({"detail": {"message": str(exc)}}, status_code=409)

    @app.post("/api/simulate")
    async def simulate() -> dict:
        # Simulate a preset's recorded run live instead of playing it back.
        await command("simulate")
        return {"status": "ok"}

    @app.get("/api/recording")
    def recording(start: int = 0) -> Response:
        blob = runner.recording_blob(start)
        if blob is None:
            return Response(status_code=204)
        return Response(
            blob["data"],
            media_type="application/octet-stream",
            headers={
                "X-Recording-Generation": str(blob["generation"]),
                "X-Recording-Id": str(blob["serial"]),
                "X-Recording-Start": str(blob["start"]),
                "X-Recording-Frames": str(blob["frames"]),
                "X-Recording-Bytes": str(len(blob["data"])),
                "Cache-Control": "no-store",
            },
        )

    @app.post("/api/reset")
    async def reset() -> dict:
        runner.request("reset")
        return {"status": "ok"}

    @app.post("/api/pause")
    async def pause() -> dict:
        await command("pause")
        return {"status": "ok"}

    @app.post("/api/resume")
    async def resume() -> dict:
        await command("resume")
        return {"status": "ok"}

    @app.post("/api/auto-restart")
    async def auto_restart(enabled: bool) -> dict:
        await command("auto_restart", enabled=enabled)
        return {"status": "ok", "enabled": enabled}

    @app.get("/api/console")
    async def console(after: int = 0) -> JSONResponse:
        return JSONResponse(runner.console.since(after))

    @app.get("/api/data.csv")
    def data_csv() -> PlainTextResponse:
        slug = "".join(c if c.isalnum() else "-" for c in config.title.lower()).strip("-")
        return PlainTextResponse(
            runner.csv(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=sofa-{slug}.csv"},
        )

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app
