"""FastAPI app serving the browser-based SOFA cube simulation viewer.

Run manually with:
	uvicorn backend.main:app --host 0.0.0.0 --port 8000
or via the run_docker.bat shortcut at the repo root.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.simulation import SimulationRunner


class ParamUpdate(BaseModel):
    floorStiffness: Optional[float] = None
    floorDamping: Optional[float] = None
    rayleighMass: Optional[float] = None
    rayleighStiffness: Optional[float] = None
    lateralFrictionDamping: Optional[float] = None

_STATIC_DIR = Path(__file__).parent.parent / "static"

_clients: set[WebSocket] = set()
_loop: Optional[asyncio.AbstractEventLoop] = None
_runner: Optional[SimulationRunner] = None


def _on_frame(frame: dict) -> None:
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(frame), _loop)


async def _broadcast(frame: dict) -> None:
    message = json.dumps(frame)
    dead = []
    for ws in _clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop, _runner
    _loop = asyncio.get_running_loop()
    _runner = SimulationRunner(on_frame=_on_frame)
    _runner.start()
    yield
    if _runner is not None:
        _runner.stop()


app = FastAPI(title="SOFA Cube Simulation", lifespan=lifespan)


@app.websocket("/ws/sim")
async def ws_sim(websocket: WebSocket) -> None:
    await websocket.accept()
    _clients.add(websocket)
    try:
        while True:
            # The client never sends anything meaningful; this just blocks
            # until disconnect so we can drop the socket from _clients.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)


@app.post("/api/reset")
async def reset_simulation() -> dict:
    if _runner is not None:
        _runner.reset()
    return {"status": "ok"}


@app.post("/api/pause")
async def pause_simulation() -> dict:
    if _runner is not None:
        _runner.set_paused(True)
    return {"status": "ok"}


@app.post("/api/resume")
async def resume_simulation() -> dict:
    if _runner is not None:
        _runner.set_paused(False)
    return {"status": "ok"}


@app.post("/api/auto-reset")
async def set_auto_reset(enabled: bool) -> dict:
    if _runner is not None:
        _runner.set_auto_reset(enabled)
    return {"status": "ok", "enabled": enabled}


@app.get("/api/info")
async def get_info() -> dict:
    return _runner.get_info() if _runner is not None else {}


@app.get("/api/params")
async def get_params() -> dict:
    return _runner.get_params() if _runner is not None else {}


@app.post("/api/params")
async def set_params(update: ParamUpdate) -> dict:
    if _runner is not None:
        changes = {k: v for k, v in update.model_dump().items() if v is not None}
        _runner.set_params(**changes)
        return _runner.get_params()
    return {}


@app.post("/api/params/default")
async def reset_params() -> dict:
    return _runner.reset_params() if _runner is not None else {}


@app.get("/api/data.csv")
def download_data() -> PlainTextResponse:
    # A plain (sync) def so FastAPI runs it in its worker threadpool --
    # building this string can be a few thousand rows and shouldn't block
    # the event loop that's also driving the websocket broadcast.
    csv_text = _runner.get_log_csv() if _runner is not None else ""
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sofa-cube-sim.csv"},
    )


# Registered after the API routes above, so those specific paths are matched
# first -- a StaticFiles mount at "/" would otherwise try (and fail) to
# resolve them as files.
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
