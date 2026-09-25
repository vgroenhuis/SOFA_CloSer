"""FastAPI app serving the browser-based SOFA double pendulum viewer.

Run manually with:
	uvicorn backend.main:app --host 0.0.0.0 --port 8000

Follows the MultiSimDocker scene contract (see scenes/README.md): relative
URLs in the frontend, GET endpoints are read-only, everything that changes
the simulation is a POST.
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
    l1: Optional[float] = None
    m1_kg: Optional[float] = None
    l2: Optional[float] = None
    m2_kg: Optional[float] = None
    anchor_z: Optional[float] = None
    gravity: Optional[float] = None
    theta1_initial_deg: Optional[float] = None
    omega1_initial: Optional[float] = None
    theta2_initial_deg: Optional[float] = None
    omega2_initial: Optional[float] = None


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


app = FastAPI(title="SOFA Double Pendulum", lifespan=lifespan)


@app.websocket("/ws/sim")
async def ws_sim(websocket: WebSocket) -> None:
    await websocket.accept()
    _clients.add(websocket)
    try:
        while True:
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


@app.get("/api/info")
async def get_info() -> dict:
    return _runner.get_info() if _runner is not None else {}


@app.get("/api/params")
async def get_params() -> dict:
    return _runner.get_params() if _runner is not None else {}


@app.post("/api/params")
async def set_params(update: ParamUpdate) -> dict:
    if _runner is None:
        return {}
    changes = {k: v for k, v in update.model_dump().items() if v is not None}
    return _runner.set_params(**changes)


@app.post("/api/params/default")
async def reset_params() -> dict:
    return _runner.reset_params() if _runner is not None else {}


@app.get("/api/data.csv")
def download_data() -> PlainTextResponse:
    csv_text = _runner.get_log_csv() if _runner is not None else ""
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sofa-double-pendulum.csv"},
    )


app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
