"""Saved runs: a stopped simulation kept for playback at /run/<token>.

"Save run" on a simulation's page (its owner) has the orchestrator fetch the
run from the simulation (GET api/export: a zip with what a browser needs to
play it back -- the setup message with the meshes, the status, the chart
history, a full frame, the recording, the console, the data, the scene's
images -- and none of the scene's code) and keep it as
DATA_DIR/runs/<token>.zip. The token is random and long: whoever has the URL
may watch the run, whatever the scene's level (hidden or admin-only
included); the lobby lists a run only to whoever may see its level.

Playback needs no simulation: the orchestrator answers the viewer's requests
(the scene's own viewer.js, from sofaweb's static folder, in its read-only
"viewer" role) from the zip, so a run still plays after its scene changed or
was removed. The formats it depends on are the setup / frame messages and
the recording's (sofaweb's recording.FORMAT_VERSION, kept in meta.json):
changes to those must stay readable by the viewer.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import secrets
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

from . import config

RUNS_DIR = config.DATA_DIR / "runs"
VIEWER_DIR = config.SOFAWEB_DIR / "sofaweb" / "static"
TOKEN = re.compile(r"^[A-Za-z0-9_-]{16,40}$")
MAX_RUN_BYTES = 400 * 2**20  # a run's zip (the recording itself stops at 200 MB raw)

_TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".svg": "image/svg+xml",
          ".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json", ".woff2": "font/woff2"}


def new_token() -> str:
    return secrets.token_urlsafe(15)  # 20 characters, 120 random bits


def path_of(token: str) -> Optional[Path]:
    return RUNS_DIR / f"{token}.zip" if TOKEN.match(token or "") else None


def check_bundle(data: bytes) -> dict:
    """The meta.json of an exported run (ValueError if it isn't one)."""
    if len(data) > MAX_RUN_BYTES:
        raise ValueError(f"the run is too big to keep ({len(data) >> 20} MB)")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
            for needed in ("meta.json", "scene.json", "messages.json", "recording.gz", "recording.json"):
                if needed not in names:
                    raise ValueError(f"{needed} is missing")
            return json.loads(z.read("meta.json"))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a zip: {exc}") from exc


def store_bundle(token: str, data: bytes) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    target = path_of(token)
    tmp = RUNS_DIR / f".{token}.{os.getpid()}.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, target)


def remove_bundle(token: str) -> None:
    for path in (path_of(token), thumbnail_path(token)):
        if path is not None:
            path.unlink(missing_ok=True)


def thumbnail_path(token: str) -> Optional[Path]:
    return RUNS_DIR / f"{token}.jpg" if TOKEN.match(token or "") else None


def has_thumbnail(token: str) -> bool:
    path = thumbnail_path(token)
    return path is not None and path.is_file()


def store_thumbnail(token: str, data_url: str) -> bool:
    """The view when the run was saved (a JPEG data URL from the viewer's canvas); anything else is ignored."""
    prefix = "data:image/jpeg;base64,"
    if not data_url.startswith(prefix):
        return False
    try:
        data = base64.b64decode(data_url[len(prefix):], validate=True)
    except ValueError:
        return False
    if not data.startswith(b"\xff\xd8"):
        return False
    thumbnail_path(token).write_bytes(data)
    return True


def _read(token: str, *names: str) -> list[Optional[bytes]]:
    path = path_of(token)
    if path is None or not path.is_file():
        raise FileNotFoundError(token)
    out: list[Optional[bytes]] = []
    with zipfile.ZipFile(path) as z:
        present = set(z.namelist())
        for name in names:
            out.append(z.read(name) if name in present else None)
    return out


async def read(token: str, *names: str) -> list[Optional[bytes]]:
    return await asyncio.to_thread(_read, token, *names)


def _json(data: Optional[bytes], default):
    return json.loads(data) if data is not None else default


async def serve(token: str, path: str, query: dict) -> Response:
    """GET <root>run/<token>/v/<path>: the viewer's page and files, and its api/... answered from the run."""
    try:
        if path in ("", "index.html"):
            return FileResponse(VIEWER_DIR / "index.html", media_type="text/html")
        if path == "api/scene":
            (scene,) = await read(token, "scene.json")
            return Response(scene, media_type="application/json")
        if path == "api/info":
            (info,) = await read(token, "info.json")
            return Response(info or b"{}", media_type="application/json")
        if path == "api/params":
            (params,) = await read(token, "params.json")
            return Response(params or b"{}", media_type="application/json")
        if path == "api/console":
            (lines,) = await read(token, "console.json")
            after = int(query.get("after") or 0)
            return JSONResponse([line for line in _json(lines, []) if line.get("seq", 0) > after])
        if path == "api/data.csv":
            (csv, meta) = await read(token, "data.csv", "meta.json")
            slug = "".join(c if c.isalnum() else "-" for c in _json(meta, {}).get("title", "run").lower()).strip("-")
            return PlainTextResponse((csv or b"t\n").decode(), media_type="text/csv", headers={
                "Content-Disposition": f"attachment; filename=sofa-{slug}-{token[:6]}.csv"})
        if path == "api/recording":
            if int(query.get("start") or 0) > 0:
                return Response(status_code=204)  # (it came whole the first time)
            data, info = await read(token, "recording.gz", "recording.json")
            info = _json(info, {})
            return Response(data, media_type="application/octet-stream", headers={
                "X-Recording-Generation": str(info.get("generation", 0)),
                "X-Recording-Id": str(info.get("serial", 0)),
                "X-Recording-Start": "0",
                "X-Recording-Frames": str(info.get("frames", 0)),
                "X-Recording-Bytes": str(len(data or b"")),
                "Cache-Control": "private, max-age=86400",
            })
        if path.startswith("images/"):
            name = path[len("images/"):]
            data, media = await read(token, f"images/{name}", "images.json")
            if data is None or "/" in name:
                return Response(status_code=404)
            return Response(data, media_type=_json(media, {}).get(name, "application/octet-stream"),
                            headers={"Cache-Control": "private, max-age=86400"})
        if path.startswith("api/") or path.startswith("choices/"):
            return Response(status_code=404)
    except FileNotFoundError:
        return Response("No such saved run.", status_code=404, media_type="text/plain")
    # The viewer's own files (viewer.js, textures.js, ...): sofaweb's, as for every simulation.
    target = (VIEWER_DIR / path).resolve()
    if VIEWER_DIR.resolve() not in target.parents or not target.is_file():
        return Response(status_code=404)
    return FileResponse(target, media_type=_TYPES.get(target.suffix.lower()))


async def websocket(ws: WebSocket, token: str) -> None:
    """The viewer's ws/sim: what a simulation sends a newly connected browser, then nothing."""
    try:
        (order,) = await read(token, "messages.json")
        names = _json(order, [])
        messages = await read(token, *names)
    except FileNotFoundError:
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        for name, data in zip(names, messages):
            if data is None:
                continue
            if name.endswith(".bin"):
                await ws.send_bytes(data)
            else:
                await ws.send_text(data.decode())
        while True:
            await ws.receive_text()  # (nothing is expected from the browser)
    except WebSocketDisconnect:
        pass
