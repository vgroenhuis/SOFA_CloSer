"""Reverse proxy from /sim/<id>/... to the simulation container.

Access rule enforced here (the "scene contract", see scenes/README.md):
  * GET/HEAD requests and server->browser WebSocket traffic are public, so
    anyone can watch.
  * Any other HTTP method, and browser->server WebSocket messages, only get
    through for the owner of the simulation (valid claim key).
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

import httpx
from fastapi import WebSocket
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from websockets.asyncio.client import connect as ws_connect

log = logging.getLogger(__name__)

# Hop-by-hop headers (RFC 7230 6.1) plus ones we deliberately don't forward:
# the orchestrator's own cookies (claim keys, admin session) never reach a
# simulation container.
_DROP_REQUEST_HEADERS = {
    "host", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "cookie", "x-msd-token",
}
_DROP_RESPONSE_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "set-cookie",
}

READ_ONLY_METHODS = {"GET", "HEAD", "OPTIONS"}


class SimProxy:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0), follow_redirects=False)
        # sim_id -> {"owner": n, "viewer": n} open WebSocket connections.
        self.connections: dict[str, dict[str, int]] = defaultdict(lambda: {"owner": 0, "viewer": 0})

    async def close(self) -> None:
        await self._client.aclose()

    async def probe(self, base_url: str) -> bool:
        """True once the simulation's web server answers at all."""
        try:
            response = await self._client.get(base_url + "/", timeout=2.0)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def http(self, request: Request, base_url: str, path: str) -> Response:
        url = f"{base_url}/{path}"
        if request.url.query:
            url += f"?{request.url.query}"
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in _DROP_REQUEST_HEADERS]
        upstream_request = self._client.build_request(
            request.method, url, headers=headers, content=request.stream()
        )
        try:
            upstream = await self._client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            log.info("Upstream %s unavailable: %s", url, exc)
            return Response("Simulation is not reachable (yet).", status_code=502, media_type="text/plain")
        response_headers = {
            k: v for k, v in upstream.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS
        }
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream.aclose),
        )

    async def websocket(self, websocket: WebSocket, ws_url: str, sim_id: str, is_owner: bool, on_owner_message) -> None:
        role = "owner" if is_owner else "viewer"
        try:
            upstream = await ws_connect(ws_url, max_size=None, open_timeout=10)
        except Exception as exc:
            log.info("Upstream websocket %s unavailable: %s", ws_url, exc)
            await websocket.close(code=1013)  # try again later
            return

        await websocket.accept()
        self.connections[sim_id][role] += 1

        async def browser_to_sim() -> None:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if not is_owner:
                    continue  # watchers are read-only
                on_owner_message()
                if message.get("text") is not None:
                    await upstream.send(message["text"])
                elif message.get("bytes") is not None:
                    await upstream.send(message["bytes"])

        async def sim_to_browser() -> None:
            async for message in upstream:
                if isinstance(message, str):
                    await websocket.send_text(message)
                else:
                    await websocket.send_bytes(message)

        tasks = [asyncio.create_task(browser_to_sim()), asyncio.create_task(sim_to_browser())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.connections[sim_id][role] = max(0, self.connections[sim_id][role] - 1)
            await upstream.close()
            try:
                await websocket.close()
            except RuntimeError:
                pass  # already closed by the browser

    def viewer_counts(self, sim_id: str) -> dict[str, int]:
        counts = self.connections.get(sim_id)
        return dict(counts) if counts else {"owner": 0, "viewer": 0}
