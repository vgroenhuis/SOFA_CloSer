"""MultiSimDocker orchestrator.

One FastAPI process that
  * serves the lobby (/), the per-simulation viewer page (/view/<id>) and
    the admin page (/admin),
  * starts/stops one Docker container per claimed simulation, within a
    configurable pool size,
  * reverse-proxies /sim/<id>/... (HTTP + WebSocket) to that container,
    letting everyone watch but only the key holder control it,
  * runs a reaper loop that returns idle / crashed simulations to the pool.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import HTTPConnection

from . import auth, catalog, config
from .catalog import Scene
from .docker_mgr import DockerManager
from .proxy import READ_ONLY_METHODS, SimProxy
from .store import Sim, Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("msd")

_SIM_ID = re.compile(r"^[a-z0-9]{6}$")
ADMIN_COOKIE = "msd_admin"


class State:
    store: Store
    docker: DockerManager
    proxy: SimProxy
    secret: bytes
    # Serializes capacity checks + inserts so two simultaneous claims can't
    # both take the last free slot.
    claim_lock: asyncio.Lock
    # Sims with a container operation in flight (start/restart); the reaper
    # leaves them alone so it doesn't mistake a restarting container for a
    # crashed one.
    busy: set[str]
    login_limiter = auth.RateLimiter(max_events=5, window_seconds=300)
    reclaim_limiter = auth.RateLimiter(max_events=20, window_seconds=300)


S = State()


# -- helpers ---------------------------------------------------------------


def api_error(status: int, message: str, **extra) -> HTTPException:
    return HTTPException(status_code=status, detail={"message": message, **extra})


def client_addr(conn: HTTPConnection) -> str:
    if config.TRUST_PROXY_HEADERS:
        forwarded = conn.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return conn.client.host if conn.client else ""


def is_secure(conn: HTTPConnection) -> bool:
    if config.TRUST_PROXY_HEADERS and conn.headers.get("x-forwarded-proto"):
        return conn.headers["x-forwarded-proto"] == "https"
    return conn.url.scheme in ("https", "wss")


def presented_token(conn: HTTPConnection, sim_id: str) -> str:
    return conn.headers.get("x-msd-token") or conn.cookies.get(auth.token_cookie_name(sim_id)) or ""


def owns(conn: HTTPConnection, sim: Sim) -> bool:
    return auth.token_matches(presented_token(conn, sim.id), sim.token_hash)


def set_claim_cookie(response: Response, conn: HTTPConnection, sim_id: str, token: str) -> None:
    response.set_cookie(
        auth.token_cookie_name(sim_id),
        token,
        max_age=config.TOKEN_COOKIE_DAYS * 86400,
        httponly=True,
        samesite="lax",
        secure=is_secure(conn),
        path="/",
    )


def get_sim_or_404(sim_id: str) -> Sim:
    sim = S.store.get_sim(sim_id) if _SIM_ID.match(sim_id) else None
    if sim is None:
        released = S.store.last_release(sim_id) if _SIM_ID.match(sim_id) else None
        if released:
            raise api_error(410, "This simulation has ended.", reason=released["detail"], endedAt=released["ts"])
        raise api_error(404, "No such simulation.")
    return sim


def require_owner(conn: HTTPConnection, sim: Sim) -> None:
    if not owns(conn, sim):
        raise api_error(403, "You need this simulation's key to do that.")


def upstream_base(sim: Sim, scheme: str = "http") -> str:
    scene = catalog.get_scene(sim.scene_id)
    port = scene.port if scene else 8000
    return f"{scheme}://{sim.container_name}:{port}"


def sim_info(sim: Sim, conn: Optional[HTTPConnection] = None, admin: bool = False) -> dict:
    limits = S.store.get_limits()
    now = time.time()
    scene = catalog.get_scene(sim.scene_id)
    counts = S.proxy.viewer_counts(sim.id)
    mine = owns(conn, sim) if conn is not None else False
    held = sim.is_held(now)
    info = {
        "id": sim.id,
        "sceneId": sim.scene_id,
        "sceneTitle": scene.title if scene else sim.scene_id,
        "status": sim.status,
        "ownerName": sim.owner_name,
        "createdAt": sim.created_at,
        "expiresAt": sim.expires_at(limits.idle_timeout_minutes),
        "held": held,
        "holdUntil": sim.hold_until if held else None,
        "watchers": counts["viewer"],
        "ownerConnected": counts["owner"] > 0,
        "mine": mine,
        "serverTime": now,
    }
    if mine or admin:
        info["lastActive"] = sim.last_active
        info["holdReason"] = sim.hold_reason if held else ""
    if admin:
        info["clientAddr"] = sim.client_addr
        info["containerName"] = sim.container_name
    return info


async def release(sim_id: str, reason: str) -> None:
    sim = S.store.get_sim(sim_id)
    if sim is None or not S.store.delete_sim(sim_id):
        return
    S.store.log("released", sim_id, reason)
    log.info("Released %s (%s)", sim_id, reason)
    await asyncio.to_thread(S.docker.remove, sim.container_name)


async def create_sim(
    scene: Scene,
    owner_name: str,
    addr: str,
    *,
    enforce_client_limit: bool = True,
    ignore_capacity: bool = False,
    hold_until: Optional[float] = None,
    hold_reason: str = "",
) -> tuple[Sim, str]:
    if not await asyncio.to_thread(S.docker.image_exists, scene.image):
        raise api_error(503, f"The image for '{scene.title}' hasn't been built yet. Ask the administrator.")

    async with S.claim_lock:
        limits = S.store.get_limits()
        sims = S.store.list_sims()
        if len(sims) >= limits.max_sims and not ignore_capacity:
            next_free = min((s.expires_at(limits.idle_timeout_minutes) for s in sims), default=None)
            raise api_error(
                409,
                "All simulation slots are in use. You can watch a running simulation in the meantime.",
                nextFreeAt=next_free,
            )
        if enforce_client_limit and limits.max_claims_per_client > 0:
            mine = [s for s in sims if s.client_addr == addr]
            if len(mine) >= limits.max_claims_per_client:
                raise api_error(
                    429,
                    "You already have a simulation running. Release it before starting another.",
                    simIds=[s.id for s in mine],
                )

        sim_id = auth.new_sim_id()
        while S.store.get_sim(sim_id) is not None:
            sim_id = auth.new_sim_id()
        token = auth.new_token()
        now = time.time()
        sim = Sim(
            id=sim_id,
            scene_id=scene.id,
            container_name=f"{config.CONTAINER_PREFIX}{sim_id}",
            status="starting",
            token_hash=auth.hash_token(token),
            owner_name=owner_name,
            client_addr=addr,
            created_at=now,
            last_active=now,
            hold_until=hold_until,
            hold_reason=hold_reason if hold_until else "",
        )
        S.store.insert_sim(sim)
        S.busy.add(sim_id)

    S.store.log("claimed", sim_id, f"scene={scene.id} by={owner_name or '-'} from={addr}")
    try:
        await asyncio.to_thread(S.docker.start, sim_id, sim.container_name, scene)
    except Exception as exc:
        log.exception("Starting %s failed", sim_id)
        await release(sim_id, f"failed to start container: {exc}")
        raise api_error(500, "The simulation container failed to start.") from exc
    finally:
        S.busy.discard(sim_id)
    return sim, token


async def restart_sim(sim: Sim) -> None:
    S.busy.add(sim.id)
    try:
        S.store.set_status(sim.id, "starting")
        S.store.touch(sim.id)
        await asyncio.to_thread(S.docker.restart, sim.container_name)
    finally:
        S.busy.discard(sim.id)


def fmt_utc(epoch: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(epoch))


def validate_hold(sim: Sim, until: float, reason: str, max_hours: Optional[float]) -> None:
    now = time.time()
    if until <= now:
        raise api_error(400, "The hold must end in the future.")
    if max_hours is not None and until > now + max_hours * 3600 + 60:
        raise api_error(400, f"A hold can last at most {max_hours:g} hours.")
    if len(reason.strip()) < 3:
        raise api_error(400, "Please give a reason for the hold.")


# -- reaper ----------------------------------------------------------------


async def reap_once() -> None:
    limits = S.store.get_limits()
    # Containers are listed *before* sims: a sim row is always inserted
    # before its container is created, so any container seen here whose row
    # is missing really is an orphan.
    containers = await asyncio.to_thread(S.docker.managed_containers)
    sims = S.store.list_sims()
    known = {s.container_name for s in sims}
    now = time.time()

    for sim in sims:
        if sim.id in S.busy:
            continue
        state = containers.get(sim.container_name)
        if sim.status == "starting":
            if state == "running" and await S.proxy.probe(upstream_base(sim)):
                S.store.set_status(sim.id, "running")
                S.store.touch(sim.id)  # the idle clock starts once it's usable
                S.store.log("ready", sim.id)
            elif state in (None, "exited", "dead"):
                await release(sim.id, f"container stopped during startup ({state or 'missing'})")
            elif now - sim.last_active > config.START_TIMEOUT_SECONDS:
                await release(sim.id, f"did not become ready within {config.START_TIMEOUT_SECONDS:g}s")
            continue

        if state != "running":
            await release(sim.id, f"container stopped unexpectedly ({state or 'missing'})")
        elif now > sim.expires_at(limits.idle_timeout_minutes):
            reason = "hold ended, then idle timeout" if sim.hold_until else "idle timeout"
            await release(sim.id, reason)

    for name in containers:
        if name not in known and name.startswith(config.CONTAINER_PREFIX):
            log.info("Removing orphaned container %s", name)
            S.store.log("orphan-removed", detail=name)
            await asyncio.to_thread(S.docker.remove, name)


async def reaper_loop() -> None:
    while True:
        try:
            await reap_once()
        except Exception:
            log.exception("Reaper pass failed")
        await asyncio.sleep(config.REAPER_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.store = Store(config.DATA_DIR / "state.db")
    S.docker = DockerManager()
    S.proxy = SimProxy()
    S.secret = config.load_secret_key()
    S.claim_lock = asyncio.Lock()
    S.busy = set()
    await asyncio.to_thread(S.docker.ensure_network)
    if not config.ADMIN_PASSWORD:
        log.warning("MSD_ADMIN_PASSWORD is not set -- the admin page is disabled.")
    reaper = asyncio.create_task(reaper_loop())
    yield
    reaper.cancel()
    await S.proxy.close()


app = FastAPI(title="MultiSimDocker", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def revalidate_by_default(request: Request, call_next):
    # Without this, browsers heuristically cache the pages' scripts and
    # keep running old code after an update. "no-cache" still allows
    # caching, it just revalidates (cheap, via ETag / Last-Modified). Applies
    # to proxied scene pages too, unless the scene sets its own policy.
    response = await call_next(request)
    if request.method == "GET" and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-cache"
    return response


# -- pages -----------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def page_lobby() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/view/{sim_id}", include_in_schema=False)
async def page_view(sim_id: str) -> FileResponse:
    return FileResponse(config.STATIC_DIR / "view.html")


@app.get("/admin", include_in_schema=False)
async def page_admin() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "admin.html")


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    return {"ok": True}


@app.get("/scenes/{scene_id}/thumbnail", include_in_schema=False)
async def scene_thumbnail(scene_id: str) -> FileResponse:
    scene = catalog.get_scene(scene_id)
    if scene is None or scene.thumbnail is None:
        raise HTTPException(404)
    return FileResponse(scene.thumbnail, headers={"Cache-Control": "max-age=300"})


# -- public API ------------------------------------------------------------


class CreateSimRequest(BaseModel):
    sceneId: str
    ownerName: str = Field(default="", max_length=40)


class HoldRequest(BaseModel):
    until: float  # unix epoch seconds
    reason: str = Field(max_length=300)


class HeartbeatRequest(BaseModel):
    active: bool = True


class ReclaimRequest(BaseModel):
    key: str = Field(max_length=64)


@app.get("/api/scenes")
async def api_scenes() -> list[dict]:
    return [s.public() for s in catalog.list_scenes()]


@app.get("/api/status")
async def api_status(request: Request) -> dict:
    limits = S.store.get_limits()
    sims = S.store.list_sims()
    return {
        "capacity": limits.max_sims,
        "idleTimeoutMinutes": limits.idle_timeout_minutes,
        "maxHoldHours": limits.max_hold_hours,
        "maxHeldSims": limits.max_held_sims,
        "sims": [sim_info(s, request) for s in sims],
        "serverTime": time.time(),
    }


@app.post("/api/sims")
async def api_create_sim(body: CreateSimRequest, request: Request) -> JSONResponse:
    scene = catalog.get_scene(body.sceneId)
    if scene is None:
        raise api_error(404, "Unknown scene.")
    sim, token = await create_sim(scene, body.ownerName.strip(), client_addr(request))
    response = JSONResponse({"sim": sim_info(sim), "key": token})
    set_claim_cookie(response, request, sim.id, token)
    return response


@app.get("/api/sims/{sim_id}")
async def api_get_sim(sim_id: str, request: Request) -> dict:
    return sim_info(get_sim_or_404(sim_id), request)


@app.post("/api/sims/{sim_id}/heartbeat")
async def api_heartbeat(sim_id: str, body: HeartbeatRequest, request: Request) -> dict:
    sim = get_sim_or_404(sim_id)
    require_owner(request, sim)
    if body.active:
        S.store.touch(sim.id)
        sim = S.store.get_sim(sim.id) or sim
    return sim_info(sim, request)


@app.get("/api/sims/{sim_id}/key")
async def api_get_key(sim_id: str, request: Request) -> dict:
    """Echo the caller's own key back, so the viewer page can show it again
    (the cookie holding it is HttpOnly)."""
    sim = get_sim_or_404(sim_id)
    require_owner(request, sim)
    return {"key": auth.normalize_token(presented_token(request, sim.id))}


@app.put("/api/sims/{sim_id}/hold")
async def api_set_hold(sim_id: str, body: HoldRequest, request: Request) -> dict:
    sim = get_sim_or_404(sim_id)
    require_owner(request, sim)
    limits = S.store.get_limits()
    validate_hold(sim, body.until, body.reason, limits.max_hold_hours)
    async with S.claim_lock:
        others_held = [s for s in S.store.list_sims() if s.id != sim.id and s.is_held()]
        if len(others_held) >= limits.max_held_sims:
            raise api_error(
                409,
                f"At most {limits.max_held_sims} simulation(s) can be on hold at the same time, "
                "and that limit is reached. Contact the administrator if you need an exception.",
            )
        S.store.set_hold(sim.id, body.until, body.reason.strip())
    S.store.touch(sim.id)
    S.store.log("hold", sim.id, f"until={fmt_utc(body.until)} reason={body.reason.strip()}")
    return sim_info(S.store.get_sim(sim.id) or sim, request)


@app.delete("/api/sims/{sim_id}/hold")
async def api_clear_hold(sim_id: str, request: Request) -> dict:
    sim = get_sim_or_404(sim_id)
    require_owner(request, sim)
    S.store.set_hold(sim.id, None, "")
    S.store.touch(sim.id)
    S.store.log("hold-cleared", sim.id, "by owner")
    return sim_info(S.store.get_sim(sim.id) or sim, request)


@app.post("/api/sims/{sim_id}/restart")
async def api_restart(sim_id: str, request: Request) -> dict:
    sim = get_sim_or_404(sim_id)
    require_owner(request, sim)
    S.store.log("restarted", sim.id, "by owner")
    await restart_sim(sim)
    return sim_info(S.store.get_sim(sim.id) or sim, request)


@app.post("/api/sims/{sim_id}/release")
async def api_release(sim_id: str, request: Request) -> JSONResponse:
    sim = get_sim_or_404(sim_id)
    require_owner(request, sim)
    await release(sim.id, "released by owner")
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.token_cookie_name(sim.id), path="/")
    return response


@app.post("/api/sims/{sim_id}/forget")
async def api_forget(sim_id: str) -> JSONResponse:
    """Drop the key from this browser (e.g. on a shared demo PC) without
    releasing the simulation; it can be reclaimed later with the key."""
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.token_cookie_name(sim_id), path="/")
    return response


@app.post("/api/reclaim")
async def api_reclaim(body: ReclaimRequest, request: Request) -> JSONResponse:
    if not S.reclaim_limiter.allow(client_addr(request)):
        raise api_error(429, "Too many attempts. Wait a few minutes and try again.")
    key = auth.normalize_token(body.key)
    sim = S.store.find_by_token_hash(auth.hash_token(key)) if key else None
    if sim is None:
        raise api_error(404, "No running simulation matches that key. It may have expired.")
    S.store.touch(sim.id)
    S.store.log("reclaimed", sim.id, f"from={client_addr(request)}")
    response = JSONResponse({"simId": sim.id})
    set_claim_cookie(response, request, sim.id, key)
    return response


# -- admin API -------------------------------------------------------------


def require_admin(request: Request) -> None:
    if not config.ADMIN_PASSWORD:
        raise api_error(503, "The admin page is disabled: set MSD_ADMIN_PASSWORD.")
    if not auth.verify_admin_session(S.secret, request.cookies.get(ADMIN_COOKIE)):
        raise api_error(401, "Not logged in.")


class LoginRequest(BaseModel):
    password: str = Field(max_length=200)


class LimitsRequest(BaseModel):
    max_sims: Optional[int] = Field(default=None, ge=0, le=100)
    idle_timeout_minutes: Optional[float] = Field(default=None, gt=0, le=24 * 60)
    max_hold_hours: Optional[float] = Field(default=None, gt=0, le=24 * 14)
    max_held_sims: Optional[int] = Field(default=None, ge=0, le=100)
    max_claims_per_client: Optional[int] = Field(default=None, ge=0, le=100)


class AdminCreateRequest(BaseModel):
    sceneId: str
    ownerName: str = Field(default="", max_length=40)
    holdUntil: Optional[float] = None
    holdReason: str = Field(default="", max_length=300)
    ignoreCapacity: bool = False


class AdminHoldRequest(BaseModel):
    until: Optional[float] = None  # None clears the hold
    reason: str = Field(default="", max_length=300)


@app.post("/admin/api/login")
async def admin_login(body: LoginRequest, request: Request) -> JSONResponse:
    if not config.ADMIN_PASSWORD:
        raise api_error(503, "The admin page is disabled: set MSD_ADMIN_PASSWORD.")
    addr = client_addr(request)
    if not S.login_limiter.allow(addr):
        raise api_error(429, "Too many login attempts. Wait a few minutes.")
    if not auth.check_admin_password(config.ADMIN_PASSWORD, body.password):
        S.store.log("admin-login-failed", detail=f"from={addr}")
        raise api_error(401, "Wrong password.")
    S.store.log("admin-login", detail=f"from={addr}")
    response = JSONResponse({"ok": True})
    response.set_cookie(
        ADMIN_COOKIE,
        auth.sign_admin_session(S.secret, config.ADMIN_SESSION_HOURS),
        max_age=int(config.ADMIN_SESSION_HOURS * 3600),
        httponly=True,
        samesite="strict",
        secure=is_secure(request),
        path="/",
    )
    return response


@app.post("/admin/api/logout")
async def admin_logout() -> JSONResponse:
    response = JSONResponse({"ok": True})
    response.delete_cookie(ADMIN_COOKIE, path="/")
    return response


@app.get("/admin/api/state", dependencies=[Depends(require_admin)])
async def admin_state() -> dict:
    sims = S.store.list_sims()
    scenes = catalog.list_scenes()
    image_flags = await asyncio.gather(*(asyncio.to_thread(S.docker.image_exists, s.image) for s in scenes))
    containers = await asyncio.to_thread(S.docker.managed_containers)
    return {
        "limits": S.store.get_limits().as_dict(),
        "sims": [
            {**sim_info(s, admin=True), "containerState": containers.get(s.container_name)} for s in sims
        ],
        "scenes": [
            {
                **scene.public(),
                "image": scene.image,
                "imageAvailable": available,
                "buildable": scene.buildable,
                "build": S.docker.builds.get(scene.id),
            }
            for scene, available in zip(scenes, image_flags)
        ],
        "events": S.store.recent_events(150),
        "serverTime": time.time(),
    }


@app.get("/admin/api/stats", dependencies=[Depends(require_admin)])
async def admin_stats() -> dict:
    sims = S.store.list_sims()
    stats = await asyncio.gather(*(asyncio.to_thread(S.docker.stats, s.container_name) for s in sims))
    return {s.id: st for s, st in zip(sims, stats)}


@app.put("/admin/api/limits", dependencies=[Depends(require_admin)])
async def admin_set_limits(body: LimitsRequest) -> dict:
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    limits = S.store.set_limits(**changes)
    S.store.log("limits-changed", detail=", ".join(f"{k}={v}" for k, v in changes.items()))
    return limits.as_dict()


@app.delete("/admin/api/limits", dependencies=[Depends(require_admin)])
async def admin_reset_limits() -> dict:
    S.store.log("limits-reset")
    return S.store.reset_limits().as_dict()


@app.post("/admin/api/sims", dependencies=[Depends(require_admin)])
async def admin_create_sim(body: AdminCreateRequest) -> dict:
    scene = catalog.get_scene(body.sceneId)
    if scene is None:
        raise api_error(404, "Unknown scene.")
    if body.holdUntil is not None:
        if body.holdUntil <= time.time():
            raise api_error(400, "The hold must end in the future.")
    sim, token = await create_sim(
        scene,
        body.ownerName.strip() or "admin",
        "admin",
        enforce_client_limit=False,
        ignore_capacity=body.ignoreCapacity,
        hold_until=body.holdUntil,
        hold_reason=body.holdReason.strip() or "set up by admin",
    )
    return {"sim": sim_info(sim, admin=True), "key": token}


@app.post("/admin/api/sims/{sim_id}/release", dependencies=[Depends(require_admin)])
async def admin_release(sim_id: str) -> dict:
    get_sim_or_404(sim_id)
    await release(sim_id, "released by admin")
    return {"ok": True}


@app.post("/admin/api/sims/{sim_id}/restart", dependencies=[Depends(require_admin)])
async def admin_restart(sim_id: str) -> dict:
    sim = get_sim_or_404(sim_id)
    S.store.log("restarted", sim.id, "by admin")
    await restart_sim(sim)
    return {"ok": True}


@app.put("/admin/api/sims/{sim_id}/hold", dependencies=[Depends(require_admin)])
async def admin_set_hold(sim_id: str, body: AdminHoldRequest) -> dict:
    sim = get_sim_or_404(sim_id)
    if body.until is None:
        S.store.set_hold(sim.id, None, "")
        S.store.log("hold-cleared", sim.id, "by admin")
    else:
        # Admins aren't bound by max_hold_hours / max_held_sims.
        validate_hold(sim, body.until, body.reason or "set by admin", None)
        S.store.set_hold(sim.id, body.until, body.reason.strip() or "set by admin")
        S.store.log(
            "hold", sim.id,
            f"by admin until={fmt_utc(body.until)} reason={body.reason.strip()}",
        )
    S.store.touch(sim.id)
    return {"ok": True}


@app.post("/admin/api/sims/{sim_id}/reissue-key", dependencies=[Depends(require_admin)])
async def admin_reissue_key(sim_id: str) -> dict:
    """New key for a simulation (e.g. the owner lost theirs); the old key
    stops working immediately."""
    sim = get_sim_or_404(sim_id)
    token = auth.new_token()
    S.store.set_token_hash(sim.id, auth.hash_token(token))
    S.store.log("key-reissued", sim.id, "by admin")
    return {"key": token}


@app.get("/admin/api/sims/{sim_id}/logs", dependencies=[Depends(require_admin)])
async def admin_logs(sim_id: str) -> Response:
    sim = get_sim_or_404(sim_id)
    text = await asyncio.to_thread(S.docker.logs, sim.container_name)
    return Response(text, media_type="text/plain; charset=utf-8")


@app.post("/admin/api/scenes/{scene_id}/build", dependencies=[Depends(require_admin)])
async def admin_build_scene(scene_id: str) -> dict:
    scene = catalog.get_scene(scene_id)
    if scene is None or not scene.buildable:
        raise api_error(404, "Unknown scene, or it has no Dockerfile.")
    S.store.log("image-build", detail=f"scene={scene.id} image={scene.image}")
    # Fire and forget: builds can take minutes; the admin page polls status.
    asyncio.get_running_loop().run_in_executor(None, S.docker.build_image, scene, config.SCENES_DIR / scene.id)
    return {"ok": True}


# -- simulation proxy ------------------------------------------------------


@app.get("/sim/{sim_id}", include_in_schema=False)
async def sim_root_redirect(sim_id: str, request: Request) -> RedirectResponse:
    # Scene frontends use relative URLs, which only resolve under the
    # simulation's prefix when the page URL ends in a slash.
    target = f"/sim/{sim_id}/" + (f"?{request.url.query}" if request.url.query else "")
    return RedirectResponse(target, status_code=307)


@app.api_route(
    "/sim/{sim_id}/{path:path}",
    methods=["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"],
    include_in_schema=False,
)
async def sim_http(sim_id: str, path: str, request: Request) -> Response:
    sim = get_sim_or_404(sim_id)
    if sim.status != "running":
        return Response("Simulation is starting...", status_code=503, media_type="text/plain")
    if request.method not in READ_ONLY_METHODS:
        require_owner(request, sim)
        S.store.touch(sim.id)
    return await S.proxy.http(request, upstream_base(sim), path)


@app.websocket("/sim/{sim_id}/{path:path}")
async def sim_ws(websocket: WebSocket, sim_id: str, path: str) -> None:
    sim = S.store.get_sim(sim_id) if _SIM_ID.match(sim_id) else None
    if sim is None or sim.status != "running":
        await websocket.close(code=1013)
        return
    url = f"{upstream_base(sim, 'ws')}/{path}"
    if websocket.url.query:
        url += f"?{websocket.url.query}"
    await S.proxy.websocket(websocket, url, sim.id, owns(websocket, sim), lambda: S.store.touch(sim.id))


app.mount("/assets", StaticFiles(directory=str(config.STATIC_DIR)), name="assets")
