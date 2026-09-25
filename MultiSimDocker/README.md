# MultiSimDocker

A small web platform for running [SOFA](https://www.sofa-framework.org/)
simulations for many visitors at once. Each simulation runs in its own
Docker container, drawn from a pool of limited size:

- **Claim**: a visitor picks a scene in the lobby and gets their own live
  simulation, plus a **key** (`XXXX-XXXX-XXXX-XXXX`) that lets them take
  control again from any browser.
- **Watch**: every running simulation is public. Anyone can open it
  read-only and navigate the 3D scene themselves, but only the key holder
  can change it.
- **Idle return**: a simulation that nobody has used for a while (15 min by
  default) goes back to the pool automatically.
- **Hold**: an owner can reserve their simulation for longer, e.g. for a
  live demo later in the day, by giving a duration or end time plus a reason
  (up to 12 h by default; only 1 simulation can be on hold at a time).
- **Admin**: a password-protected page to see everything, release or
  restart simulations, set or clear holds, issue a new key, read container
  logs, change limits at runtime, start simulations, and build scene images.

## Quick start

Requires Docker (Docker Desktop on Windows).

```bash
run.bat
```

(or `./run.sh` on Linux/macOS). On first run this copies `.env.example` to
`.env`. **Set `MSD_ADMIN_PASSWORD` in `.env`**, then run the script again.
It builds all scene images and starts the orchestrator:

- Lobby: <http://127.0.0.1:8080/>
- Admin: <http://127.0.0.1:8080/admin>

## Architecture

```
 browser ──HTTP/WS──▶ orchestrator (FastAPI, port 8080)
                        │  /               lobby
                        │  /view/<id>      viewer page (top bar + iframe)
                        │  /admin          admin page
                        │  /api/...        claim / hold / release / reclaim
                        │  /sim/<id>/...   reverse proxy ──▶ msd-sim-<id>:8000
                        │
                        └─ Docker socket: starts/stops msd-sim-<id> containers
                           on the private "msd-net" network (no host ports)
```

- **orchestrator/**: one Python process (FastAPI + Docker SDK + SQLite in
  a named volume).
  - `app/main.py`: routes, claim logic, reaper loop.
  - `app/proxy.py`: HTTP/WebSocket reverse proxy and the owner/watcher rule.
  - `app/docker_mgr.py`: container lifecycle and image builds.
  - `app/store.py`: SQLite state: active sims, limits, event log.
  - `app/auth.py`: keys, admin sessions, rate limiting.
  - `app/catalog.py`: the scene catalog.
  - `static/`: lobby, viewer and admin pages (plain JS, no build step).
- **scenes/**: one folder per scene, each with its own Dockerfile. See
  [scenes/README.md](scenes/README.md) for how to add one and what contract
  its web app must follow. Currently:
  - `cube-drop`: rigid cube bouncing on a floor (from `../TestSimInDocker`).
  - `double-pendulum`: chaotic double pendulum with live parameters (from
    `verilogscripts/SOFA/DoublePendulum`).
  - `rod-with-balls`, `asymmetric-square-tube`, `hollow-cube`,
    `pneunet-finger`: the original scripts from `verilogscripts/SOFA/`,
    run unmodified through sofaweb.
- **scenes/_base/**: the shared `msd-sofa-base` image (SOFA v26.06 with
  SoftRobots) and **sofaweb**, a runtime that runs an existing SOFA scene
  script headlessly and streams its visual models to a generic three.js
  viewer, with parameter panel, charts, live controls and console. Adding
  another script from `verilogscripts/SOFA/` is mostly writing a short
  config file; see [scenes/README.md](scenes/README.md).

### Ownership and keys

- A claim creates a random 80-bit key. Only its SHA-256 hash is stored.
- The key is set as an HttpOnly cookie for that simulation, so the
  claiming browser is recognised automatically; the **Key** button in the
  viewer's toolbar shows the key when it's needed elsewhere.
- **Reclaiming**: enter the key in the lobby ("Have a key?") or on the
  viewer page ("I have the key"), or open a reclaim link
  `https://host/#reclaim=KEY`. The key is in the URL fragment, so it never
  reaches server logs.
- Scripts can send the key as an `X-MSD-Token` header instead of a cookie.
- If a key is lost, the admin can issue a new one. The old key stops
  working immediately.

### Watchers vs. owner

The proxy enforces this for every scene: `GET`/`HEAD` and server→browser
WebSocket traffic are open to all; any other HTTP method and browser→server
WebSocket messages require the owner's key. Scenes additionally get
`?role=viewer` so they can hide their controls.

### Idle detection and holds

A simulation's expiry time is `max(last activity + idle timeout, hold end)`.
The following count as activity:

- heartbeats from the owner's viewer page, sent only while the tab is
  visible and the owner has used the mouse or keyboard (including inside
  the simulation) in the last 2 minutes;
- control requests the owner sends to the simulation;
- reclaiming the simulation.

An open but untouched tab therefore doesn't block a slot forever. Three
minutes before expiry, the owner sees a countdown with an **I'm still here**
button.

A reaper runs every 3 s. It:

- marks simulations ready once they answer;
- releases expired ones;
- releases simulations whose container crashed or didn't start within
  `MSD_START_TIMEOUT_SECONDS`;
- removes orphaned `msd-sim-*` containers, e.g. after an orchestrator
  restart.

Every release, with its reason, goes into the event log shown on the admin
page. A visitor who opens an ended simulation sees why it ended.

### Limits

Defaults come from `.env` (see [.env.example](.env.example)). They can be
changed live on the admin page. Those changes are stored in the data volume
and override `.env` until you click **Reset to server defaults**.

| Setting | Default | Meaning |
|---|---|---|
| `MSD_MAX_SIMS` | 3 | Simultaneous simulations (containers) |
| `MSD_IDLE_TIMEOUT_MINUTES` | 15 | Minutes without owner activity before release |
| `MSD_MAX_HOLD_HOURS` | 12 | Longest hold a visitor can set |
| `MSD_MAX_HELD_SIMS` | 1 | Simultaneous holds (so holds can't take every slot) |
| `MSD_MAX_CLAIMS_PER_CLIENT` | 1 | Simulations per client IP (0 = unlimited) |
| `MSD_SIM_CPUS` / `MSD_SIM_MEMORY` | 1.0 / 2g | Per-container resource caps |

Admins aren't bound by the hold limits. They can also start a simulation
beyond `MAX_SIMS`.

## Deployment notes

- **Client IP and Docker Desktop.** On Docker Desktop (Windows/macOS), every
  visitor appears to come from the same Docker gateway IP. With
  `MSD_MAX_CLAIMS_PER_CLIENT=1`, only one visitor in total can hold a
  simulation there. For local testing, set it to `0`. On a Linux host, real
  client IPs are visible. Behind a reverse proxy, set
  `MSD_TRUST_PROXY_HEADERS=true` so `X-Forwarded-For` is used.
- **HTTPS.** Put a TLS-terminating reverse proxy (Caddy, nginx, Traefik) in
  front and set `MSD_TRUST_PROXY_HEADERS=true`, so cookies get the `Secure`
  flag. It must forward WebSocket upgrades on `/sim/`.
- **Docker socket.** The orchestrator mounts `/var/run/docker.sock`, which
  is root-equivalent on the host. Keep the orchestrator image trusted, and
  keep the admin password strong.
- **State** lives in the `msd-data` volume: SQLite database plus the
  admin-session signing key. Running simulations survive an orchestrator
  restart and are picked up again.
- **Adding a scene**: add `scenes/<id>/` with a Dockerfile and `scene.json`,
  then run `build_scenes.bat <id>` or click **Build image** on the admin
  page. No restart needed. sofaweb scenes build `FROM msd-sofa-base`, which
  `build_scenes` builds first; the admin page's button only builds the
  scene itself, so run the script once after changing `scenes/_base`.
- **Disk space**: `cube-drop` and `double-pendulum` still use the older
  SOFA v24.06 image; the sofaweb scenes use v26.06-full. Both base images
  are a few GB each.

## Possible extensions

- A waiting queue ("notify me when a slot frees up") instead of only showing
  the earliest expected free time.
- Advance bookings (reserving a slot for a future time without running a
  container until then). The current hold keeps a running simulation
  alive.
- Pausing held simulations while nobody is watching, to save CPU.

## Author

Vincent Groenhuis

## AI disclaimer

Claude Code (Opus 5.5) was used to generate the orchestrator, sofaweb, the
web versions of the scenes, and most documentation. The scene scripts in
`scenes/*/original/` are the unmodified originals from `verilogscripts/SOFA/`.
