# MultiSim

A small web platform for running [SOFA](https://www.sofa-framework.org/)
simulations for many visitors at once, from a pool of limited size:

- **Claim**: a visitor picks a scene in the lobby and gets their own live
  simulation, plus a **key** (`XXXX-XXXX-XXXX-XXXX`) that lets them take
  control again from any browser. Visitors usually try scenes one after
  another, so starting a new simulation releases the ones that browser
  already owns, unless they're on hold.
- **Watch**: every running simulation is public. Anyone can open it
  read-only and navigate the 3D scene themselves, but only the key holder
  can change it.
- **Idle return**: a simulation that nobody has used for a while (15 min by
  default) goes back to the pool automatically.
- **Hold**: an owner can reserve their simulation for longer, e.g. for a
  live demo later in the day, by giving a duration or end time plus a reason
  (up to 12 h by default; only 1 simulation can be on hold at a time).
  A held simulation also survives its owner starting another one.
- **Scene visibility**: the admin sets each scene to **public**,
  **private** or **admin-only**. Private scenes and their running
  simulations are hidden from the public lobby; a separate **access
  password** (e.g. for team members) unlocks them in a browser, without
  giving access to the admin page. Admin-only scenes are for the admin
  alone, e.g. while a scene is being prepared.
- **Admin**: a password-protected page to see everything, release or
  restart simulations, set or clear holds, issue a new key, read simulation
  logs, change limits at runtime, start simulations, and build or check
  scenes.

Each simulation runs either as a plain process (**native** backend) or in
its own Docker container (**docker** backend); everything else is the same.

## Ways to run it

| | How | Backend | Needs |
|---|---|---|---|
| [Linux, no containers](#linux-native) | `./run.sh` | native | Linux x86-64, [uv](https://docs.astral.sh/uv/), curl |
| [Docker](#docker) | `run-docker.bat` / `./run-docker.sh` | docker | Docker (Docker Desktop on Windows/macOS) |
| [LXC / LXD / Incus container](#lxc--lxd--incus) | `./run.sh` inside the container | native | an Ubuntu container |
| [SOFA desktop app](#a-scene-in-the-sofa-desktop-app) | `runSofa scenes/<id>/original/<Scene>.py` | — | SOFA v26.06 with SoftRobots |

The first three run the whole platform. The last runs a single scene in
SOFA's own GUI, without the web parts.

### Linux (native)

```bash
./install-runtime.sh     # downloads SOFA (~230 MB) and a Python 3.12 venv into ./runtime
cp .env.example .env     # set MSD_ADMIN_PASSWORD
./run.sh                 # lobby on http://127.0.0.1:8080/, admin on /admin
```

No root needed. `run.sh` runs `install-runtime.sh` itself if `runtime/` is
missing.

### Docker

```bash
run-docker.bat           # or ./run-docker.sh on Linux/macOS
```

On first run this copies `.env.example` to `.env`. **Set
`MSD_ADMIN_PASSWORD` in `.env`**, then run the script again. It builds all
scene images (`build_scenes.bat` / `build_scenes.sh`) and starts the
orchestrator with `docker compose`:

- Lobby: <http://127.0.0.1:8080/>
- Admin: <http://127.0.0.1:8080/admin>

On Docker Desktop every visitor appears to come from the same address, so
set `MSD_MAX_CLAIMS_PER_CLIENT=0` there (see [Deployment notes](#deployment-notes)).

### LXC / LXD / Incus

Inside a system container the native backend is the simplest choice: no
Docker-in-container nesting is needed. This is how closer.ram.eemcs.utwente.nl
runs it. For example with LXD (`incus` works the same):

```bash
lxc launch ubuntu:24.04 multisim
lxc exec multisim -- su - ubuntu -c '
  curl -LsSf https://astral.sh/uv/install.sh | sh &&
  git clone https://github.com/vgroenhuis/SOFA_CloSer.git &&
  cd SOFA_CloSer/MultiSim && ./install-runtime.sh && cp .env.example .env'
lxc exec multisim -- su - ubuntu -c 'nano SOFA_CloSer/MultiSim/.env'   # set MSD_ADMIN_PASSWORD
lxc exec multisim -- su - ubuntu -c 'cd SOFA_CloSer/MultiSim && ./run.sh'
# from the host, in another terminal: forward port 8080 to the container
lxc config device add multisim web proxy listen=tcp:0.0.0.0:8080 connect=tcp:127.0.0.1:8080
```

To keep it running, install it as a [service](#running-as-a-service) inside
the container.

### A scene in the SOFA desktop app

Each sofaweb scene keeps its original SOFA script in
`scenes/<id>/original/`, unmodified, so it still runs in SOFA's GUI
(`runSofa`), e.g.:

```bash
cd scenes/hollow-cube/original
runSofa -l SofaPython3 HollowCubeInflation.py
```

Use SOFA v26.06 with the SoftRobots plugin (the binary release includes
it). `better-finger/original/` also has Windows launchers (`run_sofa.bat`,
`BetterFinger_run_imgui.bat`; set the SOFA path in `sofa_env.bat`). Each
script's `params.json` holds its parameters; `params_editor.py` edits them.

## Serving it under a path

Behind a reverse proxy the platform can live below the site root, e.g. at
`https://host/sofa/`:

1. Set `MSD_BASE_PATH=/sofa/` and `MSD_TRUST_PROXY_HEADERS=true` in `.env`.
2. Have the proxy strip the prefix and forward WebSocket upgrades; see
   [deploy/apache-sofa.conf](deploy/apache-sofa.conf) for Apache.

The pages use relative URLs and get a `<base href>` from `MSD_BASE_PATH`,
and cookies are scoped to it, so other sites on the same host never see
them. To have the lobby link back to the site's home page, set
`MSD_HOME_URL` (and `MSD_HOME_TITLE`), or let the proxy send
`X-Site-Home-Url` / `X-Site-Home-Title` headers.

## Running as a service

Native backend on a Linux host or container:

```bash
sudo bash deploy/setup.sh        # enables linger for the owner and installs multisimnative.service
```

The service runs `run.sh` as the checkout's owner. Linger gives it a
systemd user manager, which the per-simulation CPU/memory caps need. With
Docker, `docker compose` restarts the orchestrator itself
(`restart: unless-stopped`).

## Backends

| | native (`MSD_BACKEND=native`) | docker (`MSD_BACKEND=docker`) |
|---|---|---|
| A simulation is | a process on `127.0.0.1:<free port>` | a container `msd-sim-<id>` on the `msd-net` network |
| SOFA | one shared v26.06 binary release in `runtime/` | per-scene images: v26.06 (`msd-sofa-base`), v24.06 for cube-drop and double-pendulum |
| CPU / memory caps | per-simulation `systemd-run --user --scope` cgroup (`CPUQuota`, `MemoryMax`); off with a warning if unavailable | Docker limits |
| Isolation | same user and filesystem: fine for your own scenes, not for visitor-supplied code | separate filesystem and network |
| Admin "Build" button | runs the scene's self-check (`sofaweb check`) | `docker build` of the scene image |
| Orchestrator restart | simulations keep running and are re-adopted (`data/run/<name>.json` holds pid and port) | same, via container labels |

`run.sh` uses the native backend; `docker-compose.yml` sets
`MSD_BACKEND=docker`.

## Architecture

```
 browser ──HTTP/WS──▶ orchestrator (FastAPI, port 8080)
                        │  /               lobby
                        │  /view/<id>      viewer page (top bar + iframe)
                        │  /admin          admin page
                        │  /api/...        claim / hold / release / reclaim
                        │  /sim/<id>/...   reverse proxy ──▶ the simulation
                        │
                        └─ starts/stops simulations: processes (native) or
                           containers via the Docker socket (docker)
```

(All paths are relative to `MSD_BASE_PATH`.)

- **orchestrator/**: one Python process (FastAPI + SQLite).
  - `app/main.py`: routes, claim logic, reaper loop.
  - `app/proxy.py`: HTTP/WebSocket reverse proxy and the owner/watcher rule.
  - `app/native_mgr.py`, `app/docker_mgr.py`: simulation lifecycle for each backend.
  - `app/store.py`: SQLite state: active sims, limits, event log.
  - `app/auth.py`: keys, admin sessions, rate limiting.
  - `app/catalog.py`: the scene catalog.
  - `static/`: lobby, viewer and admin pages (plain JS, no build step).
- **scenes/**: one folder per scene. See [scenes/README.md](scenes/README.md)
  for how to add one and what contract its web app must follow. Currently:
  - `cube-drop`: rigid cube bouncing on a floor (from `../TestSimInDocker`).
  - `double-pendulum`: chaotic double pendulum with live parameters (from
    `verilogscripts/SOFA/DoublePendulum`).
  - `rod-with-balls`, `asymmetric-square-tube`, `hollow-cube`,
    `pneunet-finger`: the original scripts from `verilogscripts/SOFA/`,
    run unmodified through sofaweb.
  - `better-finger`: the PneuNet chamber with its channel far-end planes
    held rigid by a hard constraint (a new script, derived from
    PneuNetFinger.py, also runnable in the SOFA desktop GUI).
- **scenes/_base/**: **sofaweb**, a runtime that runs an existing SOFA scene
  script headlessly and streams its visual models to a generic three.js
  viewer, with parameter panel, charts, live controls and console; plus the
  `msd-sofa-base` image it runs in under Docker. Adding another script from
  `verilogscripts/SOFA/` is mostly writing a short config file; see
  [scenes/README.md](scenes/README.md).
- **deploy/**: systemd unit and setup script (native), and an Apache
  example for serving under `/sofa/`.

### Ownership and keys

- A claim creates a random 80-bit key. Only its SHA-256 hash is stored.
- The key is set as an HttpOnly cookie for that simulation, so the
  claiming browser is recognised automatically; the **Key** button in the
  viewer's toolbar shows the key when it's needed elsewhere.
- **Reclaiming**: enter the key in the lobby ("Have a key?") or on the
  viewer page ("I have the key"), or open a reclaim link
  `https://host/sofa/#reclaim=KEY`. The key is in the URL fragment, so it
  never reaches server logs.
- Scripts can send the key as an `X-MSD-Token` header instead of a cookie.
- If a key is lost, the admin can issue a new one. The old key stops
  working immediately.

### Public, private and admin-only scenes

The admin page's scene list has a visibility dropdown per scene (a
`scene.json` can set the default with `"visibility": "public" | "private" |
"admin"`; the admin page's choice wins):

| Visibility | Shown, startable and watchable for |
|---|---|
| public | everyone |
| private | browsers unlocked with the access password, and the admin |
| admin only | the admin |

"Shown" covers the lobby's scene card and thumbnail, its running
simulations, their pages and live streams; anyone else gets "not found".
A simulation's owner (holding its key) always keeps their own simulation.
The access password (`MSD_ACCESS_PASSWORD`) is entered in the lobby's
**Private simulations** form and stays valid for
`MSD_ACCESS_SESSION_HOURS` (default a week), until **Lock**.
`MSD_ACCESS_HINT` adds a note under that form, e.g. where to find the
password.

Without `MSD_ACCESS_PASSWORD`, only the admin can use private scenes. Every
simulation still takes a slot from the shared pool, so the lobby's slot
count includes ones a visitor can't see. Making a running scene private
(or admin-only) doesn't disconnect people already watching; their next
reload is refused.

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
- releases simulations that crashed or didn't start within
  `MSD_START_TIMEOUT_SECONDS`;
- removes orphaned simulations, e.g. after an orchestrator restart.

Every release, with its reason, goes into the event log shown on the admin
page. A visitor who opens an ended simulation sees why it ended.

While logged in to the admin page, a simulation's Display, Lighting and
Parameters panels (sofaweb scenes) also show **Set as default**. It saves
that panel's state as the scene's default for every simulation started
afterwards, stored in the data directory. The admin page's scene list shows
which scenes have saved defaults, with a **Clear** button
([details](scenes/README.md#adding-an-existing-sofa-scene-script-sofaweb)).

### Limits

Defaults come from `.env` (see [.env.example](.env.example)). They can be
changed live on the admin page. Those changes are stored in the data
directory and override `.env` until you click **Reset to server defaults**.

| Setting | Default | Meaning |
|---|---|---|
| `MSD_MAX_SIMS` | 3 | Simultaneous simulations |
| `MSD_IDLE_TIMEOUT_MINUTES` | 15 | Minutes without owner activity before release |
| `MSD_MAX_HOLD_HOURS` | 12 | Longest hold a visitor can set |
| `MSD_MAX_HELD_SIMS` | 1 | Simultaneous holds (so holds can't take every slot) |
| `MSD_MAX_CLAIMS_PER_CLIENT` | 1 | Simulations per client IP not on hold (0 = unlimited); matters for other browsers at the same address, since a browser's own un-held simulation is replaced anyway |
| `MSD_SIM_CPUS` / `MSD_SIM_MEMORY` | 1.0 / 2g | Per-simulation resource caps |

Admins aren't bound by the hold limits. They can also start a simulation
beyond `MAX_SIMS`.

Other settings: `MSD_ACCESS_PASSWORD`, `MSD_ACCESS_SESSION_HOURS`, `MSD_ACCESS_HINT`,
`MSD_BACKEND`, `MSD_BASE_PATH`, `MSD_PORT`,
`MSD_TRUST_PROXY_HEADERS`; for native, `MSD_RUNTIME_DIR`, `MSD_SOFA_ROOT`,
`MSD_PYTHON`, `MSD_USE_CGROUPS` (`auto` or `off`).

## Deployment notes

- **Client IP.** On Docker Desktop (Windows/macOS), every visitor appears
  to come from the same Docker gateway IP. With
  `MSD_MAX_CLAIMS_PER_CLIENT=1`, only one visitor in total can hold a
  simulation there. For local testing, set it to `0`. Behind a reverse
  proxy, set `MSD_TRUST_PROXY_HEADERS=true` so `X-Forwarded-For` is used.
- **HTTPS.** Put a TLS-terminating reverse proxy (Apache, Caddy, nginx,
  Traefik) in front and set `MSD_TRUST_PROXY_HEADERS=true`, so cookies get
  the `Secure` flag. It must forward WebSocket upgrades on `sim/`.
- **Docker socket.** The docker backend mounts `/var/run/docker.sock`,
  which is root-equivalent on the host. Keep the orchestrator image
  trusted, and keep the admin password strong.
- **State** lives in `data/` (native) or the `msd-data` volume (docker):
  SQLite database plus the admin-session signing key. Running simulations
  survive an orchestrator restart and are picked up again.
- **Adding a scene**: add `scenes/<id>/` with a `scene.json` (and a
  Dockerfile for docker), then click **Build** on the admin page (docker:
  or run `build_scenes.bat <id>`). No restart needed. Under docker, sofaweb
  scenes build `FROM msd-sofa-base`, which `build_scenes` builds first; the
  admin page's button only builds the scene itself, so run the script once
  after changing `scenes/_base`.
- **Disk space**: under docker, `cube-drop` and `double-pendulum` still use
  the older SOFA v24.06 image; the sofaweb scenes use v26.06-full. Both
  base images are a few GB each. The native runtime is about 600 MB.

## Possible extensions

- A waiting queue ("notify me when a slot frees up") instead of only showing
  the earliest expected free time.
- Advance bookings (reserving a slot for a future time without running a
  simulation until then). The current hold keeps a running simulation
  alive.
- Pausing held simulations while nobody is watching, to save CPU.

## Author

Vincent Groenhuis

## AI disclaimer

Claude Code was used to generate the orchestrator, sofaweb, the web
versions of the scenes, the native backend, and most documentation. The
scene scripts in `scenes/*/original/` are the unmodified originals from
`verilogscripts/SOFA/`.
