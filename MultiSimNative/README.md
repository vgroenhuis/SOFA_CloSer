# MultiSimNative

[MultiSimDocker](../MultiSimDocker/) without Docker: the same lobby, claim /
watch / hold / admin platform, but each simulation is a plain host process
running SOFA's own Python bindings. Use it where Docker isn't available or
wanted (for example inside an LXD container without nesting).

Differences from MultiSimDocker:

| | MultiSimDocker | MultiSimNative |
|---|---|---|
| Runtime | one container per simulation | one process per simulation on `127.0.0.1:<free port>` |
| SOFA | v26.06 image (v24.06 for cube-drop, double-pendulum) | one shared v26.06 binary release for all scenes |
| CPU / memory caps | Docker limits | per-simulation `systemd-run --user --scope` cgroup (`CPUQuota`, `MemoryMax`); off with a warning if unavailable |
| Isolation | separate filesystem + network | same user and filesystem. Fine for your own scenes, not for visitor-supplied code |
| Admin "Build image" | docker build | runs the scene's self-check (`sofaweb check`) |
| Orchestrator restart | sims keep running, re-adopted | same (`data/run/<name>.json` holds pid and port) |

The orchestrator (`orchestrator/`) is MultiSimDocker's, with `docker_mgr.py`
replaced by `native_mgr.py`. The scenes (`scenes/`) are unchanged apart from
dropping the Dockerfiles; `sofaweb` gained a `--host` option.

## Quick start (Linux x86-64, no root needed)

Requires [uv](https://docs.astral.sh/uv/), curl, and `systemd` user sessions
for resource caps (optional).

```bash
./install-runtime.sh     # downloads SOFA (~230 MB) and a Python 3.12 venv into ./runtime
cp .env.example .env     # set MSD_ADMIN_PASSWORD
./run.sh                 # lobby on http://127.0.0.1:8080/
```

## Running as a service

```bash
sudo bash deploy/setup.sh        # enables linger for the owner and installs multisimnative.service
```

Put a reverse proxy in front (see `deploy/apache-sofa.conf`; WebSockets on
`/sim/` must be forwarded) and set `MSD_TRUST_PROXY_HEADERS=true`.

## Scenes

A scene is a folder in `scenes/` with a `scene.json`, plus either
`sofaweb_scene.py` (run through sofaweb) or a FastAPI app in `backend/`. To
use another command, set `"command"` in `scene.json` (placeholders `{python}`
and `{port}`; the app must listen on `127.0.0.1:{port}`). Scene processes
get `MSD_SIM_ID` and `MSD_SCENE_DEFAULTS` in their environment. Everything else
is as described in [../MultiSimDocker/scenes/README.md](../MultiSimDocker/scenes/README.md).

## Configuration

See [.env.example](.env.example). Extra settings here: `MSD_RUNTIME_DIR`,
`MSD_SOFA_ROOT`, `MSD_PYTHON`, `MSD_USE_CGROUPS` (`auto` or `off`).

## AI disclaimer

Claude Code was used to write the native process manager and this document.
