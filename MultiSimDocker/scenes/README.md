# Scenes

Every subfolder `scenes/<id>/` that contains a `scene.json` is a launchable
scene. The folder name is the scene id (lowercase letters, digits, `-`, `_`).

```
scenes/
  cube-drop/
    scene.json      # catalog entry (required)
    Dockerfile      # builds image msd-scene-cube-drop
    docs/screenshot.png
    backend/ static/ ...
```

## scene.json

```json
{
	"title": "Falling cube",
	"description": "Shown on the lobby card.",
	"thumbnail": "docs/screenshot.png",
	"port": 8000,
	"cpus": 1.0,
	"memory": "1g",
	"image": "optional-explicit-image:tag"
}
```

Everything except `title` is optional. `image` defaults to
`msd-scene-<id>`, which is what `build_scenes.bat` / `build_scenes.sh` and
the admin page's **Build image** button produce. `cpus` / `memory` default
to `MSD_SIM_CPUS` / `MSD_SIM_MEMORY`.

The catalog is re-read on every request: add a folder, build its image, and
it shows up in the lobby without restarting anything.

## The contract a scene container must follow

The orchestrator starts one container per claimed simulation and
reverse-proxies it at `/sim/<sim-id>/`. For that to work, a scene's web
app must:

1. **Serve HTTP (and optionally WebSockets) on `port`** (default 8000),
   bound to `0.0.0.0`. The simulation is "ready" as soon as `GET /` answers
   with a status below 500.
2. **Use relative URLs only**: `fetch("api/reset")`, not
   `fetch("/api/reset")`; for WebSockets,
   `new URL("ws/sim", location.href)` with the protocol swapped to
   `ws:`/`wss:`. The page is served at `/sim/<id>/`, so an absolute path
   would hit the orchestrator instead of the simulation.
3. **Keep GET requests read-only.** Watchers can do anything that's a
   `GET`/`HEAD`; every `POST`/`PUT`/`PATCH`/`DELETE` is rejected by the proxy
   unless it comes from the owner. Messages *from* a watcher's browser over
   a WebSocket are dropped; messages *to* it are delivered normally.
4. **Honour `?role=viewer`** (optional but recommended): the viewer page
   loads the scene with `?role=owner` or `?role=viewer`. In viewer mode,
   hide or disable the controls, since they'd fail anyway. Camera navigation
   is purely client-side, so watchers can still orbit and zoom freely.
5. **Broadcast state, don't assume one client.** Several browsers are
   connected at once (the owner in multiple tabs, plus watchers). Anything
   a watcher should see, such as pause state or changed parameters, must
   come from the server (broadcast it, or let watchers poll a GET endpoint).
6. **Not depend on persistent state.** A container is removed when its
   simulation is released and started fresh for the next claim. The
   `MSD_SIM_ID` environment variable holds the simulation's id if useful.

The proxy strips the orchestrator's cookies before forwarding, so the
scene never sees claim keys.

`cube-drop` is the reference implementation. Compared to
`../../TestSimInDocker`, only its frontend changed: relative URLs, a viewer
mode, and pause state that follows the server's broadcast.
