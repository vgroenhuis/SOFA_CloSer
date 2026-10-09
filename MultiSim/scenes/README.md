# Scenes

Every subfolder `scenes/<id>/` that contains a `scene.json` is a launchable
scene. The folder name is the scene id (lowercase letters, digits, `-`, `_`).
`_base/` is not a scene: it holds the shared `msd-sofa-base` image and the
sofaweb runtime (see [Adding an existing SOFA scene script](#adding-an-existing-sofa-scene-script-sofaweb)).

A scene runs under either backend (see [../README.md](../README.md#backends)).
The **docker** backend builds and runs its `Dockerfile`. The **native**
backend runs `sofaweb_scene.py` through the shared SOFA runtime, or a
custom app's `backend/` with uvicorn; to use another command, set
`"command"` in `scene.json` (placeholders `{python}` and `{port}`; the app
must listen on `127.0.0.1:{port}`). Either way the scene gets
`MSD_SIM_ID` and `MSD_SCENE_DEFAULTS` in its environment, and the contract
below applies ("container" means the simulation's process there).

There are two ways to make a scene:

- **Custom web app** (`cube-drop`, `double-pendulum`): your own backend and
  frontend, following the contract below. Most work, full control.
- **sofaweb** (`rod-with-balls`, `asymmetric-square-tube`, `hollow-cube`,
  `pneunet-finger`, `better-finger`): drop in an existing SOFA scene script unmodified, plus a
  short config file. The generic viewer draws its visual models, camera and
  lights; parameters, charts and live controls come from the config.

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
6. **Start paused, with media-style controls.** A new simulation waits at
   its start until the owner presses play. Use a play/pause toggle plus a
   stop button (pause, then back to the start; it stays paused there). The
   existing scenes share the same inline SVG icons. Because nothing is
   broadcast while paused, send the latest state to each browser as it
   connects.
7. **Not depend on persistent state.** A container is removed when its
   simulation is released and started fresh for the next claim. The
   `MSD_SIM_ID` environment variable holds the simulation's id if useful.
   Defaults an admin saved for the scene arrive as JSON in
   `MSD_SCENE_DEFAULTS` (optional to support; see below).

The proxy strips the orchestrator's cookies before forwarding, so the
scene never sees claim keys.

`cube-drop` is the reference implementation. Compared to
`../../TestSimInDocker`, only its frontend changed: relative URLs, a viewer
mode, and pause state that follows the server's broadcast.

`double-pendulum` is a port of `verilogscripts/SOFA/DoublePendulum`. Its
physics (the RK4 controller and the equations of motion) is unchanged.
SOFA's OpenGL visuals, the Tk parameter editor and the matplotlib live-plot
process are replaced by the browser: a three.js view in the original's
cartoon style, Chart.js graphs, and a parameter panel that restarts the
swing. The image build checks that 1 s of the chaotic default start
conserves energy to better than 0.1%.

## Adding an existing SOFA scene script (sofaweb)

sofaweb (in `_base/sofaweb/`, baked into the `msd-sofa-base` image, SOFA
v26.06 with SoftRobots) runs an original `createScene(rootNode)` script as
is:

- While `createScene` runs, `Node.addObject` is patched: `OglModel` becomes
  SOFA's GL-free `VisualModelImpl` (mappings and controllers that read or
  write its `position` keep working), lights become recorded stubs, and
  `Sofa.GL.*` plugins are skipped. Camera and background stay real.
- Every frame, the positions of all visual models are streamed to the
  browser as binary WebSocket messages: only models that moved, with
  coincident (flat-shading) vertices sent once. The browser draws them with
  the scene's own camera, colors and lights; lights a controller moves
  (e.g. a rotating key light) are streamed too.
- Parameters: `params.json` next to the script is rewritten and the script
  re-imported for every rebuild, exactly like the SOFA GUI's Reload. Field
  labels, units and display scales are read from the script's Tk
  `params_editor.py` (`_FIELDS` / `_VECTOR_FIELDS` / `_BOOL_FIELDS`), which
  is parsed, never run.
- What the script prints goes to the page's Console panel, not the
  container log.
- The simulation never runs faster than real time. It stops at a
  configurable end time (so an idle held slot doesn't burn a CPU core), and
  ends a run that diverges (NaN / runaway coordinates) with a message.

To add one:

```
scenes/<id>/
  original/          # the scene script, its params.json and params_editor.py, unmodified
  sofaweb_scene.py   # CONFIG = SceneConfig(...), see _base/sofaweb/config.py
  Dockerfile         # copy one of the existing sofaweb scenes' Dockerfile as is
  scene.json
  docs/thumbnail.jpg
```

In `sofaweb_scene.py`, at minimum set `title` and `script`. Useful extras
(all in `SceneConfig`, see the four existing configs):

| Field | Purpose |
|---|---|
| `env` | e.g. `{"MYSCENE_NO_EDITOR": "1"}` so the script doesn't try to open its Tk editor |
| `forced_params` / `hidden_params` | force e.g. `console_log_enabled: False`; hide `theme` and GL-only debug flags |
| `param_limits` | clamp user input, above all mesh resolutions, so nobody can take down the container |
| `sections`, `labels` | group and relabel the parameter panel |
| `live_params` | `key -> fn(root, value)`: applied to the running scene without a rebuild (e.g. a controller's Data field) |
| `probes`, `charts`, `readouts` | `fn(root, module) -> {name: value}` sampled every frame, plotted and shown |
| `stop_at`, `extend_by`, `end_condition`, `auto_restart` | when a run ends and what happens then |
| `view` | `"3d"` (orbit) or `"2d"` (pan/zoom, drawn in creation order for flat scenes) |
| `display` | controls in the viewer's Display panel, applied in the browser without a rebuild and per viewer: opacity sliders, colors, show/hide, and mirrored copies across symmetry planes (fixed, or following a moving plane such as the fingers' far-end plane via the visual model that marks it; a moving one can repeat as a chain, e.g. a finger of N chambers), each tied to a params.json key and a pattern for the visual models' SOFA paths, plus `femRegion` switches that hide FEM elements in a region (used for the symmetry buffer blocks); a control with `requires` is grayed out while that parameter is off. Every scene also gets, automatically: a surface wireframe overlay and (when the scene has a tetrahedral FEM mesh, which is found and streamed for this) FEM elements, element edges and nodes, each with a switch and opacity slider; the elements are colored SOFA style, in one lit blue, or by strain, unlit or lit (the default; Green-Lagrange, relative to the mesh's rest shape, on a scale up to the largest strain seen); and a Lighting panel with each light's brightness and each directional light's yaw and pitch (around the scene's up axis), following the scene's own lights until changed. The Display panel also holds the camera pose (position, look-at point, zoom), editable and remembered per viewer. Chart and readout panels collapse to their title on a click. Opacity rows get a show/hide switch in front. |

**Saved defaults.** In a browser that's logged in to the admin page, the
Display, Lighting and Parameters panels get a **Set as default** button. It
stores that panel's current state (for Parameters: the running simulation's
values) as the scene's default in the orchestrator's database, so it
survives restarts. Every simulation of the scene started afterwards gets
these defaults:
- the parameters are its starting values and what **Defaults** returns to;
- display and lighting are where a viewer starts, unless they have
  settings of their own saved in their browser, and where **Reset** returns to.

The orchestrator hands them to a container as `MSD_SCENE_DEFAULTS` (JSON
`{"params", "display", "lighting"}`) and pushes changes to the scene's
running containers via `POST /api/defaults` (a path its proxy refuses to
browsers). The admin page shows which scenes have saved defaults, with a
**Clear** button to go back to the scene's own defaults.

The Dockerfile's `python3 -m sofaweb check sofaweb_scene.py` step builds
the scene and steps it for 0.2 s during `docker build`, printing the
real-time factor and probe values, so a broken scene fails the build rather
than the first visitor.

Limitations: only what SOFA's visual models hold is drawn (triangles,
quads, `keepLines` edges, one color per model). GL-only drawing such as
`VisualStyle`'s `showForceFields` or `drawBoxes`, textures and custom
shaders is not reproduced; hide those parameters or force them off.
