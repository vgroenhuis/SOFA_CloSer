# SOFA Cube Simulation (Web Viewer)

A minimal [SOFA Framework](https://www.sofa-framework.org/) demo: a rigid cube falls
under gravity onto a floor. The physics runs headlessly in Python
(SofaPython3), and a browser page renders the scene live with three.js,
receiving the cube's position/orientation over a WebSocket.

![Screenshot of the simulation: a cube bouncing on a floor, with live height and energy graphs and adjustable physics parameters](docs/screenshot.png)

## Run

```bash
run_docker.bat
```

This builds the `sofa-cube-web` image on first run (subsequent runs just
start the existing container), then open <http://127.0.0.1:8001/>.

(Port 8001 on the host, mapped to port 8000 inside the container -- chosen
because port 8000 is already used by the `ring-web` container from the
CAD_Generators project.)

The cube drops, bounces/settles on the floor, then automatically resets and
drops again after a short pause. Click **Reset** to restart it manually at
any time.

## How it works

- `backend/simulation.py` builds the SOFA scene: a rigid `MechanicalObject`
  (`Rigid3d`) for the cube, mapped via `RigidMapping` to its 8 corner points,
  with a `PlaneForceField` acting as the floor. No GUI/OpenGL components are
  used -- SOFA never opens a window, it just steps the physics.
- A background thread steps the simulation and calls a callback with the
  cube's `[x, y, z]` position and `[qx, qy, qz, qw]` quaternion each frame.
- `backend/main.py` is a FastAPI app that relays those frames to any
  connected browser over `/ws/sim`, and exposes `POST /api/reset`.
- `static/index.html` + `static/main.js` render a three.js scene (loaded
  from a CDN) driven entirely by incoming WebSocket frames -- the browser
  does no physics of its own.

## Run without Docker

Requires a local SOFA build with SofaPython3 on `PYTHONPATH` (see the
[SOFA build docs](https://sofa-framework.github.io/doc/getting-started/build/linux/)) --
this is why the Docker image builds on top of the official
`sofaframework/sofa_nightly_ubuntu` prebuilt image instead.

```bash
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## Author

Vincent Groenhuis

## AI disclaimer

Claude Code (Sonnet 5) was used to generate the whole simulation and most documentation.
