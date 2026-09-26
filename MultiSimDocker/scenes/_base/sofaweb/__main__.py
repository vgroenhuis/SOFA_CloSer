"""Entry points:

    python3 -m sofaweb serve  sofaweb_scene.py   # the web app on port 8000
    python3 -m sofaweb check  sofaweb_scene.py   # build + step the scene (image build sanity check)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
import time
from pathlib import Path


def check(config_path: Path, seconds: float) -> int:
    import Sofa.Simulation

    from .app import load_config
    from .runner import Runner

    config = load_config(config_path)
    runner = Runner(config, config_path.parent, on_message=lambda message: None)
    runner._safe_build(runner.params)
    if runner.status != "running":
        print(runner.error, file=sys.stderr)
        print("\n".join(line["text"] for line in runner.console.since(0)[-30:]), file=sys.stderr)
        return 1
    root = runner._root
    dt = float(root.dt.value)
    steps = max(1, int(round(seconds / dt)))
    start = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            Sofa.Simulation.animate(root, dt)
    elapsed = time.monotonic() - start
    runner._sim_time = steps * dt
    scalars = runner._probe()
    print(
        f"{config.title}: {len(runner._models)} visual models, "
        f"{sum(m.vertex_count for m in runner._models)} vertices; "
        f"{steps} steps of {dt:g}s in {elapsed:.2f}s (real-time factor {steps * dt / elapsed:.2f}); "
        f"probes: {', '.join(f'{k}={v:.4g}' for k, v in scalars.items())}"
    )
    runner._unload()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="sofaweb")
    parser.add_argument("command", choices=["serve", "check"])
    parser.add_argument("config", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seconds", type=float, default=0.5, help="simulated time for `check`")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()

    if args.command == "check":
        return check(config_path, args.seconds)

    import uvicorn

    from .app import create_app

    uvicorn.run(create_app(config_path), host="0.0.0.0", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
