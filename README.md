# SOFA_CloSer

## [TestSimInDocker](TestSimInDocker/)

A [SOFA Framework](https://www.sofa-framework.org/) physics demo: a rigid
cube falls under gravity, bounces, and settles on a floor, simulated
headlessly and streamed live to a browser that renders it with three.js and
plots its energy over time. Ships as a single Docker image.

![Screenshot of the simulation: a cube bouncing on a floor, with live height and energy graphs and adjustable physics parameters](TestSimInDocker/docs/screenshot.png)

```bash
cd TestSimInDocker
run_docker.bat
```

See [TestSimInDocker/README.md](TestSimInDocker/README.md) for details.

## [MultiSim](MultiSim/)

A multi-user web platform on top of such scenes: visitors claim a simulation
from a limited pool (with a key to reclaim it), watch anyone else's running
simulation read-only, and can hold one for a later demo; idle simulations
return to the pool automatically. Includes a password-protected admin page.
Live at <https://closer.ram.eemcs.utwente.nl/sofa/>.

Simulations run as plain processes (Linux, LXC containers) or as Docker
containers, and each scene's original SOFA script still runs in the SOFA
desktop app:

```bash
cd MultiSim
./run.sh            # Linux, native processes
run-docker.bat      # Docker (or ./run-docker.sh)
```

See [MultiSim/README.md](MultiSim/README.md) for details.

## License

MIT (see [LICENSE](LICENSE)), for the code and scenes in this repository.

The simulations run on software that is not part of this repository and keeps its own license:
[SOFA](https://www.sofa-framework.org/) (LGPL-2.1) with the
[SoftRobots](https://github.com/SofaDefrost/SoftRobots) plugin (LGPL-3.0), and
[MuJoCo](https://github.com/google-deepmind/mujoco) (Apache-2.0); the Dockerfiles download SOFA, they don't contain it.
The pages load [three.js](https://threejs.org/) and [Chart.js](https://www.chartjs.org/) (both MIT) from a CDN.
A Docker image built from this repository contains SOFA and SoftRobots, so sharing such an image means passing on
their licenses (LGPL) as well.

## AI disclaimer

Claude Code was used to generate the whole simulation and most documentation.
