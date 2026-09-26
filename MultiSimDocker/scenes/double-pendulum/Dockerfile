FROM sofaframework/sofa_nightly_ubuntu:v24.06-standard

# Same base-image fixups as the cube-drop scene: the SOFA python bindings
# need libpython3.10's shared object and an explicit PYTHONPATH /
# LD_LIBRARY_PATH to be importable outside runSofa.
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip libpython3.10 \
	&& rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/opt/sofa/plugins/SofaPython3/lib/python3/site-packages
ENV LD_LIBRARY_PATH=/opt/sofa/lib:/opt/sofa/plugins/SofaPython3/lib

WORKDIR /app
COPY backend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY static/ ./static/

# Fail the build (not the first request) if the scene doesn't load, and
# check the physics: one simulated second from the default, highly chaotic
# start must conserve energy to well within 0.1%.
RUN python3 -c "\
import Sofa, Sofa.Simulation; \
from backend.simulation import build_scene, PHYSICS_DT; \
root = Sofa.Core.Node('root'); \
ctrl = build_scene(root); \
Sofa.Simulation.init(root); \
e0 = sum(ctrl.energies()); \
[Sofa.Simulation.animate(root, PHYSICS_DT) for _ in range(1000)]; \
drift = abs(sum(ctrl.energies()) - e0) / abs(e0); \
print(f'SOFA double pendulum OK: t={ctrl.sim_time:.3f}s, energy drift {drift:.2e}'); \
assert drift < 1e-3, drift"

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
