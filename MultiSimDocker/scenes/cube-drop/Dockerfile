FROM sofaframework/sofa_nightly_ubuntu:v24.06-standard

# The base image ships libpython3.10-stdlib but not the libpython3.10 shared
# object itself, which the compiled Sofa python bindings dlopen at import
# time -- without it, `import Sofa` fails with a missing .so error.
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip libpython3.10 \
	&& rm -rf /var/lib/apt/lists/*

# The base image's ENTRYPOINT only wraps `runSofa`; it doesn't configure the
# environment for a bare `import Sofa`, so PYTHONPATH/LD_LIBRARY_PATH for
# the SofaPython3 bindings are set explicitly here.
ENV PYTHONPATH=/opt/sofa/plugins/SofaPython3/lib/python3/site-packages
ENV LD_LIBRARY_PATH=/opt/sofa/lib:/opt/sofa/plugins/SofaPython3/lib

WORKDIR /app
COPY backend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY static/ ./static/

# Sanity check: fail the build immediately (instead of at the first request)
# if the Sofa python bindings or the scene's plugins/components aren't
# actually available in this image.
RUN python3 -c "\
import Sofa; \
from backend.simulation import build_scene; \
root = Sofa.Core.Node('root'); \
build_scene(root); \
Sofa.Simulation.init(root); \
print('SOFA scene builds and initializes OK')"

EXPOSE 8000
# The base image's ENTRYPOINT (/docker-entrypoint.sh) just execs this CMD
# when nothing is piped to stdin, so it's left in place unmodified.
CMD ["python3", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
