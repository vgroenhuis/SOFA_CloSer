# Shared base image (msd-sofa-base) for scenes that run an original SOFA
# scene file through the sofaweb runtime. SOFA v26.06 "full" is the same
# version the original scenes were written for, and unlike the "standard"
# image it includes the SoftRobots plugin (SurfacePressureConstraint).
FROM sofaframework/sofa_nightly_ubuntu:v26.06-full

# As in the older images: the SOFA python bindings need libpython3.10's
# shared object plus explicit PYTHONPATH / LD_LIBRARY_PATH to be importable
# outside runSofa.
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip libpython3.10 \
	&& rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/opt/sofa/plugins/SofaPython3/lib/python3/site-packages:/opt/sofaweb
ENV LD_LIBRARY_PATH=/opt/sofa/lib:/opt/sofa/plugins/SofaPython3/lib

RUN pip3 install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29" "numpy>=1.24,<2"

# The sofaweb runtime (capture layer, runner, web app and browser viewer)
# shared by every scene built on this image.
COPY sofaweb/ /opt/sofaweb/sofaweb/
