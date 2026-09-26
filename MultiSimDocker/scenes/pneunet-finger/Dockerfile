# Built on the shared sofaweb base image (scenes/_base): the original scene
# script in original/ runs unmodified; sofaweb_scene.py describes its
# telemetry, parameters and controls for the web viewer.
FROM msd-sofa-base

WORKDIR /app
COPY original/ ./original/
COPY sofaweb_scene.py ./

# Fail the build (not the first visitor) if the scene doesn't build or step.
RUN python3 -m sofaweb check sofaweb_scene.py --seconds 0.2

EXPOSE 8000
CMD ["python3", "-m", "sofaweb", "serve", "sofaweb_scene.py"]
