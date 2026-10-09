#!/bin/sh
# Builds the Docker image for every scene in scenes/<id>/ that has a
# Dockerfile, tagged msd-scene-<id> (what the orchestrator expects unless a
# scene.json sets "image" explicitly). Pass scene ids to build only those.
# The shared base image for sofaweb scenes (scenes/_base -> msd-sofa-base)
# is built first; Docker's cache makes that quick when nothing changed.
set -e
cd "$(dirname "$0")"
if [ -f scenes/_base/Dockerfile ]; then
	echo "=== Building msd-sofa-base"
	docker build -t msd-sofa-base scenes/_base
fi
if [ "$#" -gt 0 ]; then scenes="$*"; else scenes=$(ls scenes); fi
for id in $scenes; do
	if [ -f "scenes/$id/Dockerfile" ] && [ -f "scenes/$id/scene.json" ]; then
		echo "=== Building msd-scene-$id"
		docker build -t "msd-scene-$id" "scenes/$id"
	fi
done
