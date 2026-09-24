#!/bin/sh
# Builds the Docker image for every scene in scenes/<id>/ that has a
# Dockerfile, tagged msd-scene-<id> (what the orchestrator expects unless a
# scene.json sets "image" explicitly). Pass scene ids to build only those.
set -e
cd "$(dirname "$0")"
if [ "$#" -gt 0 ]; then scenes="$*"; else scenes=$(ls scenes); fi
for id in $scenes; do
	if [ -f "scenes/$id/Dockerfile" ] && [ -f "scenes/$id/scene.json" ]; then
		echo "=== Building msd-scene-$id"
		docker build -t "msd-scene-$id" "scenes/$id"
	fi
done
