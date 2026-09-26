#!/bin/sh
# Builds the scene images (cached, so quick when nothing changed) and
# (re)starts the orchestrator.
set -e
cd "$(dirname "$0")"
if [ ! -f .env ]; then
	echo "No .env found - copying .env.example. Edit it to set the admin password!"
	cp .env.example .env
fi
./build_scenes.sh
docker compose up -d --build
port=$(grep -E '^MSD_PORT=' .env | cut -d= -f2)
echo
echo "Lobby: http://127.0.0.1:${port:-8080}/"
echo "Admin: http://127.0.0.1:${port:-8080}/admin"
