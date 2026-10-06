#!/bin/sh
# Starts the orchestrator in the foreground (see deploy/ for the systemd unit).
set -e
cd "$(dirname "$0")"
if [ ! -f .env ]; then
	echo "No .env found - copying .env.example. Edit it to set the admin password!"
	cp .env.example .env
fi
[ -x runtime/venv/bin/python ] || ./install-runtime.sh
set -a; . ./.env; set +a
cd orchestrator
exec ../runtime/venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${MSD_PORT:-8080}"
