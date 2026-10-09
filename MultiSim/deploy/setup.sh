#!/bin/bash
# Root-only: installs and starts the systemd service for the user who owns
# this checkout. Run: sudo bash MultiSim/deploy/setup.sh
set -euo pipefail
dir="$(cd "$(dirname "$0")/.." && pwd)"
user="$(stat -c %U "$dir")"
uid="$(id -u "$user")"
[ -x "$dir/runtime/venv/bin/python" ] || { echo "Run ./install-runtime.sh first (as $user)."; exit 1; }
[ -f "$dir/.env" ] || { echo "Create $dir/.env first (cp .env.example .env, set MSD_ADMIN_PASSWORD)."; exit 1; }
loginctl enable-linger "$user"
sed -e "s|__USER__|$user|g" -e "s|__UID__|$uid|g" -e "s|__DIR__|$dir|g" "$dir/deploy/multisimnative.service" > /etc/systemd/system/multisimnative.service
systemctl daemon-reload
systemctl enable --now multisimnative.service
echo "Service started: systemctl status multisimnative"
