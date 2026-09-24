"""Static configuration (environment variables) plus the subset of limits an
admin can change at runtime from the admin page.

Runtime-tunable limits are stored in the SQLite `settings` table and
override the environment defaults; see `Limits` and `Store.get_limits`.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


SCENES_DIR = Path(os.environ.get("MSD_SCENES_DIR", "/scenes"))
DATA_DIR = Path(os.environ.get("MSD_DATA_DIR", "/data"))
STATIC_DIR = Path(__file__).parent.parent / "static"

# Docker network shared by the orchestrator and every simulation container;
# simulations are reached by container name on this network and never
# publish ports on the host.
DOCKER_NETWORK = os.environ.get("MSD_DOCKER_NETWORK", "msd-net")
CONTAINER_PREFIX = os.environ.get("MSD_CONTAINER_PREFIX", "msd-sim-")
IMAGE_PREFIX = os.environ.get("MSD_IMAGE_PREFIX", "msd-scene-")
MANAGED_LABEL = "msd.managed"

ADMIN_PASSWORD = os.environ.get("MSD_ADMIN_PASSWORD", "")
ADMIN_SESSION_HOURS = _env_float("MSD_ADMIN_SESSION_HOURS", 12.0)

# Per-container resource caps (a scene's scene.json may override these).
SIM_CPUS = _env_float("MSD_SIM_CPUS", 1.0)
SIM_MEMORY = os.environ.get("MSD_SIM_MEMORY", "2g")
START_TIMEOUT_SECONDS = _env_float("MSD_START_TIMEOUT_SECONDS", 120.0)

# When the orchestrator sits behind a TLS-terminating reverse proxy, trust
# its X-Forwarded-For / X-Forwarded-Proto headers for client address and
# secure-cookie decisions.
TRUST_PROXY_HEADERS = _env_bool("MSD_TRUST_PROXY_HEADERS", False)

REAPER_INTERVAL_SECONDS = 3.0
TOKEN_COOKIE_DAYS = 14


@dataclass
class Limits:
    max_sims: int = _env_int("MSD_MAX_SIMS", 3)
    idle_timeout_minutes: float = _env_float("MSD_IDLE_TIMEOUT_MINUTES", 15.0)
    max_hold_hours: float = _env_float("MSD_MAX_HOLD_HOURS", 12.0)
    # How many simulations may be on a long hold at once, so holds can never
    # starve the pool completely.
    max_held_sims: int = _env_int("MSD_MAX_HELD_SIMS", 1)
    # Simultaneous claims per client address (0 = unlimited).
    max_claims_per_client: int = _env_int("MSD_MAX_CLAIMS_PER_CLIENT", 1)

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def as_dict(self) -> dict:
        return asdict(self)


def load_secret_key() -> bytes:
    """HMAC key for admin sessions -- from the environment, or generated once
    and persisted in the data volume so sessions survive restarts."""
    env_key = os.environ.get("MSD_SECRET_KEY")
    if env_key:
        return env_key.encode()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    key_file = DATA_DIR / "secret.key"
    if not key_file.exists():
        key_file.write_text(secrets.token_hex(32))
    return key_file.read_text().strip().encode()
