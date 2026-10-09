"""Claim keys (for visitors) and signed session cookies (for the admin)."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque

# Crockford base32: no I, L, O or U, so a key read aloud or retyped from a
# screenshot survives the usual confusions (see normalize_token).
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_TOKEN_GROUPS = 4
_TOKEN_GROUP_LEN = 4  # 16 symbols * 5 bits = 80 bits of entropy


def new_token() -> str:
    raw = "".join(secrets.choice(_ALPHABET) for _ in range(_TOKEN_GROUPS * _TOKEN_GROUP_LEN))
    return "-".join(raw[i : i + _TOKEN_GROUP_LEN] for i in range(0, len(raw), _TOKEN_GROUP_LEN))


def normalize_token(token: str) -> str:
    cleaned = "".join(ch for ch in token.upper() if ch.isalnum())
    cleaned = cleaned.replace("O", "0").replace("I", "1").replace("L", "1")
    if len(cleaned) != _TOKEN_GROUPS * _TOKEN_GROUP_LEN:
        return ""
    return "-".join(cleaned[i : i + _TOKEN_GROUP_LEN] for i in range(0, len(cleaned), _TOKEN_GROUP_LEN))


def hash_token(token: str) -> str:
    # Keys are high-entropy random strings, so a plain SHA-256 is enough --
    # this only ensures a leaked database can't be used to take over sims.
    return hashlib.sha256(normalize_token(token).encode()).hexdigest()


def token_matches(token: str, token_hash: str) -> bool:
    if not token or not normalize_token(token):
        return False
    return hmac.compare_digest(hash_token(token), token_hash)


def new_sim_id() -> str:
    return "".join(secrets.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(6))


def token_cookie_name(sim_id: str) -> str:
    return f"msd_claim_{sim_id}"


# -- admin sessions ------------------------------------------------------


def sign_admin_session(secret: bytes, hours: float) -> str:
    expiry = str(int(time.time() + hours * 3600))
    signature = hmac.new(secret, f"admin:{expiry}".encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{signature}"


def verify_admin_session(secret: bytes, value: str | None) -> bool:
    if not value or "." not in value:
        return False
    expiry, signature = value.split(".", 1)
    expected = hmac.new(secret, f"admin:{expiry}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return False
    return expiry.isdigit() and int(expiry) > time.time()


def check_admin_password(configured: str, supplied: str) -> bool:
    if not configured:
        return False
    return hmac.compare_digest(configured.encode(), supplied.encode())


class RateLimiter:
    """Sliding-window limiter for brute-forceable endpoints (admin login, key
    reclaim), keyed by client address."""

    def __init__(self, max_events: int, window_seconds: float) -> None:
        self._max = max_events
        self._window = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        events = self._events[key]
        while events and now - events[0] > self._window:
            events.popleft()
        if len(events) >= self._max:
            return False
        events.append(now)
        return True
