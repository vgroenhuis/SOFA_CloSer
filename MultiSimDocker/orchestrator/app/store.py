"""SQLite persistence: active simulations, runtime limits, and an event log.

A single connection guarded by a lock is plenty here -- every query is tiny
and the orchestrator is a single process.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Limits

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sims (
    id             TEXT PRIMARY KEY,
    scene_id       TEXT NOT NULL,
    container_name TEXT NOT NULL,
    status         TEXT NOT NULL,          -- starting | running
    token_hash     TEXT NOT NULL,
    owner_name     TEXT NOT NULL DEFAULT '',
    client_addr    TEXT NOT NULL DEFAULT '',
    created_at     REAL NOT NULL,
    last_active    REAL NOT NULL,
    hold_until     REAL,
    hold_reason    TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS sims_token ON sims(token_hash);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     REAL NOT NULL,
    sim_id TEXT NOT NULL DEFAULT '',
    kind   TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS events_sim ON events(sim_id, id);
"""


@dataclass
class Sim:
    id: str
    scene_id: str
    container_name: str
    status: str
    token_hash: str
    owner_name: str
    client_addr: str
    created_at: float
    last_active: float
    hold_until: Optional[float]
    hold_reason: str

    def expires_at(self, idle_timeout_minutes: float) -> float:
        """When the reaper will return this simulation to the pool: after the
        idle timeout, but never before the end of an active hold."""
        idle_expiry = self.last_active + idle_timeout_minutes * 60.0
        return max(idle_expiry, self.hold_until or 0.0)

    def is_held(self, now: Optional[float] = None) -> bool:
        return self.hold_until is not None and self.hold_until > (now or time.time())


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _execute(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            return self._conn.execute(sql, params).rowcount

    # -- simulations ----------------------------------------------------

    def list_sims(self) -> list[Sim]:
        return [Sim(**dict(row)) for row in self._query("SELECT * FROM sims ORDER BY created_at")]

    def get_sim(self, sim_id: str) -> Optional[Sim]:
        rows = self._query("SELECT * FROM sims WHERE id = ?", (sim_id,))
        return Sim(**dict(rows[0])) if rows else None

    def find_by_token_hash(self, token_hash: str) -> Optional[Sim]:
        rows = self._query("SELECT * FROM sims WHERE token_hash = ?", (token_hash,))
        return Sim(**dict(rows[0])) if rows else None

    def insert_sim(self, sim: Sim) -> None:
        self._execute(
            "INSERT INTO sims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sim.id, sim.scene_id, sim.container_name, sim.status, sim.token_hash,
                sim.owner_name, sim.client_addr, sim.created_at, sim.last_active,
                sim.hold_until, sim.hold_reason,
            ),
        )

    def delete_sim(self, sim_id: str) -> bool:
        return self._execute("DELETE FROM sims WHERE id = ?", (sim_id,)) > 0

    def set_status(self, sim_id: str, status: str) -> None:
        self._execute("UPDATE sims SET status = ? WHERE id = ?", (status, sim_id))

    def touch(self, sim_id: str, now: Optional[float] = None) -> None:
        self._execute("UPDATE sims SET last_active = ? WHERE id = ?", (now or time.time(), sim_id))

    def set_hold(self, sim_id: str, until: Optional[float], reason: str) -> None:
        self._execute(
            "UPDATE sims SET hold_until = ?, hold_reason = ? WHERE id = ?",
            (until, reason if until is not None else "", sim_id),
        )

    def set_token_hash(self, sim_id: str, token_hash: str) -> None:
        self._execute("UPDATE sims SET token_hash = ? WHERE id = ?", (token_hash, sim_id))

    # -- runtime limits -------------------------------------------------

    def get_limits(self) -> Limits:
        limits = Limits()
        for row in self._query("SELECT key, value FROM settings"):
            if row["key"] in Limits.field_names():
                current = getattr(limits, row["key"])
                setattr(limits, row["key"], type(current)(row["value"]))
        return limits

    def set_limits(self, **changes: float) -> Limits:
        for key, value in changes.items():
            if key in Limits.field_names():
                self._execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, str(value)),
                )
        return self.get_limits()

    def reset_limits(self) -> Limits:
        self._execute("DELETE FROM settings")
        return self.get_limits()

    # -- event log ------------------------------------------------------

    def log(self, kind: str, sim_id: str = "", detail: str = "") -> None:
        self._execute(
            "INSERT INTO events (ts, sim_id, kind, detail) VALUES (?, ?, ?, ?)",
            (time.time(), sim_id, kind, detail),
        )

    def recent_events(self, limit: int = 200) -> list[dict]:
        rows = self._query("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]

    def last_release(self, sim_id: str) -> Optional[dict]:
        rows = self._query(
            "SELECT * FROM events WHERE sim_id = ? AND kind = 'released' ORDER BY id DESC LIMIT 1",
            (sim_id,),
        )
        return dict(rows[0]) if rows else None
