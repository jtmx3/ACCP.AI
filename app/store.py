from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from app.core import new_state, now_iso


class StoreError(Exception):
    pass


class NotFound(StoreError):
    pass


class Forbidden(StoreError):
    pass


class StaleRevision(StoreError):
    def __init__(self, current_revision: int):
        self.current_revision = current_revision
        super().__init__("The case changed since it was loaded.")


class Store:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.getenv("ACCP_DB_PATH", "data/accp.sqlite3")
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=10000")
            return conn
        except BaseException:
            # PRAGMA/configuration failures happen before the caller can own the
            # handle, so close here rather than relying on a later finalizer.
            conn.close()
            raise

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured SQLite connection and always close it explicitly."""
        conn = self._connect()
        try:
            yield conn
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL,
                revision INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                revision INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
        if self.db_path != ":memory:":
            try:
                os.chmod(self.db_path, 0o600)
            except OSError:
                pass

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        state = json.loads(row["state_json"])
        state["revision"] = row["revision"]
        state["updated_at"] = row["updated_at"]
        return state

    def create(self) -> tuple[dict[str, Any], str]:
        import uuid
        session_id = str(uuid.uuid4())
        token = secrets.token_urlsafe(32)
        state = new_state(session_id)
        stamp = now_iso()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                         (session_id, self._token_hash(token), 0,
                          json.dumps(state, ensure_ascii=False), stamp, stamp))
            conn.execute("INSERT INTO audit_log(session_id,revision,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
                         (session_id, 0, "session_created", "{}", stamp))
            conn.commit()
        return state, token

    def get(self, session_id: str, token: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            raise NotFound("Case not found")
        if not secrets.compare_digest(row["token_hash"], self._token_hash(token)):
            raise Forbidden("Invalid case access token")
        return self._decode(row)

    def mutate(self, session_id: str, token: str, expected_revision: int,
               event_type: str, mutator: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not row:
                conn.rollback()
                raise NotFound("Case not found")
            if not secrets.compare_digest(row["token_hash"], self._token_hash(token)):
                conn.rollback()
                raise Forbidden("Invalid case access token")
            if row["revision"] != expected_revision:
                conn.rollback()
                raise StaleRevision(row["revision"])
            state = self._decode(row)
            mutator(state)
            new_revision = expected_revision + 1
            stamp = now_iso()
            state["revision"] = new_revision
            state["updated_at"] = stamp
            conn.execute("UPDATE sessions SET revision=?,state_json=?,updated_at=? WHERE id=? AND revision=?",
                         (new_revision, json.dumps(state, ensure_ascii=False), stamp, session_id, expected_revision))
            conn.execute("INSERT INTO audit_log(session_id,revision,event_type,payload_json,created_at) VALUES(?,?,?,?,?)",
                         (session_id, new_revision, event_type,
                          json.dumps({"stage": state.get("stage"), "pending_slot": state.get("pending_slot")}, ensure_ascii=False), stamp))
            conn.commit()
            return state

    def audit(self, session_id: str, token: str) -> list[dict[str, Any]]:
        self.get(session_id, token)
        with self._connection() as conn:
            rows = conn.execute("SELECT revision,event_type,payload_json,created_at FROM audit_log WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return [{"revision": r["revision"], "event_type": r["event_type"], "payload": json.loads(r["payload_json"]), "created_at": r["created_at"]} for r in rows]
