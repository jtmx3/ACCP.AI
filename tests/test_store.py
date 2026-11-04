import tempfile
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import resume_state
from app.store import Forbidden, StaleRevision, Store


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / "case.sqlite3"))
        self.state, self.token = self.store.create()

    def tearDown(self):
        self.tmp.cleanup()

    def track_connections(self):
        opened = []
        original_connect = self.store._connect

        def connect_and_track():
            conn = original_connect()
            opened.append(conn)
            return conn

        patcher = patch.object(self.store, "_connect", side_effect=connect_and_track)
        return patcher, opened

    def assert_connections_closed(self, connections):
        self.assertTrue(connections, "expected the operation to open a connection")
        for conn in connections:
            with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed"):
                conn.execute("SELECT 1")

    def test_case_id_without_token_is_denied(self):
        with self.assertRaises(Forbidden):
            self.store.get(self.state["session_id"], "wrong-token")

    def test_stale_revision_does_not_overwrite_newer_state(self):
        def pause(state):
            state["stage"] = "paused"
        updated = self.store.mutate(self.state["session_id"], self.token, 0, "pause", pause)
        self.assertEqual(updated["revision"], 1)
        with self.assertRaises(StaleRevision):
            self.store.mutate(self.state["session_id"], self.token, 0, "stale-write", lambda s: s.update(stage="review"))
        current = self.store.get(self.state["session_id"], self.token)
        self.assertEqual(current["stage"], "paused")

    def test_resume_appends_a_durable_assistant_message(self):
        def pause(state):
            state["stage"] = "paused"
            state["pending_slot"] = "narrative"
        self.store.mutate(self.state["session_id"], self.token, 0, "pause", pause)
        resumed = self.store.mutate(self.state["session_id"], self.token, 1, "resume", resume_state)
        self.assertEqual(resumed["stage"], "interview")
        self.assertIn("رجعنا للجلسة المحفوظة", resumed["messages"][-1]["text"])
        events = self.store.audit(self.state["session_id"], self.token)
        self.assertEqual([e["event_type"] for e in events], ["session_created", "pause", "resume"])

    def test_create_read_mutate_and_audit_close_every_connection(self):
        patcher, connections = self.track_connections()
        with patcher:
            created_state, token = self.store.create()
            self.store.get(created_state["session_id"], token)
            self.store.mutate(created_state["session_id"], token, 0, "change",
                              lambda state: state.update(stage="review"))
            self.store.audit(created_state["session_id"], token)
        self.assert_connections_closed(connections)

    def test_read_authorization_failure_closes_connection(self):
        patcher, connections = self.track_connections()
        with patcher, self.assertRaises(Forbidden):
            self.store.get(self.state["session_id"], "wrong-token")
        self.assert_connections_closed(connections)

    def test_mutator_exception_rolls_back_and_closes_connection(self):
        patcher, connections = self.track_connections()

        def fail_after_read(_state):
            raise RuntimeError("synthetic mutator failure")

        with patcher, self.assertRaisesRegex(RuntimeError, "synthetic mutator failure"):
            self.store.mutate(self.state["session_id"], self.token, 0, "failed", fail_after_read)
        self.assert_connections_closed(connections)
        self.assertEqual(self.store.get(self.state["session_id"], self.token)["revision"], 0)

    def test_stale_revision_closes_connection_after_rollback(self):
        self.store.mutate(self.state["session_id"], self.token, 0, "advance", lambda state: None)
        patcher, connections = self.track_connections()
        with patcher, self.assertRaises(StaleRevision):
            self.store.mutate(self.state["session_id"], self.token, 0, "stale", lambda state: None)
        self.assert_connections_closed(connections)

    def test_create_sql_failure_closes_connection_and_rolls_back(self):
        patcher, connections = self.track_connections()
        with patch("app.store.Store._token_hash", side_effect=RuntimeError("synthetic create failure")):
            with patcher, self.assertRaisesRegex(RuntimeError, "synthetic create failure"):
                self.store.create()
        self.assert_connections_closed(connections)

    def test_connection_setup_failure_closes_partially_initialized_handle(self):
        class FailingConnection:
            def __init__(self):
                self.closed = False
                self.row_factory = None

            def execute(self, _sql):
                raise sqlite3.OperationalError("synthetic PRAGMA failure")

            def close(self):
                self.closed = True

        conn = FailingConnection()
        with patch("app.store.sqlite3.connect", return_value=conn):
            with self.assertRaisesRegex(sqlite3.OperationalError, "synthetic PRAGMA failure"):
                self.store._connect()
        self.assertTrue(conn.closed)

    def test_database_files_can_be_removed_immediately_after_store_use(self):
        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "remove-me.sqlite3"
        store = Store(str(db_path))
        state, token = store.create()
        store.get(state["session_id"], token)
        store.audit(state["session_id"], token)
        self.assertTrue(db_path.exists())
        temp_dir.cleanup()
        self.assertFalse(db_path.parent.exists())


if __name__ == "__main__":
    unittest.main()
