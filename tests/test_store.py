import tempfile
import unittest
from pathlib import Path

from app.core import resume_state
from app.store import Forbidden, StaleRevision, Store


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / "case.sqlite3"))
        self.state, self.token = self.store.create()

    def tearDown(self):
        self.tmp.cleanup()

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


if __name__ == "__main__":
    unittest.main()
