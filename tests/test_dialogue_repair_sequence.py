import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import main
from app.core import classify_current_danger
from app.store import Store


class DialogueRepairSequenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_store = main.store
        main.store = Store(str(Path(self.tmp.name) / "dialogue-repair.sqlite3"))
        self.client = TestClient(main.app)
        created = self.client.post("/api/sessions", json={})
        self.assertEqual(created.status_code, 200, created.text)
        data = created.json()
        self.session_id = data["session"]["session_id"]
        self.token = data["token"]
        self.headers = {"X-Case-Token": self.token}
        self.revision = data["session"]["revision"]

    def tearDown(self):
        self.client.close()
        main.store = self.old_store
        self.tmp.cleanup()

    def _turn(self, text):
        response = self.client.post(
            f"/api/sessions/{self.session_id}/messages",
            headers=self.headers,
            json={"expected_revision": self.revision, "text": text, "kind": "answer"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.revision = data["session"]["revision"]
        return data

    def test_university_threat_time_ongoing_contact_protest_and_review_sequence(self):
        opening = ("في شخص معي بالجامعة كان معي على علاقثه حب و كنت ابعتلو صوري و انا عارية "
                   "و لما انفصلنا صار يهددني بنشر الصور او اني ارجع ابعتلو صور اكثر")
        first = self._turn(opening)
        state = first["session"]
        facts = state["facts"]
        self.assertEqual(state["pending_slot"], "initial_time")
        self.assertEqual(facts["narrative"]["quote"], opening)
        self.assertEqual((facts["threat"]["status"], facts["threat"]["value"]), ("reported", True))
        self.assertEqual((facts["publication_threat"]["status"], facts["publication_threat"]["value"]),
                         ("reported", True))
        self.assertEqual((facts["sensitive_photos"]["status"], facts["sensitive_photos"]["value"]),
                         ("reported", True))
        self.assertNotIn("evidence", facts, "images sent in the past do not imply saved evidence")
        self.assertNotIn("published", facts, "a threat to publish does not mean publication occurred")
        self.assertNotIn("identity", facts)
        self.assertNotIn("requested_amount", facts)
        self.assertNotIn("paid_amount", facts)
        self.assertNotIn("direct_danger", [message.get("question_plan", {}).get("slot")
                                            for message in state["messages"]])
        self.assertFalse(any(term in message.get("text", "") for message in state["messages"]
                             if message.get("role") == "assistant"
                             for term in ("911", "النجدة", "اتصل بالطوارئ")))
        self.assertEqual(state["assessment"]["priority_score"], 90)

        time_answer = self._turn("قبل ساعه")
        state = time_answer["session"]
        self.assertEqual(state["facts"]["initial_time"]["value"], "قبل ساعه")
        self.assertEqual(state["facts"]["initial_time"]["quote"], "قبل ساعه")
        self.assertEqual(state["pending_slot"], "repeated")
        self.assertEqual(sum(message.get("question_plan", {}).get("slot") == "initial_time"
                             for message in state["messages"] if message.get("role") == "assistant"), 1)

        time_protest = self._turn("اخبرتك قبل ساعه")
        state = time_protest["session"]
        self.assertEqual(state["facts"]["initial_time"]["value"], "قبل ساعه")
        self.assertEqual(state["pending_slot"], "repeated")
        self.assertEqual(state["messages"][-1]["kind"], "clarification")
        self.assertIn("هل تكرر التهديد", state["messages"][-1]["text"])

        ongoing = self._turn("ما زال يراسلني الان")
        state = ongoing["session"]
        self.assertEqual((state["facts"]["ongoing_contact"]["status"],
                          state["facts"]["ongoing_contact"]["value"]), ("reported", True))
        self.assertEqual(state["pending_slot"], "published")
        self.assertEqual(state["facts"]["repeated"]["status"], "unresolved")
        self.assertEqual(state["facts"]["repeated"]["value"], None)

        protest = self._turn("قلت لك ما زال يراسلني الان")
        state = protest["session"]
        self.assertEqual(state["facts"]["ongoing_contact"]["value"], True)
        self.assertEqual(state["facts"]["repeated"]["status"], "unresolved")
        self.assertIsNone(state["facts"]["repeated"]["value"])
        self.assertEqual(state["pending_slot"], "published")
        self.assertIn("بقي عندي هذا السؤال", state["messages"][-1]["text"])
        self.assertNotIn("هل تكرر التهديد", state["messages"][-1]["text"])
        self.assertEqual(sum(message.get("question_plan", {}).get("slot") == "initial_time"
                             for message in state["messages"] if message.get("role") == "assistant"), 1)

        publication = self._turn("لا، ما نشر الصور.")
        self.assertEqual(publication["session"]["facts"]["published"]["status"], "denied")
        self.assertEqual(publication["session"]["facts"]["published"]["value"], False)
        self.assertEqual(publication["session"]["pending_slot"], "identity")

        identity = self._turn("ما بعرف مين هو.")
        self.assertEqual(identity["session"]["facts"]["identity"]["status"], "unknown")
        self.assertEqual(identity["session"]["pending_slot"], "evidence")

        evidence = self._turn("ما عندي لقطات شاشة أو تسجيلات.")
        state = evidence["session"]
        self.assertEqual(state["stage"], "review")
        self.assertEqual(state["facts"]["evidence"]["status"], "denied")
        self.assertIsNone(state["facts"]["evidence"]["value"])
        self.assertEqual(state["facts"]["sensitive_photos"]["quote"], opening)
        self.assertEqual(state["facts"]["narrative"]["quote"], opening)

    def test_unknown_actor_does_not_hide_a_saved_conversation_copy(self):
        text = "الفاعل مجهول بالنسبة إلي، لكن عندي نسخة احتياطية من المحادثة."
        result = self._turn(text)
        facts = result["session"]["facts"]
        self.assertEqual(facts["identity"]["status"], "unknown")
        self.assertEqual(facts["evidence"]["status"], "reported")
        self.assertIn("رسائل أو محادثات", facts["evidence"]["value"])
        self.assertEqual(facts["evidence"]["quote"], text)

    def test_current_physical_danger_is_reactive_but_digital_extortion_and_past_events_are_not(self):
        self.assertIs(True, classify_current_danger("أنا بخطر مباشر الآن والشخص واقف قدام بيتي ومعه سكين."))
        self.assertIs(True, classify_current_danger("بيهددني بالقتل وهو واقف عند بابي الآن."))
        self.assertIs(True, classify_current_danger("جاي لعندي الآن ومعه سكين."))
        self.assertIsNone(classify_current_danger("هو يهددني بنشر صوري الخاصة."))
        self.assertIsNone(classify_current_danger("كان واقف عند بابي قبل ساعة وهددني أمس."))
        self.assertIs(False, classify_current_danger("لا، ما في خطر مباشر علي الآن."))


if __name__ == "__main__":
    unittest.main()
