import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import main
from app.store import Store


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        main.store = Store(str(Path(self.tmp.name) / "api.sqlite3"))
        self.client = TestClient(main.app)
        created = self.client.post("/api/sessions", json={})
        self.assertEqual(created.status_code, 200)
        payload = created.json()
        self.case_id = payload["session"]["session_id"]
        self.token = payload["token"]
        self.headers = {"X-Case-Token": self.token}
        self.revision = payload["session"]["revision"]

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def turn(self, text, *, kind="answer"):
        response = self.client.post(
            f"/api/sessions/{self.case_id}/messages",
            headers=self.headers,
            json={"expected_revision": self.revision, "text": text, "kind": kind},
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.revision = data["session"]["revision"]
        return data["session"]

    def test_token_is_required_and_stale_write_is_rejected(self):
        denied = self.client.get(f"/api/sessions/{self.case_id}")
        self.assertEqual(denied.status_code, 403)
        self.turn("شخص يهددني على واتساب")
        stale = self.client.post(
            f"/api/sessions/{self.case_id}/messages",
            headers=self.headers,
            json={"expected_revision": 0, "text": "لا", "kind": "answer"},
        )
        self.assertEqual(stale.status_code, 409)
        current = self.client.get(f"/api/sessions/{self.case_id}", headers=self.headers).json()["session"]
        self.assertEqual(current["revision"], 1)
        self.assertEqual(current["pending_slot"], "direct_danger")

    def test_qa_source_scope_session_resume_and_safe_citations(self):
        why = self.turn("ليش بتسأل عن التفاصيل؟", kind="question")
        why_message = why["messages"][-2]
        self.assertFalse(why_message["external_model_used"])
        question = self.turn("كيف أحفظ الأدلة واللقطات؟", kind="question")
        answer_message = question["messages"][-2]
        self.assertIn("ما عندي مصدر موثوق", answer_message["text"])
        self.assertEqual(answer_message["citations"], [])
        response = self.client.put(
            f"/api/sessions/{self.case_id}/jurisdiction",
            headers=self.headers,
            json={"expected_revision": self.revision, "jurisdiction": "AU"},
        )
        self.assertEqual(response.status_code, 200)
        self.revision = response.json()["session"]["revision"]
        answer = self.turn("كيف أحفظ الأدلة واللقطات؟", kind="question")
        answer_message = answer["messages"][-2]
        self.assertTrue(answer_message["citations"])
        self.assertTrue(all(source["geography"] == "AU" for source in answer_message["citations"]))
        self.turn("احفظ الجلسة")
        self.assertEqual(self.client.get(f"/api/sessions/{self.case_id}", headers=self.headers).json()["session"]["stage"], "paused")
        resumed = self.client.post(f"/api/sessions/{self.case_id}/resume", headers=self.headers, json={"expected_revision": self.revision})
        self.assertEqual(resumed.status_code, 200)
        self.revision = resumed.json()["session"]["revision"]
        self.assertEqual(resumed.json()["session"]["stage"], "interview")

    def test_immediate_danger_in_question_mode_bypasses_rag(self):
        state = self.turn("الشخص وصل لعندي الآن ومعه سلاح، شو أعمل؟", kind="question")
        self.assertEqual(state["stage"], "urgent_stop")
        self.assertEqual(state["facts"]["direct_danger"]["value"], True)
        safety = state["messages"][-1]
        self.assertEqual(safety["kind"], "safety")
        self.assertEqual(safety["citations"], [])
        self.assertFalse(safety["external_model_used"])

    def test_unresolved_conflict_blocks_confirmation_until_corrected(self):
        for text in ("في شخص يهددني", "لا", "قال ادفع 100 دينار", "طلب 120 دينار أيضًا"):
            self.turn(text)
        self.assertEqual(self.turn("اعرض الملخص")["stage"], "review")
        blocked = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers, json={"expected_revision": self.revision})
        self.assertEqual(blocked.status_code, 422)
        corrected = self.turn("تصحيح: المبلغ المطلوب 100 دينار.")
        self.assertEqual(corrected["facts"]["requested_amount"]["status"], "reported")
        confirmed = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers, json={"expected_revision": self.revision})
        self.assertEqual(confirmed.status_code, 200)

    def test_full_scenario_confirmation_and_correction_are_versioned(self):
        turns = [
            "في شخص عم يهددني على واتساب.",
            "لا، ما في خطر مباشر علي الآن.",
            "قال إذا ما حولتله 100 دينار رح ينشر محادثاتي الخاصة.",
            "أمس الساعة 6 مسا.",
            "نعم، رجع بعتلي اليوم.",
            "دفعت 130 دينار.",
            "بمحفظة إلكترونية.",
            "بس أنا دفعت.",
            "لا، ما حدد مهلة.",
            "ما بعرف.",
            "بعرف رقم الواتساب بس.",
            "صور التهديد وإيصال التحويل.",
        ]
        for text in turns:
            state = self.turn(text)
        self.assertEqual(state["stage"], "review")
        self.assertEqual(state["assessment"]["priority_score"], 90)
        facts = state["facts"]
        self.assertEqual(facts["requested_amount"]["value"], "100 دينار")
        self.assertEqual(facts["paid_amount"]["value"], "130 دينار")
        self.assertEqual(facts["published"]["status"], "unknown")
        report = self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers)
        self.assertEqual(report.status_code, 200)
        self.assertIn("100 دينار", report.text)
        self.assertIn("130 دينار", report.text)
        confirmed = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers, json={"expected_revision": self.revision})
        self.assertEqual(confirmed.status_code, 200)
        self.revision = confirmed.json()["session"]["revision"]
        self.assertTrue(confirmed.json()["session"]["confirmed"])
        confirmed_version = confirmed.json()["session"]["summary_version"]
        shown = self.turn("اعرض الملخص")
        self.assertTrue(shown["confirmed"])
        self.assertEqual(shown["summary_version"], confirmed_version)
        added = self.turn("أضيف تفاصيل: الرسائل وصلتني بعد المغرب.")
        self.assertFalse(added["confirmed"])
        self.assertEqual(added["facts"]["additional_details"]["value"], "أضيف تفاصيل: الرسائل وصلتني بعد المغرب.")
        corrected = self.turn("تصحيح: المبلغ المطلوب 120 دينار، وليس 100.")
        self.assertFalse(corrected["confirmed"])
        self.assertEqual(corrected["facts"]["requested_amount"]["value"], "120 دينار")
        self.assertTrue(corrected["facts"]["requested_amount"]["history"])


if __name__ == "__main__":
    unittest.main()
