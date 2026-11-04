import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.conversation import MAX_CONTEXT_CHARS, build_bounded_context, interpret_message
from app.core import new_state
from app.store import Store


class BoundedConversationContextTests(unittest.TestCase):
    def test_context_is_bounded_relevant_and_excludes_session_secrets_and_assessment(self):
        state = new_state("synthetic-session-id")
        state["case_token"] = "DO-NOT-SEND-TOKEN"
        state["assessment"] = {"secret": "DO-NOT-SEND-ASSESSMENT"}
        state["messages"] = [{"id": "m1", "role": "assistant", "kind": "question", "text": "كم دفعت؟"}]
        for index in range(1, 20):
            state["messages"].append({"id": f"m{index + 1}", "role": "user", "kind": "chat",
                                      "text": f"تحديث مالي طويل رقم {index} " + ("معلومة " * 90)})
        state["facts"] = {
            "paid_amount": {"status": "reported", "value": "130 دينار", "quote": "دفعت 130 دينار",
                            "message_id": "m20"},
            "identity": {"status": "reported", "value": "غير مرتبط", "quote": "اسم شخص",
                         "message_id": "m20"},
        }
        state["money_events"] = [{"id": "money-1", "kind": "paid", "amount": "130 دينار",
                                  "relation": "initial", "quote": "دفعت 130 دينار", "message_id": "m20"}]
        context = build_bounded_context(state, "دفعت المبلغ، هل هذه دفعة جديدة؟", "paid_amount_relation")
        encoded = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        self.assertLessEqual(len(encoded), MAX_CONTEXT_CHARS)
        self.assertTrue(context["untrusted_user_data"])
        self.assertEqual(context["pending_question"]["slot"], "paid_amount_relation")
        self.assertTrue(context["related_money_events"])
        self.assertNotIn("DO-NOT-SEND-TOKEN", encoded)
        self.assertNotIn("DO-NOT-SEND-ASSESSMENT", encoded)
        self.assertNotIn("synthetic-session-id", encoded)
        self.assertLessEqual(len(context["recent_relevant_messages"]), 4)

    def test_message_interpreter_is_nonexclusive_and_does_not_mutate_state(self):
        result = interpret_message("تصحيح المبلغ، ما بعرف الرقم بس عندي إيصال؛ ليش بتسأل؟", "requested_amount")
        self.assertEqual(result["pending_slot"], "requested_amount")
        self.assertIn("answer_pending_question", result["goals"])
        self.assertIn("correct_prior_fact", result["goals"])
        self.assertIn("unknown", result["goals"])
        self.assertIn("ask_why", result["goals"])
        self.assertTrue(result["may_contain_multiple_goals"])


class ContextConsentApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_store = main.store
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
        main.store = self.old_store
        self.tmp.cleanup()

    def test_context_is_not_sent_unless_separately_authorized(self):
        calls = []

        def fake_extraction(text, pending, context=None, interpretation=None):
            calls.append({"text": text, "pending": pending, "context": context, "interpretation": interpretation})
            return []

        provider = {"provider": "openai-compatible", "model": "synthetic-test", "mode": "external_model",
                    "configuration_complete": True}
        with patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value=provider), \
             patch("app.main.generate_interview_extraction", side_effect=fake_extraction), \
             patch("app.main.last_call_trace", return_value={"attempted": True, "latency_ms": 1}):
            first = self.client.post(f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                json={"expected_revision": self.revision, "text": "شخص هددني", "kind": "answer",
                      "allow_external_extraction": True})
            self.assertEqual(first.status_code, 200, first.text)
            first_result = first.json()
            self.revision = first_result["session"]["revision"]
            self.assertIsNone(calls[0]["context"])
            self.assertFalse(first_result["external_context_sent"])

            second = self.client.post(f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                json={"expected_revision": self.revision, "text": "اليوم دفعت 130 دينار", "kind": "answer",
                      "allow_external_extraction": True, "allow_context_extraction": True})
            self.assertEqual(second.status_code, 200, second.text)
            self.assertIsNotNone(calls[1]["context"])
            result = second.json()
            self.assertTrue(result["external_context_sent"])
            self.assertLessEqual(result["provider_call"]["context_char_count"], MAX_CONTEXT_CHARS)
            self.assertEqual(result["provider_call"]["context_message_count"] <= 4, True)
            self.assertIsNotNone(calls[1]["interpretation"])

    def test_mixed_why_question_saves_scoped_facts_then_reasks_pending_question(self):
        with patch("app.main.model_configured", return_value=False):
            first = self.client.post(f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                json={"expected_revision": self.revision, "text": "في شخص يهددني على واتساب", "kind": "answer"})
            self.assertEqual(first.status_code, 200, first.text)
            self.revision = first.json()["session"]["revision"]
            self.assertEqual(first.json()["session"]["pending_slot"], "direct_danger")

            text = "ما بعرف مين هو، بس عندي رقم الواتساب اللي تواصل معي منه، وما بدي أكتبه هون، ليش بتسأل؟"
            result = self.client.post(f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                json={"expected_revision": self.revision, "text": text, "kind": "answer"})
        self.assertEqual(result.status_code, 200, result.text)
        state = result.json()["session"]
        self.assertEqual(state["facts"]["identity"]["status"], "unknown")
        self.assertEqual(state["facts"]["contact"]["status"], "reported")
        self.assertEqual(state["facts"]["contact_identifier_disclosure"]["status"], "declined")
        self.assertNotIn("direct_danger", state["facts"])
        self.assertEqual(state["pending_slot"], "direct_danger")
        last = state["messages"][-1]
        self.assertEqual(last["kind"], "clarification")
        self.assertEqual(last["question_plan"]["slot"], "direct_danger")
        self.assertIn("بخطر مباشر", last["text"])

    def test_mixed_guidance_question_saves_evidence_and_resumes_pending_question(self):
        guidance = {"text": "احتفظ بنسخة أصلية من الرسائل ولا تعدلها.", "citations": []}
        with patch("app.main.model_configured", return_value=False), \
             patch("app.main.answer_question", return_value=guidance):
            first = self.client.post(f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                json={"expected_revision": self.revision, "text": "في شخص يهددني على واتساب", "kind": "answer"})
            self.assertEqual(first.status_code, 200, first.text)
            self.revision = first.json()["session"]["revision"]
            self.assertEqual(first.json()["session"]["pending_slot"], "direct_danger")

            text = "عندي لقطات شاشة للرسائل، كيف أوثقها؟"
            result = self.client.post(f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                json={"expected_revision": self.revision, "text": text, "kind": "answer"})
        self.assertEqual(result.status_code, 200, result.text)
        state = result.json()["session"]
        self.assertEqual(state["facts"]["evidence"]["status"], "reported")
        self.assertNotIn("direct_danger", state["facts"])
        self.assertEqual(state["pending_slot"], "direct_danger")
        assistant = [message for message in state["messages"] if message.get("role") == "assistant"]
        self.assertTrue(any(message.get("kind") == "answer" and guidance["text"] in message["text"]
                            for message in assistant))
        self.assertEqual(assistant[-1]["kind"], "clarification")
        self.assertEqual(assistant[-1]["question_plan"]["slot"], "direct_danger")


if __name__ == "__main__":
    unittest.main()
