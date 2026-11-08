import os
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.store import Store
from app.model import InterviewUnderstanding, _chat, call_trace_history, reset_call_trace


class ConversationCompletionV33Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        main.store = Store(str(Path(self.tmp.name) / "conversation.sqlite3"))
        self.client = TestClient(main.app)
        created = self.client.post("/api/sessions", json={})
        self.assertEqual(created.status_code, 200)
        self.session_id = created.json()["session"]["session_id"]
        self.token = created.json()["token"]
        self.headers = {"X-Case-Token": self.token}
        self.revision = created.json()["session"]["revision"]

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def turn(self, text, *, kind="answer", **kwargs):
        response = self.client.post(
            f"/api/sessions/{self.session_id}/messages", headers=self.headers,
            json={"expected_revision": self.revision, "text": text, "kind": kind, **kwargs},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.revision = result["session"]["revision"]
        return result

    def establish_timing_question(self):
        self.turn("في شخص على واتساب طلب مني 100 دينار، وقال رح ينشر محادثاتي الخاصة إذا ما دفعت.")
        result = self.turn("لا، ما في خطر مباشر علي الآن، بس اليوم رجع بعتلي نفس التهديد.")
        self.assertEqual(result["session"]["pending_slot"], "initial_time")
        self.assertIn("بدأ التصرف أو وصلك أول تواصل", result["session"]["messages"][-1]["text"])

    def test_mixed_initial_and_latest_time_are_separate_and_equivalent_model_value_does_not_conflict(self):
        self.establish_timing_question()
        text = "أول رسالة وصلتني مبارح حوالي الساعة 6 المسا، أما آخر تهديد فكان اليوم."
        model_facts = [{"field": "initial_time", "status": "reported",
                        "value": "مبارح حوالي الساعة 6 مسا",
                        "quote": "أول رسالة وصلتني مبارح حوالي الساعة 6 المسا"}]
        trace = {"attempted": True, "status": "accepted", "response_valid": True,
                 "result_accepted": True, "fallback_used": False, "latency_ms": 4}
        with patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value={"provider": "ollama", "model": "fixture", "mode": "local_model"}), \
             patch("app.main.generate_interview_extraction", return_value=model_facts), \
             patch("app.main.last_call_trace", return_value=trace):
            result = self.turn(text, allow_external_extraction=True)
        state = result["session"]
        start = state["facts"]["initial_time"]
        latest = state["facts"]["latest_time"]
        self.assertNotEqual(start["status"], "conflict")
        self.assertEqual(start["value"], "مبارح حوالي الساعة 6 المسا")
        self.assertEqual(start["quote"], "أول رسالة وصلتني مبارح حوالي الساعة 6 المسا")
        self.assertEqual(latest["value"], "اليوم")
        self.assertTrue(any(item.get("quote") == "أما آخر تهديد فكان اليوم"
                            for item in latest.get("supporting_statements", [])))
        self.assertFalse(any(item.get("reason") == "same_turn_rule_model_disagreement"
                             for item in start.get("history", [])))
        self.assertNotEqual(state["pending_slot"], "initial_time")
        self.assertTrue(result.get("timings"))
        self.assertTrue(result.get("provider_calls"))

    def test_model_cannot_move_latest_time_to_initial_or_create_a_false_user_conflict(self):
        self.establish_timing_question()
        text = "أول رسالة وصلتني مبارح حوالي الساعة 6 المسا، أما آخر تهديد فكان اليوم."
        proposal = [{"field": "initial_time", "status": "reported", "value": "اليوم",
                     "quote": "آخر تهديد فكان اليوم"}]
        with patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value={"provider": "ollama", "model": "fixture", "mode": "local_model"}), \
             patch("app.main.generate_interview_extraction", return_value=proposal), \
             patch("app.main.last_call_trace", return_value={"attempted": True, "latency_ms": 1}):
            result = self.turn(text, allow_external_extraction=True)
        state = result["session"]
        self.assertEqual(state["facts"]["initial_time"]["status"], "reported")
        self.assertIn("مبارح", state["facts"]["initial_time"]["value"])
        self.assertNotEqual(state["facts"]["initial_time"]["status"], "conflict")
        trace = result.get("model_merge_trace", [])
        self.assertTrue(any(item.get("field") == "initial_time" and item.get("disposition") == "ignored"
                            for item in trace))

    def test_latest_contact_only_does_not_answer_pending_initial_time(self):
        self.establish_timing_question()
        result = self.turn("أما آخر تهديد فكان اليوم.")
        state = result["session"]
        self.assertNotIn("initial_time", state["facts"])
        self.assertEqual(state["facts"]["latest_time"]["value"], "اليوم")
        self.assertEqual(state["pending_slot"], "initial_time")
        self.assertIn("متى بدأ التصرف أو وصلك أول تواصل", state["messages"][-1]["text"])
        self.assertNotIn("تعارض", state["messages"][-1]["text"])

    def test_a_real_change_in_start_time_is_preserved_as_a_specific_user_conflict(self):
        self.establish_timing_question()
        first = self.turn("أول رسالة وصلتني مبارح حوالي الساعة 6 المسا.")
        self.assertEqual(first["session"]["facts"]["initial_time"]["value"], "مبارح حوالي الساعة 6 المسا")
        changed = self.turn("للتوضيح، أول رسالة وصلتني اليوم الساعة 9 الصبح.")
        fact = changed["session"]["facts"]["initial_time"]
        self.assertEqual(fact["status"], "conflict")
        self.assertEqual({item["value"] for item in fact["alternatives"] if item.get("value")},
                         {"مبارح حوالي الساعة 6 المسا", "اليوم الساعة 9 الصبح"})
        self.assertIn("أي وقت تقصد؟", changed["session"]["messages"][-1]["text"])
        self.assertNotIn("تعارضت إفادتك", changed["session"]["messages"][-1]["text"])
        self.assertFalse(changed["session"]["confirmed"])

    def test_revision_checked_resolution_preserves_conflict_history_and_reads_are_read_only(self):
        self.establish_timing_question()
        self.turn("أول رسالة وصلتني مبارح حوالي الساعة 6 المسا.")
        conflicted = self.turn("للتوضيح، أول رسالة وصلتني اليوم الساعة 9 الصبح.")
        fact = conflicted["session"]["facts"]["initial_time"]
        self.assertEqual(fact["status"], "conflict")
        initial_revision = self.revision
        before_history = len(self.client.get(f"/api/sessions/{self.session_id}/audit", headers=self.headers).json()["events"])
        current = self.client.get(f"/api/sessions/{self.session_id}", headers=self.headers)
        report = self.client.get(f"/api/sessions/{self.session_id}/report.json", headers=self.headers)
        self.assertEqual(current.json()["session"]["revision"], initial_revision)
        self.assertEqual(len(self.client.get(f"/api/sessions/{self.session_id}/audit", headers=self.headers).json()["events"]), before_history)
        self.assertEqual(report.json()["facts"]["initial_time"]["status"], "conflict")

        endpoint = f"/api/sessions/{self.session_id}/resolve-temporal-conflict"
        stale = self.client.post(endpoint, headers=self.headers, json={
            "expected_revision": initial_revision - 1, "field": "initial_time",
            "selected_value": "مبارح حوالي الساعة 6 المسا", "resolution_reason": "هذه إفادة البداية الصحيحة",
        })
        self.assertEqual(stale.status_code, 409)

        resolved = self.client.post(endpoint, headers=self.headers, json={
            "expected_revision": initial_revision, "field": "initial_time",
            "selected_value": "مبارح حوالي الساعة 6 المسا", "resolution_reason": "هذه إفادة البداية الصحيحة",
        })
        self.assertEqual(resolved.status_code, 200, resolved.text)
        state = resolved.json()["session"]
        self.revision = state["revision"]
        fact = state["facts"]["initial_time"]
        self.assertEqual(fact["status"], "reported")
        self.assertEqual(fact["value"], "مبارح حوالي الساعة 6 المسا")
        self.assertIn("اليوم الساعة 9 الصبح", str(fact["history"][-1]["alternatives"]))
        self.assertEqual(fact["history"][-1]["reason"], "resolved_by_user_selection")
        self.assertEqual(fact["resolution"]["reason"], "هذه إفادة البداية الصحيحة")
        self.assertFalse(state["confirmed"])
        after_resolution_revision = state["revision"]
        after_resolution_history = len(self.client.get(f"/api/sessions/{self.session_id}/audit", headers=self.headers).json()["events"])
        self.client.get(f"/api/sessions/{self.session_id}", headers=self.headers)
        self.client.get(f"/api/sessions/{self.session_id}/report.json", headers=self.headers)
        self.assertEqual(self.client.get(f"/api/sessions/{self.session_id}", headers=self.headers).json()["session"]["revision"],
                         after_resolution_revision)
        self.assertEqual(len(self.client.get(f"/api/sessions/{self.session_id}/audit", headers=self.headers).json()["events"]),
                         after_resolution_history)

    def test_actual_rag_platform_clarification_resolves_after_refresh_and_resumes_interview(self):
        ask = self.turn("كيف أحمي حسابي")
        state = ask["session"]
        self.assertEqual(state["pending_slot"], "narrative")
        self.assertEqual(state.get("pending_guidance", {}).get("status"), "awaiting_platform")
        self.assertEqual(state["messages"][-1]["kind"], "guidance_clarification")
        self.assertNotEqual(state["messages"][-1]["kind"], "question")
        fetched = self.client.get(f"/api/sessions/{self.session_id}", headers=self.headers)
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["session"]["pending_guidance"]["query"], "كيف أحمي حسابي")

        paused = self.turn("احفظ الجلسة")
        self.assertEqual(paused["session"]["stage"], "paused")
        resumed = self.client.post(f"/api/sessions/{self.session_id}/resume", headers=self.headers,
                                   json={"expected_revision": self.revision})
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.revision = resumed.json()["session"]["revision"]
        self.assertEqual(resumed.json()["session"]["messages"][-1]["kind"], "guidance_clarification")
        self.assertEqual(resumed.json()["session"]["pending_guidance"]["status"], "awaiting_platform")

        resolved = self.turn("واتساب")
        state = resolved["session"]
        answer = next(message for message in reversed(state["messages"])
                      if message["role"] == "assistant" and message["kind"] == "answer")
        self.assertIsNone(state.get("pending_guidance"))
        self.assertEqual(state["pending_slot"], "narrative")
        self.assertNotIn("account_security", state["facts"])
        ids = {citation["id"] for citation in answer["citations"]}
        self.assertIn("GLOBAL-WA-ACCOUNT-SECURITY-2026", ids)
        self.assertNotIn("GLOBAL-IG-SECURITY-2026", ids)
        self.assertIn("خطوتين", answer["text"])
        self.assertIn("نكمل من السؤال السابق", state["messages"][-1]["text"])

        report = self.client.get(f"/api/sessions/{self.session_id}/report.json", headers=self.headers).json()
        self.assertEqual(report["facts"].get("platform", {}).get("status"), None,
                         "a platform named only to disambiguate guidance is not a user incident fact")

    def test_full_synthetic_interview_stays_consistent_through_review_confirmation_and_exports(self):
        first = self.turn("في شخص على واتساب طلب مني 100 دينار، وقال رح ينشر محادثاتي الخاصة إذا ما دفعت.")
        self.assertEqual(first["session"]["pending_slot"], "direct_danger")
        self.assertEqual(first["session"]["facts"]["requested_amount"]["value"], "100 دينار")
        self.assertNotIn("direct_danger", first["session"]["facts"])

        second = self.turn("لا، ما في خطر مباشر علي الآن، بس اليوم رجع بعتلي نفس التهديد.")
        self.assertEqual(second["session"]["facts"]["direct_danger"]["status"], "denied")
        self.assertEqual(second["session"]["facts"]["latest_time"]["value"], "اليوم")
        self.assertEqual(second["session"]["pending_slot"], "initial_time")

        third_text = "أول رسالة وصلتني مبارح حوالي الساعة 6 المسا، أما آخر تهديد فكان اليوم."
        third = self.turn(third_text)
        state = third["session"]
        self.assertEqual(state["facts"]["initial_time"]["value"], "مبارح حوالي الساعة 6 المسا")
        self.assertEqual(state["facts"]["latest_time"]["value"], "اليوم")
        self.assertIn("أول رسالة وصلتني مبارح حوالي الساعة 6 المسا", state["facts"]["initial_time"]["quote"])
        self.assertEqual(state["pending_slot"], "paid_amount")

        fourth = self.turn("دفعت 130 دينار عن طريق محفظة إلكترونية.")
        self.assertEqual(fourth["session"]["facts"]["paid_amount"]["value"], "130 دينار")
        self.assertEqual(fourth["session"]["facts"]["payment_method"]["value"], "محفظة إلكترونية")
        self.assertEqual(fourth["session"]["pending_slot"], "deadline")

        guidance = self.turn("كيف أحمي حسابي على واتساب؟")
        state = guidance["session"]
        answer = next(message for message in reversed(state["messages"])
                      if message["role"] == "assistant" and message["kind"] == "answer")
        cited_ids = {citation["id"] for citation in answer["citations"]}
        self.assertIn("GLOBAL-WA-TWO-STEP-2026", cited_ids)
        self.assertNotIn("GLOBAL-IG-SECURITY-2026", cited_ids)
        self.assertIn("التحقق بخطوتين", answer["text"])
        self.assertEqual(state["pending_slot"], "deadline")
        self.assertEqual(state["facts"]["paid_amount"]["value"], "130 دينار")

        combined = self.turn("ما بعرف هويته الحقيقية، بس عندي نسخة محفوظة من المحادثة.")
        state = combined["session"]
        self.assertEqual(state["facts"]["identity"]["status"], "unknown")
        self.assertEqual(state["facts"]["evidence"]["status"], "reported")
        self.assertIn("نسخة محفوظة من المحادثة", state["facts"]["evidence"]["quote"])
        self.assertEqual(state["pending_slot"], "deadline")
        self.assertIn("سجلت المعلومات التي أجابت عن حقول أخرى", state["messages"][-1]["text"])

        publication = self.turn("ما بعرف إذا نشر شي فعلاً.")
        self.assertEqual(publication["session"]["facts"]["published"]["status"], "unknown")
        self.assertEqual(publication["session"]["pending_slot"], "deadline")
        self.assertIn("مهلة أو موعدًا للدفع", publication["session"]["messages"][-1]["text"])

        no_deadline = self.turn("ما حدد مهلة للدفع.")
        state = no_deadline["session"]
        self.assertEqual(state["facts"]["deadline"]["status"], "denied")
        self.assertEqual(state["facts"]["published"]["status"], "unknown")
        self.assertEqual(state["stage"], "review")

        correction_quote = "تصحيح: المبلغ المطلوب 120 دينار، والمدفوع بقي 130."
        corrected = self.turn(correction_quote)
        state = corrected["session"]
        self.assertEqual(state["facts"]["requested_amount"]["value"], "120 دينار")
        self.assertEqual(state["facts"]["paid_amount"]["value"], "130 دينار")
        self.assertEqual(state["facts"]["requested_amount"]["history"][-1]["value"], "100 دينار")
        self.assertEqual(state["facts"]["requested_amount"]["history"][-1]["quote"],
                         "في شخص على واتساب طلب مني 100 دينار، وقال رح ينشر محادثاتي الخاصة إذا ما دفعت.")
        paid_events = [event for event in state["money_events"]
                       if event.get("kind") == "paid" and event.get("relation") not in {"correction", "unresolved"}]
        self.assertEqual(len(paid_events), 1)
        self.assertEqual(paid_events[0]["amount"], "130 دينار")
        self.assertTrue(any(item.get("kind") == "requested" and item.get("amount") == "120 دينار"
                            and item.get("previous_amount") == "100 دينار"
                            for item in state.get("money_corrections", [])))

        explanation = self.turn("ليش الحالة إلها أولوية؟")
        self.assertIn("قاعدة فرز تجريبية", explanation["session"]["messages"][-1]["text"])
        self.assertEqual(explanation["session"]["facts"]["requested_amount"]["value"], "120 دينار")

        shown = self.turn("اعرض الملخص")
        self.assertEqual(shown["session"]["stage"], "review")
        self.assertFalse(shown["session"]["confirmed"])
        confirmed = self.client.post(f"/api/sessions/{self.session_id}/confirm", headers=self.headers,
                                     json={"expected_revision": self.revision})
        self.assertEqual(confirmed.status_code, 200, confirmed.text)
        state = confirmed.json()["session"]
        self.revision = state["revision"]
        self.assertTrue(confirmed.json()["report"]["confirmed"])
        confirmed_digest = state["confirmed_digest"]

        paused = self.turn("احفظ الجلسة")
        self.assertEqual(paused["session"]["stage"], "paused")
        self.assertTrue(paused["session"]["confirmed"])
        resumed = self.client.post(f"/api/sessions/{self.session_id}/resume", headers=self.headers,
                                   json={"expected_revision": self.revision})
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.revision = resumed.json()["session"]["revision"]
        self.assertEqual(resumed.json()["session"]["confirmed_digest"], confirmed_digest)

        report = self.client.get(f"/api/sessions/{self.session_id}/report.json", headers=self.headers)
        self.assertEqual(report.status_code, 200)
        data = report.json()
        self.assertTrue(data["confirmed"])
        self.assertEqual(data["facts"]["requested_amount"]["value"], "120 دينار")
        self.assertEqual(data["facts"]["paid_amount"]["value"], "130 دينار")
        self.assertIn(correction_quote, data["facts"]["requested_amount"]["quote"])
        markdown = self.client.get(f"/api/sessions/{self.session_id}/report.md", headers=self.headers)
        self.assertEqual(markdown.status_code, 200)
        self.assertIn("120 دينار", markdown.text)
        self.assertIn("130 دينار", markdown.text)
        self.assertIn("نسخة محفوظة من المحادثة", markdown.text)

    def test_multiple_provider_calls_keep_independent_purpose_and_latency_without_raw_data(self):
        class FakeResponse:
            def __init__(self, content):
                self.content = content

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": self.content}}]}).encode()

        with patch.dict(os.environ, {"MODEL_PROVIDER": "ollama", "MODEL_NAME": "qwen3.5:4b-q4_K_M",
                                     "MODEL_BASE_URL": "http://127.0.0.1:11434/v1"}), \
             patch("app.model.urllib.request.urlopen", side_effect=[FakeResponse("{}"), FakeResponse("{}")]):
            reset_call_trace()
            self.assertEqual(_chat([], purpose="interview_extraction"), "{}")
            self.assertEqual(_chat([], purpose="grounded_answer"), "{}")
        traces = call_trace_history()
        self.assertEqual([item["purpose"] for item in traces], ["interview_extraction", "grounded_answer"])
        self.assertTrue(all(item["attempted"] and isinstance(item["latency_ms"], (float, int)) for item in traces))
        self.assertTrue(all("prompt" not in item and "raw_response" not in item for item in traces))

    def test_consented_mixed_turn_returns_extraction_and_grounded_answer_call_traces(self):
        user_text = "في شخص يهددني على واتساب. كيف أحمي حسابي على واتساب؟"
        understanding = InterviewUnderstanding(
            [{"field": "narrative", "status": "reported", "value": "في شخص يهددني على واتساب",
              "quote": "في شخص يهددني على واتساب."}],
            {"goals": ["add_fact", "ask_guidance"], "guidance_question": "كيف أحمي حسابي على واتساب؟"},
            {"accepted": True},
        )
        traces = [
            {"purpose": "interview_extraction", "attempted": True, "status": "extracted",
             "result_accepted": True, "latency_ms": 12.0},
            {"purpose": "grounded_answer", "attempted": True, "status": "accepted",
             "result_accepted": True, "latency_ms": 23.0},
        ]
        with patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value={"provider": "ollama", "model": "fixture", "mode": "local_model"}), \
             patch("app.main.generate_interview_extraction", return_value=understanding), \
             patch("app.main.generate_grounded_answer", return_value={
                 "text": "فعّل التحقق بخطوتين من إعدادات واتساب.",
                 "source_ids": ["GLOBAL-WA-TWO-STEP-2026"],
             }) as answer_generator, \
             patch("app.main.call_trace_history", return_value=traces), \
             patch("app.main.last_call_trace", return_value=traces[-1]):
            result = self.turn(user_text, allow_external_extraction=True, allow_external=True)

        self.assertTrue(answer_generator.called)
        self.assertEqual([call["purpose"] for call in result["provider_calls"]],
                         ["interview_extraction", "grounded_answer"])
        self.assertTrue(all(call["latency_ms"] > 0 for call in result["provider_calls"]))
        self.assertFalse(result["extraction_fallback_used"])
        self.assertFalse(result["guidance_fallback_used"])
        self.assertEqual(result["session"]["facts"]["narrative"]["status"], "reported")
        answer = next(message for message in reversed(result["session"]["messages"])
                      if message.get("kind") == "answer")
        self.assertIn("خطوتين", answer["text"])
        self.assertEqual({source["id"] for source in answer["citations"]}, {"GLOBAL-WA-TWO-STEP-2026"})


if __name__ == "__main__":
    unittest.main()
