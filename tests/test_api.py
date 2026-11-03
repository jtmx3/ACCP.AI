import tempfile
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi import HTTPException

from app import main
from app.main import TurnRequest
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

    def test_partial_model_filter_status_is_safe_and_visible_to_client(self):
        facts = [{"field": "platform", "status": "reported", "value": "إنستغرام",
                  "quote": "إنستغرام"}]
        trace = {"attempted": True, "status": "partially_accepted", "result_accepted": True,
                 "fallback_used": False, "failure_code": "partial_fact_filter", "filtered_fact_count": 1,
                 "filtered_fact_fields": ["identity"], "latency_ms": 12}
        with patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value={"provider": "openai-compatible",
                                                               "model": "synthetic-test", "mode": "external_model"}), \
             patch("app.main.generate_interview_extraction", return_value=facts), \
             patch("app.main.last_call_trace", return_value=trace):
            response = self.client.post(
                f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                json={"expected_revision": self.revision, "text": "حساب مزيف على إنستغرام",
                      "kind": "answer", "allow_external_extraction": True})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["external_model_result_accepted"])
        self.assertEqual(result["provider_call"]["filtered_fact_count"], 1)
        self.assertEqual(result["provider_call"]["filtered_fact_fields"], ["identity"])
        self.assertNotIn("filtered_fact_proposals", result["provider_call"])

    def test_valid_empty_and_all_filtered_model_extractions_fall_back_without_claiming_success(self):
        cases = [
            ({"status": "valid_empty", "extraction_outcome": "valid_empty", "response_valid": True,
              "result_accepted": False, "fallback_used": False, "failure_code": "empty_extraction",
              "filtered_fact_count": 0, "filtered_fact_fields": [], "filtered_fact_code_counts": {}},
             "valid_empty"),
            ({"status": "all_facts_filtered", "extraction_outcome": "all_facts_filtered", "response_valid": True,
              "result_accepted": False, "fallback_used": False, "failure_code": "all_facts_filtered",
              "filtered_fact_count": 1, "filtered_fact_fields": ["narrative"],
              "filtered_fact_code_counts": {"value_inconsistent": 1}}, "all_facts_filtered"),
        ]
        for trace, expected_outcome in cases:
            with self.subTest(outcome=expected_outcome), \
                 patch("app.main.model_configured", return_value=True), \
                 patch("app.main.provider_config", return_value={"provider": "ollama", "model": "fixture:latest",
                                                                    "mode": "local_model"}), \
                 patch("app.main.generate_interview_extraction", return_value=[]), \
                 patch("app.main.last_call_trace", return_value=trace):
                response = self.client.post(
                    f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                    json={"expected_revision": self.revision, "text": "في شخص عم يهددني على واتساب.",
                          "kind": "answer", "allow_external_extraction": True})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.revision = result["session"]["revision"]
            self.assertFalse(result["external_model_result_accepted"])
            self.assertTrue(result["fallback_used"])
            self.assertEqual(result["fallback_to"], "rules")
            self.assertEqual(result["extraction_outcome"], expected_outcome)
            self.assertTrue(result["provider_call"]["response_valid"])
            self.assertFalse(result["provider_call"]["result_accepted"])
            if expected_outcome == "all_facts_filtered":
                self.assertEqual(result["provider_call"]["filtered_fact_code_counts"], {"value_inconsistent": 1})

    def test_clear_question_in_answer_mode_is_answered_locally_then_pending_question_resumes(self):
        before = self.turn("في شخص يهددني على واتساب")
        self.assertEqual(before["pending_slot"], "direct_danger")
        facts_before = before["facts"]
        env = {"MODEL_PROVIDER": "openai-compatible", "MODEL_BASE_URL": "https://provider.example/v1",
               "MODEL_NAME": "fixture-model", "MODEL_API_KEY": "test-only"}
        with patch.dict(os.environ, env), patch("app.main.generate_interview_extraction",
                                                side_effect=AssertionError("answer-mode question must not be sent for extraction")):
            response = self.client.post(f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                                        json={"expected_revision": self.revision, "text": "كيف أحفظ الأدلة واللقطات؟",
                                              "kind": "answer", "allow_external_extraction": True})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        state = payload["session"]
        self.revision = state["revision"]
        self.assertTrue(payload["question_detected"])
        self.assertFalse(payload["external_data_sent"])
        self.assertEqual(state["pending_slot"], "direct_danger")
        self.assertEqual(state["facts"], facts_before)
        self.assertEqual(state["messages"][-2]["kind"], "answer")
        self.assertIn("ما عندي مصدر موثوق", state["messages"][-2]["text"])
        self.assertEqual(state["messages"][-1]["kind"], "question")
        self.assertIn("هل أنت بخطر مباشر الآن؟", state["messages"][-1]["text"])

    def test_immediate_danger_in_question_mode_bypasses_rag(self):
        state = self.turn("الشخص وصل لعندي الآن ومعه سلاح، شو أعمل؟", kind="question")
        self.assertEqual(state["stage"], "urgent_stop")
        self.assertEqual(state["facts"]["direct_danger"]["value"], True)
        safety = state["messages"][-1]
        self.assertEqual(safety["kind"], "safety")
        self.assertEqual(safety["citations"], [])
        self.assertFalse(safety["external_model_used"])

    def test_unresolved_money_relation_blocks_confirmation_until_resolved(self):
        for text in ("في شخص يهددني", "لا", "قال ادفع 100 دينار", "طلب 120 دينار أيضًا"):
            self.turn(text)
        self.assertEqual(self.turn("اعرض الملخص")["stage"], "review")
        pending_report = self.client.get(f"/api/sessions/{self.case_id}/report.json", headers=self.headers).json()
        self.assertTrue(pending_report["unresolved_money_events"])
        self.assertIn("requested_amount_relation", pending_report["missing_fields"])
        self.assertIn("علاقة مبلغ غير محسومة", self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers).text)
        blocked = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers, json={"expected_revision": self.revision})
        self.assertEqual(blocked.status_code, 422)
        corrected = self.turn("المبلغ الجديد إضافي")
        self.assertEqual(corrected["facts"]["requested_amount"]["status"], "reported")
        self.assertEqual(corrected["money_events"][-1]["relation"], "additional")
        resolved_report = self.client.get(f"/api/sessions/{self.case_id}/report.json", headers=self.headers).json()
        self.assertEqual(resolved_report["unresolved_money_events"], [])
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
        self.assertIn("غير معروفة", self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers).text)
        self.assertIn("- الحالة:", self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers).text)
        self.assertIn("- القيمة:", self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers).text)
        report = self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers)
        self.assertEqual(report.status_code, 200)
        self.assertIn("100 دينار", report.text)
        self.assertIn("130 دينار", report.text)
        confirmed = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers, json={"expected_revision": self.revision})
        self.assertEqual(confirmed.status_code, 200)
        self.revision = confirmed.json()["session"]["revision"]
        self.assertTrue(confirmed.json()["session"]["confirmed"])
        confirmed_version = confirmed.json()["session"]["summary_version"]
        confirmed_json = self.client.get(f"/api/sessions/{self.case_id}/report.json", headers=self.headers)
        self.assertEqual(confirmed_json.status_code, 200)
        self.assertIn(".json", confirmed_json.headers["content-disposition"])
        confirmed_report = confirmed_json.json()
        self.assertEqual(confirmed_report["summary_version"], confirmed_version)
        self.assertTrue(confirmed_report["confirmed"])
        self.assertEqual(confirmed_report["field_labels"]["platform"], "المنصة")
        self.assertIn("published", confirmed_report["unknown_fields"])
        self.assertTrue(confirmed_report["money_events"])
        self.assertIn("source_message_at", confirmed_report["facts"]["requested_amount"])
        self.assertIn("last_updated_at", confirmed_report["facts"]["requested_amount"])
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
        latest_json = self.client.get(f"/api/sessions/{self.case_id}/report.json", headers=self.headers).json()
        self.assertEqual(latest_json["summary_version"], corrected["summary_version"])
        self.assertFalse(latest_json["confirmed"])
        self.assertEqual(latest_json["facts"]["requested_amount"]["value"], "120 دينار")
        self.assertTrue(latest_json["facts"]["requested_amount"]["source_message_at"])
        self.assertTrue(latest_json["facts"]["requested_amount"]["last_updated_at"])

    def test_confirmation_is_invalidated_when_resumed_turn_changes_facts_before_clarification_return(self):
        self.turn("في شخص يهددني")
        self.turn("اعرض الملخص")
        confirmed = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers, json={"expected_revision": self.revision})
        self.assertEqual(confirmed.status_code, 200)
        self.revision = confirmed.json()["session"]["revision"]
        prior = confirmed.json()["session"]
        self.assertTrue(prior["confirmed"])
        prior_version = prior["summary_version"]
        prior_digest = prior["confirmed_digest"]

        paused = self.turn("احفظ الجلسة")
        self.assertEqual(paused["stage"], "paused")
        self.assertTrue(paused["confirmed"])
        resumed = self.client.post(f"/api/sessions/{self.case_id}/resume", headers=self.headers, json={"expected_revision": self.revision})
        self.assertEqual(resumed.status_code, 200)
        self.revision = resumed.json()["session"]["revision"]
        self.assertEqual(resumed.json()["session"]["pending_slot"], "direct_danger")
        self.assertTrue(resumed.json()["session"]["confirmed"])

        updated = self.turn("طلب مني 100 دينار")
        self.assertEqual(updated["facts"]["requested_amount"]["value"], "100 دينار")
        self.assertEqual(updated["pending_slot"], "direct_danger")
        self.assertFalse(updated["confirmed"])
        self.assertIsNone(updated["confirmed_digest"])
        self.assertGreater(updated["summary_version"], prior_version)
        self.assertEqual(updated["confirmation_history"][0]["digest"], prior_digest)

        current = self.client.get(f"/api/sessions/{self.case_id}", headers=self.headers).json()
        self.assertFalse(current["session"]["confirmed"])
        self.assertFalse(current["report"]["confirmed"])
        self.assertEqual(current["report"]["confirmation_history"][0]["summary_version"], prior_version)
        exported = self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers)
        self.assertEqual(exported.status_code, 200)
        self.assertIn("**التأكيد:** غير مؤكد", exported.text)
        self.assertIn("تأكيد سابق؛ لا يؤكد محتوى هذا الإصدار", exported.text)
        self.assertNotIn("**التأكيد:** مؤكد من المستخدم", exported.text)

    def test_read_and_export_never_claim_confirmation_for_a_mismatched_legacy_digest(self):
        self.turn("في شخص يهددني")
        self.turn("اعرض الملخص")
        confirmed = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers, json={"expected_revision": self.revision})
        self.assertEqual(confirmed.status_code, 200)
        self.revision = confirmed.json()["session"]["revision"]
        confirmed_version = confirmed.json()["session"]["summary_version"]

        def legacy_unversioned_change(state):
            state["facts"]["additional_details"] = {"field": "additional_details", "status": "reported",
                                                       "value": "تفصيل أضيف خارج مسار الإصدار", "quote": "تفصيل أضيف خارج مسار الإصدار",
                                                       "message_id": "legacy-change", "history": []}

        stale = main.store.mutate(self.case_id, self.token, self.revision, "legacy_unversioned_change", legacy_unversioned_change)
        self.revision = stale["revision"]
        fetched = self.client.get(f"/api/sessions/{self.case_id}", headers=self.headers).json()
        self.assertFalse(fetched["session"]["confirmed"])
        self.assertFalse(fetched["report"]["confirmed"])
        self.assertEqual(fetched["report"]["confirmation_history"][0]["summary_version"], confirmed_version)
        report = self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers).text
        self.assertIn("**التأكيد:** غير مؤكد", report)
        self.assertIn("تأكيد سابق؛ لا يؤكد محتوى هذا الإصدار", report)

    def test_why_question_bypasses_rag_and_does_not_answer_the_safety_slot(self):
        before = self.turn("في شخص يهددني")
        self.assertEqual(before["pending_slot"], "direct_danger")
        facts_before = before["facts"]
        version_before = before["summary_version"]
        with patch("app.main.answer_question", side_effect=AssertionError("why intent must not call RAG")):
            response = self.client.post(f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                                        json={"expected_revision": self.revision, "text": "ليش بتسأل عن الخطر؟", "kind": "question"})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        state = payload["session"]
        self.revision = state["revision"]
        self.assertFalse(payload["external_data_sent"])
        self.assertFalse(payload["external_model_used"])
        self.assertEqual(state["pending_slot"], "direct_danger")
        self.assertEqual(state["facts"], facts_before)
        self.assertEqual(state["summary_version"], version_before)
        self.assertIn("خطوة سلامة عاجلة", state["messages"][-2]["text"])

        answer_mode = self.turn("ليش بتسأل عن الخطر؟", kind="answer")
        self.assertEqual(answer_mode["pending_slot"], "direct_danger")
        self.assertEqual(answer_mode["facts"], facts_before)
        self.assertEqual(answer_mode["summary_version"], version_before)
        self.assertIn("خطوة سلامة عاجلة", answer_mode["messages"][-2]["text"])
        self.assertIn("هل أنت بخطر مباشر الآن؟", answer_mode["messages"][-1]["text"])

    def test_two_consented_extraction_calls_overlap_across_sessions(self):
        second_state, second_token = main.store.create()
        barrier = threading.Barrier(2, timeout=4)
        calls = []
        calls_lock = threading.Lock()

        def slow_extract(text, pending):
            with calls_lock:
                calls.append((text, pending))
            barrier.wait()
            return []

        with patch.dict(os.environ, {"MODEL_PROVIDER": "openai-compatible", "MODEL_API_KEY": "test-only"}), \
             patch("app.main.generate_interview_extraction", side_effect=slow_extract):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(main.post_message, self.case_id, TurnRequest(expected_revision=0, text="شخص يهددني برسالة", allow_external_extraction=True), self.token),
                    pool.submit(main.post_message, second_state["session_id"], TurnRequest(expected_revision=0, text="حساب مزيف باسمي", allow_external_extraction=True), second_token),
                ]
                results = [future.result(timeout=8) for future in futures]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(result["external_data_sent"] for result in results))
        self.assertEqual({result["session"]["revision"] for result in results}, {1})

    def test_stale_result_after_provider_wait_is_discarded_without_retry(self):
        entered = threading.Event()
        release = threading.Event()
        call_count = 0

        def slow_extract(text, pending):
            nonlocal call_count
            call_count += 1
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            return [{"field": "platform", "status": "reported", "value": "واتساب", "quote": "واتساب"}]

        with patch.dict(os.environ, {"MODEL_PROVIDER": "openai-compatible", "MODEL_API_KEY": "test-only"}), \
             patch("app.main.generate_interview_extraction", side_effect=slow_extract):
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(main.post_message, self.case_id,
                                    TurnRequest(expected_revision=0, text="شخص يهددني على واتساب", allow_external_extraction=True), self.token)
                self.assertTrue(entered.wait(timeout=3))
                concurrent = main.post_message(self.case_id, TurnRequest(expected_revision=0, text="شخص يهددني على واتساب"), self.token)
                self.assertEqual(concurrent["session"]["revision"], 1)
                release.set()
                with self.assertRaises(HTTPException) as stale:
                    first.result(timeout=5)
        self.assertEqual(stale.exception.status_code, 409)
        self.assertEqual(call_count, 1)
        saved = main.store.get(self.case_id, self.token)
        self.assertEqual(saved["revision"], 1)

    def test_slow_grounded_answer_does_not_block_a_different_session_write(self):
        second_state, second_token = main.store.create()
        for session_id, token in ((self.case_id, self.token), (second_state["session_id"], second_token)):
            main.store.mutate(session_id, token, 0, "set_jurisdiction", lambda state: state.update({"jurisdiction": "AU"}))
        entered = threading.Event()
        release = threading.Event()
        second_done = threading.Event()

        def slow_answer(question, sources):
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            return None

        def write_second():
            result = main.post_message(second_state["session_id"], TurnRequest(expected_revision=1, text="شخص يهددني برسالة"), second_token)
            second_done.set()
            return result

        with patch.dict(os.environ, {"MODEL_PROVIDER": "openai-compatible", "MODEL_API_KEY": "test-only"}), \
             patch("app.main.generate_grounded_answer", side_effect=slow_answer):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(main.post_message, self.case_id,
                                    TurnRequest(expected_revision=1, text="كيف أحفظ الأدلة واللقطات؟", kind="question", allow_external=True), self.token)
                self.assertTrue(entered.wait(timeout=3))
                second = pool.submit(write_second)
                wrote_while_waiting = second_done.wait(timeout=2)
                release.set()
                a_result = first.result(timeout=5)
                b_result = second.result(timeout=5)
        self.assertTrue(wrote_while_waiting, "a slow provider call held SQLite's write lock")
        self.assertTrue(a_result["external_data_sent"])
        self.assertFalse(a_result["external_model_used"])
        self.assertEqual(b_result["session"]["revision"], 2)


if __name__ == "__main__":
    unittest.main()
