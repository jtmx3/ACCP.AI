import io
import json
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
        self.assertIn("المصادر الوطنية لا تمثل مسارًا محليًا خارجه", answer_message["text"])
        self.assertTrue(any(source["id"] == "AU-ESAFETY-EVIDENCE-2026" for source in answer_message["citations"]))
        self.assertTrue(all(source["scope_note"] for source in answer_message["citations"]))
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

    def test_all_filtered_model_diagnostics_separate_provider_validation_merge_and_fallback(self):
        trace = {"attempted": True, "status": "all_facts_filtered", "result_accepted": False,
                 "failure_code": "all_facts_filtered", "latency_ms": 321,
                 "extraction_outcome": "all_facts_filtered", "response_valid": True,
                 "filtered_fact_count": 1, "filtered_fact_fields": ["deadline"]}
        diagnostic = {"accepted": True, "raw_response": '{"facts":[{"private":"do not return"}]}',
                      "filtered_facts": [{"field": "deadline", "status": "reported", "value": "غدًا",
                                          "quote": "اقتباس خاص", "filter_code": "quote_mismatch",
                                          "filter_detail": "The quote does not match a current message span."}]}
        with patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value={"provider": "ollama", "model": "fixture:latest",
                                                                "mode": "local_model"}), \
             patch("app.main.generate_interview_extraction", return_value=[]), \
             patch("app.main.last_call_trace", return_value=trace), \
             patch("app.main.last_extraction_diagnostic", return_value=diagnostic):
            response = self.client.post(
                f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                json={"expected_revision": self.revision, "text": "لا، ما حدد مهلة أو موعد للدفع.",
                      "kind": "answer", "allow_external_extraction": True})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        detail = result["model_diagnostics"]
        self.assertTrue(detail["provider_response"]["received"])
        self.assertEqual(detail["provider_response"]["status"], "all_facts_filtered")
        self.assertTrue(detail["validation"]["response_valid"])
        self.assertEqual(detail["validation"]["filtered_facts"], [{
            "field": "deadline", "field_label": "مهلة الدفع", "code": "quote_mismatch",
            "detail": "The quote does not match a current message span."}])
        self.assertFalse(detail["merge"]["performed"])
        self.assertTrue(detail["fallback"]["used"])
        self.assertFalse(detail["raw_response_included"])
        self.assertNotIn("private", response.text)
        self.assertNotIn("اقتباس خاص", response.text)

    def test_negated_or_unknown_model_proposals_are_filtered_before_state_and_report(self):
        cases = [
            ("ما حصل نشر لصور الحساب.", "published", "reported", True, "value_inconsistent"),
            ("ما صار أي تهديد مباشر.", "threat", "reported", True, "value_inconsistent"),
            ("ما دفعت أي مبلغ.", "paid_amount", "unknown", None, "unsupported_status_evidence"),
            ("ما نشر الصور وهددني بنشرها.", "threat", "denied", False,
             "unsupported_status_evidence"),
            ("هددني بنشر الصور.", "published", "reported", True, "value_inconsistent"),
            ("ما صار معي تهديد.", "threat", "reported", True, "value_inconsistent"),
            ("رح ينزل الصور بكرا.", "published", "reported", True, "value_inconsistent"),
        ]
        env = {"MODEL_PROVIDER": "ollama", "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
               "MODEL_NAME": "synthetic-test:latest", "OLLAMA_REASONING_EFFORT": "none"}
        for text, field, status, value, expected_code in cases:
            with self.subTest(field=field):
                created = self.client.post("/api/sessions", json={}).json()
                session_id = created["session"]["session_id"]
                token = created["token"]
                revision = created["session"]["revision"]
                model_payload = {"message_goals": [], "guidance_question": None,
                                 "facts": [{"field": field, "status": status, "value": value,
                                            "current_quote": text}]}
                provider_payload = {"choices": [{"message": {"content": json.dumps(
                    model_payload, ensure_ascii=False)}}]}
                calls = []

                def fake_urlopen(request, timeout):
                    calls.append(request)
                    return io.BytesIO(json.dumps(provider_payload, ensure_ascii=False).encode("utf-8"))

                with patch.dict(os.environ, env, clear=True), \
                     patch("app.main.model_configured", return_value=True), \
                     patch("app.main.provider_config", return_value={"provider": "ollama",
                           "model": "synthetic-test:latest", "mode": "local_model"}), \
                     patch("app.model.urllib.request.urlopen", side_effect=fake_urlopen):
                    response = self.client.post(
                        f"/api/sessions/{session_id}/messages", headers={"X-Case-Token": token},
                        json={"expected_revision": revision, "text": text, "kind": "answer",
                              "allow_external_extraction": True})

                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(len(calls), 1)
                self.assertFalse(result["external_model_result_accepted"])
                self.assertTrue(result["extraction_fallback_used"])
                self.assertEqual(result["fallback_to"], "rules")
                validation = result["model_diagnostics"]["validation"]
                self.assertEqual(validation["accepted_fields"], [])
                self.assertEqual(validation["filtered_facts"][0]["field"], field)
                self.assertEqual(validation["filtered_facts"][0]["code"], expected_code)
                self.assertFalse(result["model_diagnostics"]["merge"]["performed"])
                for facts in (result["session"].get("facts", {}), result["report"].get("facts", {})):
                    stored = facts.get(field)
                    self.assertFalse(stored and stored.get("status") == status and stored.get("value") == value)

    def test_unsupported_decline_goal_is_diagnostic_only_and_rules_handle_deadline_denial(self):
        from app.model import InterviewUnderstanding

        def make_deadline_pending(state):
            state["stage"] = "interview"
            state["pending_slot"] = "deadline"

        pending = main.store.mutate(self.case_id, self.token, self.revision,
                                    "synthetic_test_setup", make_deadline_pending)
        self.revision = pending["revision"]
        message = "لا، ما حدد مهلة أو موعد للدفع."
        trace = {"attempted": True, "status": "valid_empty", "result_accepted": False,
                 "response_valid": True, "extraction_outcome": "valid_empty", "failure_code": "empty_extraction",
                 "latency_ms": 98, "fact_fields": [], "filtered_fact_count": 0,
                 "unsupported_message_goals": ["decline_disclosure"]}
        diagnostic = {"accepted": True, "raw_response": '{"message_goals":["decline_disclosure"],"facts":[]}',
                      "filtered_facts": [], "unsupported_message_goals": ["decline_disclosure"],
                      "message_plan": {"goals": [], "unsupported_goals": ["decline_disclosure"],
                                       "guidance_question": None}}
        understanding = InterviewUnderstanding([], diagnostic["message_plan"], diagnostic)
        config = {"provider": "ollama", "model": "fixture:qwen", "mode": "local_model", "local": True}
        with patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value=config), \
             patch("app.main.generate_interview_extraction", return_value=understanding), \
             patch("app.main.last_call_trace", return_value=trace), \
             patch("app.main.call_trace_history", return_value=[trace]), \
             patch("app.main.last_extraction_diagnostic", return_value=diagnostic):
            response = self.client.post(
                f"/api/sessions/{self.case_id}/messages", headers=self.headers,
                json={"expected_revision": self.revision, "text": message, "kind": "answer",
                      "allow_external_extraction": True})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["extraction_outcome"], "valid_empty")
        self.assertFalse(result["external_model_result_accepted"])
        self.assertFalse(result["message_plan_applied"])
        self.assertEqual(result["model_diagnostics"]["validation"]["unsupported_message_goals"],
                         ["decline_disclosure"])
        self.assertFalse(result["model_diagnostics"]["merge"]["performed"])
        self.assertTrue(result["model_diagnostics"]["fallback"]["used"])
        self.assertEqual(result["provider_call"]["fact_fields"], [])
        deadline = result["report"]["facts"].get("deadline")
        self.assertEqual(deadline["status"], "denied")
        self.assertIsNone(deadline["value"])
        self.assertIn("ما حدد مهلة أو موعد للدفع", deadline["quote"])

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
        self.assertIn("المصادر الوطنية لا تمثل مسارًا محليًا خارجه", state["messages"][-2]["text"])
        self.assertTrue(any(citation["id"] == "AU-ESAFETY-EVIDENCE-2026"
                            for citation in state["messages"][-2]["citations"]))
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

    def test_windows_f83_case_reproduces_each_payment_contact_timestamp_and_correction_stage(self):
        state = self.turn("في شخص عم يهددني على واتساب.")
        self.assertEqual(state["pending_slot"], "direct_danger")

        state = self.turn("لا، ما في خطر مباشر علي الآن.")
        self.assertEqual(state["pending_slot"], "details")
        state = self.turn("قال لي إذا ما حولتله 100 دينار رح ينشر محادثاتي الخاصة.")
        self.assertEqual(state["facts"]["requested_amount"]["value"], "100 دينار")
        self.assertEqual(state["facts"]["details"]["quote"], "قال لي إذا ما حولتله 100 دينار رح ينشر محادثاتي الخاصة.")

        state = self.turn("أول تهديد وصلني أمس الساعة 6 مساءً.")
        self.assertEqual(state["facts"]["initial_time"]["status"], "reported")
        state = self.turn("نعم، اليوم رجع هددني بنشر محادثاتي الخاصة إذا ما دفعت.")
        self.assertEqual(state["facts"]["repeated"]["value"], True)
        state = self.turn("نعم، حولتله 130 دينار.")
        self.assertEqual(state["facts"]["paid_amount"]["value"], "130 دينار")
        self.assertEqual(len(state["money_events"]), 2)

        state = self.turn("حولت المبلغ عن طريق محفظة إلكترونية.")
        self.assertEqual(state["facts"]["payment_method"]["value"], "محفظة إلكترونية")
        self.assertEqual(state["facts"]["payment_method"]["status"], "reported")
        self.assertEqual(state["pending_slot"], "deadline")
        state = self.turn("بس أنا دفعت.")
        self.assertEqual(state["facts"]["payment_method"]["status"], "reported")
        self.assertNotEqual(state["facts"]["payment_method"]["status"], "conflict")
        self.assertEqual(state["facts"]["paid_amount"]["value"], "130 دينار")
        state = self.turn("لا، ما حدد أي مهلة أو موعد للدفع.")
        self.assertEqual(state["facts"]["deadline"]["status"], "denied")
        state = self.turn("ما بعرف إذا نشر المحادثات أو لا، ما شفت أي نشر.")
        self.assertEqual(state["facts"]["published"]["status"], "unknown")

        state = self.turn("لا، ما بعرف هويته الحقيقية، بس عندي رقم الواتساب اللي تواصل معي منه، وما بدي أكتبه هون.")
        self.assertEqual(state["facts"]["identity"]["status"], "unknown")
        self.assertEqual(state["facts"]["contact"]["status"], "reported")
        self.assertIn("لم يُدرج", state["facts"]["contact"]["value"])
        self.assertEqual(state["facts"]["contact_identifier_disclosure"]["status"], "declined")
        self.assertIsNone(state["facts"]["contact_identifier_disclosure"]["value"])
        self.assertEqual(state["pending_slot"], "evidence")

        state = self.turn("عندي لقطات شاشة لرسائل التهديد على واتساب، وإيصال تحويل الـ130 دينار من المحفظة الإلكترونية.")
        self.assertEqual(state["stage"], "review")
        self.assertIsNone(state["pending_slot"])
        self.assertEqual(state["facts"]["payment_method"]["value"], "محفظة إلكترونية")
        self.assertEqual(state["facts"]["payment_method"]["status"], "reported")
        self.assertEqual(len(state["money_events"]), 2, "the receipt must not create a new paid event")
        self.assertEqual(state["facts"]["paid_amount"]["value"], "130 دينار")

        state = self.turn("مثل ما ذكرت سابقًا، دفعت عن طريق محفظة إلكترونية. وإيصال الـ130 دينار يخص نفس التحويل السابق، مش دفعة جديدة.")
        self.assertEqual(state["stage"], "review")
        self.assertIsNone(state["pending_slot"], "do not ask again for a resolved payment method")
        self.assertEqual(state["facts"]["payment_method"]["status"], "reported")
        self.assertEqual(state["facts"]["payment_method"]["value"], "محفظة إلكترونية")
        self.assertEqual(len(state["money_events"]), 2, "same-transfer receipt must not be a third payment")
        self.assertEqual(state["facts"]["paid_amount"]["value"], "130 دينار")

        original_details_quote = state["facts"]["details"]["quote"]
        original_paid_quote = state["facts"]["paid_amount"]["quote"]
        other_facts_before = {key: fact for key, fact in state["facts"].items() if key != "requested_amount"}
        prior_money_events = list(state["money_events"])
        prior_version = state["summary_version"]
        corrected = self.turn("تصحيح: المبلغ المطلوب 120 دينار، وليس 100.")
        self.assertEqual(corrected["facts"]["requested_amount"]["value"], "120 دينار")
        self.assertEqual({key: fact for key, fact in corrected["facts"].items() if key != "requested_amount"},
                         other_facts_before)
        self.assertEqual(corrected["facts"]["details"]["quote"], original_details_quote)
        self.assertEqual(corrected["facts"]["paid_amount"]["value"], "130 دينار")
        self.assertEqual(corrected["facts"]["paid_amount"]["quote"], original_paid_quote)
        self.assertEqual(corrected["money_events"][0]["amount"], "100 دينار")
        self.assertEqual(corrected["money_events"][0]["relation"], "initial")
        self.assertEqual(corrected["money_events"][1]["amount"], "130 دينار")
        self.assertEqual(corrected["money_events"][1]["relation"], "initial")
        self.assertEqual(corrected["money_events"][:2], prior_money_events)
        self.assertEqual(corrected["money_events"][2]["amount"], "120 دينار")
        self.assertEqual(corrected["money_events"][2]["relation"], "correction")
        self.assertGreater(corrected["summary_version"], prior_version)

        json_before_confirm = self.client.get(f"/api/sessions/{self.case_id}/report.json", headers=self.headers).json()
        self.assertEqual(json_before_confirm["facts"]["requested_amount"]["value"], "120 دينار")
        self.assertEqual(json_before_confirm["facts"]["paid_amount"]["value"], "130 دينار")
        self.assertEqual(json_before_confirm["facts"]["contact"]["value"], "رقم واتساب متوفر بحسب الإفادة؛ المعرّف لم يُدرج")
        self.assertEqual(json_before_confirm["facts"]["contact_identifier_disclosure"]["status"], "declined")
        self.assertEqual(json_before_confirm["facts"]["identity"]["status"], "unknown")
        self.assertIn("contact_identifier_disclosure", json_before_confirm["declined_fields"])
        self.assertNotIn("contact_identifier_disclosure", json_before_confirm["unknown_fields"])
        self.assertEqual(json_before_confirm["correction_notes"][0]["previous_value"], "100 دينار")
        self.assertEqual(json_before_confirm["correction_notes"][0]["current_value"], "120 دينار")
        message_times = {message["id"]: message["created_at"] for message in corrected["messages"]}
        for key in ("requested_amount", "paid_amount", "payment_method", "contact", "contact_identifier_disclosure", "details"):
            fact = json_before_confirm["facts"][key]
            self.assertEqual(fact["source_message_at"], message_times[fact["message_id"]], key)
            self.assertTrue(fact["last_updated_at"], key)

        markdown = self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers).text
        self.assertIn(f"- نص الإفادة: «{original_details_quote}»", markdown)
        self.assertIn("المبلغ المطلوب المعتمد حاليًا 120 دينار", markdown)
        self.assertIn("كانت الإفادة الأصلية في هذا الموضع تذكر 100 دينار", markdown)
        self.assertIn(f"وقت الرسالة الأصلية: {message_times[corrected['facts']['details']['message_id']]}", markdown)
        self.assertIn("وقت آخر تحديث للمعلومة:", markdown)

        confirmed = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers,
                                     json={"expected_revision": self.revision})
        self.assertEqual(confirmed.status_code, 200)
        self.revision = confirmed.json()["session"]["revision"]
        confirmed_version = confirmed.json()["session"]["summary_version"]
        fetched = self.client.get(f"/api/sessions/{self.case_id}", headers=self.headers).json()
        self.assertTrue(fetched["session"]["confirmed"])
        self.assertEqual(fetched["report"]["summary_version"], confirmed_version)
        self.assertEqual(fetched["report"]["facts"]["requested_amount"]["value"], "120 دينار")
        self.assertTrue(self.client.get(f"/api/sessions/{self.case_id}/report.json", headers=self.headers).json()["confirmed"])
        self.assertIn("المبلغ المطلوب المعتمد حاليًا 120 دينار",
                      self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers).text)

        later = self.turn("أضيف تفاصيل: احتفظت بنسخة أخرى من المحادثة.")
        self.assertFalse(later["confirmed"])
        self.assertEqual(later["facts"]["requested_amount"]["value"], "120 دينار")
        self.assertEqual(later["facts"]["paid_amount"]["value"], "130 دينار")
        current_export = self.client.get(f"/api/sessions/{self.case_id}/report.json", headers=self.headers).json()
        self.assertFalse(current_export["confirmed"])
        self.assertEqual(current_export["summary_version"], later["summary_version"])

        paused = self.turn("احفظ الجلسة")
        self.assertEqual(paused["stage"], "paused")
        self.assertFalse(paused["confirmed"])
        resumed = self.client.post(f"/api/sessions/{self.case_id}/resume", headers=self.headers,
                                   json={"expected_revision": self.revision})
        self.assertEqual(resumed.status_code, 200)
        self.revision = resumed.json()["session"]["revision"]
        self.assertFalse(resumed.json()["session"]["confirmed"])
        self.assertEqual(resumed.json()["report"]["summary_version"], later["summary_version"])
        self.assertEqual(resumed.json()["report"]["facts"]["paid_amount"]["value"], "130 دينار")

    def test_confirmation_is_invalidated_when_a_fact_changes_after_review_resume(self):
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
        self.assertEqual(resumed.json()["session"]["stage"], "review")
        self.assertIsNone(resumed.json()["session"]["pending_slot"])
        self.assertTrue(resumed.json()["session"]["confirmed"])

        updated = self.turn("طلب مني 100 دينار")
        self.assertEqual(updated["facts"]["requested_amount"]["value"], "100 دينار")
        self.assertEqual(updated["stage"], "review")
        self.assertIsNone(updated["pending_slot"])
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

    def test_urgent_stop_state_survives_pause_and_resume_without_interview_question(self):
        risk = self.turn("أنا بخطر مباشر الآن.")
        self.assertEqual(risk["stage"], "urgent_stop")
        self.assertIsNone(risk["pending_slot"])
        paused = self.turn("احفظ الجلسة")
        self.assertEqual(paused["stage"], "paused")
        self.assertEqual(paused["resume_stage"], "urgent_stop")
        before_resume_count = len(paused["messages"])
        response = self.client.post(f"/api/sessions/{self.case_id}/resume", headers=self.headers,
                                    json={"expected_revision": self.revision})
        self.assertEqual(response.status_code, 200, response.text)
        self.revision = response.json()["session"]["revision"]
        state = response.json()["session"]
        self.assertEqual(state["stage"], "urgent_stop")
        self.assertIsNone(state["pending_slot"])
        new_messages = state["messages"][before_resume_count:]
        self.assertEqual(len(new_messages), 1)
        self.assertEqual(new_messages[0]["kind"], "safety")
        self.assertFalse(any(message["kind"] == "question" for message in new_messages))

    def test_confirmed_review_returns_to_review_after_save_and_resume(self):
        self.turn("في شخص يهددني")
        reviewed = self.turn("اعرض الملخص")
        self.assertEqual(reviewed["stage"], "review")
        confirmed = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers,
                                     json={"expected_revision": self.revision})
        self.assertEqual(confirmed.status_code, 200, confirmed.text)
        confirmed_state = confirmed.json()["session"]
        self.revision = confirmed_state["revision"]
        confirmed_digest = confirmed_state["confirmed_digest"]
        confirmed_version = confirmed_state["summary_version"]
        message_count = len(confirmed_state["messages"])

        paused = self.turn("احفظ الجلسة")
        self.assertEqual(paused["stage"], "paused")
        self.assertEqual(paused.get("resume_stage"), "review")
        self.assertTrue(paused["confirmed"])
        resumed = self.client.post(f"/api/sessions/{self.case_id}/resume", headers=self.headers,
                                   json={"expected_revision": self.revision})
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.revision = resumed.json()["session"]["revision"]
        state = resumed.json()["session"]
        report = resumed.json()["report"]
        self.assertEqual(state["stage"], "review")
        self.assertEqual(len(state["messages"]), message_count + 3)  # save user/status + resume status
        self.assertEqual(state["confirmed_digest"], confirmed_digest)
        self.assertTrue(state["confirmed"])
        self.assertTrue(report["confirmed"])
        self.assertEqual(report["summary_version"], confirmed_version)
        self.assertEqual(self.client.get(f"/api/sessions/{self.case_id}/report.json", headers=self.headers).json()["confirmed"], True)
        self.assertIn("**التأكيد:** مؤكد من المستخدم", self.client.get(
            f"/api/sessions/{self.case_id}/report.md", headers=self.headers).text)

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
             patch("app.main.generate_interview_extraction", side_effect=slow_extract), \
             patch("app.main.last_call_trace", return_value={"attempted": True, "latency_ms": 1}):
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
             patch("app.main.generate_grounded_answer", side_effect=slow_answer), \
             patch("app.main.last_call_trace", return_value={"attempted": True, "latency_ms": 1}):
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

    def test_temporal_clarification_survives_review_confirmation_resume_and_exports(self):
        first_text = (
            "اليوم اكتشفت حساب على إنستغرام مستخدم اسمي وصورتي وبحكي للناس إنه أنا. "
            "حسابي الأصلي معي وبقدر أدخل عليه. ما بعرف مين عامل الحساب، وعندي لقطات شاشة للحساب المزيف، "
            "وما في تهديد أو طلب مصاري."
        )
        clarification = "للتوضيح: أنا اكتشفت الحساب المزيف اليوم، لكن ما بعرف متى أنشأه الشخص."
        first = self.turn(first_text)
        self.assertEqual(first["facts"]["discovery_time"]["value"], "اليوم")
        self.assertNotIn("initial_time", first["facts"])

        review = self.turn("اعرض الملخص")
        self.assertEqual(review["stage"], "review")
        confirmed = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers,
                                     json={"expected_revision": self.revision})
        self.assertEqual(confirmed.status_code, 200, confirmed.text)
        self.revision = confirmed.json()["session"]["revision"]
        self.assertTrue(confirmed.json()["session"]["confirmed"])

        updated = self.turn(clarification)
        self.assertFalse(updated["confirmed"])
        self.assertIsNone(updated["confirmed_digest"])
        self.assertEqual(updated["facts"]["discovery_time"]["value"], "اليوم")
        self.assertEqual(updated["facts"]["account_creation_time"]["status"], "unknown")
        self.assertNotIn("additional_details", updated["facts"])
        self.assertEqual(updated["facts"]["impersonation"]["quote"], first_text)
        self.assertEqual(updated["facts"]["evidence"]["quote"], first_text)
        self.assertEqual(updated["facts"]["original_account_access_available"]["quote"], first_text)

        json_report = self.client.get(f"/api/sessions/{self.case_id}/report.json", headers=self.headers).json()
        self.assertFalse(json_report["confirmed"])
        self.assertEqual(json_report["summary_version"], updated["summary_version"])
        self.assertEqual(json_report["facts"]["discovery_time"]["value"], "اليوم")
        self.assertEqual(json_report["facts"]["account_creation_time"]["status"], "unknown")
        self.assertIn("لا يُستنتج من وقت اكتشافه", json_report["field_notes"]["account_creation_time"])
        source_times = {message["id"]: message["created_at"] for message in updated["messages"]}
        for key in ("discovery_time", "account_creation_time"):
            fact = json_report["facts"][key]
            self.assertEqual(fact["source_message_at"], source_times[fact["message_id"]])
            self.assertTrue(fact["last_updated_at"])

        markdown = self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers).text
        self.assertIn("وقت اكتشاف الواقعة: اليوم", markdown)
        self.assertIn("وقت إنشاء الحساب المزيف", markdown)
        self.assertIn("ما بعرف متى أنشأه الشخص", markdown)

        self.assertEqual(self.turn("احفظ الجلسة")["stage"], "paused")
        resumed = self.client.post(f"/api/sessions/{self.case_id}/resume", headers=self.headers,
                                   json={"expected_revision": self.revision})
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.revision = resumed.json()["session"]["revision"]
        resumed_report = resumed.json()["report"]
        self.assertFalse(resumed_report["confirmed"])
        self.assertEqual(resumed_report["facts"]["discovery_time"]["value"], "اليوم")
        self.assertEqual(resumed_report["facts"]["account_creation_time"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
