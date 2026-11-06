import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.core import handle_turn, new_state, report_data, render_report
from app.model import generate_interview_extraction, validate_interview_facts_diagnostic
from app.store import Store
from scripts.evaluate_conversation_v3 import _provider_turn_metrics


class MoneyEventReferenceTrustTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "pending_question": {"slot": "paid_amount_relation", "question_ref": "m9"},
            "recent_relevant_messages": [{"message_ref": "m8", "role": "user", "text": "دفعت 130 دينار"}],
            "related_facts": [],
            "related_money_events": [{"id": "money-2", "kind": "paid", "amount": "130 دينار",
                                         "relation": "initial", "quote": "دفعت 130 دينار", "message_id": "m8"}],
        }

    def validate(self, field, value, operation="same"):
        text = "نفس الدفعة"
        payload = {"facts": [{"field": field, "status": "reported", "value": value,
                              "current_quote": text, "operation": operation, "target_event_id": "money-2"}]}
        return validate_interview_facts_diagnostic(text, json.dumps(payload, ensure_ascii=False),
                                                   "paid_amount_relation", self.context)

    def test_payment_event_reference_cannot_support_an_unrelated_identity(self):
        result = self.validate("identity", "أحمد")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["filtered_facts"][0]["filter_code"], "context_reference_invalid")

    def test_same_payment_reference_cannot_change_amount_or_currency(self):
        amount = self.validate("paid_amount", "999 دولار")
        self.assertEqual(amount["facts"], [])
        self.assertEqual(amount["filtered_facts"][0]["filter_code"], "context_reference_invalid")
        currency = self.validate("paid_amount", "130 دولار")
        self.assertEqual(currency["facts"], [])
        self.assertEqual(currency["filtered_facts"][0]["filter_code"], "context_reference_invalid")

    def test_same_payment_reference_accepts_only_same_kind_and_same_full_amount(self):
        result = self.validate("paid_amount", "130 دينار")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["facts"][0]["value"], "130 دينار")
        unrelated = self.validate("payment_method", "محفظة إلكترونية")
        self.assertEqual(unrelated["facts"], [])
        self.assertEqual(unrelated["filtered_facts"][0]["filter_code"], "context_reference_invalid")

    def test_api_rejects_invented_identity_before_merge_and_export(self):
        old_store = main.store
        with tempfile.TemporaryDirectory() as directory:
            main.store = Store(str(Path(directory) / "api.sqlite3"))
            client = TestClient(main.app)
            try:
                created = client.post("/api/sessions", json={}).json()
                session_id, token = created["session"]["session_id"], created["token"]
                headers = {"X-Case-Token": token}
                revision = created["session"]["revision"]
                first = client.post(f"/api/sessions/{session_id}/messages", headers=headers,
                                    json={"expected_revision": revision,
                                          "text": "طلب مني 100 دينار وحولت 130 دينار بالمحفظة", "kind": "answer"})
                self.assertEqual(first.status_code, 200, first.text)
                first_data = first.json()
                revision = first_data["session"]["revision"]
                self.assertEqual([(event["id"], event["kind"], event["amount"])
                                  for event in first_data["session"]["money_events"]],
                                 [("money-1", "requested", "100 دينار"),
                                  ("money-2", "paid", "130 دينار")])
                malicious = {"facts": [{"field": "identity", "status": "reported", "value": "أحمد",
                                         "current_quote": "نفس الدفعة", "operation": "same",
                                         "target_event_id": "money-2"}]}
                env = {"MODEL_PROVIDER": "openai-compatible", "MODEL_BASE_URL": "https://provider.example/v1",
                       "MODEL_NAME": "synthetic-test", "MODEL_API_KEY": "test-only"}
                with patch.dict(os.environ, env, clear=True), \
                     patch("app.main.model_configured", return_value=True), \
                     patch("app.main.provider_config", return_value={"provider": "openai-compatible",
                                                                        "model": "synthetic-test", "mode": "external_model"}), \
                     patch("app.model._chat", return_value=json.dumps(malicious, ensure_ascii=False)):
                    second = client.post(f"/api/sessions/{session_id}/messages", headers=headers,
                                         json={"expected_revision": revision, "text": "نفس الدفعة", "kind": "answer",
                                               "allow_external_extraction": True, "allow_context_extraction": True})
                self.assertEqual(second.status_code, 200, second.text)
                result = second.json()
                saved = result["session"]
                self.assertNotIn("identity", saved["facts"])
                self.assertNotIn("أحمد", json.dumps(saved, ensure_ascii=False))
                self.assertNotIn("أحمد", json.dumps(result["report"], ensure_ascii=False))
                exported = client.get(f"/api/sessions/{session_id}/report.json", headers=headers)
                self.assertEqual(exported.status_code, 200)
                self.assertNotIn("أحمد", exported.text)
            finally:
                client.close()
                main.store = old_store


class ValidatedMessagePlanTests(unittest.TestCase):
    def test_generation_result_preserves_validated_goals_when_facts_are_empty(self):
        text = "بدي نصيحة أحافظ فيها على حسابي."
        payload = {"message_goals": ["ask_guidance"], "guidance_question": text, "facts": []}
        env = {"MODEL_PROVIDER": "openai-compatible", "MODEL_BASE_URL": "https://provider.example/v1",
               "MODEL_NAME": "synthetic-test", "MODEL_API_KEY": "test-only"}
        with patch.dict(os.environ, env, clear=True), \
             patch("app.model._chat", return_value=json.dumps(payload, ensure_ascii=False)):
            result = generate_interview_extraction(text, "narrative")
        self.assertEqual(result, [])
        self.assertEqual(result.message_plan["goals"], ["ask_guidance"])
        self.assertEqual(result.message_plan["guidance_question"], text)
        self.assertTrue(result.diagnostic["accepted"])

    def setUp(self):
        self.old_store = main.store
        self.temp_dir = tempfile.TemporaryDirectory()
        main.store = Store(str(Path(self.temp_dir.name) / "api.sqlite3"))
        self.client = TestClient(main.app)
        created = self.client.post("/api/sessions", json={}).json()
        self.session_id = created["session"]["session_id"]
        self.token = created["token"]
        self.headers = {"X-Case-Token": self.token}
        self.revision = created["session"]["revision"]

    def tearDown(self):
        self.client.close()
        main.store = self.old_store
        self.temp_dir.cleanup()

    def send(self, text, **kwargs):
        response = self.client.post(f"/api/sessions/{self.session_id}/messages", headers=self.headers,
                                    json={"expected_revision": self.revision, "text": text,
                                          "kind": "answer", **kwargs})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.revision = result["session"]["revision"]
        return result

    def test_model_goal_routes_natural_guidance_without_creating_a_fact(self):
        text = "بدي نصيحة أحافظ فيها على حسابي."
        plan = {"message_goals": ["ask_guidance"], "guidance_question": text, "facts": []}
        local = {"text": "فعّل المصادقة الثنائية وراجع الجلسات النشطة.", "citations": []}
        env = {"MODEL_PROVIDER": "openai-compatible", "MODEL_BASE_URL": "https://provider.example/v1",
               "MODEL_NAME": "synthetic-test", "MODEL_API_KEY": "test-only"}
        with patch.dict(os.environ, env, clear=True), \
             patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value={"provider": "openai-compatible",
                                                                "model": "synthetic-test", "mode": "external_model"}), \
             patch("app.model._chat", return_value=json.dumps(plan, ensure_ascii=False)), \
             patch("app.main.answer_question", return_value=local) as answer:
            result = self.send(text, allow_external_extraction=True)
        answer.assert_called_once()
        state = result["session"]
        self.assertTrue(result["question_detected"])
        self.assertNotIn("بدي نصيحة", json.dumps(state.get("facts", {}), ensure_ascii=False))
        answers = [item for item in state["messages"] if item.get("role") == "assistant" and item.get("kind") == "answer"]
        self.assertTrue(any("فعّل المصادقة الثنائية" in item["text"] for item in answers))
        self.assertEqual(state["pending_slot"], "narrative")

    def test_model_guidance_goal_is_combined_with_an_actionable_fact(self):
        text = "عندي لقطات شاشة للرسائل، وبدي نصيحة أحافظ فيها على حسابي."
        guidance = "بدي نصيحة أحافظ فيها على حسابي."
        plan = {"message_goals": ["ask_guidance", "add_fact"], "guidance_question": guidance,
                "facts": [{"field": "evidence", "status": "reported", "value": "صور أو لقطات شاشة",
                           "current_quote": "عندي لقطات شاشة"}]}
        local = {"text": "راجع إعدادات الأمان وسجّل الخروج من الجلسات غير المعروفة.", "citations": []}
        env = {"MODEL_PROVIDER": "openai-compatible", "MODEL_BASE_URL": "https://provider.example/v1",
               "MODEL_NAME": "synthetic-test", "MODEL_API_KEY": "test-only"}
        with patch.dict(os.environ, env, clear=True), \
             patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value={"provider": "openai-compatible",
                                                                "model": "synthetic-test", "mode": "external_model"}), \
             patch("app.model._chat", return_value=json.dumps(plan, ensure_ascii=False)), \
             patch("app.main.answer_question", return_value=local) as answer:
            result = self.send(text, allow_external_extraction=True)
        answer.assert_called_once()
        self.assertEqual(result["session"]["facts"]["evidence"]["status"], "reported")
        self.assertTrue(any(message.get("kind") == "answer" and "راجع إعدادات الأمان" in message.get("text", "")
                            for message in result["session"]["messages"]))

    def test_mixed_payment_and_assessment_question_explains_assessment_not_payment_slot(self):
        for text in (
            "في شخص عم يهددني على واتساب.",
            "لا، ما في خطر مباشر علي الآن.",
            "قال إذا ما حولتله 100 دينار رح ينشر محادثاتي الخاصة.",
            "أمس الساعة 6 مساءً.",
            "نعم، رجع هددني اليوم.",
        ):
            result = self.send(text)
        self.assertEqual(result["session"]["assessment"]["priority_score"], 90)
        previous_pending = result["session"]["pending_slot"]
        result = self.send("دفعت 130 دينار، ليش الأولوية مرتفعة وأنا مش بخطر مباشر؟")
        state = result["session"]
        self.assertEqual(state["facts"]["paid_amount"]["value"], "130 دينار")
        answers = [message for message in state["messages"] if message.get("role") == "assistant" and message.get("kind") == "answer"]
        self.assertTrue(answers)
        explanation = answers[-1]["text"]
        self.assertIn("الدرجة الحالية 90/100", explanation)
        self.assertIn("الخطر المباشر الحالي مسجل كمنفي", explanation)
        self.assertNotIn("حتى نميّز بين المبلغ المطلوب", explanation)
        self.assertNotEqual(state["pending_slot"], previous_pending)

    def test_api_persists_correction_chain_and_exports_one_current_value_without_extra_payment(self):
        original = "شخص هددني وطلب مني 100 دينار، وحولت 130 دينار بالمحفظة."
        self.send(original)
        self.send("تصحيح: المبلغ المطلوب 120 بدل 100")
        self.send("تصحيح: المبلغ المطلوب 140 بدل 120")
        final = self.send("تصحيح: المبلغ المطلوب 160 بدل 140")
        db_path = Path(self.temp_dir.name) / "api.sqlite3"
        persisted = Store(str(db_path)).get(self.session_id, self.token)
        self.assertEqual(persisted["facts"]["requested_amount"]["value"], "160 دينار")
        self.assertEqual(persisted["facts"]["paid_amount"]["value"], "130 دينار")
        report_json = self.client.get(f"/api/sessions/{self.session_id}/report.json", headers=self.headers).json()
        report_md = self.client.get(f"/api/sessions/{self.session_id}/report.md", headers=self.headers).text
        root = next(event for event in report_json["money_events"] if event["id"] == "money-1")
        self.assertEqual(root["quote"], original)
        self.assertEqual(root["current_amount"], "160 دينار")
        self.assertEqual(sum(event["kind"] == "paid" and not event["is_correction_record"]
                             for event in report_json["money_events"]), 1)
        self.assertEqual(report_json["facts"]["requested_amount"]["value"], "160 دينار")
        self.assertTrue(report_json["correction_notes"])
        self.assertEqual(report_json["correction_notes"][0]["previous_value"], "100 دينار")
        self.assertIn("160 دينار", report_md)
        self.assertIn("100 دينار", report_md)
        self.assertIn(original, report_md)
        self.assertEqual(final["report"]["facts"]["paid_amount"]["value"], "130 دينار")


class MoneyCorrectionChainTests(unittest.TestCase):
    def test_three_corrections_resolve_to_the_original_requested_event(self):
        state = new_state("money-correction-chain")
        handle_turn(state, "طلب مني 100 دينار وحولت 130 دينار بالمحفظة")
        handle_turn(state, "تصحيح: المبلغ المطلوب 120 بدل 100")
        handle_turn(state, "تصحيح: المبلغ المطلوب 140 بدل 120")
        handle_turn(state, "تصحيح: المبلغ المطلوب 160 بدل 140")
        requested_events = [event for event in state["money_events"] if event.get("kind") == "requested"]
        corrections = state["money_corrections"]
        self.assertEqual(state["facts"]["requested_amount"]["value"], "160 دينار")
        self.assertEqual([item["target_event_id"] for item in corrections], ["money-1", "money-1", "money-1"])
        self.assertEqual([item["previous_amount"] for item in corrections], ["100 دينار", "120 دينار", "140 دينار"])
        self.assertEqual(state["facts"]["paid_amount"]["value"], "130 دينار")
        self.assertEqual(sum(event.get("kind") == "paid" for event in state["money_events"]), 1)
        report = report_data(state)
        root = next(event for event in report["money_events"] if event["id"] == "money-1")
        paid = next(event for event in report["money_events"] if event.get("kind") == "paid")
        self.assertEqual(root["current_amount"], "160 دينار")
        self.assertEqual(paid["current_amount"], "130 دينار")
        self.assertEqual(report["facts"]["requested_amount"]["value"], "160 دينار")
        self.assertEqual(report["money_corrections"][-1]["target_event_id"], "money-1")
        self.assertTrue(all(item["is_correction_record"] for item in report["money_events"] if item["relation"] == "correction"))
        markdown = render_report(state)
        self.assertIn("سجل تصحيح تدقيقي", markdown)
        self.assertIn("ليس دفعة أو طلبًا ماليًا إضافيًا", markdown)
        self.assertEqual(len(requested_events), 4)

    def test_correction_targets_current_requested_value_not_a_newer_additional_event(self):
        state = new_state("money-correction-older-target")
        handle_turn(state, "طلب مني 100 دينار")
        handle_turn(state, "طلب مني كمان 50 دينار زيادة")
        handle_turn(state, "تصحيح المبلغ المطلوب 80 بدل 100")
        correction = state["money_corrections"][-1]
        self.assertEqual(correction["target_event_id"], "money-1")
        self.assertEqual(state["facts"]["requested_amount"]["value"], "80 دينار")
        self.assertEqual(state["money_events"][1]["relation"], "additional")

    def test_ambiguous_correction_prompts_for_target_and_accepts_explicit_clarification(self):
        state = new_state("money-correction-ambiguous")
        state["facts"]["requested_amount"] = {"field": "requested_amount", "status": "reported",
            "value": "100 دينار", "quote": "ذكرت مبلغًا مطلوبًا", "message_id": "missing-source", "history": []}
        state["money_events"] = [
            {"id": "money-1", "kind": "requested", "amount": "100 دينار", "relation": "initial", "message_id": "m-old-1"},
            {"id": "money-2", "kind": "requested", "amount": "100 دينار", "relation": "additional", "message_id": "m-old-2"},
        ]
        handle_turn(state, "تصحيح المبلغ المطلوب 90 بدل 100")
        self.assertEqual(state["facts"]["requested_amount"]["value"], "100 دينار")
        self.assertEqual(state["pending_slot"], "requested_amount_relation")
        self.assertIn("أي مبلغ سابق تقصد تصحيحه؟", state["messages"][-1]["text"])
        handle_turn(state, "المبلغ الأول")
        self.assertEqual(state["facts"]["requested_amount"]["value"], "90 دينار")
        self.assertEqual(state["money_corrections"][-1]["target_event_id"], "money-1")


class EvaluatorProviderOutcomeTests(unittest.TestCase):
    def test_integrated_fallback_categories_and_merge_rejection_are_distinct(self):
        empty = _provider_turn_metrics(
            {"fallback_used": True, "extraction_outcome": "valid_empty"},
            {"attempted": True, "response_valid": True, "extraction_outcome": "valid_empty"}, [], [])
        rejected = _provider_turn_metrics(
            {"fallback_used": True, "extraction_outcome": "response_rejected"},
            {"attempted": True, "response_valid": False, "extraction_outcome": "response_rejected"}, [], [])
        empty_provider = _provider_turn_metrics(
            {"fallback_used": True, "extraction_outcome": "provider_failed"},
            {"attempted": True, "status": "empty_response", "response_valid": False,
             "extraction_outcome": "provider_failed", "failure_code": "empty_response"}, [], [])
        all_filtered = _provider_turn_metrics(
            {"fallback_used": True, "extraction_outcome": "all_facts_filtered"},
            {"attempted": True, "response_valid": True, "extraction_outcome": "all_facts_filtered"}, [], [])
        partial = _provider_turn_metrics(
            {"fallback_used": False, "extraction_outcome": "partially_extracted"},
            {"attempted": True, "response_valid": True, "extraction_outcome": "partially_extracted"},
            [{"field": "evidence"}], [{"field": "evidence", "disposition": "ignored"}])
        success = _provider_turn_metrics(
            {"fallback_used": False, "extraction_outcome": "extracted"},
            {"attempted": True, "response_valid": True, "extraction_outcome": "extracted"},
            [{"field": "threat"}], [{"field": "threat", "disposition": "agrees_with_rules"}])
        connection = _provider_turn_metrics(
            {"fallback_used": True, "extraction_outcome": "provider_failed"},
            {"attempted": True, "status": "connection_error", "failure_code": "unreachable",
             "response_valid": False, "extraction_outcome": "provider_failed"}, [], [])
        http_error = _provider_turn_metrics(
            {"fallback_used": True, "extraction_outcome": "provider_failed"},
            {"attempted": True, "status": "http_error", "failure_code": "http_503",
             "response_valid": False, "extraction_outcome": "provider_failed"}, [], [])
        routed = _provider_turn_metrics(
            {"fallback_used": False, "extraction_outcome": "extracted"},
            {"attempted": True, "response_valid": True, "extraction_outcome": "extracted"},
            [{"field": "paid_amount"}], [{"field": "paid_amount", "disposition": "routed_to_rules"}])
        corrected = _provider_turn_metrics(
            {"fallback_used": False, "extraction_outcome": "extracted"},
            {"attempted": True, "response_valid": True, "extraction_outcome": "extracted"},
            [{"field": "requested_amount"}], [{"field": "requested_amount", "disposition": "applied_correction"}])
        self.assertEqual(empty["fallback_category"], "empty_extraction")
        self.assertEqual(rejected["fallback_category"], "response_rejected")
        self.assertEqual(empty_provider["fallback_category"], "empty_provider_response")
        self.assertEqual(all_filtered["fallback_category"], "all_facts_filtered")
        self.assertFalse(partial["fallback_used"])
        self.assertTrue(partial["accepted_then_merge_rejected"])
        self.assertEqual(partial["accepted_fact_merge_dispositions"], {"ignored": 1})
        self.assertTrue(success["facts_accepted"])
        self.assertFalse(success["fallback_used"])
        self.assertEqual(connection["fallback_category"], "connection_failure")
        self.assertEqual(http_error["fallback_category"], "provider_http_error")
        self.assertEqual(routed["accepted_facts_routed_to_rules_fields"], ["paid_amount"])
        self.assertEqual(corrected["accepted_facts_applied_as_corrections_fields"], ["requested_amount"])
        self.assertEqual(sum(_provider_turn_metrics({"fallback_used": True, "extraction_outcome": "valid_empty"},
                                {"attempted": True, "response_valid": True, "extraction_outcome": "valid_empty"}, [], [])
                             ["fallback_used"] for _ in range(22)), 22)


if __name__ == "__main__":
    unittest.main()
