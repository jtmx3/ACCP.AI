import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.rag import answer_question, load_corpus
from app.store import Store


class RealCorpusGuidanceRetrievalTests(unittest.TestCase):
    def test_generic_account_security_question_asks_for_platform(self):
        for query in ("كيف أحمي حسابي", "كيف أحافظ على حسابي", "بدي نصيحة لتأمين الحساب"):
            with self.subTest(query=query):
                result = answer_question(query, None)
                self.assertEqual(result["citations"], [])
                self.assertIn("أي تطبيق", result["text"])
                self.assertIn("واتساب", result["text"])
                self.assertNotIn("Instagram", result["text"])
                self.assertNotIn("المقتطف", result["text"])

    def test_whatsapp_security_and_arabic_paraphrases_retrieve_whatsapp_sources(self):
        corpus = load_corpus()
        queries = (
            "كيف أحمي حسابي على واتساب؟",
            "كيف أحافظ على حسابي بواتساب؟",
            "بدي نصيحة لتأمين الحساب في WhatsApp",
        )
        for query in queries:
            with self.subTest(query=query):
                result = answer_question(query, None)
                citation_ids = {citation["id"] for citation in result["citations"]}
                self.assertTrue(citation_ids, result["text"])
                self.assertTrue(any(source_id.startswith("GLOBAL-WA-") for source_id in citation_ids),
                                (citation_ids, result["text"]))
                self.assertFalse(any(source_id.startswith("GLOBAL-IG-") for source_id in citation_ids),
                                 citation_ids)
                self.assertTrue(any("التحقق بخطوتين" in citation["excerpt"]
                                    or "التحقق بخطوتين" in result["text"] for citation in result["citations"]))
                self.assertIn("WhatsApp Help Center", result["text"])
                self.assertIn("واتساب", result["text"])

    def test_instagram_security_remains_platform_scoped(self):
        result = answer_question("كيف أؤمن حسابي الأصلي على إنستغرام إذا كنت ما زلت أستطيع الدخول؟", None)
        citation_ids = {citation["id"] for citation in result["citations"]}
        self.assertIn("GLOBAL-IG-SECURITY-2026", citation_ids)
        self.assertFalse(any(source_id.startswith("GLOBAL-WA-") for source_id in citation_ids))
        self.assertIn("Instagram", result["text"])
        self.assertIn("هذه إرشادات للحساب داخل Instagram", result["text"])

    def test_current_platform_overrides_prior_platform_and_ambiguous_target_clarifies(self):
        switched = answer_question(
            "بدي نصيحة أحافظ فيها على حسابي.", None,
            platform_context="إنستغرام", platform_query="بدي نصيحة أحافظ فيها على حسابي على واتساب")
        switched_ids = {citation["id"] for citation in switched["citations"]}
        self.assertTrue(any(source_id.startswith("GLOBAL-WA-") for source_id in switched_ids))
        self.assertFalse(any(source_id.startswith("GLOBAL-IG-") for source_id in switched_ids))

        ambiguous = answer_question(
            "بدي نصيحة أحافظ فيها على حسابي.", None,
            platform_context="واتساب وإنستغرام")
        self.assertEqual(ambiguous["citations"], [])
        self.assertIn("أي تطبيق", ambiguous["text"])

        multi_platform_question = answer_question("كيف أحمي حسابي على واتساب أو إنستغرام؟", None)
        self.assertEqual(multi_platform_question["citations"], [])
        self.assertIn("أي تطبيق", multi_platform_question["text"])

    def test_unrelated_and_unsupported_platform_requests_do_not_borrow_other_sources(self):
        unrelated = answer_question("طريقة تحضير الكعكة والخبز المنزلي", None)
        self.assertEqual(unrelated["citations"], [])
        self.assertIn("لا يوجد", unrelated["text"])

        unsupported = answer_question("كيف أحمي حسابي على سيجنال؟", None)
        self.assertEqual(unsupported["citations"], [])
        self.assertIn("مصدر", unsupported["text"])
        self.assertNotIn("Instagram", unsupported["text"])

    def test_whatsapp_account_recovery_uses_whatsapp_recovery_source(self):
        result = answer_question("كيف أستعيد حسابي المخترق على واتساب؟", None)
        citation_ids = {citation["id"] for citation in result["citations"]}
        self.assertIn("GLOBAL-WA-RECOVERY-2026", citation_ids)
        self.assertNotIn("GLOBAL-IG-SECURITY-2026", citation_ids)
        self.assertIn("رمز التسجيل", result["text"])
        self.assertIn("WhatsApp Help Center", result["text"])

    def test_current_country_is_separate_from_saved_jurisdiction(self):
        result = answer_question("ما هي وسائل التواصل مع وحدة الجرائم الإلكترونية بالأردن؟", "US")
        self.assertEqual(result["resolved_jurisdiction"], "JO")
        self.assertIn("JO-PSD-CYBERCRIME-UNIT-2026", {citation["id"] for citation in result["citations"]})
        self.assertNotIn("US-FTC-SCAM-2026", {citation["id"] for citation in result["citations"]})

    def test_full_current_message_supplies_topic_when_guidance_span_is_short(self):
        result = answer_question("شو أعمل؟", None,
                                 topic_query="بعد ما حولت المال، شو أعمل؟")
        self.assertIn("payment_recovery", result["resolved_topics"])
        self.assertIn("US-FTC-SCAM-2026", {citation["id"] for citation in result["citations"]})


class RoutedRealCorpusGuidanceTests(unittest.TestCase):
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
        response = self.client.post(
            f"/api/sessions/{self.session_id}/messages", headers=self.headers,
            json={"expected_revision": self.revision, "text": text, "kind": "answer", **kwargs})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.revision = result["session"]["revision"]
        return result

    def ask_with_simulated_goal_plan(self, text, question_span, goals=("ask_guidance",)):
        plan = {"message_goals": list(goals), "guidance_question": question_span, "facts": []}
        env = {"MODEL_PROVIDER": "openai-compatible", "MODEL_BASE_URL": "https://provider.example/v1",
               "MODEL_NAME": "synthetic-test", "MODEL_API_KEY": "test-only"}
        with patch.dict(os.environ, env, clear=True), \
             patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value={"provider": "openai-compatible",
                                                               "model": "synthetic-test", "mode": "external_model"}), \
             patch("app.model._chat", return_value=json.dumps(plan, ensure_ascii=False)):
            return self.send(text, allow_external_extraction=True, allow_context_extraction=True)

    def test_previous_platform_turn_drives_real_retrieval_then_explicit_change_switches_source(self):
        first = self.send("المشكلة صارت على إنستغرام.")
        self.assertEqual(first["session"]["facts"]["platform"]["value"], "إنستغرام")

        generic = "بدي نصيحة أحافظ فيها على حسابي."
        answer = self.ask_with_simulated_goal_plan(generic, generic)
        response = next(message for message in reversed(answer["session"]["messages"])
                        if message.get("role") == "assistant" and message.get("kind") == "answer")
        first_ids = {citation["id"] for citation in response["citations"]}
        self.assertIn("GLOBAL-IG-SECURITY-2026", first_ids)
        self.assertIn("المصادقة الثنائية", response["text"])
        self.assertNotIn("واتساب", " ".join(citation["title"] for citation in response["citations"]))

        changed = "بدي نصيحة أحافظ فيها على حسابي على واتساب."
        answer = self.ask_with_simulated_goal_plan(changed, changed)
        response = next(message for message in reversed(answer["session"]["messages"])
                        if message.get("role") == "assistant" and message.get("kind") == "answer")
        changed_ids = {citation["id"] for citation in response["citations"]}
        self.assertTrue(any(source_id.startswith("GLOBAL-WA-") for source_id in changed_ids))
        self.assertFalse(any(source_id.startswith("GLOBAL-IG-") for source_id in changed_ids))
        self.assertIn("التحقق بخطوتين", response["text"])

    def test_no_platform_turn_routes_to_a_short_clarification_without_wrong_source(self):
        text = "بدي نصيحة أحافظ فيها على حسابي."
        result = self.ask_with_simulated_goal_plan(text, text)
        response = next(message for message in reversed(result["session"]["messages"])
                        if message.get("role") == "assistant" and message.get("kind") == "answer")
        self.assertIn("أي تطبيق", response["text"])
        self.assertEqual(response["citations"], [])
        self.assertNotIn("GLOBAL-IG-SECURITY-2026", response["text"])
        self.assertTrue(result["question_detected"])
        self.assertEqual(result["session"]["pending_slot"], "narrative")


if __name__ == "__main__":
    unittest.main()
