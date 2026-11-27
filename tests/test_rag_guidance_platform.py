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
                self.assertIn("واتساب", result["text"])
                self.assertEqual(result["text"].count("التحقق بخطوتين"), 1)
                self.assertTrue(all(citation["publisher"] == "WhatsApp Help Center, Meta"
                                    for citation in result["citations"]))
                self.assertLessEqual(len([line for line in result["text"].splitlines()
                                          if line[:1].isdigit() and ". " in line]), 4)
                self.assertNotIn("المقتطف الداعم", result["text"])
                self.assertTrue(all(citation["excerpt"] for citation in result["citations"]))

    def test_instagram_security_remains_platform_scoped(self):
        result = answer_question("كيف أؤمن حسابي الأصلي على إنستغرام إذا كنت ما زلت أستطيع الدخول؟", None)
        citation_ids = {citation["id"] for citation in result["citations"]}
        self.assertIn("GLOBAL-IG-SECURITY-2026", citation_ids)
        self.assertFalse(any(source_id.startswith("GLOBAL-WA-") for source_id in citation_ids))
        self.assertIn("Instagram", result["text"])
        self.assertTrue(any("إنستغرام" in citation.get("scope_note", "") for citation in result["citations"]))

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
        self.assertTrue(any(citation["publisher"] == "WhatsApp Help Center, Meta"
                            for citation in result["citations"]))

    def test_current_country_is_separate_from_saved_jurisdiction(self):
        result = answer_question("ما هي وسائل التواصل مع وحدة الجرائم الإلكترونية بالأردن؟", "US")
        self.assertEqual(result["resolved_jurisdiction"], "JO")
        self.assertIn("JO-PSD-CYBERCRIME-UNIT-2026", {citation["id"] for citation in result["citations"]})
        self.assertNotIn("US-FTC-SCAM-2026", {citation["id"] for citation in result["citations"]})

    def test_specific_jordan_cybercrime_law_article_retrieves_its_source_without_platform(self):
        result = answer_question("ما هي المادة 18 من قانون الجرائم الإلكترونية الأردني؟", None)
        citations = {citation["id"]: citation for citation in result["citations"]}
        self.assertIn("JO-CYBERCRIME-LAW-ART18-2023", citations)
        self.assertEqual(set(citations), {"JO-CYBERCRIME-LAW-ART18-2023"})
        self.assertEqual(result["resolved_platform"], None)
        self.assertEqual(result["resolved_jurisdiction"], "JO")
        self.assertIn("cybercrime_law", result["resolved_topics"])
        self.assertIn("المادة 18 تتعلق", result["text"])
        self.assertIn("التهديد", result["text"])
        self.assertIn("لا تمثل تكييفًا", result["text"])
        self.assertIn("المادة 18", citations["JO-CYBERCRIME-LAW-ART18-2023"]["excerpt"])
        self.assertNotIn("أي تطبيق", result["text"])

        unsupported_article = answer_question("ما هي المادة 99 من قانون الجرائم الإلكترونية الأردني؟", None)
        self.assertEqual(unsupported_article["citations"], [])
        self.assertIn("أمتنع عن الاستنتاج", unsupported_article["text"])

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

    def ask_with_simulated_goal_plan(self, text, question_span=None, goals=("ask_guidance",)):
        goal = "ask_guidance" if "ask_guidance" in goals else None
        plan = {"facts": [], "user_intent": goal, "conversation_move": "answer_question",
                "reply_text": "هل تقصد واتساب أم إنستغرام، أم تطبيقًا آخر؟"}
        env = {"MODEL_PROVIDER": "ollama", "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
               "MODEL_NAME": "synthetic-test"}
        with patch.dict(os.environ, env, clear=True), \
             patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value={"provider": "ollama",
                                                               "model": "synthetic-test", "mode": "local_model"}), \
             patch("app.model._chat", return_value=json.dumps(plan, ensure_ascii=False)):
            return self.send(text, allow_local_model=True, allow_context_extraction=True)

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
                        if message.get("role") == "assistant" and message.get("kind") == "guidance_clarification")
        self.assertIn("أي تطبيق", response["text"])
        self.assertEqual(response["citations"], [])
        self.assertNotIn("GLOBAL-IG-SECURITY-2026", response["text"])
        self.assertTrue(result["question_detected"])
        self.assertEqual(result["session"]["pending_slot"], "narrative")

    def test_short_sourced_whatsapp_answer_resumes_pending_interview_once(self):
        first = self.send("في شخص عم يهددني على واتساب.")
        pending = first["session"]["pending_slot"]
        self.assertEqual(pending, "details")
        question = "كيف أحمي حسابي على واتساب؟"
        result = self.ask_with_simulated_goal_plan(question, question)
        messages = result["session"]["messages"]
        answer = next(message for message in reversed(messages)
                      if message.get("role") == "assistant" and message.get("kind") == "answer")
        followups = [message for message in messages if message.get("role") == "assistant"
                     and message.get("kind") == "question"
                     and message.get("text", "").startswith("نكمل من السؤال السابق:")]
        self.assertEqual(len(followups), 1)
        self.assertEqual(result["session"]["pending_slot"], pending)
        self.assertTrue(answer["citations"])
        self.assertTrue(all(citation["id"].startswith("GLOBAL-WA-") for citation in answer["citations"]))
        self.assertFalse(any(citation["id"].startswith("GLOBAL-IG-") for citation in answer["citations"]))
        self.assertIn("التحقق بخطوتين", answer["text"])
        self.assertIn("نطاق المصادر", answer["text"])
        self.assertNotIn("المقتطف الداعم", answer["text"])
        self.assertTrue(all(citation.get("url") and citation.get("excerpt") for citation in answer["citations"]))

    def test_pending_platform_clarification_survives_pause_and_resume(self):
        text = "بدي نصيحة أحافظ فيها على حسابي."
        pending = self.ask_with_simulated_goal_plan(text, text)
        self.assertEqual(pending["session"]["pending_guidance"]["status"], "awaiting_platform")
        clarification_count = sum(message.get("kind") == "guidance_clarification"
                                  for message in pending["session"]["messages"])
        paused = self.send("احفظ الجلسة")
        self.assertEqual(paused["session"]["stage"], "paused")
        response = self.client.post(f"/api/sessions/{self.session_id}/resume", headers=self.headers,
                                    json={"expected_revision": self.revision})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.revision = result["session"]["revision"]
        state = result["session"]
        self.assertEqual(state["stage"], "interview")
        self.assertEqual(state["pending_slot"], "narrative")
        self.assertEqual(state["pending_guidance"]["status"], "awaiting_platform")
        self.assertEqual(sum(message.get("kind") == "guidance_clarification" for message in state["messages"]),
                         clarification_count + 1)
        self.assertEqual(state["messages"][-1]["kind"], "guidance_clarification")
        self.assertIn("أي تطبيق", state["messages"][-1]["text"])

    def test_jordan_law_article_routes_through_api_and_real_rag_without_platform_prompt(self):
        question = "ما هي المادة 18 من قانون الجرائم الإلكترونية الأردني؟"
        result = self.ask_with_simulated_goal_plan(question, question)
        answer = next(message for message in reversed(result["session"]["messages"])
                      if message.get("role") == "assistant" and message.get("kind") == "answer")
        self.assertTrue(result["question_detected"])
        self.assertEqual({citation["id"] for citation in answer["citations"]},
                         {"JO-CYBERCRIME-LAW-ART18-2023"})
        self.assertIn("المادة 18 تتعلق", answer["text"])
        self.assertNotIn("أي تطبيق", answer["text"])
        self.assertNotIn("GLOBAL-IG-", " ".join(citation["id"] for citation in answer["citations"]))


if __name__ == "__main__":
    unittest.main()
