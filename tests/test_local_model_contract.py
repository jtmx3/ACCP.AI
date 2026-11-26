import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.model import _is_local_ollama_url, _validate_adaptive_question, provider_config
from app.store import Store


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class LocalModelContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_store = main.store
        main.store = Store(str(Path(self.tmp.name) / "local-model-contract.sqlite3"))
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()
        main.store = self.old_store
        self.tmp.cleanup()

    def _new_session(self):
        created = self.client.post("/api/sessions", json={})
        self.assertEqual(created.status_code, 200, created.text)
        result = created.json()
        return (result["session"]["session_id"], result["token"],
                result["session"]["revision"], result["session"])

    def test_rules_and_adaptive_generation_are_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            config = provider_config()
            self.assertEqual(config["provider"], "rules")
            self.assertFalse(config["configured"])
            self.assertFalse(config["adaptive_question_generation_enabled"])

            session_id, token, revision, state = self._new_session()
            self.assertNotIn("local_model_enabled", state)
            self.assertFalse(state["adaptive_questions_enabled"])
            response = self.client.put(
                f"/api/sessions/{session_id}/adaptive-question-settings",
                headers={"X-Case-Token": token},
                json={"expected_revision": revision, "enabled": True},
            )
            self.assertEqual(response.status_code, 409, response.text)
            rejected = self.client.put(
                f"/api/sessions/{session_id}/local-model-settings",
                headers={"X-Case-Token": token},
                json={"expected_revision": revision, "enabled": True},
            )
            self.assertEqual(rejected.status_code, 409, rejected.text)

    def test_new_session_does_not_probe_provider_and_can_disable_configured_loopback_model(self):
        env = {"MODEL_PROVIDER": "ollama", "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
               "MODEL_NAME": "synthetic-local-test"}
        with patch.dict(os.environ, env, clear=True), \
             patch("app.main.probe_provider", side_effect=AssertionError("session creation must not probe")):
            session_id, token, revision, state = self._new_session()
            self.assertTrue(state["local_model_enabled"])
            self.assertFalse(state["adaptive_questions_enabled"])
            response = self.client.put(
                f"/api/sessions/{session_id}/local-model-settings",
                headers={"X-Case-Token": token},
                json={"expected_revision": revision, "enabled": False},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["session"]["local_model_enabled"])
        self.assertFalse(response.json()["session"]["adaptive_questions_enabled"])

    def test_adaptive_option_is_available_with_local_ollama_but_can_be_disabled(self):
        env = {"MODEL_PROVIDER": "ollama", "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
               "MODEL_NAME": "synthetic-local-test"}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(provider_config()["adaptive_question_generation_enabled"])
        with patch.dict(os.environ, {**env, "ACCP_ADAPTIVE_MODEL_QUESTIONS": "0"}, clear=True):
            self.assertFalse(provider_config()["adaptive_question_generation_enabled"])

    def test_provider_accepts_loopback_ollama_and_rejects_remote_provider_or_endpoint(self):
        for url in ("http://127.0.0.1:11434/v1", "http://localhost:11434/v1",
                    "http://[::1]:11434/v1"):
            with self.subTest(url=url):
                self.assertTrue(_is_local_ollama_url(url))
        for url in ("https://127.0.0.1:11434/v1", "http://192.168.1.4:11434/v1",
                    "http://user:pass@localhost:11434/v1", "http://localhost:11434/v1?x=1"):
            with self.subTest(url=url):
                self.assertFalse(_is_local_ollama_url(url))

        with patch.dict(os.environ, {"MODEL_PROVIDER": "remote-provider",
                                     "MODEL_NAME": "synthetic"}, clear=True):
            self.assertFalse(provider_config()["configured"])
            self.assertEqual(provider_config()["failure_code"], "invalid_provider")

        with patch.dict(os.environ, {"MODEL_PROVIDER": "ollama",
                                     "MODEL_BASE_URL": "https://model.example/v1",
                                     "MODEL_NAME": "synthetic"}, clear=True):
            self.assertFalse(provider_config()["configured"])
            self.assertEqual(provider_config()["failure_code"], "nonlocal_endpoint_disallowed")

    def test_unified_local_generation_uses_one_call_and_never_sends_session_credentials(self):
        env = {"MODEL_PROVIDER": "ollama",
               "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
               "MODEL_NAME": "synthetic-local-test",
               "OLLAMA_REASONING_EFFORT": "none"}
        with patch.dict(os.environ, env, clear=True):
            session_id, token, revision, _ = self._new_session()
            headers = {"X-Case-Token": token}
            settings = self.client.put(
                f"/api/sessions/{session_id}/local-model-settings",
                headers=headers,
                json={"expected_revision": revision, "enabled": True},
            )
            self.assertEqual(settings.status_code, 200, settings.text)
            revision = settings.json()["session"]["revision"]

            generated = {"facts": [{"field": "threat", "status": "reported", "value": True,
                                     "current_quote": "في شخص يهددني"}],
                         "user_intent": "add_fact", "conversation_move": "ask_follow_up",
                         "reply_text": "متى بدأ هذا التصرف؟"}
            captured = []

            def fake_chat(messages, purpose):
                captured.append({"messages": messages, "purpose": purpose})
                return json.dumps(generated, ensure_ascii=False)

            user_text = "اسمي سارة ورقمي 0791234567. في شخص يهددني بنشر صوري الخاصة على واتساب."
            with patch("app.model._chat", side_effect=fake_chat):
                response = self.client.post(
                    f"/api/sessions/{session_id}/messages",
                    headers=headers,
                    json={"expected_revision": revision, "text": user_text, "kind": "answer"},
                )

            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertNotIn("adaptive_question", result)
            self.assertEqual(result["local_model_purpose"], "interactive_dialogue_turn")
            self.assertTrue(result["local_context_sent"])
            self.assertEqual(result["session"]["pending_slot"], "initial_time")
            self.assertEqual(result["session"]["messages"][-1]["text"], generated["reply_text"])
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["purpose"], "interactive_dialogue_turn")
            serialized = json.dumps(captured[0], ensure_ascii=False)
            self.assertNotIn(session_id, serialized)
            self.assertNotIn(token, serialized)
            payload = json.loads(next(message["content"] for message in captured[0]["messages"]
                                      if message["role"] == "user"))
            self.assertEqual(payload["pending_slot"], "narrative")
            self.assertEqual(payload["current_user_message"], user_text)
            self.assertNotIn("session_id", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("token", json.dumps(payload, ensure_ascii=False))

    def test_adaptive_question_validator_blocks_safety_and_sensitive_identifier_requests(self):
        context = {"base_question": "ما وسيلة التواصل المتاحة؟", "recent_bot_questions": [],
                   "features": {"contact_available": True}}
        self.assertEqual(_validate_adaptive_question(
            "هل تقدر توضح قناة التواصل المتاحة دون كتابة المعرّف؟", "contact", context), (True, None))
        self.assertEqual(_validate_adaptive_question(
            "ما رقم الواتساب أو اسم المستخدم؟", "contact", context)[1],
            "forbidden_sensitive_data_request")
        self.assertEqual(_validate_adaptive_question(
            "هل أنت بخطر جسدي الآن؟", "direct_danger", context)[1],
            "protected_or_unsupported_slot")
        self.assertEqual(_validate_adaptive_question(
            "ما اسم التطبيق ورقم الحساب؟", "platform", context)[1],
            "forbidden_sensitive_data_request")

    def test_adaptive_question_relevance_accepts_natural_paraphrases_and_rejects_off_topic_text(self):
        context = {"base_question": "متى بدأ التصرف أو وصلك أول تواصل؟", "recent_bot_questions": [],
                   "features": {"initial_time": "unknown"}}
        for question in (
            "وبالنسبة لأول تواصل، متى بلش تقريبًا؟",
            "قديش صار من وقت ما بعتلك أول رسالة؟",
            "بتتذكر تقريبًا بأي يوم بدأ يحكي معك؟",
        ):
            with self.subTest(question=question):
                self.assertEqual(_validate_adaptive_question(question, "initial_time", context), (True, None))
        self.assertEqual(_validate_adaptive_question(
            "هل عندك لقطات شاشة أو رسائل تثبت ذلك؟", "initial_time", context)[1],
            "slot_topic_mismatch")


if __name__ == "__main__":
    unittest.main()
