import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.core import _assessment, handle_turn, new_state
from app.dialogue import build_dialogue_context, validate_dialogue_reply
from app.model import generate_interactive_dialogue_turn
from app.rag import answer_question
from app.stopping import _recent_unknown_streak
from app.store import Store
from app.triage import route_assessment


ART15 = "JO-CYBERCRIME-LAW-ART15-2023"


class ConversationFactIntegrityRegressionTests(unittest.TestCase):
    def test_unknown_evidence_answer_is_saved_as_unknown_instead_of_boolean_status(self):
        state = new_state("unknown-evidence-status")
        state["pending_slot"] = "evidence"
        text = "ما بعرف إذا عندي لقطات شاشة"

        handle_turn(state, text)

        evidence = state["facts"]["evidence"]
        self.assertEqual(evidence["status"], "unknown")
        self.assertIsNone(evidence["value"])
        self.assertEqual(evidence["quote"], text)

    def test_wrap_up_does_not_become_identity_when_identity_is_pending(self):
        state = new_state("wrap-up-identity")
        handle_turn(state, "في واحد شتمني بتعليق على منشور عندي على فيسبوك")
        self.assertEqual(state["pending_slot"], "identity")
        handle_turn(state, "بس هذا اللي صار ما عندي تفاصيل ثانية")
        self.assertEqual(state["stage"], "review")
        self.assertNotIn("identity", state["facts"])
        self.assertEqual(state["pending_slot"], None)

    def test_ongoing_contact_does_not_deny_repeated_and_publication_answer_is_separate(self):
        state = new_state("repeat-vs-contact")
        turns = (
            "في شخص معي بالجامعة كان معي على علاقثه حب و كنت ابعتلو صوري و انا عارية و لما انفصلنا صار يهددني بنشر الصور او اني ارجع ابعتلو صور اكثر",
            "قبل ساعه",
            "ما زال يراسلني الان",
            "لا ما نشر شي",
        )
        for text in turns[:3]:
            handle_turn(state, text)
        self.assertEqual(state["facts"]["ongoing_contact"]["status"], "reported")
        self.assertIs(state["facts"]["ongoing_contact"]["value"], True)
        self.assertEqual(state["facts"]["repeated"]["status"], "unresolved")
        self.assertIsNone(state["facts"]["repeated"]["value"])
        self.assertEqual(state["facts"]["repeated"]["resolution_reason"],
                         "ongoing_contact_without_threat_confirmation")
        self.assertEqual(state["pending_slot"], "published")
        handle_turn(state, turns[3])
        self.assertEqual(state["facts"]["published"]["status"], "denied")
        self.assertIs(state["facts"]["published"]["value"], False)
        self.assertEqual(state["pending_slot"], "identity")
        self.assertNotIn("911", " ".join(item["text"] for item in state["messages"] if item["role"] == "assistant"))

    def test_complete_image_extortion_example_reaches_review_at_priority_90(self):
        state = new_state("complete-image-extortion")
        turns = (
            "في شخص معي بالجامعة كان معي على علاقثه حب و كنت ابعتلو صوري و انا عارية و لما انفصلنا صار يهددني بنشر الصور او اني ارجع ابعتلو صور اكثر",
            "قبل ساعه",
            "ما زال يراسلني الان",
            "لا ما نشر شي",
            "نعم بعرفه وعندي لقطات شاشة",
        )
        for text in turns:
            handle_turn(state, text)
        self.assertEqual(state["stage"], "review")
        self.assertEqual(state["facts"]["ongoing_contact"]["value"], True)
        self.assertEqual(state["facts"]["repeated"]["status"], "unresolved")
        self.assertEqual(state["facts"]["published"]["status"], "denied")
        self.assertEqual(state["assessment"]["priority_score"], 90)
        assistant_text = " ".join(item["text"] for item in state["messages"] if item["role"] == "assistant")
        self.assertNotIn("911", assistant_text)
        self.assertIn("رتبت ما أفدت به عن التهديد أو الطلب أو الحساب", assistant_text)

    def test_time_protest_acknowledges_saved_time_and_keeps_repeated_question_without_invention(self):
        state = new_state("timestamp-protest")
        for text in ("في شخص هددني بنشر صوري الخاصة", "قبل ساعة", "قلت لك قبل ساعة"):
            handle_turn(state, text)
        assistant_text = " ".join(item["text"] for item in state["messages"] if item["role"] == "assistant")
        self.assertNotIn("ذكرت أن التواصل ما زال مستمرًا", assistant_text)
        self.assertNotIn("ongoing_contact", state["facts"])
        self.assertNotIn("repeated", state["facts"])
        self.assertEqual(state["stage"], "interview")
        self.assertEqual(state["pending_slot"], "repeated")
        self.assertIn("التوقيت الذي ذكرته مسجل عندي", assistant_text)

    def test_two_unknown_answers_for_different_slots_stop_without_new_substantive_fact(self):
        state = new_state("unknown-streak-different-slots")
        handle_turn(state, "في شخص هددني برسالة وطالبني 100 دينار")
        handle_turn(state, "قال رح يفضحني")
        first_slot = state["pending_slot"]
        handle_turn(state, "ما بعرف")
        second_slot = state["pending_slot"]
        self.assertNotEqual(first_slot, second_slot)
        handle_turn(state, "ما بتذكر")
        self.assertEqual(state["stage"], "review")
        self.assertEqual(state["stopping_decision"]["stop_reason"], "diminishing_returns")

    def test_recent_unknown_streak_crosses_slots_but_not_substantive_turns(self):
        state = new_state("streak-helper")
        state["turn_signals"] = [
            {"slot": "deadline", "unknown": True, "not_remembered": False, "substantive_fact": False},
            {"slot": "identity", "unknown": False, "not_remembered": True, "substantive_fact": False},
        ]
        self.assertEqual(_recent_unknown_streak(state), 2)
        state["turn_signals"].append({"slot": "evidence", "unknown": False, "not_remembered": False,
                                      "substantive_fact": True})
        self.assertEqual(_recent_unknown_streak(state), 0)


class RetrievalAndTriageRegressionTests(unittest.TestCase):
    def test_jordan_art15_retrieves_for_both_insult_questions_without_platform(self):
        questions = (
            "شو عقوبة الشتم بتعليق بالأردن؟",
            "شو حقوقي إذا حدا شتمني على الإنترنت بالأردن؟",
        )
        for question in questions:
            with self.subTest(question=question):
                result = answer_question(question, "JO")
                self.assertEqual({item["id"] for item in result["citations"]}, {ART15})
                self.assertIsNone(result["clarification_required"])
                self.assertIsNone(result["resolved_platform"])
                self.assertIn("المادة 15", result["text"])
                self.assertNotIn("أي تطبيق", result["text"])

    def test_defamation_score_progression_private_public_and_aggravated(self):
        cases = (
            ("بعتلي رسالة خاصة فيها شتم", 35),
            ("شتمني بتعليق على منشور عام بفيسبوك", 45),
            ("كرر التشهير في منشور عام واتهم عائلتي زورًا", 55),
            ("كتب عني كلام كاذب ومسيء وشهّر بعائلتي بمنشور عام على إنستغرام", 55),
            ("شتمني في منشور عام وذكر عائلتي", 55),
            ("شتمني في منشور عام وذكر عيلتي", 55),
            ("شتمني في منشور عام وذكر أهلي", 55),
            ("شتمني في منشور عام وكتب كلام كاذب", 55),
            ("شتمني في منشور عام مع تشهير", 55),
            ("شتمني في منشور عام وشهر فيني", 55),
        )
        for narrative, expected in cases:
            state = new_state("defamation-score")
            handle_turn(state, narrative)
            state["facts"].setdefault("details", {"status": "reported", "value": narrative, "quote": narrative})
            with self.subTest(narrative=narrative):
                self.assertEqual(_assessment(state)["priority_score"], expected)
                self.assertEqual(route_assessment(state)["priority_score"], expected)

    def test_impersonation_and_extortion_scores_stay_in_requested_ranges(self):
        impersonation = new_state("impersonation-score")
        handle_turn(impersonation, "حساب مزيف على إنستغرام يدعي أنه أنا")
        self.assertIn(_assessment(impersonation)["priority_score"], range(60, 76))
        extortion = new_state("extortion-score")
        handle_turn(extortion, "شخص هددني بنشر صوري الخاصة إذا ما دفعت")
        self.assertIn(_assessment(extortion)["priority_score"], range(85, 96))


class InteractiveDialogueContractTests(unittest.TestCase):
    def test_context_is_bounded_and_never_contains_credentials_or_session_id(self):
        state = new_state("secret-session-id-must-not-leak")
        state["token"] = "secret-token-value"
        for index in range(8):
            state["messages"].append({"role": "user" if index % 2 else "assistant",
                                      "text": "تفصيل عربي " + str(index) + " x" * 1800})
        context = build_dialogue_context(state, current_text="آخر إجابة")
        encoded = json.dumps(context, ensure_ascii=False)
        self.assertLessEqual(len(encoded), 5500)
        self.assertNotIn("secret-token-value", encoded)
        self.assertNotIn("secret-session-id-must-not-leak", encoded)
        self.assertLessEqual(len(context["recent_messages"]), 6)
        self.assertLessEqual(len(context["recent_assistant_questions"]), 3)

    def test_reply_validator_accepts_free_arabic_and_rejects_unsafe_or_engine_conflicts(self):
        self.assertEqual(validate_dialogue_reply(
            "واضح أن هذا مزعج. متى وصلك أول تواصل تقريبًا؟", should_stop=False,
            recent_questions=[]), (True, None))
        unsafe = (
            "أرسل كلمة المرور حتى أساعدك.",
            "اتصل على 911 الآن.",
            "حوّل 500 دينار فورًا؟",
            "تم تقديم البلاغ رسميًا نيابة عنك.",
            "هل أرسل لك بلاغًا رسميًا؟ كيف كان ذلك؟",
        )
        for reply in unsafe:
            with self.subTest(reply=reply):
                self.assertFalse(validate_dialogue_reply(reply, should_stop=False,
                                                         context={"recent_messages": []},
                                                         recent_questions=[])[0])
        self.assertFalse(validate_dialogue_reply("هل أنت بخطر؟", should_stop=True,
                                                 recent_questions=[])[0])
        self.assertFalse(validate_dialogue_reply("هل تكرر؟", should_stop=False,
                                                 recent_questions=["هل تكرر؟"])[0])

    def test_reply_validator_normalizes_hamza_variants_before_forbidden_pattern_checks(self):
        replies = (
            "ممكن أرسل رقم الهاتف حتى نتحقق؟",
            "ممكن أرفق الأدلة هنا للتأكد؟",
            "ارفق الادله هنا من فضلك؟",
        )
        for reply in replies:
            with self.subTest(reply=reply):
                self.assertEqual(validate_dialogue_reply(
                    reply, should_stop=False, recent_questions=[]),
                    (False, "unsafe_solicitation_or_unsupported_action_claim"))

    def test_reply_validator_rejects_high_jaccard_question_paraphrase_but_allows_distinct_topic(self):
        previous = "هل تكرر التهديد أو استمر التواصل؟"
        near_duplicate = "هل استمر التواصل أو تكرر التهديد نفسه؟"
        accepted, reason = validate_dialogue_reply(
            near_duplicate, should_stop=False, recent_questions=[previous])
        self.assertFalse(accepted)
        self.assertEqual(reason, "repeated_recent_question")

        accepted, reason = validate_dialogue_reply(
            "متى بدأ التهديد وأول تواصل تقريبًا؟", should_stop=False,
            recent_questions=[previous])
        self.assertTrue(accepted)
        self.assertIsNone(reason)

    def test_interactive_dialogue_makes_one_local_call_without_session_secrets(self):
        request_text = "ما حدد مهلة للدفع"
        payload = {"facts": [], "user_intent": "answer_pending_question",
                   "conversation_move": "ask_follow_up", "reply_text": "سجلت ذلك. هل دفعت أي مبلغ؟"}
        with patch.dict(os.environ, {"MODEL_PROVIDER": "ollama", "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
                                     "MODEL_NAME": "Jais-3-3.7B-GGUF:Q4_K_M", "MODEL_TIMEOUT_SECONDS": "120",
                                     "OLLAMA_REASONING_EFFORT": "none"}, clear=True), \
             patch("app.model.urllib.request.urlopen", return_value=_JsonResponse(payload)) as urlopen:
            result = generate_interactive_dialogue_turn(request_text, {"recent_messages": []}, "deadline")
        self.assertEqual(urlopen.call_count, 1)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["facts"], [])
        request = urlopen.call_args.args[0]
        sent = json.loads(request.data.decode("utf-8"))
        def keys(value):
            if isinstance(value, dict):
                return set(value).union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value)) if value else set()
            return set()
        self.assertFalse({"session_id", "token", "session_token"}.intersection(keys(sent)))
        model_input = json.loads(sent["messages"][1]["content"])
        self.assertEqual(model_input["pending_slot"], "deadline")


class SessionCreationAndUiRegressionTests(unittest.TestCase):
    def test_session_creation_does_not_probe_model(self):
        old_store = main.store
        with tempfile.TemporaryDirectory() as directory:
            main.store = Store(str(Path(directory) / "api.sqlite3"))
            client = TestClient(main.app)
            try:
                with patch("app.main.provider_config", return_value={"configured": True, "mode": "local_model"}), \
                     patch("app.main.probe_provider", side_effect=AssertionError("create_session must not probe")):
                    response = client.post("/api/sessions", json={})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(response.json()["session"].get("local_model_enabled", False))
            finally:
                client.close()
                main.store = old_store

    def test_api_uses_one_interactive_call_and_applies_safe_natural_followup(self):
        old_store = main.store
        with tempfile.TemporaryDirectory() as directory:
            main.store = Store(str(Path(directory) / "api.sqlite3"))
            client = TestClient(main.app)
            turn = {"facts": [], "message_plan": {"goals": [], "unsupported_goals": [],
                                                    "guidance_question": None},
                    "user_intent": None, "conversation_move": "ask_follow_up",
                    "reply_text": "واضح أن احتمال نشر الصور يقلقك. هل صدر تهديد صريح بالنشر، أم أنه خوف من احتمال فقط?"}
            config = {"configured": True, "mode": "local_model", "provider": "ollama",
                      "model": "Jais-3-3.7B-GGUF:Q4_K_M"}
            try:
                with patch("app.main.provider_config", return_value=config), \
                     patch("app.main.probe_provider", side_effect=AssertionError("must not probe")), \
                     patch("app.main.model_configured", return_value=True), \
                     patch("app.main.generate_interactive_dialogue_turn", return_value=turn) as generate:
                    created = client.post("/api/sessions", json={})
                    self.assertEqual(created.status_code, 200, created.text)
                    session = created.json()["session"]
                    self.assertTrue(session["local_model_enabled"])
                    posted = client.post(
                        f"/api/sessions/{session['session_id']}/messages",
                        headers={"X-Case-Token": created.json()["token"]},
                        json={"expected_revision": session["revision"],
                              "text": "في شب معي بجامعتي ومعه صوري وأنا خايفة من أنه ينشرهم"})
                self.assertEqual(posted.status_code, 200, posted.text)
                self.assertEqual(generate.call_count, 1)
                result = posted.json()
                self.assertTrue(result["dialogue_reply"]["accepted"])
                self.assertTrue(result["dialogue_reply"]["applied"])
                self.assertEqual(result["runtime_mode"], "local_model")
                self.assertFalse(result["local_model_result_accepted"])
                self.assertEqual(result["session"]["messages"][-1]["text"], turn["reply_text"])
                self.assertEqual(result["session"]["messages"][-1]["question_plan"]["slot"],
                                 "publication_threat")
            finally:
                client.close()
                main.store = old_store


    def test_legacy_sqlite_sessions_load_resume_confirm_and_export(self):
        old_store = main.store
        with tempfile.TemporaryDirectory() as directory:
            main.store = Store(str(Path(directory) / "api.sqlite3"))
            client = TestClient(main.app)
            try:
                for stage in ("review", "paused"):
                    state, token = main.store.create()
                    with main.store._connection() as conn:
                        raw = conn.execute("SELECT state_json FROM sessions WHERE id=?",
                                           (state["session_id"],)).fetchone()[0]
                        legacy = json.loads(raw)
                        for key in ("schema_version", "turn_signals", "question_history", "slot_states",
                                    "money_corrections", "pending_guidance", "clarifications", "assessment",
                                    "confirmed_digest", "confirmation_history", "adaptive_questions_enabled",
                                    "summary_version", "last_response"):
                            legacy.pop(key, None)
                        legacy["stage"] = stage
                        if stage == "paused":
                            legacy["resume_stage"] = "interview"
                        conn.execute("UPDATE sessions SET state_json=? WHERE id=?",
                                     (json.dumps(legacy, ensure_ascii=False), state["session_id"]))
                    headers = {"X-Case-Token": token}
                    loaded = client.get(f"/api/sessions/{state['session_id']}", headers=headers)
                    self.assertEqual(loaded.status_code, 200, loaded.text)
                    if stage == "paused":
                        resumed = client.post(f"/api/sessions/{state['session_id']}/resume", headers=headers,
                                              json={"expected_revision": 0})
                        self.assertEqual(resumed.status_code, 200, resumed.text)
                        self.assertEqual(resumed.json()["session"]["stage"], "interview")
                    else:
                        markdown = client.get(f"/api/sessions/{state['session_id']}/report.md", headers=headers)
                        exported = client.get(f"/api/sessions/{state['session_id']}/report.json", headers=headers)
                        self.assertEqual(markdown.status_code, 200)
                        self.assertEqual(exported.status_code, 200)
                        self.assertIn("ملخص الحالة", markdown.text)
                        self.assertEqual(exported.json()["summary_version"], 1)
                        confirmed = client.post(f"/api/sessions/{state['session_id']}/confirm", headers=headers,
                                                json={"expected_revision": 0})
                        self.assertEqual(confirmed.status_code, 200, confirmed.text)
                        self.assertTrue(confirmed.json()["session"]["confirmed"])
            finally:
                client.close()
                main.store = old_store


class _JsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": json.dumps(self.payload, ensure_ascii=False)}}]},
                          ensure_ascii=False).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
