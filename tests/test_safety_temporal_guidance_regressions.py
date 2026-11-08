import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.model import InterviewUnderstanding
from app.store import Store


class SafetyTemporalGuidanceRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        main.store = Store(str(Path(self.tmp.name) / "regressions.sqlite3"))
        self.client = TestClient(main.app)
        self._create_session()

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def _create_session(self):
        created = self.client.post("/api/sessions", json={})
        self.assertEqual(created.status_code, 200, created.text)
        self.session_id = created.json()["session"]["session_id"]
        self.token = created.json()["token"]
        self.headers = {"X-Case-Token": self.token}
        self.revision = created.json()["session"]["revision"]
        return created.json()["session"]

    def _turn(self, text, *, kind="answer", **kwargs):
        response = self.client.post(
            f"/api/sessions/{self.session_id}/messages",
            headers=self.headers,
            json={"expected_revision": self.revision, "text": text, "kind": kind, **kwargs},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.revision = result["session"]["revision"]
        return result

    def _stored(self):
        return self.client.get(f"/api/sessions/{self.session_id}", headers=self.headers).json()["session"]

    def _establish_initial_time_prompt(self):
        first = self._turn("في شخص على واتساب طلب مني 100 دينار، وقال رح ينشر محادثاتي الخاصة إذا ما دفعت.")
        self.assertEqual(first["session"]["pending_slot"], "direct_danger")
        second = self._turn("لا، ما في خطر مباشر علي الآن، بس اليوم رجع بعتلي نفس التهديد.")
        self.assertEqual(second["session"]["pending_slot"], "initial_time")
        return second["session"]

    def _establish_payment_prompt(self):
        self._establish_initial_time_prompt()
        third = self._turn("أول رسالة وصلتني مبارح حوالي الساعة 6 المسا.")
        self.assertEqual(third["session"]["pending_slot"], "paid_amount")
        return third["session"]

    def test_direct_danger_interrupts_pending_platform_clarification_and_is_persisted(self):
        requested = self._turn("كيف أحمي حسابي")
        self.assertEqual(requested["session"]["pending_guidance"]["status"], "awaiting_platform")

        with patch("app.main.answer_question", side_effect=AssertionError("safety routing must not call RAG")):
            urgent = self.client.post(
                f"/api/sessions/{self.session_id}/messages", headers=self.headers,
                json={"expected_revision": self.revision,
                      "text": "أنا بخطر مباشر الآن والشخص واقف قدام بيتي ومعه سكين.", "kind": "answer"},
            )
        self.assertEqual(urgent.status_code, 200, urgent.text)
        self.revision = urgent.json()["session"]["revision"]
        result = urgent.json()
        state = result["session"]
        self.assertEqual(state["stage"], "urgent_stop")
        self.assertIsNone(state.get("pending_guidance"))
        danger = state["facts"]["direct_danger"]
        self.assertEqual((danger["status"], danger["value"]), ("reported", True))
        user_message = next(message for message in reversed(state["messages"])
                            if message["role"] == "user")
        self.assertEqual(danger["quote"], user_message["text"])
        self.assertEqual(result["report"]["facts"]["direct_danger"]["value"], True)
        self.assertFalse(result["external_data_sent"])
        self.assertFalse(result["external_model_used"])
        safety = next(message for message in reversed(state["messages"])
                      if message["role"] == "assistant")
        self.assertEqual(safety["kind"], "safety")
        self.assertIn("سلامتك أولًا", safety["text"])
        self.assertEqual(state["assessment"]["priority_score"], 100)

        revision_before_reads = state["revision"]
        report = self.client.get(f"/api/sessions/{self.session_id}/report.json", headers=self.headers)
        markdown = self.client.get(f"/api/sessions/{self.session_id}/report.md", headers=self.headers)
        fetched = self._stored()
        self.assertEqual(report.json()["facts"]["direct_danger"]["quote"], user_message["text"])
        self.assertIn(user_message["text"], markdown.text)
        self.assertEqual(fetched["revision"], revision_before_reads)
        self.assertEqual(fetched["facts"]["direct_danger"]["value"], True)

    def test_direct_danger_interrupts_interview_question_and_resumed_guidance(self):
        # Ordinary interview route.
        self._turn("في شخص يهددني بنشر محادثاتي الخاصة.")
        answer = self._turn("أنا بخطر مباشر الآن والشخص واقف قدام بيتي ومعه سكين.")
        self.assertEqual(answer["session"]["stage"], "urgent_stop")
        self.assertEqual(answer["session"]["facts"]["direct_danger"]["value"], True)

        # Explicit question route must still screen and save a direct-danger disclosure.
        self._create_session()
        question = self._turn("أنا بخطر مباشر الآن والشخص واقف قدام بيتي ومعه سكين.", kind="question")
        self.assertEqual(question["session"]["stage"], "urgent_stop")
        self.assertEqual(question["session"]["facts"]["direct_danger"]["status"], "reported")

        # A paused/resumed pending guidance clarification must not swallow the safety report.
        self._create_session()
        pending = self._turn("كيف أحمي حسابي")
        self.assertEqual(pending["session"]["pending_guidance"]["status"], "awaiting_platform")
        paused = self._turn("احفظ الجلسة")
        self.assertEqual(paused["session"]["stage"], "paused")
        resumed = self.client.post(f"/api/sessions/{self.session_id}/resume", headers=self.headers,
                                   json={"expected_revision": self.revision})
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.revision = resumed.json()["session"]["revision"]
        urgent = self._turn("أنا بخطر مباشر الآن والشخص واقف قدام بيتي ومعه سكين.")
        self.assertEqual(urgent["session"]["stage"], "urgent_stop")
        self.assertEqual(urgent["session"]["facts"]["direct_danger"]["value"], True)

    def test_same_day_temporal_rephrasing_is_support_not_conflict(self):
        self._establish_initial_time_prompt()
        original_quote = "أول رسالة وصلتني مبارح حوالي الساعة 6 المسا"
        original = self._turn(original_quote)["session"]["facts"]["initial_time"]
        self.assertEqual(original["status"], "reported")

        repeated_quote = "أول رسالة وصلتني أمس حوالي الساعة 6 مسا"
        repeated = self._turn(repeated_quote)["session"]["facts"]["initial_time"]
        self.assertEqual(repeated["status"], "reported")
        self.assertEqual(repeated["quote"], original_quote)
        self.assertEqual(repeated["value"], original["value"])
        self.assertTrue(any(item.get("quote") == repeated_quote
                            for item in repeated.get("supporting_statements", [])))

    def test_real_hour_or_duration_change_remains_a_temporal_conflict(self):
        self._establish_initial_time_prompt()
        self._turn("أول رسالة وصلتني أمس الساعة 6 المسا.")
        changed_hour = self._turn("أول رسالة وصلتني أمس الساعة 7 المسا.")["session"]["facts"]["initial_time"]
        self.assertEqual(changed_hour["status"], "conflict")

        self._create_session()
        self._establish_initial_time_prompt()
        self._turn("أول رسالة وصلتني قبل ساعتين.")
        changed_duration = self._turn("أول رسالة وصلتني قبل يومين.")["session"]["facts"]["initial_time"]
        self.assertEqual(changed_duration["status"], "conflict")
        self.assertEqual({item.get("value") for item in changed_duration["alternatives"] if item.get("value")},
                         {"قبل ساعتين", "قبل يومين"})

    def test_relative_day_synonyms_are_not_equated_across_message_dates(self):
        self._establish_initial_time_prompt()
        first = self._turn("أول رسالة وصلتني أمس الساعة 6 المسا.")
        fact = first["session"]["facts"]["initial_time"]
        original_message = next(message for message in first["session"]["messages"]
                                if message["id"] == fact["message_id"])
        first_date = datetime.fromisoformat(original_message["created_at"]).date()
        next_day = (first_date + timedelta(days=1)).isoformat() + "T12:00:00+00:00"
        with patch("app.core.now_iso", return_value=next_day):
            second = self._turn("أول رسالة وصلتني مبارح الساعة 6 المسا.")
        changed = second["session"]["facts"]["initial_time"]
        self.assertEqual(changed["status"], "conflict")
        self.assertEqual(len(changed["alternatives"]), 2)

    def test_first_message_marker_beats_generic_contact_verb(self):
        prior = self._establish_initial_time_prompt()
        previous_latest_message_id = prior["facts"]["latest_time"]["message_id"]
        result = self._turn("أول رسالة بعتلي إياها مبارح الساعة 6 المسا.")
        facts = result["session"]["facts"]
        self.assertEqual(facts["initial_time"]["status"], "reported")
        self.assertEqual(facts["initial_time"]["value"], "مبارح الساعة 6 المسا")
        self.assertEqual(facts["latest_time"]["value"], "اليوم")
        self.assertEqual(facts["latest_time"]["message_id"], previous_latest_message_id)

    def test_temporal_hour_is_not_recorded_as_paid_amount_when_payment_question_is_pending(self):
        self._establish_payment_prompt()
        revision_before = self.revision
        result = self._turn("أول رسالة وصلتني أمس حوالي الساعة 6 المسا.")
        state = result["session"]
        self.assertEqual(state["facts"]["requested_amount"]["value"], "100 دينار")
        self.assertNotIn("paid_amount", state["facts"])
        self.assertFalse(any(event.get("kind") == "paid" for event in state["money_events"]))
        self.assertEqual(state["pending_slot"], "paid_amount")
        self.assertEqual(state["revision"], revision_before + 1)
        self.assertEqual(result["report"]["facts"]["initial_time"]["status"], "reported")

        # A short, explicit numeric answer to the still-pending payment question remains valid.
        paid = self._turn("130")
        self.assertEqual(paid["session"]["facts"]["paid_amount"]["value"], "130")
        self.assertTrue(any(event.get("kind") == "paid" and event.get("amount") == "130"
                            for event in paid["session"]["money_events"]))

    def test_mixed_guidance_clarification_answers_real_rag_and_records_new_threat_once(self):
        asked = self._turn("كيف أحمي حسابي")
        self.assertEqual(asked["session"]["pending_guidance"]["status"], "awaiting_platform")
        self._turn("احفظ الجلسة")
        resumed = self.client.post(f"/api/sessions/{self.session_id}/resume", headers=self.headers,
                                   json={"expected_revision": self.revision})
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.revision = resumed.json()["session"]["revision"]

        result = self._turn("واتساب، واليوم رجع هددني.")
        state = result["session"]
        self.assertIsNone(state.get("pending_guidance"))
        self.assertEqual(state["facts"]["threat"]["value"], True)
        self.assertEqual(state["facts"]["latest_time"]["value"], "اليوم")
        self.assertNotIn("platform", state["facts"], "WhatsApp here scopes guidance, not the incident platform")
        question_messages = [message for message in state["messages"]
                             if message["role"] == "assistant" and message.get("kind") in {"question", "clarification"}
                             and message.get("question_plan", {}).get("slot") == state["pending_slot"]]
        self.assertEqual(len(question_messages), 1)
        self.assertEqual(state["pending_slot"], "direct_danger")
        answer = next(message for message in reversed(state["messages"])
                      if message["role"] == "assistant" and message["kind"] == "answer")
        source_ids = {citation["id"] for citation in answer["citations"]}
        self.assertTrue(source_ids & {"GLOBAL-WA-ACCOUNT-SECURITY-2026", "GLOBAL-WA-TWO-STEP-2026"})
        self.assertNotIn("GLOBAL-IG-SECURITY-2026", source_ids)
        self.assertIn("اليوم رجع هددني", state["facts"]["threat"]["quote"])

        before_get_revision = state["revision"]
        report = self.client.get(f"/api/sessions/{self.session_id}/report.json", headers=self.headers).json()
        markdown = self.client.get(f"/api/sessions/{self.session_id}/report.md", headers=self.headers).text
        fetched = self._stored()
        self.assertEqual(report["facts"]["threat"]["quote"], state["facts"]["threat"]["quote"])
        self.assertIn("اليوم رجع هددني", markdown)
        self.assertEqual(fetched["revision"], before_get_revision)

    def test_user_can_cancel_pending_guidance_and_resume_interview_without_platform_prompt(self):
        pending = self._turn("كيف أحمي حسابي")
        self.assertEqual(pending["session"]["pending_guidance"]["status"], "awaiting_platform")
        before = len(pending["session"]["messages"])
        cancelled = self._turn("خلص مش بدي نصيحة، نكمل الإفادة.")
        state = cancelled["session"]
        self.assertIsNone(state.get("pending_guidance"))
        new_assistant = [message for message in state["messages"][before:]
                         if message["role"] == "assistant"]
        self.assertEqual(len(new_assistant), 1)
        self.assertIn("نكمل", new_assistant[0]["text"])
        self.assertNotIn("أي تطبيق", new_assistant[0]["text"])
        self.assertEqual(new_assistant[0].get("question_plan", {}).get("slot"), "narrative")
        self.assertEqual(state["pending_slot"], "narrative")

    def test_guidance_cancellation_accepts_decline_or_explicit_interview_resume(self):
        for index, cancellation in enumerate(("ما بدي نصيحة.", "نكمل الإفادة.")):
            if index:
                self._create_session()
            self._turn("كيف أحمي حسابي")
            previous_count = len(self._stored()["messages"])
            result = self._turn(cancellation)
            state = result["session"]
            self.assertIsNone(state.get("pending_guidance"))
            new_assistant = [message for message in state["messages"][previous_count:]
                             if message["role"] == "assistant"]
            self.assertEqual(len(new_assistant), 1)
            self.assertEqual(new_assistant[0].get("question_plan", {}).get("slot"), "narrative")

    def test_consented_model_extraction_receives_fact_span_without_guidance_platform(self):
        self._turn("كيف أحمي حسابي")
        fake_understanding = InterviewUnderstanding([], {"goals": [], "guidance_question": None}, {"accepted": True})
        with patch("app.main.model_configured", return_value=True), \
             patch("app.main.provider_config", return_value={"provider": "ollama", "model": "fixture", "mode": "local_model"}), \
             patch("app.main.generate_interview_extraction", return_value=fake_understanding) as extraction:
            self._turn("واتساب، واليوم رجع هددني.", allow_external_extraction=True)
        extraction.assert_called_once()
        self.assertEqual(extraction.call_args.args[0], "واليوم رجع هددني")

    def test_explicit_temporal_resolution_still_checks_revision_and_keeps_conflict_history(self):
        self._establish_initial_time_prompt()
        self._turn("أول رسالة وصلتني أمس الساعة 6 المسا.")
        conflicted = self._turn("للتوضيح، أول رسالة وصلتني اليوم الساعة 9 الصبح.")
        fact = conflicted["session"]["facts"]["initial_time"]
        self.assertEqual(fact["status"], "conflict")
        revision = self.revision
        endpoint = f"/api/sessions/{self.session_id}/resolve-temporal-conflict"
        stale = self.client.post(endpoint, headers=self.headers, json={
            "expected_revision": revision - 1, "field": "initial_time",
            "selected_value": "أمس الساعة 6 المسا", "resolution_reason": "أختار إفادة الساعة السادسة",
        })
        self.assertEqual(stale.status_code, 409)
        resolved = self.client.post(endpoint, headers=self.headers, json={
            "expected_revision": revision, "field": "initial_time",
            "selected_value": "أمس الساعة 6 المسا", "resolution_reason": "أختار إفادة الساعة السادسة",
        })
        self.assertEqual(resolved.status_code, 200, resolved.text)
        resolved_fact = resolved.json()["session"]["facts"]["initial_time"]
        self.assertEqual(resolved_fact["status"], "reported")
        self.assertEqual(resolved_fact["resolution"]["selected_value"], "أمس الساعة 6 المسا")
        self.assertEqual(resolved_fact["history"][-1]["reason"], "resolved_by_user_selection")
