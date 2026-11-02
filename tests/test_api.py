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

    def test_immediate_danger_in_question_mode_bypasses_rag(self):
        state = self.turn("الشخص وصل لعندي الآن ومعه سلاح، شو أعمل؟", kind="question")
        self.assertEqual(state["stage"], "urgent_stop")
        self.assertEqual(state["facts"]["direct_danger"]["value"], True)
        safety = state["messages"][-1]
        self.assertEqual(safety["kind"], "safety")
        self.assertEqual(safety["citations"], [])
        self.assertFalse(safety["external_model_used"])

    def test_unresolved_conflict_blocks_confirmation_until_corrected(self):
        for text in ("في شخص يهددني", "لا", "قال ادفع 100 دينار", "طلب 120 دينار أيضًا"):
            self.turn(text)
        self.assertEqual(self.turn("اعرض الملخص")["stage"], "review")
        blocked = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers, json={"expected_revision": self.revision})
        self.assertEqual(blocked.status_code, 422)
        corrected = self.turn("تصحيح: المبلغ المطلوب 100 دينار.")
        self.assertEqual(corrected["facts"]["requested_amount"]["status"], "reported")
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
        report = self.client.get(f"/api/sessions/{self.case_id}/report.md", headers=self.headers)
        self.assertEqual(report.status_code, 200)
        self.assertIn("100 دينار", report.text)
        self.assertIn("130 دينار", report.text)
        confirmed = self.client.post(f"/api/sessions/{self.case_id}/confirm", headers=self.headers, json={"expected_revision": self.revision})
        self.assertEqual(confirmed.status_code, 200)
        self.revision = confirmed.json()["session"]["revision"]
        self.assertTrue(confirmed.json()["session"]["confirmed"])
        confirmed_version = confirmed.json()["session"]["summary_version"]
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
