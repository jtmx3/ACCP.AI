import json
import unittest
from pathlib import Path

from app.model import validate_interview_facts_diagnostic


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "tests" / "extraction_contrastive_v1.json"


class ContrastiveExtractionValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))

    def test_fixture_contrastive_targets_are_accepted_and_exact(self):
        """Synthetic expected outputs exercise the real project validator, not an LLM."""
        self.assertTrue(self.suite["synthetic"])
        self.assertEqual(len(self.suite["cases"]), 17)
        for case in self.suite["cases"]:
            with self.subTest(case=case["id"]):
                payload = {
                    "message_goals": case["expected_goals"],
                    "guidance_question": None,
                    "facts": case["expected_facts"],
                }
                result = validate_interview_facts_diagnostic(
                    case["message"], json.dumps(payload, ensure_ascii=False),
                    pending_slot=case["pending_slot"])
                self.assertTrue(result["accepted"], result)
                self.assertEqual(result["facts"], case["expected_facts"])
                self.assertEqual(result["filtered_facts"], [])

    def test_unsupported_decline_goal_is_reported_but_not_applied(self):
        text = "لا، ما حدد مهلة أو موعد للدفع."
        payload = {"message_goals": ["answer_pending_question", "decline_disclosure"],
                   "guidance_question": None, "facts": []}
        result = validate_interview_facts_diagnostic(
            text, json.dumps(payload, ensure_ascii=False), pending_slot="deadline")
        self.assertTrue(result["accepted"])
        plan = result["message_plan"]
        self.assertEqual(plan["goals"], ["answer_pending_question"])
        self.assertEqual(plan["unsupported_goals"], ["decline_disclosure"])
        self.assertEqual(result["unsupported_message_goals"], ["decline_disclosure"])

    def test_supported_decline_goal_requires_matching_current_message(self):
        text = "ما بدي أذكر الموعد."
        payload = {"message_goals": ["answer_pending_question", "decline_disclosure"],
                   "guidance_question": None, "facts": []}
        result = validate_interview_facts_diagnostic(
            text, json.dumps(payload, ensure_ascii=False), pending_slot="deadline")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["message_plan"]["goals"], payload["message_goals"])
        self.assertEqual(result["unsupported_message_goals"], [])

    def test_denied_reported_deadline_is_rejected_when_quote_states_positive_deadline(self):
        text = "حدد الموعد بكرا."
        payload = {"message_goals": ["answer_pending_question"], "guidance_question": None,
                   "facts": [{"field": "deadline", "status": "denied", "value": None,
                              "quote": text}]}
        result = validate_interview_facts_diagnostic(
            text, json.dumps(payload, ensure_ascii=False), pending_slot="deadline")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["filtered_facts"][0]["filter_code"], "unsupported_status_evidence")

    def test_unknown_deadline_cannot_be_reported_as_denial(self):
        text = "ما بعرف إذا حدد مهلة للدفع."
        payload = {"message_goals": ["answer_pending_question", "unknown"], "guidance_question": None,
                   "facts": [{"field": "deadline", "status": "denied", "value": None,
                              "quote": text}]}
        result = validate_interview_facts_diagnostic(
            text, json.dumps(payload, ensure_ascii=False), pending_slot="deadline")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["filtered_facts"][0]["filter_code"], "unsupported_status_evidence")

    def test_denied_text_uses_null_and_denied_binary_uses_false(self):
        cases = [
            ("ما حدد مهلة", {"field": "deadline", "status": "denied", "value": None,
                             "quote": "ما حدد مهلة"}, True),
            ("ما في تهديد", {"field": "threat", "status": "denied", "value": False,
                             "quote": "ما في تهديد"}, True),
            ("ما حدد مهلة", {"field": "deadline", "status": "denied", "value": False,
                             "quote": "ما حدد مهلة"}, False),
            ("ما في تهديد", {"field": "threat", "status": "denied", "value": None,
                             "quote": "ما في تهديد"}, False),
        ]
        for text, fact, should_be_accepted in cases:
            with self.subTest(fact=fact):
                payload = {"message_goals": [], "guidance_question": None, "facts": [fact]}
                result = validate_interview_facts_diagnostic(
                    text, json.dumps(payload, ensure_ascii=False), pending_slot=fact["field"])
                self.assertTrue(result["accepted"])
                self.assertEqual(bool(result["facts"]), should_be_accepted)
                if not should_be_accepted:
                    self.assertEqual(result["filtered_facts"][0]["filter_code"], "value_inconsistent")


if __name__ == "__main__":
    unittest.main()
