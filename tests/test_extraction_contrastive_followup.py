import json
import unittest
from pathlib import Path

from app.model import validate_interview_facts_diagnostic


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "tests" / "extraction_contrastive_followup_v1.json"


class ContrastiveFollowupFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))

    def test_followup_set_is_separate_predeclared_and_outside_prompt_examples(self):
        original = json.loads((ROOT / "tests" / "extraction_contrastive_v1.json").read_text(encoding="utf-8"))
        self.assertTrue(self.suite["synthetic"])
        self.assertEqual(self.suite["edition"], "1.0")
        self.assertNotEqual(self.suite["suite_id"], original["suite_id"])
        self.assertEqual(len(self.suite["cases"]), 14)
        self.assertEqual(len({case["id"] for case in self.suite["cases"]}), 14)

    def test_followup_expected_facts_are_accepted_by_real_validator(self):
        for case in self.suite["cases"]:
            with self.subTest(case=case["id"]):
                payload = {"message_goals": case["expected_goals"], "guidance_question": None,
                           "facts": case["expected_facts"]}
                result = validate_interview_facts_diagnostic(
                    case["message"], json.dumps(payload, ensure_ascii=False),
                    pending_slot=case["pending_slot"], bounded_context=case.get("bounded_context"))
                self.assertTrue(result["accepted"], result)
                self.assertEqual(result["message_plan"]["goals"], case["expected_goals"])
                self.assertEqual(result["facts"], case["expected_facts"])
                self.assertEqual(result["filtered_facts"], [])

    def test_question_without_reported_fact_remains_valid_empty(self):
        case = next(item for item in self.suite["cases"] if item["id"] == "F11-deadline-question-no-fact")
        payload = {"message_goals": case["expected_goals"], "guidance_question": None, "facts": []}
        result = validate_interview_facts_diagnostic(case["message"], json.dumps(payload, ensure_ascii=False),
                                                     case["pending_slot"])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["facts"], [])

    def test_unknown_memory_and_decline_are_not_interchangeable(self):
        cases = [
            ("F02-deadline-unknown-variant", "not_remembered"),
            ("F03-deadline-not-remembered-variant", "unknown"),
            ("F04-deadline-decline-variant", "denied"),
        ]
        by_id = {case["id"]: case for case in self.suite["cases"]}
        for case_id, wrong_status in cases:
            case = by_id[case_id]
            wrong_fact = dict(case["expected_facts"][0], status=wrong_status)
            result = validate_interview_facts_diagnostic(
                case["message"], json.dumps({"facts": [wrong_fact]}, ensure_ascii=False),
                case["pending_slot"], case.get("bounded_context"))
            with self.subTest(case=case_id, status=wrong_status):
                self.assertTrue(result["accepted"])
                self.assertEqual(result["facts"], [])
                self.assertEqual(result["filtered_facts"][0]["filter_code"], "unsupported_status_evidence")


if __name__ == "__main__":
    unittest.main()
