import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.model import InterviewUnderstanding
from scripts import evaluate_extraction_contrastive as evaluator


class ContrastiveEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.case = evaluator._load_suite(evaluator.DEFAULT_SUITE)["cases"][0]

    def test_score_separates_unsupported_goal_omission_and_filtered_fact(self):
        accepted = []
        filtered = [{"field": "deadline", "status": "reported", "value": "غدًا",
                     "filter_code": "unsupported_status_evidence",
                     "filter_detail": "حالة غير مدعومة في الاقتباس."}]
        plan = {"goals": ["answer_pending_question"], "unsupported_goals": ["decline_disclosure"]}
        diagnostic = {"accepted": True, "raw_response": "synthetic only"}
        trace = {"attempted": True, "extraction_outcome": "all_facts_filtered", "latency_ms": 123}
        result = evaluator.score_case(self.case, accepted, filtered, plan, diagnostic, trace, 130)
        self.assertEqual(result["goals"]["false_positive"], ["decline_disclosure"])
        self.assertEqual(result["facts"]["filtered_expected_fields"], ["deadline"])
        self.assertEqual(result["facts"]["omitted_fields"], [])
        self.assertEqual(result["outcome"], "all_facts_filtered")
        self.assertNotIn("raw_response", json.dumps(result))

    def test_valid_empty_response_counts_as_omission_not_filtering(self):
        result = evaluator.score_case(
            self.case, [], [], {"goals": [], "unsupported_goals": []},
            {"accepted": True, "raw_response": "synthetic"},
            {"attempted": True, "extraction_outcome": "valid_empty", "latency_ms": 42}, 45)
        self.assertEqual(result["outcome"], "valid_empty")
        self.assertEqual(result["facts"]["omitted_fields"], ["deadline"])
        self.assertEqual(result["facts"]["filtered_expected_fields"], [])
        self.assertEqual(result["facts"]["filtered_count"], 0)

    def test_empty_provider_content_is_still_a_received_provider_response(self):
        result = evaluator.score_case(
            self.case, [], [], {}, {"accepted": False, "raw_response": None},
            {"attempted": True, "status": "empty_response", "failure_code": "empty_response",
             "latency_ms": 12}, 14)
        self.assertTrue(result["provider_attempted"])
        self.assertTrue(result["provider_response_received"])
        self.assertFalse(result["response_valid"])

    def test_rejected_response_is_not_scored_as_goal_miss_or_fact_omission(self):
        result = evaluator.score_case(
            self.case, [], [], {}, {"accepted": False, "rejection_code": "invalid_json"},
            {"attempted": True, "status": "response_rejected", "latency_ms": 15}, 18)
        self.assertFalse(result["goals"]["scored"])
        self.assertIsNone(result["goals"]["model_output"])
        self.assertIsNone(result["goals"]["false_negative"])
        self.assertEqual(result["facts"]["omitted_fields"], [])
        self.assertEqual(result["facts"]["field_results"][0]["disposition"], "response_rejected")

    def test_correct_goal_and_fact_are_scored_as_exact_matches(self):
        expected = {"id": "synthetic-correct", "pending_slot": "deadline",
                    "message": "حدد الموعد بكرا.", "expected_goals": ["answer_pending_question"],
                    "expected_facts": [{"field": "deadline", "status": "reported",
                                         "value": "بكرا", "quote": "حدد الموعد بكرا"}]}
        result = evaluator.score_case(
            expected, [{"field": "deadline", "status": "reported", "value": "بكرا",
                        "quote": "حدد الموعد بكرا"}], [],
            {"goals": ["answer_pending_question"], "unsupported_goals": []},
            {"accepted": True, "raw_response": "synthetic"},
            {"attempted": True, "extraction_outcome": "extracted", "latency_ms": 11}, 14)
        self.assertTrue(result["goals"]["exact_match"])
        self.assertEqual(result["facts"]["field_results"][0]["disposition"], "accepted_match")
        self.assertEqual(result["facts"]["omitted_fields"], [])

    def test_quality_gate_requires_complete_perfect_coverage(self):
        complete = {"attempted_cases": 17, "provider_responses_valid": 17,
                    "goal_scored_cases": 17, "goal_exact_case_matches": 17,
                    "goal_false_positive": 0, "goal_false_negative": 0,
                    "unscored_expected_fact_count": 0,
                    "fact_field_status_value_end_to_end_coverage": 1.0,
                    "accepted_fact_exact_match_precision": 1.0,
                    "unexpected_accepted_field_count": 0}
        self.assertTrue(evaluator.evaluation_passes(complete, 17))
        incomplete = dict(complete, goal_unscored_cases=1, unscored_expected_fact_count=1)
        self.assertFalse(evaluator.evaluation_passes(incomplete, 17))

    def test_suite_runner_uses_project_extraction_path_and_is_not_integrated_evaluation(self):
        plan = {"goals": ["answer_pending_question"], "unsupported_goals": []}
        diagnostic = {"accepted": True, "raw_response": '{"facts":[]}', "filtered_facts": [],
                      "message_plan": plan}
        trace = {"attempted": True, "status": "valid_empty", "response_valid": True,
                 "extraction_outcome": "valid_empty", "latency_ms": 10}
        empty = InterviewUnderstanding([], plan, diagnostic)
        config = {"provider": "ollama", "local": True, "configured": True,
                  "base_url_host": "127.0.0.1", "model": "synthetic-test", "ollama_reasoning_effort": "none"}
        with patch.object(evaluator, "provider_config", return_value=config), \
             patch.object(evaluator, "probe_provider", return_value={"service_reachable": True,
                 "model_available": True, "elapsed_ms": 2}), \
             patch.object(evaluator, "generate_interview_extraction", return_value=empty) as generate, \
             patch.object(evaluator, "last_extraction_diagnostic", return_value=diagnostic), \
             patch.object(evaluator, "last_call_trace", return_value=trace):
            result = evaluator.evaluate_suite(evaluator.DEFAULT_SUITE)
        self.assertEqual(generate.call_count, result["case_count"])
        self.assertEqual(result["evaluation_type"], "model_extraction_only_contrastive")
        self.assertFalse(result["integrated_system_evaluation"])
        self.assertFalse(result["fallback_performed"])
        self.assertFalse(result["raw_provider_responses_saved"])
        self.assertEqual(result["summary"]["valid_empty_cases"], result["case_count"])
        self.assertEqual(result["summary"]["omitted_expected_fact_count"], result["summary"]["expected_fact_count"])

    def test_result_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            evaluator.save_result({"synthetic": True}, path)
            with self.assertRaises(FileExistsError):
                evaluator.save_result({"synthetic": False}, path)
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["synthetic"])


if __name__ == "__main__":
    unittest.main()
