import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.model import ALLOWED_EXTRACTION_FIELDS
from scripts.evaluate_scenarios import (
    _candidate_matches,
    _candidate_oracle_class,
    _extraction_case_passed,
    _legacy_composite_case_passed,
    _response_fact_proposals,
    _safe_response_diagnostic,
    run,
)


class ModelEvaluationPolicyTests(unittest.TestCase):
    def test_direct_danger_is_engine_owned_not_an_extraction_expectation(self):
        self.assertNotIn("direct_danger", ALLOWED_EXTRACTION_FIELDS)
        # The model extraction metric is independent of rule-engine safety output.
        self.assertTrue(_extraction_case_passed({"threat": True}, False))
        self.assertFalse(_extraction_case_passed({"threat": True, "repeated": False}, False))

    def test_extraction_pass_does_not_require_integrated_pass_or_zero_fallback(self):
        self.assertTrue(_extraction_case_passed({"threat": True}, False))
        self.assertFalse(_extraction_case_passed({"threat": True}, True))
        self.assertIsNone(_extraction_case_passed({}, False))

    def test_legacy_composite_metric_is_retained_with_its_old_definition(self):
        calls = [{"result_accepted": True, "fallback_used": False}]
        self.assertTrue(_legacy_composite_case_passed(True, {"threat": True}, calls))
        # This fixed-v2 comparator necessarily fails a safety-engine-owned direct_danger expectation.
        self.assertFalse(_legacy_composite_case_passed(
            True, {"threat": True, "direct_danger": False}, calls))
        self.assertFalse(_legacy_composite_case_passed(
            True, {"threat": True}, [{"result_accepted": False, "fallback_used": True}]))
        self.assertFalse(_legacy_composite_case_passed(True, {"threat": True}, []))
        self.assertFalse(_legacy_composite_case_passed(
            True, {"threat": True}, [{"result_accepted": True, "fallback_used": False,
                                      "filtered_fact_count": 1}]))

    def test_candidate_values_and_negation_must_match_fixed_oracle(self):
        expected = {"paid_amount": {"status": "denied", "value": None}}
        self.assertTrue(_candidate_matches({"status": "denied", "value": None}, expected["paid_amount"]))
        self.assertFalse(_candidate_matches({"status": "reported", "value": "50"}, expected["paid_amount"]))

    def test_unlisted_candidate_is_unscored_not_assumed_false_positive(self):
        self.assertEqual(_candidate_oracle_class(
            {"field": "platform", "status": "reported", "value": "واتساب"},
            {"threat": {"status": "reported", "value": True}}, set()), "oracle_unspecified")

    def test_only_explicit_forbidden_positive_is_counted_as_such(self):
        candidate = {"field": "published", "status": "reported", "value": True}
        self.assertEqual(_candidate_oracle_class(candidate, {}, {"published"}), "explicit_forbidden_positive")
        self.assertEqual(_candidate_oracle_class(
            {"field": "published", "status": "unknown", "value": None}, {}, {"published"}),
            "oracle_unspecified")

    def test_rejected_synthetic_response_can_be_diagnosed_without_preserving_credentials(self):
        raw = '{"facts":[{"field":"threat","status":"reported","value":true,"quote":"هددني"}],"api_key":"sk-secret"}'
        diagnostic = _safe_response_diagnostic(raw)
        proposals = _response_fact_proposals(raw)
        self.assertNotIn("sk-secret", diagnostic["response_excerpt"])
        self.assertIn("[REDACTED]", diagnostic["response_excerpt"])
        self.assertEqual(proposals[0]["field"], "threat")
        self.assertEqual(proposals[0]["quote"], "هددني")

    def test_model_runner_records_sanitized_rejection_and_keeps_fallback_out_of_extraction_score(self):
        raw = '{"facts":[{"field":"threat","status":"reported","value":true,"quote":"نص غير موجود"}],"api_key":"sk-secret"}'
        dataset = {
            "dataset_id": "synthetic-unit-eval",
            "version": 1,
            "classification": "synthetic; unit-test fixture only",
            "oracle_basis": "fixed unit-test expectation",
            "cases": [{"id": "SYN-1", "category": "test",
                       "steps": [{"action": "message", "text": "شخص هددني برسالة."}],
                       "expected_facts": {"threat": {"status": "reported", "value": True}}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset.json"
            output_path = Path(tmp) / "result.json"
            dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
            with patch("scripts.evaluate_scenarios.provider_config", return_value={
                    "configured": True, "provider": "test", "model": "synthetic-test"}), \
                 patch("scripts.evaluate_scenarios.model_configured", return_value=True), \
                 patch("scripts.evaluate_scenarios.probe_provider", return_value={
                     "service_reachable": True, "model_available": True}), \
                 patch("scripts.evaluate_scenarios.test_provider_generation", return_value={"success": True}), \
                 patch("app.main.model_configured", return_value=True), \
                 patch("app.main.generate_interview_extraction", return_value=None), \
                 patch("scripts.evaluate_scenarios.last_extraction_diagnostic", return_value={
                     "accepted": False, "json_valid": True, "schema_valid": True,
                     "rejection_code": "quote_mismatch", "rejection_detail": "synthetic quote mismatch",
                     "raw_response": raw}):
                result = run(dataset_path, output_path, mode="model")
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            call = saved["cases"][0]["model_evaluation"]["calls"][0]
            self.assertEqual(call["rejection_code"], "quote_mismatch")
            self.assertIn("response_excerpt", call)
            self.assertNotIn("sk-secret", call["response_excerpt"])
            self.assertEqual(call["rejected_fact_proposals"][0]["quote"], "نص غير موجود")
            self.assertTrue(saved["cases"][0]["integrated_passed"])
            self.assertFalse(saved["cases"][0]["extraction_passed"])
            self.assertEqual(result["result"]["fallback_successes_counted_as_extraction_success"], 0)
            self.assertEqual(result["result"]["rejection_code_counts"], {"quote_mismatch": 1})

    def test_model_runner_separates_filtered_facts_from_accepted_extraction_and_legacy_score(self):
        dataset = {
            "dataset_id": "synthetic-filter-unit-eval", "version": 1,
            "classification": "synthetic; unit-test fixture only", "oracle_basis": "fixed expectation",
            "cases": [{"id": "SYN-FILTER", "category": "test",
                       "steps": [{"action": "message", "text": "شخص هددني برسالة."}],
                       "expected_facts": {"threat": {"status": "reported", "value": True}}}],
        }
        filtered = {"field": "identity", "status": "unknown", "value": None,
                    "quote": "شخص هددني برسالة.", "filter_code": "unsupported_status_evidence",
                    "filter_detail": "no explicit identity uncertainty"}
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset.json"
            output_path = Path(tmp) / "result.json"
            dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
            with patch("scripts.evaluate_scenarios.provider_config", return_value={
                    "configured": True, "provider": "test", "model": "synthetic-test"}), \
                 patch("scripts.evaluate_scenarios.model_configured", return_value=True), \
                 patch("scripts.evaluate_scenarios.probe_provider", return_value={
                     "service_reachable": True, "model_available": True}), \
                 patch("scripts.evaluate_scenarios.test_provider_generation", return_value={"success": True}), \
                 patch("app.main.model_configured", return_value=True), \
                 patch("app.main.generate_interview_extraction", return_value=[{
                     "field": "threat", "status": "reported", "value": True, "quote": "هددني"}]), \
                 patch("scripts.evaluate_scenarios.last_extraction_diagnostic", return_value={
                     "accepted": True, "json_valid": True, "schema_valid": True,
                     "facts": [{"field": "threat", "status": "reported", "value": True, "quote": "هددني"}],
                     "filtered_facts": [filtered]}):
                result = run(dataset_path, output_path, mode="model")
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            case = saved["cases"][0]
            call = case["model_evaluation"]["calls"][0]
            self.assertEqual(call["candidate_facts"][0]["field"], "threat")
            self.assertEqual(call["filtered_fact_count"], 1)
            self.assertEqual(call["filtered_fact_proposals"][0]["field"], "identity")
            self.assertTrue(case["extraction_passed"])
            self.assertTrue(case["integrated_passed"])
            self.assertFalse(case["legacy_v2_composite_passed"])
            self.assertEqual(result["result"]["filtered_fact_proposal_count"], 1)
            self.assertEqual(result["result"]["filtered_fact_proposal_code_counts"],
                             {"unsupported_status_evidence": 1})
            self.assertEqual(case["model_evaluation"]["all_response_proposal_oracle_classes"]["oracle_unspecified"], 1)


if __name__ == "__main__":
    unittest.main()
