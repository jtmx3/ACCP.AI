import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_conversation_v3


class ConversationV3EvaluatorTests(unittest.TestCase):
    def test_model_preflight_failure_records_not_run_without_generation_or_case_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "not-run.json"
            argv = ["evaluate_conversation_v3", "--mode", "model", "--output", str(output)]
            with patch.object(sys, "argv", argv), \
                    patch.object(evaluate_conversation_v3, "probe_provider", return_value={
                        "configured": True, "service_reachable": False, "model_available": False,
                        "failure_code": "unreachable", "provider": "ollama", "model": "fixture:latest",
                    }), \
                    patch.object(evaluate_conversation_v3, "test_provider_generation") as generation, \
                    patch.object(evaluate_conversation_v3, "run_suite") as run_suite:
                result = evaluate_conversation_v3.main()

            self.assertEqual(result, 2)
            generation.assert_not_called()
            run_suite.assert_not_called()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["evaluation_version"], "conversation-v3-2")
            self.assertEqual(report["model_run_status"], "not_run")
            self.assertEqual(report["provider_attempts"], 0)
            self.assertEqual(report["results"], [])
            self.assertEqual(report["reason"], "unreachable")

    def test_summary_counts_all_22_integrated_fallbacks_and_separates_latencies(self):
        with tempfile.TemporaryDirectory(dir=evaluate_conversation_v3.ROOT) as tmp:
            root = Path(tmp)
            suite_path = root / "synthetic-22.json"
            suite_path.write_text(json.dumps({"suite_id": "synthetic-metric-regression", "version": "1",
                "synthetic_internal_cases": True, "expectations_locked_before_evaluation": True,
                "cases": [{"case_id": "metric-case", "turns": [
                    {"text": f"تفصيل إضافي رقم {index}", "expect": {}} for index in range(22)
                ]}]}, ensure_ascii=False), encoding="utf-8")
            provider = {"provider": "openai-compatible", "model": "synthetic-test", "mode": "external_model"}
            trace = {"attempted": True, "status": "valid_empty", "response_valid": True,
                     "result_accepted": False, "extraction_outcome": "valid_empty", "failure_code": "empty_extraction",
                     "latency_ms": 2.0}
            with patch.object(evaluate_conversation_v3, "model_configured", return_value=True), \
                 patch.object(evaluate_conversation_v3, "provider_config", return_value=provider), \
                 patch.object(evaluate_conversation_v3, "generate_interview_extraction", return_value=[]), \
                 patch.object(evaluate_conversation_v3, "last_call_trace", return_value=trace), \
                 patch("app.main.model_configured", return_value=True), \
                 patch("app.main.provider_config", return_value=provider), \
                 patch("app.main.generate_interview_extraction", return_value=[]), \
                 patch("app.main.last_call_trace", return_value=trace):
                report = evaluate_conversation_v3.run_suite(suite_path, mode="model")

        self.assertEqual(report["evaluation_version"], "conversation-v3-2")
        self.assertEqual(report["model_eligible_turns"], 22)
        self.assertEqual(report["provider_attempts"], 22)
        self.assertEqual(report["provider_fallbacks"], 22)
        self.assertEqual(report["provider_fallback_rate"], 1.0)
        self.assertEqual(report["provider_fallback_categories"], {"empty_extraction": 22})
        self.assertEqual(report["provider_extraction_outcomes"], {"valid_empty": 22})
        self.assertEqual(report["provider_accepted_responses"], 22)
        self.assertEqual(report["provider_extractions_with_accepted_facts"], 0)
        self.assertEqual(report["provider_accepted_fact_count"], 0)
        self.assertIsNotNone(report["latency"]["provider_latency_p50_ms"])
        self.assertIsNotNone(report["latency"]["integration_elapsed_p50_ms"])
        self.assertIsNotNone(report["latency"]["turn_total_elapsed_p50_ms"])
        self.assertGreaterEqual(report["latency"]["turn_total_elapsed_p50_ms"],
                                report["latency"]["integration_elapsed_p50_ms"])


if __name__ == "__main__":
    unittest.main()
