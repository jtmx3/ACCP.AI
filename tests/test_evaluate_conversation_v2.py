import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_conversation_v2


class ConversationV2EvaluatorTests(unittest.TestCase):
    def test_model_extraction_and_rule_merge_are_reported_separately(self):
        suite = {
            "suite_id": "SYNTHETIC-MOCK-EVALUATOR",
            "authored_before_logic_changes": True,
            "synthetic_internal_cases": True,
            "cases": [{
                "id": "mock-evidence",
                "seed_facts": {},
                "pending_slot": "evidence",
                "turns": [{
                    "text": "عندي لقطات شاشة للرسائل",
                    "expect": {"facts": {"evidence": {"status": "reported"}},
                               "absent": ["direct_danger"]},
                }],
            }],
        }
        candidate = [{"field": "evidence", "status": "reported", "value": "لقطات الشاشة متوفرة",
                      "quote": "عندي لقطات شاشة للرسائل"}]
        trace = {"attempted": True, "status": "extracted", "provider": "synthetic",
                 "model": "synthetic-fixture", "latency_ms": 12.5, "response_valid": True,
                 "extraction_outcome": "extracted", "result_accepted": True}
        diagnostic = {"accepted": True, "filtered_facts": [], "filtered_fact_code_counts": {}}
        provider = {"provider": "synthetic", "model": "synthetic-fixture", "configuration_complete": True,
                    "mode": "external_model"}

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp_dir:
            path = Path(temp_dir) / "suite.json"
            path.write_text(json.dumps(suite, ensure_ascii=False), encoding="utf-8")
            with patch.object(evaluate_conversation_v2, "reset_call_trace"), \
                 patch.object(evaluate_conversation_v2, "generate_interview_extraction", return_value=candidate), \
                 patch.object(evaluate_conversation_v2, "last_call_trace", return_value=trace), \
                 patch.object(evaluate_conversation_v2, "last_extraction_diagnostic", return_value=diagnostic), \
                 patch.object(evaluate_conversation_v2, "build_bounded_context",
                              wraps=evaluate_conversation_v2.build_bounded_context) as context_builder:
                model_result = evaluate_conversation_v2.run_suite(path, "model", provider)
                context_builder.assert_not_called()
                evaluate_conversation_v2.run_suite(path, "model", provider, include_bounded_context=True)
                context_builder.assert_called_once()

            rules_result = evaluate_conversation_v2.run_suite(
                path, "rules", {"provider": "rules", "model": None, "configuration_complete": False})

        self.assertEqual(model_result["model_extraction_eligible_turns"], 1)
        self.assertEqual(model_result["provider_calls_attempted"], 1)
        self.assertEqual(model_result["model_expected_fact_matches"], 1)
        self.assertEqual(model_result["model_expected_fact_omissions"], 0)
        self.assertEqual(model_result["model_forbidden_additions"], 0)
        self.assertEqual(model_result["integrated_api_fallbacks_not_counted_as_model_success"], 0)
        self.assertGreaterEqual(model_result["integrated_model_proposals_applied_as_conflicts"], 1)
        self.assertEqual(model_result["integrated_model_proposals_superseded_by_rules"], 0)
        self.assertEqual(rules_result["mode"], "rules")
        self.assertEqual(rules_result["model_extraction_eligible_turns"], 0)
        self.assertEqual(rules_result["provider_calls_attempted"], 0)
        self.assertEqual(rules_result["model_expected_fact_checks"], 0)


if __name__ == "__main__":
    unittest.main()
