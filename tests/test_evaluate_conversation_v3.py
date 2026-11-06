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
            self.assertEqual(report["model_run_status"], "not_run")
            self.assertEqual(report["provider_attempts"], 0)
            self.assertEqual(report["results"], [])
            self.assertEqual(report["reason"], "unreachable")


if __name__ == "__main__":
    unittest.main()
