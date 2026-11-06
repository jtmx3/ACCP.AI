import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_guidance_live as live


class GuidanceLiveHarnessTests(unittest.TestCase):
    def test_report_creation_never_overwrites_a_previous_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            output.write_text('{"old":true}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                live._write_new_report(output, {"new": True})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"old": True})

    def test_default_output_paths_are_distinct_new_run_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = live._output_path(Path(tmp), None)
            second = live._output_path(Path(tmp), None)
            self.assertNotEqual(first, second)
            self.assertTrue(first.name.startswith("guidance-live-"))
            self.assertEqual(first.suffix, ".json")

    def test_generic_security_case_requires_model_plan_and_no_citation(self):
        scenario = live.SCENARIOS[0]
        result = {
            "session": {"messages": [{"role": "assistant", "kind": "answer",
                "text": "أي تطبيق تقصد؟", "citations": []}]},
            "message_plan_applied": True,
            "message_plan": {"goals": ["ask_guidance"]},
            "external_context_sent": True,
            "provider_call": {"context_fact_count": 0},
        }
        observed = live._evaluate_result(scenario, result, None)
        self.assertTrue(observed["passed"], observed["checks"])

    def test_whatsapp_case_rejects_instagram_only_citation(self):
        scenario = live.SCENARIOS[1]
        result = {"session": {"messages": [{"role": "assistant", "kind": "answer",
            "text": "خطوات أمان الحساب.", "citations": [{"id": "GLOBAL-IG-SECURITY-2026"}]}]}}
        observed = live._evaluate_result(scenario, result, None)
        self.assertFalse(observed["checks"]["has_whatsapp_source"])
        self.assertFalse(observed["checks"]["excludes_instagram_source"])
        self.assertFalse(observed["passed"])

    def test_prior_platform_case_requires_explicitly_sent_seeded_context(self):
        scenario = live.SCENARIOS[2]
        result = {"session": {"messages": [{"role": "assistant", "kind": "answer",
            "text": "فعّل التحقق بخطوتين.", "citations": [{"id": "GLOBAL-WA-ACCOUNT-SECURITY-2026"}]}]},
            "message_plan_applied": True,
            "message_plan": {"goals": ["ask_guidance"]},
            "external_context_sent": False,
            "provider_call": {"context_fact_count": 0}}
        observed = live._evaluate_result(scenario, result, True)
        self.assertTrue(observed["checks"]["has_whatsapp_source"])
        self.assertFalse(observed["checks"]["bounded_context_sent"])
        self.assertFalse(observed["checks"]["bounded_context_has_seeded_fact"])
        self.assertFalse(observed["passed"])

    def test_api_provider_trace_proves_context_sent_without_equating_it_to_answer_acceptance(self):
        scenario = live.SCENARIOS[2]
        result = {"session": {"messages": [{"role": "assistant", "kind": "answer",
            "text": "راجع قائمة الأجهزة المرتبطة.", "citations": [{"id": "GLOBAL-WA-ACCOUNT-SECURITY-2026"}]}]},
            "message_plan_applied": True,
            "message_plan": {"goals": ["ask_guidance"]},
            "external_model_result_accepted": False,
            "provider_call": {"context_sent": True, "context_fact_count": 1}}
        observed = live._evaluate_result(scenario, result, True)
        self.assertTrue(observed["checks"]["bounded_context_sent"])
        self.assertTrue(observed["checks"]["bounded_context_has_seeded_fact"])
        self.assertTrue(observed["checks"]["model_plan_applied"])
        self.assertFalse(observed["external_model_result_accepted"])
        self.assertTrue(observed["passed"])

    def test_cli_stops_before_network_when_expected_ollama_model_is_not_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "not-run.json"
            with patch.object(sys, "argv", ["evaluate_guidance_live", "--include-bounded-context",
                                             "--output", str(output)]), \
                 patch.object(live, "provider_config", return_value={"provider": "rules", "mode": "rules",
                                                                       "model": None, "configuration_complete": False}), \
                 patch.object(live, "probe_provider") as probe, \
                 patch.object(live, "test_provider_generation") as generation, \
                 patch.object(live, "run_live_scenarios") as run:
                code = live.main_cli()
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(code, 2)
        self.assertEqual(report["test_status"], "not_run")
        self.assertEqual(report["reason"], "expected_local_ollama_model_not_configured")
        probe.assert_not_called()
        generation.assert_not_called()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
