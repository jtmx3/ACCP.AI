import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.diagnose_ollama_extraction import diagnose, load_bounded_context, save_result


OLLAMA_CONFIG = {"provider": "ollama", "local": True, "configured": True,
                 "base_url_host": "127.0.0.1", "model": "fixture:latest",
                 "ollama_reasoning_effort": "none"}


class DiagnoseOllamaExtractionTests(unittest.TestCase):
    def _run_diagnostic(self, *, raw_path=None, facts=None):
        raw = '{"facts":[],"private_echo":"synthetic-only"}'
        diagnostic = {"accepted": True, "raw_response": raw,
                      "filtered_facts": [{"field": "deadline", "status": "reported", "value": "غدًا",
                                          "quote": "تسريب لا يظهر افتراضيًا", "filter_code": "quote_mismatch",
                                          "filter_detail": "The quote does not match the current message."}]}
        with patch("scripts.diagnose_ollama_extraction.provider_config", return_value=OLLAMA_CONFIG), \
             patch("scripts.diagnose_ollama_extraction.probe_provider", return_value={
                 "service_reachable": True, "model_available": True, "failure_code": None, "elapsed_ms": 2.5}), \
             patch("scripts.diagnose_ollama_extraction.generate_interview_extraction", return_value=facts or []), \
             patch("scripts.diagnose_ollama_extraction.last_call_trace", return_value={
                 "attempted": True, "status": "all_facts_filtered", "failure_code": "all_facts_filtered",
                 "latency_ms": 1450.25, "extraction_outcome": "all_facts_filtered"}), \
             patch("scripts.diagnose_ollama_extraction.last_extraction_diagnostic", return_value=diagnostic):
            return diagnose("رسالة اصطناعية للاختبار", "deadline", save_raw_response=raw_path)

    def test_default_diagnostic_is_safe_and_does_not_save_raw_response(self):
        result = self._run_diagnostic()
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertTrue(result["provider_response_received"])
        self.assertTrue(result["response_valid"])
        self.assertEqual(result["extraction_outcome"], "all_facts_filtered")
        self.assertEqual(result["filtered_facts"][0]["field"], "deadline")
        self.assertEqual(result["filtered_facts"][0]["filter_code"], "quote_mismatch")
        self.assertNotIn("synthetic-only", serialized)
        self.assertNotIn("تسريب لا يظهر", serialized)
        self.assertFalse(result["raw_response_saved"])
        self.assertFalse(result["merge_performed"])
        self.assertFalse(result["fallback_performed"])

    def test_explicit_raw_save_creates_new_local_file_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw-response.txt"
            result = self._run_diagnostic(raw_path=path)
            self.assertTrue(result["raw_response_saved"])
            self.assertEqual(path.read_text(encoding="utf-8"), '{"facts":[],"private_echo":"synthetic-only"}')
            with self.assertRaises(FileExistsError):
                self._run_diagnostic(raw_path=path)

    def test_refuses_non_loopback_ollama_endpoint_before_network(self):
        remote = {**OLLAMA_CONFIG, "base_url_host": "example.invalid"}
        with patch("scripts.diagnose_ollama_extraction.provider_config", return_value=remote), \
             patch("scripts.diagnose_ollama_extraction.probe_provider") as probe:
            with self.assertRaisesRegex(RuntimeError, "loopback"):
                diagnose("synthetic message")
            probe.assert_not_called()

    def test_context_file_must_be_bounded_and_must_not_include_secrets(self):
        valid = {"context_version": "accp-limited-context-v2", "untrusted_user_data": True,
                 "pending_question": {"slot": "deadline", "question_ref": "m4"},
                 "recent_relevant_messages": [], "related_facts": [], "related_money_events": [],
                 "context_char_limit": 4500}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "context.json"
            path.write_text(json.dumps(valid, ensure_ascii=False), encoding="utf-8")
            result = load_bounded_context(path)
            self.assertLessEqual(result["serialized_char_count"], 4500)
            invalid = {**valid, "session_token": "should-not-be-here"}
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "مفتاحًا محظورًا"):
                load_bounded_context(path)

    def test_result_json_is_written_to_a_new_file_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            result = {"provider": "ollama", "generation_attempted": False, "extraction_outcome": "not_run"}
            saved = save_result(result, path)
            self.assertEqual(saved, path.resolve())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["extraction_outcome"], "not_run")
            with self.assertRaises(FileExistsError):
                save_result({"generation_attempted": False}, path)


if __name__ == "__main__":
    unittest.main()
