import json
import os
import unittest
from unittest.mock import patch

from app.model import (
    generate_grounded_answer,
    generate_interview_extraction,
    validate_extractive_claims,
    validate_interview_facts,
)


class OptionalModelTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{
            "id": "AU-TEST",
            "title": "Evidence guide",
            "publisher": "Example publisher",
            "geography": "AU",
            "text": "Preserve relevant records.",
            "excerpt": "Preserve relevant records.",
            "url": "https://example.test/evidence",
        }]
        self.env = patch.dict(os.environ, {
            "MODEL_PROVIDER": "openai-compatible",
            "MODEL_API_KEY": "test-only-not-a-secret",
            "MODEL_BASE_URL": "https://provider.example/v1",
            "MODEL_NAME": "test-model",
            "MODEL_TIMEOUT_SECONDS": "1",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_provider_timeout_returns_none_for_local_fallback(self):
        with patch("app.model.urllib.request.urlopen", side_effect=TimeoutError("synthetic timeout")):
            self.assertIsNone(generate_grounded_answer("كيف أحفظ الأدلة؟", self.sources))

    def test_grounded_answer_shows_only_an_exact_source_excerpt(self):
        payload = {"claims": [{"source_id": "AU-TEST", "quote": "Preserve relevant records."}]}
        with patch("app.model._chat", return_value=json.dumps(payload)):
            answer = generate_grounded_answer("How do I preserve records?", self.sources)
        self.assertEqual(answer, {"text": "«Preserve relevant records.» [1]", "source_ids": ["AU-TEST"]})

    def test_hallucinated_free_text_is_rejected_even_with_a_valid_source_number(self):
        self.assertIsNone(validate_extractive_claims("The police guarantee a refund [1].", self.sources))
        payload = {"claims": [{"source_id": "AU-TEST", "quote": "The police guarantee a refund."}]}
        self.assertIsNone(validate_extractive_claims(json.dumps(payload), self.sources))

    def test_interview_output_requires_schema_exact_quote_and_field_consistency(self):
        text = "طلب مني ١٠٠ دينار وأنا ما دفعت"
        valid = {"facts": [
            {"field": "requested_amount", "status": "reported", "value": "100 دينار", "quote": "طلب مني ١٠٠ دينار"},
            {"field": "paid_amount", "status": "denied", "value": None, "quote": "ما دفعت"},
        ]}
        facts = validate_interview_facts(text, json.dumps(valid, ensure_ascii=False))
        self.assertEqual([item["field"] for item in facts], ["requested_amount", "paid_amount"])
        bad_quote = {"facts": [dict(valid["facts"][0], quote="طلب مني ٢٠٠ دينار")]}
        self.assertIsNone(validate_interview_facts(text, json.dumps(bad_quote, ensure_ascii=False)))
        bad_amount = {"facts": [dict(valid["facts"][0], value="300 دينار")]}
        self.assertIsNone(validate_interview_facts(text, json.dumps(bad_amount, ensure_ascii=False)))
        forbidden = {"facts": [{"field": "direct_danger", "status": "reported", "value": True, "quote": text}]}
        self.assertIsNone(validate_interview_facts(text, json.dumps(forbidden, ensure_ascii=False)))

    def test_extraction_request_contains_only_current_message_and_pending_slot(self):
        text = "شخص يهددني برسالة"
        candidate = {"facts": [{"field": "threat", "status": "reported", "value": True, "quote": text}]}
        with patch("app.model._chat", return_value=json.dumps(candidate, ensure_ascii=False)) as chat:
            result = generate_interview_extraction(text, "narrative")
        self.assertEqual(result[0]["field"], "threat")
        sent = json.loads(chat.call_args.args[0][1]["content"])
        self.assertEqual(set(sent), {"pending_question_slot", "current_user_message", "allowed_fact_fields"})
        self.assertEqual(sent["current_user_message"], text)
        self.assertEqual(sent["pending_question_slot"], "narrative")
        self.assertNotIn("session_id", str(sent))
        self.assertNotIn("transcript", str(sent))

    def test_malformed_or_mismatched_extraction_falls_back(self):
        text = "طلب مني 100 دينار"
        with patch("app.model._chat", return_value='{"facts":[{"field":"requested_amount","status":"reported","value":"200 دينار","quote":"طلب مني 100 دينار"}]}'):
            self.assertIsNone(generate_interview_extraction(text, "narrative"))
        with patch("app.model._chat", return_value='not JSON'):
            self.assertIsNone(generate_interview_extraction(text, "narrative"))


if __name__ == "__main__":
    unittest.main()
