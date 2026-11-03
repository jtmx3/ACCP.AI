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

    def test_contact_must_be_an_identifier_or_an_explicitly_available_number(self):
        generic_text = "التهديد وصل برسالة خاصة."
        generic = {"facts": [{"field": "contact", "status": "reported", "value": "رسالة خاصة",
                              "quote": generic_text}]}
        self.assertIsNone(validate_interview_facts(generic_text, json.dumps(generic, ensure_ascii=False)))

        available_text = "ما بعرف مين هو، بس عندي رقم واتساب إله وما بدي أكتبه هون."
        available = {"facts": [{"field": "contact", "status": "reported",
                                 "value": "رقم واتساب متوفر بحسب الإفادة؛ المعرّف لم يُدرج",
                                 "quote": available_text}]}
        self.assertIsNotNone(validate_interview_facts(available_text, json.dumps(available, ensure_ascii=False)))

        literal_text = "رقم واتساب 0791234567"
        literal = {"facts": [{"field": "contact", "status": "reported", "value": "0791234567",
                               "quote": literal_text}]}
        self.assertIsNotNone(validate_interview_facts(literal_text, json.dumps(literal, ensure_ascii=False)))

    def test_repeated_threat_requires_harm_context_not_a_repeated_payment_request(self):
        payment_text = "بعدها رجع طلب مني 50 دينار."
        payment = {"facts": [{"field": "repeated", "status": "reported", "value": True,
                              "quote": payment_text}]}
        self.assertIsNone(validate_interview_facts(payment_text, json.dumps(payment, ensure_ascii=False)))

        threat_text = "رجع هددني مرة ثانية عبر رسالة."
        threat = {"facts": [{"field": "repeated", "status": "reported", "value": True,
                             "quote": threat_text}]}
        self.assertIsNotNone(validate_interview_facts(threat_text, json.dumps(threat, ensure_ascii=False)))

        denial_text = "ما تكرر بعد الرسالة الأولى."
        denial = {"facts": [{"field": "repeated", "status": "denied", "value": False,
                             "quote": denial_text}]}
        self.assertIsNotNone(validate_interview_facts(denial_text, json.dumps(denial, ensure_ascii=False)))

    def test_narrative_value_must_match_its_verbatim_quote(self):
        text = "في شخص هددني برسالة."
        paraphrase = {"facts": [{"field": "narrative", "status": "reported",
                                 "value": "تعرضت لتهديد برسالة", "quote": text}]}
        self.assertIsNone(validate_interview_facts(text, json.dumps(paraphrase, ensure_ascii=False)))
        verbatim = {"facts": [{"field": "narrative", "status": "reported", "value": text, "quote": text}]}
        self.assertIsNotNone(validate_interview_facts(text, json.dumps(verbatim, ensure_ascii=False)))

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
