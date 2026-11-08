import io
import json
import os
import unittest
from unittest.mock import patch

from app.model import (
    generate_grounded_answer,
    generate_interview_extraction,
    last_call_trace,
    last_extraction_diagnostic,
    validate_extractive_claims,
    validate_interview_facts,
    validate_interview_facts_diagnostic,
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
        self.assertEqual(validate_interview_facts(text, json.dumps(bad_quote, ensure_ascii=False)), [])
        bad_amount = {"facts": [dict(valid["facts"][0], value="300 دينار")]}
        self.assertEqual(validate_interview_facts(text, json.dumps(bad_amount, ensure_ascii=False)), [])
        forbidden = {"facts": [{"field": "direct_danger", "status": "reported", "value": True, "quote": text}]}
        self.assertIsNone(validate_interview_facts(text, json.dumps(forbidden, ensure_ascii=False)))

    def test_contact_must_be_an_identifier_or_an_explicitly_available_number(self):
        generic_text = "التهديد وصل برسالة خاصة."
        generic = {"facts": [{"field": "contact", "status": "reported", "value": "رسالة خاصة",
                              "quote": generic_text}]}
        self.assertEqual(validate_interview_facts(generic_text, json.dumps(generic, ensure_ascii=False)), [])

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
        self.assertEqual(validate_interview_facts(payment_text, json.dumps(payment, ensure_ascii=False)), [])

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
        empty_value = {"facts": [{"field": "narrative", "status": "reported", "value": None, "quote": text}]}
        empty_diagnostic = validate_interview_facts_diagnostic(
            text, json.dumps(empty_value, ensure_ascii=False))
        self.assertTrue(empty_diagnostic["accepted"])
        self.assertEqual(empty_diagnostic["facts"], [])
        self.assertEqual(empty_diagnostic["filtered_facts"][0]["filter_code"], "value_inconsistent")
        paraphrase = {"facts": [{"field": "narrative", "status": "reported",
                                 "value": "تعرضت لتهديد برسالة", "quote": text}]}
        self.assertEqual(validate_interview_facts(text, json.dumps(paraphrase, ensure_ascii=False)), [])
        verbatim = {"facts": [{"field": "narrative", "status": "reported", "value": text, "quote": text}]}
        self.assertIsNotNone(validate_interview_facts(text, json.dumps(verbatim, ensure_ascii=False)))

    def test_exact_supported_arabic_variants_are_not_rejected_by_narrow_anchors(self):
        examples = [
            ("قال رح ينشر محادثاتي الخاصة إذا ما حولتله 100 دينار.", "threat", "reported", True,
             "قال رح ينشر محادثاتي الخاصة إذا ما حولتله 100 دينار."),
            ("كانت الرسالة تقول إنه سيلاحقني إذا خرجت من البيت.", "threat", "reported", True,
             "كانت الرسالة تقول إنه سيلاحقني إذا خرجت من البيت."),
            ("رح أضرك إذا حكيت عن الموضوع", "threat", "reported", True, "رح أضرك إذا حكيت عن الموضوع"),
            ("كأنه أنا", "impersonation", "reported", True, "كأنه أنا"),
            ("ما حدا دخل حسابي", "account_access", "denied", False, "ما حدا دخل حسابي"),
            ("كرر نفس التهديد", "repeated", "reported", True, "كرر نفس التهديد"),
            ("إنه ينزل محادثاتي الخاصة", "publication_threat", "reported", True, "إنه ينزل محادثاتي الخاصة"),
            ("قال برسالة إنه رح يرسل محادثاتي الخاصة إذا ما دفعت.", "publication_threat", "reported", True,
             "قال برسالة إنه رح يرسل محادثاتي الخاصة إذا ما دفعت."),
            ("ما بعرف إذا نزّلها فعلًا أو لا.", "published", "unknown", None,
             "ما بعرف إذا نزّلها فعلًا أو لا."),
            ("الحساب حاط اسمي وصورتي", "published", "reported", True, "حاط اسمي وصورتي"),
            ("رسالة ثانية", "contact_repeated", "reported", True, "رسالة ثانية"),
        ]
        for text, field, status, value, quote in examples:
            with self.subTest(field=field, quote=quote):
                payload = {"facts": [{"field": field, "status": status, "value": value, "quote": quote}]}
                self.assertIsNotNone(validate_interview_facts(text, json.dumps(payload, ensure_ascii=False)))

    def test_new_threat_anchors_do_not_override_clear_negation(self):
        for text, quote, field in [
            ("ما في أي تهديد", "ما في أي تهديد", "threat"),
            ("ما حدا دخل حسابي", "ما حدا دخل حسابي", "account_access"),
            ("ما بعرف إذا نشرها أو لا", "ما بعرف إذا نشرها أو لا", "published"),
            ("ما نشرها", "ما نشرها", "published"),
        ]:
            with self.subTest(field=field):
                payload = {"facts": [{"field": field, "status": "reported", "value": True, "quote": quote}]}
                diagnostic = validate_interview_facts_diagnostic(text, json.dumps(payload, ensure_ascii=False))
                self.assertTrue(diagnostic["accepted"])
                self.assertEqual(diagnostic["facts"], [])
                self.assertEqual(diagnostic["filtered_facts"][0]["filter_code"], "value_inconsistent")

    def test_evidence_positive_claim_cannot_quote_explicit_evidence_absence(self):
        text = "ما عندي نسخ ولا لقطات محفوظة من الرسائل."
        payload = {"facts": [{"field": "evidence", "status": "reported", "value": text, "quote": text}]}
        diagnostic = validate_interview_facts_diagnostic(text, json.dumps(payload, ensure_ascii=False))
        self.assertTrue(diagnostic["accepted"])
        self.assertEqual(diagnostic["filtered_facts"][0]["filter_code"], "value_inconsistent")

    def test_identity_means_suspect_identity_not_reporters_name_or_photo(self):
        reporter_quote = "حساب مزيف حاط اسمي وصورتي"
        reporter_identity = {"facts": [{"field": "identity", "status": "reported",
                                         "value": "اسمي وصورتي", "quote": "اسمي وصورتي"}]}
        diagnostic = validate_interview_facts_diagnostic(reporter_quote,
                                                          json.dumps(reporter_identity, ensure_ascii=False))
        self.assertTrue(diagnostic["accepted"])
        self.assertEqual(diagnostic["filtered_facts"][0]["filter_code"], "value_inconsistent")

        suspect_quote = "المشتبه اسمه سامر"
        suspect_identity = {"facts": [{"field": "identity", "status": "reported",
                                        "value": "سامر", "quote": "سامر"}]}
        self.assertIsNotNone(validate_interview_facts(suspect_quote,
                                                      json.dumps(suspect_identity, ensure_ascii=False)))

    def test_unknown_identity_requires_explicit_unknown_text_not_absence_of_information(self):
        no_identity_statement = "كانت الرسالة على واتساب تقول إنه سيلاحقني إذا خرجت من البيت."
        inferred_unknown = {"facts": [
            {"field": "identity", "status": "unknown", "value": None, "quote": no_identity_statement},
            {"field": "platform", "status": "reported", "value": "واتساب", "quote": "واتساب"},
        ]}
        diagnostic = validate_interview_facts_diagnostic(
            no_identity_statement, json.dumps(inferred_unknown, ensure_ascii=False), pending_slot="narrative")
        self.assertTrue(diagnostic["accepted"])
        self.assertEqual([fact["field"] for fact in diagnostic["facts"]], ["platform"])
        self.assertEqual(diagnostic["filtered_facts"][0]["filter_code"], "unsupported_status_evidence")
        self.assertEqual(diagnostic["filtered_facts"][0]["field"], "identity")

        explicit_unknown_text = "ما بعرف مين هو."
        explicit_unknown = {"facts": [{"field": "identity", "status": "unknown", "value": None,
                                       "quote": explicit_unknown_text}]}
        self.assertIsNotNone(validate_interview_facts_diagnostic(
            explicit_unknown_text, json.dumps(explicit_unknown, ensure_ascii=False), pending_slot="identity")["facts"])

    def test_partial_fact_filter_is_visible_in_safe_provider_trace(self):
        text = "كان في حساب مزيف على واتساب"
        facts = {"facts": [
            {"field": "platform", "status": "reported", "value": "واتساب", "quote": "واتساب"},
            {"field": "identity", "status": "unknown", "value": None, "quote": text},
        ]}
        envelope = {"choices": [{"message": {"content": json.dumps(facts, ensure_ascii=False)}}]}
        with patch("app.model.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(envelope).encode("utf-8"))):
            result = generate_interview_extraction(text, "narrative")
        self.assertEqual([fact["field"] for fact in result], ["platform"])
        trace = last_call_trace()
        self.assertTrue(trace["result_accepted"])
        self.assertEqual(trace["status"], "partially_accepted")
        self.assertEqual(trace["filtered_fact_count"], 1)
        self.assertEqual(trace["filtered_fact_fields"], ["identity"])
        self.assertNotIn("filtered_fact_proposals", trace)

    def test_ollama_extraction_uses_project_prompt_and_separates_empty_filtered_and_extracted(self):
        text = "في شخص عم يهددني على واتساب."
        outputs = [
            {"facts": []},
            {"facts": [{"field": "narrative", "status": "reported", "value": None, "quote": text}]},
            {"facts": [{"field": "narrative", "status": "reported", "value": text, "quote": text}]},
        ]
        request_bodies = []

        def fake_urlopen(request, timeout):
            request_bodies.append(json.loads(request.data.decode("utf-8")))
            content = json.dumps(outputs[len(request_bodies) - 1], ensure_ascii=False)
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8"))

        env = {"MODEL_PROVIDER": "ollama", "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
               "MODEL_NAME": "fixture:latest", "MODEL_REASONING_EFFORT": "high"}
        with patch.dict(os.environ, env, clear=True), patch("app.model.urllib.request.urlopen", side_effect=fake_urlopen):
            empty = generate_interview_extraction(text, "narrative")
            empty_trace = last_call_trace()
            filtered = generate_interview_extraction(text, "narrative")
            filtered_trace = last_call_trace()
            extracted = generate_interview_extraction(text, "narrative")
            extracted_trace = last_call_trace()

        self.assertEqual(empty, [])
        self.assertEqual(empty_trace["extraction_outcome"], "valid_empty")
        self.assertTrue(empty_trace["response_valid"])
        self.assertFalse(empty_trace["result_accepted"])
        self.assertEqual(filtered, [])
        self.assertEqual(filtered_trace["extraction_outcome"], "all_facts_filtered")
        self.assertEqual(filtered_trace["filtered_fact_code_counts"], {"value_inconsistent": 1})
        self.assertFalse(filtered_trace["result_accepted"])
        self.assertEqual(extracted_trace["extraction_outcome"], "extracted")
        self.assertTrue(extracted_trace["result_accepted"])
        for body in request_bodies:
            self.assertEqual(body["reasoning_effort"], "none")
            self.assertNotIn("reasoning", body)
            system_prompt = body["messages"][0]["content"]
            self.assertIn("reported الثنائية true", system_prompt)
            self.assertIn("الحقول الثنائية value=false", system_prompt)
            self.assertIn("الحقول النصية/الزمنية/المبالغ value=null", system_prompt)
            self.assertIn("narrative", system_prompt)
            self.assertIn("current_quote", system_prompt)
            self.assertIn("لا تضع operation في الإجابة المباشرة", system_prompt)
            self.assertIn("operation=same أو support أو correct", system_prompt)
            request_text = body["messages"][1]["content"]
            self.assertEqual(json.loads(request_text)["current_user_message"], text)

    def test_ollama_reasoning_effort_is_independent_and_explicitly_configurable(self):
        text = "في شخص عم يهددني على واتساب."
        payload = {"facts": [{"field": "narrative", "status": "reported", "value": text, "quote": text}]}
        bodies = []

        def fake_urlopen(request, timeout):
            bodies.append(json.loads(request.data.decode("utf-8")))
            content = json.dumps(payload, ensure_ascii=False)
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8"))

        env = {"MODEL_PROVIDER": "ollama", "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
               "MODEL_NAME": "fixture:latest", "OLLAMA_REASONING_EFFORT": "low",
               "MODEL_REASONING_EFFORT": "high"}
        with patch.dict(os.environ, env, clear=True), patch("app.model.urllib.request.urlopen", side_effect=fake_urlopen):
            result = generate_interview_extraction(text, "narrative")
        self.assertEqual(result[0]["field"], "narrative")
        self.assertEqual(bodies[0]["reasoning_effort"], "low")
        self.assertNotIn("reasoning", bodies[0])

    def test_empty_provider_content_is_not_confused_with_valid_empty_facts(self):
        envelope = {"choices": [{"message": {"content": ""}}]}
        with patch("app.model.urllib.request.urlopen",
                   return_value=io.BytesIO(json.dumps(envelope).encode("utf-8"))):
            result = generate_interview_extraction("في شخص عم يهددني على واتساب.", "narrative")
        trace = last_call_trace()
        self.assertIsNone(result)
        self.assertEqual(trace["failure_code"], "empty_response")
        self.assertEqual(trace["extraction_outcome"], "provider_failed")
        self.assertFalse(trace["response_valid"])

    def test_invalid_provider_envelope_has_safe_failure_code_and_type(self):
        with patch("app.model.urllib.request.urlopen", return_value=io.BytesIO(b"{}")):
            self.assertIsNone(generate_interview_extraction("شخص هددني", "narrative"))
        self.assertEqual(last_call_trace()["failure_code"], "invalid_provider_response")
        diagnostic = last_extraction_diagnostic()
        self.assertEqual(diagnostic["provider_error_type"], "KeyError")
        self.assertIsNone(diagnostic["raw_response"])

    def test_extraction_request_contains_only_current_message_and_pending_slot(self):
        text = "شخص يهددني برسالة"
        candidate = {"facts": [{"field": "threat", "status": "reported", "value": True, "quote": text}]}
        with patch("app.model._chat", return_value=json.dumps(candidate, ensure_ascii=False)) as chat:
            result = generate_interview_extraction(text, "narrative")
        self.assertEqual(result[0]["field"], "threat")
        sent = json.loads(chat.call_args.args[0][1]["content"])
        self.assertEqual(set(sent), {"pending_question_slot", "current_user_message", "allowed_fact_fields_and_types"})
        self.assertEqual(sent["current_user_message"], text)
        self.assertEqual(sent["allowed_fact_fields_and_types"]["threat"], "binary")
        self.assertEqual(sent["pending_question_slot"], "narrative")
        self.assertNotIn("session_id", str(sent))
        self.assertNotIn("transcript", str(sent))

    def test_extraction_request_includes_only_explicit_bounded_context(self):
        text = "نفس المبلغ الأول"
        context = {"context_version": "synthetic-v1", "pending_question": {"slot": "requested_amount_relation"},
                   "related_facts": [], "related_money_events": [], "untrusted_user_data": True}
        interpretation = {"goals": ["answer_pending_question"], "pending_slot": "requested_amount_relation",
                          "interpretation_only": True}
        candidate = {"facts": []}
        with patch("app.model._chat", return_value=json.dumps(candidate, ensure_ascii=False)) as chat:
            result = generate_interview_extraction(text, "requested_amount_relation", context, interpretation)
        self.assertEqual(result, [])
        sent = json.loads(chat.call_args.args[0][1]["content"])
        self.assertEqual(sent["bounded_untrusted_context"], context)
        self.assertEqual(sent["local_message_goal_hints"], interpretation)
        self.assertEqual(sent["current_user_message"], text)
        self.assertNotIn("session_id", json.dumps(sent, ensure_ascii=False))
        self.assertNotIn("case_token", json.dumps(sent, ensure_ascii=False))
        system = chat.call_args.args[0][0]["content"]
        self.assertIn("تلميحات الأهداف غير موثوقة", system)
        self.assertIn("ask_question", system)

    def test_contextual_yes_is_valid_only_for_the_referenced_pending_question(self):
        text = "نعم"
        context = {
            "pending_question": {"slot": "repeated", "question_ref": "m8", "text": "هل تكرر التهديد؟"},
            "recent_relevant_messages": [{"message_ref": "m8", "role": "assistant", "text": "هل تكرر التهديد؟"}],
            "related_facts": [], "related_money_events": [],
        }
        candidate = {"facts": [{"field": "repeated", "status": "reported", "value": True,
                                "current_quote": text, "question_ref": "m8"}]}
        accepted = validate_interview_facts_diagnostic(text, json.dumps(candidate, ensure_ascii=False),
                                                       "repeated", context)
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["facts"][0]["field"], "repeated")
        self.assertEqual(accepted["facts"][0]["question_ref"], "m8")

        candidate["facts"][0]["question_ref"] = "m404"
        rejected = validate_interview_facts_diagnostic(text, json.dumps(candidate, ensure_ascii=False),
                                                       "repeated", context)
        self.assertEqual(rejected["filtered_facts"][0]["filter_code"], "context_reference_invalid")

    def test_same_operation_requires_and_preserves_an_authorized_target_fact(self):
        text = "نفس المحفظة الإلكترونية."
        fact_ref = "fact:payment_method:m2"
        context = {"pending_question": {"slot": "payment_method", "question_ref": "m3"},
                   "recent_relevant_messages": [{"message_ref": "m2", "role": "user",
                                                "text": "حولت عن طريق محفظة إلكترونية."}],
                   "related_facts": [{"fact_ref": fact_ref, "field": "payment_method", "status": "reported",
                                      "value": "محفظة إلكترونية", "quote": "حولت عن طريق محفظة إلكترونية.",
                                      "message_ref": "m2"}], "related_money_events": []}
        payload = {"facts": [{"field": "payment_method", "status": "reported", "value": "محفظة إلكترونية",
                              "current_quote": text, "operation": "same", "target_fact_id": fact_ref,
                              "context_refs": [fact_ref]}]}
        result = validate_interview_facts_diagnostic(text, json.dumps(payload, ensure_ascii=False),
                                                     "payment_method", context)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["facts"][0]["target_fact_id"], fact_ref)
        self.assertEqual(result["facts"][0]["operation"], "same")

        payload["facts"][0]["target_fact_id"] = "fact:identity:m2"
        rejected = validate_interview_facts_diagnostic(text, json.dumps(payload, ensure_ascii=False),
                                                       "payment_method", context)
        self.assertEqual(rejected["filtered_facts"][0]["filter_code"], "context_reference_invalid")

    def test_operations_are_minimal_and_reference_only_when_authorized(self):
        text = "الموعد نفسه هو بكرا."
        baseline_fact = {"field": "deadline", "status": "reported", "value": "بكرا", "quote": text}

        no_operation = validate_interview_facts_diagnostic(
            text, json.dumps({"facts": [baseline_fact]}, ensure_ascii=False), "deadline")
        self.assertEqual(no_operation["facts"], [baseline_fact])

        unsupported = dict(baseline_fact, operation="unrecorded-operation")
        rejected_unsupported = validate_interview_facts_diagnostic(
            text, json.dumps({"facts": [unsupported]}, ensure_ascii=False), "deadline")
        self.assertEqual(rejected_unsupported["filtered_facts"][0]["filter_code"], "operation_unsupported")
        self.assertNotIn("unrecorded-operation", json.dumps(rejected_unsupported, ensure_ascii=False))

        missing_target = dict(baseline_fact, operation="same")
        rejected_missing = validate_interview_facts_diagnostic(
            text, json.dumps({"facts": [missing_target]}, ensure_ascii=False), "deadline")
        self.assertEqual(rejected_missing["filtered_facts"][0]["filter_code"], "operation_target_missing")

        arbitrary_direct_op = dict(baseline_fact, operation="answer")
        rejected_answer_op = validate_interview_facts_diagnostic(
            text, json.dumps({"facts": [arbitrary_direct_op]}, ensure_ascii=False), "deadline")
        self.assertEqual(rejected_answer_op["filtered_facts"][0]["filter_code"], "operation_unsupported")

    def test_same_operation_requires_explicit_reference_wording_even_with_target(self):
        text = "المحفظة الإلكترونية متاحة عندي."
        fact_ref = "fact:payment_method:synthetic-previous-message"
        context = {
            "pending_question": {"slot": "payment_method", "question_ref": "synthetic-question"},
            "recent_relevant_messages": [{"message_ref": "synthetic-previous-message", "role": "user",
                                         "text": "أستخدم المحفظة الإلكترونية."}],
            "related_facts": [{"fact_ref": fact_ref, "field": "payment_method", "status": "reported",
                               "value": "محفظة إلكترونية", "quote": "أستخدم المحفظة الإلكترونية.",
                               "message_ref": "synthetic-previous-message"}],
            "related_money_events": [],
        }
        proposal = {"facts": [{"field": "payment_method", "status": "reported",
                               "value": "محفظة إلكترونية", "quote": text, "operation": "same",
                               "target_fact_id": fact_ref}]}
        result = validate_interview_facts_diagnostic(text, json.dumps(proposal, ensure_ascii=False),
                                                      "payment_method", context)
        self.assertEqual(result["filtered_facts"][0]["filter_code"], "operation_reference_unsupported")

    def test_deadline_value_must_preserve_before_after_and_during_qualifiers(self):
        cases = [
            ("قال ادفع قبل بكرا.", "بكرا", False),
            ("قال ادفع قبل بكرا.", "قبل بكرا", True),
            ("الموعد بعد نهاية الشهر.", "نهاية الشهر", False),
            ("الموعد بعد نهاية الشهر.", "بعد نهاية الشهر", True),
            ("السداد خلال يومين.", "يومين", False),
            ("السداد خلال يومين.", "خلال يومين", True),
        ]
        for text, value, should_accept in cases:
            with self.subTest(text=text, value=value):
                fact = {"field": "deadline", "status": "reported", "value": value, "quote": text}
                result = validate_interview_facts_diagnostic(
                    text, json.dumps({"facts": [fact]}, ensure_ascii=False), "deadline")
                self.assertTrue(result["accepted"])
                self.assertEqual(bool(result["facts"]), should_accept)
                if not should_accept:
                    self.assertEqual(result["filtered_facts"][0]["filter_code"], "value_inconsistent")

    def test_deadline_constraints_keep_direction_target_and_offset_associated(self):
        cases = [
            ("موعد السداد قبل نهاية الأسبوع بيوم.", "قبل نهاية الأسبوع", False),
            ("موعد السداد قبل نهاية الأسبوع بيوم.", "قبل نهاية الأسبوع بيوم", True),
            ("موعد السداد قبل نهاية الأسبوع بيوم واحد.", "قبل نهاية الأسبوع بيوم واحد", True),
            ("موعد السداد قبل نهاية الأسبوع بيوم واحد.", "قبل نهاية الأسبوع بيوم", False),
            ("الموعد قبل بكرا وبعد الساعة 6.", "قبل الساعة 6 وبعد بكرا", False),
            ("الموعد قبل بكرا وبعد الساعة 6.", "قبل بكرا وبعد الساعة 6", True),
            ("الموعد قبل بكرا وبعد الساعة 6.", "قبل بكرا", False),
            ("موعد السداد قبل العيد.", "قبل العيد", False),
            ("بعثلي وقال موعد السداد قبل نهاية الأسبوع بيوم وأنا كنت مرتبك.",
             "قبل نهاية الأسبوع بيوم", True),
            ("موعد السداد قبل نهاية الأسبوع بيوم.", "قبل نهاية الأسبوع بيومين", False),
        ]
        for text, value, should_accept in cases:
            with self.subTest(text=text, value=value):
                fact = {"field": "deadline", "status": "reported", "value": value, "quote": text}
                result = validate_interview_facts_diagnostic(
                    text, json.dumps({"facts": [fact]}, ensure_ascii=False), "deadline")
                self.assertTrue(result["accepted"])
                self.assertEqual(bool(result["facts"]), should_accept)
                if not should_accept:
                    self.assertEqual(result["filtered_facts"][0]["filter_code"], "value_inconsistent")

    def test_prompt_simplification_reduces_request_size_without_extra_calls(self):
        text = "شو معنى المهلة؟"
        response = {"message_goals": ["ask_question"], "guidance_question": text, "facts": []}
        captured = []

        def fake_urlopen(request, timeout):
            captured.append(bytes(request.data))
            envelope = {"choices": [{"message": {"content": json.dumps(response, ensure_ascii=False)}}]}
            return io.BytesIO(json.dumps(envelope, ensure_ascii=False).encode("utf-8"))

        with patch("app.model.urllib.request.urlopen", side_effect=fake_urlopen):
            result = generate_interview_extraction(text, "deadline")
        self.assertEqual(result, [])
        self.assertEqual(len(captured), 1)
        self.assertLess(len(captured[0]), 8030)
        self.assertEqual(last_call_trace()["request_body_bytes"], len(captured[0]))
        self.assertEqual(last_call_trace()["provider_attempt_count"], 1)

    def test_malformed_extraction_falls_back_but_semantic_fact_is_filtered(self):
        text = "طلب مني 100 دينار"
        with patch("app.model._chat", return_value='{"facts":[{"field":"requested_amount","status":"reported","value":"200 دينار","quote":"طلب مني 100 دينار"}]}'):
            self.assertEqual(generate_interview_extraction(text, "narrative"), [])
        with patch("app.model._chat", return_value='not JSON'):
            self.assertIsNone(generate_interview_extraction(text, "narrative"))

    def test_diagnostic_codes_distinguish_invalid_and_truncated_json(self):
        invalid = validate_interview_facts_diagnostic("text", "not JSON")
        truncated = validate_interview_facts_diagnostic("text", '{"facts":[{"field":"threat"')
        self.assertEqual(invalid["rejection_code"], "invalid_json")
        self.assertFalse(invalid["json_valid"])
        self.assertEqual(truncated["rejection_code"], "truncated_json")

    def test_diagnostic_codes_distinguish_unsupported_field_and_status(self):
        field = {"facts": [{"field": "direct_danger", "status": "reported", "value": True, "quote": "threat"}]}
        status = {"facts": [{"field": "threat", "status": "maybe", "value": True, "quote": "threat"}]}
        self.assertEqual(validate_interview_facts_diagnostic("threat", json.dumps(field))["rejection_code"],
                         "unsupported_field_or_status")
        self.assertEqual(validate_interview_facts_diagnostic("threat", json.dumps(status))["rejection_code"],
                         "unsupported_field_or_status")

    def test_diagnostic_codes_distinguish_quote_and_value_inconsistency(self):
        wrong_quote = {"facts": [{"field": "threat", "status": "reported", "value": True,
                                  "quote": "لم يرد هذا الاقتباس"}]}
        wrong_value = {"facts": [{"field": "paid_amount", "status": "reported", "value": "300 دينار",
                                  "quote": "دفعت 30 دينار"}]}
        quote_diag = validate_interview_facts_diagnostic("هددني", json.dumps(wrong_quote, ensure_ascii=False))
        value_diag = validate_interview_facts_diagnostic("دفعت 30 دينار", json.dumps(wrong_value, ensure_ascii=False))
        self.assertEqual(quote_diag["filtered_facts"][0]["filter_code"], "quote_mismatch")
        self.assertEqual(value_diag["filtered_facts"][0]["filter_code"], "value_inconsistent")

    def test_diagnostic_reports_json_schema_and_semantic_validity_separately(self):
        malformed_shape = validate_interview_facts_diagnostic("هددني", '{"facts":{}}')
        quote_failure = validate_interview_facts_diagnostic("هددني", json.dumps({"facts": [{
            "field": "threat", "status": "reported", "value": True, "quote": "اقتباس آخر"}]}, ensure_ascii=False))
        accepted = validate_interview_facts_diagnostic("هددني", json.dumps({"facts": [{
            "field": "threat", "status": "reported", "value": True, "quote": "هددني"}]}, ensure_ascii=False))
        self.assertEqual(malformed_shape["rejection_code"], "other_validation")
        self.assertTrue(malformed_shape["json_valid"])
        self.assertFalse(malformed_shape["schema_valid"])
        self.assertTrue(quote_failure["schema_valid"])
        self.assertTrue(quote_failure["accepted"])
        self.assertEqual(quote_failure["filtered_facts"][0]["filter_code"], "quote_mismatch")
        self.assertTrue(accepted["accepted"])
        self.assertTrue(accepted["json_valid"] and accepted["schema_valid"])

    def test_filtered_response_detail_is_internal_not_in_public_provider_trace(self):
        text = "في شخص يهددني"
        raw = json.dumps({"facts": [{"field": "threat", "status": "reported", "value": True,
                                     "quote": "نص مخترع"}]}, ensure_ascii=False)
        def fake_chat(_messages, **_kwargs):
            from app.model import _update_call_trace
            _update_call_trace(attempted=True, status="generated", result_accepted=False,
                               failure_code=None, latency_ms=1.0)
            return raw
        with patch("app.model._chat", side_effect=fake_chat):
            self.assertEqual(generate_interview_extraction(text, "narrative"), [])
        self.assertEqual(last_extraction_diagnostic()["filtered_facts"][0]["filter_code"], "quote_mismatch")
        self.assertEqual(last_extraction_diagnostic()["raw_response"], raw)
        self.assertNotIn("raw_response", last_call_trace())
        self.assertEqual(last_call_trace()["failure_code"], "all_facts_filtered")
        self.assertEqual(last_call_trace()["extraction_outcome"], "all_facts_filtered")
        self.assertEqual(last_call_trace()["filtered_fact_count"], 1)


if __name__ == "__main__":
    unittest.main()
