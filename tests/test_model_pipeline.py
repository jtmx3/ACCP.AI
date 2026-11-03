import unittest

from app.core import handle_turn, last_model_merge_trace, new_state


class ModelPipelineTests(unittest.TestCase):
    def test_merge_trace_distinguishes_rule_agreement_from_rule_precedence(self):
        text = "شخص هددني برسالة."
        state = new_state("synthetic-model-agrees-with-rules")
        handle_turn(state, text, model_facts=[{"field": "threat", "status": "reported", "value": True,
                                               "quote": "هددني"}])
        self.assertEqual(last_model_merge_trace()[0]["disposition"], "agrees_with_rules")
        self.assertTrue(last_model_merge_trace()[0]["rule_extraction_agrees"])

        state = new_state("synthetic-model-disagrees-with-rules")
        handle_turn(state, text, model_facts=[{"field": "threat", "status": "denied", "value": False,
                                               "quote": "هددني"}])
        self.assertEqual(last_model_merge_trace()[0]["disposition"], "superseded_by_rules")
        self.assertFalse(last_model_merge_trace()[0]["rule_extraction_agrees"])

    def test_model_amount_suggestion_cannot_overwrite_initial_amount_or_skip_relation_question(self):
        state = new_state("synthetic-model-money-case")
        turns = [
            "شخص هددني برسالة وطلب مني 100 دينار.",
            "لا يوجد خطر مباشر الآن.",
            "كتب أنه سيؤذيني إذا رفضت الدفع.",
            "بدأ قبل يومين.",
            "ما تكرر بعد الرسالة الأولى.",
        ]
        for text in turns:
            handle_turn(state, text)
        second_amount = "بعدها رجع طلب مني 50 دينار."
        candidate = [{"field": "requested_amount", "status": "reported", "value": "50 دينار",
                      "quote": "طلب مني 50 دينار"}]
        handle_turn(state, second_amount, model_facts=candidate)

        self.assertEqual(state["facts"]["requested_amount"]["value"], "100 دينار")
        self.assertEqual(state["facts"]["requested_amount"]["status"], "reported")
        self.assertEqual(state["money_events"][-1]["amount"], "50 دينار")
        self.assertEqual(state["money_events"][-1]["relation"], "unresolved")
        self.assertEqual(state["pending_slot"], "requested_amount_relation")
        self.assertEqual(last_model_merge_trace(), [{"field": "requested_amount", "disposition": "routed_to_rules",
                                                     "reason": "relation_aware_money_event_parser",
                                                     "rule_extraction_agrees": False,
                                                     "rule_extraction_outcome": "differs_or_no_fact_recorded"}])

    def test_payment_only_model_details_do_not_skip_pending_incident_details(self):
        state = new_state("synthetic-model-details-case")
        handle_turn(state, "صاحب إعلان طلب مني 100 دينار على جهاز مستعمل، وبعدها دفعت 100 دينار بالمحفظة.")
        self.assertEqual(state["pending_slot"], "details")

        text = "بعدين دفعت 20 دينار زيادة رسوم توصيل."
        candidates = [
            {"field": "paid_amount", "status": "reported", "value": "20 دينار", "quote": "دفعت 20 دينار"},
            {"field": "details", "status": "reported", "value": "رسوم توصيل إضافية",
             "quote": "20 دينار زيادة رسوم توصيل"},
        ]
        handle_turn(state, text, model_facts=candidates)

        self.assertEqual(state["facts"]["paid_amount"]["value"], "100 دينار")
        self.assertEqual(state["money_events"][-1]["amount"], "20 دينار")
        self.assertEqual(state["money_events"][-1]["relation"], "additional")
        self.assertNotIn("details", state["facts"])
        self.assertEqual(state["pending_slot"], "details")
        self.assertIn({"field": "details", "disposition": "ignored",
                       "reason": "payment_only_is_not_incident_detail",
                       "merge_rejection_code": "pending_context_conflict"}, last_model_merge_trace())

    def test_later_model_narrative_does_not_conflict_with_opening_account(self):
        state = new_state("synthetic-model-narrative-case")
        opening = "في شخص هددني برسالة وطالبني 100 دينار."
        later = "رجع بعتلي رسالة ثانية اليوم وطلب مني أحوّل المبلغ نفسه."
        handle_turn(state, opening)
        original_narrative = state["facts"]["narrative"]["value"]
        self.assertEqual(state["pending_slot"], "direct_danger")

        candidate = [{"field": "narrative", "status": "reported", "value": later, "quote": later}]
        handle_turn(state, later, model_facts=candidate)

        self.assertEqual(state["facts"]["narrative"]["status"], "reported")
        self.assertEqual(state["facts"]["narrative"]["value"], original_narrative)
        self.assertEqual(state["pending_slot"], "direct_danger")
        self.assertIn({"field": "narrative", "disposition": "ignored",
                       "reason": "preserve_original_narrative_without_explicit_correction"},
                      last_model_merge_trace())


if __name__ == "__main__":
    unittest.main()
