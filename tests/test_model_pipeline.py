import unittest

from app.core import handle_turn, new_state


class ModelPipelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
