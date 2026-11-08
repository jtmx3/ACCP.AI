import unittest

from app.core import new_state, report_data, render_report


class ReportDataTests(unittest.TestCase):
    def test_json_and_markdown_preserve_arabic_labels_unknowns_conflicts_and_money_relations(self):
        state = new_state("synthetic-report-case")
        source_time = "2025-02-03T04:05:06+00:00"
        update_time = "2025-02-03T04:07:08+00:00"
        state["messages"].append({"id": "m2", "role": "user", "text": "إنستغرام؛ ما بعرف هويته؛ طلب 100 دينار",
                                  "created_at": source_time, "kind": "answer"})
        state["facts"] = {
            "platform": {
                "field": "platform", "status": "conflict", "value": None,
                "quote": "إنستغرام", "message_id": "m2", "updated_at": update_time,
                "alternatives": [{"status": "reported", "value": "تيك توك", "quote": "تيك توك"}],
            },
            "identity": {
                "field": "identity", "status": "unknown", "value": None,
                "quote": "ما بعرف هويته", "message_id": "m2", "updated_at": update_time,
            },
            "contact_identifier_disclosure": {
                "field": "contact_identifier_disclosure", "status": "declined", "value": None,
                "quote": "ما بدي أكتبه هون", "message_id": "m2", "updated_at": update_time,
            },
            "payment_method": {
                "field": "payment_method", "status": "not_remembered", "value": None,
                "quote": "ما بتذكر وسيلة الدفع", "message_id": "m2", "updated_at": update_time,
            },
        }
        state["money_events"] = [{
            "id": "money-1", "kind": "requested", "amount": "100 دينار", "relation": "unresolved",
            "quote": "طلب 100 دينار", "message_id": "m2", "created_at": source_time,
            "relation_quote": "طلب 120 دينار أيضًا", "relation_message_id": "m2",
        }]

        exported = report_data(state)
        self.assertEqual(exported["field_labels"]["platform"], "المنصة")
        self.assertIn("identity", exported["unknown_fields"])
        self.assertNotIn("contact_identifier_disclosure", exported["unknown_fields"])
        self.assertIn("contact_identifier_disclosure", exported["declined_fields"])
        self.assertNotIn("payment_method", exported["unknown_fields"])
        self.assertIn("payment_method", exported["not_remembered_fields"])
        self.assertIn("platform", exported["conflicting_fields"])
        self.assertEqual(len(exported["unresolved_money_events"]), 1)
        self.assertEqual(exported["facts"]["platform"]["source_message_at"], source_time)
        self.assertEqual(exported["facts"]["platform"]["last_updated_at"], update_time)
        self.assertNotEqual(exported["facts"]["platform"]["source_message_at"], exported["generated_at"])

        markdown = render_report(state)
        self.assertIn("### المنصة", markdown)
        self.assertIn("- الحالة: تعارض فعلي يحتاج إلى توضيح", markdown)
        self.assertIn("- القيمة: لا توجد قيمة مثبتة لهذا الحقل", markdown)
        self.assertIn("التعارضات الفعلية: المنصة", markdown)
        self.assertIn("غير معروفة للمستخدم: هوية الشخص", markdown)
        self.assertIn("لا يتذكرها المستخدم: وسيلة الدفع", markdown)
        self.assertIn("فضّل المستخدم عدم الإفصاح: الإفصاح عن معرّف التواصل", markdown)
        self.assertIn("علاقة مبلغ غير محسومة", markdown)
        self.assertIn(f"وقت الرسالة الأصلية: {source_time}", markdown)
        self.assertIn(f"وقت آخر تحديث للمعلومة: {update_time}", markdown)


if __name__ == "__main__":
    unittest.main()
