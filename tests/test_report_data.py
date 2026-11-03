import unittest

from app.core import new_state, report_data, render_report


class ReportDataTests(unittest.TestCase):
    def test_json_and_markdown_preserve_arabic_labels_unknowns_conflicts_and_money_relations(self):
        state = new_state("synthetic-report-case")
        source_time = state["messages"][0]["created_at"]
        state["facts"] = {
            "platform": {
                "field": "platform", "status": "conflict", "value": None,
                "quote": "إنستغرام", "message_id": "m1", "updated_at": source_time,
                "alternatives": [{"status": "reported", "value": "تيك توك", "quote": "تيك توك"}],
            },
            "identity": {
                "field": "identity", "status": "unknown", "value": None,
                "quote": "ما بعرف هويته", "message_id": "m1", "updated_at": source_time,
            },
        }
        state["money_events"] = [{
            "id": "money-1", "kind": "requested", "amount": "100 دينار", "relation": "unresolved",
            "quote": "طلب 100 دينار", "message_id": "m1", "created_at": source_time,
            "relation_quote": "طلب 120 دينار أيضًا", "relation_message_id": "m1",
        }]

        exported = report_data(state)
        self.assertEqual(exported["field_labels"]["platform"], "المنصة")
        self.assertIn("identity", exported["unknown_fields"])
        self.assertIn("platform", exported["conflicting_fields"])
        self.assertEqual(len(exported["unresolved_money_events"]), 1)
        self.assertEqual(exported["facts"]["platform"]["source_message_at"], source_time)
        self.assertEqual(exported["facts"]["platform"]["last_updated_at"], source_time)

        markdown = render_report(state)
        self.assertIn("### المنصة", markdown)
        self.assertIn("- الحالة: تعارض فعلي يحتاج إلى توضيح", markdown)
        self.assertIn("- القيمة: لا توجد قيمة مثبتة لهذا الحقل", markdown)
        self.assertIn("التعارضات الفعلية: المنصة", markdown)
        self.assertIn("علاقة مبلغ غير محسومة", markdown)
        self.assertIn("وقت الرسالة الأصلية:", markdown)
        self.assertIn("وقت آخر تحديث للمعلومة:", markdown)


if __name__ == "__main__":
    unittest.main()
