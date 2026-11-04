import copy
import unittest

from app.core import QUESTIONS, handle_turn, last_model_merge_trace, new_state, render_report, report_data


FIRST = (
    "اليوم اكتشفت حساب على إنستغرام مستخدم اسمي وصورتي وبحكي للناس إنه أنا. "
    "حسابي الأصلي معي وبقدر أدخل عليه. ما بعرف مين عامل الحساب، وعندي لقطات شاشة للحساب المزيف، "
    "وما في تهديد أو طلب مصاري."
)
CLARIFICATION = "للتوضيح: أنا اكتشفت الحساب المزيف اليوم، لكن ما بعرف متى أنشأه الشخص."


class TemporalClarificationTests(unittest.TestCase):
    def test_exact_review_clarification_separates_discovery_from_creation_and_invalidates_confirmation(self):
        state = new_state("temporal-exact-case")
        handle_turn(state, FIRST)
        self.assertEqual(state["facts"]["discovery_time"]["status"], "reported")
        self.assertEqual(state["facts"]["discovery_time"]["value"], "اليوم")
        self.assertNotIn("initial_time", state["facts"], "discovery is not event start")
        self.assertEqual(state["stage"], "review")
        self.assertIsNone(state["pending_slot"])
        self.assertIn("هذا غير وقت اكتشافك", QUESTIONS["initial_time"])
        self.assertIn("يختلف عن وقت اكتشافك", QUESTIONS["account_creation_time"])

        untouched_fields = ("impersonation", "platform", "identity", "evidence", "original_account_access_available")
        protected_before = {key: copy.deepcopy(state["facts"][key]) for key in untouched_fields}
        state["stage"] = "review"
        state["pending_slot"] = None
        state["confirmed"] = True
        state["confirmed_digest"] = "previously-confirmed-version"
        previous_version = state["summary_version"]

        handle_turn(state, CLARIFICATION)
        self.assertEqual(state["stage"], "review")
        self.assertIsNone(state["pending_slot"])
        self.assertEqual(state["facts"]["discovery_time"]["value"], "اليوم")
        self.assertEqual(state["facts"]["account_creation_time"]["status"], "unknown")
        self.assertIsNone(state["facts"]["account_creation_time"]["value"])
        self.assertIn("ما بعرف متى أنشأه الشخص", state["facts"]["account_creation_time"]["quote"])
        self.assertNotIn("additional_details", state["facts"])
        for key in untouched_fields:
            self.assertEqual(state["facts"][key], protected_before[key], key)
        self.assertNotIn("requested_amount", state["facts"])
        self.assertNotIn("paid_amount", state["facts"])
        self.assertEqual(state["facts"]["threat"]["status"], "denied")
        self.assertFalse(state["confirmed"])
        self.assertIsNone(state["confirmed_digest"])
        self.assertGreater(state["summary_version"], previous_version)

        report = report_data(state)
        self.assertEqual(report["facts"]["discovery_time"]["value"], "اليوم")
        self.assertEqual(report["facts"]["account_creation_time"]["status"], "unknown")
        self.assertIn("لا يدل وحده على وقت حدوثها", report["field_notes"]["discovery_time"])
        markdown = render_report(state)
        self.assertIn("وقت اكتشاف الواقعة: اليوم", markdown)
        self.assertIn("وقت إنشاء الحساب المزيف", markdown)
        self.assertIn("وقت إنشاء الحساب المزيف؛ لا يُستنتج من وقت اكتشافه", markdown)
        self.assertIn(FIRST, markdown)
        self.assertIn("للتوضيح: أنا اكتشفت الحساب المزيف اليوم", markdown)
        self.assertIn("ما بعرف متى أنشأه الشخص", markdown)

    def test_legacy_discovery_misfiled_as_initial_time_is_reclassified_with_history(self):
        state = new_state("temporal-legacy-case")
        handle_turn(state, FIRST)
        original_message = next(message for message in state["messages"] if message["role"] == "user")
        # Simulate the old parser's persisted interpretation, while keeping its original source intact.
        state["facts"]["initial_time"] = {
            "field": "initial_time", "status": "reported", "value": FIRST, "quote": FIRST,
            "message_id": original_message["id"], "question_id": "m1", "updated_at": original_message["created_at"],
            "history": [],
        }
        state["stage"] = "review"
        state["pending_slot"] = None

        handle_turn(state, CLARIFICATION)
        initial_time = state["facts"]["initial_time"]
        self.assertEqual(initial_time["status"], "unknown")
        self.assertIsNone(initial_time["value"])
        self.assertEqual(initial_time["history"][-1]["quote"], FIRST)
        self.assertEqual(initial_time["history"][-1]["message_id"], original_message["id"])
        self.assertEqual(initial_time["history"][-1]["reason"], "temporal_clarification")
        self.assertEqual(state["facts"]["discovery_time"]["value"], "اليوم")
        self.assertEqual(state["facts"]["account_creation_time"]["status"], "unknown")
        markdown = render_report(state)
        self.assertIn(f"الإفادة السابقة: «{FIRST}»", markdown)

    def test_discovery_and_creation_can_have_distinct_relative_times(self):
        state = new_state("temporal-distinct-times")
        handle_turn(state, "اليوم اكتشفت حساب مزيف على إنستغرام باسمي.")
        state["stage"] = "review"
        state["pending_slot"] = None
        handle_turn(state, "للتوضيح: اكتشفت الحساب اليوم، لكن الشخص أنشأه قبل أسبوعين.")
        self.assertEqual(state["facts"]["discovery_time"]["value"], "اليوم")
        self.assertEqual(state["facts"]["account_creation_time"]["value"], "قبل أسبوعين")
        self.assertNotEqual(state["facts"]["discovery_time"]["value"], state["facts"]["account_creation_time"]["value"])
        self.assertNotIn("initial_time", state["facts"])
        self.assertIn("قبل أسبوعين", state["facts"]["account_creation_time"]["quote"])

    def test_unknown_creation_time_does_not_overwrite_existing_discovery_time(self):
        state = new_state("temporal-unknown-is-scoped")
        handle_turn(state, "اليوم اكتشفت حساب مزيف على إنستغرام باسمي.")
        discovery = copy.deepcopy(state["facts"]["discovery_time"])
        state["stage"] = "review"
        state["pending_slot"] = None
        handle_turn(state, "ما بعرف متى أنشأه الشخص.")
        self.assertEqual(state["facts"]["discovery_time"]["value"], discovery["value"])
        self.assertEqual(state["facts"]["discovery_time"]["quote"], discovery["quote"])
        self.assertEqual(state["facts"]["account_creation_time"]["status"], "unknown")

    def test_passive_unknown_creation_wording_is_scoped_to_creation_time(self):
        state = new_state("temporal-passive-unknown")
        handle_turn(state, "اليوم اكتشفت حساب مزيف على إنستغرام باسمي.")
        discovery_quote = state["facts"]["discovery_time"]["quote"]
        state["stage"] = "review"
        state["pending_slot"] = None
        handle_turn(state, "ما بعرف متى أُنشئ الحساب المزيف.")
        self.assertEqual(state["facts"]["discovery_time"]["value"], "اليوم")
        self.assertEqual(state["facts"]["discovery_time"]["quote"], discovery_quote)
        self.assertEqual(state["facts"]["account_creation_time"]["status"], "unknown")
        self.assertIn("أُنشئ الحساب المزيف", state["facts"]["account_creation_time"]["quote"])

    def test_model_suggestion_cannot_reclassify_discovery_as_event_start(self):
        state = new_state("temporal-model-merge")
        handle_turn(state, FIRST, model_facts=[{
            "field": "initial_time", "status": "reported", "value": "اليوم", "quote": FIRST,
        }])
        self.assertNotIn("initial_time", state["facts"])
        trace = last_model_merge_trace()
        self.assertTrue(any(item.get("field") == "initial_time" and item.get("reason") == "discovery_time_is_not_event_start"
                            for item in trace))


if __name__ == "__main__":
    unittest.main()
