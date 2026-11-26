import json
import unittest
from pathlib import Path

from app.core import _identity_actor_unknown, compose_question, handle_turn, new_state
from app.dialogue import compose_fallback_question, validate_reply_text
from app.nlu import analyze_message, normalize_arabic
from app.stopping import evaluate_stopping_decision
from app.triage import classify_track, route_assessment


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "tests" / "conversation_rebuild_v1_synthetic.json"


class ArabicNluTests(unittest.TestCase):
    def test_shami_normalization_handles_spelling_and_repeated_letters(self):
        self.assertEqual(normalize_arabic("تيلغرام والبارررحه"), "تلغرام ومبارح")
        self.assertEqual(normalize_arabic("آلصوره"), "الصوره")

    def test_verbal_abuse_terms_match_prefixes_and_public_digital_context(self):
        for message in (
            "كتب عني كلام كاذب ومسيء بمنشور عام على إنستغرام",
            "نشر تعليقًا يهينني على فيسبوك",
            "بعتلي رسالة خاصة فيها سب وإهانة",
            "شهّر فيني بتعليقات تيليغرام",
        ):
            with self.subTest(message=message):
                signals = analyze_message(message)
                self.assertTrue(signals["verbal_abuse"])
                self.assertTrue(signals["digital_context"])

    def test_evidence_mention_alone_is_not_an_abuse_incident_or_user_held_evidence(self):
        message = "عندي سكرينشوت وإيميلات محفوظة تثبت الرسائل المسيئة."
        signals = analyze_message(message)
        state = new_state("evidence-only")
        handle_turn(state, message)
        self.assertTrue(signals["evidence_only_statement"])
        self.assertEqual(classify_track(state), "exploratory_intake")
        self.assertEqual(state["facts"]["evidence"]["status"], "reported")

    def test_singular_saved_screenshot_is_recorded_as_reported_evidence(self):
        state = new_state("singular-saved-screenshot")
        state["pending_slot"] = "evidence"
        text = "عندي لقطة شاشة محفوظة"

        handle_turn(state, text)

        evidence = state["facts"]["evidence"]
        self.assertEqual(evidence["status"], "reported")
        self.assertEqual(evidence["value"], "صور أو لقطات شاشة")
        self.assertEqual(evidence["quote"], text)

    def test_unknown_actor_identity_is_scoped_to_person_not_other_facts(self):
        self.assertTrue(_identity_actor_unknown("ما بعرف صاحب الحساب"))
        self.assertTrue(_identity_actor_unknown("ما بعرف مين هددني"))
        self.assertFalse(_identity_actor_unknown("ما بعرف إذا صار تهديد"))
        state = new_state("identity-unknown")
        message = "كتب عني كلام كاذب ومسيء بمنشور عام على إنستغرام، وما بعرف صاحب الحساب."
        handle_turn(state, message)
        self.assertEqual(state["facts"]["identity"]["status"], "unknown")
        self.assertEqual(state["facts"]["platform"]["value"], "إنستغرام")

    def test_private_images_and_fear_do_not_create_evidence_or_emergency(self):
        state = new_state("private-images")
        handle_turn(state, "في شب معي بالجامعة ومعه صوري وأنا خايفة ينشرهم أو يبعتهم لأهلي")
        self.assertNotIn("evidence", state["facts"])
        self.assertNotIn("direct_danger", state["facts"])
        self.assertNotEqual(state["stage"], "urgent_stop")
        self.assertEqual(state["pending_slot"], "publication_threat")
        self.assertNotIn("911", " ".join(m["text"] for m in state["messages"] if m["role"] == "assistant"))

    def test_user_sample_sequence_is_flexible_and_does_not_stop_before_review_is_ready(self):
        state = new_state("sample-sequence")
        first = "في شب معي بالجامعة ومعه صوري وأنا خايفة ينشرهم أو يبعتهم لأهلي"
        handle_turn(state, first)
        first_question = state["messages"][-1]["text"]
        self.assertIn("هل صدر منه تهديد صريح", first_question)
        handle_turn(state, "بدأ اليوم الصبح")
        self.assertEqual(state["facts"]["initial_time"]["status"], "reported")
        self.assertIn("الصبح", state["facts"]["initial_time"]["value"])
        self.assertEqual(state["facts"]["initial_time"]["quote"], "بدأ اليوم الصبح")
        second_question = state["messages"][-1]["text"]
        self.assertNotEqual(first_question, second_question)
        self.assertEqual(state["pending_slot"], "publication_threat")
        handle_turn(state, "لا")
        self.assertEqual(state["facts"]["publication_threat"]["status"], "denied")
        self.assertEqual(state["pending_slot"], "published")
        handle_turn(state, "لا")
        self.assertEqual(state["facts"]["published"]["status"], "denied")
        self.assertEqual(state["pending_slot"], "unclassified_details")
        self.assertNotIn("evidence", state["facts"])
        self.assertNotIn("direct_danger", state["facts"])
        self.assertNotEqual(state["stage"], "review")
        handle_turn(state, "هذا كل شي")
        self.assertEqual(state["stage"], "review")
        self.assertEqual(state["stopping_decision"]["stop_reason"], "user_wrap_up")


class TriageAndStoppingTests(unittest.TestCase):
    def test_low_priority_defamation_track_stops_when_core_fields_are_present(self):
        state = new_state("low-priority-complete")
        state["facts"].update({
            "narrative": {"status": "reported", "value": "شتمني بتعليق على فيسبوك", "quote": "شتمني بتعليق على فيسبوك"},
            "details": {"status": "reported", "value": "شتيمة في تعليق", "quote": "شتمني بتعليق"},
            "platform": {"status": "reported", "value": "فيسبوك", "quote": "على فيسبوك"},
            "identity": {"status": "unknown", "value": None, "quote": "ما بعرفه"},
            "evidence": {"status": "denied", "value": None, "quote": "ما عندي لقطة"},
        })
        self.assertEqual(classify_track(state), "defamation_or_abuse")
        self.assertEqual(route_assessment(state)["priority_score"], 45)
        decision = evaluate_stopping_decision(state)
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.stop_reason, "track_core_complete")
        self.assertEqual(decision.turn_budget, 4)

    def test_distinct_question_budget_and_repeated_unknown_stop(self):
        state = new_state("budget")
        state["facts"]["narrative"] = {"status": "reported", "value": "في شخص يلاحقني برسائل متكررة", "quote": "في شخص يلاحقني برسائل متكررة"}
        state["question_history"].extend({"slot": slot} for slot in (
            "initial_time", "identity", "evidence", "platform", "repeated", "latest_time"))
        decision = evaluate_stopping_decision(state)
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.stop_reason, "adaptive_turn_budget")
        self.assertEqual(decision.turn_budget, 6)
        state["question_history"] = [{"slot": "narrative"}]
        state["turn_signals"] = [
            {"slot": "initial_time", "unknown": True},
            {"slot": "identity", "unknown": True},
        ]
        self.assertEqual(evaluate_stopping_decision(state).stop_reason, "continue_focused_interview")
        state["turn_signals"].append({"slot": "identity", "unknown": True})
        self.assertEqual(evaluate_stopping_decision(state).stop_reason, "diminishing_returns")

    def test_direct_danger_is_only_a_reactive_track(self):
        safe = new_state("digital-fear")
        handle_turn(safe, "خايفة ينشر صوري الخاصة وما في تهديد مباشر وما حدا قريب مني")
        self.assertNotEqual(safe["stage"], "urgent_stop")
        danger = new_state("explicit-danger")
        handle_turn(danger, "شخص قال إنه جاي لعندي هسا ومعه سكين")
        self.assertEqual(danger["stage"], "urgent_stop")
        self.assertEqual(classify_track(danger), "direct_danger")

    def test_negative_impersonation_examples_remain_exploratory(self):
        for text in (
            "في حساب اسمه قريب من اسمي وصورته مختلفة، وما حكى للناس إنه أنا.",
            "حساب معجبين حاط صورتي ومكتوب بوضوح إنه فانز.",
        ):
            state = new_state("not-impersonation")
            handle_turn(state, text)
            with self.subTest(text=text):
                self.assertEqual(classify_track(state), "exploratory_intake")


class DialogueSafetyTests(unittest.TestCase):
    def test_publication_followup_is_grounded_in_fear_without_inventing_a_threat(self):
        state = new_state("fear-only-followup")
        handle_turn(state, "في شب معي بالجامعة ومعه صوري وأنا خايفة ينشرهم أو يبعتهم لأهلي")
        self.assertEqual(state["pending_slot"], "publication_threat")
        question = state["messages"][-1]["text"]
        self.assertIn("هل صدر منه تهديد صريح", question)
        self.assertIn("خوف من احتمال فقط", question)
        self.assertEqual(question.count("؟") + question.count("?"), 1)
        self.assertNotIn("911", question)
        self.assertNotIn("ارفع", question)

    def test_fallback_question_varies_and_stays_single_focused_question(self):
        state = new_state("phrasing")
        first = compose_fallback_question(state, "initial_time", {})
        state["question_history"].append({"slot": "initial_time"})
        second = compose_fallback_question(state, "initial_time", {})
        self.assertNotEqual(first, second)
        self.assertEqual(first.count("؟") + first.count("?"), 1)
        self.assertEqual(second.count("؟") + second.count("?"), 1)

    def test_dialogue_reply_rejects_unsafe_or_unsolicited_emergency_text(self):
        self.assertEqual(validate_reply_text("هل تتذكر تقريبًا متى بدأ أول تواصل؟"), (True, None))
        self.assertEqual(validate_reply_text("هل أنت بخطر جسدي الآن؟")[1], "unsolicited_emergency_content")
        self.assertEqual(validate_reply_text("أرسل كلمة المرور حتى نتابع؟")[1], "unsafe_solicitation_or_claim")


class FrozenSyntheticConversationTests(unittest.TestCase):
    SLOT_GROUPS = {
        "details_or_identity_or_evidence": {"details", "identity", "evidence", "platform"},
        "platform_or_identity": {"platform", "identity"},
        "identity_or_evidence": {"identity", "evidence"},
        "platform_or_time_or_identity_or_evidence": {"platform", "initial_time", "identity", "evidence"},
        "time_or_identity_or_evidence": {"initial_time", "identity", "evidence"},
        "time_or_repetition_or_identity_or_evidence": {"initial_time", "repeated", "identity", "evidence"},
        "deadline_or_publication_or_time": {"deadline", "published", "publication_threat", "initial_time"},
        "time_or_repetition_or_publication_or_identity": {"initial_time", "repeated", "publication_threat", "identity"},
        "time_or_account_recovery_or_identity": {"initial_time", "account_access", "identity"},
        "time_or_payment_method_or_identity_or_evidence": {"initial_time", "payment_method", "identity", "evidence"},
        "open_incident_description": {"unclassified_details", "narrative"},
        "clarify_identity_claim": {"unclassified_details", "incident_type_review"},
        "clarify_behavior": {"unclassified_details", "incident_type_review"},
        "clarify_digital_event_without_physical_danger": {"publication_threat", "unclassified_details", "incident_type_review"},
        "clarify_repeated_harm_vs_contact": {"repeated"},
        "ask_what_incident_the_evidence_relates_to": {"unclassified_details", "incident_type_review"},
    }

    def test_all_twenty_frozen_synthetic_cases_keep_their_track_and_first_focus(self):
        dataset = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
        self.assertEqual(dataset["version"], "1.0")
        self.assertEqual(len(dataset["cases"]), 20)
        self.assertTrue(dataset["frozen_before_implementation"])
        for case in dataset["cases"]:
            with self.subTest(case=case["id"]):
                state = new_state("frozen-" + case["id"])
                handle_turn(state, case["messages"][0])
                if case["expected_first_focus"] == "answer_guidance_not_emergency":
                    # These are questions, not incident disclosures; the API-level RAG route is tested elsewhere.
                    self.assertNotEqual(state["stage"], "urgent_stop")
                    self.assertNotIn("direct_danger", state["facts"])
                    continue
                self.assertEqual(classify_track(state), case["expected_track"])
                if case["expected_first_focus"] == "reactive_safety_stop":
                    self.assertEqual(state["stage"], "urgent_stop")
                    continue
                expected_focus = self.SLOT_GROUPS[case["expected_first_focus"]]
                if state["pending_slot"] is None:
                    # A frozen focus label is satisfied when the initial message already resolves
                    # every field in that focus group; stopping is preferable to asking redundantly.
                    self.assertEqual(state["stage"], "review")
                    self.assertTrue(all(slot in state["facts"] for slot in expected_focus))
                else:
                    self.assertIn(state["pending_slot"], expected_focus)
                self.assertNotEqual(state["stage"], "urgent_stop")
                assistant_text = " ".join(message["text"] for message in state["messages"] if message["role"] == "assistant")
                self.assertNotIn("911", assistant_text)
                self.assertNotIn("خطر جسدي", assistant_text)


if __name__ == "__main__":
    unittest.main()
