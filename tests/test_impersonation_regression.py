import json
import unittest
from pathlib import Path

from app.core import handle_turn, last_model_merge_trace, new_state


CASES = json.loads(Path(__file__).with_name("impersonation_regression_cases.json").read_text(encoding="utf-8"))


class ArabicImpersonationRegressionTests(unittest.TestCase):
    def assert_fact_source(self, state, field, expected_quote=None):
        fact = state["facts"][field]
        source = next(message for message in state["messages"] if message["id"] == fact["message_id"])
        self.assertIn(fact["quote"], source["text"])
        if expected_quote is not None:
            self.assertEqual(fact["quote"], expected_quote)
        return fact

    def assert_all_fact_sources(self, state):
        messages = {message["id"]: message["text"] for message in state["messages"]}
        for key, fact in state["facts"].items():
            with self.subTest(field=key):
                self.assertIn(fact.get("message_id"), messages)
                if fact.get("quote"):
                    self.assertIn(fact["quote"], messages[fact["message_id"]])
                for support in fact.get("supporting_statements", []):
                    self.assertIn(support.get("message_id"), messages)
                    self.assertIn(support.get("quote", ""), messages[support["message_id"]])

    def test_exact_three_turn_sequence_is_resolved_by_rules_stage_by_stage(self):
        state = new_state("synthetic-impersonation-three-turn")
        first, second, third = CASES["sequence"]

        handle_turn(state, first)
        self.assertEqual(state["facts"]["impersonation"]["status"], "reported")
        self.assertIs(state["facts"]["impersonation"]["value"], True)
        self.assertEqual(state["facts"]["details"]["status"], "reported")
        self.assertEqual(state["facts"]["evidence"]["status"], "reported")
        self.assertNotIn("unclassified_details", state["facts"])
        self.assertEqual(state["facts"]["identity"]["status"], "unknown")
        self.assertIs(state["facts"]["original_account_access_available"]["value"], True)
        self.assertNotIn("account_access", state["facts"])
        self.assertEqual(state["facts"]["threat"]["status"], "denied")
        self.assertNotIn("requested_amount", state["facts"])
        self.assertNotIn("paid_amount", state["facts"])
        self.assert_fact_source(state, "impersonation", first)
        self.assert_fact_source(state, "details", first)
        self.assert_fact_source(state, "evidence", first)
        self.assert_fact_source(state, "identity", first)
        self.assert_fact_source(state, "original_account_access_available", first)

        first_evidence = state["facts"]["evidence"]["quote"]
        handle_turn(state, second)
        self.assertEqual(state["facts"]["impersonation"]["status"], "reported")
        self.assertEqual(state["facts"]["evidence"]["status"], "reported")
        self.assertNotIn("alternatives", state["facts"]["evidence"])
        self.assertEqual(state["facts"]["evidence"]["quote"], first_evidence)
        self.assertIn(second, [entry["quote"] for entry in state["facts"]["details"].get("supporting_statements", [])])
        self.assertNotIn("unclassified_details", state["facts"])
        self.assertNotEqual(state["pending_slot"], "details")
        self.assertNotEqual(state["pending_slot"], "unclassified_details")
        self.assert_fact_source(state, "details", first)

        handle_turn(state, third)
        evidence = state["facts"]["evidence"]
        self.assertEqual(evidence["status"], "reported")
        self.assertNotIn("alternatives", evidence)
        self.assertIn(third, [entry["quote"] for entry in evidence.get("supporting_statements", [])])
        self.assertNotEqual(state["pending_slot"], "details")
        self.assertNotEqual(state["pending_slot"], "unclassified_details")
        self.assert_fact_source(state, "evidence", first)
        support = next(item for item in evidence["supporting_statements"] if item["quote"] == third)
        source = next(message for message in state["messages"] if message["id"] == support["message_id"])
        self.assertEqual(source["text"], third)
        self.assertNotIn("account_access", state["facts"])
        self.assertNotIn("requested_amount", state["facts"])
        self.assertNotIn("paid_amount", state["facts"])
        self.assert_all_fact_sources(state)

    def test_paraphrased_clear_claims_are_classified_without_fixed_keyword(self):
        for index, text in enumerate(CASES["positive_variants"]):
            with self.subTest(text=text):
                state = new_state(f"synthetic-impersonation-variant-{index}")
                handle_turn(state, text)
                self.assertEqual(state["facts"].get("impersonation", {}).get("status"), "reported")
                self.assertIs(state["facts"]["impersonation"]["value"], True)
                self.assertEqual(state["facts"]["details"]["status"], "reported")
                self.assert_fact_source(state, "impersonation", text)

    def test_name_photo_lookalike_and_fan_cases_are_not_auto_classified(self):
        for index, text in enumerate(CASES["negative_boundaries"]):
            with self.subTest(text=text):
                state = new_state(f"synthetic-not-impersonation-{index}")
                handle_turn(state, text)
                fact = state["facts"].get("impersonation")
                if index in {2, len(CASES["negative_boundaries"]) - 1}:
                    self.assertEqual(fact["status"], "denied")
                    self.assertIs(fact["value"], False)
                else:
                    self.assertIsNone(fact, "name/photo similarity, an unknown claim, or fan account is not proof of impersonation")
                self.assertNotIn("account_access", state["facts"])

    def test_conflicting_mocked_model_facts_create_user_resolvable_conflicts(self):
        state = new_state("synthetic-impersonation-mocked-merge")
        text = CASES["sequence"][0]
        handle_turn(state, text, model_facts=[
            {"field": "impersonation", "status": "denied", "value": False, "quote": text},
            {"field": "evidence", "status": "denied", "value": None, "quote": text},
        ])
        for field in ("impersonation", "evidence"):
            fact = state["facts"][field]
            self.assertEqual(fact["status"], "conflict")
            self.assertEqual({item["source"] for item in fact["alternatives"]}, {"rules", "model"})
            self.assertEqual({item["status"] for item in fact["alternatives"]}, {"reported", "denied"})
        dispositions = {(item["field"], item["disposition"]) for item in last_model_merge_trace()}
        self.assertIn(("impersonation", "conflict_created"), dispositions)
        self.assertIn(("evidence", "conflict_created"), dispositions)
        # Synthetic proposals above test disagreement recording only, not a real provider call.


if __name__ == "__main__":
    unittest.main()
