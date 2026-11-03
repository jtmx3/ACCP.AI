import unittest

from scripts.evaluate_scenarios import _candidate_matches, _model_case_passed


class ModelEvaluationPolicyTests(unittest.TestCase):
    def test_fallback_success_is_not_counted_as_model_success(self):
        integrated_passed = True
        expected_matches = {"threat": True, "details": True}
        calls = [
            {"result_accepted": True, "fallback_used": False},
            {"result_accepted": False, "fallback_used": True},
        ]
        self.assertFalse(_model_case_passed(integrated_passed, expected_matches, calls))

    def test_model_case_requires_valid_accepted_candidates_and_all_expected_facts(self):
        accepted = [{"result_accepted": True, "fallback_used": False}]
        self.assertTrue(_model_case_passed(True, {"threat": True}, accepted))
        self.assertFalse(_model_case_passed(True, {"threat": False}, accepted))
        self.assertFalse(_model_case_passed(False, {"threat": True}, accepted))
        self.assertFalse(_model_case_passed(True, {"threat": True}, []))

    def test_candidate_values_must_match_the_fixed_oracle(self):
        expected = {"status": "reported", "value_contains": "100 دينار"}
        self.assertTrue(_candidate_matches({"status": "reported", "value": "طلب 100 دينار"}, expected))
        self.assertFalse(_candidate_matches({"status": "reported", "value": "طلب 120 دينار"}, expected))
        self.assertFalse(_candidate_matches({"status": "denied", "value": None}, expected))


if __name__ == "__main__":
    unittest.main()
