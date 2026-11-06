from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.conversation import build_bounded_context, interpret_message
from app.core import _user_intent, classify_current_danger, is_clear_user_question
from app.main import app
from app.model import (
    ALLOWED_EXTRACTION_FIELDS,
    generate_interview_extraction,
    last_call_trace,
    last_extraction_diagnostic,
    model_configured,
    provider_config,
    probe_provider,
    reset_call_trace,
    test_provider_generation,
)
from app.store import Store

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "tests" / "conversation_v3_acceptance.json"
TOPIC_IDS = {
    "account_security": {"GLOBAL-IG-SECURITY-2026"},
    "evidence": {"AU-ESAFETY-EVIDENCE-2026", "AU-ESAFETY-IMAGE-ABUSE-2026"},
    "impersonation_reporting": {"GLOBAL-IG-IMPERSONATION-2026"},
    "payment_recovery": {"US-FTC-SCAM-2026"},
}


def load_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if not suite.get("synthetic_internal_cases") or not (
        suite.get("frozen_before_application_fixes") or suite.get("expectations_locked_before_evaluation")
    ):
        raise ValueError("v3 suite must be synthetic with explicitly locked expectations")
    if not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError("v3 acceptance suite has no cases")
    return suite


def _match_fact(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    if not actual:
        return False, "fact absent"
    for key, wanted in expected.items():
        observed = actual.get(key)
        if key == "value_contains":
            value = actual.get("value")
            if wanted not in str(value or ""):
                return False, f"value did not contain {wanted!r}: {value!r}"
        elif observed != wanted:
            return False, f"{key}: expected {wanted!r}, got {observed!r}"
    return True, "matched"


def _assert_turn(session: dict[str, Any], previous: dict[str, Any], before_message_count: int,
                 expected: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    facts = session.get("facts", {})
    for field, wanted in expected.get("facts", {}).items():
        if field == "pending_slot":
            add("pending_slot", session.get("pending_slot") == wanted, session.get("pending_slot"))
            continue
        ok, detail = _match_fact(facts.get(field, {}), wanted)
        add(f"fact:{field}", ok, detail)
    for field in expected.get("no_fact_fields", []):
        add(f"no_fact:{field}", field not in facts, "absent" if field not in facts else facts[field])
    for field in expected.get("no_true_facts", []):
        fact = facts.get(field, {})
        ok = not (fact.get("status") == "reported" and fact.get("value") is True)
        add(f"not_true:{field}", ok, fact if fact else "absent")
    for field in expected.get("not_pending", []):
        ok = session.get("pending_slot") != field
        add(f"not_pending:{field}", ok, session.get("pending_slot"))
    if "pending_slot" in expected:
        add("pending_slot", session.get("pending_slot") == expected["pending_slot"], session.get("pending_slot"))
    if "money_event_count" in expected:
        count = len(session.get("money_events", []))
        add("money_event_count", count == expected["money_event_count"], count)
    if "paid_event_count" in expected:
        count = sum(1 for event in session.get("money_events", [])
                    if event.get("kind") == "paid" and event.get("relation") != "correction")
        add("paid_event_count", count == expected["paid_event_count"], count)
    for kind, wanted_relations in expected.get("money_relations", {}).items():
        actual_relations = []
        for event in session.get("money_events", []):
            if event.get("kind") != kind:
                continue
            if event.get("relation"):
                actual_relations.append(event["relation"])
            actual_relations.extend(item.get("relation") for item in event.get("history", [])
                                    if item.get("relation"))
        add(f"money_relations:{kind}", actual_relations == wanted_relations,
            {"expected": wanted_relations, "actual": actual_relations})

    new_messages = session.get("messages", [])[before_message_count:]
    assistant = [m for m in new_messages if m.get("role") == "assistant"]
    answers = [m for m in assistant if m.get("kind") == "answer"]
    followups = [m for m in assistant if m.get("kind") in {"question", "clarification"}]
    if "assistant_answer" in expected:
        add("assistant_answer", bool(answers) == expected["assistant_answer"],
            [{"kind": m.get("kind"), "text": m.get("text"), "citations": m.get("citations", [])} for m in answers])
    if "single_followup" in expected:
        add("single_followup", len(followups) == 1,
            [{"kind": m.get("kind"), "text": m.get("text")} for m in followups])
    if expected.get("ask_jurisdiction"):
        joined = "\n".join(m.get("text", "") for m in answers)
        add("ask_jurisdiction", "اذكر البلد" in joined or "تحديد البلد" in joined, joined)
    if "answer_contains" in expected:
        joined = "\n".join(m.get("text", "") for m in answers)
        add("answer_contains", expected["answer_contains"] in joined, joined)
    if "citation_topic" in expected:
        topic = expected["citation_topic"]
        wanted_ids = TOPIC_IDS[topic]
        citation_ids = {source.get("id") for message in answers for source in message.get("citations", [])}
        add(f"citation_topic:{topic}", bool(citation_ids & wanted_ids), sorted(citation_ids))
    return checks


def _provider_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * percentile
    lower, upper = int(rank), min(int(rank) + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def _provider_turn_metrics(api_result: dict[str, Any], model_trace: dict[str, Any],
                           proposals: list[dict[str, Any]],
                           merge_trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify a model attempt using the integrated API's actual fallback decision."""
    attempted = bool(model_trace.get("attempted"))
    fallback_used = bool(api_result.get("fallback_used"))
    outcome = api_result.get("extraction_outcome") or model_trace.get("extraction_outcome")
    status = model_trace.get("status")
    failure_code = model_trace.get("failure_code")
    if not attempted:
        fallback_category = "not_attempted" if fallback_used else None
    elif not fallback_used:
        fallback_category = None
    elif status == "connection_error" or failure_code in {"timeout", "unreachable"}:
        fallback_category = "connection_failure"
    elif status == "http_error" or str(failure_code).startswith("http_"):
        fallback_category = "provider_http_error"
    elif status == "empty_response":
        fallback_category = "empty_provider_response"
    elif outcome == "response_rejected" or (model_trace.get("response_valid") is False and outcome != "provider_failed"):
        fallback_category = "response_rejected"
    elif outcome == "valid_empty":
        fallback_category = "empty_extraction"
    elif outcome == "all_facts_filtered":
        fallback_category = "all_facts_filtered"
    elif outcome == "provider_failed":
        fallback_category = "provider_failure"
    else:
        fallback_category = "no_accepted_extraction"

    ignored_fields = sorted({item.get("field") for item in merge_trace
                             if item.get("disposition") == "ignored" and item.get("field")})
    accepted_fields = sorted({item.get("field") for item in proposals if item.get("field")})
    accepted_dispositions: dict[str, int] = {}
    accepted_merge_trace = [item for item in merge_trace if item.get("field") in set(accepted_fields)]
    for item in accepted_merge_trace:
        disposition = item.get("disposition", "unrecorded")
        accepted_dispositions[disposition] = accepted_dispositions.get(disposition, 0) + 1
    merge_rejected = sorted(set(ignored_fields) & set(accepted_fields))
    routed_fields = sorted({item.get("field") for item in accepted_merge_trace
                            if item.get("disposition") == "routed_to_rules"})
    correction_fields = sorted({item.get("field") for item in accepted_merge_trace
                                if item.get("disposition") == "applied_correction"})
    conflict_fields = sorted({item.get("field") for item in accepted_merge_trace
                              if item.get("disposition") in {"conflict_created", "applied_conflict"}})
    return {"attempted": attempted, "fallback_used": fallback_used,
            "fallback_category": fallback_category, "extraction_outcome": outcome,
            "response_valid": model_trace.get("response_valid"),
            "facts_accepted": bool(proposals), "accepted_fact_count": len(proposals),
            "accepted_then_merge_rejected_fields": merge_rejected,
            "accepted_then_merge_rejected": bool(merge_rejected),
            "accepted_fact_merge_dispositions": accepted_dispositions,
            "accepted_facts_routed_to_rules_fields": routed_fields,
            "accepted_facts_applied_as_corrections_fields": correction_fields,
            "accepted_facts_preserved_as_conflicts_fields": conflict_fields}


def replay_raw(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    total_checks = passed_checks = guidance_checks = guidance_passed = 0
    repeat_checks = repeated = 0
    for case in raw.get("results", []):
        case_result = {"case_id": case.get("case_id"), "turns": []}
        for turn in case.get("turns", []):
            session = {"facts": turn.get("facts", {}), "pending_slot": turn.get("pending_slot"),
                       "money_events": turn.get("money_events", []),
                       "money_corrections": turn.get("money_corrections", []),
                       "messages": turn.get("assistant_after_turn", turn.get("new_assistant_messages", []))}
            checks = _assert_turn(session, {}, 0, turn.get("expected", {}))
            for item in checks:
                total_checks += 1
                passed_checks += int(item["passed"])
                if item["check"] == "assistant_answer" or item["check"].startswith("citation_topic:"):
                    guidance_checks += 1
                    guidance_passed += int(item["passed"])
                if item["check"].startswith("not_pending:"):
                    repeat_checks += 1
                    repeated += int(not item["passed"])
            case_result["turns"].append({"turn": turn.get("turn"), "checks": checks,
                                         "passed": all(item["passed"] for item in checks)})
        case_result["passed"] = bool(case_result["turns"]) and all(t["passed"] for t in case_result["turns"])
        cases.append(case_result)
    return {"source_artifact": str(path), "synthetic_internal_cases": bool(raw.get("synthetic_internal_cases")),
            "provider_calls_attempted": raw.get("provider_calls_attempted", raw.get("provider_attempts", 0)),
            "integrated_system": {"passed_cases": sum(c["passed"] for c in cases), "total_cases": len(cases),
                                  "case_rate": sum(c["passed"] for c in cases) / len(cases) if cases else None,
                                  "passed_assertions": passed_checks, "total_assertions": total_checks,
                                  "assertion_rate": passed_checks / total_checks if total_checks else None},
            "guidance": {"passed": guidance_passed, "checks": guidance_checks,
                         "rate": guidance_passed / guidance_checks if guidance_checks else None},
            "repeat_question_checks": {"checked": repeat_checks, "repeated_or_unresolved": repeated},
            "model_extraction_only": "not_recorded_in_source_artifact", "results": cases}


def run_suite(suite_path: Path, mode: str = "rules", with_context: bool = False) -> dict[str, Any]:
    suite_path = suite_path.resolve()
    suite = load_suite(suite_path)
    provider = provider_config()
    configured = bool(model_configured())
    results: list[dict[str, Any]] = []
    system_checks = system_passed = 0
    model_checks = model_matches = model_omissions = model_forbidden = 0
    model_expected_checks = model_forbidden_checks = 0
    guidance_checks = guidance_passed = followup_checks = followup_repeats = 0
    provider_eligible_turns = 0
    provider_attempts = provider_accepted = provider_rejected = provider_fallbacks = 0
    provider_extractions_with_facts = 0
    provider_accepted_fact_count = 0
    fallback_categories: dict[str, int] = {}
    extraction_outcomes: dict[str, int] = {}
    turns_with_accepted_facts_ignored_by_merge = 0
    accepted_facts_ignored_by_merge = 0
    accepted_facts_routed_to_rules = 0
    accepted_facts_applied_as_corrections = 0
    accepted_facts_preserved_as_conflicts = 0
    accepted_fact_merge_dispositions: dict[str, int] = {}
    provider_latencies: list[float] = []
    model_pipeline_latencies: list[float] = []
    integration_latencies: list[float] = []
    total_turn_latencies: list[float] = []
    first_call_latency: float | None = None
    subsequent_call_latencies: list[float] = []
    calls_by_case: dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="accp-conversation-v3-") as temp_dir:
        store = Store(str(Path(temp_dir) / "evaluation.sqlite3"))
        with patch("app.main.store", store), TestClient(app) as client:
            for case in suite["cases"]:
                created = client.post("/api/sessions", json={})
                created.raise_for_status()
                session_id = created.json()["session"]["session_id"]
                token = created.json()["token"]
                headers = {"X-Case-Token": token}
                current_resp = client.get(f"/api/sessions/{session_id}", headers=headers)
                current_resp.raise_for_status()
                current = current_resp.json()["session"]
                case_result = {"case_id": case["case_id"],
                               "known_prior_holdout_regression": bool(case.get("known_prior_holdout_regression")),
                               "turns": []}
                calls_by_case[case["case_id"]] = 0

                for turn_index, turn in enumerate(case["turns"], start=1):
                    before_resp = client.get(f"/api/sessions/{session_id}", headers=headers)
                    before_resp.raise_for_status()
                    before = before_resp.json()["session"]
                    before_message_count = len(before.get("messages", []))
                    text = turn["text"]
                    kind = turn.get("kind", "answer")
                    extraction_eligible = (mode == "model" and configured and kind == "answer"
                        and not is_clear_user_question(text) and _user_intent(text) is None
                        and classify_current_danger(text, before.get("pending_slot")) is not True)
                    provider_eligible_turns += int(extraction_eligible)
                    model_facts = None
                    model_trace: dict[str, Any] = {}
                    model_diag: dict[str, Any] = {}
                    model_elapsed_ms: float | None = None
                    if extraction_eligible:
                        reset_call_trace()
                        context = build_bounded_context(before, text, before.get("pending_slot")) if with_context else None
                        started = time.perf_counter()
                        model_facts = generate_interview_extraction(
                            text, before.get("pending_slot"), context,
                            interpret_message(text, before.get("pending_slot")))
                        model_elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                        model_trace = last_call_trace() or {}
                        model_diag = last_extraction_diagnostic() or {}
                        if model_trace.get("attempted"):
                            provider_attempts += 1
                            calls_by_case[case["case_id"]] += 1
                            latency = float(model_trace.get("latency_ms", model_elapsed_ms))
                            provider_latencies.append(latency)
                            model_pipeline_latencies.append(float(model_elapsed_ms))
                            if calls_by_case[case["case_id"]] == 1:
                                first_call_latency = latency if first_call_latency is None else first_call_latency
                            else:
                                subsequent_call_latencies.append(latency)
                            if model_trace.get("response_valid") is True:
                                provider_accepted += 1
                            elif model_trace.get("extraction_outcome") == "response_rejected" or model_trace.get("status") == "response_rejected":
                                provider_rejected += 1

                        # Extraction-only scoring is done before integration, against the same frozen expectations.
                        for field, wanted in turn.get("expect", {}).get("facts", {}).items():
                            if field not in ALLOWED_EXTRACTION_FIELDS:
                                continue
                            if _match_fact(before.get("facts", {}).get(field, {}), wanted)[0]:
                                continue
                            model_expected_checks += 1
                            candidate = next((fact for fact in model_facts or [] if fact.get("field") == field), {})
                            ok, _ = _match_fact(candidate, wanted)
                            model_matches += int(ok)
                            model_omissions += int(not ok)
                        forbidden_fields = set(turn.get("expect", {}).get("no_fact_fields", []))
                        forbidden_fields.update(turn.get("expect", {}).get("no_true_facts", []))
                        for field in forbidden_fields:
                            model_forbidden_checks += 1
                            model_forbidden += int(any(f.get("field") == field for f in model_facts or []))

                    payload = {"expected_revision": before["revision"], "text": text, "kind": kind,
                               "allow_external": False,
                               "allow_external_extraction": extraction_eligible,
                               "allow_context_extraction": extraction_eligible and with_context}
                    request_started = time.perf_counter()
                    if extraction_eligible:
                        with patch("app.main.model_configured", return_value=True), \
                             patch("app.main.provider_config", return_value=provider), \
                             patch("app.main.generate_interview_extraction", return_value=model_facts), \
                             patch("app.main.last_call_trace", return_value=model_trace):
                            response = client.post(f"/api/sessions/{session_id}/messages", json=payload, headers=headers)
                    else:
                        response = client.post(f"/api/sessions/{session_id}/messages", json=payload, headers=headers)
                    integration_elapsed_ms = round((time.perf_counter() - request_started) * 1000, 3)
                    integration_latencies.append(integration_elapsed_ms)
                    turn_total_elapsed_ms = round((model_elapsed_ms or 0.0) + integration_elapsed_ms, 3)
                    total_turn_latencies.append(turn_total_elapsed_ms)
                    turn_result: dict[str, Any] = {"turn": turn_index, "text": text,
                                                   "expected": turn.get("expect", {}),
                                                   "http_status": response.status_code,
                                                   "model_extraction": {"eligible": extraction_eligible,
                                                                           "elapsed_ms": model_elapsed_ms,
                                                                        "provider_latency_ms": model_trace.get("latency_ms"),
                                                                        "integration_elapsed_ms": integration_elapsed_ms,
                                                                        "turn_total_elapsed_ms": turn_total_elapsed_ms,
                                                                        "provider_call": model_trace,
                                                                        "diagnostic": model_diag,
                                                                        "proposals": model_facts or []}}
                    if response.status_code != 200:
                        turn_result["checks"] = [{"check": "http_200", "passed": False, "detail": response.text}]
                        turn_result["passed"] = False
                        case_result["turns"].append(turn_result)
                        continue
                    api_result = response.json()
                    after = api_result["session"]
                    model_metrics = _provider_turn_metrics(
                        api_result, model_trace, list(model_facts or []),
                        api_result.get("model_merge_trace", []))
                    if extraction_eligible:
                        provider_fallbacks += int(model_metrics["fallback_used"])
                        if model_metrics.get("fallback_category"):
                            category = model_metrics["fallback_category"]
                            fallback_categories[category] = fallback_categories.get(category, 0) + 1
                        outcome = model_metrics.get("extraction_outcome")
                        if outcome:
                            extraction_outcomes[outcome] = extraction_outcomes.get(outcome, 0) + 1
                        provider_extractions_with_facts += int(model_metrics["facts_accepted"])
                        provider_accepted_fact_count += model_metrics["accepted_fact_count"]
                        turns_with_accepted_facts_ignored_by_merge += int(model_metrics["accepted_then_merge_rejected"])
                        accepted_facts_ignored_by_merge += len(model_metrics["accepted_then_merge_rejected_fields"])
                        accepted_facts_routed_to_rules += len(model_metrics["accepted_facts_routed_to_rules_fields"])
                        accepted_facts_applied_as_corrections += len(model_metrics["accepted_facts_applied_as_corrections_fields"])
                        accepted_facts_preserved_as_conflicts += len(model_metrics["accepted_facts_preserved_as_conflicts_fields"])
                        for disposition, count in model_metrics["accepted_fact_merge_dispositions"].items():
                            accepted_fact_merge_dispositions[disposition] = accepted_fact_merge_dispositions.get(disposition, 0) + count
                    checks = _assert_turn(after, before, before_message_count, turn.get("expect", {}))
                    turn_result.update({"pending_slot": after.get("pending_slot"), "stage": after.get("stage"),
                                        "facts": after.get("facts", {}), "money_events": after.get("money_events", []),
                                        "money_corrections": after.get("money_corrections", []),
                                        "new_assistant_messages": [m for m in after.get("messages", [])[before_message_count:]
                                                                    if m.get("role") == "assistant"],
                                        "external_data_sent": api_result.get("external_data_sent"),
                                        "fallback_used": api_result.get("fallback_used"),
                                        "extraction_outcome": api_result.get("extraction_outcome"),
                                        "message_plan_applied": api_result.get("message_plan_applied", False),
                                        "model_turn_metrics": model_metrics,
                                        "model_merge_trace": api_result.get("model_merge_trace", []),
                                        "checks": checks,
                                        "passed": all(item["passed"] for item in checks)})
                    for check in checks:
                        system_checks += 1
                        system_passed += int(check["passed"])
                        if check["check"].startswith("citation_topic:") or check["check"] == "assistant_answer":
                            guidance_checks += 1
                            guidance_passed += int(check["passed"])
                        if check["check"].startswith("not_pending:"):
                            followup_checks += 1
                            followup_repeats += int(not check["passed"])
                    case_result["turns"].append(turn_result)
                    current = after
                case_result["passed"] = bool(case_result["turns"]) and all(t.get("passed", False) for t in case_result["turns"])
                results.append(case_result)

    total_turns = sum(len(case["turns"]) for case in results)
    passing_cases = sum(1 for case in results if case.get("passed"))
    percentiles = {
        "provider_latency_p50_ms": _provider_percentile(provider_latencies, 0.50),
        "provider_latency_p95_ms": _provider_percentile(provider_latencies, 0.95),
        "first_provider_request_ms": first_call_latency,
        "subsequent_provider_p50_ms": _provider_percentile(subsequent_call_latencies, 0.50),
        "subsequent_provider_p95_ms": _provider_percentile(subsequent_call_latencies, 0.95),
        "model_pipeline_elapsed_p50_ms": _provider_percentile(model_pipeline_latencies, 0.50),
        "integration_elapsed_p50_ms": _provider_percentile(integration_latencies, 0.50),
        "integration_elapsed_p95_ms": _provider_percentile(integration_latencies, 0.95),
        "turn_total_elapsed_p50_ms": _provider_percentile(total_turn_latencies, 0.50),
        "turn_total_elapsed_p95_ms": _provider_percentile(total_turn_latencies, 0.95),
    }
    return {
        "evaluation_version": "conversation-v3-2",
        "mode": mode,
        "suite_id": suite["suite_id"],
        "suite_version": suite["version"],
        "frozen_expectations_file": str(suite_path.relative_to(ROOT)),
        "synthetic_internal_cases": True,
        "frozen_before_application_fixes": bool(suite.get("frozen_before_application_fixes")),
        "expectations_locked_before_evaluation": bool(suite.get("expectations_locked_before_evaluation")),
        "suite_provenance": suite.get("provenance", "pre-fix frozen v3 suite"),
        "known_holdout_regressions_are_not_unseen": True,
        "provider_configured": configured,
        "provider": {k: v for k, v in provider.items() if k in {"provider", "model", "mode"}} if provider else None,
        "with_bounded_context": with_context,
        "model_eligible_turns": provider_eligible_turns,
        "provider_attempts": provider_attempts,
        "provider_fallback_rate": provider_fallbacks / provider_eligible_turns if provider_eligible_turns else None,
        "provider_accepted_responses": provider_accepted,
        "provider_rejected_responses": provider_rejected,
        "provider_extractions_with_accepted_facts": provider_extractions_with_facts,
        "provider_accepted_fact_count": provider_accepted_fact_count,
        "provider_fallbacks": provider_fallbacks,
        "provider_fallback_categories": fallback_categories,
        "provider_extraction_outcomes": extraction_outcomes,
        "accepted_fact_merge_dispositions": accepted_fact_merge_dispositions,
        "accepted_facts_ignored_by_merge": accepted_facts_ignored_by_merge,
        "accepted_facts_routed_to_rules": accepted_facts_routed_to_rules,
        "accepted_facts_applied_as_corrections": accepted_facts_applied_as_corrections,
        "accepted_facts_preserved_as_conflicts": accepted_facts_preserved_as_conflicts,
        "turns_with_accepted_facts_ignored_by_merge": turns_with_accepted_facts_ignored_by_merge,
        "integrated_system": {"passed_cases": passing_cases, "total_cases": len(results),
                              "case_rate": passing_cases / len(results) if results else None,
                              "passed_assertions": system_passed, "total_assertions": system_checks,
                              "assertion_rate": system_passed / system_checks if system_checks else None},
        "model_extraction_only": {"expected_fact_checks": model_expected_checks,
                                  "matched_facts": model_matches, "omitted_or_mismatched_facts": model_omissions,
                                  "extraction_match_rate": model_matches / model_expected_checks if model_expected_checks else None,
                                  "forbidden_field_checks": model_forbidden_checks,
                                  "unsupported_forbidden_proposals": model_forbidden},
        "guidance": {"answer_and_topic_checks_passed": guidance_passed,
                     "answer_and_topic_checks": guidance_checks,
                     "rate": guidance_passed / guidance_checks if guidance_checks else None},
        "repeat_question_checks": {"checked": followup_checks, "repeated_or_unresolved": followup_repeats},
        "latency": percentiles,
        "turn_count": total_turns,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen synthetic ACCP.AI Conversation v3 cases.")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--mode", choices=("rules", "model"), default="rules")
    parser.add_argument("--with-context", action="store_true", help="Permit explicitly consented bounded context in model mode.")
    parser.add_argument("--preflight-timeout", type=float, default=5.0)
    parser.add_argument("--replay-raw", type=Path, help="Re-score a saved synthetic raw artifact using current frozen assertions.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replay_raw:
        replayed = replay_raw(args.replay_raw)
        rendered = json.dumps(replayed, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(f"Saved: {args.output}")
        print(json.dumps({key: replayed[key] for key in ("source_artifact", "integrated_system", "guidance",
                                                        "repeat_question_checks", "model_extraction_only")},
                         ensure_ascii=False, indent=2))
        return 0 if replayed["integrated_system"]["case_rate"] == 1.0 else 1
    preflight = generation_test = None
    if args.mode == "model":
        preflight = probe_provider(timeout=max(0.1, args.preflight_timeout))
        if not preflight.get("configured") or not preflight.get("service_reachable") or not preflight.get("model_available"):
            report = {"evaluation_version": "conversation-v3-2", "mode": "model", "model_run_status": "not_run",
                      "provider_configured": bool(preflight.get("configured")), "provider": preflight,
                      "model_eligible_turns": 0, "provider_attempts": 0, "provider_fallback_rate": None,
                      "provider_accepted_responses": 0,
                      "provider_rejected_responses": 0, "provider_extractions_with_accepted_facts": 0,
                      "provider_accepted_fact_count": 0,
                      "provider_fallbacks": 0, "provider_fallback_categories": {},
                      "provider_extraction_outcomes": {}, "accepted_fact_merge_dispositions": {},
                      "accepted_facts_ignored_by_merge": 0, "accepted_facts_routed_to_rules": 0,
                      "accepted_facts_applied_as_corrections": 0, "accepted_facts_preserved_as_conflicts": 0,
                      "turns_with_accepted_facts_ignored_by_merge": 0,
                      "results": [],
                      "reason": preflight.get("failure_code") or "provider_preflight_failed",
                      "note": "No conversation extraction call was made; no model result is claimed."}
            rendered = json.dumps(report, ensure_ascii=False, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n", encoding="utf-8")
                print(f"Saved: {args.output}")
            print(rendered)
            return 2
        generation_test = test_provider_generation()
        if not generation_test.get("success"):
            report = {"evaluation_version": "conversation-v3-2", "mode": "model", "model_run_status": "not_run",
                      "provider_configured": True, "provider": preflight, "preflight_generation": generation_test,
                      "model_eligible_turns": 0, "provider_attempts": 0, "provider_fallback_rate": None,
                      "provider_accepted_responses": 0,
                      "provider_rejected_responses": 0, "provider_extractions_with_accepted_facts": 0,
                      "provider_accepted_fact_count": 0,
                      "provider_fallbacks": 0, "provider_fallback_categories": {},
                      "provider_extraction_outcomes": {}, "accepted_fact_merge_dispositions": {},
                      "accepted_facts_ignored_by_merge": 0, "accepted_facts_routed_to_rules": 0,
                      "accepted_facts_applied_as_corrections": 0, "accepted_facts_preserved_as_conflicts": 0,
                      "turns_with_accepted_facts_ignored_by_merge": 0,
                      "results": [],
                      "reason": generation_test.get("failure_code") or "preflight_generation_failed",
                      "note": "Fixed-text generation did not succeed; no conversation extraction call was made."}
            rendered = json.dumps(report, ensure_ascii=False, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n", encoding="utf-8")
                print(f"Saved: {args.output}")
            print(rendered)
            return 2
    report = run_suite(args.suite, args.mode, args.with_context)
    report["model_run_status"] = "completed" if args.mode == "model" else "not_applicable"
    if args.mode == "model":
        report["preflight"] = preflight
        report["preflight_generation"] = generation_test
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")
    print(json.dumps({key: report.get(key) for key in (
        "evaluation_version", "mode", "model_run_status", "preflight", "preflight_generation",
        "provider_configured", "provider", "model_eligible_turns", "provider_attempts", "provider_fallback_rate",
        "provider_accepted_responses", "provider_rejected_responses", "provider_extractions_with_accepted_facts",
        "provider_accepted_fact_count",
        "provider_fallbacks", "provider_fallback_categories", "provider_extraction_outcomes",
        "accepted_fact_merge_dispositions", "accepted_facts_ignored_by_merge", "accepted_facts_routed_to_rules",
        "accepted_facts_applied_as_corrections", "accepted_facts_preserved_as_conflicts",
        "turns_with_accepted_facts_ignored_by_merge", "integrated_system", "model_extraction_only",
        "guidance", "repeat_question_checks", "latency")},
        ensure_ascii=False, indent=2))
    return 0 if report["integrated_system"]["case_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
