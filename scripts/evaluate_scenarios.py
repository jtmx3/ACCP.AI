#!/usr/bin/env python3
"""Evaluate the fixed synthetic Arabic corpus in separate rules or live-model modes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app import core, main
from app.model import (ALLOWED_EXTRACTION_FIELDS, last_extraction_diagnostic, model_configured,
                       probe_provider, provider_config, test_provider_generation)
from app.store import Store

DEFAULT_DATASET = ROOT / "tests" / "evaluation_scenarios.json"
DEFAULT_RULES_OUTPUT = ROOT / "reports" / "evaluation" / "rules-final.json"
MODEL_EVALUATION_VERSION = 5
MODEL_EVALUATION_FIELDS = frozenset(ALLOWED_EXTRACTION_FIELDS)
ENGINE_OWNED_FIELDS = frozenset({"direct_danger", "priority_score", "confirmed", "stage"})


def _is_reported_positive(fact: dict[str, Any] | None) -> bool:
    if not fact or fact.get("status") != "reported":
        return False
    value = fact.get("value")
    return value is True or not isinstance(value, bool)


def _check_fact(actual: dict[str, Any] | None, expected: dict[str, Any], field: str) -> list[str]:
    problems: list[str] = []
    if actual is None:
        return [f"missing expected fact: {field}"]
    if actual.get("status") != expected["status"]:
        problems.append(f"{field}.status expected {expected['status']!r}, got {actual.get('status')!r}")
    if "value" in expected and actual.get("value") != expected["value"]:
        problems.append(f"{field}.value expected {expected['value']!r}, got {actual.get('value')!r}")
    if "value_contains" in expected and expected["value_contains"] not in str(actual.get("value")):
        problems.append(f"{field}.value expected to contain {expected['value_contains']!r}, got {actual.get('value')!r}")
    return problems


def _candidate_matches(candidate: dict[str, Any], expected: dict[str, Any]) -> bool:
    if candidate.get("status") != expected.get("status"):
        return False
    if "value" in expected and candidate.get("value") != expected["value"]:
        return False
    if "value_contains" in expected and expected["value_contains"] not in str(candidate.get("value")):
        return False
    return True


def _legacy_composite_case_passed(integrated_passed: bool, expected_fact_matches: dict[str, bool],
                                  provider_calls: list[dict[str, Any]]) -> bool:
    """Retain the v2 comparator: all expected fields, integrated pass, and no fallback."""
    if not integrated_passed or not provider_calls or not all(expected_fact_matches.values()):
        return False
    return all(call.get("result_accepted") is True and call.get("fallback_used") is False
               and not call.get("filtered_fact_count")
               for call in provider_calls)


def _extraction_case_passed(eligible_expected_fact_matches: dict[str, bool],
                            has_forbidden_positive: bool) -> bool | None:
    """Score extraction only; excludes engine-owned fields and never consults integrated pass/fallback."""
    if not eligible_expected_fact_matches:
        return None
    return all(eligible_expected_fact_matches.values()) and not has_forbidden_positive


def _is_positive(fact: dict[str, Any]) -> bool:
    return fact.get("status") == "reported" and (
        fact.get("value") is True or not isinstance(fact.get("value"), bool))


def _candidate_oracle_class(candidate: dict[str, Any], expected: dict[str, Any], forbidden: set[str]) -> str:
    field = candidate.get("field")
    if field in forbidden and _is_positive(candidate):
        return "explicit_forbidden_positive"
    if field not in expected:
        return "oracle_unspecified"
    if _candidate_matches(candidate, expected[field]):
        return "matches_final_expectation"
    return "does_not_match_final_expectation"


def _safe_response_diagnostic(raw: str | None) -> dict[str, Any]:
    """Keep bounded diagnostics for fixed synthetic runs; redact credential-like strings."""
    if not isinstance(raw, str):
        return {"response_excerpt": None, "response_sha256": None, "response_truncated": False}
    redacted = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*", r"\1[REDACTED]", raw)
    redacted = re.sub(r"(?i)((?:\"?(?:api[_ -]?key|access[_ -]?token|secret|password)\"?)\s*[:=]\s*\"?)[^,\s}\]\"]+",
                      r"\1[REDACTED]", redacted)
    limit = 4000
    return {"response_excerpt": redacted[:limit],
            "response_sha256": hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest(),
            "response_truncated": len(redacted) > limit}


def _response_fact_proposals(raw: str | None) -> list[dict[str, Any]]:
    """Extract only fact-shaped fields from a rejected JSON response for synthetic diagnostics."""
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    facts = parsed.get("facts") if isinstance(parsed, dict) else None
    if not isinstance(facts, list):
        return []
    return [{key: item.get(key) for key in ("field", "status", "value", "quote") if key in item}
            for item in facts if isinstance(item, dict)]


def _check_events(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    remaining = list(actual)
    for requirement in expected:
        found = None
        for event in remaining:
            if event.get("kind") != requirement.get("kind") or event.get("relation") != requirement.get("relation"):
                continue
            amount = str(event.get("amount"))
            if requirement.get("amount_contains") not in amount:
                continue
            found = event
            break
        if found is None:
            problems.append(f"missing money event {requirement!r}; actual={actual!r}")
        else:
            remaining.remove(found)
    return problems


def _run_case(client: TestClient, case: dict[str, Any], mode: str = "rules",
              candidate_log: list[dict[str, Any]] | None = None,
              merge_log: list[dict[str, Any]] | None = None,
              evaluation_context: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    failures: list[str] = []
    timings: list[float] = []
    provider_calls: list[dict[str, Any]] = []
    candidate_start = len(candidate_log or [])
    created = client.post("/api/sessions", json={})
    if created.status_code != 200:
        return {"id": case["id"], "passed": False, "integrated_passed": False,
                "model_passed": False, "failures": [f"session creation failed: HTTP {created.status_code}"],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
    payload = created.json()
    session = payload["session"]
    session_id, token = session["session_id"], payload["token"]
    revision = session["revision"]
    headers = {"X-Case-Token": token}
    assistant_count_before = sum(1 for m in session.get("messages", []) if m.get("role") == "assistant")
    for index, step in enumerate(case["steps"], 1):
        if evaluation_context is not None:
            evaluation_context["step"] = index
        candidate_before = len(candidate_log or [])
        merge_before = len(merge_log or [])
        step_started = time.perf_counter()
        action = step["action"]
        if action == "message":
            kind = step.get("kind", "answer")
            body = {
                "expected_revision": revision,
                "text": step["text"],
                "kind": kind,
                "allow_external": mode == "model" and kind == "question",
                "allow_external_extraction": mode == "model" and kind == "answer",
            }
            response = client.post(f"/api/sessions/{session_id}/messages", headers=headers, json=body)
        elif action == "confirm":
            response = client.post(f"/api/sessions/{session_id}/confirm", headers=headers,
                                   json={"expected_revision": revision})
        elif action == "resume":
            response = client.post(f"/api/sessions/{session_id}/resume", headers=headers,
                                   json={"expected_revision": revision})
        else:
            failures.append(f"step {index}: unsupported action {action!r}")
            break
        timings.append((time.perf_counter() - step_started) * 1000)
        if response.status_code != 200:
            detail = response.json().get("detail", response.text)
            failures.append(f"step {index} {action}: HTTP {response.status_code}: {detail}")
            break
        result = response.json()
        session = result["session"]
        revision = session["revision"]
        call = result.get("provider_call") or {}
        if call.get("attempted"):
            entries = (candidate_log or [])[candidate_before:]
            merge_entries = (merge_log or [])[merge_before:]
            diagnostic = entries[-1].get("diagnostic", {}) if entries else {}
            call_record = {
                "step": index,
                "result_accepted": bool(call.get("result_accepted")),
                "fallback_used": bool(call.get("fallback_used")),
                "failure_code": call.get("failure_code"),
                "latency_ms": call.get("latency_ms"),
                "fact_fields": call.get("fact_fields", []),
                "provider": call.get("provider"),
                "model": call.get("model"),
                "json_valid": diagnostic.get("json_valid"),
                "schema_valid": diagnostic.get("schema_valid"),
                "rejection_code": diagnostic.get("rejection_code") or call.get("failure_code"),
                "rejection_detail": diagnostic.get("rejection_detail"),
                "filtered_fact_count": len(diagnostic.get("filtered_facts", [])),
                "filtered_fact_proposals": diagnostic.get("filtered_facts", []),
                "candidate_facts": [fact for entry in entries for fact in (entry.get("facts") or [])],
                "merge_outcomes": merge_entries,
            }
            if not call_record["result_accepted"]:
                excerpt = {key: diagnostic[key] for key in ("response_excerpt", "response_sha256", "response_truncated")
                           if key in diagnostic}
                call_record.update(excerpt)
                call_record["rejected_fact_proposals"] = diagnostic.get("rejected_fact_proposals", [])
            provider_calls.append(call_record)
        assistant_messages = [m for m in session.get("messages", []) if m.get("role") == "assistant"]
        new_messages = assistant_messages[assistant_count_before:]
        assistant_count_before = len(assistant_messages)
        emitted_questions = [m for m in new_messages if m.get("kind") == "question"]
        if len(emitted_questions) > 1:
            failures.append(f"step {index}: emitted {len(emitted_questions)} separate next-question messages")

    actual_facts = session.get("facts", {})
    for field, expected in case.get("expected_facts", {}).items():
        failures.extend(_check_fact(actual_facts.get(field), expected, field))
    for field in case.get("forbidden_positive", []):
        if _is_reported_positive(actual_facts.get(field)):
            failures.append(f"forbidden positive inference: {field}={actual_facts[field].get('value')!r}")

    expected_stage = case.get("expected_stage")
    if expected_stage is not None and session.get("stage") != expected_stage:
        failures.append(f"stage expected {expected_stage!r}, got {session.get('stage')!r}")
    expected_pending = case.get("expected_pending_slot")
    if "expected_pending_slot" in case and session.get("pending_slot") != expected_pending:
        failures.append(f"pending_slot expected {expected_pending!r}, got {session.get('pending_slot')!r}")
    assessment = session.get("assessment") or {}
    if "expected_priority_score" in case and assessment.get("priority_score") != case["expected_priority_score"]:
        failures.append(f"priority_score expected {case['expected_priority_score']!r}, got {assessment.get('priority_score')!r}")
    if "expected_priority_rule" in case and (assessment.get("priority_rule") or "none") != case["expected_priority_rule"]:
        failures.append(f"priority_rule expected {case['expected_priority_rule']!r}, got {(assessment.get('priority_rule') or 'none')!r}")

    expected_state = case.get("expected_state", {})
    if "confirmed" in expected_state and session.get("confirmed") is not expected_state["confirmed"]:
        failures.append(f"confirmed expected {expected_state['confirmed']!r}, got {session.get('confirmed')!r}")
    if "confirmation_history_count" in expected_state:
        actual = len(session.get("confirmation_history") or [])
        if actual != expected_state["confirmation_history_count"]:
            failures.append(f"confirmation_history_count expected {expected_state['confirmation_history_count']}, got {actual}")
    if "requested_amount_history_contains" in expected_state:
        history = actual_facts.get("requested_amount", {}).get("history", [])
        if not any(expected_state["requested_amount_history_contains"] in str(item.get("value")) for item in history):
            failures.append(f"requested_amount history did not retain {expected_state['requested_amount_history_contains']!r}")
    if "requested_amount_supporting_statement_contains" in expected_state:
        support = actual_facts.get("requested_amount", {}).get("supporting_statements", [])
        if not any(expected_state["requested_amount_supporting_statement_contains"] in str(item.get("quote")) for item in support):
            failures.append(f"requested_amount has no supporting statement containing {expected_state['requested_amount_supporting_statement_contains']!r}")
    if "money_events" in expected_state:
        failures.extend(_check_events(session.get("money_events", []), expected_state["money_events"]))

    expected_response = case.get("expected_response_contains")
    if expected_response:
        assistant_text = "\n".join(m.get("text", "") for m in session.get("messages", []) if m.get("role") == "assistant")
        if expected_response not in assistant_text:
            failures.append(f"no assistant response contained {expected_response!r}")

    report = client.get(f"/api/sessions/{session_id}/report.md", headers=headers)
    if report.status_code != 200:
        failures.append(f"report export failed: HTTP {report.status_code}")

    integrated_passed = not failures
    candidate_calls = (candidate_log or [])[candidate_start:]
    candidate_facts = [fact for call in candidate_calls for fact in (call.get("facts") or [])]
    expected_facts = case.get("expected_facts", {})
    all_expected_fact_matches = {}
    eligible_expected_fact_matches = {}
    excluded_expected_fields = {}
    for field, expected in expected_facts.items():
        matches = [fact for fact in candidate_facts if fact.get("field") == field and _candidate_matches(fact, expected)]
        all_expected_fact_matches[field] = bool(matches)
        if field in MODEL_EVALUATION_FIELDS:
            eligible_expected_fact_matches[field] = bool(matches)
        else:
            excluded_expected_fields[field] = "engine_owned_or_not_in_model_extraction_allowlist"
    uncovered_model_facts = [field for field, matched in eligible_expected_fact_matches.items() if not matched]
    forbidden = set(case.get("forbidden_positive", []))
    forbidden_candidates = [fact for fact in candidate_facts
                            if fact.get("field") in forbidden and _is_positive(fact)]
    oracle_classes = [_candidate_oracle_class(fact, expected_facts, forbidden) for fact in candidate_facts]
    oracle_class_counts = {label: oracle_classes.count(label) for label in sorted(set(oracle_classes))}
    rejected_proposals = [fact for call in provider_calls for fact in call.get("rejected_fact_proposals", [])]
    filtered_proposals = [fact for call in provider_calls for fact in call.get("filtered_fact_proposals", [])]
    all_response_proposals = candidate_facts + rejected_proposals + filtered_proposals
    proposal_classes = [_candidate_oracle_class(fact, expected_facts, forbidden) for fact in all_response_proposals]
    proposal_class_counts = {label: proposal_classes.count(label) for label in sorted(set(proposal_classes))}
    all_forbidden_positives = [fact for fact in all_response_proposals
                               if fact.get("field") in forbidden and _is_positive(fact)]
    extractive_case_passed = (_extraction_case_passed(eligible_expected_fact_matches, bool(forbidden_candidates))
                              if mode == "model" else None)
    legacy_composite_passed = (_legacy_composite_case_passed(
        integrated_passed, all_expected_fact_matches, provider_calls) if mode == "model" else None)
    rejected_calls = sum(not call["result_accepted"] for call in provider_calls)
    merge_outcomes = [event for call in provider_calls for event in call.get("merge_outcomes", [])]
    disposition_counts: dict[str, int] = {}
    for event in merge_outcomes:
        disposition_counts[event["disposition"]] = disposition_counts.get(event["disposition"], 0) + 1
    passed = integrated_passed if mode == "rules" else extractive_case_passed is True
    elapsed_ms = (time.perf_counter() - started) * 1000
    result = {
        "id": case["id"],
        "category": case["category"],
        "passed": passed,
        "integrated_passed": integrated_passed,
        "extraction_passed": extractive_case_passed,
        "legacy_v2_composite_passed": legacy_composite_passed,
        "failures": failures,
        "actual": {
            "stage": session.get("stage"),
            "pending_slot": session.get("pending_slot"),
            "priority_score": assessment.get("priority_score"),
            "priority_rule": assessment.get("priority_rule"),
            "facts": actual_facts,
            "expected_facts": {field: actual_facts.get(field) for field in expected_facts},
            "money_events": session.get("money_events", []),
            "confirmed": session.get("confirmed"),
            "confirmation_history_count": len(session.get("confirmation_history") or []),
            "question_messages": [m.get("text") for m in session.get("messages", []) if m.get("role") == "assistant" and m.get("kind") == "question"],
            "last_assistant_message": next((m.get("text") for m in reversed(session.get("messages", [])) if m.get("role") == "assistant"), None),
        },
        "latency_ms": {"total": round(elapsed_ms, 3), "per_step": [round(value, 3) for value in timings]},
    }
    if mode == "model":
        result["model_evaluation"] = {
            "calls": provider_calls,
            "calls_attempted": len(provider_calls),
            "responses_with_valid_json": sum(call.get("json_valid") is True for call in provider_calls),
            "responses_with_valid_schema": sum(call.get("schema_valid") is True for call in provider_calls),
            "validated_fact_calls_accepted": len(provider_calls) - rejected_calls,
            "results_rejected": rejected_calls,
            "fallback_count": sum(bool(call.get("fallback_used")) for call in provider_calls),
            "candidate_fact_count": len(candidate_facts),
            "candidate_facts": candidate_facts,
            "candidate_oracle_classes": oracle_class_counts,
            "candidate_expected_mismatches": oracle_class_counts.get("does_not_match_final_expectation", 0),
            "candidate_explicit_forbidden_positives": len(forbidden_candidates),
            "all_response_explicit_forbidden_positives": len(all_forbidden_positives),
            "rejected_fact_proposal_count": len(rejected_proposals),
            "filtered_fact_proposal_count": len(filtered_proposals),
            "filtered_fact_proposal_code_counts": {
                code: sum(1 for fact in filtered_proposals if fact.get("filter_code") == code)
                for code in sorted({fact.get("filter_code") for fact in filtered_proposals if fact.get("filter_code")})
            },
            "all_response_proposal_oracle_classes": proposal_class_counts,
            "eligible_expected_fact_matches": eligible_expected_fact_matches,
            "all_expected_fact_matches_legacy": all_expected_fact_matches,
            "eligible_expected_facts": sorted(eligible_expected_fact_matches),
            "excluded_expected_fields": excluded_expected_fields,
            "expected_facts_not_covered_by_model": uncovered_model_facts,
            "eligible_expected_fact_count": len(eligible_expected_fact_matches),
            "eligible_expected_facts_matched": sum(eligible_expected_fact_matches.values()),
            "merge_disposition_counts": disposition_counts,
            "merge_field_outcomes": merge_outcomes,
            "extraction_pass_does_not_depend_on_integrated_pass_or_fallback": True,
            "legacy_v2_composite_definition": "integrated pass AND all original expected fields matched (including engine-owned fields) AND every provider call accepted with no fallback and no per-fact suppression",
            "v5_filtering_policy": "whole-response rejection is reserved for malformed JSON or schema-level errors; individual quote, value, negation, field-evidence, and unsupported-status failures suppress only that fact; valid facts in the same response are retained; every suppressed proposal has a specific code and is never credited as an accepted fact",
        }
    return result


def _model_slug(text: str) -> str:
    clean = "".join(character.lower() if character.isalnum() else "-" for character in text).strip("-")
    while "--" in clean:
        clean = clean.replace("--", "-")
    return clean[:60] or "unnamed"


def run(dataset_path: Path, output_path: Path, mode: str = "rules") -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset.get("classification", "").find("synthetic") < 0:
        raise ValueError("Evaluation dataset must be explicitly labeled synthetic.")
    if mode not in {"rules", "model"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "model" and output_path.resolve() == DEFAULT_RULES_OUTPUT.resolve():
        raise ValueError("Model output must use a separate artifact path from rules-final.json.")

    config = provider_config() if mode == "model" else None
    preflight = probe_provider() if mode == "model" else None
    generation_test = None
    blocked_reason = None
    if mode == "model":
        if not model_configured():
            blocked_reason = "provider_not_configured"
        elif not preflight["service_reachable"] or not preflight["model_available"]:
            blocked_reason = preflight.get("failure_code") or "provider_preflight_failed"
        else:
            generation_test = test_provider_generation()
            if not generation_test["success"]:
                blocked_reason = generation_test.get("failure_code") or "generation_test_failed"

    candidate_log: list[dict[str, Any]] = []
    merge_log: list[dict[str, Any]] = []
    evaluation_context: dict[str, Any] = {"step": None}
    original_generator = main.generate_interview_extraction
    original_handle_turn = main.handle_turn

    def capture_generator(text: str, pending_slot: str | None):
        facts = original_generator(text, pending_slot)
        diagnostic = last_extraction_diagnostic() or {}
        entry = {"step": evaluation_context.get("step"), "accepted": facts is not None,
                 "facts": facts or [], "diagnostic": {k: v for k, v in diagnostic.items() if k != "raw_response"}}
        if facts is None and diagnostic.get("raw_response") is not None:
            entry["diagnostic"]["rejected_fact_proposals"] = _response_fact_proposals(diagnostic["raw_response"])
            entry["diagnostic"].update(_safe_response_diagnostic(diagnostic["raw_response"]))
        candidate_log.append(entry)
        return facts

    def capture_handle_turn(state, text: str, model_facts=None, **kwargs):
        result = original_handle_turn(state, text, model_facts=model_facts, **kwargs)
        for event in core.last_model_merge_trace():
            merge_log.append({"step": evaluation_context.get("step"), **event})
        return result

    with tempfile.TemporaryDirectory(prefix="accp-eval-") as tmp:
        main.store = Store(str(Path(tmp) / "evaluation.sqlite3"))
        with TestClient(main.app) as client:
            if blocked_reason:
                results = []
            elif mode == "model":
                with patch("app.main.generate_interview_extraction", side_effect=capture_generator), \
                     patch("app.main.handle_turn", side_effect=capture_handle_turn):
                    results = [_run_case(client, case, mode=mode, candidate_log=candidate_log,
                                         merge_log=merge_log, evaluation_context=evaluation_context)
                               for case in dataset["cases"]]
            else:
                results = [_run_case(client, case, mode=mode) for case in dataset["cases"]]

    all_failures = [failure for result in results for failure in result["failures"]]
    latencies = [result["latency_ms"]["total"] for result in results]
    model_calls = [call for result in results for call in result.get("model_evaluation", {}).get("calls", [])]
    call_latencies = [call["latency_ms"] for call in model_calls if isinstance(call.get("latency_ms"), (int, float))]
    integrated_passes = sum(bool(result.get("integrated_passed")) for result in results)
    extraction_passes = sum(result.get("extraction_passed") is True for result in results)
    extraction_denominator = sum(result.get("extraction_passed") is not None for result in results)
    legacy_passes = sum(result.get("legacy_v2_composite_passed") is True for result in results)
    accepted_calls = sum(bool(call.get("result_accepted")) for call in model_calls)
    rejected_calls = len(model_calls) - accepted_calls
    rejection_counts: dict[str, int] = {}
    for call in model_calls:
        if not call.get("result_accepted"):
            code = call.get("rejection_code") or "unclassified_failure"
            rejection_counts[code] = rejection_counts.get(code, 0) + 1
    json_response_calls = [call for call in model_calls if call.get("json_valid") is not None]
    schema_response_calls = [call for call in model_calls if call.get("schema_valid") is not None]
    eligible_expected_count = sum(item["model_evaluation"]["eligible_expected_fact_count"]
                                  for item in results if "model_evaluation" in item)
    eligible_matched_count = sum(item["model_evaluation"]["eligible_expected_facts_matched"]
                                 for item in results if "model_evaluation" in item)
    eligible_missing_count = eligible_expected_count - eligible_matched_count
    candidate_classes: dict[str, int] = {}
    response_proposal_classes: dict[str, int] = {}
    merge_dispositions: dict[str, int] = {}
    merge_rejection_codes: dict[str, int] = {}
    explicit_forbidden = 0
    all_response_forbidden = 0
    for case_result in results:
        evaluation = case_result.get("model_evaluation", {})
        explicit_forbidden += evaluation.get("candidate_explicit_forbidden_positives", 0)
        all_response_forbidden += evaluation.get("all_response_explicit_forbidden_positives", 0)
        for label, count in evaluation.get("candidate_oracle_classes", {}).items():
            candidate_classes[label] = candidate_classes.get(label, 0) + count
        for label, count in evaluation.get("all_response_proposal_oracle_classes", {}).items():
            response_proposal_classes[label] = response_proposal_classes.get(label, 0) + count
        for label, count in evaluation.get("merge_disposition_counts", {}).items():
            merge_dispositions[label] = merge_dispositions.get(label, 0) + count
        for call in evaluation.get("calls", []):
            for event in call.get("merge_outcomes", []):
                code = event.get("merge_rejection_code")
                if code:
                    merge_rejection_codes[code] = merge_rejection_codes.get(code, 0) + 1
    run_status = "blocked" if blocked_reason else "completed"
    output = {
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["version"],
        "evaluation_definition_version": MODEL_EVALUATION_VERSION,
        "evaluation_definition": {
            "model_extraction_eligible_fields": sorted(MODEL_EVALUATION_FIELDS),
            "excluded_expected_field_policy": "engine-owned safety/decision fields are not part of model-extraction coverage; notably direct_danger is classified and recorded by the safety engine, not proposed by interview extraction",
            "extraction_case_pass": "all expected facts in eligible fields match at least one validator-accepted model candidate and no explicitly forbidden positive candidate occurs; independent of integrated pass and fallback",
            "integrated_case_pass": "all original fixed scenario expectations pass against the application session, including safety/triage/stage/questions and rule fallback",
            "legacy_v2_composite": "integrated pass AND all original expected fields match candidates (including direct_danger) AND every provider call accepted with no fallback or per-fact suppression; retained only as historical comparator, not model accuracy",
            "candidate_oracle_limit": "unlisted candidate fields are oracle_unspecified, not assumed false; only explicit forbidden_positive entries count as explicit unsupported-positive candidates",
            "fact_filtering": "quote, value, negation, field-evidence, and unsupported-status proposals rejected per field are separately listed and included in all-response proposal counts; extraction coverage uses only validator-accepted facts",
        },
        "classification": dataset["classification"],
        "oracle_basis": dataset["oracle_basis"],
        "mode": mode,
        "provider_configured": bool(config and config.get("configured")),
        "provider": config,
        "provider_preflight": preflight,
        "provider_generation_test": generation_test,
        "run_status": run_status,
        "blocked_reason": blocked_reason,
        "model_call_attempted": bool(model_calls) or bool(generation_test and generation_test.get("attempted")),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "result": {
            "case_count": len(results),
            "cases_passed": extraction_passes if mode == "model" else integrated_passes,
            "cases_failed": ((extraction_denominator - extraction_passes) if mode == "model" else
                             (len(results) - integrated_passes)),
            "integrated_cases_passed": integrated_passes,
            "integrated_case_denominator": len(results),
            "extraction_cases_passed": extraction_passes if mode == "model" else None,
            "extraction_case_denominator": extraction_denominator if mode == "model" else None,
            "legacy_v2_composite_cases_passed": legacy_passes if mode == "model" else None,
            "legacy_v2_composite_denominator": len(results) if mode == "model" else None,
            "assertion_failure_count": len(all_failures),
            "provider_calls_attempted": len(model_calls),
            "provider_responses_with_valid_json": sum(call.get("json_valid") is True for call in model_calls),
            "provider_responses_with_valid_json_denominator": len(json_response_calls),
            "provider_responses_with_valid_schema": sum(call.get("schema_valid") is True for call in model_calls),
            "provider_responses_with_valid_schema_denominator": len(schema_response_calls),
            "validated_fact_calls_accepted": accepted_calls,
            "partially_filtered_provider_calls": sum(bool(call.get("filtered_fact_count")) for call in model_calls),
            "filtered_fact_proposal_count": sum(
                len(call.get("filtered_fact_proposals", [])) for call in model_calls),
            "filtered_fact_proposal_code_counts": {
                code: sum(1 for call in model_calls for fact in call.get("filtered_fact_proposals", [])
                         if fact.get("filter_code") == code)
                for code in sorted({fact.get("filter_code") for call in model_calls
                                    for fact in call.get("filtered_fact_proposals", [])
                                    if fact.get("filter_code")})
            },
            "provider_calls_rejected_or_failed": rejected_calls,
            "rejection_code_counts": rejection_counts,
            "fallback_count": sum(bool(call.get("fallback_used")) for call in model_calls),
            "eligible_expected_fact_instances": eligible_expected_count,
            "eligible_expected_fact_matches": eligible_matched_count,
            "eligible_expected_fact_omissions": eligible_missing_count,
            "eligible_expected_fact_match_rate": round(eligible_matched_count / eligible_expected_count, 6) if eligible_expected_count else None,
            "accepted_candidate_oracle_class_counts": candidate_classes,
            "explicit_forbidden_positive_candidate_count": explicit_forbidden,
            "all_response_proposal_oracle_class_counts": response_proposal_classes,
            "all_response_explicit_forbidden_positive_count": all_response_forbidden,
            "model_merge_disposition_counts": merge_dispositions,
            "merge_rejection_code_counts": merge_rejection_codes,
            "fallback_successes_counted_as_extraction_success": 0,
            "median_provider_latency_ms": round(statistics.median(call_latencies), 3) if call_latencies else None,
            "p95_provider_latency_ms": round(sorted(call_latencies)[min(len(call_latencies) - 1, int(len(call_latencies) * 0.95))], 3) if call_latencies else None,
            "median_case_latency_ms": round(statistics.median(latencies), 3) if latencies else None,
            "p95_case_latency_ms": round(sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))], 3) if latencies else None,
            "total_elapsed_ms": round(sum(latencies), 3),
        },
        "cases": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("rules", "model"), default="rules",
                        help="rules keeps local-only behavior; model requires a configured, reachable provider and explicit synthetic-message consent")
    args = parser.parse_args()
    if args.output is None:
        if args.mode == "rules":
            output_path = DEFAULT_RULES_OUTPUT
        else:
            config = provider_config()
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_path = ROOT / "reports" / "evaluation" / f"model-{_model_slug(config['provider'])}-{_model_slug(config['model'] or 'unconfigured')}-{stamp}.json"
    else:
        output_path = args.output
    result = run(args.dataset, output_path, args.mode)
    print(json.dumps({"mode": result["mode"], "run_status": result["run_status"],
                      "blocked_reason": result["blocked_reason"], **result["result"],
                      "provider": result.get("provider"), "model": (result.get("provider") or {}).get("model"),
                      "output": str(output_path)}, ensure_ascii=False, indent=2))
    return 0 if result["run_status"] == "completed" and result["result"]["cases_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
