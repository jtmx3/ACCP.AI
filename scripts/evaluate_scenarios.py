#!/usr/bin/env python3
"""Evaluate the fixed synthetic Arabic corpus in separate rules or live-model modes."""
from __future__ import annotations

import argparse
import json
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

from app import main
from app.model import model_configured, probe_provider, provider_config, test_provider_generation
from app.store import Store

DEFAULT_DATASET = ROOT / "tests" / "evaluation_scenarios.json"
DEFAULT_RULES_OUTPUT = ROOT / "reports" / "evaluation" / "rules-final.json"


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


def _model_case_passed(integrated_passed: bool, expected_fact_matches: dict[str, bool],
                       provider_calls: list[dict[str, Any]]) -> bool:
    """Credit a scenario to the model only with valid model coverage and zero fallback calls."""
    if not integrated_passed or not provider_calls or not all(expected_fact_matches.values()):
        return False
    return all(call.get("result_accepted") is True and call.get("fallback_used") is False
               for call in provider_calls)


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
              candidate_log: list[dict[str, Any]] | None = None) -> dict[str, Any]:
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
            provider_calls.append({
                "step": index,
                "result_accepted": bool(call.get("result_accepted")),
                "fallback_used": bool(call.get("fallback_used")),
                "failure_code": call.get("failure_code"),
                "latency_ms": call.get("latency_ms"),
                "fact_fields": call.get("fact_fields", []),
                "provider": call.get("provider"),
                "model": call.get("model"),
            })
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
    expected_fact_matches = {}
    for field, expected in case.get("expected_facts", {}).items():
        matches = [fact for fact in candidate_facts if fact.get("field") == field and _candidate_matches(fact, expected)]
        expected_fact_matches[field] = bool(matches)
    uncovered_model_facts = [field for field, matched in expected_fact_matches.items() if not matched]
    rejected_calls = sum(not call["result_accepted"] for call in provider_calls)
    model_passed = bool(mode == "model" and _model_case_passed(integrated_passed, expected_fact_matches, provider_calls)
                         and not uncovered_model_facts)
    passed = integrated_passed if mode == "rules" else model_passed
    elapsed_ms = (time.perf_counter() - started) * 1000
    result = {
        "id": case["id"],
        "category": case["category"],
        "passed": passed,
        "integrated_passed": integrated_passed,
        "model_passed": model_passed,
        "failures": failures,
        "actual": {
            "stage": session.get("stage"),
            "pending_slot": session.get("pending_slot"),
            "priority_score": assessment.get("priority_score"),
            "priority_rule": assessment.get("priority_rule"),
            "facts": actual_facts,
            "expected_facts": {field: actual_facts.get(field) for field in case.get("expected_facts", {})},
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
            "results_accepted": len(provider_calls) - rejected_calls,
            "results_rejected": rejected_calls,
            "fallback_count": rejected_calls,
            "candidate_fact_count": len(candidate_facts),
            "candidate_facts": candidate_facts,
            "expected_fact_matches": expected_fact_matches,
            "expected_facts_not_covered_by_model": uncovered_model_facts,
            "fallback_success_counted_as_model_success": False,
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
    original_generator = main.generate_interview_extraction

    def capture_generator(text: str, pending_slot: str | None):
        facts = original_generator(text, pending_slot)
        candidate_log.append({"accepted": facts is not None, "facts": facts})
        return facts

    with tempfile.TemporaryDirectory(prefix="accp-eval-") as tmp:
        main.store = Store(str(Path(tmp) / "evaluation.sqlite3"))
        with TestClient(main.app) as client:
            if blocked_reason:
                results = []
            elif mode == "model":
                with patch("app.main.generate_interview_extraction", side_effect=capture_generator):
                    results = [_run_case(client, case, mode=mode, candidate_log=candidate_log) for case in dataset["cases"]]
            else:
                results = [_run_case(client, case, mode=mode) for case in dataset["cases"]]

    all_failures = [failure for result in results for failure in result["failures"]]
    latencies = [result["latency_ms"]["total"] for result in results]
    model_calls = [call for result in results for call in result.get("model_evaluation", {}).get("calls", [])]
    call_latencies = [call["latency_ms"] for call in model_calls if isinstance(call.get("latency_ms"), (int, float))]
    integrated_passes = sum(bool(result.get("integrated_passed")) for result in results)
    model_passes = sum(bool(result.get("model_passed")) for result in results)
    accepted_calls = sum(bool(call.get("result_accepted")) for call in model_calls)
    rejected_calls = len(model_calls) - accepted_calls
    run_status = "blocked" if blocked_reason else "completed"
    output = {
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["version"],
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
            "cases_passed": model_passes if mode == "model" else integrated_passes,
            "cases_failed": (len(results) - model_passes) if mode == "model" else (len(results) - integrated_passes),
            "integrated_cases_passed": integrated_passes,
            "model_cases_passed": model_passes if mode == "model" else None,
            "assertion_failure_count": len(all_failures),
            "provider_calls_attempted": len(model_calls),
            "provider_results_accepted": accepted_calls,
            "provider_results_rejected": rejected_calls,
            "fallback_count": rejected_calls,
            "fallback_successes_counted_as_model_success": 0,
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
