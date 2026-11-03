#!/usr/bin/env python3
"""Run the separately authored synthetic Arabic evaluation set against the local API."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app import main
from app.store import Store

DEFAULT_DATASET = ROOT / "tests" / "evaluation_scenarios.json"


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


def _run_case(client: TestClient, case: dict[str, Any], mode: str = "rules") -> dict[str, Any]:
    started = time.perf_counter()
    failures: list[str] = []
    timings: list[float] = []
    created = client.post("/api/sessions", json={})
    if created.status_code != 200:
        return {"id": case["id"], "passed": False, "failures": [f"session creation failed: HTTP {created.status_code}"], "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
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
            body = {
                "expected_revision": revision,
                "text": step["text"],
                "kind": step.get("kind", "answer"),
                "allow_external": False,
                "allow_external_extraction": False,
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

    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "id": case["id"],
        "category": case["category"],
        "passed": not failures,
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


def run(dataset_path: Path, output_path: Path, mode: str = "rules") -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset.get("classification", "").find("synthetic") < 0:
        raise ValueError("Evaluation dataset must be explicitly labeled synthetic.")
    with tempfile.TemporaryDirectory(prefix="accp-eval-") as tmp:
        main.store = Store(str(Path(tmp) / "evaluation.sqlite3"))
        with TestClient(main.app) as client:
            results = [_run_case(client, case, mode=mode) for case in dataset["cases"]]
    all_failures = [failure for result in results for failure in result["failures"]]
    latencies = [result["latency_ms"]["total"] for result in results]
    output = {
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["version"],
        "classification": dataset["classification"],
        "oracle_basis": dataset["oracle_basis"],
        "mode": mode,
        "provider_configured": False,
        "model_call_attempted": False,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "result": {
            "case_count": len(results),
            "cases_passed": sum(bool(result["passed"]) for result in results),
            "cases_failed": sum(not result["passed"] for result in results),
            "assertion_failure_count": len(all_failures),
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
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "evaluation" / "rules-final.json")
    parser.add_argument("--mode", choices=("rules",), default="rules", help="This runner currently measures local rules only; it does not invoke a model.")
    args = parser.parse_args()
    result = run(args.dataset, args.output, args.mode)
    print(json.dumps({"mode": result["mode"], **result["result"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0 if result["result"]["cases_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
