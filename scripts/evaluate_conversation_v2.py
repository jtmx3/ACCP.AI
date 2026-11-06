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

from app.core import new_state, classify_current_danger, is_clear_user_question, _user_intent
from app.main import app
from app.store import Store
from app.conversation import build_bounded_context, interpret_message
from app.model import (ALLOWED_EXTRACTION_FIELDS, generate_interview_extraction,
                       last_call_trace, last_extraction_diagnostic, reset_call_trace)

ROOT = Path(__file__).resolve().parents[1]


def load_suite(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frozen = data.get("authored_before_logic_changes") or (
        data.get("authored_after_implementation") and data.get("phrasing_not_consulted_during_repairs"))
    if not frozen or not data.get("synthetic_internal_cases"):
        raise ValueError(f"Suite {path} is missing frozen synthetic provenance metadata")
    return data


def seed_case(store: Store, case_id: str, token: str, revision: int,
              seed_facts: dict[str, Any], pending_slot: str | None,
              money_events: list[dict[str, Any]]) -> None:
    def apply(state: dict[str, Any]) -> None:
        state["messages"] = [
            {"id": "m1", "role": "assistant", "text": "[synthetic setup question]", "created_at": state["created_at"], "kind": "question"},
            {"id": "m2", "role": "user", "text": "[synthetic setup facts]", "created_at": state["created_at"], "kind": "chat"},
            {"id": "m3", "role": "assistant", "text": "[synthetic pending question]", "created_at": state["created_at"], "kind": "question"},
        ]
        state["pending_slot"] = pending_slot
        state["stage"] = "interview"
        state["facts"] = {}
        for field, fact in seed_facts.items():
            state["facts"][field] = {
                "status": fact["status"], "value": fact.get("value"),
                "quote": fact.get("quote", "[synthetic setup fact]"),
                "message_id": "m2", "question_id": "m1", "updated_at": state["created_at"],
                "history": list(fact.get("history", [])),
            }
        state["money_events"] = []
        for item in money_events:
            event = dict(item)
            event.setdefault("message_id", "m2")
            state["money_events"].append(event)
    store.mutate(case_id, token, revision, "synthetic_evaluation_seed", apply)


def fact_assertion(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    for key in ("status", "value"):
        if key not in expected:
            continue
        wanted = expected[key]
        observed = actual.get(key)
        if key == "value" and isinstance(wanted, dict) and "contains" in wanted:
            if wanted["contains"] not in str(observed or ""):
                return False, f"{key}: expected substring {wanted['contains']!r}, got {observed!r}"
        elif observed != wanted:
            return False, f"{key}: expected {wanted!r}, got {observed!r}"
    return True, "matched"


def run_suite(path: Path, mode: str, provider: dict[str, Any],
              include_bounded_context: bool = False) -> dict[str, Any]:
    suite = load_suite(path)
    results: list[dict[str, Any]] = []
    total_fact_checks = passed_fact_checks = 0
    total_forbidden_checks = passed_forbidden_checks = 0
    total_dialogue_checks = passed_dialogue_checks = 0
    provider_attempts = model_nonempty_extractions = fallbacks = 0
    model_valid_empty = model_all_filtered = model_rejected = 0
    model_fact_checks = model_fact_matches = 0
    model_eligible_turns = model_proposals = 0
    model_forbidden_addition_checks = model_forbidden_additions = 0
    merge_applied = merge_applied_conflict = merge_agreed = merge_ignored = merge_routed = merge_superseded = 0
    provider_latencies: list[float] = []
    end_to_end_latencies: list[float] = []

    with tempfile.TemporaryDirectory(prefix="accp-conversation-v2-") as temp_dir:
        store = Store(str(Path(temp_dir) / "evaluation.sqlite3"))
        with patch("app.main.store", store), TestClient(app) as client:
            for case in suite["cases"]:
                created = client.post("/api/sessions")
                created.raise_for_status()
                session = created.json()["session"]
                case_id, token = session["session_id"], created.json()["token"]
                seed_case(store, case_id, token, session["revision"], case.get("seed_facts", {}),
                          case.get("pending_slot"), case.get("money_events", []))
                headers = {"X-Case-Token": token}
                session_result = client.get(f"/api/sessions/{case_id}", headers=headers)
                session_result.raise_for_status()
                revision = session_result.json()["session"]["revision"]
                case_steps: list[dict[str, Any]] = []
                for index, turn in enumerate(case["turns"], start=1):
                    before_resp = client.get(f"/api/sessions/{case_id}", headers=headers)
                    before_resp.raise_for_status()
                    before = before_resp.json()["session"]
                    before_facts = json.dumps(before.get("facts", {}), ensure_ascii=False, sort_keys=True)
                    before_messages = len(before.get("messages", []))
                    turn_kind = turn.get("kind", "answer")
                    extraction_eligible = (mode == "model" and turn_kind == "answer"
                        and not is_clear_user_question(turn["text"])
                        and _user_intent(turn["text"]) is None
                        and classify_current_danger(turn["text"], before.get("pending_slot")) is not True)
                    model_facts = None
                    model_diagnostic: dict[str, Any] = {}
                    model_trace: dict[str, Any] = {}
                    if extraction_eligible:
                        model_eligible_turns += 1
                        reset_call_trace()
                        context = (build_bounded_context(before, turn["text"], before.get("pending_slot"))
                                   if include_bounded_context else None)
                        model_facts = generate_interview_extraction(
                            turn["text"], before.get("pending_slot"), context,
                            interpret_message(turn["text"], before.get("pending_slot")))
                        model_trace = last_call_trace() or {}
                        model_diagnostic = last_extraction_diagnostic() or {}
                        if model_trace.get("attempted"):
                            provider_attempts += 1
                            if model_trace.get("latency_ms") is not None:
                                provider_latencies.append(float(model_trace["latency_ms"]))
                            outcome = model_trace.get("extraction_outcome")
                            if model_facts:
                                model_nonempty_extractions += 1
                            elif outcome == "valid_empty":
                                model_valid_empty += 1
                            elif outcome == "all_facts_filtered":
                                model_all_filtered += 1
                            elif outcome in {"provider_failed", "response_rejected"}:
                                model_rejected += 1
                        for field, wanted in turn.get("expect", {}).get("facts", {}).items():
                            if field not in ALLOWED_EXTRACTION_FIELDS:
                                continue
                            old_fact = before.get("facts", {}).get(field, {})
                            if fact_assertion(old_fact, wanted)[0]:
                                continue
                            model_fact_checks += 1
                            candidate = next((item for item in model_facts or [] if item.get("field") == field), {})
                            matched, _ = fact_assertion(candidate, wanted)
                            model_fact_matches += int(matched)
                        model_proposals += len(model_facts or [])
                        forbidden = set(turn.get("expect", {}).get("absent", []))
                        for field in forbidden:
                            model_forbidden_addition_checks += 1
                            model_forbidden_additions += int(any(item.get("field") == field for item in model_facts or []))
                        if turn.get("expect", {}).get("no_invented_amounts"):
                            model_forbidden_addition_checks += 1
                            numeric_amount = any(item.get("field") in {"requested_amount", "paid_amount"}
                                and any(char.isdigit() for char in str(item.get("value") or ""))
                                for item in model_facts or [])
                            model_forbidden_additions += int(numeric_amount)
                    payload = {
                        "expected_revision": before["revision"], "text": turn["text"],
                        "kind": turn_kind,
                        "allow_external": False,
                        "allow_external_extraction": extraction_eligible,
                        "allow_context_extraction": extraction_eligible and include_bounded_context,
                    }
                    start = time.perf_counter()
                    if extraction_eligible:
                        with patch("app.main.model_configured", return_value=True), \
                             patch("app.main.provider_config", return_value=provider), \
                             patch("app.main.generate_interview_extraction", return_value=model_facts), \
                             patch("app.main.last_call_trace", return_value=model_trace):
                            response = client.post(f"/api/sessions/{case_id}/messages", json=payload, headers=headers)
                    else:
                        response = client.post(f"/api/sessions/{case_id}/messages", json=payload, headers=headers)
                    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
                    end_to_end_latencies.append(elapsed_ms)
                    if response.status_code != 200:
                        case_steps.append({"turn": index, "text": turn["text"], "http_status": response.status_code,
                                           "failure": response.text, "passed": False})
                        total_dialogue_checks += 1
                        continue
                    result = response.json()
                    current = result["session"]
                    merge_trace = result.get("model_merge_trace", []) if extraction_eligible else []
                    for item in merge_trace:
                        disposition = item.get("disposition")
                        if disposition in {"applied", "applied_correction"}:
                            merge_applied += 1
                        elif disposition in {"applied_conflict", "conflict_created"}:
                            merge_applied += 1
                            merge_applied_conflict += 1
                        elif disposition == "agrees_with_rules":
                            merge_agreed += 1
                        elif disposition == "routed_to_rules":
                            merge_routed += 1
                        elif disposition == "superseded_by_rules":
                            merge_superseded += 1
                        elif disposition == "ignored":
                            merge_ignored += 1
                    expect = turn.get("expect", {})
                    checks: list[dict[str, Any]] = []

                    for field, wanted in expect.get("facts", {}).items():
                        total_fact_checks += 1
                        ok, detail = fact_assertion(current.get("facts", {}).get(field, {}), wanted)
                        passed_fact_checks += int(ok)
                        checks.append({"type": "fact", "field": field, "passed": ok, "detail": detail})
                    for field in expect.get("absent", []):
                        total_forbidden_checks += 1
                        ok = field not in current.get("facts", {})
                        passed_forbidden_checks += int(ok)
                        checks.append({"type": "forbidden_fact", "field": field, "passed": ok,
                                       "detail": "absent" if ok else "unexpected fact recorded"})
                    if expect.get("no_invented_amounts"):
                        total_forbidden_checks += 1
                        amount_values = [str(current.get("facts", {}).get(field, {}).get("value") or "")
                                         for field in ("requested_amount", "paid_amount")]
                        ok = all(not any(char.isdigit() for char in value) for value in amount_values)
                        passed_forbidden_checks += int(ok)
                        checks.append({"type": "no_invented_amounts", "passed": ok, "detail": amount_values})
                    if expect.get("source_quote_contains"):
                        for field, fragment in expect["source_quote_contains"].items():
                            total_fact_checks += 1
                            quote = current.get("facts", {}).get(field, {}).get("quote", "")
                            ok = fragment in quote
                            passed_fact_checks += int(ok)
                            checks.append({"type": "source_quote", "field": field, "passed": ok,
                                           "detail": "source quote retained" if ok else f"missing {fragment!r}"})
                    for field, entries in expect.get("history", {}).items():
                        for wanted in entries:
                            total_fact_checks += 1
                            history = current.get("facts", {}).get(field, {}).get("history", [])
                            ok = any(all(item.get(k) == value for k, value in wanted.items()) for item in history)
                            passed_fact_checks += int(ok)
                            checks.append({"type": "history", "field": field, "passed": ok,
                                           "detail": f"historical entry {wanted!r}"})
                    if "pending_slot" in expect:
                        total_dialogue_checks += 1
                        ok = current.get("pending_slot") == expect["pending_slot"]
                        passed_dialogue_checks += int(ok)
                        checks.append({"type": "pending_slot", "passed": ok,
                                       "detail": f"expected {expect['pending_slot']!r}, got {current.get('pending_slot')!r}"})
                    if "stage" in expect:
                        total_dialogue_checks += 1
                        ok = current.get("stage") == expect["stage"]
                        passed_dialogue_checks += int(ok)
                        checks.append({"type": "stage", "passed": ok, "detail": current.get("stage")})
                    if "last_assistant_kind" in expect:
                        total_dialogue_checks += 1
                        last = next((m for m in reversed(current.get("messages", [])) if m.get("role") == "assistant"), {})
                        ok = last.get("kind") == expect["last_assistant_kind"]
                        passed_dialogue_checks += int(ok)
                        checks.append({"type": "last_assistant_kind", "passed": ok, "detail": last.get("kind")})
                    if expect.get("single_question"):
                        total_dialogue_checks += 1
                        added = current.get("messages", [])[before_messages:]
                        questions = [m for m in added if m.get("role") == "assistant" and m.get("kind") in {"question", "clarification"}]
                        ok = len(questions) <= 1
                        passed_dialogue_checks += int(ok)
                        checks.append({"type": "single_question", "passed": ok,
                                       "detail": {"count": len(questions), "text": questions[-1].get("text") if questions else None}})
                    if expect.get("no_fact_change"):
                        total_forbidden_checks += 1
                        after_facts = json.dumps(current.get("facts", {}), ensure_ascii=False, sort_keys=True)
                        ok = before_facts == after_facts
                        passed_forbidden_checks += int(ok)
                        checks.append({"type": "no_fact_change", "passed": ok, "detail": "facts unchanged" if ok else "facts changed"})
                    if expect.get("question_answered"):
                        total_dialogue_checks += 1
                        added = current.get("messages", [])[before_messages:]
                        ok = any(m.get("role") == "assistant" and m.get("kind") == "answer" for m in added)
                        passed_dialogue_checks += int(ok)
                        checks.append({"type": "question_answered", "passed": ok,
                                       "detail": "answer recorded" if ok else "no answer message"})
                    relation_expect = expect.get("money_event_relation")
                    if relation_expect:
                        total_fact_checks += 1
                        ok = any(event.get("kind") == relation_expect.get("kind")
                                 and event.get("relation") == relation_expect.get("relation")
                                 and relation_expect.get("amount_contains", "") in str(event.get("amount", ""))
                                 for event in current.get("money_events", []))
                        passed_fact_checks += int(ok)
                        checks.append({"type": "money_event_relation", "passed": ok, "detail": relation_expect})
                    if "money_event_count" in expect:
                        total_forbidden_checks += 1
                        ok = len(current.get("money_events", [])) == expect["money_event_count"]
                        passed_forbidden_checks += int(ok)
                        checks.append({"type": "money_event_count", "passed": ok,
                                       "detail": len(current.get("money_events", []))})

                    provider_call = result.get("provider_call") or {}
                    if extraction_eligible and model_trace.get("attempted") and result.get("fallback_used"):
                        fallbacks += 1
                    safe_call = {key: provider_call.get(key) for key in (
                        "attempted", "provider", "model", "status", "latency_ms", "failure_code",
                        "extraction_outcome", "response_valid", "result_accepted", "fallback_used",
                        "fact_fields", "filtered_fact_count", "filtered_fact_fields",
                        "filtered_fact_code_counts", "context_sent", "context_char_count",
                        "context_fact_count", "context_message_count") if key in provider_call}
                    model_summary = ({
                        "attempted": bool(model_trace.get("attempted")),
                        "extraction_outcome": model_trace.get("extraction_outcome"),
                        "response_valid": model_trace.get("response_valid"),
                        "accepted_fact_fields": [item.get("field") for item in model_facts or []],
                        "filtered_fact_count": len(model_diagnostic.get("filtered_facts", [])),
                        "filtered_fact_code_counts": model_diagnostic.get("filtered_fact_code_counts", {}),
                        "merge_trace": merge_trace,
                    } if extraction_eligible else None)
                    case_steps.append({"turn": index, "text": turn["text"], "http_status": 200,
                                       "pending_slot": current.get("pending_slot"), "stage": current.get("stage"),
                                       "facts": current.get("facts"), "money_events": current.get("money_events"),
                                       "checks": checks, "passed": all(item["passed"] for item in checks),
                                       "request_latency_ms": elapsed_ms, "provider_call": safe_call,
                                       "model_extraction": model_summary,
                                       "fallback_used": result.get("fallback_used", False)})
                results.append({"case_id": case["id"], "passed": all(step["passed"] for step in case_steps), "steps": case_steps})

    def latency_stats(values: list[float]) -> dict[str, float | None]:
        return {"count": len(values), "median_ms": round(statistics.median(values), 3) if values else None,
                "p95_ms": round(sorted(values)[max(0, int(0.95 * len(values) + 0.999) - 1)], 3) if values else None}

    return {
        "suite_id": suite["suite_id"], "suite_path": str(path.relative_to(ROOT)), "mode": mode,
        "synthetic_internal_cases": True,
        "provider": provider.get("provider"), "model": provider.get("model"),
        "configured": provider.get("configuration_complete", provider.get("configured", False)),
        "conversation_integration_cases_passed": sum(bool(item["passed"]) for item in results),
        "case_denominator": len(results),
        "fact_checks_passed": passed_fact_checks, "fact_checks_total": total_fact_checks,
        "prohibited_inference_checks_passed": passed_forbidden_checks,
        "prohibited_inference_checks_total": total_forbidden_checks,
        "dialogue_checks_passed": passed_dialogue_checks, "dialogue_checks_total": total_dialogue_checks,
        "provider_calls_attempted": provider_attempts,
        "model_extractions_with_at_least_one_valid_fact": model_nonempty_extractions,
        "model_extraction_eligible_turns": model_eligible_turns,
        "model_valid_empty_responses": model_valid_empty,
        "model_responses_with_all_facts_filtered": model_all_filtered,
        "model_rejected_or_failed_responses": model_rejected,
        "model_expected_fact_matches": model_fact_matches,
        "model_expected_fact_checks": model_fact_checks,
        "model_expected_fact_omissions": model_fact_checks - model_fact_matches,
        "model_expected_fact_match_rate": (round(model_fact_matches / model_fact_checks, 4)
                                            if model_fact_checks else None),
        "model_valid_fact_proposals": model_proposals,
        "model_forbidden_addition_checks": model_forbidden_addition_checks,
        "model_forbidden_additions": model_forbidden_additions,
        "integrated_model_proposals_applied": merge_applied,
        "integrated_model_proposals_applied_as_conflicts": merge_applied_conflict,
        "integrated_model_proposals_agreeing_with_rule_facts": merge_agreed,
        "integrated_model_proposals_routed_to_rule_merger": merge_routed,
        "integrated_model_proposals_ignored": merge_ignored,
        "integrated_model_proposals_superseded_by_rules": merge_superseded,
        "integrated_api_fallbacks_not_counted_as_model_success": fallbacks,
        "provider_latency": latency_stats(provider_latencies),
        "end_to_end_request_latency": latency_stats(end_to_end_latencies),
        "first_provider_call_latency_ms": provider_latencies[0] if provider_latencies else None,
        "subsequent_provider_calls_latency": latency_stats(provider_latencies[1:]),
        "case_results": results,
        "limitations": "Synthetic internal evaluation; not an independent human review or representative user sample. Model extraction is called once per eligible turn and separately scored against eligible expected facts; the captured output is injected into the API merge path to avoid a duplicate provider call. Integrated case pass rate may include rules fallback and is never reported as model extraction success.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate staged Arabic conversation cases for ACCP.AI v2.")
    parser.add_argument("--mode", choices=("rules", "model"), default="rules")
    parser.add_argument("--suite", choices=("acceptance", "development", "holdout", "both"), default="both")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-timeout", type=float, default=5,
                        help="seconds for provider service/model availability preflight; extraction uses MODEL_TIMEOUT_SECONDS")
    parser.add_argument("--include-bounded-context", action="store_true",
                        help="separately authorize bounded synthetic prior-turn context; default sends current message only")
    args = parser.parse_args()

    from app.model import provider_config, probe_provider
    config = provider_config()
    provider_probe = (probe_provider(timeout=args.preflight_timeout)
                      if args.mode == "model" and config.get("configured") else None)
    preflight_ok = bool(config.get("configured")) and bool(provider_probe) and bool(
        provider_probe.get("service_reachable") and provider_probe.get("model_available"))
    if args.mode == "model" and not preflight_ok:
        payload = {
            "evaluation_version": "conversation-v2-2",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_commit": os.popen("git rev-parse HEAD").read().strip(),
            "mode": "model", "status": "not_run",
            "provider": {key: config.get(key) for key in
                          ("provider", "mode", "model", "configuration_complete", "base_url_host")},
            "provider_probe": ({key: provider_probe.get(key) for key in
                                ("service_reachable", "model_available", "generation_success", "failure_code")}
                               if provider_probe else None),
            "results": [],
            "limitations": "Model extraction was not evaluated: provider configuration or service/model preflight failed. No extraction calls were made and no result is attributed to the model.",
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        raise SystemExit(2)

    suite_names = ["acceptance", "development", "holdout"] if args.suite == "both" else [args.suite]
    payload = {
        "evaluation_version": "conversation-v2-2",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_commit": os.popen("git rev-parse HEAD").read().strip(),
        "worktree_dirty": bool(os.popen("git status --porcelain").read().strip()),
        "mode": args.mode, "provider_probe": provider_probe,
        "results": [run_suite(ROOT / "tests" / f"conversation_v2_{name}.json", args.mode, config,
                              include_bounded_context=args.include_bounded_context)
                    for name in suite_names],
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
