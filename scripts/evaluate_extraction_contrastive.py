#!/usr/bin/env python3
"""تقييم contrastive لاستخراج النموذج عبر برومبت ACCP ومدققه، بلا دمج أو fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.fields import ALLOWED_EXTRACTION_FIELDS
from app.model import (generate_interview_extraction, last_call_trace,
                       last_extraction_diagnostic, probe_provider, provider_config,
                       reset_call_trace)

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
DEFAULT_SUITE = ROOT / "tests" / "extraction_contrastive_v1.json"


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def _fact_maps(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["field"]: item for item in items if isinstance(item, dict)
            and isinstance(item.get("field"), str)}


def _model_output_snapshot(raw_response: Any) -> dict[str, Any]:
    """Keep structured model proposals separate from validator output, without storing the raw response body."""
    if not isinstance(raw_response, str):
        return {"json_parsed": False, "response_sha256": None, "response_bytes": 0,
                "message_goals": None, "fact_proposals": []}
    encoded = raw_response.encode("utf-8", "replace")
    snapshot: dict[str, Any] = {"json_parsed": False,
                                "response_sha256": hashlib.sha256(encoded).hexdigest(),
                                "response_bytes": len(encoded), "message_goals": None,
                                "guidance_question": None, "fact_proposals": []}
    try:
        parsed = json.loads(raw_response)
    except (TypeError, ValueError):
        return snapshot
    if not isinstance(parsed, dict):
        return snapshot
    snapshot["json_parsed"] = True
    goals = parsed.get("message_goals")
    snapshot["message_goals"] = goals[:32] if isinstance(goals, list) else None
    guidance = parsed.get("guidance_question")
    snapshot["guidance_question"] = guidance[:400] if isinstance(guidance, str) else None
    proposals = parsed.get("facts")
    if isinstance(proposals, list):
        keys = ("field", "status", "value", "quote", "current_quote", "operation",
                "question_ref", "context_refs", "target_fact_id", "target_event_id")
        clean = []
        for item in proposals[:24]:
            if not isinstance(item, dict):
                clean.append({"proposal_type": type(item).__name__})
                continue
            row = {}
            for key in keys:
                value = item.get(key)
                if isinstance(value, str):
                    row[key] = value[:500]
                elif isinstance(value, list):
                    row[key] = [part[:160] if isinstance(part, str) else None for part in value[:12]]
                elif value is None or isinstance(value, (bool, int, float)):
                    row[key] = value
                elif key in item:
                    row[key] = {"unsupported_value_type": type(value).__name__}
            clean.append(row)
        snapshot["fact_proposals"] = clean
    return snapshot


def score_case(case: dict[str, Any], accepted_facts: list[dict[str, Any]],
               filtered_facts: list[dict[str, Any]], message_plan: dict[str, Any],
               diagnostic: dict[str, Any], trace: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    expected_goals = set(case.get("expected_goals", []))
    response_valid = diagnostic.get("accepted") is True
    supported_goals = set(message_plan.get("goals", [])) if response_valid else set()
    unsupported_goals = set(message_plan.get("unsupported_goals", [])) if response_valid else set()
    model_goals = supported_goals | unsupported_goals
    expected_facts = _fact_maps(case.get("expected_facts", []))
    accepted = _fact_maps(accepted_facts)
    filtered = _fact_maps(filtered_facts)

    fact_rows = []
    omitted_fields = []
    filtered_expected_fields = []
    for field, expected in expected_facts.items():
        actual = accepted.get(field)
        rejected = filtered.get(field)
        if actual is None:
            if rejected is not None:
                disposition = "filtered"
                filtered_expected_fields.append(field)
            elif not response_valid:
                disposition = "response_rejected"
            else:
                disposition = "omitted"
                omitted_fields.append(field)
        else:
            status_ok = actual.get("status") == expected.get("status")
            value_ok = actual.get("value") == expected.get("value")
            exact = status_ok and value_ok
            disposition = "accepted_match" if exact else "accepted_mismatch"
        fact_rows.append({"field": field, "expected_status": expected.get("status"),
                          "expected_value": expected.get("value"),
                          "expected_quote": expected.get("quote"),
                          "actual_status": actual.get("status") if actual else None,
                          "actual_value": actual.get("value") if actual else None,
                          "actual_quote": actual.get("quote") if actual else None,
                          "quote_match": actual.get("quote") == expected.get("quote") if actual else False,
                          "disposition": disposition,
                          "filter_code": rejected.get("filter_code") if rejected else None,
                          "filter_detail": rejected.get("filter_detail") if rejected else None})

    unexpected_accepted = sorted(set(accepted) - set(expected_facts))
    unexpected_proposals = [{"field": accepted[field].get("field"),
                             "status": accepted[field].get("status"),
                             "value": accepted[field].get("value"),
                             "quote": accepted[field].get("quote")}
                            for field in unexpected_accepted]
    goal_tp = len(expected_goals & model_goals) if response_valid else 0
    goal_fp = len(model_goals - expected_goals) if response_valid else 0
    goal_fn = len(expected_goals - model_goals) if response_valid else 0
    filtered_rows = [{"field": item.get("field"), "status": item.get("status"),
                      "value": item.get("value"), "filter_code": item.get("filter_code"),
                      "filter_detail": item.get("filter_detail")}
                     for item in filtered_facts if isinstance(item, dict)]
    provider_response_received = bool(diagnostic.get("raw_response")) or trace.get("status") == "empty_response"
    raw_model_output = _model_output_snapshot(diagnostic.get("raw_response"))
    outcome = trace.get("extraction_outcome") or (
        "extracted" if accepted else "all_facts_filtered" if filtered_facts else
        "valid_empty" if diagnostic.get("accepted") else "provider_failed")
    return {
        "case_id": case["id"], "pending_slot": case.get("pending_slot"),
        "message": case["message"], "outcome": outcome,
        "provider_attempted": bool(trace.get("attempted")),
        "provider_response_received": provider_response_received,
        "json_valid": diagnostic.get("json_valid"),
        "schema_valid": diagnostic.get("schema_valid"),
        "response_valid": response_valid,
        "provider_latency_ms": trace.get("latency_ms"), "total_elapsed_ms": round(elapsed_ms, 3),
        "request_body_bytes": trace.get("request_body_bytes"),
        "prompt_char_count": trace.get("prompt_char_count"),
        "provider_attempt_count": trace.get("provider_attempt_count", int(bool(trace.get("attempted")))),
        "model_raw_output": raw_model_output,
        "validator_accepted_facts": accepted_facts,
        "validator_filtered_proposals": filtered_rows,
        "application_normalizations": [],
        "goals": {"scored": response_valid, "expected": sorted(expected_goals),
                  "model_output": sorted(model_goals) if response_valid else None,
                  "supported": sorted(supported_goals), "unsupported": sorted(unsupported_goals),
                  "exact_match": model_goals == expected_goals if response_valid else None,
                  "true_positive": sorted(expected_goals & model_goals) if response_valid else None,
                  "false_positive": sorted(model_goals - expected_goals) if response_valid else None,
                  "false_negative": sorted(expected_goals - model_goals) if response_valid else None},
        "facts": {"expected_count": len(expected_facts), "accepted_count": len(accepted_facts),
                  "filtered_count": len(filtered_facts), "omitted_fields": omitted_fields,
                  "filtered_expected_fields": filtered_expected_fields,
                  "unexpected_accepted_fields": unexpected_accepted,
                  "unexpected_accepted_facts": unexpected_proposals,
                  "field_results": fact_rows, "filtered_proposals": filtered_rows,
                  "proposed_fact_count": len(accepted_facts) + len(filtered_facts)},
        "validation_rejection_code": diagnostic.get("rejection_code"),
        "validation_rejection_detail": diagnostic.get("rejection_detail"),
    }


def _load_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(suite, dict) or suite.get("synthetic") is not True:
        raise ValueError("لا يُشغّل هذا التقييم إلا ملف سيناريوهات معلنًا كبيانات اصطناعية.")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("ملف السيناريوهات فارغ أو غير صالح.")
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str)
                or not isinstance(case.get("message"), str)
                or case.get("pending_slot") not in ALLOWED_EXTRACTION_FIELDS
                or not isinstance(case.get("expected_goals"), list)
                or not isinstance(case.get("expected_facts"), list)):
            raise ValueError("هناك حالة ناقصة الحقول أو تتجاوز مخطط مجموعة التقييم.")
        for fact in case["expected_facts"]:
            if (not isinstance(fact, dict) or fact.get("field") not in ALLOWED_EXTRACTION_FIELDS
                    or fact.get("quote") not in case["message"]):
                raise ValueError(f"توقع حقل أو اقتباس غير مسند في الحالة {case['id']}.")
    return suite


def evaluate_suite(suite_path: Path = DEFAULT_SUITE, *, probe_timeout: float = 5.0) -> dict[str, Any]:
    suite = _load_suite(suite_path)
    config = provider_config()
    if config.get("provider") != "ollama" or not config.get("local"):
        raise RuntimeError("التقييم الحي يقبل إعداد Ollama المحلي فقط.")
    if not config.get("configured") or config.get("base_url_host") not in LOOPBACK_HOSTS:
        raise RuntimeError("إعداد Ollama غير مكتمل أو لا يشير إلى loopback؛ لم يُرسل أي طلب.")

    probe = probe_provider(timeout=probe_timeout)
    output: dict[str, Any] = {
        "evaluation_type": "model_extraction_only_contrastive",
        "evaluation_version": suite.get("edition"), "scoring_version": "2.0",
        "prompt_contract_version": "minimal-reference-ops-v2", "suite_id": suite.get("suite_id"),
        "synthetic": True, "provider": "ollama", "model": config.get("model"),
        "reasoning_effort": config.get("ollama_reasoning_effort"),
        "service_reachable": bool(probe.get("service_reachable")),
        "model_available": bool(probe.get("model_available")),
        "probe_elapsed_ms": probe.get("elapsed_ms"), "probe_failure_code": probe.get("failure_code"),
        "prompt_source": "app.model.generate_interview_extraction",
        "validator_source": "app.model.validate_interview_facts_diagnostic",
        "integrated_system_evaluation": False, "rule_merge_performed": False,
        "fallback_performed": False, "raw_provider_responses_saved": False,
        "raw_model_proposals_saved": True,
        "case_count": len(suite["cases"]), "cases": [],
    }
    if not probe.get("service_reachable") or not probe.get("model_available"):
        output["run_status"] = "not_run"
        output["cases"] = [{"case_id": case["id"], "outcome": "not_run",
                             "provider_attempted": False} for case in suite["cases"]]
        output["summary"] = {"attempted_cases": 0, "reason": probe.get("failure_code") or "provider_unavailable"}
        return output

    results: list[dict[str, Any]] = []
    for case in suite["cases"]:
        reset_call_trace()
        started = time.perf_counter()
        try:
            result = generate_interview_extraction(case["message"], case["pending_slot"],
                                                   case.get("bounded_context"))
            elapsed_ms = (time.perf_counter() - started) * 1000
            diagnostic = last_extraction_diagnostic() or {}
            trace = last_call_trace() or {}
            facts = list(result) if result is not None else []
            plan = getattr(result, "message_plan", diagnostic.get("message_plan", {})) if result is not None else diagnostic.get("message_plan", {})
            plan = plan if isinstance(plan, dict) else {}
            filtered = diagnostic.get("filtered_facts", [])
            results.append(score_case(case, facts, filtered if isinstance(filtered, list) else [],
                                      plan, diagnostic, trace, elapsed_ms))
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            trace = last_call_trace() or {}
            diagnostic = last_extraction_diagnostic() or {}
            results.append({"case_id": case["id"], "pending_slot": case.get("pending_slot"),
                            "message": case["message"], "outcome": "evaluation_error",
                            "provider_attempted": bool(trace.get("attempted")),
                            "provider_latency_ms": trace.get("latency_ms"),
                            "total_elapsed_ms": round(elapsed_ms, 3),
                            "error_type": type(exc).__name__,
                            "validation_rejection_code": diagnostic.get("rejection_code")})

    scored_cases = [item for item in results if item.get("goals", {}).get("scored")]
    scored_case_ids = {item["case_id"] for item in scored_cases}
    expected_goal_count = sum(len(case.get("expected_goals", [])) for case in suite["cases"]
                              if case["id"] in scored_case_ids)
    unscored_goal_count = sum(len(case.get("expected_goals", [])) for case in suite["cases"]
                              if case["id"] not in scored_case_ids)
    all_goal_tp = sum(len(item.get("goals", {}).get("true_positive") or []) for item in results)
    all_goal_fp = sum(len(item.get("goals", {}).get("false_positive") or []) for item in results)
    all_goal_fn = sum(len(item.get("goals", {}).get("false_negative") or []) for item in results)
    expected_fact_count = sum(len(case.get("expected_facts", [])) for case in suite["cases"])
    fact_rows = [row for item in results for row in item.get("facts", {}).get("field_results", [])]
    scored_expected_fact_count = sum(item.get("facts", {}).get("expected_count", 0)
                                     for item in results if item.get("response_valid"))
    unscored_expected_fact_count = expected_fact_count - scored_expected_fact_count
    latencies = [item["provider_latency_ms"] for item in results
                 if isinstance(item.get("provider_latency_ms"), (int, float))]
    outcomes = [item.get("outcome") for item in results]
    filter_code_counts: dict[str, int] = {}
    for item in results:
        for proposal in item.get("facts", {}).get("filtered_proposals", []):
            code = proposal.get("filter_code") or "unspecified"
            filter_code_counts[code] = filter_code_counts.get(code, 0) + 1
    rejection_code_counts: dict[str, int] = {}
    for item in results:
        code = item.get("validation_rejection_code")
        if code:
            rejection_code_counts[code] = rejection_code_counts.get(code, 0) + 1
    accepted_fact_count = sum(item.get("facts", {}).get("accepted_count", 0) for item in results)
    exact_fact_match_count = sum(row.get("disposition") == "accepted_match" for row in fact_rows)
    exact_fact_and_quote_match_count = sum(row.get("disposition") == "accepted_match" and row.get("quote_match")
                                           for row in fact_rows)
    unexpected_accepted_count = sum(len(item.get("facts", {}).get("unexpected_accepted_fields", []))
                                    for item in results)
    quote_match_count = sum(bool(row.get("quote_match")) for row in fact_rows)
    response_bytes = [item["model_raw_output"]["response_bytes"] for item in results
                      if item.get("model_raw_output", {}).get("response_bytes")]
    request_sizes = [item["request_body_bytes"] for item in results
                     if isinstance(item.get("request_body_bytes"), int)]
    request_prompt_chars = [item["prompt_char_count"] for item in results
                            if isinstance(item.get("prompt_char_count"), int)]
    call_attempts = sum(item.get("provider_attempt_count", 0) for item in results)
    output["cases"] = results
    output["run_status"] = "completed"
    output["summary"] = {
        "attempted_cases": sum(bool(item.get("provider_attempted")) for item in results),
        "provider_attempt_count": call_attempts,
        "provider_attempts_per_case": _ratio(call_attempts, len(results)),
        "provider_responses_received": sum(bool(item.get("provider_response_received")) for item in results),
        "json_valid_responses": sum(item.get("json_valid") is True for item in results),
        "schema_valid_responses": sum(item.get("schema_valid") is True for item in results),
        "provider_responses_valid": sum(bool(item.get("response_valid")) for item in results),
        "goal_scored_cases": len(scored_cases), "goal_unscored_cases": len(results) - len(scored_cases),
        "unscored_expected_goal_count": unscored_goal_count,
        "goal_exact_case_matches": sum(item.get("goals", {}).get("exact_match") is True for item in scored_cases),
        "goal_true_positive": all_goal_tp, "goal_false_positive": all_goal_fp,
        "goal_false_negative": all_goal_fn,
        "goal_precision": _ratio(all_goal_tp, all_goal_tp + all_goal_fp),
        "goal_recall": _ratio(all_goal_tp, all_goal_tp + all_goal_fn),
        "expected_goal_count": expected_goal_count,
        "expected_fact_count": expected_fact_count,
        "scored_expected_fact_count": scored_expected_fact_count,
        "unscored_expected_fact_count": unscored_expected_fact_count,
        "fact_field_status_matches": sum(row.get("actual_status") == row.get("expected_status")
                                          and row.get("disposition", "").startswith("accepted") for row in fact_rows),
        "fact_field_status_value_matches": exact_fact_match_count,
        "fact_field_status_accuracy": _ratio(sum(row.get("actual_status") == row.get("expected_status")
                                                  and row.get("disposition", "").startswith("accepted") for row in fact_rows),
                                              scored_expected_fact_count),
        "fact_field_status_value_accuracy": _ratio(sum(row.get("disposition") == "accepted_match" for row in fact_rows),
                                                    scored_expected_fact_count),
        "fact_field_status_value_end_to_end_coverage": _ratio(exact_fact_match_count, expected_fact_count),
        "fact_quote_matches": quote_match_count,
        "fact_quote_accuracy": _ratio(quote_match_count, scored_expected_fact_count),
        "fact_field_full_status_value_quote_matches": exact_fact_and_quote_match_count,
        "fact_field_full_status_value_quote_coverage": _ratio(exact_fact_and_quote_match_count,
                                                              expected_fact_count),
        "accepted_fact_count": accepted_fact_count,
        "accepted_fact_exact_match_precision": _ratio(exact_fact_match_count, accepted_fact_count),
        "accepted_fact_full_match_precision": _ratio(exact_fact_and_quote_match_count, accepted_fact_count),
        "unexpected_accepted_field_count": unexpected_accepted_count,
        "omitted_expected_fact_count": sum(len(item.get("facts", {}).get("omitted_fields", [])) for item in results),
        "filtered_expected_fact_count": sum(len(item.get("facts", {}).get("filtered_expected_fields", [])) for item in results),
        "response_rejected_expected_fact_count": sum(
            sum(row.get("disposition") == "response_rejected" for row in item.get("facts", {}).get("field_results", []))
            for item in results),
        "filtered_proposal_count": sum(item.get("facts", {}).get("filtered_count", 0) for item in results),
        "filtered_proposals_by_code": filter_code_counts,
        "response_rejections_by_code": rejection_code_counts,
        "valid_empty_cases": outcomes.count("valid_empty"),
        "all_facts_filtered_cases": outcomes.count("all_facts_filtered"),
        "cases_with_accepted_facts": sum(item.get("facts", {}).get("accepted_count", 0) > 0 for item in results),
        "provider_latency_ms_median": round(statistics.median(latencies), 3) if latencies else None,
        "provider_latency_ms_p95": _percentile(latencies, 0.95),
        "request_body_bytes_total": sum(request_sizes),
        "request_body_bytes_median": round(statistics.median(request_sizes), 3) if request_sizes else None,
        "request_body_bytes_p95": _percentile(request_sizes, 0.95),
        "prompt_char_count_median": round(statistics.median(request_prompt_chars), 3) if request_prompt_chars else None,
        "model_response_bytes_total": sum(response_bytes),
        "model_response_bytes_median": round(statistics.median(response_bytes), 3) if response_bytes else None,
        "integrated_system_metrics_included": False,
        "rules_fallback_success_counted_as_model_success": False,
        "application_normalization_or_repair_count": 0,
    }
    return output


def save_result(result: dict[str, Any], requested_path: Path | None = None) -> Path:
    if requested_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        requested_path = ROOT / "reports" / "evaluation" / f"extraction-contrastive-{stamp}-{uuid.uuid4().hex[:8]}.json"
    path = requested_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    result["result_file"] = str(path)
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def evaluation_passes(summary: dict[str, Any], case_count: int) -> bool:
    """Require complete scoring; a partial/unscored run is not a passing evaluation."""
    return (
        summary.get("attempted_cases") == case_count
        and summary.get("provider_responses_valid") == case_count
        and summary.get("goal_scored_cases") == case_count
        and summary.get("goal_exact_case_matches") == case_count
        and summary.get("goal_false_positive") == 0
        and summary.get("goal_false_negative") == 0
        and summary.get("unscored_expected_fact_count") == 0
        and summary.get("fact_field_status_value_end_to_end_coverage") == 1.0
        and summary.get("fact_quote_accuracy") == 1.0
        and summary.get("accepted_fact_exact_match_precision") == 1.0
        and summary.get("accepted_fact_full_match_precision") == 1.0
        and summary.get("unexpected_accepted_field_count") == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE,
                        help="مجموعة اصطناعية ثابتة؛ الافتراضي tests/extraction_contrastive_v1.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="ملف JSON جديد؛ افتراضيًا ينشئ اسمًا فريدًا في reports/evaluation")
    parser.add_argument("--probe-timeout", type=float, default=5.0)
    args = parser.parse_args()
    try:
        result = evaluate_suite(args.suite, probe_timeout=args.probe_timeout)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        result = {"evaluation_type": "model_extraction_only_contrastive", "synthetic": True,
                  "run_status": "not_run", "error": str(exc), "integrated_system_evaluation": False,
                  "fallback_performed": False, "raw_provider_responses_saved": False}
    try:
        path = save_result(result, args.output)
    except OSError as exc:
        print(json.dumps({"error": "تعذر حفظ ملف النتائج محليًا.", "error_type": type(exc).__name__}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("run_status") != "completed":
        return 1
    summary = result.get("summary", {})
    return 0 if evaluation_passes(summary, result.get("case_count", 0)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
