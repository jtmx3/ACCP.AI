from __future__ import annotations

import argparse
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app import main
from app.model import probe_provider, provider_config, test_provider_generation
from app.store import Store

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROVIDER = "ollama"
EXPECTED_MODEL = "qwen3.5:4b-q4_K_M"

# These are synthetic smoke cases. They are not an accuracy benchmark and contain no user data.
SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "generic-security-no-platform",
        "text": "بدي نصيحة أحافظ فيها على حسابي.",
        "prior_platform": None,
        "expected": "clarify_platform",
        "requires_model_goal": "ask_guidance",
    },
    {
        "id": "explicit-whatsapp-security",
        "text": "كيف أحمي حسابي على واتساب؟",
        "prior_platform": None,
        "expected": "whatsapp_security",
    },
    {
        "id": "previous-whatsapp-platform-context",
        "text": "بدي نصيحة أحافظ فيها على حسابي.",
        "prior_platform": "المشكلة صارت على واتساب.",
        "expected": "whatsapp_security",
        "requires_model_goal": "ask_guidance",
        "requires_context": True,
    },
    {
        "id": "current-platform-switch-overrides-prior",
        "text": "كيف أحمي حسابي على واتساب؟",
        "prior_platform": "المشكلة صارت على إنستغرام.",
        "expected": "whatsapp_security",
    },
    {
        "id": "ambiguous-two-platforms",
        "text": "كيف أحمي حسابي على واتساب وإنستغرام؟",
        "prior_platform": None,
        "expected": "clarify_platform",
    },
    {
        "id": "unrelated-topic-abstains",
        "text": "كيف أعمل كعكة شوكولاتة؟",
        "prior_platform": None,
        "expected": "abstain",
    },
]

TRACE_FIELDS = (
    "attempted", "status", "failure_code", "latency_ms", "provider", "model",
    "result_accepted", "fallback_used", "context_sent", "context_char_count",
    "context_fact_count", "context_message_count", "message_plan_applied", "message_goals",
)


def _safe_provider_summary(config: dict[str, Any]) -> dict[str, Any]:
    keys = ("provider", "mode", "model", "base_url_host", "configuration_complete",
            "ollama_reasoning_effort")
    return {key: config.get(key) for key in keys}


def _safe_probe_summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("configured", "service_reachable", "model_available", "models_count", "failure_code",
            "provider", "model")
    return {key: result.get(key) for key in keys}


def _safe_generation_summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("success", "attempted", "latency_ms", "failure_code", "provider", "model")
    return {key: result.get(key) for key in keys}


def _output_path(output_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested if requested.is_absolute() else ROOT / requested
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"guidance-live-{stamp}-{uuid.uuid4().hex[:10]}.json"


def _write_new_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects every prior run, including when the caller supplied --output.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _send(client: TestClient, case_id: str, token: str, revision: int, text: str,
          *, allow_model: bool, allow_context: bool) -> dict[str, Any]:
    response = client.post(
        f"/api/sessions/{case_id}/messages",
        headers={"X-Case-Token": token},
        json={
            "expected_revision": revision,
            "text": text,
            "kind": "answer",
            "allow_external": allow_model,
            "allow_external_extraction": allow_model,
            "allow_context_extraction": allow_context,
        },
    )
    if response.status_code != 200:
        raise RuntimeError(f"message request returned HTTP {response.status_code}")
    return response.json()


def _evaluate_result(scenario: dict[str, Any], result: dict[str, Any],
                     seed_platform_ok: bool | None) -> dict[str, Any]:
    session = result.get("session", {})
    messages = session.get("messages", [])
    answers = [item for item in messages if item.get("role") == "assistant" and item.get("kind") == "answer"]
    answer = answers[-1] if answers else {}
    answer_text = str(answer.get("text") or "")
    citations = answer.get("citations") or []
    citation_ids = sorted({str(item.get("id")) for item in citations if item.get("id")})
    model_plan = result.get("message_plan") or {}
    goals = model_plan.get("goals", []) if isinstance(model_plan, dict) else []
    trace = result.get("provider_call") or {}
    safe_trace = {key: trace.get(key) for key in TRACE_FIELDS if key in trace}
    checks: dict[str, bool] = {
        "assistant_answer_present": bool(answer_text),
        "seed_platform_recorded": seed_platform_ok is not False,
    }
    required_goal = scenario.get("requires_model_goal")
    if required_goal:
        checks["model_plan_applied"] = bool(result.get("message_plan_applied"))
        checks[f"model_goal:{required_goal}"] = required_goal in goals
    if scenario.get("requires_context"):
        context_sent = bool(result.get("external_context_sent", trace.get("context_sent")))
        checks["bounded_context_sent"] = context_sent
        checks["bounded_context_has_seeded_fact"] = bool(trace.get("context_fact_count", 0))
    expected = scenario["expected"]
    if expected == "clarify_platform":
        checks["clarifies_platform"] = any(marker in answer_text for marker in ("أي تطبيق", "أي منصة", "تحديد التطبيق"))
        checks["no_platform_mismatch_citation"] = not citations
    elif expected == "whatsapp_security":
        checks["has_whatsapp_source"] = any(source_id.startswith("GLOBAL-WA-") for source_id in citation_ids)
        checks["excludes_instagram_source"] = "GLOBAL-IG-SECURITY-2026" not in citation_ids
    elif expected == "instagram_security":
        checks["has_instagram_source"] = "GLOBAL-IG-SECURITY-2026" in citation_ids
        checks["excludes_whatsapp_source"] = not any(source_id.startswith("GLOBAL-WA-") for source_id in citation_ids)
    elif expected == "abstain":
        checks["no_citations"] = not citations
        checks["abstains_without_source"] = any(marker in answer_text for marker in (
            "لا يوجد في قاعدة المعرفة", "ما عندي مصدر موثوق", "لا أستطيع تقديم إرشاد"))

    public_citations = [
        {key: source.get(key) for key in ("id", "title", "publisher", "url", "geography", "platforms", "topics", "excerpt") if key in source}
        for source in citations
    ]
    return {
        "scenario_id": scenario["id"],
        "synthetic_user_text": scenario["text"],
        "synthetic_prior_platform_message": scenario.get("prior_platform"),
        "expected_behavior": expected,
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
        "answer_text": answer_text,
        "retrieved_sources": public_citations,
        "message_plan": {"applied": bool(result.get("message_plan_applied")), "goals": goals},
        "external_context_sent": bool(result.get("external_context_sent", trace.get("context_sent"))),
        "external_model_result_accepted": bool(result.get("external_model_result_accepted")),
        "provider_call": safe_trace,
        "seed_platform_recorded": seed_platform_ok,
    }


def run_live_scenarios(include_bounded_context: bool = False) -> list[dict[str, Any]]:
    if not include_bounded_context:
        raise ValueError("The live guidance harness requires explicit --include-bounded-context consent; only synthetic context is used.")
    previous_store = main.store
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="accp-guidance-live-") as temp_dir:
        main.store = Store(str(Path(temp_dir) / "guidance.sqlite3"))
        try:
            with TestClient(main.app) as client:
                for scenario in SCENARIOS:
                    scenario_started = time.perf_counter()
                    created = client.post("/api/sessions", json={})
                    if created.status_code != 200:
                        results.append({"scenario_id": scenario["id"], "passed": False,
                                        "error": f"session creation returned HTTP {created.status_code}"})
                        continue
                    payload = created.json()
                    session = payload["session"]
                    case_id, token = session["session_id"], payload["token"]
                    revision = session["revision"]
                    seed_platform_ok: bool | None = None
                    try:
                        if scenario.get("prior_platform"):
                            seeded = _send(client, case_id, token, revision, scenario["prior_platform"],
                                           allow_model=False, allow_context=False)
                            seed_session = seeded["session"]
                            revision = seed_session["revision"]
                            platform_fact = seed_session.get("facts", {}).get("platform", {})
                            seed_platform_ok = platform_fact.get("status") == "reported"
                        result = _send(client, case_id, token, revision, scenario["text"],
                                       allow_model=True, allow_context=True)
                        item = _evaluate_result(scenario, result, seed_platform_ok)
                        item["elapsed_ms"] = round((time.perf_counter() - scenario_started) * 1000, 3)
                        item["external_data_sent"] = bool(result.get("external_data_sent"))
                        item["external_model_purpose"] = result.get("external_model_purpose")
                        item["fact_or_answer_fallback_used"] = bool(result.get("fallback_used"))
                        item["fact_or_answer_fallback_to"] = result.get("fallback_to")
                        results.append(item)
                    except Exception as exc:
                        results.append({"scenario_id": scenario["id"], "passed": False,
                                        "error": f"{type(exc).__name__}: {exc}",
                                        "seed_platform_recorded": seed_platform_ok,
                                        "elapsed_ms": round((time.perf_counter() - scenario_started) * 1000, 3)})
        finally:
            main.store = previous_store
    return results


def main_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Run synthetic Arabic guidance-routing and real-corpus retrieval checks through the configured local Ollama model."
    )
    parser.add_argument("--include-bounded-context", action="store_true",
                        help="Explicitly consent to sending only the synthetic bounded context in these scenarios.")
    parser.add_argument("--preflight-timeout", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "evaluation")
    parser.add_argument("--output", type=Path, help="Optional new report path; existing files are never overwritten.")
    args = parser.parse_args()
    if not args.include_bounded_context:
        parser.error("--include-bounded-context is required; this script sends synthetic prior-turn context to the model.")

    output_path = _output_path(args.output_dir, args.output)
    config = provider_config()
    report: dict[str, Any] = {
        "evaluation_version": "guidance-live-smoke-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_status": "not_run",
        "synthetic_only": True,
        "bounded_context_consent": True,
        "provider_config": _safe_provider_summary(config),
        "preflight": None,
        "preflight_generation": None,
        "scenarios": [],
        "summary": {"passed": 0, "total": len(SCENARIOS), "failed": 0},
        "note": "A model preflight and live API calls are required. This is a small synthetic smoke test, not an accuracy benchmark or proof of real-world readiness.",
    }

    if config.get("provider") != EXPECTED_PROVIDER or config.get("model") != EXPECTED_MODEL or not config.get("configuration_complete"):
        report["reason"] = "expected_local_ollama_model_not_configured"
        _write_new_report(output_path, report)
        print(json.dumps({"report": str(output_path), "test_status": report["test_status"],
                          "reason": report["reason"], "provider_config": report["provider_config"]}, ensure_ascii=False, indent=2))
        return 2

    preflight = probe_provider(timeout=max(0.1, args.preflight_timeout))
    report["preflight"] = _safe_probe_summary(preflight)
    if not preflight.get("configured") or not preflight.get("service_reachable") or not preflight.get("model_available"):
        report["reason"] = preflight.get("failure_code") or "ollama_preflight_failed"
        _write_new_report(output_path, report)
        print(json.dumps({"report": str(output_path), "test_status": report["test_status"],
                          "reason": report["reason"], "preflight": report["preflight"]}, ensure_ascii=False, indent=2))
        return 2

    generation = test_provider_generation()
    report["preflight_generation"] = _safe_generation_summary(generation)
    if not generation.get("success"):
        report["reason"] = generation.get("failure_code") or "ollama_generation_failed"
        _write_new_report(output_path, report)
        print(json.dumps({"report": str(output_path), "test_status": report["test_status"],
                          "reason": report["reason"], "preflight_generation": report["preflight_generation"]}, ensure_ascii=False, indent=2))
        return 2

    results = run_live_scenarios(include_bounded_context=True)
    passed = sum(bool(item.get("passed")) for item in results)
    report["scenarios"] = results
    report["summary"] = {"passed": passed, "total": len(results), "failed": len(results) - passed}
    report["test_status"] = "completed"
    _write_new_report(output_path, report)
    print(json.dumps({"report": str(output_path), "test_status": report["test_status"],
                      "provider": EXPECTED_PROVIDER, "model": EXPECTED_MODEL,
                      "summary": report["summary"],
                      "scenario_results": [{"id": item.get("scenario_id"), "passed": item.get("passed"),
                                            "checks": item.get("checks"), "retrieved_sources": [source.get("id") for source in item.get("retrieved_sources", [])]}
                                           for item in results]}, ensure_ascii=False, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
