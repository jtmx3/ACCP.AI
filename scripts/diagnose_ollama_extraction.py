#!/usr/bin/env python3
"""نفّذ طلب استخراج ACCP.AI ومدققه الفعليين مرة واحدة عبر Ollama المحلي."""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import QUESTIONS
from app.model import (generate_interview_extraction, last_call_trace,
                       last_extraction_diagnostic, probe_provider,
                       provider_config, reset_call_trace)

MAX_CONTEXT_CHARS = 4500
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
FORBIDDEN_CONTEXT_KEYS = {"token", "case_token", "session_token", "session_id", "api_key", "authorization"}


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).strip().lower() in FORBIDDEN_CONTEXT_KEYS
                   or _contains_forbidden_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def load_bounded_context(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    context = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(context, dict) or context.get("context_version") != "accp-limited-context-v2":
        raise ValueError("ملف السياق يجب أن يكون ناتجًا عن build_bounded_context بإصدار accp-limited-context-v2.")
    if _contains_forbidden_key(context):
        raise ValueError("ملف السياق يحتوي مفتاحًا محظورًا مثل رمز جلسة أو اعتماد.")
    serialized = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > MAX_CONTEXT_CHARS:
        raise ValueError(f"حجم السياق {len(serialized)} حرفًا ويتجاوز الحد {MAX_CONTEXT_CHARS}.")
    if not context.get("untrusted_user_data"):
        raise ValueError("يجب أن يعلن السياق صراحة أنه بيانات مستخدم غير موثوقة.")
    context["serialized_char_count"] = len(serialized)
    return context


def diagnose(message: str, pending_slot: str | None = None, *,
             context_file: Path | None = None, save_raw_response: Path | None = None,
             probe_timeout: float = 5.0) -> dict[str, Any]:
    config = provider_config()
    if config.get("provider") != "ollama" or not config.get("local"):
        raise RuntimeError("هذه الأداة تسمح بمزود Ollama المحلي فقط.")
    if not config.get("configured") or config.get("base_url_host") not in LOOPBACK_HOSTS:
        raise RuntimeError("إعداد Ollama غير مكتمل أو لا يشير إلى عنوان loopback؛ لم يُرسل أي طلب.")
    context = load_bounded_context(context_file)
    probe = probe_provider(timeout=probe_timeout)
    output: dict[str, Any] = {
        "provider": "ollama", "model": config.get("model"),
        "service_reachable": bool(probe.get("service_reachable")),
        "model_available": bool(probe.get("model_available")),
        "probe_failure_code": probe.get("failure_code"),
        "probe_elapsed_ms": probe.get("elapsed_ms"),
        "context_supplied": context is not None,
        "context_char_count": context.get("serialized_char_count", 0) if context else 0,
        "prompt_source": "app.model.generate_interview_extraction",
        "validator_source": "app.model.validate_interview_facts_diagnostic",
        "merge_performed": False, "fallback_performed": False,
        "raw_response_saved": False,
    }
    if not probe.get("service_reachable") or not probe.get("model_available"):
        output.update({"generation_attempted": False, "provider_response_received": False,
                       "response_valid": False, "extraction_outcome": "not_run"})
        return output

    reset_call_trace()
    started = time.perf_counter()
    facts = generate_interview_extraction(message, pending_slot, bounded_context=context)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    trace = last_call_trace() or {}
    diagnostic = last_extraction_diagnostic() or {}
    filtered = diagnostic.get("filtered_facts", [])
    raw = diagnostic.get("raw_response")
    if facts:
        outcome = "extracted"
    elif filtered:
        outcome = "all_facts_filtered"
    elif diagnostic.get("accepted"):
        outcome = "valid_empty"
    else:
        outcome = trace.get("extraction_outcome", "provider_failed")
    output.update({
        "generation_attempted": bool(trace.get("attempted")),
        "provider_response_received": bool(isinstance(raw, str) and raw.strip()),
        "provider_status": trace.get("status"),
        "provider_failure_code": trace.get("failure_code"),
        "provider_latency_ms": trace.get("latency_ms"),
        "total_elapsed_ms": elapsed_ms,
        "response_valid": diagnostic.get("accepted") is True,
        "extraction_outcome": outcome,
        "accepted_fact_count": len(facts or []),
        "accepted_facts": [{"field": item.get("field"), "status": item.get("status"),
                            "value": item.get("value"), "quote": item.get("quote")}
                           for item in (facts or []) if isinstance(item, dict)],
        "filtered_facts": [{"field": item.get("field"), "status": item.get("status"),
                            "filter_code": item.get("filter_code", "other_validation"),
                            "filter_detail": item.get("filter_detail", "لم يُسجل تفصيل التحقق.")}
                           for item in filtered if isinstance(item, dict)],
    })
    if save_raw_response is not None and isinstance(raw, str):
        path = save_raw_response.expanduser().resolve()
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(raw)
        output["raw_response_saved"] = True
        output["raw_response_path"] = str(path)
    return output


def save_result(output: dict[str, Any], requested_path: Path | None = None) -> Path:
    if requested_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        requested_path = ROOT / "reports" / "diagnostics" / f"ollama-extraction-{stamp}-{uuid.uuid4().hex[:8]}.json"
    path = requested_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    output["result_file"] = str(path)
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", required=True, help="رسالة اختبار اصطناعية أو مسموح بها محليًا")
    parser.add_argument("--pending-slot", choices=sorted(QUESTIONS), default=None)
    parser.add_argument("--context-file", type=Path, default=None,
                        help="JSON سياق محدود منشأ بـbuild_bounded_context، وليس ملف حالة كاملًا")
    parser.add_argument("--probe-timeout", type=float, default=5.0)
    parser.add_argument("--save-raw-response", type=Path, default=None,
                        help="حفظ محلي صريح للرد الخام في ملف جديد؛ لا يكتب فوق ملف موجود")
    parser.add_argument("--output", type=Path, default=None,
                        help="مسار تقرير JSON جديد؛ افتراضيًا يولد البرنامج مسارًا فريدًا في reports/diagnostics")
    args = parser.parse_args()
    try:
        result = diagnose(args.message, args.pending_slot, context_file=args.context_file,
                          save_raw_response=args.save_raw_response, probe_timeout=args.probe_timeout)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        result = {"error": str(exc), "generation_attempted": False,
                  "provider_response_received": False, "response_valid": False,
                  "merge_performed": False, "fallback_performed": False}
    except Exception as exc:  # Never expose request/response bodies, URLs, or credentials.
        result = {"error": "فشل فحص التشخيص.", "error_type": type(exc).__name__,
                  "generation_attempted": False, "provider_response_received": False,
                  "response_valid": False, "merge_performed": False, "fallback_performed": False}
    try:
        save_result(result, args.output)
    except OSError as exc:
        print(json.dumps({"error": "تعذر حفظ ملف النتائج محليًا.", "error_type": type(exc).__name__,
                          "generation_attempted": result.get("generation_attempted", False)},
                         ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("generation_attempted") and result.get("provider_response_received") else 1


if __name__ == "__main__":
    raise SystemExit(main())
