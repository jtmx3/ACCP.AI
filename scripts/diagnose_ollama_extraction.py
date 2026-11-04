#!/usr/bin/env python3
"""Run the real ACCP.AI extraction prompt against a configured local Ollama model."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model import (generate_interview_extraction, last_call_trace,
                       last_extraction_diagnostic, provider_config)


def diagnose(message: str, pending_slot: str | None = None) -> dict[str, Any]:
    config = provider_config()
    if config["provider"] != "ollama":
        raise RuntimeError("Set MODEL_PROVIDER=ollama before running this diagnostic.")
    if not config["configured"]:
        raise RuntimeError("Ollama configuration is incomplete; set MODEL_BASE_URL and MODEL_NAME.")

    started = time.perf_counter()
    facts = generate_interview_extraction(message, pending_slot)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    trace = last_call_trace() or {}
    diagnostic = last_extraction_diagnostic() or {}
    filtered = diagnostic.get("filtered_facts", [])

    if facts:
        outcome = "extracted"
    elif filtered:
        outcome = "all_facts_filtered"
    elif diagnostic.get("accepted"):
        outcome = "valid_empty"
    else:
        outcome = trace.get("extraction_outcome", "provider_failed")

    return {
        "provider": "ollama",
        "model": config["model"],
        "reasoning_effort": config["ollama_reasoning_effort"],
        "elapsed_ms": elapsed_ms,
        "provider_latency_ms": trace.get("latency_ms"),
        "response_valid": bool(diagnostic.get("accepted")),
        "extraction_outcome": outcome,
        "accepted_facts": facts or [],
        "filtered_fact_reasons": [
            {"field": item.get("field"), "code": item.get("filter_code"),
             "reason": item.get("filter_detail")}
            for item in filtered
        ],
        "failure_code": trace.get("failure_code"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", required=True, help="Current synthetic/test message sent to local Ollama")
    parser.add_argument("--pending-slot", default=None, help="Optional pending ACCP.AI question field")
    args = parser.parse_args()
    try:
        result = diagnose(args.message, args.pending_slot)
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["extraction_outcome"] in {"extracted", "valid_empty", "all_facts_filtered"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
