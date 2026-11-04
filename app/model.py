from __future__ import annotations

import contextvars
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

BINARY_FIELDS = {
    "threat", "impersonation", "publication_threat", "repeated", "published",
    "active_unauthorized_funds", "fraud_description", "account_access", "harassment", "contact_repeated",
}
ALLOWED_EXTRACTION_FIELDS = {
    "narrative", "threat", "impersonation", "platform", "details", "publication_threat",
    "active_unauthorized_funds", "fraud_description", "initial_time", "latest_time", "repeated",
    "requested_amount", "paid_amount", "payment_method", "deadline", "published", "identity",
    "contact", "evidence", "account_access", "harassment", "contact_repeated",
}
ALLOWED_STATUSES = {"reported", "denied", "unknown", "not_remembered", "declined"}
FIELD_ANCHORS = {
    "threat": ("تهديد", "هدد", "يهدد", "ابتزاز", "يبتز", "يلاحق", "لاحق", "يؤذ", "اذي", "يضر", "اضر", "ضرر"),
    "impersonation": ("انتحال", "منتحل", "ينتحل", "حساب مزيف", "باسمي", "كأنه انا", "كأنها انا"),
    "publication_threat": ("ينشر", "نشر", "ينزل", "نزل", "بفضح", "يفضح", "يهدد"),
    "repeated": ("تكرر", "كرر", "تكرار", "مستمر", "مرة ثانية", "مره ثانيه", "رسالة ثانية", "رساله ثانيه", "رجع بعت", "رجع ارسل", "رجع هدد"),
    "published": ("نشر", "نزل", "انتشرت", "بعت الصور", "حاط"),
    "active_unauthorized_funds": ("يسحب", "تنسحب", "بدون اذني", "دون اذني", "غير ماذون"),
    "fraud_description": ("نصب", "احتيال", "خدعني", "احتيال مالي"),
    "account_access": ("اخترق", "دخل على حساب", "دخل حساب", "سرق حساب"),
    "harassment": ("مضايقة", "تحرش", "يلاحق", "بلاحق", "بضايق"),
    "contact_repeated": ("تكرر", "كرر", "رسالة ثانية", "رساله ثانيه", "رجع بعت", "عدة رسائل", "مرات"),
}

_CALL_TRACE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("accp_model_call_trace", default=None)
_EXTRACTION_DIAGNOSTIC: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("accp_extraction_diagnostic", default=None)


def provider_config() -> dict[str, Any]:
    """Return safe provider configuration facts; never return a key or full URL."""
    provider = os.getenv("MODEL_PROVIDER", "rules").strip().lower() or "rules"
    local = provider == "ollama"
    base_url = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:11434/v1" if local else "https://api.openai.com/v1").strip().rstrip("/")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini" if provider == "openai-compatible" else "").strip()
    key_present = bool(os.getenv("MODEL_API_KEY")) or local
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or ""
    valid_provider = provider in {"rules", "ollama", "openai-compatible"}
    configured = provider in {"ollama", "openai-compatible"} and bool(base_url and model and key_present and valid_provider)
    mode = "local_model" if local and configured else "external_model" if provider == "openai-compatible" and configured else "rules"
    ollama_reasoning_effort = os.getenv("OLLAMA_REASONING_EFFORT", "none").strip().lower()
    if ollama_reasoning_effort not in {"none", "low", "medium", "high"}:
        ollama_reasoning_effort = "none"
    external_reasoning_effort = os.getenv("MODEL_REASONING_EFFORT", "").strip() or None
    return {
        "provider": provider if valid_provider else "invalid",
        "mode": mode,
        "local": local,
        "model": model or None,
        "base_url_host": host or None,
        "api_key_present": key_present,
        "configuration_complete": configured,
        "configured": configured,
        "valid_provider": valid_provider,
        "ollama_reasoning_effort": ollama_reasoning_effort if local else None,
        "external_reasoning_effort": external_reasoning_effort if provider == "openai-compatible" else None,
    }


def model_configured() -> bool:
    return bool(provider_config()["configured"])


def reset_call_trace() -> None:
    _CALL_TRACE.set(None)


def last_call_trace() -> dict[str, Any] | None:
    trace = _CALL_TRACE.get()
    return dict(trace) if trace else None


def last_extraction_diagnostic() -> dict[str, Any] | None:
    """Internal extraction-only diagnostics; raw response is never part of API call trace."""
    diagnostic = _EXTRACTION_DIAGNOSTIC.get()
    return dict(diagnostic) if diagnostic else None


def _update_call_trace(**values: Any) -> None:
    trace = dict(_CALL_TRACE.get() or {})
    trace.update(values)
    _CALL_TRACE.set(trace)


def _base_url() -> str:
    cfg = provider_config()
    if cfg["local"]:
        return os.getenv("MODEL_BASE_URL", "http://127.0.0.1:11434/v1").strip().rstrip("/")
    return os.getenv("MODEL_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")


def _model_name() -> str:
    return str(provider_config()["model"] or "")


def _api_key() -> str:
    if provider_config()["local"]:
        return os.getenv("MODEL_API_KEY", "ollama")
    return os.getenv("MODEL_API_KEY", "")


def _timeout_seconds() -> float:
    try:
        return max(1.0, float(os.getenv("MODEL_TIMEOUT_SECONDS", "60")))
    except ValueError:
        return 60.0


def _chat(messages: list[dict[str, str]]) -> str | None:
    cfg = provider_config()
    if not cfg["configured"]:
        _update_call_trace(attempted=False, status="not_configured", result_accepted=False, latency_ms=None)
        return None
    body: dict[str, Any] = {
        "model": _model_name(),
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    if cfg["local"]:
        body.update({"temperature": 0, "max_tokens": 1200,
                     "reasoning_effort": cfg["ollama_reasoning_effort"]})
    else:
        body["max_completion_tokens"] = 1200
        effort = os.getenv("MODEL_REASONING_EFFORT", "").strip()
        if effort:
            body["reasoning"] = {"effort": effort}
    request = urllib.request.Request(
        f"{_base_url()}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=_timeout_seconds()) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        if not isinstance(content, str) or not content.strip():
            _update_call_trace(attempted=True, status="empty_response", result_accepted=False,
                               failure_code="empty_response", latency_ms=elapsed)
            return None
        _update_call_trace(attempted=True, status="generated", result_accepted=False,
                           failure_code=None, latency_ms=elapsed)
        return content
    except urllib.error.HTTPError as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        _update_call_trace(attempted=True, status="http_error", result_accepted=False,
                           failure_code=f"http_{exc.code}", latency_ms=elapsed)
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        reason = getattr(exc, "reason", exc)
        code = "timeout" if isinstance(reason, (TimeoutError,)) or "timed out" in str(reason).lower() else "unreachable"
        _update_call_trace(attempted=True, status="connection_error", result_accepted=False,
                           failure_code=code, latency_ms=elapsed)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        _update_call_trace(attempted=True, status="invalid_response", result_accepted=False,
                           failure_code="invalid_provider_response", provider_error_type=type(exc).__name__,
                           latency_ms=elapsed)
    return None


def _models_url(cfg: dict[str, Any]) -> str:
    base = _base_url()
    if cfg["local"]:
        root = re.sub(r"/v1/?$", "", base)
        return f"{root}/api/tags"
    return f"{base}/models"


def probe_provider(timeout: float = 5.0) -> dict[str, Any]:
    """Non-generative probe: configuration, service reachability, and selected model presence."""
    cfg = provider_config()
    result = {
        "provider": cfg["provider"], "mode": cfg["mode"], "model": cfg["model"],
        "configured": cfg["configured"], "service_reachable": False,
        "model_available": False, "models_count": 0, "failure_code": None,
    }
    if not cfg["configured"]:
        result["failure_code"] = "not_configured"
        return result
    headers = {"Accept": "application/json"}
    if not cfg["local"]:
        headers["Authorization"] = f"Bearer {_api_key()}"
    request = urllib.request.Request(_models_url(cfg), headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if cfg["local"]:
            models = payload.get("models", [])
            names = [item.get("name") or item.get("model") for item in models if isinstance(item, dict)]
        else:
            models = payload.get("data", [])
            names = [item.get("id") for item in models if isinstance(item, dict)]
        names = [str(name) for name in names if name]
        result["service_reachable"] = True
        result["models_count"] = len(names)
        result["model_available"] = _model_name() in names
        result["available_model_names"] = names
        if not result["model_available"]:
            result["failure_code"] = "model_not_found"
    except urllib.error.HTTPError as exc:
        result["failure_code"] = f"http_{exc.code}"
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        result["failure_code"] = "timeout" if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower() else "unreachable"
    except (ValueError, KeyError, TypeError):
        result["failure_code"] = "invalid_model_list"
    return result


def test_provider_generation() -> dict[str, Any]:
    """Make an explicit, fixed, non-user-data generation check and require valid JSON."""
    started = time.perf_counter()
    reset_call_trace()
    prompt = [
        {"role": "system", "content": "أعد JSON صالحًا فقط بالشكل {\"ok\":true}. لا تضف أي نص آخر."},
        {"role": "user", "content": "اختبار اتصال ثابت. أعد {\"ok\":true}."},
    ]
    content = _chat(prompt)
    success = False
    if content:
        try:
            success = json.loads(content).get("ok") is True
        except (ValueError, TypeError, AttributeError):
            success = False
    trace = last_call_trace() or {}
    if trace.get("attempted"):
        trace["generation_test_success"] = success
        trace["status"] = "generation_succeeded" if success else "generation_rejected"
        trace["result_accepted"] = success
        if not success and not trace.get("failure_code"):
            trace["failure_code"] = "invalid_test_json"
        _CALL_TRACE.set(trace)
    return {
        "success": success,
        "attempted": bool(trace.get("attempted")),
        "latency_ms": trace.get("latency_ms", round((time.perf_counter() - started) * 1000, 3)),
        "failure_code": trace.get("failure_code"),
        "provider": provider_config()["provider"],
        "model": provider_config()["model"],
    }


def validate_extractive_claims(payload: str, sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Only display verbatim source spans; free-form generated claims are never shown."""
    try:
        parsed = json.loads(payload)
        claims = parsed.get("claims")
        if not isinstance(claims, list) or not claims or len(claims) > 4:
            return None
        source_map = {str(source.get("id")): source for source in sources}
        source_order = [str(source.get("id")) for source in sources]
        checked: list[tuple[str, str]] = []
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) != {"source_id", "quote"}:
                return None
            source_id = claim.get("source_id")
            quote = claim.get("quote")
            source = source_map.get(source_id) if isinstance(source_id, str) else None
            excerpt = source.get("excerpt") if source else None
            if not isinstance(quote, str) or not quote.strip() or not isinstance(excerpt, str) or quote not in excerpt:
                return None
            checked.append((source_id, quote))
        used_ids = list(dict.fromkeys(source_id for source_id, _ in checked))
        lines = []
        for source_id, quote in checked:
            citation_number = source_order.index(source_id) + 1
            lines.append(f"«{quote}» [{citation_number}]")
        return {"text": "\n".join(lines), "source_ids": used_ids}
    except (ValueError, TypeError, AttributeError):
        return None


def generate_grounded_answer(question: str, sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Ask only for source quote selection; returned prose is built from verified literal excerpts."""
    if not model_configured() or not sources:
        return None
    evidence = [{"source_id": str(source["id"]), "title": source.get("title"),
                 "excerpt": source.get("excerpt", "")} for source in sources]
    messages = [
        {"role": "system", "content": "أعد JSON فقط بالشكل {\"claims\":[{\"source_id\":\"...\",\"quote\":\"...\"}]}. اختر من المقتطفات اقتباسًا حرفيًا قصيرًا يدعم جواب السؤال. لا تكتب أي claim أو نص حر. يجب أن يكون quote سلسلة حرفية متصلة من مقتطف المصدر نفسه. إذا لم يوجد مقتطف يجيب السؤال، أعد {\"claims\":[]}. لا تستنتج أو تضف أي معلومات."},
        {"role": "user", "content": json.dumps({"question": question, "sources": evidence}, ensure_ascii=False)},
    ]
    draft = _chat(messages)
    result = validate_extractive_claims(draft, sources) if draft else None
    if last_call_trace() and last_call_trace().get("attempted"):
        _update_call_trace(result_accepted=result is not None,
                           status="accepted" if result is not None else (last_call_trace() or {}).get("status", "rejected"),
                           failure_code=None if result is not None else (last_call_trace() or {}).get("failure_code") or "schema_rejected")
    return result


def _normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text.lower())
    return text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")


def _threat_quote_supported(normalized_quote: str) -> bool:
    explicit_harm = ("تهديد", "هدد", "يهدد", "ابتزاز", "يبتز", "يلاحق", "لاحق", "يؤذ", "اذي", "يضر", "اضر", "ضرر")
    if any(marker in normalized_quote for marker in explicit_harm):
        return True
    publication = ("ينشر", "نشر", "ينزل", "نزل", "يفضح", "فضح", "يرسل")
    condition = ("اذا", "لو", "ان لم", "ما دفعت", "لم ادفع", "ما حول", "لم احول", "ما استجبت", "اذا ما", "اذا لم")
    return any(marker in normalized_quote for marker in publication) and any(marker in normalized_quote for marker in condition)


def _publication_threat_quote_supported(normalized_quote: str) -> bool:
    ordinary_markers = tuple(_normalize_arabic(anchor) for anchor in FIELD_ANCHORS["publication_threat"])
    if any(marker in normalized_quote for marker in ordinary_markers):
        return True
    send_material = ("يرسل", "يبعث")
    private_material = ("محادث", "صور", "مقاطع", "مراسلات")
    condition = ("اذا", "لو", "ان لم", "ما دفعت", "لم ادفع", "ما حول", "لم احول")
    return (any(marker in normalized_quote for marker in send_material)
            and any(marker in normalized_quote for marker in private_material)
            and any(marker in normalized_quote for marker in condition))


def _reported_binary_conflicts_with_clear_negation(field: str, normalized_quote: str) -> bool:
    negation_by_field = {
        "threat": ("ما في تهديد", "ما في اي تهديد", "ما حدا هدد", "ما هددني", "لم يهددني", "لا يهددني"),
        "account_access": ("ما حدا دخل", "ما دخل حسابي", "لم يدخل حسابي", "ما اخترق حسابي", "لم يخترق حسابي"),
        "published": ("ما نشر", "لم ينشر", "ما نزل", "لم ينزل", "ما بعرف اذا نشر", "ما بعرف اذا نزل",
                      "مش متاكد اذا نشر", "مش متاكد اذا نزل"),
    }
    return any(phrase in normalized_quote for phrase in negation_by_field.get(field, ()))


def _reported_text_conflicts_with_clear_negation(field: str, normalized_quote: str) -> bool:
    negation_by_field = {
        "evidence": ("ما عندي نسخ", "ما عندي دليل", "ما في دليل", "لا يوجد دليل",
                     "لا املك نسخة", "لم احفظ", "ما حفظت", "غير محفوظ"),
    }
    return any(phrase in normalized_quote for phrase in negation_by_field.get(field, ()))


def _nonreported_status_has_support(field: str, status: str, normalized_quote: str,
                                    pending_slot: str | None) -> bool:
    unknown_markers = ("ما بعرف", "لا اعرف", "مش عارف", "غير معروف", "ما بتذكر", "لا اتذكر",
                       "مش متذكر", "مش متاكد", "ما اعرف")
    memory_markers = ("ما بتذكر", "لا اتذكر", "مش متذكر", "ناسي", "ناسية")
    decline_markers = ("ما بدي", "لا اريد", "مش حاب", "ما حاب", "لن اكتب", "لا ارغب", "مش مستعد")
    field_cues = {
        "identity": ("هوي", "مين هو", "مين صاحب", "اسمه", "اسم الحساب", "رقم التواصل"),
        "published": ("نشر", "ينشر", "نزل", "ينزل", "رفع", "منشور"),
        "requested_amount": ("طلب", "مبلغ", "دينار", "ريال", "دولار"),
        "paid_amount": ("دفع", "دفعت", "حول", "حوالة", "تحويل", "مبلغ"),
        "payment_method": ("دفع", "دفعت", "حول", "حوالة", "تحويل", "محفظة", "بطاقة"),
        "evidence": ("دليل", "نسخ", "لقطات", "رسائل", "محفوظ"),
        "contact": ("رقم", "حساب", "معرف", "اسم المستخدم", "واتساب"),
        "initial_time": ("متى", "وقت", "مبارح", "امبارح", "اليوم", "الاسبوع", "قبل"),
        "latest_time": ("متى", "وقت", "مبارح", "امبارح", "اليوم", "الاسبوع", "قبل"),
    }
    cues_match = any(cue in normalized_quote for cue in field_cues.get(field, ()))
    if status == "not_remembered":
        return any(marker in normalized_quote for marker in memory_markers) and (field == pending_slot or cues_match)
    if status == "declined":
        return any(marker in normalized_quote for marker in decline_markers) and (field == pending_slot or cues_match)
    if status != "unknown":
        return True
    explicit_unknown = any(marker in normalized_quote for marker in unknown_markers)
    if field == pending_slot and (explicit_unknown or normalized_quote.strip() in {"لا", "مش عارف", "ما بعرف"}):
        return True
    if explicit_unknown and cues_match:
        return True
    if field == "paid_amount" and re.search(r"\d", normalized_quote.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))) is None:
        return any(cue in normalized_quote for cue in field_cues["paid_amount"])
    return False


def _contact_value_is_specific(value: str, quote: str) -> bool:
    normalized_value = _normalize_arabic(value)
    normalized_quote = _normalize_arabic(quote)
    channels = ("رقم واتساب", "رقم تواصل", "رقم حساب", "حساب تواصل", "اسم حساب", "اسم مستخدم")
    availability = ("عندي", "معي", "متوفر", "المتوفر", "احتفظت")
    if any(channel in normalized_value for channel in channels) and any(marker in normalized_quote for marker in availability):
        return True
    digits = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    quote_digits = re.sub(r"\D", "", quote.translate(digits))
    value_digits = re.sub(r"\D", "", value.translate(digits))
    if len(quote_digits) >= 6 and quote_digits in value_digits:
        return True
    handle = re.search(r"(?<!\w)@[A-Za-z0-9_.-]{2,}", quote)
    if handle and handle.group(0) in value:
        return True
    username = re.search(r"(?:اسم المستخدم|username)\s*[:=]?\s*([\w.-]{2,})", quote, re.IGNORECASE)
    return bool(username and username.group(1) in value)


def _json_failure_code(payload: str, error: json.JSONDecodeError) -> str:
    stripped = payload.rstrip()
    looks_like_json = stripped.startswith(("{", "["))
    incomplete_end = not stripped.endswith(("}", "]"))
    error_at_end = error.pos >= max(0, len(stripped) - 2)
    unterminated = "unterminated" in error.msg.lower()
    return "truncated_json" if looks_like_json and (incomplete_end or error_at_end or unterminated) else "invalid_json"


def validate_interview_facts_diagnostic(text: str, payload: str,
                                        pending_slot: str | None = None) -> dict[str, Any]:
    """Return the unchanged validation decision plus a specific, non-secret rejection reason."""
    json_valid = False
    schema_valid = False

    def reject(code: str, detail: str) -> dict[str, Any]:
        return {"accepted": False, "facts": None, "json_valid": json_valid,
                "schema_valid": schema_valid, "rejection_code": code, "rejection_detail": detail}

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        return reject(_json_failure_code(payload, exc), f"JSON parse error at character {exc.pos}: {exc.msg}")
    except (TypeError, ValueError) as exc:
        return reject("invalid_json", f"JSON parse error: {type(exc).__name__}")
    json_valid = True
    if not isinstance(parsed, dict):
        return reject("other_validation", "Top-level JSON value must be an object.")
    facts = parsed.get("facts")
    if not isinstance(facts, list):
        return reject("other_validation", "The facts property must be an array.")
    if len(facts) > 24:
        return reject("other_validation", "The facts array exceeds the 24-item limit.")

    seen: set[str] = set()
    for index, fact in enumerate(facts):
        prefix = f"facts[{index}]"
        if not isinstance(fact, dict) or set(fact) != {"field", "status", "value", "quote"}:
            return reject("other_validation", f"{prefix} must contain exactly field, status, value, and quote.")
        field, status, value, quote = fact["field"], fact["status"], fact["value"], fact["quote"]
        if not isinstance(field, str) or field not in ALLOWED_EXTRACTION_FIELDS:
            return reject("unsupported_field_or_status", f"{prefix} uses an unsupported extraction field.")
        if not isinstance(status, str) or status not in ALLOWED_STATUSES:
            return reject("unsupported_field_or_status", f"{prefix} uses an unsupported fact status.")
        if field in seen:
            return reject("other_validation", f"{prefix} repeats field {field!r} in one response.")
        if not isinstance(quote, str):
            return reject("other_validation", f"{prefix}.quote must be a string.")
        seen.add(field)
    schema_valid = True

    checked: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for index, fact in enumerate(facts):
        prefix = f"facts[{index}]"
        field, status, value, quote = fact["field"], fact["status"], fact["value"], fact["quote"]
        reason: tuple[str, str] | None = None
        if not isinstance(quote, str) or not quote or quote not in text:
            reason = ("quote_mismatch", f"{prefix}.quote is not a non-empty verbatim span of the current message.")
        else:
            normalized_quote = _normalize_arabic(quote)
            anchors = FIELD_ANCHORS.get(field, ())
            if field == "threat":
                anchored = _threat_quote_supported(normalized_quote)
            elif field == "publication_threat":
                anchored = _publication_threat_quote_supported(normalized_quote)
            else:
                anchored = not anchors or any(_normalize_arabic(anchor) in normalized_quote for anchor in anchors)
            if not anchored:
                reason = ("quote_mismatch", f"{prefix}.quote does not contain the evidence anchor required for {field!r}.")
        if reason is None and status == "reported":
            if field in BINARY_FIELDS:
                if value is not True:
                    reason = ("value_inconsistent", f"Reported binary field {field!r} must have value true.")
                elif _reported_binary_conflicts_with_clear_negation(field, normalized_quote):
                    reason = ("value_inconsistent", f"Reported {field!r} conflicts with explicit negation in its quote.")
            elif not isinstance(value, str) or not value.strip():
                reason = ("value_inconsistent", f"Reported text field {field!r} must have a non-empty string value.")
            elif _reported_text_conflicts_with_clear_negation(field, normalized_quote):
                reason = ("value_inconsistent", f"Reported {field!r} conflicts with explicit absence in its quote.")
            if reason is None and field == "narrative" and value != quote:
                reason = ("value_inconsistent", "The narrative value must be exactly the quoted span.")
            if reason is None and field == "identity":
                normalized_value = _normalize_arabic(value)
                victim_owned_markers = ("اسمي", "صورتي", "حسابي", "رقمي")
                if any(marker in normalized_value for marker in victim_owned_markers):
                    reason = ("value_inconsistent", "Reported identity refers to the reporter's own identifier, not the suspect.")
                elif normalized_value not in normalized_quote:
                    reason = ("value_inconsistent", "Reported identity value is not literally supported by its quote.")
            if reason is None and field == "contact" and not _contact_value_is_specific(value, quote):
                reason = ("value_inconsistent", "The contact value is a generic channel, not a supported identifier or explicitly available contact.")
            if reason is None and field == "repeated" and not any(term in normalized_quote for term in (
                "تهديد", "هدد", "ابتزاز", "ابتز", "مضايق", "تحرش", "لاحق", "يؤذ", "اذي"
            )):
                reason = ("value_inconsistent", "Repeated harm lacks an explicit harm-context marker in its quote.")
            if reason is None and field in {"requested_amount", "paid_amount"}:
                quote_digits = re.sub(r"\D", "", quote.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
                value_digits = re.sub(r"\D", "", value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
                if value_digits and value_digits not in quote_digits:
                    reason = ("value_inconsistent", f"The numeric value for {field!r} is absent from its quote.")
        elif reason is None and status == "denied":
            if field in BINARY_FIELDS and value is not False:
                reason = ("value_inconsistent", f"Denied binary field {field!r} must have value false.")
            if field not in BINARY_FIELDS and value is not None:
                reason = ("value_inconsistent", f"Denied text field {field!r} must have null value.")
        elif reason is None and value is not None:
            reason = ("value_inconsistent", f"Non-reported status {status!r} must have null value.")
        if reason is None and status in {"unknown", "not_remembered", "declined"} and not _nonreported_status_has_support(
                field, status, normalized_quote, pending_slot):
            reason = ("unsupported_status_evidence", f"Status {status!r} for {field!r} lacks explicit or field-specific support.")
        if reason is not None:
            filtered.append({"field": field, "status": status, "value": value, "quote": quote,
                             "filter_code": reason[0], "filter_detail": reason[1]})
            continue
        checked.append({"field": field, "status": status, "value": value, "quote": quote})
    return {"accepted": True, "facts": checked, "filtered_facts": filtered,
            "json_valid": True, "schema_valid": True,
            "rejection_code": None, "rejection_detail": None}


def validate_interview_facts(text: str, payload: str) -> list[dict[str, Any]] | None:
    """Compatibility wrapper; validation requirements remain identical."""
    return validate_interview_facts_diagnostic(text, payload)["facts"]


def generate_interview_extraction(text: str, pending_slot: str | None) -> list[dict[str, Any]] | None:
    """Extract only from the current user turn, optionally after per-message consent.

    No session ID, transcript, previous answers, credentials or assessment are sent.
    """
    _EXTRACTION_DIAGNOSTIC.set(None)
    if not model_configured():
        return None
    schema = sorted(ALLOWED_EXTRACTION_FIELDS)
    user = {
        "pending_question_slot": pending_slot,
        "current_user_message": text,
        "allowed_fact_fields": schema,
    }
    messages = [
        {"role": "system", "content": (
            "استخرج الوقائع التي صرّح بها المستخدم في الرسالة الحالية فقط. أعد JSON فقط بالشكل "
            "{\"facts\":[{\"field\":\"اسم الحقل المسموح\",\"status\":\"reported أو denied أو unknown أو not_remembered أو declined\",\"value\":\"قيمة من النوع المحدد أدناه\",\"quote\":\"اقتباس حرفي\"}]}. "
            "يجب أن يحتوي كل عنصر على الحقول الأربعة field,status,value,quote فقط. كل quote مقطع حرفي متصل من رسالة المستخدم وبالحروف نفسها. "
            "أنواع القيمة والأمثلة منفصلة: للحقول الثنائية فقط "
            "(threat, impersonation, publication_threat, repeated, published, active_unauthorized_funds, fraud_description, account_access, harassment, contact_repeated): "
            "reported تكون value=true، وdenied تكون value=false؛ مثال {\"field\":\"threat\",\"status\":\"reported\",\"value\":true,\"quote\":\"في شخص عم يهددني\"}. "
            "للحقول النصية (ومنها narrative, platform, details, identity, contact, initial_time, latest_time, requested_amount, paid_amount, payment_method, deadline, evidence): "
            "reported تتطلب قيمة نصية غير فارغة ومسنَدة حرفيًا؛ مثال {\"field\":\"narrative\",\"status\":\"reported\",\"value\":\"في شخص عم يهددني على واتساب.\",\"quote\":\"في شخص عم يهددني على واتساب.\"}. "
            "قيمة narrative عند reported يجب أن تساوي quote حرفيًا. لا تضع null أو قيمة فارغة لواقعة reported، ولا تعمم null على الحقول. "
            "للحالة غير المعلومة الصريحة فقط، استخدم مثالًا منفصلًا مثل "
            "{\"field\":\"identity\",\"status\":\"unknown\",\"value\":null,\"quote\":\"ما بعرف مين هو\"}; "
            "وللحقول النصية المنفية استخدم denied مع value=null، وللثنائية المنفية denied مع false. "
            "لا تستخدم unknown/not_remembered/declined لمجرد غياب المعلومة؛ يجب أن ينص الاقتباس على عدم المعرفة أو التذكر أو الرفض وأن يتعلق بالحقل. "
            "لا تخترع قيمة أو اقتباسًا. راعِ النفي والسياق، ولا تقبل ادعاءً إيجابيًا يناقض نفيًا صريحًا. "
            "لا تستنتج خطرًا مباشرًا أو درجة أو مرحلة أو تأكيدًا. لا تصدر حقلًا غير مذكور ولا يمثل جوابًا واضحًا للسؤال المعلق. "
            "المبالغ reported تتطلب الرقم نفسه في الرسالة؛ هوية المشتبه به ليست اسم المتضرر؛ repeated يصف تكرار الضرر لا تكرار طلب المال؛ contact ليس نوع رسالة عامًّا. "
            "إذا لم توجد حقيقة واضحة أعد {\"facts\":[]}. حقيقة واحدة فقط لكل field."
        )},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]
    payload = _chat(messages)
    if payload:
        diagnostic = validate_interview_facts_diagnostic(text, payload, pending_slot=pending_slot)
        facts = diagnostic["facts"]
        diagnostic["raw_response"] = payload
        _EXTRACTION_DIAGNOSTIC.set(diagnostic)
    else:
        facts = None
        trace = last_call_trace() or {}
        _EXTRACTION_DIAGNOSTIC.set({
            "accepted": False, "facts": None,
            "rejection_code": trace.get("failure_code") or "generation_failed",
            "rejection_detail": (f"No extraction response; provider status={trace.get('status', 'unknown')}; "
                                 f"failure_code={trace.get('failure_code') or 'unknown'}; "
                                 f"provider_error_type={trace.get('provider_error_type') or 'not_recorded'}."),
            "provider_error_type": trace.get("provider_error_type"),
            "raw_response": None,
        })
    trace = last_call_trace()
    if trace and trace.get("attempted"):
        diag = last_extraction_diagnostic() or {}
        accepted_count = len(facts or [])
        filtered = diag.get("filtered_facts", [])
        response_valid = bool(diag.get("accepted"))
        if not response_valid:
            outcome = "provider_failed" if not payload else "response_rejected"
            status = trace.get("status") if not payload else "response_rejected"
            failure_code = diag.get("rejection_code") or trace.get("failure_code") or "generation_failed"
        elif accepted_count and filtered:
            outcome, status, failure_code = "partially_extracted", "partially_accepted", "partial_fact_filter"
        elif accepted_count:
            outcome, status, failure_code = "extracted", "extracted", None
        elif filtered:
            outcome, status, failure_code = "all_facts_filtered", "all_facts_filtered", "all_facts_filtered"
        else:
            outcome, status, failure_code = "valid_empty", "valid_empty", "empty_extraction"
        _update_call_trace(result_accepted=accepted_count > 0,
                           response_valid=response_valid,
                           extraction_outcome=outcome,
                           status=status,
                           failure_code=failure_code,
                           fact_fields=[item["field"] for item in facts] if facts is not None else [],
                           filtered_fact_count=len(filtered),
                           filtered_fact_fields=[item["field"] for item in filtered],
                           filtered_fact_code_counts={code: sum(1 for item in filtered if item.get("filter_code") == code)
                                                      for code in sorted({item.get("filter_code", "other_validation") for item in filtered})})
    return facts
