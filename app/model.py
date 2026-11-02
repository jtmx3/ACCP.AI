from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import re
from typing import Any

BINARY_FIELDS = {
    "threat", "impersonation", "publication_threat", "repeated", "published",
    "active_unauthorized_funds", "fraud_description", "account_access", "harassment",
}
ALLOWED_EXTRACTION_FIELDS = {
    "narrative", "threat", "impersonation", "platform", "details", "publication_threat",
    "active_unauthorized_funds", "fraud_description", "initial_time", "latest_time", "repeated",
    "requested_amount", "paid_amount", "payment_method", "deadline", "published", "identity",
    "contact", "evidence", "account_access", "harassment", "contact_repeated",
}
ALLOWED_STATUSES = {"reported", "denied", "unknown", "not_remembered", "declined"}
FIELD_ANCHORS = {
    "threat": ("تهديد", "هدد", "يهدد", "ابتزاز", "يبتز"),
    "impersonation": ("انتحال", "منتحل", "ينتحل", "حساب مزيف", "باسمي"),
    "publication_threat": ("ينشر", "نشر", "بفضح", "يهدد"),
    "repeated": ("تكرر", "مستمر", "مرة ثانية", "رجع"),
    "published": ("نشر", "نزل", "انتشرت", "بعت الصور"),
    "active_unauthorized_funds": ("يسحب", "تنسحب", "بدون اذني", "دون اذني", "غير ماذون"),
    "fraud_description": ("نصب", "احتيال", "خدعني", "احتيال مالي"),
    "account_access": ("اخترق", "دخل على حساب", "سرق حساب"),
    "harassment": ("مضايقة", "تحرش", "يلاحق", "بلاحق", "بضايق"),
    "contact_repeated": ("تكرر", "رجع بعت", "عدة رسائل", "مرات"),
}


def model_configured() -> bool:
    return os.getenv("MODEL_PROVIDER", "rules").strip().lower() == "openai-compatible" and bool(os.getenv("MODEL_API_KEY"))


def _chat(messages: list[dict[str, str]]) -> str | None:
    if not model_configured():
        return None
    base = os.getenv("MODEL_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("MODEL_TIMEOUT_SECONDS", "15"))) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else None
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError, OSError):
        return None


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
    return validate_extractive_claims(draft, sources) if draft else None


def validate_interview_facts(text: str, payload: str) -> list[dict[str, Any]] | None:
    """Reject malformed facts, unsupported fields/statuses, and quotes not found verbatim in this turn."""
    try:
        parsed = json.loads(payload)
        facts = parsed.get("facts")
        if not isinstance(facts, list) or len(facts) > 24:
            return None
        checked: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fact in facts:
            if not isinstance(fact, dict) or set(fact) != {"field", "status", "value", "quote"}:
                return None
            field, status, value, quote = fact["field"], fact["status"], fact["value"], fact["quote"]
            if not isinstance(field, str) or field not in ALLOWED_EXTRACTION_FIELDS or field in {"direct_danger", "priority_score", "confirmed", "stage"}:
                return None
            if field in seen or status not in ALLOWED_STATUSES:
                return None
            if not isinstance(quote, str) or not quote or quote not in text:
                return None
            normalized_quote = quote.lower().replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
            anchors = FIELD_ANCHORS.get(field, ())
            if anchors and not any(anchor in normalized_quote for anchor in anchors):
                return None
            if status == "reported":
                if field in BINARY_FIELDS:
                    if value is not True:
                        return None
                elif not isinstance(value, str) or not value.strip():
                    return None
                if field in {"requested_amount", "paid_amount"}:
                    quote_digits = re.sub(r"\D", "", quote.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
                    value_digits = re.sub(r"\D", "", value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
                    if value_digits and value_digits not in quote_digits:
                        return None
            elif status == "denied":
                if field in BINARY_FIELDS and value is not False:
                    return None
                if field not in BINARY_FIELDS and value is not None:
                    return None
            elif value is not None:
                return None
            checked.append({"field": field, "status": status, "value": value, "quote": quote})
            seen.add(field)
        return checked
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def generate_interview_extraction(text: str, pending_slot: str | None) -> list[dict[str, Any]] | None:
    """Extract only from the current user turn, optionally after per-message consent.

    No session ID, transcript, previous answers, credentials or assessment are sent.
    """
    if not model_configured():
        return None
    schema = sorted(ALLOWED_EXTRACTION_FIELDS)
    user = {
        "pending_question_slot": pending_slot,
        "current_user_message": text,
        "allowed_fact_fields": schema,
    }
    messages = [
        {"role": "system", "content": "استخرج الوقائع المصرّح بها في الرسالة الحالية فقط، مع مراعاة النفي والاقتباس والافتراض واختلاف الجمل. أعد JSON فقط بالشكل {\"facts\":[{\"field\":\"...\",\"status\":\"reported|denied|unknown|not_remembered|declined\",\"value\":null,\"quote\":\"...\"}]}. كل quote يجب أن تكون مقطعًا حرفيًا متصلًا من نص المستخدم وبالحروف نفسها. لا تخترع قيمًا. لا تستنتج خطرًا مباشرًا أو درجة أو مرحلة أو تأكيدًا. للحقول الثنائية استخدم reported مع true أو denied مع false. للمجهول أو النفي للحقول النصية استخدم value=null. أخرج facts فارغة عند غياب معلومة واضحة. يمكن إخراج عدة حقائق إذا وردت في الرسالة، لكن حقيقة واحدة فقط لكل field."},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]
    payload = _chat(messages)
    return validate_interview_facts(text, payload) if payload else None
