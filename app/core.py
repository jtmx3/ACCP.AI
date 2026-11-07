from __future__ import annotations

import hashlib
import json
import re
import contextvars
from datetime import datetime, timezone
from typing import Any

POLICY_VERSION = "ACCP-TRIAGE-0.1.0"
SCHEMA_VERSION = 1

QUESTIONS: dict[str, str] = {
    "narrative": "احكيلي شو صار معك.",
    "unclassified_details": "ما قدرت أحدد نوع الواقعة من الوصف وحده. ممكن توضح شو حدث خطوة بخطوة أو شو الرسالة أو التصرف اللي وصلك؟",
    "direct_danger": "هل أنت بخطر مباشر الآن؟",
    "details": "شو قال لك تحديدًا أو شو صار؟",
    "initial_time": "متى بدأ التصرف أو وصلك أول تواصل؟ هذا غير وقت اكتشافك للواقعة.",
    "account_creation_time": "متى أُنشئ الحساب المزيف، بحسب ما تعرف؟ هذا يختلف عن وقت اكتشافك له.",
    "repeated": "هل تكرر التهديد أو المضايقة المؤثرة؟ ليس مجرد استمرار الرسائل أو التواصل.",
    "requested_amount": "هل طُلب منك مبلغ؟ إذا نعم، كم كان المبلغ المطلوب؟",
    "requested_amount_relation": "هل المبلغ الجديد مبلغ إضافي، أم تصحيح للمبلغ المطلوب السابق؟",
    "paid_amount": "هل دفعت أو حولت أي مبلغ؟ إذا نعم، كم؟",
    "paid_amount_relation": "هل المبلغ الجديد دفعة إضافية، أم تصحيح لمبلغ الدفع السابق؟",
    "payment_method": "بأي وسيلة دفعت؟ لا ترسل كلمة مرور أو رمز تحقق.",
    "deadline": "هل حدد الشخص مهلة أو موعدًا للدفع؟",
    "published": "هل نُشر شيء فعلًا، بحسب علمك؟",
    "identity": "هل بتعرف هوية الشخص، ولا المتوفر عندك بس اسم حساب أو رقم تواصل؟",
    "evidence": "شو الأدلة المتوفرة عندك بحسب إفادتك؟ لا ترفع أو ترسل محتوى حساس هون.",
}

LABELS = {
    "narrative": "وصف الواقعة", "unclassified_details": "توضيح واقعة غير مصنفة", "incident_type_review": "تصنيف نوع الواقعة (يحتاج مراجعة)",
    "threat": "وجود تهديد", "impersonation": "انتحال شخصية", "platform": "المنصة",
    "account_access": "دخول غير مصرح به للحساب الأصلي", "original_account_access_available": "القدرة على الدخول للحساب الأصلي", "harassment": "مضايقة مذكورة",
    "contact_repeated": "تكرار التواصل دون استنتاج تهديد", "publication_threat": "تهديد بالنشر",
    "direct_danger": "الخطر المباشر الحالي", "active_unauthorized_funds": "أموال معرضة حاليًا لتصرف غير مأذون",
    "fraud_description": "وصف خداع مالي", "details": "تفاصيل الواقعة",
    "initial_time": "وقت بدء التصرف أو أول تواصل", "discovery_time": "وقت اكتشاف الواقعة",
    "account_creation_time": "وقت إنشاء الحساب المزيف", "latest_time": "أحدث تواصل", "repeated": "تكرار التهديد أو المضايقة",
    "contact_repeated": "تكرار التواصل دون استنتاج تهديد", "harassment": "مضايقة مذكورة",
    "requested_amount": "المبلغ المطلوب", "paid_amount": "المبلغ المدفوع", "payment_method": "وسيلة الدفع",
    "requested_amount_relation": "علاقة المبلغ المطلوب السابق", "paid_amount_relation": "علاقة مبلغ الدفع السابق",
    "deadline": "مهلة الدفع", "published": "حدوث النشر", "identity": "هوية الشخص",
    "contact": "معرّف التواصل", "contact_identifier_disclosure": "الإفصاح عن معرّف التواصل",
    "evidence": "الأدلة المذكورة", "additional_details": "تفاصيل إضافية من المستخدم",
    "jurisdiction": "الدولة أو الاختصاص",
}

QUESTION_RATIONALES = {
    "narrative": ("لفهم الواقعة قبل طرح أسئلة أدق", "لم يظهر بعد وصف كافٍ لما حدث"),
    "unclassified_details": ("لتحديد نوع الواقعة من وصف المستخدم لا من التخمين", "الوصف الحالي لا يحدد تصرفًا أو واقعة واضحة"),
    "direct_danger": ("للتحقق من الحاجة إلى مسار السلامة الآن", "لم تُحسم حالة الخطر المباشر الحالي"),
    "details": ("لفهم التصرف أو الرسالة المحددة", "نوع الواقعة معروف لكن تفاصيلها المؤثرة ما زالت ناقصة"),
    "initial_time": ("لفصل وقت بدء التصرف عن وقت اكتشافه", "وقت البداية أو أول تواصل غير معلوم"),
    "account_creation_time": ("لعدم استنتاج إنشاء الحساب من وقت اكتشافه", "وقت إنشاء الحساب غير معلوم على نحو مستقل"),
    "repeated": ("للتمييز بين تكرار الضرر واستمرار التواصل فقط", "لم يُحسم هل تكرر التهديد أو المضايقة"),
    "requested_amount": ("لتسجيل المبلغ المطلوب دون تخمين", "ذُكر طلب مالي لكن رقمه غير معلوم أو غير مسجل"),
    "requested_amount_relation": ("لمنع جمع مبلغين أو استبدال السابق دون سند", "العلاقة بين المبلغ الجديد والمطلوب السابق غير محسومة"),
    "paid_amount": ("لفصل ما دُفع فعليًا عن المبلغ المطلوب", "لم يُحسم وقوع الدفع أو مقداره"),
    "paid_amount_relation": ("لمنع تكرار احتساب التحويل نفسه", "العلاقة بين التحويل الجديد والدفع السابق غير محسومة"),
    "payment_method": ("لتوثيق الوسيلة المذكورة دون طلب بيانات دخول", "وسيلة الدفع غير مذكورة أو ما زالت متعارضة"),
    "deadline": ("للتمييز بين طلب المال ووجود مهلة", "لم تُذكر مهلة الدفع أو نفيها بوضوح"),
    "published": ("للفصل بين التهديد بالنشر وحدوثه فعلًا", "حدوث النشر لم يُحسم بحسب الإفادة"),
    "identity": ("لتمييز هوية الفاعل عن قناة التواصل المتاحة", "هوية الفاعل أو معرف التواصل ما زال غير محسوم"),
    "evidence": ("لتوثيق ما ذكر المستخدم توفره دون طلب رفعه", "نوع الأدلة المتوفرة أو نفيها لم يُحسم"),
}

FIELD_NOTES = {
    "initial_time": "وقت بدء التصرف أو أول تواصل، وليس وقت اكتشاف الواقعة.",
    "discovery_time": "وقت اكتشاف المستخدم للواقعة؛ لا يدل وحده على وقت حدوثها أو إنشاء الحساب.",
    "account_creation_time": "وقت إنشاء الحساب المزيف؛ لا يُستنتج من وقت اكتشافه.",
}

FIELD_ORDER = ["narrative", "direct_danger", "details", "initial_time", "discovery_time", "account_creation_time",
               "repeated", "requested_amount", "paid_amount", "payment_method", "deadline", "published", "identity", "evidence"]
TERMINAL = {"reported", "denied", "unknown", "not_remembered", "declined", "unresolved", "conflict"}
_MODEL_MERGE_TRACE: contextvars.ContextVar[list[dict[str, Any]]] = contextvars.ContextVar("accp_model_merge_trace", default=[])


def last_model_merge_trace() -> list[dict[str, Any]]:
    """Internal per-turn model merge telemetry; never persisted or serialized in session state."""
    return [dict(item) for item in _MODEL_MERGE_TRACE.get()]


def _trace_model_merge(field: Any, disposition: str, reason: str, **details: Any) -> None:
    trace = list(_MODEL_MERGE_TRACE.get())
    trace.append({"field": field if isinstance(field, str) else None,
                  "disposition": disposition, "reason": reason, **details})
    _MODEL_MERGE_TRACE.set(trace)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    return text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")


def new_state(session_id: str) -> dict[str, Any]:
    stamp = now_iso()
    opening = {"id": "m1", "role": "assistant", "text": QUESTIONS["narrative"], "created_at": stamp, "kind": "question"}
    opening["question_plan"] = question_plan({"facts": {}, "money_events": []}, "narrative")
    return {"schema_version": SCHEMA_VERSION, "session_id": session_id, "stage": "interview", "revision": 0,
            "created_at": stamp, "updated_at": stamp, "messages": [opening], "facts": {}, "money_events": [],
            "money_corrections": [], "pending_guidance": None,
            "pending_slot": "narrative", "clarifications": {}, "assessment": None,
            "confirmed": False, "confirmed_digest": None, "confirmation_history": [],
            "summary_version": 1, "last_response": opening["text"]}


def _is_unknown(text: str) -> str | None:
    t = normalize(text)
    if any(x in t for x in ("ما بعرف", "مش عارف", "مش عارفه", "لا اعرف", "غير معروف", "مجهول", "ما عندي فكرة", "مش متاكد", "مش متاكده")):
        return "unknown"
    if any(x in t for x in ("ما بتذكر", "مش متذكر", "مش متذكره", "لا اتذكر", "ناسي", "ناسية")):
        return "not_remembered"
    if any(x in t for x in ("ما بدي اجاوب", "بفضل ما اجاوب", "افضل عدم الاجابة", "مش حاب اجاوب", "بفضل ما احكي")):
        return "declined"
    return None


_UNKNOWN_FIELD_CUES = {
    "identity": ("مين", "من هو", "هوية", "الشخص", "الفاعل", "هويته", "هويتها", "مجهول"),
    "evidence": ("دليل", "اثبات", "صور", "لقطات", "رسائل", "محادثات", "اثار"),
    "direct_danger": ("خطر", "امان", "سلامة", "بأذى", "باذى", "مهدد الان"),
    "initial_time": ("بدا", "بداية", "اول تواصل", "اول رسالة", "متى", "تاريخ", "وقت"),
    "discovery_time": ("اكتشفت", "اكتشاف", "انتبهت", "لاحظت", "متى", "تاريخ", "وقت"),
    "account_creation_time": ("انشا", "انشأ", "انشاء", "انعمل", "انشائه", "موعد انشائه", "متى", "تاريخ", "وقت"),
    "requested_amount": ("مبلغ", "دينار", "كم", "طلب", "المطلوب"),
    "paid_amount": ("دفعت", "حولت", "تحويل", "مدفوع", "دينار", "مبلغ"),
    "requested_amount_relation": ("مبلغ", "طلب", "المطلوب", "اضافي", "تصحيح", "نفس"),
    "paid_amount_relation": ("دفعة", "دفع", "حولت", "تحويل", "اضافي", "تصحيح", "نفس"),
    "payment_method": ("وسيلة", "محفظة", "بطاقة", "تحويل", "بنك", "نقد"),
    "deadline": ("مهلة", "موعد", "قبل", "وقت للدفع", "تاريخ للدفع"),
    "published": ("نشر", "ينشر", "نزل", "انتشر"),
    "repeated": ("تكرر", "تكرار", "مرة ثانية", "مره ثانيه", "رجع"),
}


def _unknown_status_for_pending(text: str, pending: str | None,
                                status: str | None = None) -> str | None:
    status = status if status is not None else _is_unknown(text)
    if not status:
        return None
    if not pending:
        return status
    target = "requested_amount" if pending == "requested_amount_relation" else (
        "paid_amount" if pending == "paid_amount_relation" else pending)
    for clause in _clauses(text):
        clause_status = _is_unknown(clause)
        if not clause_status:
            continue
        normalized = normalize(clause)
        matched_fields = {field for field, markers in _UNKNOWN_FIELD_CUES.items()
                          if any(normalize(marker) in normalized for marker in markers)}
        if target == "identity" and any(marker in normalized for marker in ("متى", "متي", "وقت", "تاريخ", "موعد")) \
                and any(marker in normalized for marker in ("انشا", "انشاء", "انعمل", "عمله")) \
                and "هوية" not in normalized and "مين" not in normalized:
            matched_fields.discard("identity")
        if not matched_fields or target in matched_fields:
            return clause_status
    return None


def _unknown_targets_publication(text: str) -> bool:
    publication_markers = ("نشر", "ينشر", "نزل", "انتشر")
    unknown_cues = ("اذا", "هل", "ما اذا", "لو", "صار", "حصل")
    for clause in _clauses(text):
        normalized = normalize(clause)
        if _is_unknown(clause) and any(marker in normalized for marker in publication_markers) \
                and any(marker in normalized for marker in unknown_cues):
            return True
    return False


def _yes_no(text: str) -> bool | None:
    t = normalize(text)
    if re.search(r"(^|[\s،,])(?:لا|لأ|ابدا|كلا|مش|ما في|ما كان|ما صار|ما حدد)(?:$|[\s،,.؛])", t):
        return False
    if re.search(r"(^|[\s،,])(?:نعم|اه|آه|ايوه|ايوا|صحيح|أكيد|اكيد)(?:$|[\s،,.؛])", t):
        return True
    return None


def _amount_matches(text: str) -> list[tuple[int, int, str]]:
    pattern = r"(?<!\d)(\d[\d,٫٬]*(?:\.\d+)?|[٠-٩][٠-٩,٫٬]*(?:\.[٠-٩]+)?)(?!\d)\s*(?:دينار|دنانير|د\.ا|شيكل|دولار|ريال|ليرة)?"
    results = []
    temporal_spans = []
    for time_pattern in globals().get("_TIME_EXPRESSION_PATTERNS", ()):
        temporal_spans.extend((item.start(), item.end()) for item in re.finditer(time_pattern, text))
    for match in re.finditer(pattern, text):
        if any(match.start() < end and match.end() > start for start, end in temporal_spans):
            continue
        amount = re.sub(r"\s+", " ", match.group(0).strip())
        amount = amount.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        results.append((match.start(), match.end(), amount))
    return results


def _first_amount(text: str) -> str | None:
    matches = _amount_matches(text)
    return matches[0][2] if matches else None


def _amount_after_marker(text: str, markers: tuple[str, ...]) -> str | None:
    normalized = normalize(text)
    amounts = _amount_matches(text)
    candidates: list[tuple[int, int, str]] = []
    for marker in markers:
        marker = normalize(marker)
        start = normalized.find(marker)
        if start < 0:
            continue
        end = start + len(marker)
        following = next((item for item in amounts if item[0] >= end), None)
        if following:
            candidates.append((following[0] - end, following[0], following[2]))
    return min(candidates)[2] if candidates else None


def _fact_value(state: dict[str, Any], key: str) -> dict[str, Any] | None:
    return state["facts"].get(key)


def _message_calendar_date(state: dict[str, Any], message_id: str | None) -> str | None:
    if not message_id:
        return None
    message = next((item for item in state.get("messages", []) if item.get("id") == message_id), None)
    if not message or not message.get("created_at"):
        return None
    try:
        return datetime.fromisoformat(str(message["created_at"]).replace("Z", "+00:00")).date().isoformat()
    except (TypeError, ValueError):
        return None


def _has_relative_temporal_anchor(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = normalize(value)
    return any(marker in normalized for marker in (
        "اليوم", "امس", "مبارح", "البارحه", "قبل", "منذ", "الاسبوع الماضي", "الشهر الماضي", "السنه الماضيه",
    ))


def _temporal_values_equivalent(state: dict[str, Any], old: dict[str, Any], value: Any,
                               message_id: str | None) -> bool:
    old_value = old.get("value")
    old_signature = _temporal_signature(old_value)
    new_signature = _temporal_signature(value)
    if old_signature is None or old_signature != new_signature:
        return False
    if _has_relative_temporal_anchor(old_value) or _has_relative_temporal_anchor(value):
        old_date = _message_calendar_date(state, old.get("message_id"))
        new_date = _message_calendar_date(state, message_id)
        return old_date is not None and old_date == new_date
    return True


def _record(state: dict[str, Any], key: str, status: str, value: Any, quote: str,
            message_id: str, question_id: str | None, reason: str = "user_statement",
            reference: dict[str, Any] | None = None) -> bool:
    if status not in TERMINAL:
        raise ValueError(f"Invalid fact status: {status}")
    old = _fact_value(state, key)
    reference_meta = {name: reference[name] for name in
                      ("operation", "target_fact_id", "target_event_id", "context_refs")
                      if reference and name in reference}
    temporal_keys = {"initial_time", "latest_time", "discovery_time", "account_creation_time"}
    if (old and key in temporal_keys and old.get("status") == status == "reported"
            and old.get("value") != value and _temporal_values_equivalent(state, old, value, message_id)):
        # Preserve the original wording/value and let the normal same-value path
        # attach this message as supporting evidence.
        value = old.get("value")
    if (old and old.get("status") != "conflict" and old.get("message_id") == message_id
            and reason == "optional_model_extraction"
            and (old.get("status") != status or old.get("value") != value)):
        alternatives = old.get("alternatives") or [{
            "status": old.get("status"), "value": old.get("value"), "quote": old.get("quote"),
            "message_id": old.get("message_id"), "source": old.get("source", "rules"),
        }]
        alternatives.append({"status": status, "value": value, "quote": quote,
                             "message_id": message_id, "source": "model", **reference_meta})
        old.setdefault("history", []).append({"status": old.get("status"), "value": old.get("value"),
            "quote": old.get("quote"), "message_id": old.get("message_id"), "at": now_iso(),
            "reason": "same_turn_rule_model_disagreement", "source": old.get("source", "rules")})
        old.update({"status": "conflict", "value": None, "alternatives": alternatives,
                    "quote": quote, "message_id": message_id, "question_id": question_id,
                    "updated_at": now_iso(), "source": "conflict"})
        return True
    temporal_field = key in {"direct_danger", "repeated", "active_unauthorized_funds", "latest_time",
                             "initial_time", "discovery_time", "account_creation_time"}
    if old and key in {"latest_time", "initial_time", "discovery_time", "account_creation_time"} \
            and old.get("status") in TERMINAL and old.get("status") != "conflict" \
            and (old.get("status") != status or old.get("value") != value):
        explicit_correction = reason == "temporal_clarification" or any(
            marker in normalize(quote) for marker in ("تصحيح", "قصدي", "اقصد", "المعلومة الصحيحة", "بدل"))
        if not explicit_correction:
            alternatives = old.get("alternatives", [{"status": old["status"], "value": old.get("value"),
                "quote": old.get("quote"), "message_id": old.get("message_id"), "source": old.get("source", "rules")}])
            alternatives.append({"status": status, "value": value, "quote": quote, "message_id": message_id,
                                 "source": "model" if reason == "optional_model_extraction" else "rules", **reference_meta})
            old.setdefault("history", []).append({"status": old["status"], "value": old.get("value"),
                "quote": old.get("quote"), "message_id": old.get("message_id"), "at": now_iso(),
                "reason": "conflicting_temporal_user_statement"})
            old.update({"status": "conflict", "value": None, "alternatives": alternatives,
                        "quote": quote, "message_id": message_id, "question_id": question_id,
                        "updated_at": now_iso(), "source": "conflict"})
            return True
        old.setdefault("history", []).append({"status": old["status"], "value": old.get("value"), "quote": old.get("quote"),
                                               "message_id": old.get("message_id"), "at": now_iso(),
                                               "reason": "temporal_clarification"})
    elif old and temporal_field and old.get("status") in TERMINAL and old.get("status") != "conflict" and (old.get("status") != status or old.get("value") != value):
        old.setdefault("history", []).append({"status": old["status"], "value": old.get("value"), "quote": old.get("quote"),
                                               "message_id": old.get("message_id"), "at": now_iso(),
                                               "reason": reason if reason != "user_statement" else "new_temporal_information"})
    elif old and old.get("status") in TERMINAL and old.get("status") != "conflict":
        same = old.get("status") == status and old.get("value") == value
        if same:
            if old.get("message_id") == message_id:
                return False
            supports = old.setdefault("supporting_statements", [])
            if not any(item.get("message_id") == message_id for item in supports):
                supports.append({"quote": quote, "message_id": message_id, "question_id": question_id, "at": now_iso(),
                                 **({"source": "model", **reference_meta} if reference_meta else {})})
                old["updated_at"] = now_iso()
                return True
            return False
        if key == "details" and old.get("status") == status == "reported" and isinstance(old.get("value"), str) and isinstance(value, str):
            supports = old.setdefault("supporting_statements", [])
            if not any(item.get("message_id") == message_id for item in supports):
                supports.append({"quote": quote, "message_id": message_id, "question_id": question_id, "at": now_iso(),
                                 **({"source": "model", **reference_meta} if reference_meta else {})})
                old["updated_at"] = now_iso()
                return True
            return False
        if old.get("status") in {"unknown", "not_remembered", "unresolved"} and status in {"reported", "denied"}:
            old.setdefault("history", []).append({"status": old["status"], "value": old.get("value"),
                "quote": old.get("quote"), "message_id": old.get("message_id"), "at": now_iso(),
                "reason": "later_supported_information"})
            old.update({"status": status, "value": value, "quote": quote, "message_id": message_id,
                        "question_id": question_id, "updated_at": now_iso(),
                        "source": "model" if reason == "optional_model_extraction" else "rules"})
            if reference_meta:
                old["reference"] = reference_meta
            return True
        is_correction = reason == "temporal_clarification" or any(
            w in normalize(quote) for w in ("تصحيح", "قصدي", "اقصد", "عدّل", "عدلي", "المعلومة الصحيحة", "بدل")
        )
        if not is_correction:
            alternatives = old.get("alternatives", [{"status": old["status"], "value": old.get("value"),
                "quote": old.get("quote"), "message_id": old.get("message_id"), "source": old.get("source", "rules")}])
            alternatives.append({"status": status, "value": value, "quote": quote, "message_id": message_id,
                                 "source": "model" if reason == "optional_model_extraction" else "rules",
                                 **reference_meta})
            history = old.setdefault("history", [])
            history.append({"status": old["status"], "value": old.get("value"), "quote": old.get("quote"), "message_id": old.get("message_id"), "at": now_iso(), "reason": "conflicting_statement"})
            old.update({"status": "conflict", "value": None, "alternatives": alternatives, "quote": quote,
                        "message_id": message_id, "question_id": question_id, "updated_at": now_iso()})
            return True
        old.setdefault("history", []).append({"status": old["status"], "value": old.get("value"), "quote": old.get("quote"),
                                               "message_id": old.get("message_id"), "at": now_iso(),
                                               "reason": "temporal_clarification" if reason == "temporal_clarification" else "explicit_correction"})
    elif old and old.get("status") == "conflict":
        alternatives = old.get("alternatives", [])
        explicit_correction = any(w in normalize(quote) for w in ("تصحيح", "قصدي", "اقصد", "عدّل", "عدلي", "المعلومة الصحيحة", "بدل"))
        matches = any(item.get("status") == status and item.get("value") == value for item in alternatives)
        if status in {"reported", "denied"} and not matches and not explicit_correction:
            alternatives.append({"status": status, "value": value, "quote": quote, "message_id": message_id,
                                 "source": "model" if reason == "optional_model_extraction" else "rules",
                                 **reference_meta})
            old["alternatives"] = alternatives
            old["quote"] = quote
            old["message_id"] = message_id
            old["updated_at"] = now_iso()
            return True
        old.setdefault("history", []).append({"status": "conflict", "alternatives": alternatives, "quote": old.get("quote"),
                                               "message_id": old.get("message_id"), "at": now_iso(), "reason": "conflict_resolved"})
    history = old.get("history", []) if old else []
    state["facts"][key] = {"field": key, "status": status, "value": value, "quote": quote,
                            "message_id": message_id, "question_id": question_id, "updated_at": now_iso(),
                            "source": "model" if reason == "optional_model_extraction" else "rules", "history": history}
    if reference_meta:
        state["facts"][key]["reference"] = reference_meta
    return True


def _set_answer(state: dict[str, Any], key: str, value: Any, status: str, text: str, mid: str,
                question_id: str | None, changed: set[str], reason: str = "user_statement") -> None:
    if _record(state, key, status, value, text, mid, question_id, reason):
        changed.add(key)


def _append_additional_detail(state: dict[str, Any], text: str, message_id: str, question_id: str | None) -> bool:
    fact = state["facts"].get("additional_details")
    if not fact:
        return _record(state, "additional_details", "reported", text, text, message_id, question_id)
    supports = fact.setdefault("supporting_statements", [])
    if any(item.get("message_id") == message_id for item in supports):
        return False
    supports.append({"quote": text, "message_id": message_id, "question_id": question_id, "at": now_iso()})
    fact["updated_at"] = now_iso()
    return True


def _threat_polarity(text: str, already_reported: bool) -> bool | None:
    clauses = _clauses(_strip_quoted_spans(text))
    result: bool | None = None
    for clause in clauses:
        if not any(term in clause for term in ("تهديد", "هدد", "يهدد", "بهدد", "ابتزاز", "يبتز")):
            continue
        denied = bool(re.search(r"(?:ما\s+(?:فيه?\s+)?|مش\s+(?:فيه?\s+)?|مو\s+(?:فيه?\s+)?|لا\s+يوجد\s+)(?:اي\s+)?(?:تهديد|ابتزاز)", clause) or
                       re.search(r"(?:ما|مش|مو|لم|لا)\s+(?:هددني|هدد|يهددني|يهدد|ابتزني|يبتزني)", clause))
        if denied:
            result = False
        elif (any(term in clause for term in ("هددني", "يهددني", "بهددني", "ابتزني", "يبتزني", "التهديد مش مستمر", "التهديد غير مستمر")) or
              re.search(r"(?:^|\s)(?:و)?هدد(?:ني)?\s+(?:انه|ينشر|يؤذي|سي|رح|راح|يلاحق|بانه)", clause) or
              (already_reported and "التهديد" in clause)):
            result = True
    return result


def _evidence_polarity(text: str, allow_bare: bool = False) -> tuple[bool, bool, list[str], bool]:
    positive: list[str] = []
    categories = {
        "صور أو لقطات شاشة": ("صور", "صوري", "صورتي", "لقطات", "سكرين"),
        "رسائل أو محادثات": ("رسائل", "رسالة", "محادث", "مسجات"),
        "إيصال أو سجل دفع": ("ايصال", "إيصال", "حوالة", "تحويل"),
    }
    negative_mentions = False
    unknown_mentions = False
    possession_markers = ("عندي", "معي", "احتفظت", "محتفظ", "حفظت", "لدي", "صورت الشاشة", "اخذت لقطة", "أخذت لقطة")
    for clause in _clauses(_strip_quoted_spans(text)):
        normalized_clause = normalize(clause)
        matched_labels = []
        for label, terms in categories.items():
            # Possessives such as "صورتي" refer to the impersonated profile
            # image, not automatically to evidence held by the user.
            explicit_term = any(term in normalized_clause for term in terms)
            if label == "صور أو لقطات شاشة" and explicit_term and not any(
                marker in normalized_clause for marker in ("لقطات", "سكرين", "صور شاشة", "صورة شاشة")
            ):
                explicit_term = bool(re.search(r"(?<!\w)(?:صور|صورة|صوري|صورتي)(?!\w)", normalized_clause))
            if explicit_term:
                matched_labels.append(label)
        if not matched_labels:
            continue
        if _is_unknown(clause):
            unknown_mentions = True
            continue
        if re.search(r"(?:ما\s+عندي|ليس\s+لدي|مش\s+عندي|ما\s+في|لا\s+يوجد|ولا\s+عندي)\s+[^،؛.]*", normalized_clause):
            negative_mentions = True
            continue
        has_possession = any(marker in normalized_clause for marker in possession_markers)
        if has_possession or allow_bare:
            positive.extend(matched_labels)
    return bool(positive), negative_mentions, list(dict.fromkeys(positive)), unknown_mentions


def _impersonation_polarity(text: str) -> bool | None:
    """Classify explicit identity claims, not mere name/photo resemblance."""
    normalized = normalize(_strip_quoted_spans(text))
    clauses = _clauses(_strip_quoted_spans(text))
    account_context = any(term in normalized for term in ("حساب", "بروفايل", "صفحه", "صفحة", "profile"))
    if not account_context:
        return None

    denied = bool(re.search(
        r"(?:ما\s+في|مش|مو|لا\s+يوجد)\s+(?:اي\s+)?(?:حساب\s+)?(?:مزيف|وهمي|منتحل)|"
        r"(?:ما\s+في|لا\s+يوجد)\s+حساب\s+منتحل|"
        r"(?:ما|مش|مو|لا)\s+(?:بحكي|بيحكي|يحكي|بقول|بيقول|يقول|بدعي|يدعي)\s+(?:للناس\s+)?(?:انه\s+)?انا",
        normalized,
    ))
    explicit_claim_markers = (
        "انتحال", "منتحل", "ينتحل", "انتحل", "ينتحل شخصيتي", "انتحال شخصيتي", "انتحال هويتي",
        "تنتحل دوري", "تنتحل هويتي", "بصفته انا", "بصفتي انا",
    )
    claim_pattern = re.compile(
        r"(?:بحكي|بيحكي|يحكي|بقول|بيقول|يقول|بدعي|يدعي|يتظاهر|يزعم)\s+"
        r"[^،؛.]{0,45}\b(?:انه|اني)\s+انا"
    )
    self_identification_pattern = re.compile(
        r"(?:يعرف|يعر[ّ]?ف|يقدم|يقدّم|يقدم)\s+(?:حال[هها]|نفس[هها])"
        r"[^،؛.]{0,45}(?:على\s+)?(?:انه|اني)\s+انا"
    )
    explicit_claim = any(
        (any(marker in normalize(clause) for marker in explicit_claim_markers)
         or claim_pattern.search(normalize(clause)) or self_identification_pattern.search(normalize(clause)))
        and not _is_unknown(clause)
        and not re.search(r"(?:ما|مش|مو|لا)\s+(?:بحكي|بيحكي|يحكي|بقول|بيقول|يقول|بدعي|يدعي|ينتحل)", normalize(clause))
        for clause in clauses
    )

    fan_or_lookalike = any(term in normalized for term in ("معجبين", "حساب معجب", "fan account", "مشابه", "شبيه"))
    uncertain_claim = any(_is_unknown(clause) and re.search(r"(?:يدعي|بدعي|يحكي|بقول|انه\s+انا)", normalize(clause)) for clause in clauses)
    fake_profile = bool(re.search(r"(?:حساب(?:ا)?|بروفايل|صفحه)\s+(?:ا)?(?:مزيف|وهمي)", normalized))
    identity_reference = any(term in normalized for term in ("باسمي", "اسمي", "صورتي", "صوري", "شخصيتي", "هويتي"))
    if denied:
        return False
    if explicit_claim:
        return True
    if fake_profile and identity_reference and not fan_or_lookalike and not uncertain_claim:
        return True
    return None


def _impersonation_behavior_described(text: str) -> bool:
    normalized = normalize(_strip_quoted_spans(text))
    if re.search(
        r"(?:بحكي|بيحكي|يحكي|بقول|بيقول|يقول|بدعي|يدعي|يتظاهر|يزعم)\s+"
        r"[^،؛.]{0,45}\b(?:انه|اني)\s+انا",
        normalized,
    ):
        return True
    if re.search(r"(?:يعرف|يعر[ّ]?ف|يقدم|يقدّم)\s+(?:حاله|نفسه)[^،؛.]{0,45}(?:انه|اني)\s+انا", normalized):
        return True
    return any(term in normalized for term in (
        "ينتحل شخصيتي", "انتحال شخصيتي", "ينتحل هويتي", "انتحال هويتي", "تنتحل دوري", "تنتحل هويتي",
        "بصفته انا", "بصفتي انا",
    ))


def _identity_actor_unknown(text: str) -> bool:
    identity_subject = re.compile(r"(?:مين|من)\s+(?:عامل|عمل|انشا|انشأ|مسوي|وراء|صاحب|يدير)|هوية\s+(?:الشخص|الفاعل)")
    person_reference = re.compile(r"(?:الشخص|الفاعل|فاعل|هويته|هويتها|مين\s+هو|مين\s+هي|مينه|مينها)")
    for clause in _clauses(_strip_quoted_spans(text)):
        normalized = normalize(clause)
        if not _is_unknown(clause):
            continue
        if identity_subject.search(normalized):
            return True
        if person_reference.search(normalized):
            creation_time_only = (any(marker in normalized for marker in ("متى", "متي", "وقت", "تاريخ"))
                                  and any(marker in normalized for marker in ("انشا", "انشأ", "انعمل", "عمله"))
                                  and "هوية" not in normalized)
            if not creation_time_only:
                return True
    return False


def _original_account_access_available(text: str) -> bool:
    for clause in _clauses(_strip_quoted_spans(text)):
        normalized_clause = normalize(clause)
        if not re.search(r"(?:حسابي|الحساب)\s+(?:ال)?اصلي", normalized_clause):
            continue
        if any(marker in normalized_clause for marker in ("بقدر ادخل", "اقدر ادخل", "استطيع الدخول", "بقدر افوت", "ما زال معي", "ما زلت ادخل")):
            return True
    return False


_NARRATIVE_STOPWORDS = {"في", "من", "على", "عن", "الى", "انا", "هو", "هي", "كان", "كانت", "هذا", "هذه", "مع", "بس", "لكن", "عندي", "علي", "ما", "لا", "لم", "مش", "مو"}
_NARRATIVE_LABELS = {"ابتزاز", "الابتزاز", "تهديد", "التهديد", "احتيال", "الاحتيال", "نصب", "اختراق", "انتحال", "مضايقة", "تحرش", "نشر"}


def _narrative_has_description(text: Any) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    tokens = re.findall(r"[\u0600-\u06ff0-9a-zA-Z]+", normalize(text))
    informative = {token for token in tokens if len(token) > 1 and token not in _NARRATIVE_STOPWORDS and token not in _NARRATIVE_LABELS}
    return len(informative) >= 2


def _specific_contact_identifier(text: str) -> str | None:
    handle = re.search(r"(?<!\w)@[A-Za-z0-9_.-]{2,}", text)
    if handle:
        return handle.group(0)
    phone = re.search(r"(?<!\w)(?:\+|00)?[0-9٠-٩][0-9٠-٩\s().-]{5,}[0-9٠-٩](?!\w)", text)
    if phone:
        return re.sub(r"\s+", " ", phone.group(0)).strip()
    named = re.search(r"(?:اسم المستخدم|username)\s*[:=]?\s*([\w.-]{2,})", text, re.IGNORECASE)
    return named.group(1) if named else None


def _contact_availability(text: str) -> str | None:
    normalized = normalize(text)
    if any(marker in normalized for marker in ("ما عندي رقم", "ما معي رقم", "ما في رقم", "لا يوجد رقم")):
        return None
    contact_patterns = (
        (r"(?<!\w)رقم\s+(?:ال)?واتساب(?!\w)", "رقم واتساب"),
        (r"(?<!\w)رقم\s+(?:ال)?تواصل(?!\w)", "رقم تواصل"),
        (r"(?<!\w)رقم\s+(?:ال)?حساب(?!\w)", "رقم حساب"),
        (r"(?<!\w)(?:ال)?حساب\s+(?:ال)?تواصل(?!\w)", "حساب تواصل"),
        (r"(?<!\w)اسم\s+(?:ال)?حساب(?!\w)", "اسم حساب"),
        (r"(?<!\w)اسم\s+(?:ال)?مستخدم(?!\w)", "اسم مستخدم"),
    )
    contact_type = next((label for pattern, label in contact_patterns if re.search(pattern, normalized)), None)
    available = any(marker in normalized for marker in ("عندي", "معي", "متوفر", "المتوفر", "احتفظت"))
    if not (contact_type and available):
        return None
    return _specific_contact_identifier(text) or f"{contact_type} متوفر بحسب الإفادة؛ المعرّف لم يُدرج"


def _declined_contact_identifier(text: str) -> bool:
    normalized = normalize(text)
    return bool(re.search(
        r"(?:ما بدي|مش حاب(?:ب)?|مش حابة|بفضل ما|افضل عدم)\s+"
        r"(?:اكتب(?:ه|ها)?|اذكر(?:ه|ها)?|اعطي(?:ه|ها)?|اشارك(?:ه|ها)?|ارسل(?:ه|ها)?|ابعث(?:ه|ها)?|ادخل(?:ه|ها)?)",
        normalized,
    ))


def _canonical_payment_method(text: str) -> str | None:
    normalized = normalize(text)
    clitic = r"(?:[وبل])?(?:ال)?"
    if re.search(rf"(?<!\w){clitic}محفظ[ةه]\s+(?:ال)?الكتروني[ةه](?!\w)", normalized):
        return "محفظة إلكترونية"
    if re.search(rf"(?<!\w){clitic}كليك(?!\w)", normalized):
        return "كليك"
    if re.search(rf"(?<!\w){clitic}بطاق[ةه](?!\w)", normalized):
        return "بطاقة"
    if re.search(rf"(?<!\w){clitic}(?:تحويل|حوال[ةه])\s+(?:ال)?بنكي(?:[ةه])?(?!\w)", normalized):
        return "تحويل بنكي"
    if re.search(rf"(?<!\w){clitic}بنك(?!\w)", normalized):
        return "بنك"
    if re.search(r"(?<!\w)paypal(?!\w)", normalized):
        return "PayPal"
    if re.search(r"(?<!\w)western\s+union(?!\w)", normalized):
        return "Western Union"
    if re.search(rf"(?<!\w){clitic}محفظ[ةه](?!\w)", normalized):
        return "محفظة (النوع غير محدد)"
    if re.search(rf"(?<!\w){clitic}حوال[ةه](?!\w)", normalized):
        return "حوالة (النوع غير محدد)"
    return None


def _incident_detail_candidate(text: str) -> bool:
    normalized = normalize(text)
    return len(text.strip()) > 7 and any(marker in normalized for marker in (
        "رسالة", "كتب", "قال", "اتصل", "هدد", "تهديد", "ينشر", "نشر", "صوري", "محادثات",
        "حساب", "دخل", "اخترق", "انتحال", "ينتحل", "اعلان", "رابط", "كلمة المرور", "غير كلمة",
        "استولى", "استخدم", "خدعني", "ابتزاز", "لاحق", "يضربني", "سيؤذيني", "يؤذيني",
    ))


_TIME_EXPRESSION_PATTERNS = (
    r"(?<!\w)(?:اليوم|أمس|امس|البارحة|مبارح)(?:\s+(?:حوالي|تقريباً|تقريبا|نحو|قرابة))?(?:\s+(?:على\s+)?(?:الساعة\s*)?[0-9٠-٩]{1,2}(?::[0-9٠-٩]{2})?(?:\s*(?:المسا|مسا|مساء|الصبح|الصباح|صباحا|بالليل|ليلا))?)?(?!\w)",
    r"(?<!\w)(?:الساعة\s*)?[0-9٠-٩]{1,2}:[0-9٠-٩]{2}(?!\w)",
    r"(?<!\w)(?:قبل|منذ)\s+(?:ساعتين|ساعة|ساعات|دقائق|دقيقة|يومين|أسبوعين|اسبوعين|أسبوع|اسبوع|شهرين|سنتين|[0-9٠-٩]+\s*(?:ثانية|دقيقة|ساعة|يوم|أيام|اسبوع|أسبوع|شهر|سنة)?)(?!\w)",
    r"(?<!\w)(?:الأسبوع|الاسبوع|الشهر|السنة)\s+(?:الماضي|الفائت)(?!\w)",
    r"(?<!\w)الساعة\s+[0-9٠-٩]{1,2}(?::[0-9٠-٩]{2})?(?!\w)",
    r"(?<!\w)بتاريخ\s+[0-9٠-٩]{1,4}[/-][0-9٠-٩]{1,2}(?:[/-][0-9٠-٩]{1,4})?(?!\w)",
)


def _time_expression(text: str) -> str | None:
    for pattern in _TIME_EXPRESSION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    stripped = text.lstrip()
    if stripped.startswith("و") and len(stripped) > 1:
        return _time_expression(stripped[1:])
    return None


def _temporal_clauses(text: str) -> list[str]:
    return [part.strip() for part in re.split(
        r"[,،؛.!?؟\n]+|(?=\s+(?:أما|اما|لكن|بينما|بس|غير أن|إلا أن)\s+)", text
    ) if part.strip()]


def _is_discovery_clause(clause: str) -> bool:
    normalized = normalize(clause)
    explicit = any(marker in normalized for marker in ("اكتشفت", "اكتشف", "انتبهت", "لاحظت"))
    found_profile = "لقيت" in normalized and any(marker in normalized for marker in ("حساب", "صفحه", "صفحة", "بروفايل"))
    return bool((explicit or found_profile) and _time_expression(clause))


_CREATION_MARKERS = ("انشا", "انشاء", "انشئ", "انشي", "انعمل", "انعملت", "انشائه", "انشايه",
                     "انشائها", "انشاءه", "created account")
_INITIAL_EVENT_MARKERS = ("اول", "بدا", "بداية", "اول تواصل", "اول رسالة", "اول تهديد", "وصلني")
_LATEST_EVENT_MARKERS = ("اخر", "احدث", "رجع", "كرر", "مرة ثانية", "مره ثانيه", "بعتلي", "ارسللي",
                         "تواصل معي مجددا", "اليوم رجع")


def _temporal_event_field(clause: str, pending: str | None = None) -> str | None:
    normalized = normalize(clause)
    if any(marker in normalized for marker in _CREATION_MARKERS):
        return "account_creation_time"
    if any(marker in normalized for marker in ("اكتشفت", "اكتشف", "انتبهت", "لاحظت")):
        return "discovery_time"
    if any(marker in normalized for marker in _INITIAL_EVENT_MARKERS):
        return "initial_time"
    if any(marker in normalized for marker in _LATEST_EVENT_MARKERS):
        return "latest_time"
    if pending in {"initial_time", "discovery_time", "account_creation_time", "latest_time"}:
        return pending
    return None


def _temporal_signature(value: Any) -> tuple[Any, ...] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = normalize(value).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    expression = _time_expression(normalized)
    if expression:
        normalized = normalize(expression).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    normalized = normalized.replace("البارحه", "امس").replace("مبارح", "امس")
    for spelling in ("المسا", "مسا", "مساء"):
        normalized = normalized.replace(spelling, "مساء")
    duration = re.search(r"\b(قبل|منذ)\s+(.+)", normalized)
    if duration:
        return ("relative_duration", duration.group(1), re.sub(r"\s+", " ", duration.group(2)).strip())
    day = next((marker for marker in ("اليوم", "امس", "الاسبوع الماضي", "الشهر الماضي", "السنه الماضيه") if marker in normalized), None)
    hour_match = re.search(r"(?:الساعه\s*)?(\d{1,2})(?::(\d{2}))?", normalized)
    hour = (int(hour_match.group(1)), int(hour_match.group(2) or 0)) if hour_match else None
    period = next((marker for marker in ("صباحا", "الصباح", "الصبح", "مسا", "المسا", "مساء", "بالليل", "ليلا") if marker in normalized), None)
    approximate = any(marker in normalized for marker in ("حوالي", "تقريبا", "نحو", "قرابه"))
    if day or hour or period:
        return (day, hour, period, approximate)
    return (re.sub(r"\s+", " ", normalized).strip(),)


def _temporal_candidate_matches(key: str, candidate: dict[str, Any], text: str,
                                pending: str | None, rule_fact: dict[str, Any] | None) -> tuple[bool, str]:
    if candidate.get("status") != "reported":
        return True, ""
    quote = candidate.get("quote")
    value = candidate.get("value")
    if not isinstance(quote, str) or not quote or quote not in text:
        return False, "temporal_quote_not_in_current_message"
    source_clauses = [clause for clause in _temporal_clauses(text) if quote in clause]
    if not source_clauses:
        return False, "temporal_quote_event_not_identified"
    fields = {_temporal_event_field(clause, pending) for clause in source_clauses}
    fields.discard(None)
    if fields != {key}:
        return False, "temporal_quote_event_mismatch"
    quoted_time = next((_time_expression(clause) for clause in source_clauses if _time_expression(clause)), None)
    proposed_signature = _temporal_signature(value)
    quoted_signature = _temporal_signature(quoted_time) if quoted_time else None
    if proposed_signature is None or quoted_signature is None or proposed_signature != quoted_signature:
        return False, "temporal_value_not_supported_by_quote"
    if rule_fact and rule_fact.get("message_id") and rule_fact.get("quote"):
        rule_quote = rule_fact.get("quote")
        if rule_quote in text and _temporal_event_field(rule_quote, pending) != key:
            return False, "temporal_rule_fact_event_mismatch"
    return True, ""


def _legacy_initial_time_is_discovery(fact: dict[str, Any] | None) -> bool:
    if not fact:
        return False
    quote = str(fact.get("quote") or fact.get("value") or "")
    normalized = normalize(quote)
    if not any(marker in normalized for marker in ("اكتشفت", "اكتشف", "انتبهت", "لاحظت")) or not _time_expression(quote):
        return False
    event_markers = ("بدا", "بدأ", "وصلني", "اول تواصل", "اول رسالة", "اول تهديد", "انشا", "انشئ", "انشي")
    return not any(marker in normalized for marker in event_markers)


def _extract_temporal_facts(state: dict[str, Any], text: str, mid: str, question_id: str | None,
                            pending: str | None, changed: set[str]) -> None:
    for clause in _temporal_clauses(text):
        normalized = normalize(clause)
        time_value = _time_expression(clause)
        discovery_clause = _is_discovery_clause(clause)
        if discovery_clause:
            _set_answer(state, "discovery_time", time_value, "reported", clause, mid, question_id, changed)
            if _legacy_initial_time_is_discovery(state["facts"].get("initial_time")):
                _set_answer(state, "initial_time", None, "unknown", clause, mid, question_id, changed,
                            reason="temporal_clarification")

        creation_clause = any(marker in normalized for marker in _CREATION_MARKERS)
        creation_unknown = creation_clause and _is_unknown(clause) and any(marker in normalized for marker in ("متى", "متي", "وقت", "تاريخ", "موعد"))
        if creation_unknown:
            quote_clause = re.sub(r"^\s*(?:لكن|بس)\s+", "", clause)
            _set_answer(state, "account_creation_time", None, "unknown", quote_clause, mid, question_id, changed)
            if "initial_time" not in changed and _legacy_initial_time_is_discovery(state["facts"].get("initial_time")):
                _set_answer(state, "initial_time", None, "unknown", quote_clause, mid, question_id, changed,
                            reason="temporal_clarification")
        elif creation_clause and time_value:
            _set_answer(state, "account_creation_time", time_value, "reported", clause, mid, question_id, changed)

        if pending == "account_creation_time" and not creation_clause and not discovery_clause:
            if _unknown_status_for_pending(clause, "account_creation_time"):
                _set_answer(state, "account_creation_time", None, "unknown", clause, mid, question_id, changed)
            elif time_value:
                _set_answer(state, "account_creation_time", time_value, "reported", clause, mid, question_id, changed)

        if time_value:
            event_field = _temporal_event_field(clause, pending)
            if event_field in {"initial_time", "discovery_time", "account_creation_time", "latest_time"}:
                if event_field == "initial_time" and discovery_clause:
                    continue
                if event_field == "latest_time":
                    _set_answer(state, "latest_time", time_value, "reported", clause, mid, question_id, changed)
                elif event_field == "initial_time" and not creation_clause and not discovery_clause:
                    _set_answer(state, "initial_time", time_value, "reported", clause, mid, question_id, changed)


def _discovery_is_only_known_time(text: str) -> bool:
    clauses = _temporal_clauses(text)
    has_discovery_time = any(
        _is_discovery_clause(clause)
        for clause in clauses
    )
    if not has_discovery_time:
        return False
    explicit_event_markers = ("بدا", "وصلني", "اول تواصل", "اول رسالة", "اول تهديد", "انشا", "انشاء", "انشئ", "انشي")
    return not any(any(marker in normalize(clause) for marker in explicit_event_markers) and _time_expression(clause)
                   for clause in clauses)


def _extract(state: dict[str, Any], text: str, mid: str, pending: str | None) -> set[str]:
    t = normalize(text)
    changed: set[str] = set()
    unknown_status = _is_unknown(text)
    pending_unknown_status = _unknown_status_for_pending(text, pending, unknown_status)
    question_id = state["messages"][-2]["id"] if len(state["messages"]) >= 2 and state["messages"][-2]["role"] == "assistant" else None
    _extract_temporal_facts(state, text, mid, question_id, pending, changed)

    # Preserve exact user wording as provenance; short answers are anchored to question_id.
    threat_polarity = _threat_polarity(text, state["facts"].get("threat", {}).get("value") is True)
    if threat_polarity is not None:
        if threat_polarity and (pending == "narrative" or not state["facts"].get("narrative")) and _narrative_has_description(text):
            _set_answer(state, "narrative", text, "reported", text, mid, question_id, changed)
        _set_answer(state, "threat", threat_polarity, "reported" if threat_polarity else "denied", text, mid, question_id, changed)
    impersonation_polarity = _impersonation_polarity(text)
    if impersonation_polarity is not None:
        _set_answer(state, "impersonation", impersonation_polarity,
                    "reported" if impersonation_polarity else "denied", text, mid, question_id, changed)
        if impersonation_polarity and _impersonation_behavior_described(text) and _narrative_has_description(text):
            _set_answer(state, "details", text, "reported", text, mid, question_id, changed)
    platform_names = (("واتساب", "واتساب"), ("whatsapp", "واتساب"), ("انستغرام", "إنستغرام"), ("انستجرام", "إنستغرام"), ("instagram", "إنستغرام"), ("فيسبوك", "فيسبوك"), ("facebook", "فيسبوك"), ("سناب", "سناب شات"), ("snapchat", "سناب شات"), ("تيك توك", "تيك توك"), ("tiktok", "تيك توك"))
    platform = next((name for token, name in platform_names if token in t), None)
    if platform:
        _set_answer(state, "platform", platform, "reported", text, mid, question_id, changed)
    if any(k in t for k in ("محادثات", "صوري", "صور خاصة", "ينشر", "ينشرها", "نشر", "فضح")):
        if any(normalize(k) in t for k in ("إذا ما", "إذا لم", "رح ينشر", "تهديد بالنشر")) or _threat_polarity(text, state["facts"].get("threat", {}).get("value") is True) is True:
            _set_answer(state, "details", text, "reported", text, mid, question_id, changed)
            _set_answer(state, "publication_threat", True, "reported", text, mid, question_id, changed)
            _set_answer(state, "threat", True, "reported", text, mid, question_id, changed)
        elif pending == "published" and not unknown_status and ("نشر" in t or "نزل" in t):
            yn = _yes_no(text)
            if yn is not None:
                _set_answer(state, "published", yn, "reported" if yn else "denied", text, mid, question_id, changed)
        if any(k in t for k in ("نشرها فعلا", "نشر فعلا", "نزلها", "انتشرت")):
            _set_answer(state, "published", True, "reported", text, mid, question_id, changed)
    if unknown_status and _unknown_targets_publication(text):
        _set_answer(state, "published", None, unknown_status, text, mid, question_id, changed)
    elif any(k in t for k in ("ما نشر", "ما نزل", "لم ينشر", "لم تنشر", "ما اننشرت")):
        _set_answer(state, "published", False, "denied", text, mid, question_id, changed)
    account_access_signal = any(normalize(marker) in t for marker in ("سرق", "دخل على حساب", "دخل حساب", "اخترق"))
    account_access_denial = any(normalize(marker) in t for marker in ("ما انسرق", "لم يسرق", "ما اخترق", "لم يخترق", "مش مخترق", "ما حدا دخل على حساب", "ما حدا دخل حساب", "ما دخل على حساب", "لم يدخل على حساب", "ما دخل حساب"))
    if account_access_signal and not account_access_denial:
        _set_answer(state, "account_access", True, "reported", text, mid, question_id, changed)
    if _original_account_access_available(text):
        _set_answer(state, "original_account_access_available", True, "reported", text, mid, question_id, changed)
    if any(k in t for k in ("عم يسحب", "تنسحب اموال", "تحويل بدون اذني", "تحويل دون اذني", "خصم بدون اذني", "سحب غير ماذون")):
        _set_answer(state, "active_unauthorized_funds", True, "reported", text, mid, question_id, changed)
    if any(normalize(k) in t for k in ("نصب علي", "اننصب علي", "خدعني حتى أدفع", "خداع مالي واضح", "احتيال مالي واضح")):
        _set_answer(state, "fraud_description", True, "reported", text, mid, question_id, changed)

    if pending == "narrative" and not state["facts"].get("narrative") and _narrative_has_description(text):
        _set_answer(state, "narrative", text, "reported", text, mid, question_id, changed)
    if pending == "unclassified_details" and _narrative_has_description(text) and impersonation_polarity is not True:
        _set_answer(state, "unclassified_details", text, "reported", text, mid, question_id, changed)

    if any(term in t for term in ("بلاحقني", "ملاحقة", "بضايقني", "تعرضت لمضايقة", "عم بتعرض لمضايقة", "تحرش بي", "بتحرش فيني")) and not any(term in t for term in ("ما في مضايقة", "مش مضايقة", "ما بتحرش")):
        _set_answer(state, "harassment", True, "reported", text, mid, question_id, changed)
    repeat_positive = False
    repeat_negative = False
    for clause in _clauses(_strip_quoted_spans(text)):
        if any(term in clause for term in ("مستمر", "تكرر", "كرر", "مرة ثانية", "مره ثانيه", "رجع بعت", "رجع ارسل", "رجع هدد", "رجع يهدد", "عاد هدد", "عاد يهدد")):
            if re.search(r"(?:ما|لم|مش|مو|لا)\s+(?:يتكرر|تكرر|كرر|مستمر|رجع)", clause) or "مش مستمر" in clause or "غير مستمر" in clause:
                repeat_negative = True
            else:
                repeat_positive = True
    has_harm_context = state["facts"].get("threat", {}).get("value") is True or state["facts"].get("harassment", {}).get("value") is True
    if repeat_positive and has_harm_context:
        _set_answer(state, "repeated", True, "reported", text, mid, question_id, changed)
    elif repeat_positive:
        _set_answer(state, "contact_repeated", True, "reported", text, mid, question_id, changed)
    elif repeat_negative and (has_harm_context or pending == "repeated"):
        _set_answer(state, "repeated", False, "denied", text, mid, question_id, changed)
    elif pending == "repeated" and (yn := _yes_no(text)) is not None:
        _set_answer(state, "repeated", yn, "reported" if yn else "denied", text, mid, question_id, changed)
    elif pending == "repeated" and pending_unknown_status:
        _set_answer(state, "repeated", None, pending_unknown_status, text, mid, question_id, changed)

    if pending in {"requested_amount_relation", "paid_amount_relation"}:
        if not _resolve_money_relation(state, text, mid, question_id, pending, changed) and pending_unknown_status:
            unresolved_relation = _unresolved_money_relation(state)
            if unresolved_relation and unresolved_relation[1] == pending:
                unresolved_relation[0]["relation"] = "unknown"
                unresolved_relation[0]["relation_quote"] = text
                unresolved_relation[0]["relation_message_id"] = mid
                changed.add("money_events")
                _set_answer(state, pending, None, pending_unknown_status, text, mid, question_id, changed)
    elif pending not in {"requested_amount_relation", "paid_amount_relation"}:
        unresolved_relation = _unresolved_money_relation(state)
        if unresolved_relation and _resolve_money_relation(state, text, mid, question_id, unresolved_relation[1], changed):
            changed.add("money_events")

    amount = _first_amount(text)
    payment_clauses = _payment_clauses(text)
    request_markers = ("طلب", "طالب", "ادفع", "مطلوب", "اذا ما حولت", "اذا لم تحول", "اذا ما دفعت", "اذا لم تدفع", "لو ما دفعت", "لو لم تدفع")
    request_records = [(clause, _amount_after_marker(clause, request_markers) or _first_amount(clause))
                       for clause in payment_clauses if _is_request_clause(clause)]
    paid_markers = ("دفعت", "حولتله", "حولت له", "حوالة", "حولت مبلغ", "تم التحويل", "حولت")
    paid_records = [(clause, _amount_after_marker(clause, paid_markers) or _first_amount(clause))
                    for clause in payment_clauses if _is_completed_payment_clause(clause)]
    unquantified_paid_clauses = [clause for clause in payment_clauses
                                 if _is_completed_payment_clause(clause) and not _first_amount(clause)]
    denied_request_signal = any(_denied_request(clause) for clause in payment_clauses)
    denied_payment_signal = not paid_records and any(
        _negated_payment(clause) and not _conditional_payment(clause)
        and not any(marker in clause for marker in ("مرة ثانية", "مره ثانيه", "مرة اخرى", "مره اخرى"))
        for clause in payment_clauses)
    state.setdefault("money_events", [])

    existing_requested = state["facts"].get("requested_amount")
    for clause, requested_value in request_records:
        if not requested_value:
            continue
        relation = _money_relation(clause, requested_value, existing_requested)
        if relation == "correction":
            requested_value = _inherit_correction_currency(requested_value,
                                                           (existing_requested or {}).get("value"))
            if _add_money_event(state, "requested", requested_value, relation, text, mid, question_id):
                changed.add("money_events")
                _set_answer(state, "requested_amount", requested_value, "reported", text, mid, question_id, changed)
                existing_requested = state["facts"].get("requested_amount")
            else:
                if _add_money_event(state, "requested", requested_value, "unresolved", text, mid, question_id):
                    changed.add("money_events")
                    event = state["money_events"][-1]
                    event["correction_target_ambiguous"] = sum(
                        item.get("kind") == "requested" and item.get("amount") is not None
                        and item.get("relation") in {"initial", "additional", "same"}
                        and not item.get("target_event_id")
                        for item in state["money_events"]
                    ) > 1
        elif relation == "initial":
            if _add_money_event(state, "requested", requested_value, relation, text, mid, question_id):
                changed.add("money_events")
            _set_answer(state, "requested_amount", requested_value, "reported", text, mid, question_id, changed)
            existing_requested = state["facts"].get("requested_amount")
        elif relation == "additional":
            if _add_money_event(state, "requested", requested_value, relation, text, mid, question_id):
                changed.add("money_events")
            if existing_requested and existing_requested.get("status") == "reported":
                _set_answer(state, "requested_amount", existing_requested.get("value"), "reported", text, mid, question_id, changed)
        elif relation == "same":
            _set_answer(state, "requested_amount", existing_requested.get("value"), "reported", text, mid, question_id, changed)
        else:
            if _add_money_event(state, "requested", requested_value, "unresolved", text, mid, question_id):
                changed.add("money_events")

    requested_signal = bool(request_records)
    if request_records and not any(value for _, value in request_records):
        if existing_requested and existing_requested.get("status") == "reported" and existing_requested.get("value"):
            _set_answer(state, "requested_amount", existing_requested.get("value"), "reported", text, mid, question_id, changed)
        elif pending == "requested_amount":
            _set_answer(state, "requested_amount", "المبلغ المطلوب غير محدد بحسب الإفادة", "reported", text, mid, question_id, changed)

    existing_paid = state["facts"].get("paid_amount")
    if denied_payment_signal:
        _set_answer(state, "paid_amount", None, "denied", text, mid, question_id, changed)
    for clause, paid_value in paid_records:
        if paid_value:
            relation = _money_relation(clause, paid_value, existing_paid)
            if relation == "correction":
                paid_value = _inherit_correction_currency(paid_value, (existing_paid or {}).get("value"))
                if _add_money_event(state, "paid", paid_value, relation, text, mid, question_id):
                    changed.add("money_events")
                    _set_answer(state, "paid_amount", paid_value, "reported", text, mid, question_id, changed)
                    existing_paid = state["facts"].get("paid_amount")
                else:
                    if _add_money_event(state, "paid", paid_value, "unresolved", text, mid, question_id):
                        changed.add("money_events")
                        event = state["money_events"][-1]
                        event["correction_target_ambiguous"] = sum(
                            item.get("kind") == "paid" and item.get("amount") is not None
                            and item.get("relation") in {"initial", "additional", "same"}
                            and not item.get("target_event_id")
                            for item in state["money_events"]
                        ) > 1
            elif relation == "initial":
                if _add_money_event(state, "paid", paid_value, relation, text, mid, question_id):
                    changed.add("money_events")
                _set_answer(state, "paid_amount", paid_value, "reported", text, mid, question_id, changed)
                existing_paid = state["facts"].get("paid_amount")
            elif relation == "additional":
                if _add_money_event(state, "paid", paid_value, relation, text, mid, question_id):
                    changed.add("money_events")
                if existing_paid and existing_paid.get("status") == "reported":
                    _set_answer(state, "paid_amount", existing_paid.get("value"), "reported", text, mid, question_id, changed)
            elif relation == "same":
                _set_answer(state, "paid_amount", existing_paid.get("value"), "reported", text, mid, question_id, changed)
            else:
                if _add_money_event(state, "paid", paid_value, "unresolved", text, mid, question_id):
                    changed.add("money_events")
        elif existing_paid and existing_paid.get("status") == "reported" and existing_paid.get("value"):
            _set_answer(state, "paid_amount", existing_paid.get("value"), "reported", text, mid, question_id, changed)
        elif pending == "paid_amount":
            _set_answer(state, "paid_amount", "دفع بحسب الإفادة؛ المبلغ غير محدد", "reported", text, mid, question_id, changed)

    if unquantified_paid_clauses and not any(value for _, value in paid_records) and not denied_payment_signal:
        existing_paid = state["facts"].get("paid_amount")
        if existing_paid and existing_paid.get("status") == "reported" and existing_paid.get("value"):
            _set_answer(state, "paid_amount", existing_paid.get("value"), "reported", text, mid, question_id, changed)
        else:
            _set_answer(state, "paid_amount", "دفع بحسب الإفادة؛ المبلغ غير محدد", "reported", text, mid, question_id, changed)
            if _add_unquantified_money_event(state, "paid", text, mid, question_id):
                changed.add("money_events")

    if pending == "paid_amount" and not paid_records and not denied_payment_signal:
        if pending_unknown_status:
            _set_answer(state, "paid_amount", None, pending_unknown_status, text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is False:
            _set_answer(state, "paid_amount", None, "denied", text, mid, question_id, changed)
        elif amount and not request_records:
            if _add_money_event(state, "paid", amount, "initial", text, mid, question_id):
                changed.add("money_events")
            _set_answer(state, "paid_amount", amount, "reported", text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is True:
            _set_answer(state, "paid_amount", "نعم، المبلغ غير محدد", "reported", text, mid, question_id, changed)

    if pending == "requested_amount" and not request_records:
        if pending_unknown_status:
            _set_answer(state, "requested_amount", None, pending_unknown_status, text, mid, question_id, changed)
        elif denied_request_signal or (yn := _yes_no(text)) is False:
            _set_answer(state, "requested_amount", None, "denied", text, mid, question_id, changed)
        elif amount and not paid_records:
            if _add_money_event(state, "requested", amount, "initial", text, mid, question_id):
                changed.add("money_events")
            _set_answer(state, "requested_amount", amount, "reported", text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is True:
            _set_answer(state, "requested_amount", "المبلغ المطلوب غير محدد بحسب الإفادة", "reported", text, mid, question_id, changed)

    if (method := _canonical_payment_method(text)):
        _set_answer(state, "payment_method", method, "reported", text, mid, question_id, changed)

    if pending == "deadline":
        if pending_unknown_status:
            _set_answer(state, "deadline", None, pending_unknown_status, text, mid, question_id, changed)
        else:
            no_deadline = any(k in t for k in ("ما حدد مهلة", "ما حدد موعد", "ما حدد وقت", "ما كان في مهلة", "بدون مهلة", "ما حدد"))
            yes_deadline = any(k in t for k in ("حدد مهلة", "حدد موعد", "مهلة يوم", "مهلة ساعة", "قبل يوم", "قبل ساعة"))
            if no_deadline:
                _set_answer(state, "deadline", None, "denied", text, mid, question_id, changed)
            elif yes_deadline:
                _set_answer(state, "deadline", text, "reported", text, mid, question_id, changed)
    elif any(k in t for k in ("ما حدد مهلة", "ما حدد موعد", "ما حدد وقت", "ما كان في مهلة", "بدون مهلة")):
        _set_answer(state, "deadline", None, "denied", text, mid, question_id, changed)
    elif any(k in t for k in ("حدد مهلة", "حدد موعد", "مهلة يوم", "مهلة ساعة")):
        _set_answer(state, "deadline", text, "reported", text, mid, question_id, changed)

    if pending == "published":
        if pending_unknown_status:
            _set_answer(state, "published", None, pending_unknown_status, text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is not None:
            _set_answer(state, "published", yn, "reported" if yn else "denied", text, mid, question_id, changed)
        elif any(k in t for k in ("نشر", "نزل", "بعت الصور", "انتشرت")):
            _set_answer(state, "published", True, "reported", text, mid, question_id, changed)

    if (contact_value := _contact_availability(text)):
        _set_answer(state, "contact", contact_value, "reported", text, mid, question_id, changed)
        if _declined_contact_identifier(text):
            _set_answer(state, "contact_identifier_disclosure", None, "declined", text, mid, question_id, changed)

    explicit_identity_unknown = _identity_actor_unknown(text)
    identity_unknown = explicit_identity_unknown or (pending == "identity" and pending_unknown_status is not None)
    if identity_unknown:
        _set_answer(state, "identity", None, "unknown" if explicit_identity_unknown else pending_unknown_status,
                    text, mid, question_id, changed)
    elif pending == "identity":
        if pending_unknown_status:
            _set_answer(state, "identity", None, pending_unknown_status, text, mid, question_id, changed)
        elif any(k in t for k in ("رقم", "حساب", "اسم مستخدم", "username")):
            identifier = _specific_contact_identifier(text)
            contact_value = identifier or "ذُكر وجود رقم أو حساب تواصل دون إدخال معرّفه"
            _set_answer(state, "contact", contact_value, "reported", text, mid, question_id, changed)
            if any(k in t for k in ("بس", "فقط", "ما بعرف مين", "ما بعرف هويته")):
                _set_answer(state, "identity", None, "unknown", text, mid, question_id, changed)
            elif any(k in t for k in ("بعرف هويته", "اعرف هويته", "بعرف الشخص")):
                _set_answer(state, "identity", "معروفة بحسب الإفادة", "reported", text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is not None:
            _set_answer(state, "identity", "معروف بحسب الإفادة" if yn else None, "reported" if yn else "denied", text, mid, question_id, changed)
        elif len(text) > 2:
            _set_answer(state, "identity", text, "reported", text, mid, question_id, changed)
    elif (identifier := _specific_contact_identifier(text)):
        _set_answer(state, "contact", identifier, "reported", text, mid, question_id, changed)

    evidence_positive, evidence_denied, evidence_types, evidence_unknown = _evidence_polarity(text, allow_bare=pending == "evidence")
    if pending == "evidence" or evidence_positive or evidence_denied:
        if evidence_unknown or (pending == "evidence" and pending_unknown_status):
            _set_answer(state, "evidence", None, evidence_unknown or pending_unknown_status,
                        text, mid, question_id, changed)
        elif evidence_positive:
            _set_answer(state, "evidence", "; ".join(evidence_types), "reported", text, mid, question_id, changed)
        elif evidence_denied:
            _set_answer(state, "evidence", None, "denied", text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is not None:
            _set_answer(state, "evidence", "متوفر، النوع غير محدد" if yn else None, "reported" if yn else "denied", text, mid, question_id, changed)

    # Generic terminal answer states, with provenance tied to the asked slot.
    if pending in QUESTIONS and pending not in changed and pending_unknown_status and pending not in ("narrative", "details"):
        _set_answer(state, pending, None, pending_unknown_status, text, mid, question_id, changed)
    if pending == "details" and not state["facts"].get("details") and _incident_detail_candidate(text) and unknown_status is None:
        _set_answer(state, "details", text, "reported", text, mid, question_id, changed)
    if pending == "direct_danger" and not state["facts"].get("direct_danger") and pending_unknown_status:
        _set_answer(state, "direct_danger", None, pending_unknown_status, text, mid, question_id, changed)
    if pending is None and state.get("stage") in {"review", "urgent_stop"} and any(k in t for k in ("تفاصيل", "الوصف", "كلامه", "المعلومة الصحيحة")) and any(k in t for k in ("تصحيح", "قصدي", "اقصد", "عدّل", "بدل")):
        _set_answer(state, "details", text, "reported", text, mid, question_id, changed)
    return changed


def _user_intent(text: str) -> str | None:
    t = normalize(text)
    assessment_terms = ("اولوية", "الاولويه", "الدرجه", "التصنيف", "التصنيف", "الفرز", "التقييم")
    why_terms = ("ليش", "لماذا", "كيف احتسب", "سبب")
    if (any(normalize(w) in t for w in ("ليش الأولوية", "ليش الدرجة", "لماذا الدرجة", "كيف احتسبت الأولوية", "سبب التصنيف"))
            or (any(term in t for term in why_terms) and any(term in t for term in assessment_terms))):
        return "assessment_why"
    if any(normalize(w) in t for w in ("ليش بتسأل", "ليش بدك تعرف", "لماذا تسأل", "ليش السؤال")):
        return "why"
    if any(normalize(w) in t for w in ("احفظ الجلسة", "وقف المقابلة", "توقف المقابلة", "بدي أوقف المقابلة", "خلص بدي أوقف", "بكمل بعدين", "نكمل لاحقًا")):
        return "pause"
    if any(normalize(w) in t for w in ("اعرض المعلومات", "ورجيني الملخص", "اعرض الملخص", "شو سجلت", "المعلومات اللي عندك")):
        return "show"
    return None


def _why_target_slot(text: str, pending: str | None) -> str:
    normalized = normalize(text)
    field_markers = (
        ("deadline", ("مهلة", "موعد للدفع", "قبل الدفع")),
        ("payment_method", ("وسيلة الدفع", "طريقة الدفع", "المحفظة", "وسيلة")),
        ("requested_amount", ("المبلغ المطلوب", "طلب المال", "المبلغ")),
        ("paid_amount", ("المبلغ المدفوع", "التحويل", "الدفعة", "دفعت")),
        ("evidence", ("الادلة", "الأدلة", "اللقطات", "اثبات")),
        ("identity", ("هوية", "مين الشخص", "هوية الشخص")),
        ("initial_time", ("بداية الواقعة", "بدأ التصرف", "اول تواصل")),
        ("account_creation_time", ("انشاء الحساب", "موعد انشائه")),
        ("direct_danger", ("الخطر المباشر", "خطر مباشر", "سلامتك")),
        ("repeated", ("التكرار", "تكرار التهديد", "تكرار المضايقة")),
        ("published", ("حدوث النشر", "اذا نشر", "النشر فعلا")),
    )
    for slot, markers in field_markers:
        if any(normalize(marker) in normalized for marker in markers):
            return slot
    return pending or "narrative"


def is_clear_user_question(text: str) -> bool:
    """Recognize clear questions in answer mode without reclassifying incident facts."""
    raw = _strip_quoted_spans(text).strip()
    if not raw or _is_unknown(raw) or _user_intent(raw) in {"pause", "show"}:
        return False
    normalized = normalize(raw)
    prefixes = ("كيف", "هل", "متى", "وين", "اين", "شو", "ايش", "اش", "ليش", "لماذا", "ماذا", "مين", "من", "كم",
                "ما معنى", "ما الذي", "ما هي", "ما هو", "ما الخطوات")

    def starts_with_question(clause: str) -> bool:
        clause = clause.strip()
        return any(clause == prefix or clause.startswith(prefix + " ") for prefix in prefixes)

    clauses = [part.strip() for part in re.split(r"[،,؛;\n]+", normalized) if part.strip()]
    if clauses and starts_with_question(clauses[0]):
        return True
    fact_markers = ("هددني", "يهددني", "تهديد", "ابتزاز", "ينتحل", "انتحال", "حساب مزيف", "خدعني", "نصب", "احتيال",
                    "دفعت", "حولت", "طلب مني", "سحب غير", "وصلتني", "نشر صوري")
    has_incident_assertion = any(normalize(marker) in normalized for marker in fact_markers)
    if len(clauses) > 1 and starts_with_question(clauses[-1]) and not has_incident_assertion:
        return True
    if raw.endswith(("?", "؟")) and not has_incident_assertion:
        return True
    return _user_intent(raw) == "why"


def _strip_quoted_spans(text: str) -> str:
    return re.sub(r"«[^»]*»|“[^”]*”|\"[^\"]*\"|'[^']*'", " ", text)


def _clauses(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,،؛.!?؟\n]+|\b(?:بس|لكن|غير أن|إلا أن)\b", normalize(text)) if part.strip()]


def _payment_clauses(text: str) -> list[str]:
    parts: list[str] = []
    for clause in _clauses(text):
        parts.extend(part.strip() for part in re.split(r"\s+(?=وانا\b|واني\b|وهو\b|وهي\b|ثم\b|بعدها\b|بعد ذلك\b)", clause) if part.strip())
    return parts


def _conditional_payment(clause: str) -> bool:
    return bool(re.search(r"\b(?:و)?(?:اذا|لو)\s+(?:ما|لم)\s+(?:ادفع|دفعت|احول|حولت(?:له|لها)?|حولت\s+له|تحول|تدفع)\b", clause))


def _negated_payment(clause: str) -> bool:
    return bool(re.search(r"(?:^|\s)(?:و)?(?:ما|لم|مش|مو|لا)\s+(?:(?:رح|راح)\s+)?(?:ادفع|دفعت|احول|حولت|حول)(?=\s|$)", clause))


def _denied_request(clause: str) -> bool:
    return any(marker in clause for marker in (
        "ما طلب", "لم يطلب", "لم اطلب", "ما طلب مني", "ما طلبت", "ما في طلب", "ما كان في طلب",
        "ما كان فيه طلب", "لم يكن هناك طلب",
    ))


def _is_request_clause(clause: str) -> bool:
    if _denied_request(clause):
        return False
    if _conditional_payment(clause):
        return True
    if any(marker in clause for marker in ("طلب مني", "طلبت مني", "يطلب مني", "تطلب مني", "طالبني", "مطلوب", "بدك تحول", "ادفع له", "ادفعله", "طلبه", "اللي طلبه")) or re.search(r"(?:^|\s|و)(?:طلب|طلبت|طالب)(?:ني)?(?:\s|$)", clause):
        return True
    if "ادفع" not in clause or _negated_payment(clause):
        return False
    if any(marker in clause for marker in ("خدعني حتى ادفع", "خدعني حتي ادفع", "حتى ادفع", "حتي ادفع")):
        return False
    return bool(_first_amount(clause) or any(marker in clause for marker in ("ادفع له", "ادفعله", "ادفع لها", "ادفعلها")))


def _has_unquantified_request(text: str) -> bool:
    request_markers = ("طلب", "طالب", "ادفع", "مطلوب", "اذا ما حولت", "اذا لم تحول",
                       "اذا ما دفعت", "اذا لم تدفع", "لو ما دفعت", "لو لم تدفع")
    for clause in _payment_clauses(text):
        denial_scopes_over_request = bool(re.search(
            r"^(?:و)?(?:ما في|لا يوجد|ما كان في|ما كان فيه|لم يكن هناك)\b.*(?:طلب|طالب|مطلوب)", clause))
        if (_is_request_clause(clause) and not _denied_request(clause) and not denial_scopes_over_request
                and not (_amount_after_marker(clause, request_markers) or _first_amount(clause))):
            return True
    return False


def _is_completed_payment_clause(clause: str) -> bool:
    if _conditional_payment(clause) or _negated_payment(clause):
        return False
    return any(marker in clause for marker in ("دفعت", "حولتله", "حولت له", "حوالة", "حولت مبلغ", "تم التحويل", "حولت", "وحولت"))


def _amount_digits(value: Any) -> str:
    if value is None:
        return ""
    normalized = str(value).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return re.sub(r"\D", "", normalized)


def _same_amount(left: Any, right: Any) -> bool:
    left_digits, right_digits = _amount_digits(left), _amount_digits(right)
    return bool(left_digits and right_digits and left_digits == right_digits)


def _money_currency(value: Any) -> str | None:
    normalized = normalize(str(value or ""))
    for currency, aliases in {
        "dinar": ("دينار", "دنانير", "د.ا"), "shekel": ("شيكل",),
        "dollar": ("دولار",), "riyal": ("ريال",), "lira": ("ليرة",),
    }.items():
        if any(normalize(alias) in normalized for alias in aliases):
            return currency
    return None


def _same_full_financial_amount(left: Any, right: Any) -> bool:
    if not _same_amount(left, right):
        return False
    return _money_currency(left) == _money_currency(right)


def _money_relation(text: str, amount: str, existing: dict[str, Any] | None) -> str:
    clause = normalize(text)
    has_current_reported_amount = bool(existing and existing.get("status") == "reported" and existing.get("value"))
    if has_current_reported_amount and any(marker in clause for marker in (
            "تصحيح", "المبلغ الصحيح", "بدل", "قصدي", "اقصد", "تعديل للمبلغ")):
        return "correction"
    if any(marker in clause for marker in ("زيادة", "اضافي", "اضافية", "كمان", "مبلغ ثاني", "مبلغ اخر", "دفعة ثانية", "دفعة اخرى")):
        return "additional"
    if not existing or existing.get("status") != "reported" or not existing.get("value"):
        return "initial"
    return "same" if _same_amount(amount, existing.get("value")) else "unresolved"


def _inherit_correction_currency(amount: str, current_value: Any) -> str:
    """A correction to the same amount field inherits its known currency if omitted."""
    if not isinstance(current_value, str) or not _amount_digits(amount):
        return amount
    currency_terms = ("دينار", "دنانير", "د.ا", "شيكل", "دولار", "ريال", "ليرة")
    if any(normalize(term) in normalize(amount) for term in currency_terms):
        return amount
    old_currency = next((term for term in currency_terms if normalize(term) in normalize(current_value)), None)
    return f"{amount} {old_currency}" if old_currency else amount


def _money_event_current_amount(state: dict[str, Any], event: dict[str, Any]) -> Any:
    corrections = [item for item in state.get("money_corrections", [])
                   if item.get("target_event_id") == event.get("id")]
    return corrections[-1].get("amount") if corrections else event.get("amount")


def _original_event_for_correction(state: dict[str, Any], kind: str) -> dict[str, Any] | None:
    """Return the uniquely supported original event for a field-level correction."""
    fact_key = "requested_amount" if kind == "requested" else "paid_amount"
    current_fact = state.get("facts", {}).get(fact_key, {})
    events = [event for event in state.get("money_events", [])
              if event.get("kind") == kind and event.get("amount") is not None
              and event.get("relation") in {"initial", "additional", "same"}
              and not event.get("target_event_id")]
    if not events:
        return None
    current_value = current_fact.get("value")
    matches = [event for event in events
               if _same_full_financial_amount(current_value, _money_event_current_amount(state, event))]
    if len(matches) == 1:
        return matches[0]
    source_id = current_fact.get("message_id")
    source_events = [event for event in state.get("money_events", [])
                     if event.get("message_id") == source_id and event.get("kind") == kind]
    source_roots = []
    for source_event in source_events:
        target_id = source_event.get("target_event_id") if source_event.get("relation") == "correction" else source_event.get("id")
        root = next((event for event in events if event.get("id") == target_id), None)
        if root and root not in source_roots:
            source_roots.append(root)
    return source_roots[0] if len(source_roots) == 1 else None


def _add_money_event(state: dict[str, Any], kind: str, amount: str, relation: str, quote: str,
                     message_id: str, question_id: str | None) -> bool:
    events = state.setdefault("money_events", [])
    if relation == "correction":
        existing = _original_event_for_correction(state, kind)
        if existing:
            corrections = state.setdefault("money_corrections", [])
            if any(item.get("target_event_id") == existing.get("id") and item.get("message_id") == message_id
                   and item.get("amount") == amount for item in corrections):
                return False
            previous_amount = _money_event_current_amount(state, existing)
            corrections.append({"id": f"money-correction-{len(corrections) + 1}",
                                "target_event_id": existing.get("id"), "kind": kind,
                                "previous_amount": previous_amount, "amount": amount,
                                "relation": "correction", "quote": quote, "message_id": message_id,
                                "question_id": question_id, "created_at": now_iso()})
            events.append({"id": f"money-{len(events) + 1}", "kind": kind, "amount": amount,
                           "relation": "correction", "quote": quote, "message_id": message_id,
                           "question_id": question_id, "created_at": now_iso(),
                           "target_event_id": existing.get("id")})
            return True
        return False
    if any(event.get("message_id") == message_id and event.get("kind") == kind and
           event.get("amount") == amount and event.get("relation") == relation for event in events):
        return False
    events.append({"id": f"money-{len(events) + 1}", "kind": kind, "amount": amount, "relation": relation,
                   "quote": quote, "message_id": message_id, "question_id": question_id, "created_at": now_iso()})
    return True


def _add_unquantified_money_event(state: dict[str, Any], kind: str, quote: str,
                                  message_id: str, question_id: str | None) -> bool:
    """Record that a request/payment was stated while preserving the absent amount as unknown."""
    events = state.setdefault("money_events", [])
    if any(event.get("message_id") == message_id and event.get("kind") == kind and event.get("amount") is None
           for event in events):
        return False
    events.append({"id": f"money-{len(events) + 1}", "kind": kind, "amount": None,
                   "relation": "initial", "quote": quote, "message_id": message_id,
                   "question_id": question_id, "created_at": now_iso(), "amount_status": "not_stated"})
    return True


def _unresolved_money_relation(state: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    for event in reversed(state.get("money_events", [])):
        if event.get("relation") == "unresolved":
            slot = "requested_amount_relation" if event.get("kind") == "requested" else "paid_amount_relation"
            return event, slot
    return None


def _resolve_money_relation(state: dict[str, Any], text: str, mid: str, question_id: str | None,
                            pending: str, changed: set[str]) -> bool:
    unresolved = _unresolved_money_relation(state)
    if not unresolved or unresolved[1] != pending:
        return False
    event, _ = unresolved
    normalized = normalize(text)
    correction_target = None
    if event.get("correction_target_ambiguous"):
        candidates = [item for item in state.get("money_events", [])
                     if item.get("kind") == event.get("kind") and item.get("amount") is not None
                     and item.get("relation") in {"initial", "additional", "same"}
                     and not item.get("target_event_id")]
        matching = [item for item in candidates
                    if _same_full_financial_amount(_money_event_current_amount(state, item), text)]
        if len(matching) == 1:
            correction_target = matching[0]
        else:
            ordinal = next((index for index, markers in enumerate((("الاول", "اول مبلغ", "اولى"),
                                                                    ("الثاني", "الثانية", "المبلغ الثاني")))
                            if any(marker in normalized for marker in markers)), None)
            if ordinal is not None and ordinal < len(candidates):
                correction_target = candidates[ordinal]
        if not correction_target:
            return False
        relation = "correction"
    elif any(marker in normalized for marker in ("نفس المبلغ", "نفس الدفعة", "المبلغ الاول", "الدفعة الاولى",
                                                 "مش دفعة جديدة", "ما في تحويل ثاني", "ما تغير", "هو نفسه")):
        relation = "same"
    elif any(marker in normalized for marker in ("زيادة", "اضافي", "اضافية", "كمان", "مبلغ ثاني", "مبلغ اخر", "دفعة ثانية", "دفعة اخرى")):
        relation = "additional"
    elif any(marker in normalized for marker in ("تصحيح", "تعديل", "بدل", "المبلغ الصحيح", "قصدي", "اقصد")):
        relation = "correction"
    else:
        return False
    event["relation"] = relation
    event["relation_quote"] = text
    event["relation_message_id"] = mid
    event["relation_updated_at"] = now_iso()
    if correction_target:
        event["target_event_id"] = correction_target.get("id")
        event["correction_target_ambiguous"] = False
    fact_key = "requested_amount" if event["kind"] == "requested" else "paid_amount"
    relation_key = pending
    if relation == "correction":
        old = state["facts"].get(fact_key)
        if correction_target:
            state.setdefault("money_corrections", []).append({
                "id": f"money-correction-{len(state.get('money_corrections', [])) + 1}",
                "target_event_id": correction_target.get("id"), "kind": event.get("kind"),
                "previous_amount": _money_event_current_amount(state, correction_target),
                "amount": event.get("amount"), "relation": "correction", "quote": event.get("quote"),
                "message_id": event.get("message_id"), "question_id": event.get("question_id"),
                "created_at": now_iso(), "resolved_by_message_id": mid,
            })
        if old:
            old.setdefault("history", []).append({"status": old.get("status"), "value": old.get("value"),
                "quote": old.get("quote"), "message_id": old.get("message_id"), "at": now_iso(), "reason": "explicit_amount_relation_correction"})
            old.update({"status": "reported", "value": event["amount"], "quote": event["quote"],
                        "message_id": event["message_id"], "updated_at": now_iso()})
            old.setdefault("supporting_statements", []).append({"quote": text, "message_id": mid, "question_id": question_id, "at": now_iso()})
    elif state["facts"].get(fact_key):
        fact = state["facts"][fact_key]
        supports = fact.setdefault("supporting_statements", [])
        if not any(item.get("message_id") == mid for item in supports):
            supports.append({"quote": text, "message_id": mid, "question_id": question_id, "at": now_iso()})
    _set_answer(state, relation_key, relation, "reported", text, mid, question_id, changed)
    changed.add("money_events")
    return True


def _is_hypothetical_or_metalinguistic(clause: str) -> bool:
    return any(marker in clause for marker in ("معني كلمة", "معني عبارة", "ماذا تعني", "ما معني", "مثال علي", "افترض", "تخيل", "لو كنت", "اذا كنت", "لو كان", "اذا كان", "هل تعني عبارة"))


def _danger_denial(clause: str) -> bool:
    return bool(re.search(r"(?:انا\s*)?(?:مش|مو|لست|ما\s+في|ما\s+عندي|لا\s+يوجد|لا\s+اشعر)\s+(?:في\s+)?(?:اي\s+)?ب?خطر(?:\s+مباشر)?", clause) or
                    re.search(r"(?:ما\s+في|لا\s+يوجد)\s+(?:اي\s+)?خطر\s+(?:مباشر\s+)?(?:علي|الان|هسا)", clause))


def _danger_affirmation(clause: str) -> bool:
    if _is_hypothetical_or_metalinguistic(clause) or _danger_denial(clause):
        return False
    first_person = any(marker in clause for marker in ("انا", "علي", "عندي", "بيتي", "قريب مني"))
    immediate = any(marker in clause for marker in ("مباشر", "الان", "هسا", "هلا", "هلأ", "حاليا", "اللحظة", "وصل", "جاي", "واقف", "قدام"))
    if ("خطر" in clause or "بخطر" in clause) and first_person and immediate:
        return True
    if any(marker in clause for marker in ("معه سلاح", "بيده سلاح", "رح يأذيني", "راح يأذيني", "سيؤذيني")) and immediate:
        return True
    return False


def classify_current_danger(text: str, pending: str | None = None) -> bool | None:
    """Return True/False only for a clear current first-person risk disclosure/answer."""
    text_without_quotes = _strip_quoted_spans(text)
    clauses = _clauses(text_without_quotes)
    # A concrete current physical danger in any clause takes priority over a separate denial.
    if any(_danger_affirmation(clause) for clause in clauses):
        return True
    if pending == "direct_danger":
        unknown = _unknown_status_for_pending(text, "direct_danger")
        if unknown:
            return None
        if (answer := _yes_no(text)) is not None and len(normalize(text).split()) <= 8:
            return answer
    if any(_danger_denial(clause) for clause in clauses):
        return False
    return None


def is_immediate_danger_disclosure(text: str, pending: str | None = None) -> bool:
    return classify_current_danger(text, pending) is True


def screen_current_danger(state: dict[str, Any], text: str, message: dict[str, Any], pending: str | None) -> bool:
    classification = classify_current_danger(text, pending)
    if classification is None:
        return False
    question_id = state["messages"][-2]["id"] if len(state["messages"]) >= 2 and state["messages"][-2]["role"] == "assistant" else None
    status = "reported" if classification else "denied"
    if _record(state, "direct_danger", status, classification, text, message["id"], question_id):
        state["confirmed"] = False
        state["confirmed_digest"] = None
        state["summary_version"] += 1
    if not classification:
        state["assessment"] = _assessment(state)
        return False
    state["stage"] = "urgent_stop"
    state["pending_slot"] = None
    state["assessment"] = _assessment(state)
    if (state.get("jurisdiction") or "").upper() == "JO":
        text_reply = "سلامتك أولًا. إذا كنت داخل الأردن وفي خطر مباشر، اتصل الآن بالرقم الموحد 911 أو اطلب مساعدة شخص موثوق قريب منك. لم يجرِ التطبيق اتصالًا أو بلاغًا نيابة عنك."
        try:
            from app.rag import load_corpus
            source = next(item for item in load_corpus()["sources"] if item["id"] == "JO-PSD-EMERGENCY-911-2026")
            citation = {key: source.get(key) for key in ("id", "title", "publisher", "url", "published_or_updated", "ingested_at", "geography", "content_version")}
            citations = [citation]
        except (StopIteration, OSError, ValueError, KeyError):
            citations = []
    else:
        text_reply = "سلامتك أولًا. إذا كنت بخطر مباشر، ابتعد لمكان آمن إن أمكن، وتواصل الآن مع خدمات الطوارئ المحلية أو شخص موثوق قريب منك. لا أستطيع تحديد رقم طوارئ من دون معرفة موقعك."
        citations = []
    response = _add_message(state, "assistant", text_reply, "safety")
    response["citations"] = citations
    response["external_model_used"] = False
    return True


def _has(state: dict[str, Any], key: str) -> bool:
    fact = state["facts"].get(key)
    if key == "narrative":
        if not fact:
            return False
        if fact.get("status") in {"unknown", "not_remembered", "declined", "unresolved"}:
            return True
        return fact.get("status") == "reported" and _narrative_has_description(fact.get("value"))
    if key == "requested_amount" and fact and fact.get("status") == "reported" and fact.get("value") == "المبلغ المطلوب غير محدد بحسب الإفادة":
        return False
    return bool(fact and fact.get("status") in TERMINAL)


def _reported_true(state: dict[str, Any], key: str) -> bool:
    fact = state["facts"].get(key, {})
    return fact.get("status") == "reported" and fact.get("value") is True


def _incident_profile(state: dict[str, Any]) -> dict[str, bool]:
    facts = state["facts"]
    context = normalize(" ".join(str(facts.get(key, {}).get("value", "")) for key in ("narrative", "details", "additional_details")))
    threat = _reported_true(state, "threat")
    impersonation = _reported_true(state, "impersonation")
    harassment = _reported_true(state, "harassment")
    account_access = _reported_true(state, "account_access")
    financial = any(_reported_true(state, key) for key in ("active_unauthorized_funds", "fraud_description"))
    financial = financial or any(facts.get(key, {}).get("status") == "reported" for key in ("requested_amount", "paid_amount", "payment_method"))
    publication = _reported_true(state, "publication_threat") or facts.get("published", {}).get("status") in {"reported", "denied"}
    publication = publication or (_narrative_has_description(context) and any(word in context for word in ("ينشر", "نشر", "انتشرت", "صور خاصة", "محادثات خاصة")))
    incident = threat or impersonation or harassment or account_access or financial or publication
    return {"threat": threat, "impersonation": impersonation, "harassment": harassment, "account_access": account_access,
            "financial": financial, "publication": publication, "incident": incident}


def _relevant_slots(state: dict[str, Any]) -> list[str]:
    profile = _incident_profile(state)
    slots = ["narrative"]
    if not profile["incident"]:
        if state["facts"].get("narrative", {}).get("status") != "unresolved":
            slots.append("unclassified_details")
        slots.append("incident_type_review")
    if profile["threat"]:
        slots.append("direct_danger")
    if profile["threat"] or profile["account_access"] or profile["financial"]:
        slots.extend(("details", "initial_time"))
    elif profile["impersonation"]:
        slots.append("details")
    elif profile["incident"]:
        slots.append("initial_time")
    if profile["threat"] or profile["harassment"]:
        slots.append("repeated")
    if profile["financial"]:
        slots.extend(("requested_amount", "paid_amount"))
        if state["facts"].get("paid_amount", {}).get("status") == "reported" and state["facts"].get("paid_amount", {}).get("value"):
            slots.append("payment_method")
        if state["facts"].get("requested_amount", {}).get("status") == "reported" and state["facts"].get("requested_amount", {}).get("value"):
            slots.append("deadline")
    if profile["publication"]:
        slots.append("published")
    if profile["threat"] or profile["impersonation"] or profile["account_access"]:
        slots.append("identity")
    if profile["incident"]:
        slots.append("evidence")
    unresolved_relation = _unresolved_money_relation(state)
    if unresolved_relation:
        slots.append(unresolved_relation[1])
    return list(dict.fromkeys(slots))


def _has_known(state: dict[str, Any], key: str) -> bool:
    fact = state["facts"].get(key)
    if key == "narrative":
        return bool(fact and fact.get("status") == "reported" and _narrative_has_description(fact.get("value")))
    return bool(fact and fact.get("status") in {"reported", "denied"})


def _next_slot(state: dict[str, Any]) -> str | None:
    facts = state["facts"]
    for key, fact in facts.items():
        if fact.get("status") == "conflict" and key in QUESTIONS:
            return key
    unresolved_relation = _unresolved_money_relation(state)
    if unresolved_relation:
        return unresolved_relation[1]
    for key in _relevant_slots(state):
        if key not in QUESTIONS:
            continue
        if not _has(state, key):
            return key
    return None


def _short_answer(text: str) -> str:
    t = normalize(text)
    if "دفعت" in t or "حولت" in t:
        return "فهمت إنك ذكرت الدفع. "
    if "واتساب" in t:
        return "فهمت إن الواقعة كانت على واتساب. "
    return "فهمت عليك. "


def _explain_why(slot: str) -> str:
    reasons = {
        "unclassified_details": "لأن الكلمة أو الوصف الحالي لا يكفي لتحديد ما حدث، ولا أريد افتراض نوع الواقعة.",
        "direct_danger": "حتى نعرف إذا في حاجة لخطوة سلامة عاجلة الآن.",
        "details": "حتى نوثق الكلام أو التصرف كما ورد، من دون ما نستنتج تفاصيل ما ذكرتها.",
        "initial_time": "حتى نوثق متى بدأ التصرف أو وصلك أول تواصل، لا متى اكتشفته.",
        "account_creation_time": "حتى نفصل وقت إنشاء الحساب عن وقت اكتشافه، ولا نفترض أن الوقتين واحد.",
        "repeated": "لأن استمرار التواصل أو تكراره قد يغيّر أولوية المراجعة.",
        "requested_amount": "حتى نوثق المبلغ المطلوب بشكل منفصل عن أي مبلغ دُفع.",
        "paid_amount": "حتى نميّز بين المبلغ المطلوب والمبلغ الذي دفعته فعلًا.",
        "payment_method": "حتى نعرف نوع وسيلة الدفع، من دون طلب كلمة مرور أو رمز تحقق.",
        "deadline": "حتى نوثق إذا كان في مهلة محددة، وهذا مختلف عن مجرد حصول الدفع.",
        "published": "حتى نفرّق بين التهديد بالنشر وحدوث النشر فعلًا.",
        "identity": "حتى نميّز بين معرفة هوية الشخص ووجود رقم أو حساب تواصل فقط.",
        "evidence": "حتى نوثق الأدلة المذكورة، من دون الادعاء إننا استلمناها أو فحصناها.",
    }
    return reasons.get(slot, "حتى نوثق المعلومات المهمة من دون افتراض شيء غير مذكور.")


def _assessment(state: dict[str, Any]) -> dict[str, Any]:
    facts = state["facts"]
    profile = _incident_profile(state)
    score: int | None = None
    priority_rule: str | None = None
    supporting_fields: list[str] = []
    route = "استيضاح أو مراجعة بشرية"
    reasons: list[str] = []
    danger = facts.get("direct_danger", {})
    repeated = facts.get("repeated", {})
    threat = facts.get("threat", {})
    details = facts.get("details", {})
    normalized_details = normalize(str(details.get("value") or ""))
    impersonation = facts.get("impersonation", {})
    if danger.get("status") == "reported" and danger.get("value") is True:
        score, route = 100, "توقف السلامة وتوجيه عاجل مناسب"
        priority_rule, supporting_fields = "R100_DIRECT_DANGER", ["direct_danger"]
        reasons.append("خطر مباشر حالي بحسب إفادة المستخدم")
    elif _reported_true(state, "active_unauthorized_funds"):
        score, route = 95, "مراجعة عاجلة وفق السياسة"
        priority_rule, supporting_fields = "R95_ACTIVE_FUNDS", ["active_unauthorized_funds"]
        reasons.append("أموال معرضة حاليًا لتحويل أو سحب دون إذن")
    elif repeated.get("status") == "reported" and repeated.get("value") is True and (profile["threat"] or profile["harassment"]):
        score, route = 90, "مراجعة بشرية بأولوية مرتفعة"
        priority_rule, supporting_fields = "R90_REPEATED_HARM", (["threat", "repeated"] if profile["threat"] else ["harassment", "repeated"])
        reasons.append("تكرار تهديد أو مضايقة مؤثرة مذكور صراحة بحسب الإفادة")
    elif _reported_true(state, "account_access") or _reported_true(state, "fraud_description"):
        score, route = 90, "مراجعة بشرية بأولوية مرتفعة"
        priority_rule = "R90_ACCOUNT_OR_FRAUD"
        supporting_fields = [key for key in ("account_access", "fraud_description") if _reported_true(state, key)]
        reasons.append("وصف مدعوم لدخول غير مصرح به أو خداع مالي")
    elif (profile["threat"] or profile["impersonation"]) and details.get("status") == "reported" and details.get("value") and any(normalize(term) in normalized_details for term in ("رسالة", "ينشر", "ينزل", "نشر", "محادثات", "حساب", "اسمي", "صورتي", "صوري", "هدد", "ابتزاز", "لاحق", "قتل", "سلاح", "دينار", "محفظة", "رقم", "سيؤذيني", "يؤذيني", "سيؤذي", "يضربني", "يقتلني")):
        score, route = 85, "مراجعة بشرية"
        priority_rule = "R85_DETAILED_THREAT_OR_IMPERSONATION"
        supporting_fields = (["threat", "details"] if profile["threat"] else ["impersonation", "details"])
        reasons.append("تهديد أو انتحال شخصية مدعوم بتفاصيل بحسب الإفادة")
    elif profile["threat"] or profile["impersonation"]:
        reasons.append("وردت إشارة إلى تهديد أو انتحال، لكن التفاصيل غير كافية لتطبيق درجة 85")
    elif not facts:
        reasons.append("المعلومات غير كافية لتحديد درجة")
    elif not profile["incident"]:
        reasons.append("لم يتحدد نوع الواقعة من القواعد؛ يلزم توضيح أو مراجعة بشرية، ولا تعني نسبة الاكتمال إثبات التصنيف")
    if facts.get("publication_threat", {}).get("value") is True:
        reasons.append("تهديد بالنشر مذكور؛ لا يثبت حدوث النشر")
    if impersonation.get("value") is True:
        reasons.append("وجود حساب منتحل بحسب الإفادة؛ لا يثبت اختراق الحساب الأصلي")
    if facts.get("paid_amount", {}).get("status") == "reported":
        reasons.append("دفع مالي مذكور بحسب الإفادة؛ لا يثبت انتهاء التهديد")
    if facts.get("published", {}).get("status") in {"unknown", "not_remembered", "declined"}:
        reasons.append("حدوث النشر غير معلوم أو لم تتم الإجابة عنه")
    unresolved_fields = [LABELS.get(key, key) for key, fact in facts.items() if fact.get("status") == "unresolved"]
    if unresolved_fields:
        reasons.append("بقيت حقول غير محسومة بعد محاولة التوضيح: " + "، ".join(unresolved_fields))
    # Completeness is about known fields, not confidence or proof.
    relevant = _relevant_slots(state)
    answered = sum(1 for k in relevant if _has_known(state, k))
    return {"policy_version": POLICY_VERSION, "priority_score": score, "priority_rule": priority_rule, "supporting_fields": supporting_fields,
            "priority_label": "عاجلة" if score == 100 else "مرتفعة" if score and score >= 90 else "عادية/تحتاج مراجعة" if score else "غير محددة",
            "route": route, "reasons": reasons, "completeness_percent": round(100 * answered / len(relevant)),
            "relevant_fields": relevant, "completeness_note": f"{answered} من {len(relevant)} الحقول ذات الصلة لها إفادة أو نفي صريح؛ المجهول وغير المحسوم لا يُحسبان كحقائق مثبتة.",
            "extraction_quality": "إفادة نصية تحتاج مراجعة المستخدم؛ لا يوجد تحقق مستقل",
            "independent_verification": False, "calibrated_probability": False}


def explain_assessment(state: dict[str, Any]) -> str:
    assessment = _assessment(state)
    score = assessment.get("priority_score")
    reasons = assessment.get("reasons", [])
    if score is None:
        body = "لا توجد درجة أولوية محددة في القواعد للمعلومات الحالية."
    else:
        body = f"الدرجة الحالية {score}/100 هي ناتج قاعدة فرز تجريبية، وليست احتمالًا إحصائيًا أو إثباتًا."
        fields = assessment.get("supporting_fields", [])
        if fields:
            body += " الحقول التي فعّلت القاعدة: " + "، ".join(LABELS.get(field, field) for field in fields) + "."
    danger = state.get("facts", {}).get("direct_danger", {})
    if danger.get("status") == "denied":
        body += " الخطر المباشر الحالي مسجل كمنفي بحسب إفادتك، ولا أستنتج منه سبب الدرجة الأخرى."
    if reasons:
        body += " سبب الفرز المسجل: " + "؛ ".join(reasons) + "."
    return body + " راجع المعلومات واطلب مراجعة بشرية؛ لا يقدّم التطبيق بلاغًا رسميًا أو نصيحة قانونية."


def _add_message(state: dict[str, Any], role: str, text: str, kind: str = "chat") -> dict[str, Any]:
    msg = {"id": f"m{len(state['messages']) + 1}", "role": role, "text": text, "created_at": now_iso(), "kind": kind}
    state["messages"].append(msg)
    state["last_response"] = text
    return msg


def question_plan(state: dict[str, Any], slot: str | None) -> dict[str, str] | None:
    """Explain the target and trigger for a rule-selected question without adding facts."""
    if not slot or slot not in QUESTIONS:
        return None
    reason, condition = QUESTION_RATIONALES.get(slot, ("لاستكمال المعلومة ذات الصلة", "المعلومة غير محسومة"))
    fact = state.get("facts", {}).get(slot, {})
    if fact.get("status") == "conflict" and slot in {"initial_time", "latest_time", "discovery_time", "account_creation_time"}:
        reason = "توجد إفادتان زمنيتان مختلفتان لهذا الحدث، ولا أختار إحداهما نيابةً عنك"
        condition = "وقت الحدث غير محسوم بين البدائل المسجلة"
    plan = {"slot": slot, "field_label": LABELS.get(slot, slot), "reason": reason, "condition": condition}
    unresolved = _unresolved_money_relation(state) if slot in {"requested_amount_relation", "paid_amount_relation"} else None
    if unresolved:
        plan["related_event_id"] = str(unresolved[0].get("id", ""))
    return plan


def _add_question_message(state: dict[str, Any], slot: str, text: str | None = None,
                          kind: str = "question") -> dict[str, Any]:
    fact = state.get("facts", {}).get(slot, {})
    if text is None and fact.get("status") == "conflict" and slot in {"initial_time", "latest_time", "discovery_time", "account_creation_time"}:
        alternatives = fact.get("alternatives", [])
        values = [str(item.get("value")) for item in alternatives if item.get("status") == "reported" and item.get("value")]
        choices = " أو ".join(dict.fromkeys(values))
        text = f"وصلتني أوقات مختلفة ل{LABELS.get(slot, 'هذا الحدث')}: {choices}. أي وقت تقصد؟ هذا اختلاف بين التفسيرات المسجلة، وليس حكمًا بأن إفادتك غير صحيحة."
    message = _add_message(state, "assistant", text if text is not None else QUESTIONS[slot], kind)
    message["question_plan"] = question_plan(state, slot)
    return message


def _finalize_changed_facts(state: dict[str, Any], changed: set[str]) -> None:
    if changed:
        state["confirmed"] = False
        state["confirmed_digest"] = None
        state["summary_version"] += 1
    state["assessment"] = _assessment(state)


def handle_turn(state: dict[str, Any], text: str, model_facts: list[dict[str, Any]] | None = None,
                message_interpretation: dict[str, Any] | None = None,
                fact_text: str | None = None) -> dict[str, Any]:
    _MODEL_MERGE_TRACE.set([])
    text = text.strip()
    if not text:
        raise ValueError("اكتب رسالة قبل الإرسال.")
    if len(text) > 5000:
        raise ValueError("الرسالة أطول من الحد المسموح (5000 حرف).")
    pending = state.get("pending_slot")
    previous_stage = state.get("stage")
    user_msg = _add_message(state, "user", text)
    if screen_current_danger(state, text, user_msg, pending):
        return state
    intent = _user_intent(text)
    why_reply = None
    actionable_goals = {"answer_pending_question", "correct_prior_fact", "add_fact", "unknown", "not_remembered",
                         "decline_disclosure", "confirm_or_reference_prior_fact"}
    if intent == "why" and actionable_goals.intersection((message_interpretation or {}).get("goals", [])):
        intent = None
    if intent == "pause":
        state["stage"] = "paused"
        state["pending_slot"] = pending
        _add_message(state, "assistant", "حفظت الجلسة. تقدر ترجع لها وتكمل من نفس المكان.", "status")
        return state
    if intent == "why":
        reason = _explain_why(_why_target_slot(text, pending))
        prompt = QUESTIONS.get(pending or "", "احكيلي أي تفصيل بتحب تضيفه.")
        _add_question_message(state, pending if pending in QUESTIONS else "narrative",
                              f"{reason}\n\n{prompt}")
        return state
    if intent == "show":
        state["stage"] = "review"
        state["pending_slot"] = None
        state["assessment"] = _assessment(state)
        _add_message(state, "assistant", "هذا ملخص أولي لما سجلته. راجعه وصحح أي معلومة قبل التأكيد.", "review")
        return state

    rule_text = (fact_text or text).strip()
    unquantified_request = _has_unquantified_request(rule_text)
    changed = _extract(state, rule_text, user_msg["id"], pending)
    question_id = state["messages"][-2]["id"] if len(state["messages"]) >= 2 and state["messages"][-2]["role"] == "assistant" else None
    for candidate in model_facts or []:
        key = candidate.get("field")
        if key in {"initial_time", "latest_time", "discovery_time", "account_creation_time"}:
            if key == "initial_time" and _discovery_is_only_known_time(text):
                _trace_model_merge(key, "ignored", "discovery_time_is_not_event_start")
                continue
            rule_fact = state["facts"].get(key)
            supported, rejection_reason = _temporal_candidate_matches(key, candidate, text, pending, rule_fact)
            if not supported:
                _trace_model_merge(key, "ignored", rejection_reason,
                                   merge_rejection_code="temporal_event_or_value_mismatch")
                continue
            if key in changed and rule_fact:
                same_semantics = (rule_fact.get("status") == candidate.get("status")
                                  and _temporal_signature(rule_fact.get("value"))
                                  == _temporal_signature(candidate.get("value")))
                if same_semantics:
                    _trace_model_merge(key, "agrees_with_rules", "same_turn_temporal_meaning_matches_rule_extraction",
                                       rule_extraction_agrees=True, rule_extraction_outcome="semantic_agreement")
                else:
                    _trace_model_merge(key, "ignored", "same_turn_temporal_interpretation_differs_from_supported_rule_fact",
                                       merge_rejection_code="temporal_interpretation_mismatch")
                continue
        if key in {"direct_danger", "priority_score", "confirmed", "stage"}:
            _trace_model_merge(key, "ignored", "engine_owned_safety_or_decision_field")
            continue
        # Amount facts must pass through the money-event relation parser below; otherwise
        # a model proposal can overwrite the initial request before clarification.
        if key in {"requested_amount", "paid_amount", "requested_amount_relation", "paid_amount_relation"}:
            rule_fact = state["facts"].get(key, {})
            agrees = rule_fact.get("status") == candidate.get("status") and rule_fact.get("value") == candidate.get("value")
            _trace_model_merge(key, "routed_to_rules", "relation_aware_money_event_parser",
                               rule_extraction_agrees=agrees,
                               rule_extraction_outcome="agrees" if agrees else "differs_or_no_fact_recorded")
            continue
        if key == "payment_method" and candidate.get("status") == "reported":
            quote = str(candidate.get("quote", ""))
            proposed_value = str(candidate.get("value", ""))
            quote_method = _canonical_payment_method(quote)
            proposed_method = _canonical_payment_method(proposed_value)
            value_appears_verbatim = normalize(proposed_value) in normalize(quote)
            if quote_method and proposed_method != quote_method and not value_appears_verbatim:
                _trace_model_merge(key, "ignored", "model_payment_method_value_not_supported_by_current_quote",
                                   merge_rejection_code="value_inconsistent")
                continue
        if key in changed:
            rule_fact = state["facts"].get(key, {})
            agrees = rule_fact.get("status") == candidate.get("status") and rule_fact.get("value") == candidate.get("value")
            if agrees:
                _trace_model_merge(key, "agrees_with_rules", "same_message_fact_matches_rule_extraction",
                                   rule_extraction_agrees=True, rule_extraction_outcome="agrees")
            else:
                if _record(state, key, candidate["status"], candidate["value"], candidate["quote"],
                           user_msg["id"], question_id, "optional_model_extraction", candidate):
                    changed.add(key)
                _trace_model_merge(key, "conflict_created", "same_turn_rule_model_disagreement_requires_user_resolution",
                                   rule_extraction_agrees=False, rule_extraction_outcome="conflict")
            continue
        if key == "details" and not _incident_detail_candidate(candidate.get("quote", "")):
            _trace_model_merge(key, "ignored", "payment_only_is_not_incident_detail",
                               merge_rejection_code="pending_context_conflict")
            continue
        if key == "narrative" and state["facts"].get("narrative") and pending != "narrative":
            correction_text = normalize(candidate.get("quote", ""))
            if not any(marker in correction_text for marker in ("تصحيح", "قصدي", "اقصد", "المعلومة الصحيحة")):
                _trace_model_merge(key, "ignored", "preserve_original_narrative_without_explicit_correction")
                continue
        old = state["facts"].get(key)
        was_correction = bool(old and any(marker in normalize(candidate.get("quote", "")) for marker in
                                           ("تصحيح", "قصدي", "اقصد", "عدّل", "عدلي", "المعلومة الصحيحة", "بدل")))
        if _record(state, key, candidate["status"], candidate["value"], candidate["quote"],
                   user_msg["id"], question_id, "optional_model_extraction", candidate):
            changed.add(key)
            disposition = "applied_conflict" if state["facts"].get(key, {}).get("status") == "conflict" else (
                "applied_correction" if was_correction and old else "applied")
            _trace_model_merge(key, disposition, "model_suggestion_recorded_in_case")
        else:
            _trace_model_merge(key, "ignored", "duplicate_or_no_state_change")
    addition_marker = any(marker in normalize(text) for marker in ("اضيف", "تفصيل اضافي", "معلومة اضافية", "نسيت اقول"))
    if pending is None and state.get("stage") in {"review", "urgent_stop"} and intent is None and (not changed or addition_marker):
        if _append_additional_detail(state, text, user_msg["id"], None):
            changed.add("additional_details")
    new_relation = _unresolved_money_relation(state)
    new_relation_this_turn = bool(new_relation and new_relation[0].get("message_id") == user_msg["id"])
    temporal_conflict_slot = next((key for key, fact in state.get("facts", {}).items()
                                   if fact.get("status") == "conflict"
                                   and key in {"initial_time", "latest_time", "discovery_time", "account_creation_time"}), None)
    if temporal_conflict_slot and intent is None:
        _finalize_changed_facts(state, changed)
        state["stage"] = "interview"
        state["pending_slot"] = temporal_conflict_slot
        _add_question_message(state, temporal_conflict_slot, None, "clarification")
        return state
    pending_still_needed = (pending in QUESTIONS and pending not in {"narrative", "unclassified_details"}
                            and not (new_relation_this_turn and pending not in {"requested_amount_relation", "paid_amount_relation"})
                            and pending in _relevant_slots(state) and not _has(state, pending))
    if pending_still_needed and intent is None:
        if changed - {pending}:
            _finalize_changed_facts(state, changed)
            state["pending_slot"] = pending
            prompt = "سجلت المعلومات التي أجابت عن حقول أخرى. بقي عندي هذا السؤال للتوضيح: " + QUESTIONS[pending]
            _add_question_message(state, pending, prompt, "clarification")
            return state
        # Do not let an unrelated useful fact move the conversation past its unresolved active question.
        attempts = state["clarifications"].get(pending, 0)
        if attempts < 1:
            state["clarifications"][pending] = attempts + 1
            _finalize_changed_facts(state, changed)
            state["pending_slot"] = pending
            prompt = "سجلت المعلومة التي أضفتها. بقي عندي هذا السؤال للتوضيح: " + QUESTIONS[pending]
            _add_question_message(state, pending,
                                  (why_reply + "\n\n" if why_reply else "") + prompt,
                                  "clarification")
            return state
        if pending in {"requested_amount_relation", "paid_amount_relation"}:
            unresolved_relation = _unresolved_money_relation(state)
            if unresolved_relation and unresolved_relation[1] == pending:
                unresolved_relation[0]["relation"] = "unresolved_after_clarification"
                unresolved_relation[0]["relation_quote"] = text
                unresolved_relation[0]["relation_message_id"] = user_msg["id"]
                changed.add("money_events")
        if _record(state, pending, "unresolved", None, text, user_msg["id"], state["messages"][-2]["id"] if len(state["messages"]) > 1 else None, "clarification_limit"):
            changed.add(pending)

    _finalize_changed_facts(state, changed)

    if state["facts"].get("direct_danger", {}).get("status") == "reported" and state["facts"].get("direct_danger", {}).get("value") is True:
        state["stage"] = "urgent_stop"
        state["pending_slot"] = None
        state["assessment"] = _assessment(state)
        _add_message(state, "assistant", "سلامتك أولًا. إذا كنت بخطر مباشر، ابتعد لمكان آمن إن أمكن، وتواصل مع خدمات الطوارئ المحلية أو شخص موثوق قريب منك الآن. ما بقدر أحدد رقم طوارئ من دون معرفة بلدك. حفظت إفادتك ويمكن استكمالها لاحقًا.", "safety")
        return state

    next_slot = ("requested_amount" if unquantified_request and not _has(state, "requested_amount")
                 else _next_slot(state))
    if next_slot:
        if previous_stage in {"review", "urgent_stop"} and pending is None:
            state["stage"] = "urgent_stop" if previous_stage == "urgent_stop" else "review"
            state["pending_slot"] = None
            state["assessment"] = _assessment(state)
            _add_message(state, "assistant", (why_reply + "\n\n" if why_reply else "") +
                         "سجلت التحديث. راجع الملخص الحالي وأي معلومات ما زالت غير معروفة قبل التأكيد.", "review")
            return state
        state["stage"] = "interview"
        state["pending_slot"] = next_slot
        response_kind = "clarification" if next_slot == pending and next_slot in {"narrative", "unclassified_details"} else "question"
        unresolved_money = _unresolved_money_relation(state)
        temporal_conflict = (state.get("facts", {}).get(next_slot, {}).get("status") == "conflict"
                             and next_slot in {"initial_time", "latest_time", "discovery_time", "account_creation_time"})
        question_text = None if temporal_conflict else QUESTIONS[next_slot]
        if unresolved_money and unresolved_money[0].get("correction_target_ambiguous"):
            kind_label = "مبلغًا مطلوبًا" if unresolved_money[0].get("kind") == "requested" else "دفعة"
            question_text = f"ذكرت أكثر من {kind_label}. أي مبلغ سابق تقصد تصحيحه؟ اذكر المبلغ أو ترتيبه، من دون إعادة إرسال معلومات غير لازمة."
        if temporal_conflict:
            _add_question_message(state, next_slot, None, response_kind)
        else:
            _add_question_message(state, next_slot, (why_reply + "\n\n" if why_reply else "") + question_text, response_kind)
        return state
    state["pending_slot"] = None
    state["stage"] = "review"
    state["assessment"] = _assessment(state)
    _add_message(state, "assistant", (why_reply + "\n\n" if why_reply else "") +
                 "اكتملت الأسئلة الأساسية بحسب المعلومات المتاحة. راجع الملخص، وصحح أو أضف تفاصيل قبل التأكيد.", "review")
    return state


def resume_state(state: dict[str, Any]) -> dict[str, Any]:
    if state["stage"] == "paused":
        state["stage"] = "interview"
        pending_guidance = state.get("pending_guidance")
        if isinstance(pending_guidance, dict) and pending_guidance.get("status") == "awaiting_platform":
            prompt = pending_guidance.get("clarification_text") or "لأي تطبيق تقصد؟ اذكر اسم التطبيق لأعرض مصدره الرسمي فقط."
            _add_message(state, "assistant", prompt, "guidance_clarification")
            return state
        slot = state.get("pending_slot") or _next_slot(state)
        state["pending_slot"] = slot
        text = QUESTIONS.get(slot, "تقدر تراجع الملخص أو تضيف معلومة جديدة.")
        _add_question_message(state, slot, "رجعنا للجلسة المحفوظة.\n\n" + text, "question")
    return state


def display_status(fact: dict[str, Any] | None) -> str:
    if not fact:
        return "غير مسجلة"
    statuses = {
        "reported": "مذكورة بحسب الإفادة", "denied": "منفية صراحة بحسب الإفادة",
        "unknown": "غير معروفة للمستخدم", "not_remembered": "لا يتذكرها المستخدم",
        "declined": "فضّل المستخدم عدم الإجابة", "unresolved": "غير محسومة بعد الاستيضاح",
        "conflict": "تعارض فعلي يحتاج إلى توضيح",
    }
    return statuses.get(fact.get("status"), "حالة غير متاحة")


def display_value(fact: dict[str, Any] | None) -> str:
    if not fact or fact.get("status") != "reported":
        return "لا توجد قيمة مثبتة لهذا الحقل"
    value = fact.get("value")
    if isinstance(value, bool):
        return "نعم بحسب الإفادة" if value else "لا بحسب الإفادة"
    return str(value) if value is not None else "غير محدد بحسب الإفادة"


def _report_correction_notes(state: dict[str, Any], message_times: dict[str, Any]) -> list[dict[str, Any]]:
    facts = state.get("facts", {})
    requested = facts.get("requested_amount", {})
    if requested.get("status") != "reported":
        return []
    current_amount = requested.get("value")
    if not current_amount:
        return []
    correction_event = next((item for item in reversed(state.get("money_corrections", []))
                             if item.get("kind") == "requested"
                             and _same_full_financial_amount(item.get("amount"), current_amount)), None)
    if not correction_event:
        return []
    root_event = next((item for item in state.get("money_events", [])
                       if item.get("id") == correction_event.get("target_event_id")
                       and item.get("kind") == "requested"), None)
    original_amount = root_event.get("amount") if root_event else None
    if not original_amount or _same_full_financial_amount(original_amount, current_amount):
        return []
    previous = next((item for item in reversed(requested.get("history", []))
                     if item.get("reason") in {"explicit_correction", "explicit_amount_relation_correction"}
                     and item.get("value") is not None), None)
    source_message_id = root_event.get("message_id") if root_event else None
    correction_message_id = correction_event.get("message_id")
    target_field = "requested_amount"
    for field in ("details", "narrative"):
        source_fact = facts.get(field, {})
        quote = source_fact.get("quote")
        if source_fact.get("message_id") == source_message_id and isinstance(quote, str) and any(
            _same_full_financial_amount(original_amount, amount) for _, _, amount in _amount_matches(quote)
        ):
            target_field = field
            break
    return [{
        "target_field": target_field,
        "corrected_field": "requested_amount",
        "current_value": current_amount,
        "previous_value": original_amount,
        "previous_quote": root_event.get("quote") if root_event else None,
        "previous_message_id": source_message_id,
        "previous_message_at": message_times.get(source_message_id),
        "immediate_previous_value": previous.get("value") if previous else None,
        "immediate_previous_quote": previous.get("quote") if previous else None,
        "correction_quote": correction_event.get("quote"),
        "correction_message_id": correction_message_id,
        "correction_message_at": message_times.get(correction_message_id),
        "money_event_id": correction_event.get("id"),
    }]


def report_data(state: dict[str, Any]) -> dict[str, Any]:
    assessment = _assessment(state)
    current_confirmation = confirmation_is_current(state)
    message_times = {message.get("id"): message.get("created_at") for message in state.get("messages", [])}
    facts = {}
    for key, fact in state.get("facts", {}).items():
        facts[key] = {**fact, "source_message_at": message_times.get(fact.get("message_id")),
                      "last_updated_at": fact.get("updated_at")}
    money_corrections = [{**item, "source_message_at": message_times.get(item.get("message_id"))}
                         for item in state.get("money_corrections", [])]
    corrections_by_event: dict[str, list[dict[str, Any]]] = {}
    for correction in money_corrections:
        corrections_by_event.setdefault(correction.get("target_event_id"), []).append(correction)
    money_events = []
    for event in state.get("money_events", []):
        root_event_id = event.get("target_event_id") if event.get("relation") == "correction" else event.get("id")
        corrections = corrections_by_event.get(root_event_id, []) if event.get("relation") != "correction" else []
        money_events.append({**event,
            "current_amount": corrections[-1].get("amount") if corrections else event.get("amount"),
            "corrections": corrections,
            "is_correction_record": event.get("relation") == "correction",
            "source_message_at": message_times.get(event.get("message_id")),
            "relation_message_at": message_times.get(corrections[-1].get("message_id")) if corrections else message_times.get(event.get("relation_message_id"))})
    unknown_statuses = {"unknown", "not_remembered", "declined"}
    missing_fields = [key for key in _relevant_slots(state) if not _has(state, key)]
    unknown_fields = [key for key, fact in facts.items() if fact.get("status") in unknown_statuses]
    unresolved_fields = [key for key, fact in facts.items() if fact.get("status") == "unresolved"]
    conflicting_fields = [key for key, fact in facts.items() if fact.get("status") == "conflict"]
    field_keys = sorted(set(facts) | set(_relevant_slots(state)))
    unresolved_money_events = [event for event in money_events
                               if event.get("relation") in {"unresolved", "unresolved_after_clarification"}]
    history = list(state.get("confirmation_history") or [])
    old_digest = state.get("confirmed_digest")
    if state.get("confirmed") and old_digest and not current_confirmation and not any(item.get("digest") == old_digest for item in history):
        confirmed_message = next((message for message in reversed(state.get("messages", [])) if message.get("kind") == "confirmation"), {})
        history.append({"summary_version": state["summary_version"], "digest": old_digest,
                        "confirmed_at": confirmed_message.get("created_at"), "legacy": True})
    return {"case_id": state["session_id"], "summary_version": state["summary_version"], "stage": state["stage"],
            "case_created_at": state.get("created_at"), "case_updated_at": state.get("updated_at"),
            "facts": facts, "money_events": money_events, "money_corrections": money_corrections, "assessment": assessment,
            "correction_notes": _report_correction_notes(state, message_times),
            "field_labels": {key: LABELS.get(key, "معلومة إضافية") for key in field_keys},
            "field_notes": dict(FIELD_NOTES),
            "missing_fields": missing_fields, "unknown_fields": unknown_fields,
            "unresolved_fields": unresolved_fields, "conflicting_fields": conflicting_fields,
            "unresolved_money_events": unresolved_money_events, "confirmed": current_confirmation,
            "confirmation_history": history,
            "generated_at": now_iso(), "notice": "هذا ملخص لإفادة المستخدم للمراجعة فقط. لا يمثل بلاغًا رسميًا أو تحققًا مستقلًا أو إثباتًا للواقعة."}


def render_report(state: dict[str, Any]) -> str:
    data = report_data(state)
    facts = data["facts"]
    lines = [f"# ملخص الحالة {state['session_id']}", "", f"**إصدار الملخص:** {state['summary_version']}  ",
             f"**وقت إنشاء التقرير:** {data['generated_at']}  ", f"**وقت آخر تحديث للحالة:** {data['case_updated_at'] or 'غير متاح'}  ",
             f"**التأكيد:** {'مؤكد من المستخدم' if data['confirmed'] else 'غير مؤكد'}", "",
             "## سجل التأكيد", ""]
    if data["confirmation_history"]:
        for item in data["confirmation_history"]:
            is_current = data["confirmed"] and item.get("digest") == state.get("confirmed_digest")
            status = "تأكيد النسخة الحالية" if is_current else "تأكيد سابق؛ لا يؤكد محتوى هذا الإصدار"
            lines.append(f"- الإصدار {item.get('summary_version', 'غير معروف')}: {status}؛ الوقت: {item.get('confirmed_at') or 'غير متاح'}")
    else:
        lines.append("- لا توجد تأكيدات سابقة محفوظة.")
    lines += ["", "## وصف الواقعة", "", display_value(facts.get("narrative")), "", "## التسلسل الزمني", "",
             f"- وقت اكتشاف الواقعة: {display_value(facts.get('discovery_time'))}",
             f"- وقت بدء التصرف أو أول تواصل: {display_value(facts.get('initial_time'))}",
             f"- أحدث تواصل: {display_value(facts.get('latest_time'))}", "", "## الوقائع ومصادرها", ""]
    ordered = ["threat", "impersonation", "platform", "details", "unclassified_details", "additional_details", "publication_threat", "initial_time", "discovery_time", "account_creation_time", "latest_time", "repeated", "requested_amount", "paid_amount", "payment_method", "deadline", "published", "identity", "contact", "evidence", "original_account_access_available", "account_access", "fraud_description", "active_unauthorized_funds", "direct_danger"]
    for key in ordered:
        fact = facts.get(key)
        if not fact:
            continue
        lines += [f"### {LABELS.get(key, key)}", f"- الحالة: {display_status(fact)}",
                  f"- القيمة: {display_value(fact)}"]
        if FIELD_NOTES.get(key):
            lines.append(f"- توضيح دلالي: {FIELD_NOTES[key]}")
        if fact.get("quote"):
            lines.append(f"- نص الإفادة: «{fact['quote']}»")
        for note in data["correction_notes"]:
            if note.get("target_field") == key:
                lines.append(
                    f"- توضيح مستند إلى تصحيح لاحق: المبلغ المطلوب المعتمد حاليًا "
                    f"{note['current_value']}؛ وكانت الإفادة الأصلية في هذا الموضع تذكر "
                    f"{note['previous_value']}. بقي الاقتباس الأصلي أعلاه حرفيًا دون تعديل."
                )
                if note.get("correction_quote"):
                    lines.append(f"- نص تصحيح المبلغ المطلوب: «{note['correction_quote']}»")
        if fact.get("message_id"):
            lines.append(f"- مرجع الرسالة: {fact['message_id']}")
        lines.append(f"- وقت الرسالة الأصلية: {fact.get('source_message_at') or 'غير متاح'}")
        lines.append(f"- وقت آخر تحديث للمعلومة: {fact.get('last_updated_at') or 'غير متاح'}")
        for previous in fact.get("history", []):
            old_value = previous.get("value")
            old_quote = previous.get("quote")
            lines.append(f"- سجل سابق ({previous.get('reason', 'تعديل')}): {old_value if old_value is not None else previous.get('status', 'غير محدد')}")
            if old_quote:
                lines.append(f"  - الإفادة السابقة: «{old_quote}»")
        for support in fact.get("supporting_statements", []):
            lines.append(f"- إفادة داعمة إضافية ({support.get('message_id', 'رسالة')}): «{support.get('quote', '')}»")
        if fact.get("alternatives"):
            lines.append(f"- عدد الإفادات البديلة المحفوظة: {len(fact['alternatives'])}")
        lines.append("")
    assessment = data["assessment"]
    lines += ["## الفرز", "", f"- السياسة: {assessment['policy_version']}",
              f"- أولوية المراجعة: {assessment['priority_score']}/100" if assessment["priority_score"] is not None else "- أولوية المراجعة: غير محددة",
              f"- المسار المقترح: {assessment['route']}", f"- اكتمال المعلومات: {assessment['completeness_percent']}% — {assessment['completeness_note']}",
              f"- جودة الاستخراج: {assessment['extraction_quality']}", "- التحقق المستقل: غير متوفر", "- الدرجة ليست احتمالًا لوقوع الواقعة أو ثبوتها.", ""]
    relation_labels = {"initial": "أولي", "additional": "إضافي", "same": "مماثل لمبلغ سابق",
                       "correction": "تصحيح", "unresolved": "غير محسوم", "unresolved_after_clarification": "غير محسوم بعد التوضيح",
                       "unknown": "غير معروف"}
    kind_labels = {"requested": "مطلوب", "paid": "مدفوع"}
    lines += ["", "## سجل الأحداث المالية", ""]
    if data["money_events"]:
        for event in data["money_events"]:
            original_amount = event.get("amount") or "غير محدد"
            current_amount = event.get("current_amount")
            if event.get("is_correction_record"):
                kind = kind_labels.get(event.get("kind"), "مالي")
                lines.append(f"- سجل تصحيح تدقيقي للمبلغ {kind}: القيمة المصححة {original_amount}؛ "
                             "ليس دفعة أو طلبًا ماليًا إضافيًا.")
                if event.get("quote"):
                    lines.append(f"  - نص التصحيح: «{event['quote']}»")
                continue
            current_note = f"؛ القيمة المعتمدة حاليًا بعد التصحيح: {current_amount} (كانت الإفادة الأصلية: {original_amount})" if current_amount and current_amount != event.get("amount") else ""
            lines.append(f"- مبلغ {kind_labels.get(event.get('kind'), 'مالي')}: {original_amount}{current_note}؛ العلاقة: {relation_labels.get(event.get('relation'), 'غير معروفة')}؛ وقت الرسالة الأصلية: {event.get('source_message_at') or 'غير متاح'}")
            if event.get("quote"):
                lines.append(f"  - نص الإفادة: «{event['quote']}»")
            for correction in event.get("corrections", []):
                lines.append(f"  - تصحيح لاحق: {correction.get('previous_amount')} ← {correction.get('amount')}؛ نص التصحيح: «{correction.get('quote') or ''}»")
            if event.get("relation_quote"):
                lines.append(f"  - نص توضيح العلاقة: «{event['relation_quote']}»؛ الوقت: {event.get('relation_message_at') or 'غير متاح'}")
    else:
        lines.append("- لم يُسجل حدث مالي.")
    lines += ["", "### أسباب النتيجة", ""] + [f"- {r}" for r in assessment["reasons"] or ["لا توجد إشارة كافية لتحديد درجة."]] + ["", "## المعلومات الناقصة أو غير المحسومة", ""]
    missing = [LABELS.get(k, "معلومة إضافية") for k in data["missing_fields"]]
    unknown = [LABELS.get(k, "معلومة إضافية") for k in data["unknown_fields"]]
    unresolved = [LABELS.get(k, "معلومة إضافية") for k in data["unresolved_fields"]]
    conflicts = [LABELS.get(k, "معلومة إضافية") for k in data["conflicting_fields"]]
    lines += ["- لم تُسجل بعد: " + (", ".join(missing) if missing else "لا توجد حقول مقابلة أساسية ناقصة."),
              "- غير معروفة/لم يتذكرها المستخدم/فضّل عدم الإجابة: " + (", ".join(unknown) if unknown else "لا توجد حقول بهذه الحالات."),
              "- غير محسومة: " + (", ".join(unresolved) if unresolved else "لا توجد."),
              "- التعارضات الفعلية: " + (", ".join(conflicts) if conflicts else "لا توجد تعارضات مسجلة.")]
    if data["unresolved_money_events"]:
        for event in data["unresolved_money_events"]:
            lines.append(f"- علاقة مبلغ غير محسومة: {kind_labels.get(event.get('kind'), 'مالي')} {event.get('amount') or 'غير محدد'}.")
    lines += ["", "## إرشادات موثقة", ""]
    citations = {}
    for message in state["messages"]:
        for citation in message.get("citations", []):
            citations[citation["id"]] = citation
    if citations:
        for citation in citations.values():
            lines.append(f"- [{citation['title']}]({citation['url']}) — {citation['publisher']}؛ النطاق: {citation['geography']}؛ تاريخ المصدر: {citation['published_or_updated']}؛ إصدار الفهرس: {citation['content_version']}")
    else:
        lines.append("لم تُسترجع إرشادات موثقة في المحادثة. لا تُعرض توصية قانونية أو جهة تواصل دون تحديد الاختصاص ومصدر مناسب.")
    lines += ["", f"> {data['notice']}", ""]
    return "\n".join(lines)


def summary_digest(state: dict[str, Any]) -> str:
    canonical = json.dumps({"facts": state["facts"], "assessment": _assessment(state),
                            "summary_version": state["summary_version"]}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def confirmation_is_current(state: dict[str, Any]) -> bool:
    stored_digest = state.get("confirmed_digest")
    return bool(state.get("confirmed") and isinstance(stored_digest, str) and stored_digest == summary_digest(state))
