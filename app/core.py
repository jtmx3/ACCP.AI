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
    "initial_time": "متى بدأت الواقعة أو وصلك أول تواصل؟ اذكر الوقت كما تتذكره.",
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
    "initial_time": "وقت البداية", "latest_time": "أحدث تواصل", "repeated": "تكرار التهديد أو المضايقة",
    "contact_repeated": "تكرار التواصل دون استنتاج تهديد", "harassment": "مضايقة مذكورة",
    "requested_amount": "المبلغ المطلوب", "paid_amount": "المبلغ المدفوع", "payment_method": "وسيلة الدفع",
    "requested_amount_relation": "علاقة المبلغ المطلوب السابق", "paid_amount_relation": "علاقة مبلغ الدفع السابق",
    "deadline": "مهلة الدفع", "published": "حدوث النشر", "identity": "هوية الشخص",
    "contact": "معرّف التواصل", "contact_identifier_disclosure": "الإفصاح عن معرّف التواصل",
    "evidence": "الأدلة المذكورة", "additional_details": "تفاصيل إضافية من المستخدم",
    "jurisdiction": "الدولة أو الاختصاص",
}

FIELD_ORDER = ["narrative", "direct_danger", "details", "initial_time", "repeated", "requested_amount",
               "paid_amount", "payment_method", "deadline", "published", "identity", "evidence"]
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
    return {"schema_version": SCHEMA_VERSION, "session_id": session_id, "stage": "interview", "revision": 0,
            "created_at": stamp, "updated_at": stamp, "messages": [opening], "facts": {}, "money_events": [],
            "pending_slot": "narrative", "clarifications": {}, "assessment": None,
            "confirmed": False, "confirmed_digest": None, "confirmation_history": [],
            "summary_version": 1, "last_response": opening["text"]}


def _is_unknown(text: str) -> str | None:
    t = normalize(text)
    if any(x in t for x in ("ما بعرف", "مش عارف", "مش عارفه", "لا اعرف", "غير معروف", "مش متاكد", "مش متاكده")):
        return "unknown"
    if any(x in t for x in ("ما بتذكر", "مش متذكر", "مش متذكره", "لا اتذكر", "ناسي", "ناسية")):
        return "not_remembered"
    if any(x in t for x in ("ما بدي اجاوب", "بفضل ما اجاوب", "افضل عدم الاجابة", "مش حاب اجاوب", "بفضل ما احكي")):
        return "declined"
    return None


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
    for match in re.finditer(pattern, text):
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


def _record(state: dict[str, Any], key: str, status: str, value: Any, quote: str,
            message_id: str, question_id: str | None, reason: str = "user_statement") -> bool:
    if status not in TERMINAL:
        raise ValueError(f"Invalid fact status: {status}")
    old = _fact_value(state, key)
    temporal_field = key in {"direct_danger", "repeated", "active_unauthorized_funds", "latest_time"}
    if old and temporal_field and old.get("status") in TERMINAL and old.get("status") != "conflict" and (old.get("status") != status or old.get("value") != value):
        old.setdefault("history", []).append({"status": old["status"], "value": old.get("value"), "quote": old.get("quote"),
                                               "message_id": old.get("message_id"), "at": now_iso(), "reason": "new_temporal_information"})
    elif old and old.get("status") in TERMINAL and old.get("status") != "conflict":
        same = old.get("status") == status and old.get("value") == value
        if same:
            if old.get("message_id") == message_id:
                return False
            supports = old.setdefault("supporting_statements", [])
            if not any(item.get("message_id") == message_id for item in supports):
                supports.append({"quote": quote, "message_id": message_id, "question_id": question_id, "at": now_iso()})
                old["updated_at"] = now_iso()
                return True
            return False
        if key == "details" and old.get("status") == status == "reported" and isinstance(old.get("value"), str) and isinstance(value, str):
            supports = old.setdefault("supporting_statements", [])
            if not any(item.get("message_id") == message_id for item in supports):
                supports.append({"quote": quote, "message_id": message_id, "question_id": question_id, "at": now_iso()})
                old["updated_at"] = now_iso()
                return True
            return False
        is_correction = any(w in normalize(quote) for w in ("تصحيح", "قصدي", "اقصد", "عدّل", "عدلي", "المعلومة الصحيحة", "بدل"))
        if not is_correction:
            alternatives = old.get("alternatives", [{"status": old["status"], "value": old.get("value"), "quote": old.get("quote")}])
            alternatives.append({"status": status, "value": value, "quote": quote, "message_id": message_id})
            history = old.setdefault("history", [])
            history.append({"status": old["status"], "value": old.get("value"), "quote": old.get("quote"), "message_id": old.get("message_id"), "at": now_iso(), "reason": "conflicting_statement"})
            old.update({"status": "conflict", "value": None, "alternatives": alternatives, "quote": quote,
                        "message_id": message_id, "question_id": question_id, "updated_at": now_iso()})
            return True
        old.setdefault("history", []).append({"status": old["status"], "value": old.get("value"), "quote": old.get("quote"),
                                               "message_id": old.get("message_id"), "at": now_iso(), "reason": "explicit_correction"})
    elif old and old.get("status") == "conflict":
        alternatives = old.get("alternatives", [])
        explicit_correction = any(w in normalize(quote) for w in ("تصحيح", "قصدي", "اقصد", "عدّل", "عدلي", "المعلومة الصحيحة", "بدل"))
        matches = any(item.get("status") == status and item.get("value") == value for item in alternatives)
        if status in {"reported", "denied"} and not matches and not explicit_correction:
            alternatives.append({"status": status, "value": value, "quote": quote, "message_id": message_id})
            old["alternatives"] = alternatives
            old["quote"] = quote
            old["message_id"] = message_id
            old["updated_at"] = now_iso()
            return True
        old.setdefault("history", []).append({"status": "conflict", "alternatives": alternatives, "quote": old.get("quote"),
                                               "message_id": old.get("message_id"), "at": now_iso(), "reason": "conflict_resolved"})
    history = old.get("history", []) if old else []
    state["facts"][key] = {"field": key, "status": status, "value": value, "quote": quote,
                            "message_id": message_id, "question_id": question_id, "updated_at": now_iso(), "history": history}
    return True


def _set_answer(state: dict[str, Any], key: str, value: Any, status: str, text: str, mid: str,
                question_id: str | None, changed: set[str]) -> None:
    if _record(state, key, status, value, text, mid, question_id):
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
    possession_markers = ("عندي", "معي", "احتفظت", "حفظت", "لدي", "صورت الشاشة", "اخذت لقطة", "أخذت لقطة")
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
        "بصفته انا", "بصفتي انا",
    )
    claim_pattern = re.compile(
        r"(?:بحكي|بيحكي|يحكي|بقول|بيقول|يقول|بدعي|يدعي|يتظاهر|يزعم)\s+"
        r"[^،؛.]{0,45}\b(?:انه|اني)\s+انا"
    )
    explicit_claim = any(
        (any(marker in normalize(clause) for marker in explicit_claim_markers) or claim_pattern.search(normalize(clause)))
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
    return any(term in normalized for term in (
        "ينتحل شخصيتي", "انتحال شخصيتي", "ينتحل هويتي", "انتحال هويتي", "بصفته انا", "بصفتي انا",
    ))


def _identity_actor_unknown(text: str) -> bool:
    identity_subject = re.compile(r"(?:مين|من)\s+(?:عامل|عمل|انشا|انشأ|مسوي|وراء|صاحب|يدير)|هوية\s+(?:الشخص|الفاعل)")
    return any(identity_subject.search(normalize(clause)) and _is_unknown(clause)
               for clause in _clauses(_strip_quoted_spans(text)))


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


def _extract(state: dict[str, Any], text: str, mid: str, pending: str | None) -> set[str]:
    t = normalize(text)
    changed: set[str] = set()
    unknown_status = _is_unknown(text)
    question_id = state["messages"][-2]["id"] if len(state["messages"]) >= 2 and state["messages"][-2]["role"] == "assistant" else None

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
        elif pending == "published" and not unknown_status and ("نشر" in t or "نزل" in t):
            yn = _yes_no(text)
            if yn is not None:
                _set_answer(state, "published", yn, "reported" if yn else "denied", text, mid, question_id, changed)
        if any(k in t for k in ("نشرها فعلا", "نشر فعلا", "نزلها", "انتشرت")):
            _set_answer(state, "published", True, "reported", text, mid, question_id, changed)
    if unknown_status and any(k in t for k in ("نشر", "ينشر", "نزل", "انتشرت")):
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

    # Independently extract facts volunteered before their question.
    has_time_reference = bool(re.search(r"أمس|امس|اليوم|البارحة|مبارح|الاسبوع الماضي|قبل يوم", text))
    if has_time_reference:
        if any(k in t for k in ("اول", "بدا", "وصلني", "بداية", "اول تهديد")) or pending in {"initial_time", "narrative"}:
            _set_answer(state, "initial_time", text, "reported", text, mid, question_id, changed)
        if any(k in t for k in ("اخر تواصل", "اخر مره", "رجع", "كرر", "مرة ثانية", "مره ثانيه", "بعتلي", "ارسللي", "تواصل معي مجددا")):
            _set_answer(state, "latest_time", text, "reported", text, mid, question_id, changed)

    if any(term in t for term in ("بلاحقني", "ملاحقة", "بضايقني", "تعرضت لمضايقة", "عم بتعرض لمضايقة", "تحرش بي", "بتحرش فيني")) and not any(term in t for term in ("ما في مضايقة", "مش مضايقة", "ما بتحرش")):
        _set_answer(state, "harassment", True, "reported", text, mid, question_id, changed)
    repeat_positive = False
    repeat_negative = False
    for clause in _clauses(_strip_quoted_spans(text)):
        if any(term in clause for term in ("مستمر", "تكرر", "كرر", "مرة ثانية", "مره ثانيه", "رجع بعت", "رجع ارسل")):
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
    elif pending == "repeated" and unknown_status:
        _set_answer(state, "repeated", None, unknown_status, text, mid, question_id, changed)

    if pending in {"requested_amount_relation", "paid_amount_relation"}:
        if not _resolve_money_relation(state, text, mid, question_id, pending, changed) and unknown_status:
            unresolved_relation = _unresolved_money_relation(state)
            if unresolved_relation and unresolved_relation[1] == pending:
                unresolved_relation[0]["relation"] = "unknown"
                unresolved_relation[0]["relation_quote"] = text
                unresolved_relation[0]["relation_message_id"] = mid
                changed.add("money_events")
                _set_answer(state, pending, None, unknown_status, text, mid, question_id, changed)
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
    denied_request_signal = any(_denied_request(clause) for clause in payment_clauses)
    denied_payment_signal = not paid_records and any(_negated_payment(clause) and not _conditional_payment(clause) for clause in payment_clauses)
    state.setdefault("money_events", [])

    existing_requested = state["facts"].get("requested_amount")
    for clause, requested_value in request_records:
        if not requested_value:
            continue
        relation = _money_relation(clause, requested_value, existing_requested)
        if relation in {"initial", "correction"}:
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
            if relation in {"initial", "correction"}:
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

    if pending == "paid_amount" and not paid_records and not denied_payment_signal:
        if unknown_status:
            _set_answer(state, "paid_amount", None, unknown_status, text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is False:
            _set_answer(state, "paid_amount", None, "denied", text, mid, question_id, changed)
        elif amount and not request_records:
            if _add_money_event(state, "paid", amount, "initial", text, mid, question_id):
                changed.add("money_events")
            _set_answer(state, "paid_amount", amount, "reported", text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is True:
            _set_answer(state, "paid_amount", "نعم، المبلغ غير محدد", "reported", text, mid, question_id, changed)

    if pending == "requested_amount" and not request_records:
        if unknown_status:
            _set_answer(state, "requested_amount", None, unknown_status, text, mid, question_id, changed)
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
        if unknown_status:
            _set_answer(state, "deadline", None, unknown_status, text, mid, question_id, changed)
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
        if unknown_status:
            _set_answer(state, "published", None, unknown_status, text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is not None:
            _set_answer(state, "published", yn, "reported" if yn else "denied", text, mid, question_id, changed)
        elif any(k in t for k in ("نشر", "نزل", "بعت الصور", "انتشرت")):
            _set_answer(state, "published", True, "reported", text, mid, question_id, changed)

    if (contact_value := _contact_availability(text)):
        _set_answer(state, "contact", contact_value, "reported", text, mid, question_id, changed)
        if _declined_contact_identifier(text):
            _set_answer(state, "contact_identifier_disclosure", None, "declined", text, mid, question_id, changed)

    identity_unknown = _identity_actor_unknown(text) or (pending == "identity" and unknown_status is not None)
    if identity_unknown:
        _set_answer(state, "identity", None, "unknown", text, mid, question_id, changed)
    elif pending == "identity":
        if unknown_status:
            _set_answer(state, "identity", None, unknown_status, text, mid, question_id, changed)
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
        if evidence_unknown or (pending == "evidence" and unknown_status):
            _set_answer(state, "evidence", None, "unknown", text, mid, question_id, changed)
        elif evidence_positive:
            _set_answer(state, "evidence", "; ".join(evidence_types), "reported", text, mid, question_id, changed)
        elif evidence_denied:
            _set_answer(state, "evidence", None, "denied", text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is not None:
            _set_answer(state, "evidence", "متوفر، النوع غير محدد" if yn else None, "reported" if yn else "denied", text, mid, question_id, changed)

    # Generic terminal answer states, with provenance tied to the asked slot.
    if pending in QUESTIONS and pending not in changed and unknown_status and pending not in ("narrative", "details"):
        _set_answer(state, pending, None, unknown_status, text, mid, question_id, changed)
    if pending == "initial_time" and not state["facts"].get("initial_time") and any(k in t for k in ("أمس", "امس", "اليوم", "البارحة", "مبارح", "الساعة", "صباح", "مسا")):
        _set_answer(state, "initial_time", text, "reported", text, mid, question_id, changed)
    if pending == "details" and not state["facts"].get("details") and _incident_detail_candidate(text) and unknown_status is None:
        _set_answer(state, "details", text, "reported", text, mid, question_id, changed)
    if pending == "direct_danger" and not state["facts"].get("direct_danger") and unknown_status:
        _set_answer(state, "direct_danger", None, unknown_status, text, mid, question_id, changed)
    if pending is None and state.get("stage") in {"review", "urgent_stop"} and any(k in t for k in ("تفاصيل", "الوصف", "كلامه", "المعلومة الصحيحة")) and any(k in t for k in ("تصحيح", "قصدي", "اقصد", "عدّل", "بدل")):
        _set_answer(state, "details", text, "reported", text, mid, question_id, changed)
    return changed


def _user_intent(text: str) -> str | None:
    t = normalize(text)
    if any(normalize(w) in t for w in ("ليش بتسأل", "ليش بدك تعرف", "لماذا تسأل", "ليش السؤال")):
        return "why"
    if any(normalize(w) in t for w in ("احفظ الجلسة", "وقف المقابلة", "توقف المقابلة", "بدي أوقف المقابلة", "خلص بدي أوقف", "بكمل بعدين", "نكمل لاحقًا")):
        return "pause"
    if any(normalize(w) in t for w in ("اعرض المعلومات", "ورجيني الملخص", "اعرض الملخص", "شو سجلت", "المعلومات اللي عندك")):
        return "show"
    return None


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
        return any(re.match(rf"^{re.escape(prefix)}(?:$|\\s)", clause) for prefix in prefixes)

    clauses = [part.strip() for part in re.split(r"[،,؛;\\n]+", normalized) if part.strip()]
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
    return any(marker in clause for marker in ("ما طلب", "لم يطلب", "لم اطلب", "ما طلب مني", "ما طلبت"))


def _is_request_clause(clause: str) -> bool:
    if _denied_request(clause):
        return False
    if _conditional_payment(clause):
        return True
    if any(marker in clause for marker in ("طلب مني", "طلبت مني", "يطلب مني", "تطلب مني", "طالبني", "مطلوب", "بدك تحول", "ادفع له", "ادفعله")) or re.search(r"(?:^|\s|و)(?:طلب|طلبت|طالب)(?:ني)?(?:\s|$)", clause):
        return True
    if "ادفع" not in clause or _negated_payment(clause):
        return False
    if any(marker in clause for marker in ("خدعني حتى ادفع", "خدعني حتي ادفع", "حتى ادفع", "حتي ادفع")):
        return False
    return bool(_first_amount(clause) or any(marker in clause for marker in ("ادفع له", "ادفعله", "ادفع لها", "ادفعلها")))


def _is_completed_payment_clause(clause: str) -> bool:
    if _conditional_payment(clause) or _negated_payment(clause):
        return False
    return any(marker in clause for marker in ("دفعت", "حولتله", "حولت له", "حوالة", "حولت مبلغ", "تم التحويل")) or bool(re.search(r"\bحولت\s", clause))


def _amount_digits(value: Any) -> str:
    if value is None:
        return ""
    normalized = str(value).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return re.sub(r"\D", "", normalized)


def _same_amount(left: Any, right: Any) -> bool:
    left_digits, right_digits = _amount_digits(left), _amount_digits(right)
    return bool(left_digits and right_digits and left_digits == right_digits)


def _money_relation(text: str, amount: str, existing: dict[str, Any] | None) -> str:
    clause = normalize(text)
    if any(marker in clause for marker in ("تصحيح", "المبلغ الصحيح", "بدل", "قصدي", "اقصد", "تعديل للمبلغ")):
        return "correction"
    if any(marker in clause for marker in ("زيادة", "اضافي", "اضافية", "كمان", "مبلغ ثاني", "مبلغ اخر", "دفعة ثانية", "دفعة اخرى")):
        return "additional"
    if not existing or existing.get("status") != "reported" or not existing.get("value"):
        return "initial"
    return "same" if _same_amount(amount, existing.get("value")) else "unresolved"


def _add_money_event(state: dict[str, Any], kind: str, amount: str, relation: str, quote: str,
                     message_id: str, question_id: str | None) -> bool:
    events = state.setdefault("money_events", [])
    if any(event.get("message_id") == message_id and event.get("kind") == kind and
           event.get("amount") == amount and event.get("relation") == relation for event in events):
        return False
    events.append({"id": f"money-{len(events) + 1}", "kind": kind, "amount": amount, "relation": relation,
                   "quote": quote, "message_id": message_id, "question_id": question_id, "created_at": now_iso()})
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
    if any(marker in normalized for marker in ("زيادة", "اضافي", "اضافية", "كمان", "مبلغ ثاني", "مبلغ اخر", "دفعة ثانية", "دفعة اخرى")):
        relation = "additional"
    elif any(marker in normalized for marker in ("تصحيح", "تعديل", "بدل", "المبلغ الصحيح", "قصدي", "اقصد")):
        relation = "correction"
    else:
        return False
    event["relation"] = relation
    event["relation_quote"] = text
    event["relation_message_id"] = mid
    event["relation_updated_at"] = now_iso()
    fact_key = "requested_amount" if event["kind"] == "requested" else "paid_amount"
    relation_key = pending
    if relation == "correction":
        old = state["facts"].get(fact_key)
        if old:
            old.setdefault("history", []).append({"status": old.get("status"), "value": old.get("value"),
                "quote": old.get("quote"), "message_id": old.get("message_id"), "at": now_iso(), "reason": "explicit_amount_relation_correction"})
            old.update({"status": "reported", "value": event["amount"], "quote": event["quote"],
                        "message_id": event["message_id"], "updated_at": now_iso()})
            old.setdefault("supporting_statements", []).append({"quote": text, "message_id": mid, "question_id": question_id, "at": now_iso()})
    elif state["facts"].get(fact_key):
        fact = state["facts"][fact_key]
        fact.setdefault("supporting_statements", []).append({"quote": text, "message_id": mid, "question_id": question_id, "at": now_iso()})
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
        unknown = _is_unknown(text)
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
    if profile["threat"] or profile["impersonation"] or profile["account_access"] or profile["financial"]:
        slots.extend(("details", "initial_time"))
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
        "initial_time": "حتى نفرّق بين وقت بداية الواقعة وأحدث تواصل.",
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


def _add_message(state: dict[str, Any], role: str, text: str, kind: str = "chat") -> dict[str, Any]:
    msg = {"id": f"m{len(state['messages']) + 1}", "role": role, "text": text, "created_at": now_iso(), "kind": kind}
    state["messages"].append(msg)
    state["last_response"] = text
    return msg


def _finalize_changed_facts(state: dict[str, Any], changed: set[str]) -> None:
    if changed:
        state["confirmed"] = False
        state["confirmed_digest"] = None
        state["summary_version"] += 1
    state["assessment"] = _assessment(state)


def handle_turn(state: dict[str, Any], text: str, model_facts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
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
    if intent == "pause":
        state["stage"] = "paused"
        state["pending_slot"] = pending
        _add_message(state, "assistant", "حفظت الجلسة. تقدر ترجع لها وتكمل من نفس المكان.", "status")
        return state
    if intent == "why":
        reason = _explain_why(pending or "")
        prompt = QUESTIONS.get(pending or "", "احكيلي أي تفصيل بتحب تضيفه.")
        _add_message(state, "assistant", f"{reason}\n\n{prompt}", "question")
        return state
    if intent == "show":
        state["stage"] = "review"
        state["pending_slot"] = None
        state["assessment"] = _assessment(state)
        _add_message(state, "assistant", "هذا ملخص أولي لما سجلته. راجعه وصحح أي معلومة قبل التأكيد.", "review")
        return state

    changed = _extract(state, text, user_msg["id"], pending)
    question_id = state["messages"][-2]["id"] if len(state["messages"]) >= 2 and state["messages"][-2]["role"] == "assistant" else None
    for candidate in model_facts or []:
        key = candidate.get("field")
        if key in {"direct_danger", "priority_score", "confirmed", "stage"}:
            _trace_model_merge(key, "ignored", "engine_owned_safety_or_decision_field")
            continue
        # Amount facts must pass through the money-event relation parser below; otherwise
        # a model proposal can overwrite the initial request before clarification.
        if key in {"requested_amount", "paid_amount"}:
            rule_fact = state["facts"].get(key, {})
            agrees = rule_fact.get("status") == candidate.get("status") and rule_fact.get("value") == candidate.get("value")
            _trace_model_merge(key, "routed_to_rules", "relation_aware_money_event_parser",
                               rule_extraction_agrees=agrees,
                               rule_extraction_outcome="agrees" if agrees else "differs_or_no_fact_recorded")
            continue
        if key in changed:
            rule_fact = state["facts"].get(key, {})
            agrees = rule_fact.get("status") == candidate.get("status") and rule_fact.get("value") == candidate.get("value")
            _trace_model_merge(key, "agrees_with_rules" if agrees else "superseded_by_rules",
                               "deterministic_extraction_already_changed_field",
                               rule_extraction_agrees=agrees,
                               rule_extraction_outcome="agrees" if agrees else "differs_or_no_fact_recorded")
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
        if _record(state, key, candidate["status"], candidate["value"], candidate["quote"], user_msg["id"], question_id, "optional_model_extraction"):
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
    if pending and pending not in changed and pending in QUESTIONS and intent is None and _next_slot(state) == pending:
        # Keep indirect answers from silently answering the asked slot. Allow one understandable clarification.
        attempts = state["clarifications"].get(pending, 0)
        if attempts < 1:
            state["clarifications"][pending] = attempts + 1
            _finalize_changed_facts(state, changed)
            _add_message(state, "assistant", _short_answer(text) + QUESTIONS[pending], "clarification")
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

    next_slot = _next_slot(state)
    if next_slot:
        if previous_stage in {"review", "urgent_stop"} and pending is None:
            state["stage"] = "urgent_stop" if previous_stage == "urgent_stop" else "review"
            state["pending_slot"] = None
            state["assessment"] = _assessment(state)
            _add_message(state, "assistant", "سجلت التحديث. راجع الملخص الحالي وأي معلومات ما زالت غير معروفة قبل التأكيد.", "review")
            return state
        state["stage"] = "interview"
        state["pending_slot"] = next_slot
        _add_message(state, "assistant", QUESTIONS[next_slot], "question")
        return state
    state["pending_slot"] = None
    state["stage"] = "review"
    state["assessment"] = _assessment(state)
    _add_message(state, "assistant", "اكتملت الأسئلة الأساسية بحسب المعلومات المتاحة. راجع الملخص، وصحح أو أضف تفاصيل قبل التأكيد.", "review")
    return state


def resume_state(state: dict[str, Any]) -> dict[str, Any]:
    if state["stage"] == "paused":
        state["stage"] = "interview"
        slot = state.get("pending_slot") or _next_slot(state)
        state["pending_slot"] = slot
        text = QUESTIONS.get(slot, "تقدر تراجع الملخص أو تضيف معلومة جديدة.")
        _add_message(state, "assistant", "رجعنا للجلسة المحفوظة.\n\n" + text, "question")
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
    details = facts.get("details", {})
    if requested.get("status") != "reported" or details.get("status") != "reported":
        return []
    current_amount = requested.get("value")
    current_message_id = requested.get("message_id")
    if not current_amount or not current_message_id:
        return []
    correction_event = next((event for event in reversed(state.get("money_events", []))
                             if event.get("kind") == "requested" and event.get("relation") == "correction"
                             and event.get("message_id") == current_message_id
                             and _same_amount(event.get("amount"), current_amount)), None)
    if not correction_event:
        return []
    previous = next((item for item in reversed(requested.get("history", []))
                     if item.get("reason") in {"explicit_correction", "explicit_amount_relation_correction"}
                     and item.get("value") is not None), None)
    if not previous or _same_amount(previous.get("value"), current_amount):
        return []
    previous_digits = _amount_digits(previous.get("value"))
    detail_quotes = [details.get("quote", ""), details.get("value", "")]
    if not previous_digits or not any(
        _amount_digits(amount) == previous_digits
        for quote in detail_quotes if isinstance(quote, str)
        for _, _, amount in _amount_matches(quote)
    ):
        return []
    return [{
        "target_field": "details",
        "corrected_field": "requested_amount",
        "current_value": current_amount,
        "previous_value": previous.get("value"),
        "previous_quote": previous.get("quote"),
        "previous_message_id": previous.get("message_id"),
        "previous_message_at": message_times.get(previous.get("message_id")),
        "correction_quote": requested.get("quote"),
        "correction_message_id": current_message_id,
        "correction_message_at": message_times.get(current_message_id),
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
    money_events = []
    for event in state.get("money_events", []):
        money_events.append({**event,
            "source_message_at": message_times.get(event.get("message_id")),
            "relation_message_at": message_times.get(event.get("relation_message_id"))})
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
            "facts": facts, "money_events": money_events, "assessment": assessment,
            "correction_notes": _report_correction_notes(state, message_times),
            "field_labels": {key: LABELS.get(key, "معلومة إضافية") for key in field_keys},
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
             f"- بداية الواقعة: {display_value(facts.get('initial_time'))}",
             f"- أحدث تواصل: {display_value(facts.get('latest_time'))}", "", "## الوقائع ومصادرها", ""]
    ordered = ["threat", "impersonation", "platform", "details", "unclassified_details", "additional_details", "publication_threat", "initial_time", "latest_time", "repeated", "requested_amount", "paid_amount", "payment_method", "deadline", "published", "identity", "contact", "evidence", "original_account_access_available", "account_access", "fraud_description", "active_unauthorized_funds", "direct_danger"]
    for key in ordered:
        fact = facts.get(key)
        if not fact:
            continue
        lines += [f"### {LABELS.get(key, key)}", f"- الحالة: {display_status(fact)}",
                  f"- القيمة: {display_value(fact)}"]
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
            lines.append(f"- مبلغ {kind_labels.get(event.get('kind'), 'مالي')}: {event.get('amount') or 'غير محدد'}؛ العلاقة: {relation_labels.get(event.get('relation'), 'غير معروفة')}؛ وقت الرسالة الأصلية: {event.get('source_message_at') or 'غير متاح'}")
            if event.get("quote"):
                lines.append(f"  - نص الإفادة: «{event['quote']}»")
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
