from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

POLICY_VERSION = "ACCP-TRIAGE-0.1.0"
SCHEMA_VERSION = 1

QUESTIONS: dict[str, str] = {
    "narrative": "احكيلي شو صار معك.",
    "direct_danger": "هل أنت بخطر مباشر الآن؟",
    "details": "شو قال لك تحديدًا أو شو صار؟",
    "initial_time": "متى بدأت الواقعة أو وصلك أول تواصل؟ اذكر الوقت كما تتذكره.",
    "repeated": "هل تكرر التهديد أو المضايقة المؤثرة؟ ليس مجرد استمرار الرسائل أو التواصل.",
    "requested_amount": "هل طُلب منك مبلغ؟ إذا نعم، كم كان المبلغ المطلوب؟",
    "paid_amount": "هل دفعت أو حولت أي مبلغ؟ إذا نعم، كم؟",
    "payment_method": "بأي وسيلة دفعت؟ لا ترسل كلمة مرور أو رمز تحقق.",
    "deadline": "هل حدد الشخص مهلة أو موعدًا للدفع؟",
    "published": "هل نُشر شيء فعلًا، بحسب علمك؟",
    "identity": "هل بتعرف هوية الشخص، ولا المتوفر عندك بس اسم حساب أو رقم تواصل؟",
    "evidence": "شو الأدلة المتوفرة عندك بحسب إفادتك؟ لا ترفع أو ترسل محتوى حساس هون.",
}

LABELS = {
    "narrative": "وصف الواقعة", "threat": "وجود تهديد", "impersonation": "انتحال شخصية", "harassment": "مضايقة مذكورة",
    "contact_repeated": "تكرار التواصل دون استنتاج تهديد", "publication_threat": "تهديد بالنشر",
    "direct_danger": "الخطر المباشر الحالي", "active_unauthorized_funds": "أموال معرضة حاليًا لتصرف غير مأذون",
    "fraud_description": "وصف خداع مالي", "details": "تفاصيل الواقعة",
    "initial_time": "وقت البداية", "latest_time": "أحدث تواصل", "repeated": "تكرار التهديد أو المضايقة",
    "contact_repeated": "تكرار التواصل دون استنتاج تهديد", "harassment": "مضايقة مذكورة",
    "requested_amount": "المبلغ المطلوب", "paid_amount": "المبلغ المدفوع", "payment_method": "وسيلة الدفع",
    "deadline": "مهلة الدفع", "published": "حدوث النشر", "identity": "هوية الشخص",
    "contact": "معرّف التواصل", "evidence": "الأدلة المذكورة", "additional_details": "تفاصيل إضافية من المستخدم",
    "jurisdiction": "الدولة أو الاختصاص",
}

FIELD_ORDER = ["narrative", "direct_danger", "details", "initial_time", "repeated", "requested_amount",
               "paid_amount", "payment_method", "deadline", "published", "identity", "evidence"]
TERMINAL = {"reported", "denied", "unknown", "not_remembered", "declined", "unresolved", "conflict"}


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
            "created_at": stamp, "updated_at": stamp, "messages": [opening], "facts": {},
            "pending_slot": "narrative", "clarifications": {}, "assessment": None,
            "confirmed": False, "confirmed_digest": None, "summary_version": 1, "last_response": opening["text"]}


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
        elif any(term in clause for term in ("هددني", "يهددني", "بهددني", "هددني", "ابتزني", "يبتزني", "التهديد مش مستمر", "التهديد غير مستمر")) or (already_reported and "التهديد" in clause):
            result = True
    return result


def _evidence_polarity(text: str) -> tuple[bool, bool, list[str]]:
    positive: list[str] = []
    categories = {
        "صور أو لقطات شاشة": ("صور", "صوري", "لقطات", "سكرين"),
        "رسائل أو محادثات": ("رسائل", "رسالة", "محادث", "مسجات"),
        "إيصال أو سجل دفع": ("ايصال", "إيصال", "حوالة", "تحويل"),
    }
    negative_mentions = False
    for clause in _clauses(text):
        for label, terms in categories.items():
            if not any(term in normalize(clause) for term in terms):
                continue
            if re.search(r"(?:ما\s+عندي|ليس\s+لدي|مش\s+عندي|ما\s+في|لا\s+يوجد|ولا\s+عندي)\s+[^،؛.]*", normalize(clause)):
                negative_mentions = True
            elif any(term in normalize(clause) for term in terms):
                positive.append(label)
    return bool(positive), negative_mentions, list(dict.fromkeys(positive))


def _specific_contact_identifier(text: str) -> str | None:
    handle = re.search(r"(?<!\w)@[A-Za-z0-9_.-]{2,}", text)
    if handle:
        return handle.group(0)
    phone = re.search(r"(?<!\w)(?:\+|00)?[0-9٠-٩][0-9٠-٩\s().-]{5,}[0-9٠-٩](?!\w)", text)
    if phone:
        return re.sub(r"\s+", " ", phone.group(0)).strip()
    named = re.search(r"(?:اسم المستخدم|username)\s*[:=]?\s*([\w.-]{2,})", text, re.IGNORECASE)
    return named.group(1) if named else None


def _extract(state: dict[str, Any], text: str, mid: str, pending: str | None) -> set[str]:
    t = normalize(text)
    changed: set[str] = set()
    unknown_status = _is_unknown(text)
    question_id = state["messages"][-2]["id"] if len(state["messages"]) >= 2 and state["messages"][-2]["role"] == "assistant" else None

    # Preserve exact user wording as provenance; short answers are anchored to question_id.
    threat_polarity = _threat_polarity(text, state["facts"].get("threat", {}).get("value") is True)
    if threat_polarity is not None:
        if threat_polarity and (pending == "narrative" or not state["facts"].get("narrative")):
            _set_answer(state, "narrative", text, "reported", text, mid, question_id, changed)
        _set_answer(state, "threat", threat_polarity, "reported" if threat_polarity else "denied", text, mid, question_id, changed)
    impersonation_denied = any(term in t for term in ("ما في حساب منتحل", "مش حساب منتحل", "لا يوجد حساب منتحل", "ما في حد منتحل"))
    impersonation_context = any(term in t for term in ("انتحال", "منتحل", "ينتحل", "ينتحلني", "عامل حساب", "حساب مزيف")) and any(term in t for term in ("باسمي", "اسمي", "صورتي", "ينتحل", "اصحابي", "اصدقائي"))
    if impersonation_denied:
        _set_answer(state, "impersonation", False, "denied", text, mid, question_id, changed)
    elif impersonation_context:
        _set_answer(state, "impersonation", True, "reported", text, mid, question_id, changed)
    platform_names = (("واتساب", "واتساب"), ("whatsapp", "واتساب"), ("انستغرام", "إنستغرام"), ("انستجرام", "إنستغرام"), ("instagram", "إنستغرام"), ("فيسبوك", "فيسبوك"), ("facebook", "فيسبوك"), ("سناب", "سناب شات"), ("snapchat", "سناب شات"), ("تيك توك", "تيك توك"), ("tiktok", "تيك توك"))
    platform = next((name for token, name in platform_names if token in t), None)
    if platform:
        _set_answer(state, "platform", platform, "reported", text, mid, question_id, changed)
    if any(k in t for k in ("محادثات", "صوري", "صور خاصة", "ينشر", "ينشرها", "نشر", "فضح")):
        if any(k in t for k in ("إذا ما", "اذا ما", "رح ينشر", "تهديد بالنشر")) or ("ينشر" in t and any(k in t for k in ("بهدد", "يهدد", "هددني"))):
            _set_answer(state, "details", text, "reported", text, mid, question_id, changed)
            _set_answer(state, "publication_threat", True, "reported", text, mid, question_id, changed)
        elif pending == "published" and ("نشر" in t or "نزل" in t):
            yn = _yes_no(text)
            if yn is not None:
                _set_answer(state, "published", yn, "reported" if yn else "denied", text, mid, question_id, changed)
        if any(k in t for k in ("نشرها فعلا", "نشر فعلا", "نزلها", "انتشرت")):
            _set_answer(state, "published", True, "reported", text, mid, question_id, changed)
    if unknown_status and any(k in t for k in ("نشر", "ينشر", "نزل", "انتشرت")):
        _set_answer(state, "published", None, unknown_status, text, mid, question_id, changed)
    elif any(k in t for k in ("ما نشر", "ما نزل", "لم ينشر", "لم تنشر", "ما اننشرت")):
        _set_answer(state, "published", False, "denied", text, mid, question_id, changed)
    if ("سرق" in t or "دخل على حساب" in t or "اخترق" in t) and not any(k in t for k in ("ما انسرق", "لم يسرق", "ما اخترق", "لم يخترق", "مش مخترق")):
        _set_answer(state, "account_access", True, "reported", text, mid, question_id, changed)
    if any(k in t for k in ("عم يسحب", "تنسحب اموال", "تحويل بدون اذني", "تحويل دون اذني", "خصم بدون اذني", "سحب غير ماذون")):
        _set_answer(state, "active_unauthorized_funds", True, "reported", text, mid, question_id, changed)
    if any(k in t for k in ("نصب علي", "اننصب علي", "خدعني حتى ادفع", "خداع مالي واضح", "احتيال مالي واضح")):
        _set_answer(state, "fraud_description", True, "reported", text, mid, question_id, changed)

    if pending == "narrative" and not state["facts"].get("narrative"):
        _set_answer(state, "narrative", text, "reported", text, mid, question_id, changed)

    # Independently extract facts volunteered before their question.
    has_time_reference = bool(re.search(r"أمس|امس|اليوم|البارحة|مبارح|الاسبوع الماضي|قبل يوم", text))
    if has_time_reference:
        if any(k in t for k in ("اول", "بدا", "وصلني", "بداية", "اول تهديد")) or pending in {"initial_time", "narrative"}:
            _set_answer(state, "initial_time", text, "reported", text, mid, question_id, changed)
        if any(k in t for k in ("اخر تواصل", "اخر مره", "رجع", "كرر", "مرة ثانية", "مره ثانيه", "بعتلي", "ارسللي", "تواصل معي مجددا")):
            _set_answer(state, "latest_time", text, "reported", text, mid, question_id, changed)

    if any(term in t for term in ("مضايقة", "تحرش", "بلاحقني", "ملاحقة", "بضايقني")) and not any(term in t for term in ("ما في مضايقة", "مش مضايقة", "ما بتحرش")):
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

    amount = _first_amount(text)
    requested_amount = _amount_after_marker(text, ("طلب", "طالب", "ادفع", "مطلوب", "اذا ما حولت", "اذا لم تحول"))
    paid_amount = _amount_after_marker(text, ("دفعت", "حولتله", "حولت له", "حوالة", "حولت مبلغ", "تم التحويل", "حولت"))
    conditional_transfer = "اذا ما حولت" in t or "اذا لم تحول" in t
    denied_request_signal = any(k in t for k in ("ما طلب", "لم يطلب", "لم اطلب", "ما طلب مني", "ما طلبت"))
    requested_signal = not denied_request_signal and any(k in t for k in ("طلب", "طالب", "ادفع", "ادفعله", "ادفع له", "مطلوب", "بدك تحول", "اذا ما حولت"))
    denied_payment_signal = any(k in t for k in ("ما دفعت", "لم ادفع", "ما حولت", "لم احول", "ما حولتله", "ما حولت له")) and not conditional_transfer
    paid_signal = not denied_payment_signal and (any(k in t for k in ("دفعت", "حوالة", "حولت مبلغ", "تم التحويل")) or (
        not conditional_transfer and any(k in t for k in ("حولتله", "حولت له"))
    ) or ("حولت " in t and not conditional_transfer))
    if requested_signal:
        _set_answer(state, "requested_amount", requested_amount or amount or text, "reported", text, mid, question_id, changed)
    existing_paid = state["facts"].get("paid_amount")
    if denied_payment_signal:
        _set_answer(state, "paid_amount", None, "denied", text, mid, question_id, changed)
    elif paid_signal:
        if paid_amount:
            _set_answer(state, "paid_amount", paid_amount, "reported", text, mid, question_id, changed)
        elif pending == "paid_amount" or not existing_paid or existing_paid.get("status") == "denied":
            _set_answer(state, "paid_amount", "دفع بحسب الإفادة؛ المبلغ غير محدد", "reported", text, mid, question_id, changed)
        elif existing_paid.get("status") == "reported":
            _set_answer(state, "paid_amount", existing_paid.get("value"), "reported", text, mid, question_id, changed)
    elif pending == "paid_amount":
        if unknown_status:
            _set_answer(state, "paid_amount", None, unknown_status, text, mid, question_id, changed)
        elif denied_payment_signal or (yn := _yes_no(text)) is False:
            _set_answer(state, "paid_amount", None, "denied", text, mid, question_id, changed)
        elif amount and not requested_signal:
            _set_answer(state, "paid_amount", paid_amount or amount, "reported", text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is True:
            _set_answer(state, "paid_amount", "نعم، المبلغ غير محدد", "reported", text, mid, question_id, changed)
    if pending == "requested_amount" and not requested_signal:
        if unknown_status:
            _set_answer(state, "requested_amount", None, unknown_status, text, mid, question_id, changed)
        elif denied_request_signal:
            _set_answer(state, "requested_amount", None, "denied", text, mid, question_id, changed)
        elif (yn := _yes_no(text)) is not None and not yn:
            _set_answer(state, "requested_amount", None, "denied", text, mid, question_id, changed)
        elif (requested_amount or amount) and not paid_signal:
            _set_answer(state, "requested_amount", requested_amount or amount, "reported", text, mid, question_id, changed)

    if any(k in t for k in ("محفظة", "كليك", "كليك", "بطاقة", "بنك", "تحويل بنكي", "western union", "paypal")):
        method = next((x for x in ("محفظة إلكترونية", "كليك", "بطاقة", "تحويل بنكي", "بنك", "PayPal", "Western Union") if normalize(x) in t), text)
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

    if pending == "identity":
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

    evidence_positive, evidence_denied, evidence_types = _evidence_polarity(text)
    evidence_possession = pending == "evidence" or any(term in t for term in ("عندي", "معي", "احتفظت", "حفظت الرسالة", "صورت الشاشة", "اخذت لقطة", "أخذت لقطة", "مسجل عندي", "لدي لقطات"))
    evidence_positive = evidence_positive and evidence_possession
    if pending == "evidence" or evidence_positive or evidence_denied:
        if unknown_status:
            _set_answer(state, "evidence", None, unknown_status, text, mid, question_id, changed)
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
    if pending == "details" and not state["facts"].get("details") and len(text) > 7 and unknown_status is None:
        _set_answer(state, "details", text, "reported", text, mid, question_id, changed)
    if pending == "direct_danger" and not state["facts"].get("direct_danger") and unknown_status:
        _set_answer(state, "direct_danger", None, unknown_status, text, mid, question_id, changed)
    if pending is None and state.get("stage") in {"review", "urgent_stop"} and any(k in t for k in ("تفاصيل", "الوصف", "كلامه", "المعلومة الصحيحة")) and any(k in t for k in ("تصحيح", "قصدي", "اقصد", "عدّل", "بدل")):
        _set_answer(state, "details", text, "reported", text, mid, question_id, changed)
    return changed


def _user_intent(text: str) -> str | None:
    t = normalize(text)
    if any(w in t for w in ("ليش بتسأل", "ليش بدك تعرف", "لماذا تسال", "ليش السؤال")):
        return "why"
    if any(w in t for w in ("احفظ الجلسة", "وقف المقابلة", "توقف المقابلة", "بدي اوقف المقابلة", "خلص بدي اوقف", "بكمل بعدين", "نكمل لاحقا")):
        return "pause"
    if any(w in t for w in ("اعرض المعلومات", "ورجيني الملخص", "اعرض الملخص", "شو سجلت", "المعلومات اللي عندك")):
        return "show"
    return None


def _strip_quoted_spans(text: str) -> str:
    return re.sub(r"«[^»]*»|“[^”]*”|\"[^\"]*\"|'[^']*'", " ", text)


def _clauses(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,،؛.!?؟\n]+|\b(?:بس|لكن|غير أن|إلا أن)\b", normalize(text)) if part.strip()]


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
    return bool(fact and fact.get("status") in TERMINAL)


def _reported_true(state: dict[str, Any], key: str) -> bool:
    fact = state["facts"].get(key, {})
    return fact.get("status") == "reported" and fact.get("value") is True


def _incident_profile(state: dict[str, Any]) -> dict[str, bool]:
    facts = state["facts"]
    context = normalize(" ".join(str(facts.get(key, {}).get("value", "")) for key in ("narrative", "details", "additional_details", "fraud_description")))
    threat = _reported_true(state, "threat")
    impersonation = _reported_true(state, "impersonation")
    harassment = _reported_true(state, "harassment")
    account_access = _reported_true(state, "account_access")
    financial = any(_reported_true(state, key) for key in ("active_unauthorized_funds", "fraud_description"))
    financial = financial or any(facts.get(key, {}).get("status") == "reported" for key in ("requested_amount", "paid_amount"))
    financial = financial or any(word in context for word in ("دينار", "دولار", "ريال", "مبلغ", "فلوس", "مال", "ادفع", "دفعت", "تحويل", "حولت", "بنك", "محفظة", "دفع"))
    publication = _reported_true(state, "publication_threat") or facts.get("published", {}).get("status") in {"reported", "denied"}
    publication = publication or any(word in context for word in ("ينشر", "نشر", "انتشرت", "صور خاصة", "محادثات خاصة"))
    incident = threat or impersonation or harassment or account_access or financial or publication
    return {"threat": threat, "impersonation": impersonation, "harassment": harassment, "account_access": account_access,
            "financial": financial, "publication": publication, "incident": incident}


def _relevant_slots(state: dict[str, Any]) -> list[str]:
    profile = _incident_profile(state)
    slots = ["narrative"]
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
    return list(dict.fromkeys(slots))


def _has_known(state: dict[str, Any], key: str) -> bool:
    fact = state["facts"].get(key)
    return bool(fact and fact.get("status") in {"reported", "denied"})


def _next_slot(state: dict[str, Any]) -> str | None:
    facts = state["facts"]
    for key, fact in facts.items():
        if fact.get("status") == "conflict" and key in QUESTIONS:
            return key
    for key in _relevant_slots(state):
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
    elif (profile["threat"] or profile["impersonation"]) and details.get("status") == "reported" and details.get("value") and any(term in normalize(str(details.get("value"))) for term in ("رسالة", "ينشر", "نشر", "حساب", "اسمي", "صورتي", "ابتزاز", "لاحق", "قتل", "سلاح", "دينار", "محفظة", "رقم")):
        score, route = 85, "مراجعة بشرية"
        priority_rule = "R85_DETAILED_THREAT_OR_IMPERSONATION"
        supporting_fields = (["threat", "details"] if profile["threat"] else ["impersonation", "details"])
        reasons.append("تهديد أو انتحال شخصية مدعوم بتفاصيل بحسب الإفادة")
    elif profile["threat"] or profile["impersonation"]:
        reasons.append("وردت إشارة إلى تهديد أو انتحال، لكن التفاصيل غير كافية لتطبيق درجة 85")
    elif not facts:
        reasons.append("المعلومات غير كافية لتحديد درجة")
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


def handle_turn(state: dict[str, Any], text: str, model_facts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
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
        if key in {"direct_danger", "priority_score", "confirmed", "stage"} or key in changed:
            continue
        if _record(state, key, candidate["status"], candidate["value"], candidate["quote"], user_msg["id"], question_id, "optional_model_extraction"):
            changed.add(key)
    addition_marker = any(marker in normalize(text) for marker in ("اضيف", "تفصيل اضافي", "معلومة اضافية", "نسيت اقول"))
    if pending is None and state.get("stage") in {"review", "urgent_stop"} and intent is None and (not changed or addition_marker):
        if _append_additional_detail(state, text, user_msg["id"], None):
            changed.add("additional_details")
    if pending and pending not in changed and pending in QUESTIONS and intent is None and _next_slot(state) == pending:
        # Keep indirect answers from silently answering the asked slot. Allow one understandable clarification.
        attempts = state["clarifications"].get(pending, 0)
        if attempts < 1:
            state["clarifications"][pending] = attempts + 1
            _add_message(state, "assistant", _short_answer(text) + QUESTIONS[pending], "clarification")
            return state
        if _record(state, pending, "unresolved", None, text, user_msg["id"], state["messages"][-2]["id"] if len(state["messages"]) > 1 else None, "clarification_limit"):
            changed.add(pending)

    if changed:
        state["confirmed"] = False
        state["confirmed_digest"] = None
        state["summary_version"] += 1

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


def display_value(fact: dict[str, Any] | None) -> str:
    if not fact:
        return "لم يُذكر"
    status = fact.get("status")
    if status == "denied":
        return "منفي صراحة بحسب الإفادة"
    if status == "unknown":
        return "المستخدم لا يعرف"
    if status == "not_remembered":
        return "المستخدم لا يتذكر"
    if status == "declined":
        return "المستخدم فضّل عدم الإجابة"
    if status == "unresolved":
        return "لم تُحسم بعد محاولة التوضيح"
    if status == "conflict":
        return "إفادات متعارضة — تحتاج توضيحًا"
    value = fact.get("value")
    if isinstance(value, bool):
        return "نعم بحسب الإفادة" if value else "لا بحسب الإفادة"
    return str(value) if value is not None else "غير محدد بحسب الإفادة"


def report_data(state: dict[str, Any]) -> dict[str, Any]:
    assessment = state.get("assessment") or _assessment(state)
    return {"case_id": state["session_id"], "summary_version": state["summary_version"], "stage": state["stage"],
            "facts": state["facts"], "assessment": assessment, "confirmed": state["confirmed"],
            "generated_at": now_iso(), "notice": "هذا ملخص لإفادة المستخدم للمراجعة فقط. لا يمثل بلاغًا رسميًا أو تحققًا مستقلًا أو إثباتًا للواقعة."}


def render_report(state: dict[str, Any]) -> str:
    data = report_data(state)
    facts = state["facts"]
    lines = [f"# ملخص الحالة {state['session_id']}", "", f"**إصدار الملخص:** {state['summary_version']}  ",
             f"**تاريخ إنشاء التقرير:** {data['generated_at']}  ", f"**التأكيد:** {'مؤكد من المستخدم' if state['confirmed'] else 'غير مؤكد'}", "",
             "## وصف الواقعة", "", display_value(facts.get("narrative")), "", "## التسلسل الزمني", "",
             f"- بداية الواقعة: {display_value(facts.get('initial_time'))}",
             f"- أحدث تواصل: {display_value(facts.get('latest_time'))}", "", "## الوقائع ومصادرها", ""]
    ordered = ["threat", "impersonation", "platform", "details", "additional_details", "publication_threat", "initial_time", "latest_time", "repeated", "requested_amount", "paid_amount", "payment_method", "deadline", "published", "identity", "contact", "evidence", "account_access", "fraud_description", "active_unauthorized_funds", "direct_danger"]
    for key in ordered:
        fact = facts.get(key)
        if not fact:
            continue
        lines += [f"### {LABELS.get(key, key)}", f"- الحالة: {display_value(fact)}"]
        if fact.get("quote"):
            lines.append(f"- نص الإفادة: «{fact['quote']}»")
        if fact.get("message_id"):
            lines.append(f"- مرجع الرسالة: {fact['message_id']}")
        if fact.get("updated_at"):
            lines.append(f"- وقت التسجيل: {fact['updated_at']}")
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
    lines += ["### أسباب النتيجة", ""] + [f"- {r}" for r in assessment["reasons"] or ["لا توجد إشارة كافية لتحديد درجة."]] + ["", "## النواقص والتعارضات", ""]
    missing = [LABELS.get(k, k) for k in _relevant_slots(state) if not _has(state, k)]
    lines += ["- " + (", ".join(missing) if missing else "لا توجد حقول مقابلة أساسية غير محسومة.")]
    conflicts = [LABELS.get(k, k) for k, f in facts.items() if f.get("status") == "conflict"]
    unresolved = [LABELS.get(k, k) for k, f in facts.items() if f.get("status") == "unresolved"]
    lines += ["- الحقول غير المحسومة بعد التوضيح: " + (", ".join(unresolved) if unresolved else "لا توجد."),
              "- التعارضات: " + (", ".join(conflicts) if conflicts else "لا توجد تعارضات مسجلة."), "", "## إرشادات موثقة", ""]
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
    canonical = json.dumps({"facts": state["facts"], "assessment": state.get("assessment") or _assessment(state),
                            "summary_version": state["summary_version"]}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
