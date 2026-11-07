from __future__ import annotations

import json
import re
from typing import Any

MAX_CONTEXT_CHARS = 4_500
MAX_RECENT_MESSAGES = 4
MAX_MESSAGE_CHARS = 500
MAX_FACTS = 10
MAX_MONEY_EVENTS = 4

QUESTION_TEXT = {
    "narrative": "احكيلي شو صار معك.",
    "direct_danger": "هل أنت بخطر مباشر الآن؟",
    "details": "شو قال لك تحديدًا أو شو صار؟",
    "initial_time": "متى بدأ التصرف أو وصلك أول تواصل؟",
    "discovery_time": "متى اكتشفت الواقعة؟",
    "account_creation_time": "متى أُنشئ الحساب، بحسب ما تعرف؟",
    "requested_amount": "كم كان المبلغ المطلوب؟",
    "requested_amount_relation": "هل المبلغ إضافة أم تصحيح؟",
    "paid_amount": "هل دفعت؟ وكم؟",
    "paid_amount_relation": "هل هذا مبلغ دفع جديد أم نفس الدفعة؟",
    "payment_method": "بأي وسيلة دفعت؟",
    "deadline": "هل حُددت مهلة للدفع؟",
    "published": "هل حدث النشر فعلًا؟",
    "identity": "هل تعرف هوية الشخص أم لديك قناة تواصل فقط؟",
    "evidence": "ما الأدلة المتوفرة بحسب إفادتك؟",
    "repeated": "هل تكرر التهديد أو المضايقة؟",
}

FINANCIAL_FIELDS = {"requested_amount", "paid_amount", "payment_method", "deadline",
                    "requested_amount_relation", "paid_amount_relation"}
FIELD_GROUPS = {
    "narrative": {"narrative", "details", "threat", "impersonation", "platform"},
    "unclassified_details": {"narrative", "details", "threat", "impersonation", "platform"},
    "direct_danger": {"threat", "direct_danger"},
    "details": {"narrative", "details", "threat", "impersonation", "publication_threat", "published"},
    "initial_time": {"initial_time", "discovery_time", "account_creation_time", "latest_time"},
    "discovery_time": {"discovery_time", "initial_time", "account_creation_time"},
    "account_creation_time": {"discovery_time", "account_creation_time", "initial_time"},
    "repeated": {"threat", "harassment", "repeated", "contact_repeated"},
    "requested_amount": FINANCIAL_FIELDS,
    "requested_amount_relation": FINANCIAL_FIELDS,
    "paid_amount": FINANCIAL_FIELDS,
    "paid_amount_relation": FINANCIAL_FIELDS,
    "payment_method": FINANCIAL_FIELDS,
    "deadline": FINANCIAL_FIELDS,
    "published": {"publication_threat", "published", "details"},
    "identity": {"identity", "contact", "contact_identifier_disclosure", "impersonation"},
    "evidence": {"evidence", "identity", "details"},
}

QUESTION_PREFIXES = (
    "كيف", "هل", "متى", "وين", "اين", "شو", "ايش", "اش", "ليش", "لماذا", "ماذا", "مين", "من", "كم",
    "ما معنى", "ما الذي", "ما هي", "ما هو", "ما الخطوات",
)


def normalize_goal_text(text: str) -> str:
    normalized = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text.lower())
    return (normalized.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            .replace("ى", "ي").replace("ة", "ه"))


def _question_prefix_at(text: str, start: int) -> bool:
    tail = text[start:]
    return any(tail == prefix or tail.startswith(prefix + " ") for prefix in QUESTION_PREFIXES)


def extract_guidance_question(text: str) -> str | None:
    """Return a literal current-message question span, prioritizing a final question clause."""
    if not text.strip():
        return None
    # Prefer an explicit question-mark suffix. Preserve the exact original substring.
    question_end = max(text.rfind("؟"), text.rfind("?"))
    limit = question_end + 1 if question_end >= 0 else len(text)
    starts = [match.start() for match in re.finditer(r"(?<!\S)(?:كيف|هل|متى|وين|أين|اين|شو|ايش|اش|ليش|لماذا|ماذا|مين|من|كم|ما معنى|ما الذي|ما هي|ما هو|ما الخطوات)(?=\s)", text[:limit])]
    for start in reversed(starts):
        if not _question_prefix_at(normalize_goal_text(text), len(normalize_goal_text(text[:start]))):
            continue
        raw = text[start:limit].strip(" \t\r\n,،;؛.")
        if raw and (question_end >= 0 or start == 0 or text[max(0, start - 2):start].strip() in {",", "،", ";", "؛", ".", "!"}):
            return raw
    # A question may appear in the first clause and have no terminal punctuation.
    first_clause = re.split(r"[،,؛;.!?؟\n]+", text, maxsplit=1)[0].strip()
    if first_clause and _question_prefix_at(normalize_goal_text(first_clause), 0):
        return first_clause
    return None


def extract_fact_text(text: str, guidance_question: str | None = None) -> str:
    """Remove only the exact question span; preserve all remaining user wording verbatim."""
    if not guidance_question:
        return text.strip()
    start = text.rfind(guidance_question)
    if start < 0:
        return text.strip()
    fact_text = (text[:start] + " " + text[start + len(guidance_question):]).strip()
    return re.sub(r"^[\s,،;؛.!?؟]+|[\s,،;؛.!?؟]+$", "", fact_text).strip()


def _has_explicit_fact_clause(text: str) -> bool:
    normalized = normalize_goal_text(text)
    markers = (
        "عندي ", "معي ", "ما بعرف", "ما اعرف", "مش عارف", "لا اعرف", "ما بتذكر", "مش متذكر",
        "ما بدي اكتب", "ما بدي اذكر", "دفعت", "حولت", "طلب مني", "طلب كمان", "اكتشفت", "اكتشف",
        "انتبهت", "لاحظت", "بدأت", "بدا", "في شخص", "شخص هدد", "تهديد", "انتحل", "تنتحل",
        "حساب مزيف", "بقدر ادخل", "لقيت صور", "لقطات شاشة", "ما في خطر", "لا يوجد تهديد",
        "هددني", "هدد", "رجع هدد", "اليوم رجع", "بعتلي",
        "ما بعرف اذا نشر", "ما بعرف اذا نشرها", "ما بعرف إذا نشر",
    )
    if any(normalize_goal_text(marker) in normalized for marker in markers):
        return True
    return bool(re.search(r"(?:^|\s)\d+\s*(?:دينار|ريال|دولار|درهم)(?:\s|$)", normalized))


def interpret_message(text: str, pending_slot: str | None) -> dict[str, Any]:
    """Return non-exclusive role hints and exact spans; never changes session state."""
    value = text.strip()
    normalized = normalize_goal_text(value)
    question = extract_guidance_question(value)
    fact_text = extract_fact_text(value, question)
    goals: list[str] = []
    if pending_slot and fact_text:
        goals.append("answer_pending_question")
    if question:
        goals.append("ask_question")
        q = normalize_goal_text(question)
        if any(marker in q for marker in ("ليش بتسال", "ليش عم تسال", "ليش بدك تعرف", "لماذا تسال", "سبب السؤال")):
            goals.append("ask_why")
        if any(marker in q for marker in ("الاولويه", "الدرجة", "الدرجه", "التصنيف", "النتيجه")) and any(marker in q for marker in ("ليش", "لماذا", "كيف")):
            goals.append("ask_assessment_reason")
        if any(marker in q for marker in (
            "شو اعمل", "ماذا افعل", "كيف اوثق", "كيف احمي", "كيف احفظ", "كيف ابلغ", "كيف اتصرف",
            "شو بقدر اعمل", "ما الخطوات", "كيف امن", "كيف اامن",
        )):
            goals.append("ask_guidance")
    if any(marker in normalized for marker in ("تصحيح", "قصدي", "اقصد", "المبلغ الصحيح", "المعلومة الصحيحة", "بدل", "مش هيك")):
        goals.append("correct_prior_fact")
    if any(marker in normalized for marker in ("اضيف", "تفصيل اضافي", "معلومة اضافية", "نسيت اقول", "كمان شغلة")):
        goals.append("add_fact")
    elif _has_explicit_fact_clause(fact_text):
        goals.append("add_fact")
    if any(marker in normalized for marker in ("نفس المبلغ", "نفس الشخص", "مثل ما قلت", "زي ما حكيت", "مش دفعة جديدة", "ما تغير", "الشخص نفسه")):
        goals.append("confirm_or_reference_prior_fact")
    if any(marker in normalized for marker in ("ما بعرف", "مش عارف", "لا اعرف", "مش معروف", "مجهول", "ما عندي فكرة")):
        goals.append("unknown")
    if any(marker in normalized for marker in ("ما بتذكر", "مش متذكر", "لا اتذكر", "ناسي")):
        goals.append("not_remembered")
    if any(marker in normalized for marker in ("ما بدي اكتب", "ما بدي اذكر", "ما بدي احكي", "افضل عدم", "بفضل ما")):
        goals.append("decline_disclosure")
    if any(marker in normalized for marker in ("احفظ الجلسة", "وقف المقابلة", "توقف المقابلة", "بكمل بعدين")):
        goals.append("pause")
    if any(marker in normalized for marker in ("اعرض المعلومات", "ورجيني الملخص", "اعرض الملخص", "شو سجلت", "المعلومات اللي عندك")):
        goals.append("review_summary")
    return {
        "goals": list(dict.fromkeys(goals)),
        "pending_slot": pending_slot,
        "guidance_question": question,
        "fact_text": fact_text,
        "may_contain_multiple_goals": len(goals) > 1,
        "interpretation_only": True,
    }


def has_actionable_fact_goal(text: str, pending_slot: str | None = None) -> bool:
    goals = set(interpret_message(text, pending_slot)["goals"])
    return bool(goals & {"answer_pending_question", "correct_prior_fact", "add_fact", "unknown", "not_remembered",
                         "decline_disclosure", "confirm_or_reference_prior_fact"})


def _related_fields(text: str, pending_slot: str | None, state: dict[str, Any]) -> set[str]:
    fields = set(FIELD_GROUPS.get(pending_slot or "", set()))
    folded = normalize_goal_text(text)
    financial_markers = ("دفع", "دفعت", "حول", "تحويل", "مبلغ", "دينار", "مهلة", "دفعة", "طلب كمان")
    if any(marker in folded for marker in financial_markers):
        fields.update(FINANCIAL_FIELDS)
        fields.update(field for field in state.get("facts", {}) if field in FINANCIAL_FIELDS)
    if any(marker in folded for marker in ("اكتشف", "انشاء", "انعمل", "انشأ", "اليوم", "متى", "انتبهت", "لاحظت")):
        fields.update({"initial_time", "discovery_time", "account_creation_time", "latest_time"})
    if any(marker in folded for marker in ("هوية", "مين", "رقم", "واتساب", "معرف", "تواصل", "مجهول")):
        fields.update({"identity", "contact", "contact_identifier_disclosure"})
    if any(marker in folded for marker in ("دليل", "اثبات", "لقطات", "صور", "رسائل", "محادثات", "نسخة")):
        fields.update({"evidence", "details"})
    if not fields:
        fields.update({"narrative", "details"})
    return fields


def build_bounded_context(state: dict[str, Any], current_text: str, pending_slot: str | None) -> dict[str, Any]:
    """Build source-referenced, bounded context; never include tokens, session IDs, or assessment scores."""
    fields = _related_fields(current_text, pending_slot, state)
    facts_out: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for field in sorted(fields):
        fact = state.get("facts", {}).get(field)
        if not isinstance(fact, dict):
            continue
        item = {
            "fact_ref": f"fact:{field}:{fact.get('message_id') or 'unknown'}",
            "field": field, "status": fact.get("status"), "value": fact.get("value"),
            "quote": str(fact.get("quote") or "")[:MAX_MESSAGE_CHARS],
            "message_ref": fact.get("message_id"),
        }
        facts_out.append(item)
        if fact.get("message_id"):
            source_ids.add(str(fact["message_id"]))
    facts_out = facts_out[-MAX_FACTS:]

    pending_message = next((m for m in reversed(state.get("messages", []))
                            if m.get("role") == "assistant" and m.get("kind") in {"question", "clarification"}), None)
    recent: list[dict[str, Any]] = []
    user_messages = [m for m in state.get("messages", [])[:-1] if m.get("role") == "user"]
    selected_ids = source_ids | {m.get("id") for m in user_messages[-2:]}
    candidates = [m for m in user_messages if m.get("id") in selected_ids]
    for message in candidates[-MAX_RECENT_MESSAGES:]:
        recent.append({"message_ref": message.get("id"), "role": "user",
                       "text": str(message.get("text") or "")[:MAX_MESSAGE_CHARS]})
    if pending_message:
        recent.append({"message_ref": pending_message.get("id"), "role": "assistant_question",
                       "text": str(pending_message.get("text") or "")[:MAX_MESSAGE_CHARS]})
    recent = recent[-MAX_RECENT_MESSAGES:]

    events: list[dict[str, Any]] = []
    if fields & FINANCIAL_FIELDS:
        for event in state.get("money_events", [])[-MAX_MONEY_EVENTS:]:
            events.append({key: event.get(key) for key in
                           ("id", "kind", "amount", "relation", "quote", "message_id", "relation_message_id")
                           if key in event})
    context: dict[str, Any] = {
        "context_version": "accp-limited-context-v2",
        "pending_question": {"slot": pending_slot,
                             "text": QUESTION_TEXT.get(pending_slot or ""),
                             "question_ref": pending_message.get("id") if pending_message else None},
        "recent_relevant_messages": recent,
        "related_facts": facts_out,
        "related_money_events": events,
        "context_char_limit": MAX_CONTEXT_CHARS,
        "untrusted_user_data": True,
    }

    def encoded_size() -> int:
        return len(json.dumps(context, ensure_ascii=False, separators=(",", ":")))

    # Keep the question reference and current turn semantics; reduce only older quotes/messages first.
    if encoded_size() > MAX_CONTEXT_CHARS:
        for fact in facts_out:
            fact["quote"] = fact.get("quote", "")[:160]
        for message in recent[:-1]:
            message["text"] = message.get("text", "")[:200]
    if encoded_size() > MAX_CONTEXT_CHARS:
        context["related_facts"] = facts_out[:5]
        context["recent_relevant_messages"] = recent[-2:]
        context["related_money_events"] = events[-2:]
    if encoded_size() > MAX_CONTEXT_CHARS:
        context["related_facts"] = []
        context["recent_relevant_messages"] = recent[-1:]
        context["related_money_events"] = []
    if encoded_size() > MAX_CONTEXT_CHARS:
        context["recent_relevant_messages"] = []
    if encoded_size() > MAX_CONTEXT_CHARS:
        # These small references are bounded in normal operation; fail closed rather than exceed the contract.
        context["pending_question"] = {"slot": pending_slot, "question_ref": None}
    context["serialized_char_count"] = encoded_size()
    if context["serialized_char_count"] > MAX_CONTEXT_CHARS:
        raise ValueError("Bounded extraction context exceeds its character limit")
    return context
