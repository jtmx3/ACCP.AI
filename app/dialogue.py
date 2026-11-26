"""Natural Arabic dialogue phrasing, bounded context, and safe local reply validation."""
from __future__ import annotations

import json
import re
from typing import Any

from app.nlu import normalize_arabic
from app.triage import classify_track

MAX_DIALOGUE_CONTEXT_CHARS = 5_500


def first_turn_empathy_preface(state: dict[str, Any], slot: str) -> str:
    """Brief first-follow-up care for verbal abuse and image-publication concerns."""
    user_turns = sum(1 for item in state.get("messages", [])
                     if isinstance(item, dict) and item.get("role") == "user")
    if user_turns != 1 or slot == "narrative":
        return ""
    track = classify_track(state)
    if track == "defamation_or_abuse":
        return "آسف إنك تعرضت لهالكلام؛ خذ راحتك، وما في داعي تكرر ألفاظ ما بترتاح لها. "
    facts = state.get("facts", {})
    photo_text = " ".join(str((facts.get(key) or {}).get("value") or "")
                          for key in ("narrative", "details", "unclassified_details"))
    normalized_photo_text = normalize_arabic(photo_text)
    photo_concern = any(token in normalized_photo_text for token in ("صور", "صوره", "صورتي", "صوري")) \
        and any(token in normalized_photo_text for token in ("خايف", "خايفه", "ينشر", "ينزل", "يبعت", "يرسل"))
    image_extortion = track == "extortion_fraud_takeover" and (
        ((facts.get("sensitive_photos") or {}).get("status") == "reported"
         and (facts.get("sensitive_photos") or {}).get("value") is True)
        or any(token in normalized_photo_text for token in ("صور", "صوره", "صورتي", "صوري")))
    if photo_concern or image_extortion:
        return "آسف إنك عم تمر بهالضغط؛ ما في داعي ترفع أو ترسل الصور هون. "
    return ""


def compose_fallback_question(state: dict[str, Any], slot: str, questions: dict[str, str]) -> str:
    """Select a concise, context-aware alternative for rules-only operation."""
    track = classify_track(state)
    preface = first_turn_empathy_preface(state, slot)
    alternatives: dict[str, tuple[str, ...]] = {
        "details": (
            "إذا بناسبك، شو التصرف أو الكلام اللي صار بالتحديد؟",
            "ممكن توصفلي شو قال أو عمل، بالقدر اللي بترتاح تحكيه؟",
            "شو أهم تفصيل بتحب ينحفظ عن اللي صار؟",
        ),
        "platform": ("على أي تطبيق أو موقع صار هالشي؟", "وين ظهر الكلام أو التصرف، إذا بتعرف؟"),
        "identity": (
            "بتعرف مين الشخص، ولا اللي متوفر بس حساب أو وسيلة تواصل؟ ما تكتب المعرّف نفسه.",
            "هل تعرف هوية الشخص أو عندك اسم حساب فقط؟ ما في داعي ترسل الاسم أو الرابط.",
        ),
        "evidence": (
            "شو نوع الشي المحفوظ عندك، مثل لقطة شاشة أو محادثة؟ لا ترسله هون.",
            "هل عندك نسخة أو لقطة محفوظة، أم ما توفر شيء؟ لا ترفع أي ملف.",
        ),
        "initial_time": (
            "متى بدأ التصرف أو وصلك أول تواصل تقريبًا؟ وقت الاكتشاف قد يختلف عن بدايته.",
            "هل تتذكر متى بدأ التصرف أو وصلك أول تواصل، ولو بتقدير تقريبي؟",
        ),
        "repeated": (
            "هل تكرر التهديد أو الضرر نفسه، ولا كان التواصل مستمرًا فقط؟",
            "صار نفس التصرف مرة ثانية، أم الرسائل استمرت من غير تكرار الضرر؟",
        ),
        "published": (
            "بحسب اللي تعرفه، هل نُشر المحتوى فعلًا أم ما عندك تأكيد؟ لا ترسله.",
            "هل صار نشر فعلي لحد الآن، ولا ما بتعرف؟ ما في داعي تشارك المحتوى.",
        ),
        "publication_threat": (
            "هل صدر منه تهديد صريح بالنشر أو الإرسال، أم أن خوفك من احتمال فقط؟",
            "هل قال أو فعل شيئًا يدل على تهديد بالنشر، أم أنك تخشى احتمال ذلك دون تهديد واضح؟",
        ),
        "narrative": (
            "احكيلي شو صار معك، وخذ راحتك بالتفاصيل اللي بتحب تذكرها.",
            "ممكن توصفلي الموقف بكلماتك، حتى لو باختصار؟",
        ),
    }
    options = alternatives.get(slot)
    if not options:
        return preface + questions.get(slot, "في تفصيل ثاني بتحب تضيفه؟")
    history = state.get("question_history", [])
    occurrences = sum(1 for item in history if isinstance(item, dict) and item.get("slot") == slot)
    index = max(0, occurrences - (1 if slot == "narrative" else 0)) % len(options)
    if track == "defamation_or_abuse" and slot == "details":
        return preface + ("شو انكتب أو انحكى بالتعليق المسيء؟ ما في داعي تنقل ألفاظ ما بترتاح لها.",
                "بتقدر توصف معنى التعليق أو الكلام المسيء بدون ما تعيد ألفاظه كاملة؟")[occurrences % 2]
    return preface + options[index]


def _fact_value_for_context(fact: dict[str, Any]) -> str | bool | None:
    value = fact.get("value")
    if isinstance(value, (str, bool)) or value is None:
        return value[:260] if isinstance(value, str) else value
    return None


def build_dialogue_context(state: dict[str, Any], current_text: str | None = None,
                           bounded_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a compact, data-only local dialogue context; never include credentials or session IDs."""
    messages = []
    source_messages = [item for item in state.get("messages", []) if isinstance(item, dict)]
    if current_text is not None:
        source_messages.append({"role": "user", "text": current_text, "kind": "answer"})
    for item in source_messages[-6:]:
        role = item.get("role")
        content = item.get("text")
        if role in {"assistant", "user"} and isinstance(content, str):
            messages.append({"role": role, "text": content[:520]})
    assistant_questions = [item["text"] for item in source_messages
                           if item.get("role") == "assistant" and item.get("kind") in {"question", "clarification"}
                           and isinstance(item.get("text"), str)][-3:]
    facts = []
    for field, fact in state.get("facts", {}).items():
        if not isinstance(field, str) or not isinstance(fact, dict):
            continue
        if fact.get("status") not in {"reported", "denied", "unknown", "not_remembered", "declined", "unresolved"}:
            continue
        facts.append({"field": field, "status": fact.get("status"),
                      "value": _fact_value_for_context(fact)})
    context: dict[str, Any] = {
        "recent_messages": messages,
        "confirmed_facts": facts[-18:],
        "incident_track": state.get("incident_track") or classify_track(state),
        "useful_gaps": [str(key) for key, status in state.get("slot_states", {}).items()
                        if status in {"unasked", "asked"}][:16],
        "stopping_decision": {key: value for key, value in (state.get("stopping_decision") or {}).items()
                              if key in {"should_stop", "stop_reason", "confidence_note", "recommended_next_focus"}},
        "recent_assistant_questions": assistant_questions,
        "pending_slot": state.get("pending_slot"),
    }
    if isinstance(bounded_context, dict):
        # Reference IDs are opaque, session-local fact/message references, not session credentials.
        context["pending_question"] = bounded_context.get("pending_question")
        context["recent_relevant_messages"] = bounded_context.get("recent_relevant_messages", [])[-4:]
        context["related_facts"] = bounded_context.get("related_facts", [])[-10:]
        context["related_money_events"] = bounded_context.get("related_money_events", [])[-4:]
    # Keep the entire serialized context within the documented character budget.
    def serialized() -> str:
        return json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    while len(serialized()) > MAX_DIALOGUE_CONTEXT_CHARS and context["recent_messages"]:
        context["recent_messages"].pop(0)
    if len(serialized()) > MAX_DIALOGUE_CONTEXT_CHARS:
        for item in context["recent_messages"]:
            item["text"] = item["text"][:180]
        for item in context.get("recent_relevant_messages", []):
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                item["text"] = item["text"][:180]
    while len(serialized()) > MAX_DIALOGUE_CONTEXT_CHARS and context["confirmed_facts"]:
        context["confirmed_facts"].pop(0)
    if len(serialized()) > MAX_DIALOGUE_CONTEXT_CHARS:
        context.pop("related_money_events", None)
    if len(serialized()) > MAX_DIALOGUE_CONTEXT_CHARS:
        context.pop("related_facts", None)
    context["serialized_char_count"] = len(serialized())
    return context


_COMMON_QUESTION_WORDS = frozenset({
    "هل", "شو", "ما", "متى", "كيف", "وين", "اين", "اي", "اذا", "كان", "كانت",
    "او", "و", "في", "على", "من", "عن", "الى", "مع", "لك", "لديك", "عندك",
    "ممكن", "بتقدر", "تقدر", "هلأ", "الان", "تقريبا", "بس", "هو", "هي", "انت",
})


def _uncommon_question_terms(text: str) -> set[str]:
    normalized = normalize_arabic(text)
    return {word for word in re.findall(r"[a-z0-9]+|[\u0621-\u064a]+", normalized)
            if word not in _COMMON_QUESTION_WORDS}


def _questions_are_near_duplicates(first: str, second: str) -> bool:
    first_terms = _uncommon_question_terms(first)
    second_terms = _uncommon_question_terms(second)
    if not first_terms or not second_terms:
        return False
    union = first_terms | second_terms
    return bool(union) and len(first_terms & second_terms) / len(union) >= 0.78


def validate_dialogue_reply(text: str, *, should_stop: bool, urgent_stop: bool = False,
                            context: dict[str, Any] | None = None,
                            recent_questions: list[str] | None = None) -> tuple[bool, str | None]:
    """Validate a natural reply without imposing fixed wording or slot vocabulary."""
    if not isinstance(text, str):
        return False, "reply_not_text"
    stripped = text.strip()
    normalized = normalize_arabic(stripped)
    if not (8 <= len(stripped) <= 420):
        return False, "reply_length_out_of_bounds"
    arabic_letters = len(re.findall(r"[\u0621-\u064a]", stripped))
    if arabic_letters < 5:
        return False, "reply_not_understandable_arabic"
    question_count = stripped.count("؟") + stripped.count("?")
    if (should_stop and question_count) or (not should_stop and question_count > 1):
        return False, "question_count_conflicts_with_engine_stop" if should_stop else "multiple_questions"
    forbidden_patterns = (
        r"كلم(?:ة|ه)\s*المرور", r"رمز\s*(?:التحقق|التسجيل|otp)|\botp\b",
        r"(?:ارسل|ابعث|ارفع|حمل|أرفق)\s*(?:لي\s*)?(?:أي\s*)?(?:الصور?|صور(?:ه|تين)?|لقطات?(?:\s+شاشه)?|الدليل|الأدله|ملف)",
        r"(?:اكتب|اذكر|أرسل|اعطني|اعطيني|شارك)\s+(?:الاسم|اسم المستخدم|المعرف|رقم الهاتف|رقم الواتساب|الرابط)",
        r"(?:ما|شو)\s+(?:هو\s+|هي\s+)?(?:اسم الحساب|اسم المستخدم|المعرف|رقم الهاتف|رقم الواتساب|الرابط)",
        r"تم\s+تقديم\s+(?:ال)?بلاغ", r"قدمت\s+(?:ال)?بلاغ\s+رسمي(?:ا|ً)?", r"قدمنا\s+(?:ال)?بلاغ",
    )
    if any(re.search(normalize_arabic(pattern), normalized, re.IGNORECASE)
           for pattern in forbidden_patterns):
        return False, "unsafe_solicitation_or_unsupported_action_claim"
    if not urgent_stop and re.search(r"(?<!\d)911(?!\d)|النجده|الطوارئ|الاسعاف|خطر جسدي", normalized):
        return False, "unsolicited_emergency_content"
    available_numbers = set(re.findall(r"\d+|[٠-٩]+", json.dumps(context or {}, ensure_ascii=False)))
    if set(re.findall(r"\d+|[٠-٩]+", stripped)) - available_numbers:
        return False, "unsupported_number"
    for previous in (recent_questions or [])[-3:]:
        if isinstance(previous, str) and previous.strip():
            previous_normalized = normalize_arabic(previous).strip(" ؟?")
            if (previous_normalized and (
                    previous_normalized == normalized.strip(" ؟?")
                    or _questions_are_near_duplicates(previous_normalized, normalized.strip(" ؟?")))):
                return False, "repeated_recent_question"
    return True, None


def validate_reply_text(text: str, *, allow_emergency: bool = False) -> tuple[bool, str | None]:
    """Compatibility wrapper retained for existing tests and question flows."""
    accepted, code = validate_dialogue_reply(text, should_stop=False, urgent_stop=allow_emergency,
                                             context={}, recent_questions=[])
    if code == "unsafe_solicitation_or_unsupported_action_claim":
        code = "unsafe_solicitation_or_claim"
    return accepted, code
