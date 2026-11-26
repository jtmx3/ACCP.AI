"""Natural Arabic dialogue phrasing and safe fallback question selection."""
from __future__ import annotations

from typing import Any

from app.nlu import normalize_arabic
from app.triage import classify_track


def compose_fallback_question(state: dict[str, Any], slot: str, questions: dict[str, str]) -> str:
    """Select a concise, context-aware alternative for rules-only operation.

    The question target remains application-owned. Rotating among equivalent
    phrasings is deliberately deterministic, preserving reproducibility.
    """
    track = classify_track(state)
    alternatives: dict[str, tuple[str, ...]] = {
        "details": (
            "إذا بناسبك، شو التصرف أو الكلام اللي صار بالتحديد؟",
            "ممكن توصفلي شو قال أو عمل، بالقدر اللي بترتاح تحكيه؟",
            "شو أهم تفصيل بتحب ينحفظ عن اللي صار؟",
        ),
        "platform": (
            "على أي تطبيق أو موقع صار هالشي؟",
            "وين ظهر الكلام أو التصرف، إذا بتعرف؟",
        ),
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
        "narrative": (
            "احكيلي شو صار معك، وخذ راحتك بالتفاصيل اللي بتحب تذكرها.",
            "ممكن توصفلي الموقف بكلماتك، حتى لو باختصار؟",
        ),
    }
    options = alternatives.get(slot)
    if not options:
        return questions.get(slot, "في تفصيل ثاني بتحب تضيفه؟")
    history = state.get("question_history", [])
    occurrences = sum(1 for item in history if isinstance(item, dict) and item.get("slot") == slot)
    # Keep the familiar opening on a fresh case; vary follow-ups and across tracks.
    index = max(0, occurrences - (1 if slot == "narrative" else 0)) % len(options)
    if track == "defamation_or_abuse" and slot == "details":
        return ("شو انكتب أو انحكى بالتعليق المسيء؟ ما في داعي تنقل ألفاظ ما بترتاح لها.",
                "بتقدر توصف معنى التعليق أو الكلام المسيء بدون ما تعيد ألفاظه كاملة؟")[occurrences % 2]
    return options[index]


def validate_reply_text(text: str, *, allow_emergency: bool = False) -> tuple[bool, str | None]:
    normalized = normalize_arabic(text)
    if not (8 <= len(text.strip()) <= 420):
        return False, "reply_length_out_of_bounds"
    if text.count("؟") + text.count("?") > 1:
        return False, "multiple_questions"
    forbidden = ("كلمة المرور", "رمز التحقق", "otp", "ارسل الصورة", "ارفع الملف", "ارسل الدليل",
                 "قدمت بلاغ نيابة", "تم تقديم بلاغ")
    if any(normalize_arabic(marker) in normalized for marker in forbidden):
        return False, "unsafe_solicitation_or_claim"
    if not allow_emergency and any(marker in normalized for marker in ("911", "النجدة", "الطوارئ", "خطر جسدي")):
        return False, "unsolicited_emergency_content"
    return True, None
