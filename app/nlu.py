"""Arabic/Shami utterance normalization and conservative conversation signals.

This module is a deterministic safety and interpretation aid. It does not replace
model understanding when a local model is enabled, and normalized text is never
used as a stored quote.
"""
from __future__ import annotations

import re
from typing import Any


def normalize_arabic(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", value)
    value = (value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
             .replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي"))
    value = re.sub(r"([\u0600-\u06ff])\1{2,}", r"\1", value)
    replacements = {
        "علاقثه": "علاقه", "تيلغرام": "تلغرام", "تلجرام": "تلغرام",
        "تيليجرام": "تلغرام", "امبارحه": "مبارح", "البارحه": "مبارح",
        "بعتللي": "بعتلي", "صارلو": "صارله", "صرلو": "صارله",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return re.sub(r"\s+", " ", value).strip()


_WRAP_UP = (
    "بس هذا اللي صار", "هذا كل شي", "هذا كل شيء", "ما عندي شي اضيفه", "ما عندي شيء اضيفه",
    "ما عندي تفاصيل ثانية", "ما في تفاصيل ثانية", "خلاص", "خلص", "يكفي هيك", "بدي الملخص",
    "اعرض الملخص", "ما بدي اكمل", "ما عندي اشي ثاني",
)
_UNKNOWN = ("ما بعرف", "ما اعرف", "مش عارف", "مش عارفه", "لا اعرف", "مجهول", "ما عندي فكرة", "مش متاكد")
_NOT_REMEMBERED = ("ما بتذكر", "مش متذكر", "مش متذكره", "لا اتذكر", "ناسي", "ناسية")
_PROTEST = ("قلت لك", "حكيتلك", "اخبرتك", "جاوبتك", "سبق وقلت", "حكيتلك قبل", "سألتني نفس السؤال")
_FATIGUE = ("تعبت", "زهقت", "مرهق", "مرهقة", "مش قادر اكمل", "ما بقدر اكمل", "لا اريد المتابعة")


def _has_any(normalized_text: str, markers: tuple[str, ...]) -> bool:
    return any(normalize_arabic(marker) in normalized_text for marker in markers)


def _has_term_sequence(normalized_text: str, terms: tuple[str, ...]) -> bool:
    """Match Arabic words/phrases without relying on Unicode \b semantics."""
    tokens = re.findall(r"[\u0621-\u064a]+|[a-z0-9]+", normalized_text)

    def forms(token: str) -> set[str]:
        result = {token}
        stem = token
        while len(stem) > 3 and stem[0] in "وفبكل":
            stem = stem[1:]
            result.add(stem)
        if stem.startswith("ال") and len(stem) > 3:
            result.add(stem[2:])
        return result

    for term in terms:
        target = re.findall(r"[\u0621-\u064a]+|[a-z0-9]+", normalize_arabic(term))
        if target and any(all(word in forms(candidate) for word, candidate in zip(target, tokens[index:index + len(target)]))
                          for index in range(len(tokens) - len(target) + 1)):
            return True
    return False


def analyze_message(text: str) -> dict[str, Any]:
    """Return low-risk categorical signals; do not infer unsupported case facts."""
    normalized = normalize_arabic(text)
    identity_claim = bool(re.search(
        r"(?:بحكي|بيحكي|يحكي|بقول|بيقول|يقول|بدعي|يدعي|يتظاهر|يزعم)[^،؛.!؟?]{0,55}(?:انه|اني|كانه|كانها) انا"
        r"|(?:انتحال|منتحل|ينتحل|انتحل)\s+(?:شخصيتي|هويتي)|(?:تنتحل|تنتحل دوري)", normalized))
    identity_claim_denied = _has_any(normalized, (
        "ما حكي للناس انه انا", "ما حكى للناس انه انا", "ما قال للناس انه انا",
        "ما ادعي انه انا", "ما يدعي انه انا", "مش حاكي انه انا", "ما بحكي انه انا",
        "ما بتظاهر انه انا", "ما قال انه انا",
    ))
    return {
        "wrap_up": _has_any(normalized, _WRAP_UP),
        "unknown": _has_any(normalized, _UNKNOWN),
        "not_remembered": _has_any(normalized, _NOT_REMEMBERED),
        "protest": _has_any(normalized, _PROTEST),
        "fatigue": _has_any(normalized, _FATIGUE),
        "verbal_abuse": _has_term_sequence(normalized, (
            "شتمني", "شتمتني", "شتم", "سب علي", "سبني", "سب", "اهانني", "إهانة", "مسيء",
            "مسيئة", "يهينني", "اهانني", "أساء لي", "ذم", "قدح", "شهّر في", "شهر فيني",
            "يشهر فيني", "تشهير")),
        "digital_context": _has_any(normalized, (
            "تعليق", "منشور", "بوست", "ستوري", "انستغرام", "انستجرام", "فيسبوك", "واتساب",
            "تيك توك", "تلغرام", "رسالة", "اونلاين", "على الانترنت", "حساب",
        )),
        "explicit_identity_claim": identity_claim and not identity_claim_denied,
        "lookalike_only": _has_any(normalized, (
            "اسم قريب", "اسم مشابه", "شبيه باسمي", "حساب مشابه", "ما بحكي انه انا", "ما يدعي انه انا",
            "ما حكى للناس انه انا", "ما قال للناس انه انا", "ما حكي للناس انه انا",
        )),
        "fan_account": _has_any(normalized, ("حساب معجبين", "حساب فانز", "مكتوب انه فانز", "صفحة معجبين")),
        "evidence_only_statement": bool(re.search(
            r"^\s*(?:عندي|معي|متوفر عندي).{0,70}(?:سكرين\s*شوت|لقطات?|صور|ايميلات|رسائل|محادثات).{0,60}(?:تثبت|توضح|تخص).*$",
            normalized)),
        "publication_threat": _has_any(normalized, (
            "يهدد ينشر", "يهددني بنشر", "هدد بنشر", "هددني بنشر", "تهديد بالنشر", "اذا ما رجعت", "يفضحني",
            "والا ينشر", "الا ينشر", "الا يبعت", "وإلا ينشر",
        )),
        "extortion": (_has_any(normalized, ("ابتزاز", "يطلب مصاري", "طلب مني مصاري", "طلب مني مبلغ", "اذا ما دفعت", "اذا ما حولت"))
                      or bool(re.search(r"طلب\s+مني\s+\d+\s*(?:دينار|دولار|شيكل|ريال)", normalized))),
        "account_takeover": _has_any(normalized, ("اخترق", "دخل على حسابي", "دخل حسابي", "سرق حسابي", "غير كلمة المرور")),
        "financial_deception": _has_any(normalized, ("نصب", "احتيال", "خدعني", "اختفى بعد ما حولت", "وعدني يبيعني")),
        "harassment_cue": _has_any(normalized, (
            "يلاحقني", "بضايقني", "مضايقة", "يراسلني كل يوم", "رسائل متكررة", "ما زال يبعثلي رسائل",
            "لسه يبعثلي رسائل", "ما زال يرسل رسائل", "ما زال يتواصل",
        )),
        "explicit_physical_danger": _has_any(normalized, (
            "واقف عند باب", "جاي لعندي", "معه سكين", "معه مسدس", "هددني بالقتل هسا", "يضربني الان",
        )),
    }


def is_wrap_up(text: str) -> bool:
    return bool(analyze_message(text)["wrap_up"])
