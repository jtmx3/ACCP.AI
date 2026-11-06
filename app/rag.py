from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge.json"
STOPWORDS = {
    "كيف", "هل", "ماذا", "لماذا", "شو", "ايش", "متى", "وين", "من", "على", "الى", "عن", "في", "مع", "هذا", "هذه", "ذلك", "انا", "انت", "كان", "كانت",
    "معنى", "كلمة", "لغة", "اللغة", "عربية", "العربية", "شرح", "يعني", "يعنيها", "ماهو", "هو", "هي", "افعل", "أفعل", "عبر", "اذا", "التي", "بعد", "ما", "منه", "لها", "له", "كنت", "زلت", "بس", "قد", "what", "how", "when", "where", "why", "who", "which", "would", "should", "could", "with", "from", "that", "this", "there", "their", "have", "has", "are", "the", "and", "for", "but", "you", "your", "can", "meaning", "word", "language", "arabic", "i", "me", "my", "do", "does", "if", "via", "a", "an", "of", "in", "or", "on", "it", "to",
}
MIN_QUERY_TERMS = 2
MIN_OVERLAP = 2
MIN_QUERY_COVERAGE = 0.40

PLATFORM_ALIASES = {
    "whatsapp": ("واتساب", "واتس اب", "واتسآب", "whatsapp"),
    "instagram": ("انستغرام", "انستجرام", "instagram"),
    "facebook": ("فيسبوك", "فيس بوك", "facebook"),
    "tiktok": ("تيك توك", "tiktok"),
    "snapchat": ("سناب شات", "سناب", "snapchat"),
    "telegram": ("تلغرام", "تليجرام", "تيليجرام", "telegram"),
    "signal": ("سيجنال", "signal"),
    "x": ("تويتر", "twitter", "x.com"),
}
PLATFORM_LABELS = {
    "whatsapp": "واتساب", "instagram": "إنستغرام", "facebook": "فيسبوك",
    "tiktok": "تيك توك", "snapchat": "سناب شات", "telegram": "تلغرام",
    "signal": "سيجنال", "x": "X / تويتر",
}
PLATFORM_REQUIRED_TOPICS = {
    "account_security", "account_recovery", "impersonation_reporting",
    "platform_reporting", "device_session_management",
}


class GuidanceScopeError(ValueError):
    """Raised for malformed topic/platform metadata in the local knowledge corpus."""


def load_corpus(path: Path = CORPUS_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        corpus = json.load(f)
    if not isinstance(corpus.get("sources"), list):
        raise ValueError("Knowledge corpus must contain a sources list")
    required = {"id", "title", "publisher", "url", "published_or_updated", "ingested_at", "geography", "content_version", "text", "topics", "platforms"}
    source_ids: set[str] = set()
    valid_platforms = set(PLATFORM_ALIASES)
    for source in corpus["sources"]:
        missing = required - source.keys()
        if missing:
            raise ValueError(f"Source {source.get('id')} is missing metadata: {sorted(missing)}")
        if source["id"] in source_ids:
            raise ValueError(f"Duplicate source id: {source['id']}")
        source_ids.add(source["id"])
        if not source["url"].startswith("https://") or not source["geography"]:
            raise ValueError(f"Source {source['id']} has an invalid URL or geography")
        for field in ("keywords", "required_keywords", "excluded_keywords", "topics", "platforms"):
            values = source.get(field, [])
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"Source {source['id']} has invalid {field}")
        if not source["topics"]:
            raise ValueError(f"Source {source['id']} must declare at least one retrieval topic")
        platforms = set(source["platforms"])
        if platforms - valid_platforms:
            raise ValueError(f"Source {source['id']} has unsupported platform tags: {sorted(platforms - valid_platforms)}")
        if (str(source["geography"]).upper() == "GLOBAL" or source.get("guidance_scope") == "global_platform") and not platforms:
            raise ValueError(f"Global platform source {source['id']} must declare its platform")
    if not isinstance(corpus.get("geography_labels", {}), dict):
        raise ValueError("Knowledge corpus geography_labels must be an object")
    return corpus


def available_jurisdictions(corpus: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    corpus = corpus or load_corpus()
    labels = corpus.get("geography_labels", {})
    codes = sorted({str(source["geography"]).upper() for source in corpus["sources"]
                    if str(source["geography"]).upper() != "GLOBAL"})
    return [{"code": code, "label": labels.get(code, code),
             "source_count": sum(1 for source in corpus["sources"] if str(source["geography"]).upper() == code)} for code in codes]


def _normalize(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[\u064b-\u065f\u0670]", "", value)
    return value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")


NORMALIZED_STOPWORDS = {_normalize(word) for word in STOPWORDS}
ACCOUNT_TERMS = {"حساب", "حسابي", "حسابك", "حسابه", "حسابها", "حسابات", "account", "accounts", "profile"}
PROTECTION_TERMS = {
    "احمي", "احم", "احافظ", "حافظ", "احفظ", "اؤمن", "اامن", "حمايه", "حمايه", "حمايه", "حفظ",
    "تامين", "لتامين", "امان", "امن", "مؤمن", "حمايه", "secure", "security", "protect", "protection",
    "safeguard", "safe", "keep", "securely", "two", "step", "verification", "passkey", "password",
    "المصادقه", "الثنائيه", "تحقق", "خطوتين",
}
RECOVERY_TERMS = {
    "استرداد", "استرجاع", "استرجع", "استعيد", "استعادة", "مخترق", "اختراق", "انسرق", "مسروق",
    "recover", "recovery", "regain", "hacked", "hack", "compromised", "stolen", "locked",
}


def _tokens(value: str) -> set[str]:
    result: set[str] = set()
    for token in re.findall(r"[\w\u0600-\u06ff]+", _normalize(value)):
        token = token.strip(".,،؛:؟?!()[]{}\"'“”«»_")
        if len(token) <= 2 or token in NORMALIZED_STOPWORDS:
            continue
        stem = token
        for _ in range(2):
            prefix = next((item for item in ("وال", "ولل", "بال", "كال", "لل", "ال")
                           if stem.startswith(item) and len(stem) - len(item) > 2), None)
            if not prefix:
                break
            stem = stem[len(prefix):]
        if stem not in NORMALIZED_STOPWORDS:
            result.add(stem)
    return result


def _search_terms(value: str) -> set[str]:
    terms = _tokens(value)
    if terms & ACCOUNT_TERMS:
        terms.add("__account__")
    if terms & PROTECTION_TERMS:
        terms.add("__protection__")
    if terms & RECOVERY_TERMS:
        terms.add("__recovery__")
    return terms


def _platforms_in_text(value: str) -> set[str]:
    folded = _normalize(value)
    found = set()
    for platform, aliases in PLATFORM_ALIASES.items():
        if any(_normalize(alias) in folded for alias in aliases):
            found.add(platform)
    return found


def _query_topics(value: str) -> set[str]:
    terms = _tokens(value)
    normalized = _normalize(value)
    platforms = _platforms_in_text(value)
    has_account = bool(terms & ACCOUNT_TERMS)
    topics: set[str] = set()

    if has_account and terms & PROTECTION_TERMS:
        topics.add("account_security")
    if has_account and terms & RECOVERY_TERMS:
        topics.update({"account_recovery", "account_security"})
    if platforms and terms & (PROTECTION_TERMS | RECOVERY_TERMS):
        topics.add("account_security" if terms & PROTECTION_TERMS else "account_recovery")
    if terms & {"two", "step", "verification", "passkey", "password", "المصادقه", "الثنائيه", "خطوتين"}:
        if platforms or has_account:
            topics.add("account_security")

    if terms & {"دليل", "ادله", "لقطه", "لقطات", "اثبات", "توثيق", "evidence", "screenshot", "screenshots"}:
        topics.add("evidence_preservation")
    if any(marker in normalized for marker in ("صور حميمه", "صور خاصه", "نشر صوري", "ابتزاز الصور", "image-based abuse", "intimate images", "sextortion")):
        topics.add("image_based_abuse")
    if terms & {"انتحال", "ينتحل", "انتحل", "مزيف", "مزيفه", "يدعي", "يتظاهر", "impersonation", "impersonating", "fake"}:
        topics.update({"impersonation_reporting", "platform_reporting"})

    money_action = terms & {"دفع", "دفعت", "دفعه", "تحويل", "حول", "حولت", "مصاري", "مال", "مبلغ", "bank", "payment", "paid", "transfer", "refund", "استرداد", "استرجاع", "استرجع", "استعيد", "احتيال", "محتال", "نصب", "scam", "fraud"}
    if len(money_action) >= 2 or (money_action & {"استرداد", "استرجاع", "استرجع", "refund"}):
        topics.add("payment_recovery")

    report_terms = {"بلاغ", "ابلغ", "ابلاغ", "تقرير", "report", "reporting", "وحده", "unit"}
    cybercrime_terms = {"جريمه", "جرائم", "crime", "cybercrime", "196", "ecrimes"}
    if terms & cybercrime_terms and (terms & report_terms or terms & {"196", "ecrimes", "وحده", "unit"}):
        topics.add("cybercrime_reporting")
    if terms & {"حادث", "incident"} and terms & {"سيبراني", "الكتروني", "امن", "cybersecurity", "security", "jocert"}:
        topics.add("cybersecurity_incident_reporting")
    if terms & {"طوارئ", "911", "خطر", "مباشر", "emergency"}:
        topics.add("emergency")

    return topics


def _excerpt(source: dict[str, Any], query_terms: set[str], max_chars: int = 480) -> str:
    text = str(source.get("text", "")).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?؟])\s+", text) if part.strip()]
    if not sentences:
        return text[:max_chars]
    ranked = sorted(enumerate(sentences), key=lambda item: (-len(query_terms & _search_terms(item[1])), item[0]))
    chosen: list[tuple[int, str]] = []
    total = 0
    for index, sentence in ranked:
        if not (query_terms & _search_terms(sentence)):
            continue
        if chosen and total + len(sentence) > max_chars:
            continue
        chosen.append((index, sentence))
        total += len(sentence) + 1
        if len(chosen) >= 2:
            break
    if not chosen:
        chosen = [(0, sentences[0])]
    chosen.sort(key=lambda item: item[0])
    excerpt = " ".join(sentence for _, sentence in chosen)
    return excerpt if len(excerpt) <= max_chars else excerpt[:max_chars].rsplit(" ", 1)[0] + "…"


def _explicit_scope(query: str) -> str | None:
    normalized = _normalize(query)
    cues = (("JO", ("الاردن", "بالاردن", "للاردن", "اردني", "الأردن", "jordan")),
            ("US", ("الولايات المتحدة", "امريكا", "الامريكية", "امريكي", "united states", "usa")),
            ("AU", ("استراليا", "استرالي", "australia")))
    for code, markers in cues:
        if any(_normalize(marker) in normalized for marker in markers):
            return code
    return None


def _requires_jurisdiction(query: str) -> bool:
    normalized = _normalize(query)
    return any(marker in normalized for marker in (
        "ابلغ", "ابلاغ", "بلاغ", "الشرطه", "الشرطة", "قانوني", "رقم الطوارئ", "رقم 911", "رقم 196",
        "جهة مختصة", "محكمة", "تعويض قانوني", "report cybercrime", "official report", "police", "legal advice", "emergency number",
    ))


def _resolve_platform(query: str, platform_context: str | list[str] | None,
                      platform_context_ambiguous: bool) -> tuple[str | None, bool, set[str]]:
    current = _platforms_in_text(query)
    if len(current) > 1:
        return None, True, current
    if len(current) == 1:
        return next(iter(current)), False, current
    if platform_context_ambiguous:
        return None, True, set()
    prior_text = " ".join(platform_context) if isinstance(platform_context, list) else (platform_context or "")
    prior = _platforms_in_text(prior_text)
    if len(prior) > 1:
        return None, True, prior
    return (next(iter(prior)) if prior else None), False, prior


def retrieve(query: str, jurisdiction: str | None, limit: int = 3, corpus: dict[str, Any] | None = None,
             context_text: str = "", allow_nonlocal_general: bool = False,
             platform: str | None = None, topics: set[str] | None = None) -> list[dict[str, Any]]:
    corpus = corpus or load_corpus()
    scope = (jurisdiction or _explicit_scope(query) or "").strip().upper()
    available = {item["code"] for item in available_jurisdictions(corpus)}
    if scope and scope not in available:
        return []
    query_topics = set(topics) if topics is not None else _query_topics(query)
    if not query_topics:
        return []
    query_terms = _search_terms(query)
    if len(query_terms) < MIN_QUERY_TERMS and context_text:
        query_terms |= _search_terms(context_text)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for source in corpus["sources"]:
        geography = str(source["geography"]).upper()
        general_practice = source.get("guidance_scope") == "general_practice"
        global_platform = source.get("guidance_scope") == "global_platform" or geography == "GLOBAL"
        allowed_geography = (global_platform or general_practice) if not scope else (
            geography == scope or global_platform or (allow_nonlocal_general and general_practice))
        if not allowed_geography:
            continue
        source_platforms = set(source.get("platforms", []))
        if source_platforms and (not platform or platform not in source_platforms):
            continue
        source_topics = set(source.get("topics", []))
        matched_topics = query_topics & source_topics
        if not matched_topics:
            continue
        query_tokens = _tokens(query)
        excluded = set().union(*(_tokens(term) for term in source.get("excluded_keywords", []))) if source.get("excluded_keywords") else set()
        if query_tokens & excluded:
            continue
        required = set().union(*(_tokens(term) for term in source.get("required_keywords", []))) if source.get("required_keywords") else set()
        if required and geography != "GLOBAL" and not (query_tokens & required):
            continue
        source_terms = _search_terms(" ".join((source["text"], source["title"], " ".join(source.get("keywords", [])))))
        overlap_terms = query_terms & source_terms
        overlap = len(overlap_terms)
        coverage = overlap / max(1, len(query_terms))
        # Topic, platform, and geography have already been checked independently.
        # Keyword overlap ranks sources within that safe scope; it cannot override scope mismatch.
        if not overlap and not matched_topics:
            continue
        score = len(matched_topics) * 1.0 + coverage * 0.20 + overlap * 0.03
        candidate = dict(source)
        candidate["excerpt"] = _excerpt(source, query_terms)
        if general_practice and geography != scope:
            candidate["scope_note"] = "إرشاد عام من مصدر وطني؛ لا يمثل قانونًا أو جهة محلية."
        elif global_platform:
            candidate["scope_note"] = f"إرشادات خاصة بمنصة {PLATFORM_LABELS.get(platform or '', platform or 'محددة')} فقط."
        else:
            candidate["scope_note"] = None
        scored.append((score, overlap, candidate))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]["id"]))
    return [source for _, _, source in scored[:limit]]


def answer_question(query: str, jurisdiction: str | None, context_text: str = "",
                    allow_nonlocal_general: bool = False, *,
                    platform_context: str | list[str] | None = None,
                    platform_context_ambiguous: bool = False,
                    platform_query: str | None = None,
                    scope_query: str | None = None,
                    topic_query: str | None = None) -> dict[str, Any]:
    corpus = load_corpus()
    current_topic_text = topic_query if topic_query is not None else query
    topics = _query_topics(current_topic_text)
    current_platform_text = platform_query if platform_query is not None else query
    platform, platform_ambiguous, mentioned_platforms = _resolve_platform(
        current_platform_text, platform_context, platform_context_ambiguous)
    explicit_scope = _explicit_scope(scope_query if scope_query is not None else query)
    # A country named in the current question is a deliberate current-turn scope;
    # otherwise use only the saved jurisdiction, never a country inferred from topic text.
    effective_scope = (explicit_scope or jurisdiction or "").strip().upper() or None
    platform_needed = bool(topics & PLATFORM_REQUIRED_TOPICS)

    if platform_needed and platform_ambiguous:
        response = "ظهر أكثر من تطبيق أو تعارضت معلومات المنصة. أي تطبيق تقصد في هذا السؤال؟ سأقتصر على مصدره الرسمي."
        return {"text": response, "citations": [], "corpus_version": corpus["corpus_version"],
                "resolved_platform": None, "resolved_topics": sorted(topics), "resolved_jurisdiction": effective_scope}
    if platform_needed and platform is None:
        response = "لأي تطبيق تقصد؟ اذكر اسم التطبيق، مثل واتساب أو إنستغرام، لأعرض إرشادات موثقة تخصه فقط."
        return {"text": response, "citations": [], "corpus_version": corpus["corpus_version"],
                "resolved_platform": None, "resolved_topics": sorted(topics), "resolved_jurisdiction": effective_scope}

    sources = retrieve(current_topic_text, effective_scope, corpus=corpus, context_text=context_text,
                       allow_nonlocal_general=(allow_nonlocal_general or not _requires_jurisdiction(query)),
                       platform=platform, topics=topics)
    if not sources:
        available = {item["code"] for item in available_jurisdictions(corpus)}
        if not effective_scope and _requires_jurisdiction(query):
            response = "يعتمد مسار الإبلاغ أو رقم الطوارئ على البلد. اذكر البلد الذي تسأل عنه لأبحث في مصدر رسمي مناسب؛ لن أفترض موقعك."
        elif effective_scope and effective_scope not in available:
            response = "ما عندي مصدر موثوق محلي لهذا الاختصاص في قاعدة المعرفة الحالية؛ لا أستطيع تقديم إرشاد محلي موثوق."
        elif platform and platform_needed:
            label = PLATFORM_LABELS.get(platform, platform)
            response = f"ما عندي في قاعدة المعرفة الحالية مصدر رسمي مناسب لمنصة {label} لهذا الموضوع؛ لن أعرض خطوات من منصة أخرى."
        elif not topics and not effective_scope:
            response = "لا يوجد في قاعدة المعرفة مقتطف كافٍ يجيب عن هذا السؤال. لا أستنتج خطوات أو نتائج غير مدعومة بمصدر موثوق."
        else:
            response = "ما عندي مصدر موثوق أو مقتطف كافٍ يطابق موضوع السؤال ضمن النطاق المحدد؛ أمتنع عن الاستنتاج من تطابق لفظي ضعيف."
        return {"text": response, "citations": [], "corpus_version": corpus["corpus_version"],
                "resolved_platform": platform, "resolved_topics": sorted(topics), "resolved_jurisdiction": effective_scope}

    if effective_scope:
        lines = ["بحسب المقتطفات الموثقة؛ تظهر جهة المصدر ونطاقها بجوار كل مقتطف:"]
    else:
        lines = ["هذه مقتطفات موثقة من دون افتراض بلدك؛ المصادر الوطنية ليست مسارات محلية خارج بلدها:"]
    citations = []
    for number, source in enumerate(sources, start=1):
        excerpt = source["excerpt"]
        lines.append(f"\n[{number}] {source['title']} — {source['publisher']}")
        for step in source.get("answer_steps", []):
            lines.append(f"\n- {step}")
        if not source.get("answer_steps"):
            lines.append(f"\n«{excerpt}»")
        if source.get("scope_note"):
            lines.append(f"\nنطاق المصدر: {source['scope_note']}")
        lines.append(f"\nالمقتطف الداعم: «{excerpt}»")
        citations.append({"id": source["id"], "title": source["title"], "publisher": source["publisher"], "url": source["url"],
                          "published_or_updated": source["published_or_updated"], "ingested_at": source["ingested_at"],
                          "geography": source["geography"], "guidance_scope": source.get("guidance_scope"),
                          "topics": source.get("topics", []), "platforms": source.get("platforms", []),
                          "scope_note": source.get("scope_note"), "content_version": source["content_version"], "excerpt": excerpt})
    lines.append("\nهذه مقتطفات من المصادر فقط؛ لا تثبت ما حدث في حالتك ولا تمثل نصيحة قانونية. إذا لم يُجب المقتطف عن سؤالك، أمتنع عن إضافة استنتاج.")
    return {"text": "".join(lines), "citations": citations, "corpus_version": corpus["corpus_version"],
            "resolved_platform": platform, "resolved_topics": sorted(topics), "resolved_jurisdiction": effective_scope}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate the local RAG corpus metadata")
    parser.add_argument("--reindex", action="store_true", help="Validate and reload source metadata; does not build an embeddings or semantic index")
    args = parser.parse_args()
    if args.reindex:
        corpus = load_corpus()
        print(f"Validated corpus metadata {corpus['corpus_version']} with {len(corpus['sources'])} sources; no semantic index was built.")
