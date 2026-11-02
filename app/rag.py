from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge.json"
STOPWORDS = {
    "كيف", "هل", "ماذا", "لماذا", "شو", "ايش", "متى", "وين", "من", "على", "الى", "عن", "في", "مع", "هذا", "هذه", "ذلك", "انا", "انت", "كان", "كانت",
    "معنى", "كلمة", "لغة", "اللغة", "عربية", "العربية", "شرح", "يعني", "يعنيها", "ماهو", "هو", "هي", "افعل", "أفعل", "عبر", "اذا", "التي", "بعد", "ما", "منه", "لها", "له", "what", "how", "when", "where", "why", "who", "which", "would", "should", "could", "with", "from", "that", "this", "there", "their", "have", "has", "are", "the", "and", "for", "but", "you", "your", "can", "meaning", "word", "language", "arabic", "i", "me", "my", "do", "does", "if", "via", "a", "an", "of", "in", "or", "on", "it", "to",
}
MIN_QUERY_TERMS = 2
MIN_OVERLAP = 2
MIN_QUERY_COVERAGE = 0.40


def load_corpus(path: Path = CORPUS_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        corpus = json.load(f)
    if not isinstance(corpus.get("sources"), list):
        raise ValueError("Knowledge corpus must contain a sources list")
    required = {"id", "title", "publisher", "url", "published_or_updated", "ingested_at", "geography", "content_version", "text"}
    source_ids: set[str] = set()
    for source in corpus["sources"]:
        missing = required - source.keys()
        if missing:
            raise ValueError(f"Source {source.get('id')} is missing metadata: {sorted(missing)}")
        if source["id"] in source_ids:
            raise ValueError(f"Duplicate source id: {source['id']}")
        source_ids.add(source["id"])
        if not source["url"].startswith("https://") or not source["geography"]:
            raise ValueError(f"Source {source['id']} has an invalid URL or geography")
        for field in ("keywords", "required_keywords", "excluded_keywords"):
            values = source.get(field, [])
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"Source {source['id']} has invalid {field}")
    if not isinstance(corpus.get("geography_labels", {}), dict):
        raise ValueError("Knowledge corpus geography_labels must be an object")
    return corpus


def available_jurisdictions(corpus: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    corpus = corpus or load_corpus()
    labels = corpus.get("geography_labels", {})
    codes = sorted({str(source["geography"]).upper() for source in corpus["sources"]})
    return [{"code": code, "label": labels.get(code, code),
             "source_count": sum(1 for source in corpus["sources"] if str(source["geography"]).upper() == code)} for code in codes]


def _normalize(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[\u064b-\u065f\u0670]", "", value)
    return value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")


NORMALIZED_STOPWORDS = {_normalize(word) for word in STOPWORDS}


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


def _excerpt(source: dict[str, Any], query_terms: set[str], max_chars: int = 480) -> str:
    text = str(source.get("text", "")).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?؟])\s+", text) if part.strip()]
    if not sentences:
        return text[:max_chars]
    ranked = sorted(enumerate(sentences), key=lambda item: (-len(query_terms & _tokens(item[1])), item[0]))
    chosen: list[tuple[int, str]] = []
    total = 0
    for index, sentence in ranked:
        if not (query_terms & _tokens(sentence)):
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


def retrieve(query: str, jurisdiction: str | None, limit: int = 3, corpus: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    corpus = corpus or load_corpus()
    scope = (jurisdiction or "").strip().upper()
    available = {item["code"] for item in available_jurisdictions(corpus)}
    if scope not in available:
        return []
    query_terms = _tokens(query)
    if len(query_terms) < MIN_QUERY_TERMS:
        return []
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for source in corpus["sources"]:
        if str(source["geography"]).upper() != scope:
            continue
        excluded = set().union(*(_tokens(term) for term in source.get("excluded_keywords", []))) if source.get("excluded_keywords") else set()
        if query_terms & excluded:
            continue
        required = set().union(*(_tokens(term) for term in source.get("required_keywords", []))) if source.get("required_keywords") else set()
        if required and not (query_terms & required):
            continue
        terms = _tokens(" ".join((source["text"], source["title"], " ".join(source.get("keywords", [])))))
        overlap_terms = query_terms & terms
        overlap = len(overlap_terms)
        coverage = overlap / len(query_terms)
        if overlap < MIN_OVERLAP or coverage < MIN_QUERY_COVERAGE:
            continue
        # Prefer sources that cover several distinct query concepts, not one common word.
        score = coverage + overlap * 0.08
        candidate = dict(source)
        candidate["excerpt"] = _excerpt(source, query_terms)
        scored.append((score, overlap, candidate))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]["id"]))
    return [source for _, _, source in scored[:limit]]


def answer_question(query: str, jurisdiction: str | None) -> dict[str, Any]:
    corpus = load_corpus()
    sources = retrieve(query, jurisdiction, corpus=corpus)
    if not sources:
        scope = (jurisdiction or "").strip().upper()
        available = {item["code"] for item in available_jurisdictions(corpus)}
        if not scope:
            response = "ما عندي مصدر موثوق محلي لهذا السؤال قبل تحديد البلد أو الاختصاص. لا أستنتج قانونًا أو جهة تواصل من دون مصدر واختصاص واضحين."
        elif scope not in available:
            response = "ما عندي مصدر موثوق محلي لهذا الاختصاص في قاعدة المعرفة الحالية؛ لا أستطيع تقديم إرشاد محلي موثوق."
        else:
            response = "ما عندي مصدر موثوق أو مقتطف كافٍ يطابق سؤالك ضمن الاختصاص المحدد. أمتنع عن الاستنتاج من تطابق لفظي ضعيف."
        return {"text": response, "citations": [], "corpus_version": corpus["corpus_version"]}

    lines = ["بحسب المقتطفات الموثقة ضمن الاختصاص المحدد:"]
    citations = []
    for number, source in enumerate(sources, start=1):
        excerpt = source["excerpt"]
        lines.append(f"\n[{number}] «{excerpt}»")
        citations.append({"id": source["id"], "title": source["title"], "publisher": source["publisher"], "url": source["url"],
                          "published_or_updated": source["published_or_updated"], "ingested_at": source["ingested_at"],
                          "geography": source["geography"], "content_version": source["content_version"], "excerpt": excerpt})
    lines.append("\nهذه مقتطفات من المصادر فقط؛ لا تثبت ما حدث في حالتك ولا تمثل نصيحة قانونية. إذا لم يُجب المقتطف عن سؤالك، أمتنع عن إضافة استنتاج.")
    return {"text": "".join(lines), "citations": citations, "corpus_version": corpus["corpus_version"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate the local RAG corpus metadata")
    parser.add_argument("--reindex", action="store_true", help="Validate and reload source metadata; does not build an embeddings or semantic index")
    args = parser.parse_args()
    if args.reindex:
        corpus = load_corpus()
        print(f"Validated corpus metadata {corpus['corpus_version']} with {len(corpus['sources'])} sources; no semantic index was built.")
