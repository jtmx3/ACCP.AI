from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge.json"
STOPWORDS = {"كيف", "هل", "ماذا", "لماذا", "شو", "ايش", "متى", "وين", "من", "على", "الى", "عن", "في", "مع", "هذا", "هذه", "ذلك", "انا", "انت", "كان", "كانت", "what", "how", "when", "where", "why", "who", "which", "would", "should", "could", "with", "from", "that", "this", "there", "their", "have", "has", "are", "the", "and", "for", "but", "you", "your", "can"}


def load_corpus(path: Path = CORPUS_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        corpus = json.load(f)
    if not isinstance(corpus.get("sources"), list):
        raise ValueError("Knowledge corpus must contain a sources list")
    required = {"id", "title", "publisher", "url", "published_or_updated", "ingested_at", "geography", "content_version", "text"}
    for source in corpus["sources"]:
        missing = required - source.keys()
        if missing:
            raise ValueError(f"Source {source.get('id')} is missing metadata: {sorted(missing)}")
    return corpus


def _tokens(value: str) -> set[str]:
    value = value.lower()
    value = re.sub(r"[\u064b-\u065f\u0670]", "", value)
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    result: set[str] = set()
    for token in re.findall(r"[\w\u0600-\u06ff]+", value):
        if len(token) <= 2 or token in STOPWORDS:
            continue
        result.add(token)
        for prefix in ("وال", "بال", "كال", "لل", "ال", "و", "ب", "ل", "ا"):
            if token.startswith(prefix) and len(token) - len(prefix) > 2:
                result.add(token[len(prefix):])
                break
    return result


def retrieve(query: str, jurisdiction: str | None, limit: int = 3, corpus: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    corpus = corpus or load_corpus()
    scope = (jurisdiction or "").strip().upper()
    if scope not in {"AU", "US"}:
        return []
    query_terms = _tokens(query)
    scored = []
    for source in corpus["sources"]:
        if source["geography"].upper() != scope:
            continue
        terms = _tokens(source["text"] + " " + source["title"])
        overlap = len(query_terms & terms)
        if overlap:
            scored.append((overlap, source))
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [source for _, source in scored[:limit]]


def answer_question(query: str, jurisdiction: str | None) -> dict[str, Any]:
    sources = retrieve(query, jurisdiction)
    if not sources:
        response = "ما عندي مصدر موثوق وكافٍ أستند إليه لهذا السؤال ضمن الاختصاص المحدد. إذا كان سؤالك عن إجراء محلي، احكيلي أي بلد أو اختصاص تقصد. ما رح أخمّن قانونًا أو رقم تواصل."
        return {"text": response, "citations": [], "corpus_version": load_corpus()["corpus_version"]}
    content = " ".join(s["text"] for s in sources)
    response = content + "\n\nالمعلومات أعلاه إرشاد عام من المصادر المذكورة، وليست حكمًا على حالتك ولا تأكيدًا مستقلًا للواقعة."
    citations = [{"id": s["id"], "title": s["title"], "publisher": s["publisher"], "url": s["url"],
                  "published_or_updated": s["published_or_updated"], "ingested_at": s["ingested_at"],
                  "geography": s["geography"], "content_version": s["content_version"]} for s in sources]
    return {"text": response, "citations": citations, "corpus_version": load_corpus()["corpus_version"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate the local RAG source index")
    parser.add_argument("--reindex", action="store_true", help="Validate and reload source metadata from data/knowledge.json")
    args = parser.parse_args()
    if args.reindex:
        corpus = load_corpus()
        print(f"Validated corpus {corpus['corpus_version']} with {len(corpus['sources'])} sources.")
