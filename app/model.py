from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


def model_configured() -> bool:
    return os.getenv("MODEL_PROVIDER", "rules").strip().lower() == "openai-compatible" and bool(os.getenv("MODEL_API_KEY"))


def generate_grounded_answer(question: str, sources: list[dict[str, Any]]) -> str | None:
    """Return a draft only when explicitly configured; callers must obtain per-question consent.

    The caller passes no session transcript or incident facts. Network and provider errors return None.
    """
    if not model_configured() or not sources:
        return None
    base = os.getenv("MODEL_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    source_text = "\n\n".join(f"[{i + 1}] {s.get('title')} — {s.get('publisher')} ({s.get('geography')}): {s.get('text')}\nURL: {s.get('url')}" for i, s in enumerate(sources))
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "أجب بالعربية عن سؤال المستخدم اعتمادًا فقط على مقتطفات المصادر الموثوقة المرفقة. لا تقدم نصيحة قانونية أو رقم تواصل أو إجراء محلي خارج نطاق المصدر. لا تدّعِ التحقق من واقعة المستخدم. إذا لم تدعم المصادر جوابًا كافيًا، صرّح بذلك. لا تغيّر أولوية الفرز ولا تستنتج معلومات عن الحالة. أضف إحالات [1] أو [2] فقط للمصادر التي تدعم الجملة."},
            {"role": "user", "content": f"السؤال:\n{question}\n\nالمصادر المتاحة:\n{source_text}"},
        ],
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("MODEL_TIMEOUT_SECONDS", "15"))) as response:
            payload = json.loads(response.read().decode("utf-8"))
        answer = payload["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            return None
        answer = answer.strip()[:6000]
        def keep_supported(match: re.Match[str]) -> str:
            return match.group(0) if 1 <= int(match.group(1)) <= len(sources) else ""
        return re.sub(r"\[(\d+)\]", keep_supported, answer)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError, OSError):
        return None
