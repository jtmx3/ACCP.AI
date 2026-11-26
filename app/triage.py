"""Conservative multi-track triage and track-specific evidence requirements."""
from __future__ import annotations

from typing import Any

from app.nlu import analyze_message, normalize_arabic

TRACK_LABELS = {
    "direct_danger": "خطر جسدي مباشر مصرح به",
    "defamation_or_abuse": "إساءة لفظية أو ذم/قدح رقمي",
    "impersonation_or_harassment": "انتحال أو مضايقة تحتاج توضيحًا",
    "extortion_fraud_takeover": "ابتزاز أو احتيال أو دخول غير مأذون",
    "exploratory_intake": "استيضاح أولي",
}


def _fact(state: dict[str, Any], key: str) -> dict[str, Any]:
    value = state.get("facts", {}).get(key, {})
    return value if isinstance(value, dict) else {}


def _reported_true(state: dict[str, Any], key: str) -> bool:
    fact = _fact(state, key)
    return fact.get("status") == "reported" and fact.get("value") is True


def classify_track(state: dict[str, Any]) -> str:
    """Choose a review track from supported facts and the reporter's own narrative.

    A lookalike or a fan account is intentionally not sufficient for impersonation.
    The result is a triage route, not a legal finding.
    """
    facts = state.get("facts", {})
    narrative = " ".join(str(_fact(state, key).get("value") or _fact(state, key).get("quote") or "")
                         for key in ("narrative", "details", "unclassified_details", "additional_details"))
    cues = analyze_message(narrative)
    folded = normalize_arabic(narrative)
    if _reported_true(state, "direct_danger"):
        return "direct_danger"
    financial = any(_reported_true(state, key) for key in (
        "active_unauthorized_funds", "fraud_description", "account_access"))
    financial = financial or any(_fact(state, key).get("status") == "reported" for key in (
        "requested_amount", "paid_amount", "payment_method"))
    intimate = _reported_true(state, "sensitive_photos") or _reported_true(state, "ai_fabricated")
    publication = _reported_true(state, "publication_threat") or cues["publication_threat"]
    explicit_blackmail = cues["extortion"] or (publication and (_reported_true(state, "threat") or intimate))
    account_takeover = _reported_true(state, "account_access") or _reported_true(state, "fraud_description") or cues["account_takeover"] or cues["financial_deception"]
    if financial or explicit_blackmail or account_takeover or (publication and _reported_true(state, "threat")):
        return "extortion_fraud_takeover"
    impersonation = _reported_true(state, "impersonation") or cues["explicit_identity_claim"]
    if cues["fan_account"] or cues["lookalike_only"]:
        impersonation = False
    if impersonation or _reported_true(state, "harassment") or cues["harassment_cue"]:
        return "impersonation_or_harassment"
    if cues["verbal_abuse"] and cues["digital_context"] and not cues.get("evidence_only_statement"):
        return "defamation_or_abuse"
    # A generic threat belongs to a focused exploratory/harassment route, not a high score by itself.
    if _reported_true(state, "threat"):
        return "impersonation_or_harassment"
    if any(key in facts for key in ("narrative", "details", "unclassified_details")):
        return "exploratory_intake"
    return "exploratory_intake"


def core_fields_for_track(state: dict[str, Any], track: str) -> list[str]:
    """Minimum track-specific elements; optional details never block a useful review."""
    facts = state.get("facts", {})
    narrative = " ".join(str(_fact(state, key).get("value") or "") for key in
                         ("narrative", "details", "additional_details"))
    cues = analyze_message(narrative)
    def present(key: str) -> bool:
        fact = _fact(state, key)
        return fact.get("status") in {"reported", "denied", "unknown", "not_remembered", "declined", "unresolved", "conflict"}
    if track == "defamation_or_abuse":
        return ["narrative", "details", "platform", "identity", "evidence"]
    if track == "impersonation_or_harassment":
        # Either initial contact or discovery/creation time can satisfy the time anchor.
        fields = ["narrative", "details", "platform", "identity", "evidence"]
        if not any(present(key) for key in ("initial_time", "discovery_time", "account_creation_time")):
            fields.append("initial_time")
        return fields
    if track == "extortion_fraud_takeover":
        fields = ["narrative", "details", "identity", "evidence"]
        if not any(present(key) for key in ("initial_time", "latest_time")):
            fields.append("initial_time")
        if _reported_true(state, "threat") or _reported_true(state, "harassment"):
            if not present("repeated") and not present("ongoing_contact"):
                fields.append("repeated")
        if _reported_true(state, "publication_threat") or cues.get("publication_threat"):
            if not present("publication_threat"):
                fields.append("publication_threat")
            if not present("published"):
                fields.append("published")
        # Ask only about financial dimensions introduced by the user's account.
        requested = present("requested_amount") or present("deadline") or any(
            token in normalize_arabic(" ".join(str(_fact(state, key).get("value") or "")
                                                for key in ("narrative", "details")))
            for token in ("طلب مبلغ", "ادفع", "حول", "مهله", "موعد الدفع"))
        if requested:
            fields.extend(key for key in ("requested_amount", "paid_amount", "deadline") if not present(key))
        paid_fact = _fact(state, "paid_amount")
        if paid_fact.get("status") == "reported" and paid_fact.get("value") and not present("payment_method"):
            fields.append("payment_method")
        for event in state.get("money_events", []):
            if isinstance(event, dict) and event.get("relation") in {"unresolved", "unresolved_after_clarification"}:
                relation = "requested_amount_relation" if event.get("kind") == "requested" else "paid_amount_relation"
                if not present(relation):
                    fields.append(relation)
        return list(dict.fromkeys(fields))
    if track == "exploratory_intake" and "narrative" in facts:
        return ["narrative", "unclassified_details"]
    return ["narrative"]


def route_assessment(state: dict[str, Any]) -> dict[str, Any]:
    track = classify_track(state)
    cues = analyze_message(" ".join(str(_fact(state, key).get("value") or "") for key in ("narrative", "details")))
    if track == "defamation_or_abuse":
        text = " ".join(str(_fact(state, key).get("value") or "") for key in ("narrative", "details"))
        public = any(term in normalize_arabic(text) for term in ("تعليق", "منشور", "بوست", "ستوري", "عام"))
        repeated = _reported_true(state, "repeated")
        score = 45 if public else 35
        if repeated:
            score = 55
        return {"priority_score": score, "priority_label": "عادية/تحتاج مراجعة",
                "priority_rule": "R45_DIGITAL_VERBAL_ABUSE" if not repeated else "R55_REPEATED_DIGITAL_ABUSE",
                "supporting_fields": ["narrative", "details"] + (["platform"] if _fact(state, "platform") else []),
                "route": "مراجعة بشرية أولية لإساءة لفظية رقمية",
                "reasons": ["إساءة لفظية رقمية بحسب إفادة المستخدم؛ التصنيف تجريبي وليس حكمًا قانونيًا"],
                "risk_indicators": []}
    return {"incident_track": track, "track_label": TRACK_LABELS[track],
            "core_fields": core_fields_for_track(state, track), "signals": cues}
