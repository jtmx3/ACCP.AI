"""Smart stopping policy, separate from field extraction and risk scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.nlu import analyze_message
from app.triage import classify_track, core_fields_for_track

TURN_BUDGETS = {
    "defamation_or_abuse": 4,
    "impersonation_or_harassment": 6,
    "extortion_fraud_takeover": 8,
    "exploratory_intake": 4,
    "direct_danger": 0,
}
TERMINAL = {"reported", "denied", "unknown", "not_remembered", "declined", "unresolved", "conflict"}


@dataclass(frozen=True)
class StoppingDecision:
    should_stop: bool
    stop_reason: str
    confidence_note: str
    missing_optional_fields: list[str]
    recommended_next_focus: str | None
    incident_track: str
    turn_count: int
    turn_budget: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _answered(state: dict[str, Any], key: str) -> bool:
    fact = state.get("facts", {}).get(key, {})
    if key == "narrative":
        return fact.get("status") in TERMINAL and bool(fact.get("value") or fact.get("status") != "reported")
    return fact.get("status") in TERMINAL


def _question_count(state: dict[str, Any]) -> int:
    # Count distinct interview focuses rather than retries of an unanswered slot.
    # The opening narrative/details prompts establish the account and are outside
    # the targeted follow-up budget; retries remain visible in history and signals.
    excluded = {None, "narrative", "details", "unclassified_details", "incident_type_review"}
    return len({item.get("slot") for item in state.get("question_history", [])
                if isinstance(item, dict) and item.get("slot") not in excluded})


def _recent_unknown_streak(state: dict[str, Any]) -> int:
    signals = state.get("turn_signals", [])
    streak = 0
    slots: list[str] = []
    for item in reversed(signals):
        if isinstance(item, dict) and item.get("substantive_fact"):
            break
        if not isinstance(item, dict) or not (item.get("unknown") or item.get("not_remembered")):
            break
        streak += 1
        slot = item.get("slot")
        if isinstance(slot, str):
            slots.append(slot)
    # The stop policy requires two different question focuses, not repeated
    # unknown answers to a single pending question.
    return streak if len(set(slots)) >= 2 else min(streak, 1)


def evaluate_stopping_decision(state: dict[str, Any], turn_signals: dict[str, Any] | None = None) -> StoppingDecision:
    """Return a deterministic stopping result; high priority is never a prerequisite."""
    track = classify_track(state)
    budget = TURN_BUDGETS.get(track, 6)
    count = _question_count(state)
    signals = turn_signals or {}
    required = core_fields_for_track(state, track)
    missing_core = [key for key in required if not _answered(state, key)]
    all_relevant = state.get("slot_states", {})
    missing_optional = [key for key in all_relevant if key not in required and all_relevant.get(key) in {"unasked", "asked"}]
    if state.get("stage") == "urgent_stop" or track == "direct_danger":
        return StoppingDecision(True, "reactive_direct_danger", "توقف سلامة بناءً على تصريح صريح؛ لا تُستنتج حالة طوارئ من القلق الرقمي وحده.", missing_optional, None, track, count, budget)
    if signals.get("wrap_up"):
        next_focus = missing_core[0] if missing_core else None
        return StoppingDecision(True, "user_wrap_up", "طلب المستخدم إنهاء السرد أو مراجعة الملخص؛ النواقص ستبقى ظاهرة دون استنتاج.", missing_optional + missing_core, next_focus, track, count, budget)
    if signals.get("protest") or signals.get("fatigue"):
        return StoppingDecision(True, "user_fatigue_or_repeat_protest", "تجنب تكرار السؤال واحترم إرهاق المستخدم؛ يمكن إضافة تفاصيل لاحقًا.", missing_optional + missing_core, missing_core[0] if missing_core else None, track, count, budget)
    streak = _recent_unknown_streak(state)
    if streak >= 2:
        return StoppingDecision(True, "diminishing_returns", "تكررت إفادة عدم المعرفة أو عدم التذكر؛ توقف دون الضغط أو تحويل المجهول إلى حقيقة.", missing_optional + missing_core, missing_core[0] if missing_core else None, track, count, budget)
    if missing_core and count >= budget:
        return StoppingDecision(True, "adaptive_turn_budget", "بلغ الحوار سقف الأسئلة الملائم للمسار؛ تُعرض الفجوات بدل استمرار الاستجواب.", missing_optional + missing_core, missing_core[0], track, count, budget)
    if not missing_core:
        return StoppingDecision(True, "track_core_complete", "اكتملت العناصر الجوهرية لهذا المسار بحسب الإفادات المسجلة؛ يمكن مراجعة الملخص دون اشتراط خطورة مرتفعة.", missing_optional, None, track, count, budget)
    return StoppingDecision(False, "continue_focused_interview", "ما زالت هناك معلومة جوهرية واحدة على الأقل قد تغيّر توثيق المسار.", missing_optional, missing_core[0], track, count, budget)
