"""Synthetic fixtures for the unified local interactive-turn contract."""
from __future__ import annotations

from typing import Any


def interactive_result(facts: list[dict[str, Any]] | None = None, *,
                       goals: list[str] | None = None,
                       user_intent: str | None = None,
                       reply_text: str = "فهمت ما ذكرت. ما التفصيل الذي تريد إضافته؟",
                       conversation_move: str = "ask_follow_up") -> dict[str, Any]:
    goals = list(goals or [])
    return {
        "facts": list(facts or []),
        "user_intent": user_intent,
        "conversation_move": conversation_move,
        "reply_text": reply_text,
        "message_plan": {"goals": goals, "unsupported_goals": []},
        "diagnostic": {"accepted": True, "filtered_facts": [],
                       "unsupported_message_goals": []},
    }
