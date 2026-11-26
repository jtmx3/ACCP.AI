from __future__ import annotations

import copy
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core import (
    QUESTIONS, LABELS, _add_message, _assessment, _explain_why, _why_target_slot, explain_assessment,
    _incident_features,
    _finalize_changed_facts, _next_slot, _record, _unresolved_money_relation,
    _user_intent, classify_current_danger, confirmation_is_current, handle_turn, now_iso, normalize, render_report,
    report_data, resume_state, screen_current_danger, summary_digest, is_clear_user_question, question_plan,
    compose_question,
    last_model_merge_trace,
)
from app.rag import answer_question, available_jurisdictions, load_corpus
from app.model import (generate_grounded_answer, generate_interview_extraction, model_configured,
                       generate_adaptive_question,
                       provider_config, probe_provider, test_provider_generation, reset_call_trace,
                       last_call_trace, call_trace_history, last_extraction_diagnostic,
                       snapshot_telemetry, restore_telemetry)
from app.conversation import (build_bounded_context, extract_fact_text, extract_guidance_question,
                              has_actionable_fact_goal, interpret_message)
from app.store import Forbidden, NotFound, StaleRevision, Store

app = FastAPI(title="ACCP.AI", version="0.2.0", description="Local-first Arabic incident intake and review prototype")
store = Store()
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class TurnRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=5000)
    kind: str = "answer"
    allow_local_model: bool = False
    allow_context_extraction: bool = False
    allow_adaptive_question: bool = False

    def model_post_init(self, __context):
        if self.kind not in {"answer", "question"}:
            raise ValueError("نوع الرسالة غير مدعوم.")


class RevisionRequest(BaseModel):
    expected_revision: int = Field(ge=0)


class JurisdictionRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    jurisdiction: str = ""


class AdaptiveQuestionSettingsRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    enabled: bool = False


class LocalModelSettingsRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    enabled: bool = False


class TemporalConflictResolutionRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    field: str
    selected_value: str = Field(min_length=1, max_length=500)
    resolution_reason: str = Field(min_length=4, max_length=500)


def _state_for(session_id: str, token: str):
    try:
        return store.get(session_id, token)
    except NotFound:
        raise HTTPException(404, "الحالة غير موجودة.")
    except Forbidden:
        raise HTTPException(403, "بيانات الدخول إلى الحالة غير صحيحة.")


def _preflight(session_id: str, token: str, expected_revision: int):
    state = _state_for(session_id, token)
    if state["revision"] != expected_revision:
        raise HTTPException(409, {"message": "تغيّرت الحالة بعد تحميلها. حدّث العرض قبل إعادة المحاولة.", "current_revision": state["revision"]})
    return state


def _saved_platform_for_guidance(state: dict) -> tuple[str | None, bool]:
    """Return only a resolved platform fact; never infer one from unrelated history."""
    fact = state.get("facts", {}).get("platform")
    if not isinstance(fact, dict):
        return None, False
    if fact.get("status") == "conflict":
        return None, True
    if fact.get("status") == "reported" and isinstance(fact.get("value"), str):
        return fact["value"], False
    return None, False


def _sanitized_adaptive_question_context(state: dict, slot: str) -> dict:
    """No raw messages, quotes, session identifiers, names, amounts, or contact IDs."""
    features = _incident_features(state)
    plan = question_plan(state, slot) or {}
    feature_allowlist = {
        "threat", "impersonation", "publication_threat", "published_status", "sensitive_photos",
        "ai_fabricated", "age_or_minor_mentioned", "local_payment_method_named", "deadline_known",
        "deadline_soon", "financial", "platform", "discovery_known", "creation_known", "contact_available",
    }
    question_history = state.get("question_history", [])
    recent_slots = [item.get("slot") for item in question_history[-4:] if isinstance(item, dict)]
    recent = [compose_question(state, previous) for previous in recent_slots if previous in QUESTIONS]
    return {"field_label": plan.get("field_label"),
            "question_goal": plan.get("reason"),
            "missing_condition": plan.get("condition"),
            "language_style": "لهجة أردنية محكية بسيطة ومحترمة",
            "base_question": compose_question(state, slot),
            "recent_bot_questions": list(dict.fromkeys(recent[-3:])),
            "features": {key: features.get(key) for key in sorted(feature_allowlist)}}


_GUIDANCE_PLATFORM_PREFIX = re.compile(
    r"^\s*(?:واتساب|whatsapp|إنستغرام|انستغرام|انستجرام|instagram|فيسبوك|facebook|سناب\s*شات|سناب|تيك\s*توك|tiktok)"
    r"\s*(?:[,،;؛.!?؟]\s*|$)", re.IGNORECASE,
)


def _fact_text_after_guidance_platform(text: str) -> str:
    """Remove only a leading app name used to answer the pending guidance scope question."""
    match = _GUIDANCE_PLATFORM_PREFIX.match(text)
    if not match:
        return text.strip()
    return text[match.end():].strip(" \t\r\n,،;؛.!?؟")


def _cancels_pending_guidance(text: str) -> bool:
    normalized = normalize(text)
    declines = ("ما بدي نصيحة", "مش بدي نصيحة", "ما بدي نصيحه", "مش بدي نصيحه",
                "ما بدي ارشاد", "مش بدي ارشاد", "مش حابب نصيحة", "مش حابة نصيحة")
    resumes = ("نكمل الإفادة", "نكمل الافادة", "نكمل الافاده", "نكمل المقابلة", "نكمل المقابله",
               "نرجع للمقابلة", "نرجع للمقابله", "تابع المقابلة", "تابع المقابله")
    return any(marker in normalized for marker in declines) or any(marker in normalized for marker in resumes)


def _mutate(session_id: str, token: str, rev: int, event: str, fn):
    def apply_and_reconcile(state):
        fn(state)
        if state.get("confirmed") and not confirmation_is_current(state):
            old_digest = state.get("confirmed_digest")
            history = state.setdefault("confirmation_history", [])
            if old_digest and not any(item.get("digest") == old_digest for item in history):
                confirmed_message = next((message for message in reversed(state.get("messages", [])) if message.get("kind") == "confirmation"), {})
                history.append({"summary_version": state["summary_version"], "digest": old_digest,
                                "confirmed_at": confirmed_message.get("created_at"), "legacy": True})
            state["confirmed"] = False
            state["confirmed_digest"] = None
            state["summary_version"] += 1
            state["assessment"] = _assessment(state)

    try:
        return store.mutate(session_id, token, rev, event, apply_and_reconcile)
    except NotFound:
        raise HTTPException(404, "الحالة غير موجودة.")
    except Forbidden:
        raise HTTPException(403, "بيانات الدخول إلى الحالة غير صحيحة.")
    except StaleRevision as exc:
        raise HTTPException(409, {"message": "تغيّرت الحالة بعد تحميلها. حدّث العرض قبل إعادة المحاولة.", "current_revision": exc.current_revision})
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/health")
def health():
    config = provider_config()
    return {"status": "ok", "storage": "sqlite", "provider": config,
            "runtime_mode": config["mode"], "local_model_configured": config["configured"], "version": app.version}


@app.get("/api/provider/status")
def provider_status():
    config = provider_config()
    return {"config": config, "probe": probe_provider()}


@app.post("/api/provider/test")
def provider_generation_test():
    config = provider_config()
    if not config["configured"]:
        raise HTTPException(409, "لم يكتمل إعداد مزود النموذج.")
    probe = probe_provider()
    if not probe["service_reachable"] or not probe["model_available"]:
        return {"preflight": probe, "success": False, "attempted": False,
                "failure_code": probe["failure_code"], "provider": config["provider"], "model": config["model"]}
    return {"preflight": probe, **test_provider_generation()}


@app.get("/api/metadata")
def metadata():
    return {"jurisdictions": available_jurisdictions(), "corpus_version": load_corpus()["corpus_version"]}


@app.post("/api/sessions")
def create_session():
    defaults = None
    config = provider_config()
    if config.get("configured"):
        probe = probe_provider(timeout=1.0)
        enabled = bool(probe.get("service_reachable") and probe.get("model_available"))
        defaults = {"local_model_enabled": enabled, "model_settings_version": 1,
                    "adaptive_questions_enabled": bool(enabled and config.get("adaptive_question_generation_enabled"))}
    state, token = store.create(defaults)
    state["jurisdiction"] = None
    return {"session": state, "token": token, "token_notice": "هذا الرمز خاص بالحالة. لا تشاركه؛ لا يمكن استعادته من الخادم."}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, x_case_token: str = Header(default="")):
    state = _state_for(session_id, x_case_token)
    state["assessment"] = _assessment(state)
    report = report_data(state)
    state["confirmed"] = report["confirmed"]
    state["confirmation_history"] = report["confirmation_history"]
    if not state["confirmed"]:
        state["confirmed_digest"] = None
    return {"session": state, "report": report}


@app.put("/api/sessions/{session_id}/adaptive-question-settings")
def set_adaptive_question_settings(session_id: str, body: AdaptiveQuestionSettingsRequest,
                                   x_case_token: str = Header(default="")):
    if body.enabled and not provider_config().get("adaptive_question_generation_enabled"):
        raise HTTPException(409, "توليد الأسئلة التكيفية معطل في إعداد التطبيق.")

    def mutate(state):
        state["adaptive_questions_enabled"] = bool(body.enabled)

    state = _mutate(session_id, x_case_token, body.expected_revision, "adaptive_question_setting_changed", mutate)
    return {"session": state, "adaptive_questions_enabled": state["adaptive_questions_enabled"]}


@app.put("/api/sessions/{session_id}/local-model-settings")
def set_local_model_settings(session_id: str, body: LocalModelSettingsRequest,
                             x_case_token: str = Header(default="")):
    config = provider_config()
    if body.enabled and not config.get("configured"):
        raise HTTPException(409, "لا يمكن تفعيل النموذج إلا مع Ollama محلي مضبوط على loopback.")

    def mutate(state):
        state["local_model_enabled"] = bool(body.enabled)
        state["model_settings_version"] = 1
        state["adaptive_questions_enabled"] = bool(
            body.enabled and config.get("adaptive_question_generation_enabled"))

    state = _mutate(session_id, x_case_token, body.expected_revision, "local_model_setting_changed", mutate)
    return {"session": state, "local_model_enabled": state["local_model_enabled"],
            "adaptive_questions_enabled": state["adaptive_questions_enabled"],
            "provider": config["provider"], "model": config["model"]}


def _append_answer_before_followup(state: dict, text: str, citations: list[dict] | None = None,
                                  local_model_used: bool = False) -> None:
    if state.get("stage") == "urgent_stop":
        return
    followup = state.get("messages", [])[-1] if state.get("messages") else None
    if not followup or followup.get("role") != "assistant" or followup.get("kind") not in {
        "question", "clarification", "review"
    }:
        followup = None
    if followup:
        state["messages"].pop()
    response = _add_message(state, "assistant", text, "answer")
    response["citations"] = citations or []
    response["local_model_used"] = local_model_used
    if followup:
        if followup.get("kind") in {"question", "clarification"}:
            slot = (followup.get("question_plan") or {}).get("slot") or state.get("pending_slot")
            if slot in QUESTIONS:
                restored = _add_message(state, "assistant", followup.get("text", QUESTIONS[slot]), followup["kind"])
                restored["question_plan"] = question_plan(state, slot)
        else:
            _add_message(state, "assistant", followup.get("text", "راجع الملخص الحالي."), "review")


def _append_guidance_before_followup(state: dict, local_answer: dict, grounded_local_answer: dict | None) -> None:
    reply = grounded_local_answer["text"] if grounded_local_answer else local_answer["text"]
    reply = "سجلت الجزء الذي ذكرته في مسودة الإفادة.\n\n" + reply
    if grounded_local_answer:
        reply += "\n\nصيغ الرد من مقتطفات حرفية موثقة، ولم يُرسل سجل المقابلة. راجع المصدر ولا تعتبره نصيحة قانونية."
    used_ids = set(grounded_local_answer["source_ids"]) if grounded_local_answer else set()
    citations = [citation for citation in local_answer["citations"]
                 if not used_ids or citation["id"] in used_ids]
    _append_answer_before_followup(state, reply, citations, bool(grounded_local_answer))


@app.post("/api/sessions/{session_id}/messages")
def post_message(session_id: str, body: TurnRequest, x_case_token: str = Header(default="")):
    request_started = time.perf_counter()
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "اكتب رسالة قبل الإرسال.")

    # Authenticate and check revision before any optional network operation.
    preflight_started = time.perf_counter()
    snapshot = _preflight(session_id, x_case_token, body.expected_revision)
    request_preflight_ms = round((time.perf_counter() - preflight_started) * 1000, 3)
    session_setting_present = "local_model_enabled" in snapshot
    model_enabled_for_turn = (bool(snapshot.get("local_model_enabled")) if session_setting_present
                              else bool(body.allow_local_model))
    context_enabled_for_turn = model_enabled_for_turn and (session_setting_present or body.allow_context_extraction)
    pending = snapshot.get("pending_slot")
    pending_guidance = snapshot.get("pending_guidance")
    resolving_pending_guidance = bool(isinstance(pending_guidance, dict)
                                      and pending_guidance.get("status") == "awaiting_platform")
    urgent = classify_current_danger(text, pending) is True
    pending_guidance_cancelled = False
    mixed_pending_guidance = False
    guidance_fact_text = ""
    understanding_started = time.perf_counter()
    intent = _user_intent(text)
    if resolving_pending_guidance and not urgent and intent == "pause":
        resolving_pending_guidance = False
        message_interpretation = interpret_message(text, pending)
    elif resolving_pending_guidance and not urgent and _cancels_pending_guidance(text):
        pending_guidance_cancelled = True
        resolving_pending_guidance = False
        message_interpretation = interpret_message("", pending)
    elif resolving_pending_guidance and not urgent:
        guidance_fact_text = _fact_text_after_guidance_platform(text)
        candidate_interpretation = interpret_message(guidance_fact_text, pending)
        explicit_fact_goals = {"add_fact", "correct_prior_fact", "unknown", "not_remembered",
                               "decline_disclosure", "confirm_or_reference_prior_fact"}
        if guidance_fact_text and explicit_fact_goals.intersection(candidate_interpretation.get("goals", [])):
            resolving_pending_guidance = False
            mixed_pending_guidance = True
            message_interpretation = candidate_interpretation
            message_interpretation["fact_text"] = guidance_fact_text
        else:
            message_interpretation = {"goals": ["ask_guidance"],
                                      "guidance_question": pending_guidance.get("query"), "fact_text": ""}
    elif resolving_pending_guidance:
        message_interpretation = {"goals": ["ask_guidance"],
                                  "guidance_question": pending_guidance.get("query"), "fact_text": ""}
    else:
        message_interpretation = interpret_message(text, pending)
    if intent is None and "ask_why" in message_interpretation["goals"]:
        intent = "why"
    if intent is None and "ask_assessment_reason" in message_interpretation["goals"]:
        intent = "assessment_why"
    local_qa = None
    grounded_local_answer = None
    model_facts = None
    model_message_plan: dict[str, Any] = {"goals": [], "unsupported_goals": [], "guidance_question": None}
    message_plan_applied = False
    extraction_context = None
    local_extraction_sent = False
    local_question_sent = False
    reset_call_trace()

    # Extract role goals and current-turn facts in one call. Plan fields come only
    # from the model validator, and all routing/safety outcomes remain application-owned.
    fact_candidate = message_interpretation.get("fact_text", text)
    initial_actionable_fact = (False if resolving_pending_guidance or pending_guidance_cancelled
                               else has_actionable_fact_goal(fact_candidate, pending))
    initial_auto_question = (not resolving_pending_guidance and body.kind == "answer"
                             and is_clear_user_question(text) and not initial_actionable_fact)
    extraction_text = guidance_fact_text if mixed_pending_guidance else text
    if (body.kind == "answer" and not initial_auto_question and model_enabled_for_turn
            and model_configured() and not urgent and intent is None and not resolving_pending_guidance):
        local_extraction_sent = True
        if context_enabled_for_turn:
            extraction_context = build_bounded_context(snapshot, extraction_text, pending)
            model_facts = generate_interview_extraction(extraction_text, pending, extraction_context,
                                                        message_interpretation)
        else:
            model_facts = generate_interview_extraction(extraction_text, pending)
        candidate_plan = getattr(model_facts, "message_plan", None)
        if isinstance(candidate_plan, dict):
            plan_goals = candidate_plan.get("goals", [])
            if isinstance(plan_goals, list) and all(isinstance(goal, str) for goal in plan_goals):
                unsupported_goals = candidate_plan.get("unsupported_goals", [])
                if not isinstance(unsupported_goals, list) or not all(isinstance(goal, str) for goal in unsupported_goals):
                    unsupported_goals = []
                model_message_plan = {"goals": list(dict.fromkeys(plan_goals)),
                                      "unsupported_goals": list(dict.fromkeys(unsupported_goals)),
                                      "guidance_question": candidate_plan.get("guidance_question")}
                message_plan_applied = bool(model_message_plan["goals"])
                message_interpretation = {**message_interpretation,
                    "goals": list(dict.fromkeys(message_interpretation.get("goals", []) + plan_goals)),
                    "guidance_question": model_message_plan.get("guidance_question") or message_interpretation.get("guidance_question"),
                }
                message_interpretation["fact_text"] = (guidance_fact_text if mixed_pending_guidance else
                    extract_fact_text(text, message_interpretation.get("guidance_question")))
        if intent is None and "ask_why" in model_message_plan["goals"]:
            intent = "why"
        if intent is None and "ask_assessment_reason" in model_message_plan["goals"]:
            intent = "assessment_why"

    actionable_plan_goals = {"answer_pending_question", "correct_prior_fact", "add_fact", "unknown",
                             "not_remembered", "decline_disclosure", "confirm_or_reference_prior_fact"}
    has_actionable_fact = (initial_actionable_fact or bool(
        actionable_plan_goals.intersection(message_interpretation.get("goals", []))))
    question_goals = {"ask_question", "ask_guidance", "ask_why", "ask_assessment_reason"}
    model_question_goal = bool(set(model_message_plan["goals"]) & question_goals)
    if question_goals.intersection(message_interpretation.get("goals", [])) and not resolving_pending_guidance:
        # A pending slot alone does not turn a pure question into factual interview input.
        fact_text = message_interpretation.get("fact_text")
        has_actionable_fact = bool(model_facts or (isinstance(fact_text, str) and fact_text.strip()))
    mixed_why = intent in {"why", "assessment_why"} and has_actionable_fact
    mixed_guidance = (("ask_guidance" in message_interpretation["goals"] and has_actionable_fact)
                      or mixed_pending_guidance)
    mixed_fact_question = mixed_why or mixed_guidance
    auto_question = (body.kind == "answer" and (is_clear_user_question(text) or model_question_goal)
                     and not mixed_fact_question and not resolving_pending_guidance)
    question_mode = (resolving_pending_guidance or body.kind == "question" or auto_question) and not mixed_fact_question
    explanation_intent = intent
    if mixed_fact_question:
        intent = None
    if body.kind == "question" and intent not in {"why", "assessment_why"}:
        intent = None

    pending_guidance_turn = resolving_pending_guidance or mixed_pending_guidance
    guidance_question = (pending_guidance.get("query") if pending_guidance_turn else
                         message_interpretation.get("guidance_question")
                         or extract_guidance_question(text) or text)
    local_guidance_context = ""
    guidance_platform, guidance_platform_ambiguous = _saved_platform_for_guidance(snapshot)
    if (question_mode or mixed_guidance) and not resolving_pending_guidance:
        local_context = build_bounded_context(snapshot, text, pending)
        local_guidance_context = " ".join(
            f"{fact.get('field', '')} {fact.get('value') or ''} {fact.get('quote') or ''}"
            for fact in local_context.get("related_facts", [])
        )
    retrieval_ms = 0.0
    understanding_ms = round((time.perf_counter() - understanding_started) * 1000, 3)
    if (question_mode or mixed_guidance) and not urgent and intent not in {"why", "assessment_why"}:
        retrieval_started = time.perf_counter()
        local_qa = answer_question(guidance_question, snapshot.get("jurisdiction"),
                                   context_text="" if resolving_pending_guidance else local_guidance_context + " " + text,
                                   allow_nonlocal_general=True,
                                   platform_context=guidance_platform,
                                   platform_context_ambiguous=guidance_platform_ambiguous,
                                   platform_query=text,
                                   scope_query=(pending_guidance.get("scope_query", pending_guidance.get("query", text))
                                                if resolving_pending_guidance else text),
                                   topic_query=(pending_guidance.get("topic_query", pending_guidance.get("query", text))
                                                if pending_guidance_turn else text))
        retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 3)
    answer_generation_ms = 0.0
    if (question_mode or mixed_guidance) and not urgent and intent not in {"why", "assessment_why"}:
        if model_enabled_for_turn and model_configured() and local_qa and local_qa.get("citations"):
            local_question_sent = True
            corpus = load_corpus()
            source_map = {item["id"]: item for item in corpus["sources"]}
            selected_sources = []
            for citation in local_qa["citations"]:
                source = source_map.get(citation["id"])
                if source:
                    selected_sources.append({**source, "excerpt": citation.get("excerpt", "")})
            answer_started = time.perf_counter()
            grounded_local_answer = generate_grounded_answer(guidance_question, selected_sources)
            answer_generation_ms = round((time.perf_counter() - answer_started) * 1000, 3)

    def mutate(state):
        if not session_setting_present:
            state["adaptive_questions_enabled"] = bool(
                body.allow_adaptive_question and provider_config().get("adaptive_question_generation_enabled", False))
        if urgent:
            user_msg = _add_message(state, "user", text, "answer")
            state["updated_at"] = user_msg["created_at"]
            state["pending_guidance"] = None
            screen_current_danger(state, text, user_msg, pending)
            state["assessment"] = _assessment(state)
            return
        if pending_guidance_cancelled:
            user_msg = _add_message(state, "user", text, "guidance_clarification")
            state["updated_at"] = user_msg["created_at"]
            state["pending_guidance"] = None
            if state.get("stage") == "interview" and pending in QUESTIONS:
                state["pending_slot"] = pending
                resumed = _add_message(
                    state, "assistant", "حسنًا، ألغيت طلب الإرشاد. نكمل من السؤال الحالي: " + QUESTIONS[pending], "question")
                resumed["question_plan"] = question_plan(state, pending)
            else:
                _add_message(state, "assistant", "حسنًا، ألغيت طلب الإرشاد.", "status")
            state["assessment"] = _assessment(state)
            return
        if resolving_pending_guidance:
            _add_message(state, "user", text, "guidance_clarification")
            if local_qa and local_qa.get("clarification_required") == "platform":
                pending_request = state.get("pending_guidance") or {}
                pending_request["status"] = "awaiting_platform"
                pending_request["clarification_text"] = local_qa["text"]
                state["pending_guidance"] = pending_request
                clarification = _add_message(state, "assistant", local_qa["text"], "guidance_clarification")
                clarification["citations"] = []
                state["assessment"] = _assessment(state)
                return
            state["pending_guidance"] = None
            answer_text = grounded_local_answer["text"] if grounded_local_answer else (local_qa or {}).get("text", "")
            if grounded_local_answer:
                answer_text += "\n\nصيغ الرد من مقتطفات حرفية موثقة، ولم يُرسل سجل المقابلة. راجع المصدر ولا تعتبره نصيحة قانونية."
            answer = _add_message(state, "assistant", answer_text, "answer")
            used_ids = set(grounded_local_answer["source_ids"]) if grounded_local_answer else set()
            answer["citations"] = [citation for citation in (local_qa or {}).get("citations", [])
                                    if not used_ids or citation["id"] in used_ids]
            answer["local_model_used"] = bool(grounded_local_answer)
            if state.get("stage") == "interview" and pending in QUESTIONS:
                resumed = _add_message(state, "assistant", "نكمل من السؤال السابق: " + QUESTIONS[pending], "question")
                resumed["question_plan"] = question_plan(state, pending)
            state["assessment"] = _assessment(state)
            return
        if mixed_pending_guidance:
            state["pending_guidance"] = None
        if question_mode:
            user_msg = _add_message(state, "user", text, "question")
            state["updated_at"] = user_msg["created_at"]
            if screen_current_danger(state, text, user_msg, pending):
                return
            clarification_required = False
            if intent == "why":
                reply = _explain_why(_why_target_slot(text, pending))
                citations = []
            elif intent == "assessment_why":
                reply = explain_assessment(state)
                citations = []
            else:
                result = local_qa or answer_question(guidance_question, state.get("jurisdiction"),
                                                     context_text=local_guidance_context + " " + text,
                                                     allow_nonlocal_general=True,
                                                     platform_context=guidance_platform,
                                                     platform_context_ambiguous=guidance_platform_ambiguous,
                                                     platform_query=text, scope_query=text,
                                                     topic_query=text)
                reply = grounded_local_answer["text"] if grounded_local_answer else result["text"]
                if grounded_local_answer:
                    reply += "\n\nصيغ الرد من مقتطفات حرفية موثقة، ولم يُرسل سجل المقابلة. راجع المصدر ولا تعتبره نصيحة قانونية."
                used_ids = set(grounded_local_answer["source_ids"]) if grounded_local_answer else set()
                citations = [citation for citation in result["citations"] if not used_ids or citation["id"] in used_ids]
                clarification_required = (intent not in {"why", "assessment_why"}
                                          and result.get("clarification_required") == "platform")
            response = _add_message(state, "assistant", reply,
                                    "guidance_clarification" if clarification_required else "answer")
            response["citations"] = citations
            response["local_model_used"] = bool(grounded_local_answer)
            if clarification_required:
                state["pending_guidance"] = {"status": "awaiting_platform", "query": guidance_question,
                                             "topic_query": text, "scope_query": text,
                                             "interview_pending_slot": pending, "clarification_text": reply}
            elif state.get("stage") == "interview" and pending in QUESTIONS:
                resumed_question = _add_message(state, "assistant", "نكمل من السؤال السابق: " + QUESTIONS[pending], "question")
                resumed_question["question_plan"] = question_plan(state, pending)
                state["pending_slot"] = pending
            state["assessment"] = _assessment(state)
            return

        handle_turn(state, text, model_facts=model_facts, message_interpretation=message_interpretation,
                    fact_text=message_interpretation.get("fact_text"))
        if mixed_why and state.get("stage") != "urgent_stop":
            if explanation_intent == "assessment_why":
                explanation = explain_assessment(state)
            else:
                target_slot = _why_target_slot(text, pending)
                explanation = _explain_why(target_slot)
                paid = state.get("facts", {}).get("paid_amount", {})
                if target_slot == "deadline" and paid.get("status") == "reported":
                    explanation = "الدفع مسجل عندي؛ " + explanation
            _append_answer_before_followup(state, "سجلت الجزء الذي ذكرته في مسودة الإفادة.\n\n" + explanation)
        if mixed_guidance and local_qa is not None:
            _append_guidance_before_followup(state, local_qa, grounded_local_answer)
        state["assessment"] = _assessment(state)

    save_started = time.perf_counter()
    state = _mutate(session_id, x_case_token, body.expected_revision, "user_message", mutate)
    save_ms = round((time.perf_counter() - save_started) * 1000, 3)
    adaptive_question_result = {"attempted": False, "accepted": False, "rejection_code": None}
    adaptive_question_ms = 0.0
    adaptive_enabled = bool(model_enabled_for_turn and state.get("adaptive_questions_enabled")
                            and provider_config().get("adaptive_question_generation_enabled")
                            and model_configured() and not urgent and not question_mode
                            and state.get("stage") == "interview")
    adaptive_slot = state.get("pending_slot")
    last_message = state.get("messages", [])[-1] if state.get("messages") else {}
    if adaptive_enabled and adaptive_slot in QUESTIONS and last_message.get("role") == "assistant" \
            and last_message.get("kind") in {"question", "clarification"}:
        telemetry_before_adaptive = snapshot_telemetry()
        traces_before_adaptive = len(telemetry_before_adaptive.get("history", []))
        adaptive_started = time.perf_counter()
        adaptive_question_result = generate_adaptive_question(
            adaptive_slot, _sanitized_adaptive_question_context(state, adaptive_slot))
        adaptive_question_ms = round((time.perf_counter() - adaptive_started) * 1000, 3)
        traces_after_generation = call_trace_history()[traces_before_adaptive:]
        restore_telemetry(telemetry_before_adaptive, traces_after_generation)
        adaptive_question_result["attempted"] = any(trace.get("attempted") for trace in traces_after_generation)
        if adaptive_question_result.get("accepted") and adaptive_question_result.get("question"):
            replacement = adaptive_question_result["question"]

            def apply_adaptive_question(current_state):
                if current_state.get("stage") != "interview" or current_state.get("pending_slot") != adaptive_slot:
                    return
                candidate = current_state.get("messages", [])[-1] if current_state.get("messages") else {}
                if (candidate.get("role") == "assistant" and candidate.get("kind") in {"question", "clarification"}
                        and (candidate.get("question_plan") or {}).get("slot") == adaptive_slot):
                    candidate["text"] = replacement
                    candidate["local_model_question"] = True
                    current_state["last_response"] = replacement

            try:
                state = _mutate(session_id, x_case_token, state["revision"],
                                "adaptive_question_generated", apply_adaptive_question)
            except HTTPException as exc:
                if exc.status_code == 409:
                    state = _state_for(session_id, x_case_token)
                    adaptive_question_result["accepted"] = False
                    adaptive_question_result["question"] = None
                    adaptive_question_result["rejection_code"] = "case_changed_before_question_save"
                else:
                    raise
    # An empty/fully-filtered extraction is a valid response at most, not successful extraction.
    call_accepted = grounded_local_answer is not None or bool(model_facts)
    provider_call = last_call_trace()
    provider_calls = call_trace_history()
    if not provider_calls and provider_call and provider_call.get("attempted"):
        provider_calls = [provider_call]
    provider = provider_config()
    if local_question_sent or local_extraction_sent or adaptive_question_result.get("attempted"):
        provider_call = dict(provider_call or {})
        provider_call.update({
            "attempted": bool(provider_call.get("attempted")),
            "turn_any_result_accepted": call_accepted,
            "turn_fallback_used": not call_accepted,
            "provider": provider["provider"],
            "model": provider["model"],
            "context_sent": bool(provider_call.get("context_sent", extraction_context is not None)),
            "context_char_count": extraction_context.get("serialized_char_count") if extraction_context else 0,
            "context_fact_count": len(extraction_context.get("related_facts", [])) if extraction_context else 0,
            "context_message_count": len(extraction_context.get("recent_relevant_messages", [])) if extraction_context else 0,
            "message_plan_applied": message_plan_applied,
            "message_goals": model_message_plan["goals"],
        })
        if not call_accepted and not provider_call.get("failure_code"):
            provider_call["failure_code"] = "no_valid_result"
        if model_facts is not None:
            provider_call["fact_fields"] = [fact.get("field") for fact in model_facts]
        for call in provider_calls:
            call["provider"] = provider["provider"]
            call["model"] = provider["model"]
            if call.get("attempted") and call.get("purpose") != "adaptive_question":
                call["fallback_used"] = not bool(call.get("result_accepted"))
        for call in provider_calls:
            if call.get("purpose") != "adaptive_question":
                call.update({"turn_any_result_accepted": call_accepted,
                             "turn_fallback_used": not call_accepted,
                             "message_plan_applied": message_plan_applied,
                             "message_goals": model_message_plan["goals"]})
    extraction_fallback_used = bool(local_extraction_sent and not model_facts)
    guidance_fallback_used = bool(local_question_sent and not grounded_local_answer)
    fallback_used = extraction_fallback_used or guidance_fallback_used
    fallback_targets = []
    if extraction_fallback_used:
        fallback_targets.append("rules")
    if guidance_fallback_used:
        fallback_targets.append("local_rag")
    fallback_to = "+".join(fallback_targets) if fallback_targets else None
    extraction_diagnostic = last_extraction_diagnostic() if local_extraction_sent else None
    filtered_diagnostics = []
    for item in (extraction_diagnostic or {}).get("filtered_facts", []):
        if not isinstance(item, dict):
            continue
        field = item.get("field") if isinstance(item.get("field"), str) else None
        code = item.get("filter_code") if isinstance(item.get("filter_code"), str) else "other_validation"
        detail = item.get("filter_detail") if isinstance(item.get("filter_detail"), str) else "لم يُسجل تفصيل التحقق."
        filtered_diagnostics.append({"field": field, "field_label": LABELS.get(field, "معلومة إضافية"),
                                     "code": code, "detail": detail})
    if local_extraction_sent and not filtered_diagnostics:
        trace_fields = (provider_call or {}).get("filtered_fact_fields", [])
        trace_code_counts = (provider_call or {}).get("filtered_fact_code_counts", {})
        if isinstance(trace_fields, list) and len(trace_fields) == 1 and isinstance(trace_code_counts, dict) and len(trace_code_counts) == 1:
            field = trace_fields[0] if isinstance(trace_fields[0], str) else None
            code, count = next(iter(trace_code_counts.items()))
            if field and isinstance(code, str) and count:
                filtered_diagnostics.append({"field": field, "field_label": LABELS.get(field, "معلومة إضافية"),
                                             "code": code, "detail": "تفصيل التحقق غير محفوظ في trace هذا التشغيل."})
    model_diagnostics = None
    if local_extraction_sent or local_question_sent or adaptive_question_result.get("attempted"):
        trace_response_valid = (provider_call or {}).get("response_valid")
        trace_status = (provider_call or {}).get("status")
        raw_received = bool(isinstance((extraction_diagnostic or {}).get("raw_response"), str)
                            and (extraction_diagnostic or {}).get("raw_response", "").strip())
        model_diagnostics = {
            "provider_response": {"received": (raw_received or trace_response_valid is True
                                                   or trace_status in {"valid_empty", "all_facts_filtered", "partially_accepted"})
                                                 if local_extraction_sent else bool(grounded_local_answer),
                                  "status": trace_status,
                                  "latency_ms": (provider_call or {}).get("latency_ms"),
                                  "failure_code": (provider_call or {}).get("failure_code")},
            "validation": {"response_valid": ((extraction_diagnostic or {}).get("accepted")
                                                 if extraction_diagnostic else trace_response_valid)
                                                if local_extraction_sent else None,
                           "accepted_fields": [fact.get("field") for fact in (model_facts or [])] if local_extraction_sent else [],
                           "filtered_facts": filtered_diagnostics if local_extraction_sent else [],
                           "unsupported_message_goals": list((extraction_diagnostic or {}).get(
                               "unsupported_message_goals", [])) if local_extraction_sent else []},
            "merge": {"performed": bool(last_model_merge_trace()),
                      "field_dispositions": last_model_merge_trace() if local_extraction_sent else []},
            "fallback": {"used": fallback_used, "to": fallback_to},
            "adaptive_question": {"attempted": bool(adaptive_question_result.get("attempted")),
                                  "accepted": bool(adaptive_question_result.get("accepted")),
                                  "rejection_code": adaptive_question_result.get("rejection_code"),
                                  "fallback_to": "rules" if adaptive_question_result.get("attempted")
                                  and not adaptive_question_result.get("accepted") else None},
            "raw_response_included": False,
        }
        if provider_call is not None and local_extraction_sent:
            provider_call["validation_diagnostics"] = filtered_diagnostics
    timings = {"request_preflight_ms": request_preflight_ms, "provider_preflight_ms": None,
               "provider_preflight_performed": False, "message_understanding_ms": understanding_ms,
               "retrieval_ms": retrieval_ms, "answer_generation_ms": answer_generation_ms,
               "save_ms": save_ms, "adaptive_question_ms": adaptive_question_ms,
               "total_ms": round((time.perf_counter() - request_started) * 1000, 3)}
    return {"session": state, "report": report_data(state),
            "local_model_used": bool(grounded_local_answer) or bool(model_facts) or message_plan_applied or bool(adaptive_question_result.get("attempted")),
            "local_data_sent": any(bool(call.get("attempted")) for call in provider_calls),
            "local_model_purpose": "grounded_question" if local_question_sent else "current_message_extraction" if local_extraction_sent else "adaptive_question" if adaptive_question_result.get("attempted") else None,
            "local_context_sent": bool(provider_call and provider_call.get("context_sent")),
            "adaptive_context_sent": bool(adaptive_question_result.get("attempted")),
            "local_model_result_accepted": call_accepted,
            "adaptive_question": {"attempted": bool(adaptive_question_result.get("attempted")),
                                  "accepted": bool(adaptive_question_result.get("accepted")),
                                  "rejection_code": adaptive_question_result.get("rejection_code"),
                                  "fallback_to_rules": bool(adaptive_question_result.get("attempted")
                                                             and not adaptive_question_result.get("accepted"))},
            "message_plan_applied": message_plan_applied,
            "message_plan": model_message_plan if message_plan_applied else None,
            "model_merge_trace": last_model_merge_trace() if local_extraction_sent else [],
            "extraction_outcome": (provider_call or {}).get("extraction_outcome") if local_extraction_sent else None,
            "fallback_used": fallback_used,
            "extraction_fallback_used": extraction_fallback_used,
            "guidance_fallback_used": guidance_fallback_used,
            "fallback_to": fallback_to,
            "question_detected": auto_question or resolving_pending_guidance,
            "runtime_mode": provider["mode"] if call_accepted or message_plan_applied else "rules",
            "provider_call": provider_call, "provider_calls": provider_calls, "timings": timings,
            "model_diagnostics": model_diagnostics,
            "incident_track": state.get("incident_track") or state.get("assessment", {}).get("incident_track"),
            "stopping_decision": state.get("stopping_decision")}


@app.post("/api/sessions/{session_id}/resume")
def resume(session_id: str, body: RevisionRequest, x_case_token: str = Header(default="")):
    state = _mutate(session_id, x_case_token, body.expected_revision, "session_resumed", resume_state)
    return {"session": state, "report": report_data(state)}


@app.put("/api/sessions/{session_id}/jurisdiction")
def set_jurisdiction(session_id: str, body: JurisdictionRequest, x_case_token: str = Header(default="")):
    available = {item["code"] for item in available_jurisdictions()}
    if body.jurisdiction and body.jurisdiction not in available:
        raise HTTPException(422, "هذا الاختصاص غير متاح لعدم وجود مصادر محلية معتمدة.")

    def mutate(state):
        state["jurisdiction"] = body.jurisdiction or None

    state = _mutate(session_id, x_case_token, body.expected_revision, "jurisdiction_changed", mutate)
    return {"session": state, "report": report_data(state)}


@app.post("/api/sessions/{session_id}/confirm")
def confirm(session_id: str, body: RevisionRequest, x_case_token: str = Header(default="")):
    def mutate(state):
        if state.get("stage") not in {"review", "urgent_stop"}:
            raise ValueError("راجع المعلومات أولًا قبل تأكيدها.")
        if any(fact.get("status") == "conflict" for fact in state.get("facts", {}).values()):
            raise ValueError("وضّح المعلومات المتعارضة أو صححها قبل تأكيد الملخص.")
        if _unresolved_money_relation(state):
            raise ValueError("وضّح علاقة المبلغ غير المحسومة قبل تأكيد الملخص.")
        state["assessment"] = _assessment(state)
        state["confirmed"] = True
        state["confirmed_digest"] = summary_digest(state)
        state.setdefault("confirmation_history", []).append({"summary_version": state["summary_version"],
                                                                "digest": state["confirmed_digest"], "confirmed_at": now_iso()})
        _add_message(state, "assistant", "تم تأكيد إصدار الملخص الحالي بحسب اختيارك. هذا التأكيد لا يقدّم بلاغًا رسميًا ولا يتحقق من صحة الوقائع.", "confirmation")

    state = _mutate(session_id, x_case_token, body.expected_revision, "summary_confirmed", mutate)
    return {"session": state, "report": report_data(state)}


@app.post("/api/sessions/{session_id}/resolve-temporal-conflict")
def resolve_temporal_conflict(session_id: str, body: TemporalConflictResolutionRequest,
                              x_case_token: str = Header(default="")):
    temporal_fields = {"initial_time", "latest_time", "discovery_time", "account_creation_time"}
    if body.field not in temporal_fields:
        raise HTTPException(422, "الحقل المطلوب ليس حقلًا زمنيًا قابلًا للتوضيح.")

    def mutate(state):
        fact = state.get("facts", {}).get(body.field)
        if not isinstance(fact, dict) or fact.get("status") != "conflict":
            raise ValueError("لا يوجد تعارض زمني مفتوح لهذا الحقل.")
        alternatives = fact.get("alternatives", [])
        selected = next((item for item in alternatives if item.get("status") == "reported"
                         and item.get("value") == body.selected_value), None)
        if not selected:
            raise ValueError("اختر قيمة واردة حرفيًا ضمن بدائل التعارض المسجلة.")
        source_message = next((message for message in state.get("messages", [])
                               if message.get("id") == selected.get("message_id") and message.get("role") == "user"), None)
        if not selected.get("quote") or not source_message or selected["quote"] not in source_message.get("text", ""):
            raise ValueError("لا يمكن حسم البديل لأن اقتباسه غير قابل للتحقق من الرسالة الأصلية.")
        resolved_at = now_iso()
        was_active_question = state.get("stage") == "interview" and state.get("pending_slot") == body.field
        previous_conflict = {key: copy.deepcopy(value) for key, value in fact.items()
                             if key in {"status", "value", "quote", "message_id", "alternatives", "history", "updated_at"}}
        fact.setdefault("history", []).append({**previous_conflict,
            "at": resolved_at, "reason": "resolved_by_user_selection", "resolution_reason": body.resolution_reason})
        fact.update({"status": "reported", "value": selected["value"], "quote": selected.get("quote"),
                     "message_id": selected.get("message_id"), "question_id": selected.get("question_id"),
                     "source": selected.get("source", "rules"), "updated_at": resolved_at,
                     "resolution": {"selected_value": selected["value"], "selected_quote": selected.get("quote"),
                                    "reason": body.resolution_reason, "resolved_at": resolved_at}})
        fact.pop("alternatives", None)
        _add_message(state, "user", f"حُسم حقل {body.field} باختيار الإفادة المسجلة «{selected['value']}». السبب: {body.resolution_reason}",
                     "temporal_conflict_resolution")
        _finalize_changed_facts(state, {body.field})
        # Resolution never confirms or submits the summary.
        state["confirmed"] = False
        state["confirmed_digest"] = None
        state["assessment"] = _assessment(state)
        if was_active_question:
            next_slot = _next_slot(state)
            state["pending_slot"] = next_slot
            if next_slot:
                state["stage"] = "interview"
                next_question = _add_message(state, "assistant", QUESTIONS[next_slot], "question")
                next_question["question_plan"] = question_plan(state, next_slot)
            else:
                state["stage"] = "review"
                _add_message(state, "assistant", "تم توضيح الوقت. راجع الملخص الحالي قبل التأكيد.", "review")
        elif state.get("stage") in {"review", "urgent_stop"}:
            _add_message(state, "assistant", "تم توضيح الوقت المسجل. راجع الملخص الحالي قبل تأكيده.", "review")

    state = _mutate(session_id, x_case_token, body.expected_revision, "temporal_conflict_resolved", mutate)
    return {"session": state, "report": report_data(state)}


@app.get("/api/sessions/{session_id}/report.md")
def download_report(session_id: str, x_case_token: str = Header(default="")):
    state = _state_for(session_id, x_case_token)
    markdown = render_report(state)
    filename = f"ACCP-{session_id[:8]}-summary.md"
    return Response(content=markdown, media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/sessions/{session_id}/report.json")
def download_report_json(session_id: str, x_case_token: str = Header(default="")):
    state = _state_for(session_id, x_case_token)
    filename = f"ACCP-{session_id[:8]}-summary.json"
    return JSONResponse(content=report_data(state),
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/sessions/{session_id}/audit")
def get_audit(session_id: str, x_case_token: str = Header(default="")):
    _state_for(session_id, x_case_token)
    try:
        return {"events": store.audit(session_id, x_case_token)}
    except Forbidden:
        raise HTTPException(403, "بيانات الدخول إلى الحالة غير صحيحة.")


@app.exception_handler(Exception)
async def generic_error_handler(request, exc):
    # Do not expose implementation details or secret configuration.
    return JSONResponse(status_code=500, content={"detail": "حدث عطل تقني. راجع آخر نسخة محفوظة قبل إعادة إرسال الرسالة."})


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
