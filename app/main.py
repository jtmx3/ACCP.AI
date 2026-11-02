from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core import (
    QUESTIONS, _add_message, _assessment, _explain_why, _next_slot, _record,
    _user_intent, classify_current_danger, confirmation_is_current, handle_turn, now_iso, render_report,
    report_data, resume_state, screen_current_danger, summary_digest,
)
from app.rag import answer_question, available_jurisdictions, load_corpus
from app.model import generate_grounded_answer, generate_interview_extraction, model_configured
from app.store import Forbidden, NotFound, StaleRevision, Store

app = FastAPI(title="ACCP.AI", version="0.2.0", description="Local-first Arabic incident intake and review prototype")
store = Store()
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class TurnRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=5000)
    kind: str = "answer"
    allow_external: bool = False
    allow_external_extraction: bool = False

    def model_post_init(self, __context):
        if self.kind not in {"answer", "question"}:
            raise ValueError("نوع الرسالة غير مدعوم.")


class RevisionRequest(BaseModel):
    expected_revision: int = Field(ge=0)


class JurisdictionRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    jurisdiction: str = ""


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
    configured = model_configured()
    return {"status": "ok", "storage": "sqlite", "model_provider": "openai-compatible" if configured else "rules",
            "external_model_configured": configured, "version": app.version}


@app.get("/api/metadata")
def metadata():
    return {"jurisdictions": available_jurisdictions(), "corpus_version": load_corpus()["corpus_version"]}


@app.post("/api/sessions")
def create_session():
    state, token = store.create()
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


@app.post("/api/sessions/{session_id}/messages")
def post_message(session_id: str, body: TurnRequest, x_case_token: str = Header(default="")):
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "اكتب رسالة قبل الإرسال.")

    # Authenticate and check revision before any optional network operation.
    snapshot = _preflight(session_id, x_case_token, body.expected_revision)
    pending = snapshot.get("pending_slot")
    intent = _user_intent(text)
    if body.kind == "question" and intent != "why":
        intent = None
    urgent = classify_current_danger(text, pending) is True
    local_qa = None
    external_answer = None
    model_facts = None
    external_extraction_sent = False
    external_question_sent = False

    if body.kind == "question" and not urgent and intent != "why":
        local_qa = answer_question(text, snapshot.get("jurisdiction"))
        if body.allow_external and model_configured() and local_qa["citations"]:
            external_question_sent = True
            corpus = load_corpus()
            source_map = {item["id"]: item for item in corpus["sources"]}
            selected_sources = []
            for citation in local_qa["citations"]:
                source = source_map.get(citation["id"])
                if source:
                    selected_sources.append({**source, "excerpt": citation.get("excerpt", "")})
            external_answer = generate_grounded_answer(text, selected_sources)

    if body.kind == "answer" and body.allow_external_extraction and model_configured() and not urgent and intent is None:
        # Send only the current user message and pending slot; never the transcript or saved facts.
        external_extraction_sent = True
        model_facts = generate_interview_extraction(text, pending)

    def mutate(state):
        if body.kind == "question":
            user_msg = _add_message(state, "user", text, "question")
            state["updated_at"] = user_msg["created_at"]
            if screen_current_danger(state, text, user_msg, pending):
                return
            if intent == "why":
                reply = _explain_why(pending or "") + ("\n\n" + QUESTIONS[pending] if pending in QUESTIONS else "")
                citations = []
            else:
                result = local_qa or answer_question(text, state.get("jurisdiction"))
                reply = external_answer["text"] if external_answer else result["text"]
                if external_answer:
                    reply += "\n\nصيغ الرد من مقتطفات حرفية موثقة، ولم يُرسل سجل المقابلة. راجع المصدر ولا تعتبره نصيحة قانونية."
                used_ids = set(external_answer["source_ids"]) if external_answer else set()
                citations = [citation for citation in result["citations"] if not used_ids or citation["id"] in used_ids]
            response = _add_message(state, "assistant", reply, "answer")
            response["citations"] = citations
            response["external_model_used"] = bool(external_answer)
            if state.get("stage") == "interview" and pending in QUESTIONS:
                _add_message(state, "assistant", "نكمل من السؤال السابق: " + QUESTIONS[pending], "question")
                state["pending_slot"] = pending
            state["assessment"] = _assessment(state)
            return

        handle_turn(state, text, model_facts=model_facts)
        state["assessment"] = _assessment(state)

    state = _mutate(session_id, x_case_token, body.expected_revision, "user_message", mutate)
    return {"session": state, "report": report_data(state),
            "external_model_used": bool(external_answer) or bool(model_facts),
            "external_data_sent": external_question_sent or external_extraction_sent,
            "external_model_purpose": "grounded_question" if external_question_sent else "current_message_extraction" if external_extraction_sent else None}


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
        state["assessment"] = _assessment(state)
        state["confirmed"] = True
        state["confirmed_digest"] = summary_digest(state)
        state.setdefault("confirmation_history", []).append({"summary_version": state["summary_version"],
                                                                "digest": state["confirmed_digest"], "confirmed_at": now_iso()})
        _add_message(state, "assistant", "تم تأكيد إصدار الملخص الحالي بحسب اختيارك. هذا التأكيد لا يقدّم بلاغًا رسميًا ولا يتحقق من صحة الوقائع.", "confirmation")

    state = _mutate(session_id, x_case_token, body.expected_revision, "summary_confirmed", mutate)
    return {"session": state, "report": report_data(state)}


@app.get("/api/sessions/{session_id}/report.md")
def download_report(session_id: str, x_case_token: str = Header(default="")):
    state = _state_for(session_id, x_case_token)
    markdown = render_report(state)
    filename = f"ACCP-{session_id[:8]}-summary.md"
    return Response(content=markdown, media_type="text/markdown; charset=utf-8",
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
