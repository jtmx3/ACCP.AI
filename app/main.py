from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core import _assessment, _add_message, _explain_why, _record, _next_slot, QUESTIONS, handle_turn, is_immediate_danger_disclosure, render_report, resume_state, report_data, summary_digest
from app.rag import answer_question, load_corpus
from app.model import generate_grounded_answer, model_configured
from app.store import Forbidden, NotFound, StaleRevision, Store

app = FastAPI(title="ACCP.AI", version="0.1.0", description="Local-first Arabic incident intake and review prototype")
store = Store()
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class TurnRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=5000)
    kind: Literal["answer", "question"] = "answer"
    allow_external: bool = False


class RevisionRequest(BaseModel):
    expected_revision: int = Field(ge=0)


class JurisdictionRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    jurisdiction: Literal["", "AU", "US", "JO"]


def _state_for(session_id: str, token: str):
    try:
        return store.get(session_id, token)
    except NotFound:
        raise HTTPException(404, "الحالة غير موجودة.")
    except Forbidden:
        raise HTTPException(403, "بيانات الدخول إلى الحالة غير صحيحة.")


def _mutate(session_id: str, token: str, rev: int, event: str, fn):
    try:
        return store.mutate(session_id, token, rev, event, fn)
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


@app.post("/api/sessions")
def create_session():
    state, token = store.create()
    state["jurisdiction"] = None
    return {"session": state, "token": token, "token_notice": "هذا الرمز خاص بالحالة. لا تشاركه؛ لا يمكن استعادته من الخادم."}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, x_case_token: str = Header(default="")):
    state = _state_for(session_id, x_case_token)
    state["assessment"] = _assessment(state)
    return {"session": state, "report": report_data(state)}


@app.post("/api/sessions/{session_id}/messages")
def post_message(session_id: str, body: TurnRequest, x_case_token: str = Header(default="")):
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "اكتب رسالة قبل الإرسال.")

    def mutate(state):
        if body.kind == "question":
            question = text
            user_msg = _add_message(state, "user", question, "question")
            state["updated_at"] = user_msg["created_at"]
            normalized = question.casefold()
            pending = state.get("pending_slot")
            if is_immediate_danger_disclosure(question):
                if _record(state, "direct_danger", "reported", True, question, user_msg["id"], None):
                    state["confirmed"] = False
                    state["confirmed_digest"] = None
                    state["summary_version"] += 1
                state["stage"] = "urgent_stop"
                state["pending_slot"] = None
                state["assessment"] = _assessment(state)
                response = _add_message(state, "assistant", "سلامتك أولًا. إذا كنت بخطر مباشر، ابتعد لمكان آمن إن أمكن، وتواصل مع خدمات الطوارئ المحلية أو شخص موثوق قريب منك الآن. ما بقدر أحدد رقم طوارئ من دون معرفة بلدك.", "safety")
                response["citations"] = []
                response["external_model_used"] = False
                return
            response_external = False
            if any(word in normalized for word in ("ليش بتسأل", "ليش بدك تعرف", "لماذا تسأل", "ليش السؤال")):
                reply = _explain_why(pending or "") + ("\n\n" + QUESTIONS[pending] if pending in QUESTIONS else "")
                citations = []
            else:
                result = answer_question(question, state.get("jurisdiction"))
                reply, citations = result["text"], result["citations"]
                if body.allow_external and model_configured() and citations:
                    corpus = load_corpus()
                    source_map = {item["id"]: item for item in corpus["sources"]}
                    selected_sources = [source_map[item["id"]] for item in citations if item["id"] in source_map]
                    draft = generate_grounded_answer(question, selected_sources)
                    if draft:
                        reply = draft + "\n\nتمت صياغة هذا الرد عبر مزود خارجي بناءً على المصادر الظاهرة فقط؛ راجع الروابط ولا تعتبره نصيحة قانونية."
                        response_external = True
                    else:
                        response_external = False
                else:
                    response_external = False
            response = _add_message(state, "assistant", reply, "answer")
            response["citations"] = citations
            if body.kind == "question":
                response["external_model_used"] = response_external if not any(word in question.casefold() for word in ("ليش بتسأل", "ليش بدك تعرف", "لماذا تسأل", "ليش السؤال")) else False
            if state.get("stage") == "interview" and pending in QUESTIONS:
                follow = _add_message(state, "assistant", "نكمل من السؤال السابق: " + QUESTIONS[pending], "question")
                state["pending_slot"] = pending
            state["assessment"] = _assessment(state)
        else:
            handle_turn(state, text)
            state["assessment"] = _assessment(state)
    state = _mutate(session_id, x_case_token, body.expected_revision, "user_message", mutate)
    return {"session": state, "report": report_data(state)}


@app.post("/api/sessions/{session_id}/resume")
def resume(session_id: str, body: RevisionRequest, x_case_token: str = Header(default="")):
    state = _mutate(session_id, x_case_token, body.expected_revision, "session_resumed", resume_state)
    return {"session": state, "report": report_data(state)}


@app.put("/api/sessions/{session_id}/jurisdiction")
def set_jurisdiction(session_id: str, body: JurisdictionRequest, x_case_token: str = Header(default="")):
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
