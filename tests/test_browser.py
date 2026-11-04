import asyncio
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.parse import urlparse
from unittest.mock import patch

import uvicorn
from playwright.async_api import async_playwright, expect

from app import main
from app.store import Store


def browser_state(session_id, revision, marker, *, stage="interview", messages=None):
    return {
        "session_id": session_id,
        "revision": revision,
        "stage": stage,
        "pending_slot": "narrative" if stage == "interview" else None,
        "messages": messages or [{"id": "m1", "role": "assistant", "text": marker, "kind": "question"}],
        "facts": {},
        "assessment": {"priority_score": None, "priority_label": "غير محددة", "priority_rule": None,
                       "supporting_fields": [], "reasons": [], "relevant_fields": [], "completeness_percent": 0,
                       "completeness_note": "0 من 0", "extraction_quality": "test"},
        "confirmed": False,
        "jurisdiction": None,
    }


class BrowserWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        main.store = Store(str(Path(self.tmp.name) / "browser.sqlite3"))
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(128)
        self.port = self.socket.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(main.app, log_level="error", lifespan="off"))
        self.server_thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.socket]}, daemon=True)
        self.server_thread.start()
        for _ in range(100):
            if self.server.started:
                break
            await asyncio.sleep(0.05)
        self.assertTrue(self.server.started, "uvicorn test server did not start")
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(accept_downloads=True)
        self.page = await self.context.new_page()

    async def asyncTearDown(self):
        await self.context.close()
        await self.browser.close()
        await self.playwright.stop()
        self.server.should_exit = True
        self.server_thread.join(timeout=5)
        self.socket.close()
        self.tmp.cleanup()

    async def _send(self, text):
        await self.page.locator("#message").fill(text)
        await self.page.locator("#send").click()

    async def test_runtime_panel_distinguishes_configuration_connection_model_and_generation(self):
        config = {"provider": "openai-compatible", "mode": "external_model", "model": "synthetic-model",
                  "configuration_complete": True, "configured": True, "local": False}
        stage = {"probe": {"provider": "openai-compatible", "configured": True, "service_reachable": False,
                            "model_available": False, "models_count": 0, "failure_code": "unreachable"}}

        async def intercept(route):
            url = route.request.url
            if url.endswith("/api/health"):
                await route.fulfill(json={"status": "ok", "provider": config})
            elif url.endswith("/api/provider/status"):
                await route.fulfill(json={"config": config, **stage})
            elif url.endswith("/api/provider/test"):
                await route.fulfill(json={"success": True, "attempted": True, "latency_ms": 12,
                                          "failure_code": None, "provider": "openai-compatible",
                                          "model": "synthetic-model"})
            else:
                await route.continue_()

        await self.page.route("**/api/**", intercept)
        await self.page.goto(self.base_url)
        await expect(self.page.locator("#runtime-mode")).to_contain_text("لم يُثبت الاتصال بعد")

        await self.page.locator("#check-provider").click()
        await expect(self.page.locator("#runtime-mode")).to_contain_text("القواعد المحلية")
        await expect(self.page.locator("#runtime-detail")).to_contain_text("الخدمة غير متاحة")
        self.assertTrue(await self.page.locator("#test-provider-generation").is_disabled())

        stage["probe"] = {"provider": "openai-compatible", "configured": True, "service_reachable": True,
                           "model_available": True, "models_count": 1, "failure_code": None}
        await self.page.locator("#check-provider").click()
        await expect(self.page.locator("#runtime-mode")).to_contain_text("الخدمة والنموذج متاحان")
        await expect(self.page.locator("#runtime-detail")).to_contain_text("لم يُختبر التوليد بعد")
        self.assertTrue(await self.page.locator("#test-provider-generation").is_enabled())

        await self.page.locator("#test-provider-generation").click()
        await expect(self.page.locator("#runtime-mode")).to_contain_text("مزود خارجي — نجح اختبار التوليد الثابت")
        await expect(self.page.locator("#runtime-detail")).to_contain_text("12 مللي ثانية")

    async def test_partial_model_filter_is_reported_after_consented_send(self):
        config = {"provider": "openai-compatible", "mode": "external_model", "model": "synthetic-test",
                  "configuration_complete": True, "configured": True, "local": False}
        async def intercept(route):
            if route.request.url.endswith("/api/provider/status"):
                await route.fulfill(json={"config": config, "probe": {"configured": True,
                    "service_reachable": True, "model_available": True, "models_count": 1,
                    "failure_code": None}})
            else:
                await route.continue_()
        await self.page.route("**/api/provider/status", intercept)
        trace = {"attempted": True, "status": "partially_accepted", "result_accepted": True,
                 "fallback_used": False, "failure_code": "partial_fact_filter", "filtered_fact_count": 1,
                 "filtered_fact_fields": ["identity"], "latency_ms": 12}
        with patch("app.main.provider_config", return_value=config), \
             patch("app.main.model_configured", return_value=True), \
             patch("app.main.generate_interview_extraction", return_value=[{
                 "field": "platform", "status": "reported", "value": "إنستغرام", "quote": "إنستغرام"}]), \
             patch("app.main.last_call_trace", return_value=trace):
            await self.page.goto(self.base_url)
            await self.page.locator("#start-case").click()
            await expect(self.page.locator("#model-consent-row")).to_be_visible()
            await self.page.locator("#message").fill("ظهر حساب مزيف على إنستغرام")
            await self.page.locator("#model-consent").check()
            await self.page.locator("#send").click()
            await expect(self.page.locator("#runtime-detail")).to_contain_text("عدد الوقائع المقبولة 1")
            await expect(self.page.locator("#runtime-detail")).to_contain_text("عدد المقترحات المصفاة 1")

    async def test_valid_empty_model_response_is_not_shown_as_extraction_success(self):
        config = {"provider": "ollama", "mode": "local_model", "model": "fixture:latest",
                  "configuration_complete": True, "configured": True, "local": True,
                  "ollama_reasoning_effort": "none"}
        trace = {"attempted": True, "status": "valid_empty", "extraction_outcome": "valid_empty",
                 "response_valid": True, "result_accepted": False, "fallback_used": True,
                 "failure_code": "empty_extraction", "fact_fields": [], "filtered_fact_count": 0,
                 "filtered_fact_code_counts": {}, "latency_ms": 2980}
        async def intercept(route):
            if route.request.url.endswith("/api/provider/status"):
                await route.fulfill(json={"config": config, "probe": {"configured": True,
                    "service_reachable": True, "model_available": True, "models_count": 1,
                    "failure_code": None}})
            else:
                await route.continue_()
        await self.page.route("**/api/provider/status", intercept)
        with patch("app.main.provider_config", return_value=config), \
             patch("app.main.model_configured", return_value=True), \
             patch("app.main.generate_interview_extraction", return_value=[]), \
             patch("app.main.last_call_trace", return_value=trace):
            await self.page.goto(self.base_url)
            await self.page.locator("#start-case").click()
            await self.page.locator("#message").fill("في شخص عم يهددني على واتساب.")
            await self.page.locator("#model-consent").check()
            await self.page.locator("#send").click()
            await expect(self.page.locator("#runtime-mode")).to_contain_text("رد صالح")
            await expect(self.page.locator("#runtime-mode")).to_contain_text("بلا وقائع مستخرجة")
            await expect(self.page.locator("#status-message")).to_contain_text("لم يُعد ذلك نجاحًا للاستخراج")

    async def test_all_filtered_model_response_shows_reasons_and_local_fallback(self):
        config = {"provider": "ollama", "mode": "local_model", "model": "fixture:latest",
                  "configuration_complete": True, "configured": True, "local": True,
                  "ollama_reasoning_effort": "none"}
        trace = {"attempted": True, "status": "all_facts_filtered", "extraction_outcome": "all_facts_filtered",
                 "response_valid": True, "result_accepted": False, "fallback_used": True,
                 "failure_code": "all_facts_filtered", "fact_fields": [], "filtered_fact_count": 1,
                 "filtered_fact_fields": ["narrative"], "filtered_fact_code_counts": {"value_inconsistent": 1},
                 "latency_ms": 2980}
        async def intercept(route):
            if route.request.url.endswith("/api/provider/status"):
                await route.fulfill(json={"config": config, "probe": {"configured": True,
                    "service_reachable": True, "model_available": True, "models_count": 1,
                    "failure_code": None}})
            else:
                await route.continue_()
        await self.page.route("**/api/provider/status", intercept)
        with patch("app.main.provider_config", return_value=config), \
             patch("app.main.model_configured", return_value=True), \
             patch("app.main.generate_interview_extraction", return_value=[]), \
             patch("app.main.last_call_trace", return_value=trace):
            await self.page.goto(self.base_url)
            await self.page.locator("#start-case").click()
            await self.page.locator("#message").fill("في شخص عم يهددني على واتساب.")
            await self.page.locator("#model-consent").check()
            await self.page.locator("#send").click()
            await expect(self.page.locator("#runtime-mode")).to_contain_text("استُبعدت جميع مقترحات")
            await expect(self.page.locator("#runtime-detail")).to_contain_text("قيمة غير متسقة: 1")
            await expect(self.page.locator("#status-message")).to_contain_text("طُبقت القواعد المحلية")

    async def test_windows_f83_contact_and_corrected_amount_are_visible_in_review_and_exports(self):
        await self.page.goto(self.base_url)
        await self.page.locator("#start-case").click()
        turns = [
            "في شخص عم يهددني على واتساب.",
            "لا، ما في خطر مباشر علي الآن.",
            "قال لي إذا ما حولتله 100 دينار رح ينشر محادثاتي الخاصة.",
            "أول تهديد وصلني أمس الساعة 6 مساءً.",
            "نعم، اليوم رجع هددني بنشر محادثاتي الخاصة إذا ما دفعت.",
            "نعم، حولتله 130 دينار.",
            "حولت المبلغ عن طريق محفظة إلكترونية.",
            "بس أنا دفعت.",
            "لا، ما حدد أي مهلة أو موعد للدفع.",
            "ما بعرف إذا نشر المحادثات أو لا، ما شفت أي نشر.",
            "لا، ما بعرف هويته الحقيقية، بس عندي رقم الواتساب اللي تواصل معي منه، وما بدي أكتبه هون.",
            "عندي لقطات شاشة لرسائل التهديد على واتساب، وإيصال تحويل الـ130 دينار من المحفظة الإلكترونية.",
            "مثل ما ذكرت سابقًا، دفعت عن طريق محفظة إلكترونية. وإيصال الـ130 دينار يخص نفس التحويل السابق، مش دفعة جديدة.",
        ]
        for text in turns:
            await self._send(text)
        await expect(self.page.locator("#stage-badge")).to_have_text("مراجعة")
        facts = self.page.locator("#facts-list")
        await expect(facts).to_contain_text("هوية الشخص")
        await expect(facts).to_contain_text("معرّف التواصل")
        await expect(facts).to_contain_text("رقم واتساب متوفر بحسب الإفادة؛ المعرّف لم يُدرج")
        await expect(facts).to_contain_text("الإفصاح عن معرّف التواصل")
        await expect(facts).to_contain_text("فضّل عدم الإجابة")
        await expect(facts).to_contain_text("محفظة إلكترونية")
        await expect(facts).not_to_contain_text("تعارض فعلي يحتاج إلى توضيح")

        original_detail = "قال لي إذا ما حولتله 100 دينار رح ينشر محادثاتي الخاصة."
        await self._send("تصحيح: المبلغ المطلوب 120 دينار، وليس 100.")
        await expect(facts).to_contain_text(original_detail)
        await expect(facts).to_contain_text("المبلغ المطلوب المعتمد حاليًا 120 دينار")
        await expect(facts).to_contain_text("الإفادة الأصلية في هذا الموضع تذكر 100 دينار")
        await self.page.locator("#confirm-summary").click()
        await expect(self.page.locator("#confirmation-state")).to_contain_text("مؤكد لهذا الإصدار")

        async with self.page.expect_download() as markdown_event:
            await self.page.locator("#download-report").click()
        markdown_download = await markdown_event.value
        markdown_path = Path(self.tmp.name) / markdown_download.suggested_filename
        await markdown_download.save_as(markdown_path)
        markdown = markdown_path.read_text(encoding="utf-8")
        self.assertIn(original_detail, markdown)
        self.assertIn("المبلغ المطلوب المعتمد حاليًا 120 دينار", markdown)
        self.assertIn("وقت الرسالة الأصلية: 2026", markdown)

        displayed_version = await self.page.locator("#review").get_attribute("data-summary-version")
        async with self.page.expect_download() as json_event:
            await self.page.locator("#download-json").click()
        json_download = await json_event.value
        json_path = Path(self.tmp.name) / json_download.suggested_filename
        await json_download.save_as(json_path)
        exported = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(str(exported["summary_version"]), displayed_version)
        self.assertTrue(exported["confirmed"])
        self.assertEqual(exported["facts"]["identity"]["status"], "unknown")
        self.assertEqual(exported["facts"]["contact"]["status"], "reported")
        self.assertEqual(exported["facts"]["contact_identifier_disclosure"]["status"], "declined")
        self.assertEqual(exported["facts"]["requested_amount"]["value"], "120 دينار")
        self.assertEqual(exported["facts"]["paid_amount"]["value"], "130 دينار")
        self.assertEqual(exported["facts"]["details"]["quote"], original_detail)
        self.assertEqual(exported["correction_notes"][0]["previous_value"], "100 دينار")
        self.assertTrue(exported["facts"]["contact"]["source_message_at"])

    async def test_impersonation_sequence_ui_and_exports_keep_scoped_facts(self):
        await self.page.goto(self.base_url)
        await self.page.locator("#start-case").click()
        turns = [
            "اليوم اكتشفت حساب على إنستغرام مستخدم اسمي وصورتي وبحكي للناس إنه أنا. حسابي الأصلي معي وبقدر أدخل عليه. ما بعرف مين عامل الحساب، وعندي لقطات شاشة للحساب المزيف، وما في تهديد أو طلب مصاري.",
            "شخص عامل حساب مزيف باسمي وصورتي على إنستغرام، وبتواصل مع الناس وبدّعي إنه أنا. حسابي الأصلي ما زال معي وبقدر أدخل عليه.",
            "مثل ما ذكرت، عندي لقطات شاشة للحساب المزيف بتوضح استخدام اسمي وصورتي.",
        ]
        facts = self.page.locator("#facts-list")

        await self._send(turns[0])
        await expect(facts).to_contain_text("انتحال شخصية")
        await expect(facts).to_contain_text("الأدلة المذكورة")
        await expect(facts).to_contain_text("القدرة على الدخول للحساب الأصلي")
        await expect(facts).to_contain_text("هوية الشخص")
        await expect(facts).not_to_contain_text("توضيح واقعة غير مصنفة")
        await expect(facts).not_to_contain_text("تعارض فعلي يحتاج إلى توضيح")

        await self._send(turns[1])
        await expect(facts).not_to_contain_text("تعارض فعلي يحتاج إلى توضيح")
        await expect(self.page.locator("#stage-badge")).to_have_text("مراجعة")
        await expect(self.page.locator("#chat")).not_to_contain_text("ما قدرت أحدد نوع الواقعة")

        await self._send(turns[2])
        await expect(facts).not_to_contain_text("تعارض فعلي يحتاج إلى توضيح")
        await expect(self.page.locator("#stage-badge")).to_have_text("مراجعة")

        async with self.page.expect_download() as json_event:
            await self.page.locator("#download-json").click()
        json_download = await json_event.value
        json_path = Path(self.tmp.name) / json_download.suggested_filename
        await json_download.save_as(json_path)
        exported = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(exported["facts"]["impersonation"]["status"], "reported")
        self.assertIs(exported["facts"]["impersonation"]["value"], True)
        self.assertEqual(exported["facts"]["evidence"]["status"], "reported")
        self.assertIn(turns[2], [item["quote"] for item in exported["facts"]["evidence"]["supporting_statements"]])
        self.assertEqual(exported["facts"]["identity"]["status"], "unknown")
        self.assertIs(exported["facts"]["original_account_access_available"]["value"], True)
        self.assertNotIn("account_access", exported["facts"])
        self.assertEqual(exported["facts"]["threat"]["status"], "denied")
        self.assertNotIn("requested_amount", exported["facts"])
        self.assertNotIn("paid_amount", exported["facts"])
        self.assertNotIn("unclassified_details", exported["facts"])

        async with self.page.expect_download() as markdown_event:
            await self.page.locator("#download-report").click()
        markdown_download = await markdown_event.value
        markdown_path = Path(self.tmp.name) / markdown_download.suggested_filename
        await markdown_download.save_as(markdown_path)
        markdown = markdown_path.read_text(encoding="utf-8")
        self.assertIn(turns[0], markdown)
        self.assertIn(turns[2], markdown)
        self.assertIn("القدرة على الدخول للحساب الأصلي", markdown)

    async def test_chat_pause_resume_correction_confirmation_and_download(self):
        await self.page.add_init_script("window.__accpPrintCalled=false;window.print=()=>{window.__accpPrintCalled=true;};")
        await self.page.goto(self.base_url)
        await self.page.locator("#start-case").click()
        await expect(self.page.locator("#stage-badge")).to_have_text("مقابلة")
        await self._send("اليوم شخص عامل حسابًا مزيفًا باسمي على إنستغرام")
        await expect(self.page.locator("#chat")).to_contain_text("شو قال لك")
        await expect(self.page.locator("#chat")).to_contain_text("إنستغرام")

        await self.page.locator("#pause-case").click()
        await expect(self.page.locator("#stage-badge")).to_have_text("محفوظة")
        await self.page.locator("#pause-case").click()
        await expect(self.page.locator("#stage-badge")).to_have_text("مقابلة")

        await self._send("ينتحل الحساب شخصيتي ويتواصل مع أصدقائي")
        await self._send("ما بعرف هويته")
        await self._send("ما عندي صور ولا رسائل")
        await expect(self.page.locator("#stage-badge")).to_have_text("مراجعة")
        await expect(self.page.locator("#facts-list")).to_contain_text("منفية صراحة بحسب الإفادة")
        await expect(self.page.locator("#facts-list")).to_contain_text("المنصة")
        await expect(self.page.locator("#information-gaps")).to_contain_text("غير معروفة")
        self.assertTrue(await self.page.locator("#confirm-summary").is_enabled())

        await self._send("تصحيح: بدأت الواقعة أمس")
        await expect(self.page.locator("#facts-list")).to_contain_text("أمس")
        self.assertTrue(await self.page.locator("#confirm-summary").is_enabled())
        await self.page.locator("#confirm-summary").click()
        await expect(self.page.locator("#confirmation-state")).to_contain_text("مؤكد لهذا الإصدار")

        async with self.page.expect_download() as download_event:
            await self.page.locator("#download-report").click()
        download = await download_event.value
        target = Path(self.tmp.name) / download.suggested_filename
        await download.save_as(target)
        self.assertTrue(target.exists())
        self.assertIn("بدأت الواقعة أمس", target.read_text(encoding="utf-8"))
        displayed_version = await self.page.locator("#review").get_attribute("data-summary-version")
        self.assertEqual(await self.page.locator("#review").get_attribute("data-confirmed"), "true")

        async with self.page.expect_download() as json_event:
            await self.page.locator("#download-json").click()
        json_download = await json_event.value
        json_path = Path(self.tmp.name) / json_download.suggested_filename
        await json_download.save_as(json_path)
        json_report = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(str(json_report["summary_version"]), displayed_version)
        self.assertTrue(json_report["confirmed"])
        self.assertEqual(json_report["field_labels"]["platform"], "المنصة")
        self.assertIn("identity", json_report["unknown_fields"])
        self.assertTrue(json_report["facts"]["platform"]["source_message_at"])
        self.assertTrue(json_report["facts"]["platform"]["last_updated_at"])

        await self.page.locator("#print-report").click()
        await self.page.wait_for_function("window.__accpPrintCalled === true")
        self.assertTrue(await self.page.evaluate("window.__accpPrintCalled"))
        self.assertEqual(await self.page.locator("#review").get_attribute("data-summary-version"), displayed_version)

    async def test_delayed_old_session_response_cannot_replace_active_equal_revision_session(self):
        a_id, b_id = "case-A-12345678", "case-B-abcdefgh"
        state_a = browser_state(a_id, 4, "A initial marker")
        state_b = browser_state(b_id, 5, "B-only marker")
        a_after = browser_state(a_id, 5, "A-response marker", messages=[
            {"id": "m1", "role": "assistant", "text": "A initial marker", "kind": "question"},
            {"id": "m2", "role": "user", "text": "A turn", "kind": "chat"},
            {"id": "m3", "role": "assistant", "text": "A-response marker", "kind": "question"},
        ])
        b_after = browser_state(b_id, 6, "B-only marker", messages=[
            {"id": "m1", "role": "assistant", "text": "B-only marker", "kind": "question"},
            {"id": "m2", "role": "user", "text": "B next turn", "kind": "chat"},
            {"id": "m3", "role": "assistant", "text": "B-next-turn response", "kind": "question"},
        ])
        await self.page.add_init_script("""localStorage.setItem('accp.local.cases.v1', JSON.stringify([
          {id: 'case-A-12345678', token: 'token-A', label: 'A'},
          {id: 'case-B-abcdefgh', token: 'token-B', label: 'B'}
        ]));
        window.__aPostSettled = new Promise(resolve => window.addEventListener('accp-test-a-post-settled', () => resolve(true), {once: true}));
        const originalFetch = window.fetch.bind(window);
        window.fetch = async (...args) => {
          const response = await originalFetch(...args);
          const requestUrl = String(args[0]);
          if (requestUrl.includes('/case-A-12345678/messages')) setTimeout(() => window.dispatchEvent(new Event('accp-test-a-post-settled')), 0);
          return response;
        };""")
        release_a = asyncio.Event()
        a_started = asyncio.Event()
        a_finished = asyncio.Event()
        b_post = {}
        unexpected = []

        async def route_api(route):
            request = route.request
            path = urlparse(request.url).path
            if path == "/api/health":
                await route.fulfill(json={"external_model_configured": False})
            elif path == "/api/metadata":
                await route.fulfill(json={"jurisdictions": []})
            elif request.method == "GET" and path == f"/api/sessions/{a_id}":
                await route.fulfill(json={"session": state_a})
            elif request.method == "GET" and path == f"/api/sessions/{b_id}":
                await route.fulfill(json={"session": state_b})
            elif request.method == "POST" and path == f"/api/sessions/{a_id}/messages":
                a_started.set()
                await release_a.wait()
                await route.fulfill(json={"session": a_after, "external_data_sent": False, "external_model_used": False})
                a_finished.set()
            elif request.method == "POST" and path == f"/api/sessions/{b_id}/messages":
                b_post.update(url=path, body=json.loads(request.post_data or "{}"))
                await route.fulfill(json={"session": b_after, "external_data_sent": False, "external_model_used": False})
            else:
                unexpected.append(f"{request.method} {path}")
                await route.fulfill(status=404, json={"detail": "unexpected route"})

        await self.page.route("**/api/**", route_api)
        await self.page.goto(self.base_url)
        await expect(self.page.locator("#case-title")).to_contain_text(a_id[:8])
        await self._send("A turn")
        await asyncio.wait_for(a_started.wait(), timeout=3)

        await self.page.locator("#case-list").select_option(b_id)
        await self.page.locator("#resume-case").click()
        await expect(self.page.locator("#case-title")).to_contain_text(b_id[:8])
        await expect(self.page.locator("#chat")).to_contain_text("B-only marker")

        release_a.set()
        await asyncio.wait_for(a_finished.wait(), timeout=3)
        await self.page.evaluate("window.__aPostSettled")
        await expect(self.page.locator("#case-title")).to_contain_text(b_id[:8])
        await expect(self.page.locator("#chat")).to_contain_text("B-only marker")
        await expect(self.page.locator("#chat")).not_to_contain_text("A-response marker")

        await self._send("B next turn")
        await expect(self.page.locator("#chat")).to_contain_text("B-next-turn response")
        self.assertEqual(b_post["url"], f"/api/sessions/{b_id}/messages")
        self.assertEqual(b_post["body"]["expected_revision"], 5)
        self.assertEqual(b_post["body"]["text"], "B next turn")
        self.assertEqual(unexpected, [])

    async def test_model_consent_is_optional_and_changes_scope_by_mode(self):
        env = {"MODEL_PROVIDER": "openai-compatible", "MODEL_API_KEY": "test-only", "MODEL_BASE_URL": "https://provider.example/v1", "MODEL_NAME": "test"}
        outbound = []
        message_requests = []

        def capture_message_request(request):
            if request.method == "POST" and "/messages" in request.url:
                message_requests.append(request.post_data_json)

        self.page.on("request", capture_message_request)

        def fake_chat(messages):
            payload = json.loads(messages[1]["content"])
            outbound.append(payload)
            if not payload["sources"]:
                return json.dumps({"claims": []})
            source = payload["sources"][0]
            return json.dumps({"claims": [{"source_id": source["source_id"], "quote": source["excerpt"][:40]}]}, ensure_ascii=False)

        with patch.dict("os.environ", env), \
             patch("app.main.generate_interview_extraction", return_value=[]) as extract, \
             patch("app.main.generate_grounded_answer", wraps=main.generate_grounded_answer) as answer, \
             patch("app.model._chat", side_effect=fake_chat) as provider:
            await self.page.goto(self.base_url)
            await self.page.locator("#start-case").click()
            await expect(self.page.locator("#model-consent-row")).to_be_visible()
            self.assertFalse(await self.page.locator("#model-consent").is_checked())

            await self._send("في شخص يهددني على واتساب")
            await expect(self.page.locator("#status-message")).to_contain_text("تم حفظ الرسالة وتحديث الحالة بالقواعد المحلية.")
            extract.assert_not_called()
            self.assertFalse(await self.page.locator("#model-consent").is_checked())
            self.assertFalse(message_requests[-1]["allow_external_extraction"])
            await self.page.locator("#model-consent").check()
            await self._send("لا")
            extract.assert_called_once_with("لا", "direct_danger")
            await expect(self.page.locator("#status-message")).to_contain_text("نص الرسالة الحالية فقط")
            self.assertFalse(await self.page.locator("#model-consent").is_checked())
            self.assertTrue(message_requests[-1]["allow_external_extraction"])
            self.assertFalse(message_requests[-1]["allow_external"])

            await self.page.locator("#jurisdiction").select_option("AU")
            await expect(self.page.locator("#status-message")).to_contain_text("تم تحديث الاختصاص")
            await self.page.locator("#toggle-mode").click()
            await expect(self.page.locator("#model-consent-copy")).to_contain_text("مقتطفات المصادر المختارة")
            await self._send("كيف أحفظ الأدلة واللقطات؟")
            await expect(self.page.locator("#status-message")).to_contain_text("تم حفظ الرسالة وتحديث الحالة بالقواعد المحلية.")
            answer.assert_not_called()
            provider.assert_not_called()
            self.assertFalse(await self.page.locator("#model-consent").is_checked())
            self.assertFalse(message_requests[-1]["allow_external"])
            self.assertFalse(message_requests[-1]["allow_external_extraction"])

            await self.page.locator("#model-consent").check()
            await self._send("كيف أحفظ الأدلة واللقطات؟")
            await expect(self.page.locator("#status-message")).to_contain_text("نص السؤال الحالي")
            await expect(self.page.locator("#model-consent")).not_to_be_checked()
            self.assertTrue(message_requests[-1]["allow_external"])
            self.assertFalse(message_requests[-1]["allow_external_extraction"])
            answer.assert_called_once()
            provider.assert_called_once()
            question, sources = answer.call_args.args
            self.assertEqual(question, "كيف أحفظ الأدلة واللقطات؟")
            self.assertTrue(sources)
            self.assertTrue(all(source.get("excerpt") for source in sources))
            self.assertTrue(all(source.get("geography") == "AU" for source in sources))
            self.assertEqual(outbound[0]["question"], "كيف أحفظ الأدلة واللقطات؟")
            self.assertNotIn("في شخص يهددني على واتساب", json.dumps(outbound[0], ensure_ascii=False))
            self.assertTrue(all(set(source) == {"source_id", "title", "excerpt"} for source in outbound[0]["sources"]))
            self.assertTrue(all(source["excerpt"] for source in outbound[0]["sources"]))


if __name__ == "__main__":
    unittest.main()
