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

    async def test_chat_pause_resume_correction_confirmation_and_download(self):
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
        await expect(self.page.locator("#facts-list")).to_contain_text("منفي صراحة بحسب الإفادة")
        self.assertTrue(await self.page.locator("#confirm-summary").is_enabled())

        await self._send("تصحيح: بدأت الواقعة أمس")
        await expect(self.page.locator("#facts-list")).to_contain_text("أمس")
        self.assertTrue(await self.page.locator("#confirm-summary").is_enabled())
        await self.page.locator("#confirm-summary").click()
        await expect(self.page.locator("#confirmation-state")).to_contain_text("تم تأكيد")

        async with self.page.expect_download() as download_event:
            await self.page.locator("#download-report").click()
        download = await download_event.value
        target = Path(self.tmp.name) / download.suggested_filename
        await download.save_as(target)
        self.assertTrue(target.exists())
        self.assertIn("بدأت الواقعة أمس", target.read_text(encoding="utf-8"))

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
        with patch.dict("os.environ", env), \
             patch("app.main.generate_interview_extraction", return_value=[]) as extract, \
             patch("app.main.generate_grounded_answer", return_value=None) as answer:
            await self.page.goto(self.base_url)
            await self.page.locator("#start-case").click()
            await expect(self.page.locator("#model-consent-row")).to_be_visible()
            self.assertFalse(await self.page.locator("#model-consent").is_checked())

            await self._send("في شخص يهددني على واتساب")
            extract.assert_not_called()
            await self.page.locator("#model-consent").check()
            await self._send("لا")
            extract.assert_called_once_with("لا", "direct_danger")
            await expect(self.page.locator("#status-message")).to_contain_text("نص الرسالة الحالية فقط")
            self.assertFalse(await self.page.locator("#model-consent").is_checked())

            await self.page.locator("#jurisdiction").select_option("AU")
            await expect(self.page.locator("#status-message")).to_contain_text("تم تحديث الاختصاص")
            await self.page.locator("#toggle-mode").click()
            await expect(self.page.locator("#model-consent-copy")).to_contain_text("مقتطفات المصادر المختارة")
            await self.page.locator("#model-consent").check()
            await self._send("كيف أحفظ الأدلة واللقطات؟")
            answer.assert_called_once()
            question, sources = answer.call_args.args
            self.assertEqual(question, "كيف أحفظ الأدلة واللقطات؟")
            self.assertTrue(sources)
            self.assertTrue(all(source.get("excerpt") for source in sources))
            await expect(self.page.locator("#status-message")).to_contain_text("نص السؤال الحالي")
            self.assertFalse(await self.page.locator("#model-consent").is_checked())


if __name__ == "__main__":
    unittest.main()
