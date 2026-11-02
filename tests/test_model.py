import json
import os
import unittest
from unittest.mock import patch

from app.model import generate_grounded_answer


class OptionalModelTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{
            "id": "AU-TEST",
            "title": "Evidence guide",
            "publisher": "Example publisher",
            "geography": "AU",
            "text": "Preserve relevant records.",
            "url": "https://example.test/evidence",
        }]
        self.env = patch.dict(os.environ, {
            "MODEL_PROVIDER": "openai-compatible",
            "MODEL_API_KEY": "test-only-not-a-secret",
            "MODEL_BASE_URL": "https://provider.example/v1",
            "MODEL_NAME": "test-model",
            "MODEL_TIMEOUT_SECONDS": "1",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_provider_timeout_returns_none_for_local_fallback(self):
        with patch("app.model.urllib.request.urlopen", side_effect=TimeoutError("synthetic timeout")):
            self.assertIsNone(generate_grounded_answer("كيف أحفظ الأدلة؟", self.sources))

    def test_unsupported_citation_numbers_are_removed(self):
        payload = {"choices": [{"message": {"content": "احفظ المستندات [1]. إرشاد آخر [2]."}}]}
        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return json.dumps(payload, ensure_ascii=False).encode("utf-8")

        with patch("app.model.urllib.request.urlopen", return_value=FakeResponse()) as urlopen:
            answer = generate_grounded_answer("كيف أحفظ الأدلة؟", self.sources)
        self.assertEqual(answer, "احفظ المستندات [1]. إرشاد آخر .")
        request = urlopen.call_args.args[0]
        sent = json.loads(request.data.decode("utf-8"))
        self.assertIn("كيف أحفظ الأدلة؟", sent["messages"][1]["content"])
        self.assertNotIn("حادثة المستخدم غير المرسلة", sent["messages"][1]["content"])
        self.assertIn("Preserve relevant records.", sent["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
