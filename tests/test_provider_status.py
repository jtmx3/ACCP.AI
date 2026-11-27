import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main, model


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class ProviderStatusTests(unittest.TestCase):
    def test_rules_mode_does_not_claim_a_model_is_configured(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "rules"}, clear=True):
            config = model.provider_config()
            self.assertEqual(config["mode"], "rules")
            self.assertFalse(config["configured"])
            self.assertFalse(model.probe_provider()["service_reachable"])

    def test_ollama_probe_separates_configuration_reachability_and_model_availability(self):
        env = {
            "MODEL_PROVIDER": "ollama",
            "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
            "MODEL_NAME": "fixture:latest",
        }
        response = FakeResponse({"models": [{"name": "fixture:latest", "model": "fixture:latest"}]})
        with patch.dict(os.environ, env, clear=True), patch("app.model.urllib.request.urlopen", return_value=response) as urlopen:
            status = model.probe_provider()
            request = urlopen.call_args.args[0]
            self.assertEqual(request.get_method(), "GET")
            self.assertTrue(request.full_url.endswith("/api/tags"))
        self.assertTrue(status["configured"])
        self.assertTrue(status["service_reachable"])
        self.assertTrue(status["model_available"])
        self.assertEqual(status["models_count"], 1)

    def test_remote_provider_and_remote_ollama_endpoint_are_rejected(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "remote-provider", "MODEL_NAME": "fixture"}, clear=True):
            config = model.provider_config()
            self.assertFalse(config["configured"])
            self.assertFalse(config["valid_provider"])
            self.assertIsNone(config["base_url_host"])
        with patch.dict(os.environ, {"MODEL_PROVIDER": "ollama", "MODEL_BASE_URL": "https://host.example/v1",
                                    "MODEL_NAME": "fixture"}, clear=True), \
             patch("app.model.urllib.request.urlopen") as urlopen:
            config = model.provider_config()
            status = model.probe_provider()
            urlopen.assert_not_called()
        self.assertFalse(config["configured"])
        self.assertEqual(config["failure_code"], "nonlocal_endpoint_disallowed")
        self.assertFalse(status["service_reachable"])

    def test_fixed_generation_diagnostic_records_a_real_accepted_call(self):
        env = {
            "MODEL_PROVIDER": "ollama",
            "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
            "MODEL_NAME": "fixture-model",
            "OLLAMA_REASONING_EFFORT": "low",
        }
        content = json.dumps({"choices": [{"message": {"content": "{\"ok\":true}"}}]})
        request_bodies = []

        def fake_urlopen(request, timeout):
            request_bodies.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse(json.loads(content))

        with patch.dict(os.environ, env, clear=True), patch("app.model.urllib.request.urlopen", side_effect=fake_urlopen):
            result = model.test_provider_generation()
            trace = model.last_call_trace()
        self.assertTrue(result["attempted"])
        self.assertTrue(result["success"])
        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["model"], "fixture-model")
        assert trace is not None
        self.assertTrue(trace["result_accepted"])
        self.assertIsInstance(trace["latency_ms"], (int, float))
        self.assertEqual(request_bodies[0]["reasoning_effort"], "low")
        serialized = json.dumps(request_bodies[0], ensure_ascii=False)
        self.assertIn("اختبار اتصال ثابت", serialized)
        self.assertNotIn("نص حالة المستخدم", serialized)

    def test_status_and_generation_routes_show_only_local_ollama_configuration(self):
        env = {
            "MODEL_PROVIDER": "ollama",
            "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
            "MODEL_NAME": "fixture-model",
        }
        probe = {"configured": True, "service_reachable": True, "model_available": True,
                 "models_count": 1, "failure_code": None, "provider": "ollama",
                 "mode": "local_model", "model": "fixture-model"}
        with patch.dict(os.environ, env, clear=True), TestClient(main.app) as client:
            with patch("app.main.probe_provider", return_value=probe):
                response = client.get("/api/provider/status")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["config"]["configuration_complete"])
            self.assertEqual(response.json()["config"]["provider"], "ollama")
            with patch("app.main.probe_provider", return_value=probe), \
                 patch("app.main.test_provider_generation", return_value={"success": True, "attempted": True,
                     "latency_ms": 12.5, "failure_code": None, "provider": "ollama", "model": "fixture-model"}):
                result = client.post("/api/provider/test", json={})
            self.assertEqual(result.status_code, 200)
            self.assertTrue(result.json()["success"])
            self.assertEqual(result.json()["provider"], "ollama")

    def test_configuration_without_reachable_service_is_not_reported_as_connected(self):
        from urllib.error import URLError

        env = {
            "MODEL_PROVIDER": "ollama",
            "MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
            "MODEL_NAME": "missing-service-model",
        }
        with patch.dict(os.environ, env, clear=True), patch("app.model.urllib.request.urlopen", side_effect=URLError("connection refused")):
            status = model.probe_provider()
        self.assertTrue(status["configured"])
        self.assertFalse(status["service_reachable"])
        self.assertFalse(status["model_available"])
        self.assertEqual(status["failure_code"], "unreachable")


if __name__ == "__main__":
    unittest.main()
