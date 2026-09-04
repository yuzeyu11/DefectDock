import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from defectdock.api import create_app
from defectdock.security import SecurityMode, SecuritySettings, validate_bind_host
from defectdock.settings import RuntimeSettings


class SecuritySettingsTests(unittest.TestCase):
    def test_local_mode_is_default_and_only_allows_loopback_binding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = SecuritySettings.from_sources(temp_dir, environ={})
        self.assertEqual(settings.mode, SecurityMode.LOCAL)
        validate_bind_host(settings.mode, "localhost")
        validate_bind_host(settings.mode, "127.0.0.1")
        validate_bind_host(settings.mode, "::1")
        with self.assertRaisesRegex(ValueError, "non-loopback"):
            validate_bind_host(settings.mode, "0.0.0.0")

    def test_network_mode_requires_a_strong_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "at least 32"):
                SecuritySettings.from_sources(
                    temp_dir,
                    environ={"DEFECTDOCK_SECURITY_MODE": "network", "DEFECTDOCK_API_TOKEN": "short"},
                )
            settings = SecuritySettings.from_sources(
                temp_dir,
                environ={
                    "DEFECTDOCK_SECURITY_MODE": "network",
                    "DEFECTDOCK_API_TOKEN": "x" * 32,
                    "DEFECTDOCK_MAX_REQUEST_BYTES": "4096",
                },
            )
        self.assertEqual(settings.mode, SecurityMode.NETWORK)
        self.assertEqual(settings.max_request_bytes, 4096)
        self.assertNotIn("x" * 32, repr(settings))
        validate_bind_host(settings.mode, "0.0.0.0")


class NetworkSecurityApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.token = "network-test-token-which-is-long-enough"
        runtime = RuntimeSettings.from_sources(self.root)
        security = SecuritySettings.from_sources(
            runtime.state_dir,
            mode=SecurityMode.NETWORK,
            api_token=self.token,
            max_request_bytes=1024,
        )
        self.audit_path = security.audit_log_path
        self.app = create_app(
            runtime_settings=runtime,
            security_settings=security,
            training_enabled=False,
        )
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.state.jobs.shutdown(wait=True)
        self.temp_dir.cleanup()

    def test_health_is_public_but_other_endpoints_require_bearer_token(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["security_mode"], "network")
        self.assertTrue(health.json()["authentication_required"])

        missing = self.client.get("/api/datasets")
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.headers["www-authenticate"], "Bearer")
        wrong = self.client.get(
            "/api/datasets", headers={"Authorization": "Bearer definitely-wrong"}
        )
        self.assertEqual(wrong.status_code, 401)
        accepted = self.client.get(
            "/api/datasets",
            headers={"Authorization": f"Bearer {self.token}", "X-Request-ID": "test-request-1"},
        )
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.headers["x-request-id"], "test-request-1")

    def test_request_limit_and_audit_log_are_enforced_without_leaking_token(self):
        oversized = self.client.post(
            "/api/datasets",
            content=b"x" * 1025,
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(oversized.status_code, 413)

        rejected = self.client.post("/api/datasets")
        self.assertEqual(rejected.status_code, 401)
        lines = [json.loads(line) for line in self.audit_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([line["status"] for line in lines], [413, 401])
        self.assertTrue(lines[0]["actor"].startswith("token:"))
        self.assertEqual(lines[1]["actor"], "anonymous")
        self.assertNotIn(self.token, self.audit_path.read_text(encoding="utf-8"))

    def test_openapi_describes_network_authentication(self):
        schema = self.client.get("/openapi.json").json()
        self.assertEqual(
            schema["components"]["securitySchemes"]["BearerAuth"],
            {"type": "http", "scheme": "bearer"},
        )
        self.assertEqual(schema["paths"]["/api/health"]["get"]["security"], [])
        self.assertEqual(
            schema["paths"]["/api/datasets"]["get"]["security"],
            [{"BearerAuth": []}],
        )


if __name__ == "__main__":
    unittest.main()
