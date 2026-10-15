import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from defectdock.rate_limits import RateLimitSettings, RequestRateLimiter
from defectdock.security import SecurityBoundaryMiddleware, SecurityMode, SecuritySettings


class RateLimiterTests(unittest.TestCase):
    def test_refill_and_retry_delay_use_a_monotonic_clock(self):
        now = [100.0]
        limiter = RequestRateLimiter(RateLimitSettings(requests=2), clock=lambda: now[0])
        self.assertEqual(limiter.acquire("requests"), 0)
        self.assertEqual(limiter.acquire("requests"), 0)
        self.assertEqual(limiter.acquire("requests"), 30)
        now[0] += 15
        self.assertEqual(limiter.acquire("requests"), 15)
        now[0] += 15
        self.assertEqual(limiter.acquire("requests"), 0)
        self.assertEqual(limiter.acquire("requests"), 30)

    def test_rejected_write_does_not_consume_the_read_budget(self):
        limiter = RequestRateLimiter(
            RateLimitSettings(requests=3, mutations=1), clock=lambda: 0.0
        )
        self.assertEqual(limiter.acquire("requests", "mutations"), 0)
        self.assertEqual(limiter.acquire("requests", "mutations"), 60)
        self.assertEqual(limiter.acquire("requests"), 0)
        self.assertEqual(limiter.acquire("requests"), 0)
        self.assertEqual(limiter.acquire("requests"), 20)

    def test_idle_refill_never_exceeds_capacity(self):
        now = [0.0]
        limiter = RequestRateLimiter(RateLimitSettings(requests=1), clock=lambda: now[0])
        limiter.acquire("requests")
        now[0] = 6000.0
        self.assertEqual(limiter.acquire("requests"), 0)
        self.assertEqual(limiter.acquire("requests"), 60)

    def test_parallel_admission_cannot_overdraw_the_budget(self):
        limiter = RequestRateLimiter(RateLimitSettings(requests=10), clock=lambda: 0.0)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: limiter.acquire("requests"), range(80)))
        self.assertEqual(results.count(0), 10)
        self.assertEqual(results.count(6), 70)

    def test_environment_limits_are_validated_and_explicit_policy_wins(self):
        policy = RateLimitSettings.from_env({"DEFECTDOCK_REQUESTS_PER_MINUTE": "42"})
        self.assertEqual(policy.requests, 42)
        for invalid in ("0", "-1", "1.5", "abc", "1000001"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "REQUESTS_PER_MINUTE"):
                RateLimitSettings.from_env({"DEFECTDOCK_REQUESTS_PER_MINUTE": invalid})
        with self.assertRaises(ValueError):
            RateLimitSettings(requests=True)
        with tempfile.TemporaryDirectory() as root:
            selected = SecuritySettings.from_sources(
                root, rate_limits=policy, environ={"DEFECTDOCK_REQUESTS_PER_MINUTE": "invalid"}
            )
        self.assertEqual(selected.rate_limits.requests, 42)


class RateLimitMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.token = "a-valid-network-token-for-rate-limit-tests"
        self.now = 0.0
        self.hits = []
        self.body_reads = 0
        self.policy = SecuritySettings.from_sources(
            self.temp_dir.name,
            mode=SecurityMode.NETWORK,
            api_token=self.token,
            rate_limits=RateLimitSettings(requests=2, mutations=1, auth_failures=2),
            environ={},
        )

        async def app(scope, receive, send):
            self.hits.append(scope["path"])
            if scope["method"] == "POST":
                await receive()
                self.body_reads += 1
            await JSONResponse({"ok": True})(scope, receive, send)

        self.middleware = SecurityBoundaryMiddleware(app, self.policy)
        self.middleware.rate_limiter = RequestRateLimiter(
            self.policy.rate_limits, clock=lambda: self.now
        )
        self.client = TestClient(
            self.middleware, headers={"Authorization": f"Bearer {self.token}"}
        )

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_request_limit_returns_retry_header_request_id_and_body_free_audit(self):
        self.assertEqual(self.client.get("/api/one").status_code, 200)
        self.assertEqual(self.client.get("/api/two").status_code, 200)
        limited = self.client.get("/api/three", headers={"X-Request-ID": "limited-request"})
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(limited.headers["retry-after"], "30")
        self.assertEqual(limited.headers["x-request-id"], "limited-request")
        self.assertEqual(limited.headers["cache-control"], "no-store")
        self.assertEqual(limited.json()["retry_after_seconds"], 30)
        self.assertEqual(len(self.hits), 2)
        audit = self.policy.audit_log_path.read_text(encoding="utf-8")
        self.assertEqual(len(audit.splitlines()), 1)
        self.assertEqual(json.loads(audit)["status"], 429)
        self.assertNotIn(self.token, audit)
        self.now = 30.0
        self.assertEqual(self.client.get("/api/three").status_code, 200)

    def test_rejected_write_never_reaches_handler_or_reads_upload(self):
        self.assertEqual(self.client.post("/api/upload", content=b"first").status_code, 200)
        limited = self.client.post("/api/upload", content=b"private-upload-body")
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(limited.headers["retry-after"], "60")
        self.assertEqual(self.body_reads, 1)
        self.assertEqual(self.client.get("/api/status").status_code, 200)
        self.assertNotIn(
            "private-upload-body", self.policy.audit_log_path.read_text(encoding="utf-8")
        )

    def test_auth_failure_budget_does_not_block_valid_credentials(self):
        for _ in range(2):
            self.assertEqual(
                self.client.get("/api/data", headers={"Authorization": "Bearer wrong"}).status_code,
                401,
            )
        limited = self.client.get("/api/data", headers={"Authorization": "Bearer other-wrong"})
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(self.client.get("/api/data").status_code, 200)
        self.assertEqual(self.hits, ["/api/data"])

    def test_path_and_forwarded_header_changes_cannot_reset_budget(self):
        for index in range(2):
            self.client.get(f"/api/item/{index}", headers={"X-Forwarded-For": f"192.0.2.{index}"})
        limited = self.client.get("/api/a-new-path", headers={"X-Forwarded-For": "198.51.100.4"})
        self.assertEqual(limited.status_code, 429)

    def test_public_health_and_docs_remain_available_without_consuming_budget(self):
        self.client.get("/api/data")
        self.client.get("/api/data")
        for path in ("/api/health", "/docs", "/openapi.json"):
            for _ in range(3):
                self.assertEqual(self.client.get(path, headers={"Authorization": ""}).status_code, 200)
        self.assertEqual(self.client.get("/api/data").status_code, 429)

    def test_local_mode_preserves_unlimited_workstation_requests(self):
        self.middleware.policy = SecuritySettings.from_sources(
            Path(self.temp_dir.name), mode=SecurityMode.LOCAL, environ={}
        )
        for _ in range(5):
            self.assertEqual(self.client.post("/api/data", content=b"test").status_code, 200)


if __name__ == "__main__":
    unittest.main()
