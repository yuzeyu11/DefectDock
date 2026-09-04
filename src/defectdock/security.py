"""Local/network security boundary for the DefectDock HTTP service."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

SECURITY_MODE_ENV = "DEFECTDOCK_SECURITY_MODE"
API_TOKEN_ENV = "DEFECTDOCK_API_TOKEN"
MAX_REQUEST_BYTES_ENV = "DEFECTDOCK_MAX_REQUEST_BYTES"
DEFAULT_MAX_REQUEST_BYTES = 512 * 1024 * 1024
PUBLIC_PATHS = frozenset({"/", "/api/health", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class SecurityMode(str, Enum):
    LOCAL = "local"
    NETWORK = "network"


@dataclass(frozen=True)
class SecuritySettings:
    mode: SecurityMode
    audit_log_path: Path
    api_token: str | None = field(default=None, repr=False)
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES

    @classmethod
    def from_sources(
        cls,
        state_dir: str | Path,
        *,
        mode: str | SecurityMode | None = None,
        api_token: str | None = None,
        max_request_bytes: int | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "SecuritySettings":
        environment = os.environ if environ is None else environ
        raw_mode = mode or environment.get(SECURITY_MODE_ENV, SecurityMode.LOCAL.value)
        try:
            selected_mode = raw_mode if isinstance(raw_mode, SecurityMode) else SecurityMode(raw_mode.lower())
        except ValueError as exc:
            raise ValueError("DEFECTDOCK_SECURITY_MODE must be 'local' or 'network'") from exc

        selected_token = api_token if api_token is not None else environment.get(API_TOKEN_ENV)
        if selected_mode is SecurityMode.NETWORK and (
            selected_token is None or len(selected_token.encode("utf-8")) < 32
        ):
            raise ValueError(
                "Network mode requires DEFECTDOCK_API_TOKEN with at least 32 UTF-8 bytes"
            )

        if max_request_bytes is None:
            raw_limit = environment.get(MAX_REQUEST_BYTES_ENV, str(DEFAULT_MAX_REQUEST_BYTES))
            try:
                selected_limit = int(raw_limit)
            except ValueError as exc:
                raise ValueError("DEFECTDOCK_MAX_REQUEST_BYTES must be an integer") from exc
        else:
            selected_limit = max_request_bytes
        if selected_limit < 1024:
            raise ValueError("DEFECTDOCK_MAX_REQUEST_BYTES must be at least 1024")

        return cls(
            mode=selected_mode,
            api_token=selected_token,
            max_request_bytes=selected_limit,
            audit_log_path=Path(state_dir).resolve() / "audit.jsonl",
        )

    @property
    def token_fingerprint(self) -> str | None:
        if self.api_token is None:
            return None
        return hashlib.sha256(self.api_token.encode("utf-8")).hexdigest()[:12]


def validate_bind_host(mode: SecurityMode, host: str) -> None:
    """Refuse externally reachable bind addresses in unauthenticated local mode."""
    normalized = host.strip().strip("[]").lower()
    if mode is SecurityMode.NETWORK:
        return
    if normalized == "localhost":
        return
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as exc:
        raise ValueError(
            "Local mode only accepts localhost or a loopback IP; use --mode network with a token"
        ) from exc
    if not address.is_loopback:
        raise ValueError(
            "Local mode cannot bind to a non-loopback address; use --mode network with a token"
        )


class AuditWriter:
    """Append compact, body-free JSON audit events without logging credentials."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)


class RequestTooLarge(Exception):
    pass


class SecurityBoundaryMiddleware:
    """Enforce network authentication, request limits, request IDs and auditing."""

    def __init__(self, app: ASGIApp, policy: SecuritySettings) -> None:
        self.app = app
        self.policy = policy
        self.audit = AuditWriter(policy.audit_log_path)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", "/"))
        request_id = _request_id(headers.get("x-request-id"))
        client = scope.get("client")
        client_host = str(client[0]) if client else None
        started = time.perf_counter()
        status_code = 500
        response_started = False
        authenticated = (
            self.policy.mode is SecurityMode.LOCAL
            or self._authenticate(headers.get("authorization"))
        )
        scope.setdefault("state", {})["actor"] = self._actor(authenticated)

        async def send_with_metadata(message: Message) -> None:
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                response_headers = MutableHeaders(scope=message)
                response_headers["X-Request-ID"] = request_id
            await send(message)

        if self.policy.mode is SecurityMode.NETWORK and path not in PUBLIC_PATHS and not authenticated:
            await self._reject(
                send_with_metadata,
                401,
                "A valid Bearer token is required in network mode",
                {"WWW-Authenticate": "Bearer"},
            )
            self._audit(method, path, 401, request_id, client_host, False, started)
            return

        content_length = headers.get("content-length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError:
                await self._reject(send_with_metadata, 400, "Invalid Content-Length header")
                self._audit(method, path, 400, request_id, client_host, authenticated, started)
                return
            if declared_size > self.policy.max_request_bytes:
                await self._reject(send_with_metadata, 413, "Request body exceeds the configured limit")
                self._audit(method, path, 413, request_id, client_host, authenticated, started)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.policy.max_request_bytes:
                    raise RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send_with_metadata)
        except RequestTooLarge:
            status_code = 413
            if not response_started:
                await self._reject(send_with_metadata, 413, "Request body exceeds the configured limit")
        finally:
            if method in MUTATING_METHODS or status_code in {401, 403, 413}:
                self._audit(
                    method,
                    path,
                    status_code,
                    request_id,
                    client_host,
                    authenticated,
                    started,
                )

    def _authenticate(self, authorization: str | None) -> bool:
        if authorization is None or self.policy.api_token is None:
            return False
        scheme, separator, credentials = authorization.partition(" ")
        return bool(
            separator
            and scheme.lower() == "bearer"
            and hmac.compare_digest(credentials, self.policy.api_token)
        )

    @staticmethod
    async def _reject(
        send: Send,
        status_code: int,
        detail: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        await JSONResponse({"detail": detail}, status_code=status_code, headers=headers)(
            {"type": "http"}, _empty_receive, send
        )

    def _audit(
        self,
        method: str,
        path: str,
        status_code: int,
        request_id: str,
        client_host: str | None,
        authenticated: bool,
        started: float,
    ) -> None:
        self.audit.write(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "mode": self.policy.mode.value,
                "actor": self._actor(authenticated),
                "client": client_host,
                "method": method,
                "path": path,
                "status": status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )

    def _actor(self, authenticated: bool) -> str:
        return (
            f"token:{self.policy.token_fingerprint}"
            if authenticated and self.policy.mode is SecurityMode.NETWORK
            else "local"
            if authenticated
            else "anonymous"
        )


async def _empty_receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


def _request_id(value: str | None) -> str:
    if value and 1 <= len(value) <= 128 and all(character.isalnum() or character in "-_." for character in value):
        return value
    return uuid4().hex
