from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import asdict, dataclass
from typing import Any

from .config import PolicyConfig
from .types import RiskLevel, ToolRequest


@dataclass(slots=True, frozen=True)
class Capability:
    jti: str
    tool: str
    args_hash: str
    project_root: str
    risk: int
    issued_at: int
    expires_at: int


class CapabilityAuthority:
    """Issues short-lived, single-use capabilities bound to one exact tool request."""

    def __init__(self, secret: bytes | None = None):
        self._secret = secret or secrets.token_bytes(32)
        self._used: set[str] = set()

    @staticmethod
    def _args_hash(arguments: dict[str, Any]) -> str:
        canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def mint(self, request: ToolRequest, ttl_seconds: int = 30) -> str:
        now = int(time.time())
        capability = Capability(
            jti=secrets.token_urlsafe(12),
            tool=request.tool,
            args_hash=self._args_hash(request.arguments),
            project_root=os.path.realpath(request.project_root),
            risk=int(request.risk),
            issued_at=now,
            expires_at=now + ttl_seconds,
        )
        payload = json.dumps(asdict(capability), sort_keys=True, separators=(",", ":")).encode()
        signature = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return f"{base64.urlsafe_b64encode(payload).decode().rstrip('=')}.{base64.urlsafe_b64encode(signature).decode().rstrip('=')}"

    @staticmethod
    def _decode(value: str) -> bytes:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

    def verify(self, token: str, request: ToolRequest, consume: bool = True) -> Capability:
        try:
            payload_part, signature_part = token.split(".", 1)
            payload = self._decode(payload_part)
            signature = self._decode(signature_part)
            expected = hmac.new(self._secret, payload, hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected):
                raise PermissionError("capability signature mismatch")
            value = json.loads(payload)
            capability = Capability(**value)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise PermissionError("malformed capability") from exc

        if capability.expires_at < int(time.time()):
            raise PermissionError("capability expired")
        if capability.jti in self._used:
            raise PermissionError("capability already consumed")
        if capability.tool != request.tool:
            raise PermissionError("capability tool mismatch")
        if capability.args_hash != self._args_hash(request.arguments):
            raise PermissionError("capability arguments mismatch")
        if capability.project_root != os.path.realpath(request.project_root):
            raise PermissionError("capability project mismatch")
        if capability.risk != int(request.risk):
            raise PermissionError("capability risk mismatch")
        if consume:
            self._used.add(capability.jti)
        return capability


class PolicyEngine:
    LEVELS = {
        "read": RiskLevel.READ,
        "write": RiskLevel.WRITE,
        "execute": RiskLevel.EXECUTE,
        "destructive": RiskLevel.DESTRUCTIVE,
        "external": RiskLevel.EXTERNAL,
    }

    def __init__(self, config: PolicyConfig):
        self.config = config

    def evaluate(self, request: ToolRequest) -> tuple[bool, str]:
        rendered = json.dumps(request.arguments, sort_keys=True).lower()
        for fragment in self.config.denied_fragments:
            if fragment.lower() in rendered:
                return False, f"denied command fragment: {fragment}"
        ceiling = (
            self.LEVELS.get(self.config.max_auto_risk.lower(), RiskLevel.WRITE)
            if self.config.autonomous
            else RiskLevel.READ
        )
        if request.risk > ceiling:
            return False, f"risk {request.risk.name.lower()} exceeds autonomous ceiling {ceiling.name.lower()}"
        if request.risk >= RiskLevel.EXECUTE:
            command = request.arguments.get("command") or []
            executable = command[0] if isinstance(command, list) and command else ""
            if executable and executable not in self.config.allowed_commands:
                return False, f"executable is not allowlisted: {executable}"
        return True, "allowed by project-scoped policy"
