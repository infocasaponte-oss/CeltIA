# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
"""Gateway in front of the local inference server: per-key rate limits and a bounded
inference queue, so only `max_concurrency` requests ever reach the GPU at once."""
import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

# role -> (requests/minute, concurrent requests, max_tokens per request, tokens/minute, tokens/month or None)
DEFAULT_LIMITS = {
    "admin": (600, 4, 4096, 200_000, None),
    "premium": (60, 2, 2048, 40_000, 5_000_000),
    "user": (20, 1, 1024, 10_000, 300_000),
}


class GatewayRejection(Exception):
    def __init__(self, reason: str, retry_after: int):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


class Gateway:
    def __init__(self, max_concurrency: int = 1, queue_wait_seconds: float = 30.0, limits=None, month_usage=None,
                 month_usage_client=None):
        """`month_usage(key_id) -> int` returns tokens consumed so far this calendar month."""
        self.limits = limits or DEFAULT_LIMITS
        self.month_usage = month_usage
        self.month_usage_client = month_usage_client
        self._tokens: dict[int, deque] = defaultdict(deque)
        self.queue_wait_seconds = queue_wait_seconds
        self._gpu = asyncio.Semaphore(max_concurrency)
        self._hits: dict[int, deque] = defaultdict(deque)
        self._active: dict[int, int] = defaultdict(int)
        self._waiting = 0
        self.max_concurrency = max_concurrency

    def limits_for(self, role: str) -> dict:
        rpm, concurrent, max_tokens, tpm, monthly = self.limits.get(role, self.limits["user"])
        return {"requests_per_minute": rpm, "concurrent_requests": concurrent, "max_tokens_per_request": max_tokens,
                "tokens_per_minute": tpm, "monthly_tokens": monthly}

    def tokens_last_minute(self, key_id: int) -> int:
        now = time.monotonic()
        window = self._tokens[key_id]
        while window and now - window[0][0] > 60:
            window.popleft()
        return sum(n for _, n in window)

    def record_tokens(self, key_id, tokens: int, client_key_id=None):
        if tokens <= 0:
            return
        if key_id is not None:
            self._tokens[key_id].append((time.monotonic(), tokens))
        if client_key_id is not None:
            self._tokens[("ck", client_key_id)].append((time.monotonic(), tokens))

    def _check_client_key(self, client_key_id: int, limits: dict):
        """Per-key limits set by the account owner; unset (None) values fall back to the role limits only."""
        ref = ("ck", client_key_id)
        tpm, monthly, rpm = limits.get("tokens_per_minute"), limits.get("monthly_tokens"), limits.get("requests_per_minute")
        if tpm is not None and self.tokens_last_minute(ref) >= tpm:
            raise GatewayRejection(f"Esta clave alcanzó su límite de {tpm} tokens/minuto", 30)
        if monthly is not None and self.month_usage_client is not None and self.month_usage_client(client_key_id) >= monthly:
            raise GatewayRejection(f"Esta clave agotó su cuota mensual de {monthly} tokens", 3600)
        if rpm is not None:
            self._check_rate(ref, rpm)

    def _check_tokens(self, key_id: int, lim: dict):
        if self.tokens_last_minute(key_id) >= lim["tokens_per_minute"]:
            window = self._tokens[key_id]
            wait = max(1, int(60 - (time.monotonic() - window[0][0])) + 1) if window else 5
            raise GatewayRejection(f"Límite de {lim['tokens_per_minute']} tokens/minuto alcanzado", wait)
        monthly = lim["monthly_tokens"]
        if monthly is not None and self.month_usage is not None and self.month_usage(key_id) >= monthly:
            raise GatewayRejection(f"Cuota mensual de {monthly} tokens agotada", 3600)

    def _check_rate(self, key_id: int, rpm: int):
        now = time.monotonic()
        hits = self._hits[key_id]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= rpm:
            raise GatewayRejection(f"Límite de {rpm} peticiones/minuto alcanzado", max(1, int(60 - (now - hits[0])) + 1))
        hits.append(now)

    @asynccontextmanager
    async def slot(self, key: dict):
        """Enforce rate + per-key concurrency, then wait (bounded) for a GPU slot."""
        key_id, role = key.get("id"), key.get("role", "user")
        lim = self.limits_for(role)
        if key_id is not None:
            self._check_tokens(key_id, lim)
            self._check_rate(key_id, lim["requests_per_minute"])
            if key.get("client_key_id") is not None and key.get("client_key_limits"):
                self._check_client_key(key["client_key_id"], key["client_key_limits"])
            if self._active[key_id] >= lim["concurrent_requests"]:
                raise GatewayRejection("Demasiadas peticiones simultáneas con esta clave", 2)
            self._active[key_id] += 1
        self._waiting += 1
        try:
            try:
                await asyncio.wait_for(self._gpu.acquire(), timeout=self.queue_wait_seconds)
            except asyncio.TimeoutError:
                raise GatewayRejection("Servidor saturado, inténtalo de nuevo en unos segundos", 5)
            finally:
                self._waiting -= 1
            try:
                yield
            finally:
                self._gpu.release()
        finally:
            if key_id is not None:
                self._active[key_id] -= 1

    def status(self) -> dict:
        return {"max_concurrency": self.max_concurrency, "waiting": self._waiting}
