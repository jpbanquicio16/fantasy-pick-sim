"""Raw-layer cache and polite HTTP fetching.

Every response is stored unchanged as JSON with its fetch timestamp, keyed by source,
URL and parameters. A ``ttl`` of ``None`` means "never refetch" (past seasons).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx


class RateLimiter:
    """Minimum interval between calls, shared across threads."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._last + self.min_interval - now
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


def cache_key(source: str, url: str, params: dict | None) -> str:
    blob = json.dumps({"s": source, "u": url, "p": params or {}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


@dataclass
class RawCache:
    root: Path
    hits: int = 0
    misses: int = 0

    def path(self, source: str, key: str) -> Path:
        return self.root / source / f"{key}.json"

    def get(self, source: str, key: str, ttl: float | None) -> Any | None:
        p = self.path(source, key)
        if not p.exists():
            return None
        with p.open() as f:
            rec = json.load(f)
        if ttl is not None and time.time() - rec["fetched_at"] > ttl:
            return None
        self.hits += 1
        return rec["body"]

    def put(self, source: str, key: str, url: str, params: dict | None, body: Any) -> None:
        p = self.path(source, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump({"url": url, "params": params, "fetched_at": time.time(), "body": body}, f)
        tmp.replace(p)
        self.misses += 1

    def fetch(
        self,
        source: str,
        url: str,
        params: dict | None,
        fetcher: Callable[[], Any],
        ttl: float | None,
    ) -> Any:
        key = cache_key(source, url, params)
        body = self.get(source, key, ttl)
        if body is None:
            body = fetcher()
            self.put(source, key, url, params, body)
        return body


@dataclass
class HttpSource:
    """httpx client with a rate limiter, retries with exponential backoff, and the cache."""

    name: str
    base_url: str
    cache: RawCache
    min_interval: float = 0.1
    retries: int = 4
    timeout: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.min_interval)
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, headers=self.headers)

    def _request(self, path: str, params: dict | None) -> Any:
        last: Exception | None = None
        for attempt in range(self.retries):
            self._limiter.wait()
            try:
                r = self._client.get(path, params=params)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                time.sleep(min(30.0, 0.5 * 2**attempt))
        raise RuntimeError(f"{self.name}: GET {path} failed after {self.retries} attempts") from last

    def get(self, path: str, params: dict | None = None, ttl: float | None = 0) -> Any:
        """``ttl=0`` bypasses the cache (live polling); ``None`` caches forever."""
        if ttl == 0:
            return self._request(path, params)
        return self.cache.fetch(
            self.name, self.base_url + path, params, lambda: self._request(path, params), ttl
        )
