"""Content-addressed response cache.

When the same artifact is graded twice (rubric re-runs, mid-program
rubric changes), the cache returns the previously-computed result
rather than re-invoking the VLM. Cuts inference cost during repeated
eval runs by an order of magnitude.

Backed by an in-memory dict for simplicity; the production version
would persist to Redis or a Postgres table behind the same interface.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheEntry:
    raw_output: str
    timestamp: float


class ResponseCache:
    """LRU cache keyed by ``(image_hash, prompt_hash)``.

    Parameters
    ----------
    max_entries:
        Maximum number of cached entries. When exceeded, the
        oldest entry is evicted.
    ttl_seconds:
        Optional time-to-live. Entries older than this are treated
        as cache misses. ``None`` disables expiry.
    """

    def __init__(
        self,
        max_entries: int = 10_000,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0 when set")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(image_bytes: bytes, prompt: str) -> str:
        h = hashlib.sha256()
        h.update(image_bytes)
        h.update(b"\x00")
        h.update(prompt.encode("utf-8"))
        return h.hexdigest()

    def get(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        key = self._key(image_bytes, prompt)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if self.ttl_seconds is not None and time.time() - entry.timestamp > self.ttl_seconds:
            del self._store[key]
            self.misses += 1
            return None
        # LRU bump
        self._store.move_to_end(key)
        self.hits += 1
        return entry.raw_output

    def put(self, image_bytes: bytes, prompt: str, raw_output: str) -> None:
        key = self._key(image_bytes, prompt)
        self._store[key] = CacheEntry(raw_output=raw_output, timestamp=time.time())
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._store)
