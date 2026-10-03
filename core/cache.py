"""Hashed on-disk cache for every LLM call, search and page fetch (engineering rule 1).

Every cacheable call is keyed by sha256 of a canonical JSON encoding of its inputs,
so a repeat call with identical inputs never hits the network. An uncached repeat
call is a bug, and `CacheStats` exists so a run can assert that.

Layout: ``cache/<namespace>/<hash[:2]>/<hash>.json``, one record per call::

    {"key": ..., "namespace": ..., "created_at": ..., "payload": {...}, "value": ...}

The payload is stored alongside the value so a cache entry can be inspected and
re-keyed by hand when a prompt changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.paths import CACHE_DIR

_MISS = object()


def canonical_json(payload: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    by_namespace: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, namespace: str, event: str) -> None:
        setattr(self, event, getattr(self, event) + 1)
        ns = self.by_namespace.setdefault(namespace, {"hits": 0, "misses": 0, "writes": 0})
        ns[event] += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def reset(self) -> None:
        self.hits = self.misses = self.writes = 0
        self.by_namespace = {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hit_rate, 4),
            "by_namespace": self.by_namespace,
        }


STATS = CacheStats()


class DiskCache:
    """A namespaced cache. One instance per kind of call: ``llm``, ``search``, ``fetch``."""

    def __init__(
        self,
        namespace: str,
        root: Path | None = None,
        enabled: bool = True,
        stats: CacheStats | None = None,
    ) -> None:
        self.namespace = namespace
        self.root = (root or CACHE_DIR) / namespace
        self.enabled = enabled
        self.stats = stats if stats is not None else STATS
        self._lock = threading.Lock()

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def key_for(self, payload: Any) -> str:
        return hash_payload(payload)

    def get(self, payload: Any, default: Any = None) -> Any:
        value = self._read(self.key_for(payload))
        return default if value is _MISS else value

    def contains(self, payload: Any) -> bool:
        return self.path_for(self.key_for(payload)).exists()

    def set(self, payload: Any, value: Any, meta: dict[str, Any] | None = None) -> str:
        key = self.key_for(payload)
        self._write(key, payload, value, meta)
        return key

    def get_or_compute(
        self,
        payload: Any,
        compute: Callable[[], Any],
        meta: dict[str, Any] | None = None,
    ) -> Any:
        """The one entry point callers should use. `compute` runs only on a miss."""
        key = self.key_for(payload)
        if self.enabled:
            cached = self._read(key)
            if cached is not _MISS:
                self.stats.record(self.namespace, "hits")
                return cached
        self.stats.record(self.namespace, "misses")
        value = compute()
        if self.enabled:
            self._write(key, payload, value, meta)
        return value

    def _read(self, key: str) -> Any:
        if not self.enabled:
            return _MISS
        path = self.path_for(key)
        if not path.exists():
            return _MISS
        try:
            with path.open(encoding="utf-8") as fh:
                return json.load(fh)["value"]
        except (json.JSONDecodeError, KeyError, OSError):
            # A truncated record (killed mid-write) is treated as a miss and overwritten.
            return _MISS

    def _write(self, key: str, payload: Any, value: Any, meta: dict[str, Any] | None) -> None:
        record = {
            "key": key,
            "namespace": self.namespace,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "payload": payload,
            "value": value,
        }
        if meta:
            record["meta"] = meta
        path = self.path_for(key)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, ensure_ascii=False, default=str)
                os.replace(tmp, path)  # atomic, so a crash never leaves a half record
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        self.stats.record(self.namespace, "writes")

    def size(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json"))

    def clear(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)


_CACHES: dict[str, DiskCache] = {}
_CACHES_LOCK = threading.Lock()


def get_cache(namespace: str) -> DiskCache:
    """Shared cache per namespace. Set ``VERIMEM_DISABLE_CACHE=1`` to bypass (tests only)."""
    with _CACHES_LOCK:
        if namespace not in _CACHES:
            enabled = os.environ.get("VERIMEM_DISABLE_CACHE", "") not in {"1", "true", "True"}
            _CACHES[namespace] = DiskCache(namespace, enabled=enabled)
        return _CACHES[namespace]


def cached(namespace: str, key_fn: Callable[..., Any] | None = None) -> Callable:
    """Decorator form. By default the key is the function name plus its arguments.

    >>> @cached("search")
    ... def search(query: str, k: int = 5): ...
    """

    def decorate(fn: Callable) -> Callable:
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            payload = (
                key_fn(*args, **kwargs)
                if key_fn
                else {
                    "fn": fn.__qualname__,
                    "args": list(args),
                    "kwargs": kwargs,
                }
            )
            return get_cache(namespace).get_or_compute(payload, lambda: fn(*args, **kwargs))

        wrapper.cache_payload = key_fn or (
            lambda *a, **k: {"fn": fn.__qualname__, "args": list(a), "kwargs": k}
        )
        wrapper.cache_namespace = namespace
        return wrapper

    return decorate
