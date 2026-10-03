import json

import pytest

from core.cache import CacheStats, DiskCache, canonical_json, hash_payload


@pytest.fixture()
def cache(tmp_path):
    return DiskCache("llm", root=tmp_path, stats=CacheStats())


def test_key_is_order_independent():
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})


def test_key_separates_different_payloads():
    assert hash_payload({"prompt": "x"}) != hash_payload({"prompt": "y"})


def test_canonical_json_is_compact_and_sorted():
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'


def test_compute_runs_once_then_hits(cache):
    calls = []

    def compute():
        calls.append(1)
        return {"verdict": "Supported"}

    payload = {"model": "m", "prompt": "is the sky blue?"}
    first = cache.get_or_compute(payload, compute)
    second = cache.get_or_compute(payload, compute)

    assert first == second == {"verdict": "Supported"}
    assert len(calls) == 1, "a repeat call with identical inputs must not recompute"
    assert cache.stats.hits == 1 and cache.stats.misses == 1


def test_different_payloads_both_compute(cache):
    cache.get_or_compute({"prompt": "a"}, lambda: 1)
    cache.get_or_compute({"prompt": "b"}, lambda: 2)
    assert cache.stats.misses == 2 and cache.stats.hits == 0


def test_record_stores_payload_for_inspection(cache, tmp_path):
    payload = {"url": "https://example.org"}
    cache.set(payload, "<html/>")
    path = cache.path_for(cache.key_for(payload))
    record = json.loads(path.read_text())
    assert record["payload"] == payload
    assert record["value"] == "<html/>"
    assert record["namespace"] == "llm"
    assert record["created_at"]


def test_truncated_record_is_treated_as_miss(cache):
    payload = {"prompt": "a"}
    cache.set(payload, "value")
    cache.path_for(cache.key_for(payload)).write_text("{not json")
    assert cache.get_or_compute(payload, lambda: "recomputed") == "recomputed"


def test_disabled_cache_always_computes(tmp_path):
    cache = DiskCache("llm", root=tmp_path, enabled=False, stats=CacheStats())
    calls = []
    for _ in range(2):
        cache.get_or_compute({"p": 1}, lambda: calls.append(1))
    assert len(calls) == 2
    assert cache.size() == 0


def test_persists_across_instances(tmp_path):
    a = DiskCache("search", root=tmp_path, stats=CacheStats())
    a.set({"q": "iit bhilai"}, ["hit"])
    b = DiskCache("search", root=tmp_path, stats=CacheStats())
    assert b.get({"q": "iit bhilai"}) == ["hit"]


def test_namespaces_do_not_collide(tmp_path):
    stats = CacheStats()
    llm = DiskCache("llm", root=tmp_path, stats=stats)
    search = DiskCache("search", root=tmp_path, stats=stats)
    payload = {"x": 1}
    llm.set(payload, "from-llm")
    assert search.get(payload) is None
    assert llm.get(payload) == "from-llm"


def test_stats_track_namespaces(tmp_path):
    stats = CacheStats()
    c = DiskCache("fetch", root=tmp_path, stats=stats)
    c.get_or_compute({"u": 1}, lambda: "page")
    c.get_or_compute({"u": 1}, lambda: "page")
    assert stats.by_namespace["fetch"] == {"hits": 1, "misses": 1, "writes": 1}
    assert stats.hit_rate == 0.5


def test_cached_decorator_memoises(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIMEM_CACHE_DIR", str(tmp_path))
    import importlib

    import core.cache as cache_mod
    import core.paths as paths_mod

    importlib.reload(paths_mod)
    importlib.reload(cache_mod)

    calls = []

    @cache_mod.cached("search")
    def search(query, k=5):
        calls.append(query)
        return [f"{query}-{k}"]

    assert search("abc") == ["abc-5"]
    assert search("abc") == ["abc-5"]
    assert len(calls) == 1
    assert search("abc", k=3) == ["abc-3"]
    assert len(calls) == 2
