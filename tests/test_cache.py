"""Tests for ResponseCache."""

import time

import pytest

from vlm_grader.cache import ResponseCache


def test_get_miss_returns_none():
    c = ResponseCache()
    assert c.get(b"img", "p") is None
    assert c.misses == 1


def test_put_then_get_hit():
    c = ResponseCache()
    c.put(b"img", "p", "result")
    assert c.get(b"img", "p") == "result"
    assert c.hits == 1
    assert c.misses == 0


def test_different_inputs_dont_collide():
    c = ResponseCache()
    c.put(b"img1", "p", "r1")
    c.put(b"img2", "p", "r2")
    c.put(b"img1", "q", "r3")
    assert c.get(b"img1", "p") == "r1"
    assert c.get(b"img2", "p") == "r2"
    assert c.get(b"img1", "q") == "r3"


def test_lru_eviction():
    c = ResponseCache(max_entries=2)
    c.put(b"a", "p", "1")
    c.put(b"b", "p", "2")
    c.put(b"c", "p", "3")  # should evict (b"a", "p")
    assert c.get(b"a", "p") is None
    assert c.get(b"b", "p") == "2"
    assert c.get(b"c", "p") == "3"


def test_ttl_expiry():
    c = ResponseCache(ttl_seconds=0.05)
    c.put(b"img", "p", "fresh")
    assert c.get(b"img", "p") == "fresh"
    time.sleep(0.1)
    assert c.get(b"img", "p") is None  # expired


def test_lru_bumps_on_get():
    c = ResponseCache(max_entries=2)
    c.put(b"a", "p", "1")
    c.put(b"b", "p", "2")
    # Access "a" so it becomes most recent.
    assert c.get(b"a", "p") == "1"
    c.put(b"c", "p", "3")  # should evict "b" now, not "a"
    assert c.get(b"a", "p") == "1"
    assert c.get(b"b", "p") is None


def test_hit_rate():
    c = ResponseCache()
    c.put(b"a", "p", "1")
    c.get(b"a", "p")
    c.get(b"a", "p")
    c.get(b"missing", "p")
    assert c.hit_rate() == pytest.approx(2 / 3)


def test_invalid_max_entries():
    with pytest.raises(ValueError):
        ResponseCache(max_entries=0)


def test_invalid_ttl():
    with pytest.raises(ValueError):
        ResponseCache(ttl_seconds=0)
