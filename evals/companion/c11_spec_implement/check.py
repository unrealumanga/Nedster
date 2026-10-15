from __future__ import annotations

from pathlib import Path

from evals.companion import helpers

HIDDEN = r'''
from ttlcache import TTLCache

class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t

def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False

# construction
assert raises(ValueError, TTLCache, 0)
assert raises(ValueError, TTLCache, -3)
for bad in (True, False, 1.5, '2', None):
    assert raises(ValueError, TTLCache, bad), bad
assert raises(ValueError, TTLCache, 2, default_ttl=0)
assert raises(ValueError, TTLCache, 2, default_ttl=-1)
c = TTLCache(2)
assert raises(ValueError, c.put, "k", 1, ttl=0)
assert raises(ValueError, c.put, "k", 1, ttl=-5)
assert c.stats() == {"hits": 0, "misses": 0, "evictions": 0, "expirations": 0}

# basic get/put/default
assert c.get("a") is None and c.get("a", 7) == 7
c.put("a", 1)
assert c.get("a") == 1 and "a" in c and len(c) == 1
st = c.stats(); assert st["hits"] == 1 and st["misses"] == 2, st
st["hits"] = 99
assert c.stats()["hits"] == 1, "stats() must return a copy"

# LRU eviction order, get refreshes recency
c = TTLCache(2)
c.put("a", 1); c.put("b", 2)
assert c.get("a") == 1            # a is now most recent
c.put("c", 3)                     # evicts b
assert "b" not in c and "a" in c and "c" in c and len(c) == 2
assert c.stats()["evictions"] == 1
c.put("a", 10)                    # replace refreshes recency: order is c, a
c.put("d", 4)                     # evicts c
assert "c" not in c and c.get("a") == 10 and c.get("d") == 4
assert c.stats()["evictions"] == 2

# __contains__ does not refresh recency nor touch stats
c = TTLCache(2)
c.put("a", 1); c.put("b", 2)
before = c.stats()
assert "a" in c and "zz" not in c
assert c.stats() == before
c.put("c", 3)                     # a is still least recent -> evicted
assert "a" not in c and "b" in c

# expiry boundary and default ttl
clk = Clock()
c = TTLCache(5, default_ttl=10, clock=clk)
c.put("a", 1)
clk.t = 109.999
assert c.get("a") == 1
clk.t = 110.0
assert "a" not in c and len(c) == 0
assert c.get("a") is None
st = c.stats()
assert st["expirations"] == 1 and st["misses"] == 1 and st["hits"] == 1, st
assert c.get("a") is None and c.stats()["expirations"] == 1 and c.stats()["misses"] == 2

# per-entry ttl overrides default; ttl restarts on replace; no-ttl entries never expire
clk = Clock()
c = TTLCache(5, default_ttl=10, clock=clk)
c.put("short", 1, ttl=2); c.put("long", 2, ttl=50); c.put("dflt", 3)
clk.t += 5
assert "short" not in c and "dflt" in c and "long" in c
c.put("dflt", 4)                  # restart ttl at t+5
clk.t += 8
assert c.get("dflt") == 4         # 8 < 10
clk.t += 3
assert c.get("dflt") is None      # 11 >= 10
c2 = TTLCache(2, clock=clk)
c2.put("x", 1)
clk.t += 10**9
assert c2.get("x") == 1

# len counts only live entries; purge_expired
clk = Clock()
c = TTLCache(10, clock=clk)
for i in range(4):
    c.put(i, i, ttl=i + 1)
clk.t += 2.5                       # 0 and 1 are expired
assert len(c) == 2
assert c.purge_expired() == 2
assert c.purge_expired() == 0
assert c.stats()["expirations"] == 2 and len(c) == 2

# expired entries are dropped before live ones when evicting
clk = Clock()
c = TTLCache(2, clock=clk)
c.put("old", 1, ttl=1); c.put("live", 2)
clk.t += 5
c.put("new", 3)
assert "live" in c and "new" in c and "old" not in c
st = c.stats(); assert st["evictions"] == 0 and st["expirations"] == 1, st

# delete / clear
clk = Clock()
c = TTLCache(3, clock=clk)
c.put("a", 1, ttl=1); c.put("b", 2)
assert c.delete("b") is True and c.delete("b") is False and c.delete("nope") is False
clk.t += 2
assert c.delete("a") is False
before = c.stats()
c.put("z", 1); c.clear()
assert len(c) == 0 and "z" not in c
after = c.stats()
assert after == before, (before, after)
c.put("y", 1); assert c.get("y") == 1

# capacity 1 and non-string keys
c = TTLCache(1)
c.put((1, 2), "t"); c.put(3, "u")
assert (1, 2) not in c and c.get(3) == "u"

# Expired-key replacement is an expiration, even without capacity pressure.
clk = Clock()
c = TTLCache(3, clock=clk)
c.put('old', None, ttl=1)
clk.t += 1
c.put('old', 9)
assert c.stats()['expirations'] == 1 and c.get('old') == 9
assert c.stats()['evictions'] == 0

# None/false values are cache hits, and invalid puts leave existing data alone.
c.put('none', None); c.put('false', False)
marker = object()
assert c.get('none', marker) is None and c.get('false', marker) is False
before = c.stats()
assert raises(ValueError, c.put, 'old', 999, ttl=0)
assert c.stats() == before and c.get('old') == 9
clk = Clock()
c = TTLCache(2, clock=clk)
c.put('expired', 1, ttl=1); c.put('live', 2)
clk.t += 2
before = c.stats()
assert c.delete('expired') is False and c.delete('live') is True
assert c.stats() == before
# Clear must preserve nonzero values of every counter.
c.put('a', 1); c.put('b', 2); c.put('c', 3)
c.get('c'); c.get('missing')
c.put('short', 1, ttl=1); clk.t += 2; c.purge_expired()
before = c.stats(); c.clear()
assert c.stats() == before and len(c) == 0
print("ok")
'''


def check(root: Path, answer: str) -> tuple[bool, str]:
    spec = root / "SPEC.md"
    if spec.read_bytes() != helpers.original(__file__, "SPEC.md").read_bytes():
        return False, "SPEC.md was modified"
    ok, detail = helpers.run_py(root, HIDDEN)
    return (True, "hidden battery passed") if ok else (False, detail)
