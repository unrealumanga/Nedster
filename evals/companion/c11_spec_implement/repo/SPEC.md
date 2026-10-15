# TTLCache specification

`ttlcache.py` must provide one class, `TTLCache`, an in-memory cache with a size
limit (least-recently-used eviction) and optional per-entry expiry. Standard
library only. Not thread-safe.

## Construction

`TTLCache(capacity, default_ttl=None, clock=time.monotonic)`

* `capacity`: maximum number of live entries; an int >= 1 (not bool), otherwise `ValueError`.
* `default_ttl`: seconds an entry lives unless overridden; `None` means no expiry.
  A ttl that is not `None` must be > 0, otherwise `ValueError`.
* `clock`: zero-argument callable returning the current time in seconds. All
  time decisions use only this callable (tests pass a fake clock).

## Methods

* `put(key, value, ttl=None)`: store the entry. `ttl=None` means "use `default_ttl`".
  An explicit ttl must be > 0 (`ValueError` otherwise). Putting an existing key
  replaces its value, restarts its ttl and makes it the most recently used.
  Replacing an expired entry counts its removal as an expiration.
  If the cache is then over capacity, evict least recently used entries until it
  fits (expired entries are removed first and count as expirations, not evictions).
* `get(key, default=None)`: return the value, or `default` if the key is absent
  or expired. A hit makes the key the most recently used. An expired entry is
  removed when found.
* `__contains__(key)`: True if the key is present and not expired. Does not
  change recency and does not touch the statistics.
* `__len__()`: number of entries that are not expired; does not remove entries
  or change statistics. `__contains__` likewise leaves expired entries stored.
* `delete(key)`: remove the key; return True if it was present and live, else
  False. Removing an expired entry returns False.
* `clear()`: remove everything. Statistics are kept.
* `purge_expired()`: remove all expired entries, return how many were removed.
* `stats()`: a new dict with integer counters
  `{"hits", "misses", "evictions", "expirations"}`.
  * `get` on a live key: `hits += 1`.
  * `get` on an absent key: `misses += 1`.
  * `get` on an expired key: `misses += 1` and `expirations += 1` (it is removed).
  * Capacity eviction of a live entry: `evictions += 1`.
  * Every expired entry removed by `get`, `put` or `purge_expired`: `expirations += 1`.
  * `delete` and `clear` change no counter.

## Expiry rule

An entry stored at time `t0` with ttl `d` is expired when `clock() >= t0 + d`
(at exactly `t0 + d` it is already expired).
