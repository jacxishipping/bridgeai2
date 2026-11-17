"""Cross-worker rate limiting with Redis, DB, or in-process fallback (Phase 9).

Implementation layer for :mod:`.limiter`, which is the single entry
point for rate limiting (INV-12 / TD-D05, Wave 3B). New code must
import from :mod:`.limiter`; this module's public functions stay intact
so existing callers and tests keep working, but they are the backend,
not the interface.

Backend selection (Wave 2A / PERF-02):

* Redis URL configured and reachable -> Redis sliding window.
* Redis URL configured but unreachable -> in-memory limiter. The DB path
  is NOT used on this path: a per-request DB write under a Redis outage
  would both hammer the shared DB and give a different (single, global)
  limit from the one Redis would enforce. The outage is logged once per
  process; the in-memory buckets are per-worker by necessity.
* No Redis URL configured -> DB limiter when a connection is supplied
  (multi-worker safe), else in-memory.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import threading
import time
from typing import Any

log = logging.getLogger("ai_bridge.durable_ratelimit")

# Wave 5A (rate-limiter lockout, confirmed live 2026-11-17): the Redis
# sliding window must record ONLY admitted requests. The old code did
# ``zadd`` of the current request BEFORE counting, so every rejected
# request refreshed the window and a token that tripped the limit stayed
# 429 forever under sustained load. The Lua script below makes the
# prune -> count -> add-if-under-limit sequence atomic on real Redis
# (one round trip, no interleaving across workers). Clients without
# ``eval`` (old redis-py, test fakes) use the pipeline fallback in
# ``_redis_allow_pipeline`` / ``_async_redis_allow_pipeline``, which
# performs the same admit-only sequence in two round trips (racy under
# concurrent writers, but it can never lock the window out: rejected
# requests are never recorded).
_RATE_LIMIT_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
local ttl = math.max(2, math.floor(window * 2) + 1)
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
  redis.call('EXPIRE', key, ttl)
  return 0
end
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, ttl)
return 1
"""


def _rate_limit_ttl(window: float) -> int:
    return max(2, int(window * 2) + 1)


def _rate_limit_member(now: float) -> str:
    # Unique per request: two requests in the same microsecond must not
    # collapse into one sorted-set member (that would under-count).
    return f"{now:.6f}:{secrets.token_hex(4)}"


def _eval_not_supported(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "unknown command" in msg or "noscript" in msg or "eval" in msg


_RATE_LOCK = threading.Lock()
_MEMORY: dict[str, list[float]] = {}
_SYNC_REDIS = None
_SYNC_REDIS_OK = False
_SYNC_REDIS_FAILED_AT = 0.0
_SYNC_REDIS_FAILED_COOLDOWN_SEC = 10.0
_REDIS_UNREACHABLE_WARNED = False


def _redis_url() -> str:
    return (os.environ.get("REDIS_URL") or os.environ.get("AI_BRIDGE_REDIS_URL") or "").strip()


def redis_required() -> bool:
    """True when a Redis URL is configured (Redis is the required backend)."""
    return bool(_redis_url())


def _note_redis_unreachable(exc: Exception) -> None:
    global _REDIS_UNREACHABLE_WARNED
    if not _REDIS_UNREACHABLE_WARNED:
        _REDIS_UNREACHABLE_WARNED = True
        log.warning(
            "durable_ratelimit: Redis configured (%s) but unreachable (%s); "
            "falling back to in-memory rate limiting for this process",
            "REDIS_URL/AI_BRIDGE_REDIS_URL",
            exc,
        )


def _get_sync_redis():
    global _SYNC_REDIS, _SYNC_REDIS_OK, _SYNC_REDIS_FAILED_AT
    if _SYNC_REDIS_OK:
        return _SYNC_REDIS
    url = _redis_url()
    if not url:
        return None
    if _SYNC_REDIS is None:
        # Negative cache: do not re-attempt a dead Redis on every request.
        if time.time() - _SYNC_REDIS_FAILED_AT < _SYNC_REDIS_FAILED_COOLDOWN_SEC:
            return None
        try:
            import redis

            _SYNC_REDIS = redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
            _SYNC_REDIS.ping()
            _SYNC_REDIS_OK = True
            _SYNC_REDIS_FAILED_AT = 0.0
            return _SYNC_REDIS
        except Exception as exc:
            _SYNC_REDIS = None
            _SYNC_REDIS_OK = False
            _SYNC_REDIS_FAILED_AT = time.time()
            _note_redis_unreachable(exc)
    return None


def backend_name(conn: Any | None = None) -> str:
    """Name of the backend allow() would actually use.

    "redis" when Redis is configured and reachable, "memory" when Redis is
    configured but unreachable (the DB path is not used then), "database"
    when no Redis URL is configured and a connection is supplied, else
    "memory".
    """
    if redis_required():
        return "redis" if _get_sync_redis() is not None else "memory"
    if conn is not None:
        return "database"
    return "memory"


def _memory_allow(key: str, limit: int, window: float) -> bool:
    now = time.time()
    with _RATE_LOCK:
        bucket = [t for t in _MEMORY.get(key, []) if now - t < window]
        if len(bucket) >= limit:
            _MEMORY[key] = bucket
            return False
        bucket.append(now)
        _MEMORY[key] = bucket
        return True


def _redis_allow_pipeline(client: Any, key: str, limit: int, window: float, now: float) -> bool:
    """Admit-only fallback for Redis clients without Lua ``eval``.

    Prune expired entries, count, and only then record the request —
    and only when it is admitted. Two round trips (racy under
    concurrent writers, unlike the Lua path) but never records a
    rejected request, so the window always drains.
    """
    redis_key = f"aibridge:ratelimit:{key}"
    window_start = now - window
    ttl = _rate_limit_ttl(window)
    pipe = client.pipeline()
    pipe.zremrangebyscore(redis_key, "-inf", window_start)
    pipe.zcard(redis_key)
    _, count = pipe.execute()
    if count >= limit:
        # Rejected: refresh the TTL so the key still expires, but do
        # NOT add this request to the window. Via a pipeline: some
        # clients (notably the async fakeredis-style clients used in
        # tests) expose expire only on pipelines, not on the client.
        pipe = client.pipeline()
        pipe.expire(redis_key, ttl)
        pipe.execute()
        return False
    pipe = client.pipeline()
    pipe.zadd(redis_key, {_rate_limit_member(now): now})
    pipe.expire(redis_key, ttl)
    pipe.execute()
    return True


def _redis_allow(key: str, limit: int, window: float) -> bool | None:
    global _SYNC_REDIS, _SYNC_REDIS_OK, _SYNC_REDIS_FAILED_AT
    client = _get_sync_redis()
    if client is None:
        return None
    redis_key = f"aibridge:ratelimit:{key}"
    now = time.time()
    try:
        eval_fn = getattr(client, "eval", None)
        if callable(eval_fn):
            try:
                admitted = eval_fn(_RATE_LIMIT_LUA, 1, redis_key, now, window, limit, _rate_limit_member(now))
            except Exception as exc:
                if _eval_not_supported(exc):
                    return _redis_allow_pipeline(client, key, limit, window, now)
                raise
            return bool(admitted)
        return _redis_allow_pipeline(client, key, limit, window, now)
    except Exception as exc:
        # Redis died mid-request: drop the cached client so the next call
        # re-probes (after the negative-cache cooldown) instead of reusing
        # a dead connection, and warn once per process.
        _SYNC_REDIS = None
        _SYNC_REDIS_OK = False
        _SYNC_REDIS_FAILED_AT = time.time()
        _note_redis_unreachable(exc)
        return None


_DB_ALLOW_LOCK = threading.Lock()


def _db_allow(conn: Any, key: str, limit: int, window: float) -> bool:
    """Atomic DB rate-limit check/increment (F-05-08).

    Serializes in-process callers with ``_DB_ALLOW_LOCK``, uses ``SELECT ...
    FOR UPDATE`` inside a transaction on Postgres (with ``ON CONFLICT DO
    NOTHING`` row seeding so concurrent workers serialize on the row lock),
    and ``BEGIN IMMEDIATE`` on SQLite so cross-process writers cannot race.
    """
    setting_key = f"ratelimit.{key}"
    now = time.time()
    is_pg = (
        getattr(conn, "dialect", None) == "postgres"
        or "pg" in type(conn).__name__.lower()
    )

    def _eval_and_write(raw: Any) -> tuple[bool, str]:
        data: dict[str, Any] = {"t0": now, "count": 0}
        if raw:
            try:
                data = json.loads(str(raw))
            except json.JSONDecodeError:
                data = {"t0": now, "count": 0}
        t0 = float(data.get("t0") or now)
        count = int(data.get("count") or 0)
        if now - t0 >= window:
            t0 = now
            count = 0
        if count >= limit:
            return False, json.dumps({"t0": t0, "count": count})
        count += 1
        return True, json.dumps({"t0": t0, "count": count})

    with _DB_ALLOW_LOCK:
        if is_pg:
            txn = getattr(conn, "transaction", None)
            if txn is not None:
                with txn():
                    conn.execute(
                        "INSERT INTO settings (key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO NOTHING",
                        (setting_key, json.dumps({"t0": now, "count": 0})),
                    )
                    row = conn.execute(
                        "SELECT value FROM settings WHERE key=? FOR UPDATE",
                        (setting_key,),
                    ).fetchone()
                    raw = (row["value"] if hasattr(row, "keys") else row[0]) if row is not None else None
                    admitted, new_val = _eval_and_write(raw)
                    if admitted:
                        conn.execute(
                            "UPDATE settings SET value=? WHERE key=?",
                            (new_val, setting_key),
                        )
                    return admitted
        # SQLite (or generic DB-API connection without transaction())
        raw_conn = getattr(conn, "_conn", conn)
        began = False
        try:
            raw_conn.execute("BEGIN IMMEDIATE")
            began = True
        except Exception as exc:
            if "cannot start a transaction within a transaction" not in str(exc).lower():
                began = False
        try:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (setting_key,)).fetchone()
            raw = (row["value"] if hasattr(row, "keys") else row[0]) if row is not None else None
            admitted, new_val = _eval_and_write(raw)
            if admitted:
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (setting_key, new_val),
                )
            if began:
                raw_conn.commit()
            else:
                conn.commit()
            return admitted
        except Exception:
            if began:
                try:
                    raw_conn.rollback()
                except Exception:
                    pass
            raise


def allow(key: str, limit: int, window: float = 60.0, *, conn: Any | None = None) -> bool:
    """Return True if the request is within the rate limit.

    When a Redis URL is configured, Redis is the only durable backend:
    if it is unreachable the in-memory limiter is used and the DB path
    is skipped entirely (PERF-02). The DB path applies only when no
    Redis URL is configured at all.
    """
    if redis_required():
        redis_ok = _redis_allow(key, limit, window)
        if redis_ok is not None:
            return redis_ok
        return _memory_allow(key, limit, window)
    redis_ok = _redis_allow(key, limit, window)
    if redis_ok is not None:
        return redis_ok
    if conn is not None:
        try:
            return _db_allow(conn, key, limit, window)
        except Exception:
            pass
    return _memory_allow(key, limit, window)


_ASYNC_REDIS = None
_ASYNC_REDIS_OK = False
_ASYNC_REDIS_FAILED_AT = 0.0


async def _get_async_redis():
    """Cached redis.asyncio client, or None when unconfigured/unreachable."""
    global _ASYNC_REDIS, _ASYNC_REDIS_OK, _ASYNC_REDIS_FAILED_AT
    if _ASYNC_REDIS_OK:
        return _ASYNC_REDIS
    url = _redis_url()
    if not url:
        return None
    if _ASYNC_REDIS is None:
        if time.time() - _ASYNC_REDIS_FAILED_AT < _SYNC_REDIS_FAILED_COOLDOWN_SEC:
            return None
        try:
            import redis.asyncio as aioredis

            _ASYNC_REDIS = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=2)
            await _ASYNC_REDIS.ping()
            _ASYNC_REDIS_OK = True
            _ASYNC_REDIS_FAILED_AT = 0.0
            return _ASYNC_REDIS
        except Exception as exc:
            _ASYNC_REDIS = None
            _ASYNC_REDIS_OK = False
            _ASYNC_REDIS_FAILED_AT = time.time()
            _note_redis_unreachable(exc)
    return None


async def _async_redis_allow_pipeline(client: Any, key: str, limit: int, window: float, now: float) -> bool:
    """Async admit-only fallback for Redis clients without Lua ``eval``.

    Same sequence as :func:`_redis_allow_pipeline`: prune, count, and
    record only admitted requests.
    """
    redis_key = f"aibridge:ratelimit:{key}"
    window_start = now - window
    ttl = _rate_limit_ttl(window)
    pipe = client.pipeline()
    pipe.zremrangebyscore(redis_key, "-inf", window_start)
    pipe.zcard(redis_key)
    _, count = await pipe.execute()
    if count >= limit:
        # Rejected: refresh the TTL via a pipeline (see the sync
        # fallback) and do NOT add this request to the window.
        pipe = client.pipeline()
        pipe.expire(redis_key, ttl)
        await pipe.execute()
        return False
    pipe = client.pipeline()
    pipe.zadd(redis_key, {_rate_limit_member(now): now})
    pipe.expire(redis_key, ttl)
    await pipe.execute()
    return True


async def _async_redis_allow(key: str, limit: int, window: float) -> bool | None:
    """Async Redis sliding-window check; None when Redis is unavailable."""
    global _ASYNC_REDIS, _ASYNC_REDIS_OK, _ASYNC_REDIS_FAILED_AT
    client = await _get_async_redis()
    if client is None:
        return None
    redis_key = f"aibridge:ratelimit:{key}"
    now = time.time()
    try:
        eval_fn = getattr(client, "eval", None)
        if callable(eval_fn):
            try:
                admitted = eval_fn(_RATE_LIMIT_LUA, 1, redis_key, now, window, limit, _rate_limit_member(now))
                if asyncio.iscoroutine(admitted):
                    admitted = await admitted
            except Exception as exc:
                if _eval_not_supported(exc):
                    return await _async_redis_allow_pipeline(client, key, limit, window, now)
                raise
            return bool(admitted)
        return await _async_redis_allow_pipeline(client, key, limit, window, now)
    except Exception as exc:
        _ASYNC_REDIS = None
        _ASYNC_REDIS_OK = False
        _ASYNC_REDIS_FAILED_AT = time.time()
        _note_redis_unreachable(exc)
        return None


async def allow_async(key: str, limit: int, window: float = 60.0, *, conn: Any | None = None) -> bool:
    """Async-native version of :func:`allow` (DIST-07).

    Same backend selection as :func:`allow`: Redis when configured and
    reachable, in-memory when Redis is configured but down (the DB path
    is skipped then, PERF-02), DB only when no Redis URL is configured
    at all. The DB path is synchronous and short (one settings row); it
    runs inline rather than in a thread because it is a single indexed
    lookup + upsert.
    """
    if redis_required():
        redis_ok = await _async_redis_allow(key, limit, window)
        if redis_ok is not None:
            return redis_ok
        return _memory_allow(key, limit, window)
    redis_ok = await _async_redis_allow(key, limit, window)
    if redis_ok is not None:
        return redis_ok
    if conn is not None:
        try:
            return _db_allow(conn, key, limit, window)
        except Exception:
            pass
    return _memory_allow(key, limit, window)


def reset_for_tests() -> None:
    """Clear cached clients, once-per-process warnings, and buckets."""
    global _SYNC_REDIS, _SYNC_REDIS_OK, _SYNC_REDIS_FAILED_AT
    global _ASYNC_REDIS, _ASYNC_REDIS_OK, _ASYNC_REDIS_FAILED_AT
    global _REDIS_UNREACHABLE_WARNED
    _SYNC_REDIS = None
    _SYNC_REDIS_OK = False
    _SYNC_REDIS_FAILED_AT = 0.0
    _ASYNC_REDIS = None
    _ASYNC_REDIS_OK = False
    _ASYNC_REDIS_FAILED_AT = 0.0
    _REDIS_UNREACHABLE_WARNED = False
    with _RATE_LOCK:
        _MEMORY.clear()