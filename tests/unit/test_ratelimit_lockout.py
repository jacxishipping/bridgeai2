"""Wave 5A — rate-limiter lockout regression tests.

Reproduces the bug confirmed live 2026-11-17: the Redis sliding window
recorded rejected requests too (``zadd`` before counting), so under
sustained overload the window never drained and a token that tripped
the limit stayed 429 even after 30 s of complete silence.

Scenario (mirrors the live probe, scaled down): fill the window to the
limit, hammer it with rejected requests, advance the clock past the
window, and assert the next request is admitted. Against the old code
this fails — the rejected attempts keep the window full forever.

The fake Redis implements the exact ``eval`` + pipeline surface the
production code uses, plus a pipeline-only client (no ``eval``) to
prove the fallback path is admit-only too. No live Redis is needed.
"""
from __future__ import annotations

import asyncio
import sys
import time
import types

import pytest

from ai_bridge.control import durable_ratelimit


class FakeBroker:
    def __init__(self):
        self.zsets: dict[str, dict[str, float]] = {}
        self.eval_calls = 0

    def lua_allow(self, key: str, now: float, window: float, limit: int, member: str) -> int:
        """The admit-only semantics of _RATE_LIMIT_LUA, in Python."""
        self.eval_calls += 1
        z = self.zsets.setdefault(key, {})
        cutoff = now - window
        for m in [m for m, s in z.items() if s <= cutoff]:
            del z[m]
        if len(z) >= limit:
            return 0
        z[member] = float(now)
        return 1


class _BasePipeline:
    def __init__(self, broker: FakeBroker):
        self._broker = broker
        self._ops: list[tuple] = []

    def zremrangebyscore(self, key, mn, mx):
        self._ops.append(("zrem", key, mn, mx))
        return self

    def zadd(self, key, mapping):
        self._ops.append(("zadd", key, mapping))
        return self

    def zcard(self, key):
        self._ops.append(("zcard", key))
        return self

    def expire(self, key, ttl):
        self._ops.append(("expire", key, ttl))
        return self

    def _run(self):
        results = []
        for op in self._ops:
            if op[0] == "zrem":
                _, key, mn, mx = op
                z = self._broker.zsets.setdefault(key, {})
                lo = float("-inf") if mn == "-inf" else float(mn)
                hi = float("inf") if mx == "+inf" else float(mx)
                for m in [m for m, s in z.items() if lo <= s <= hi]:
                    del z[m]
                results.append(0)
            elif op[0] == "zadd":
                _, key, mapping = op
                z = self._broker.zsets.setdefault(key, {})
                for m, s in mapping.items():
                    z[m] = float(s)
                results.append(len(mapping))
            elif op[0] == "zcard":
                results.append(len(self._broker.zsets.get(op[1], {})))
            else:
                results.append(True)
        self._ops.clear()
        return results


class FakeSyncPipeline(_BasePipeline):
    def execute(self):
        return self._run()


class FakeAsyncPipeline(_BasePipeline):
    async def execute(self):
        return self._run()


class FakeSyncRedis:
    def __init__(self, broker: FakeBroker, with_eval: bool = True):
        self._broker = broker
        self._with_eval = with_eval

    def ping(self):
        return True

    def pipeline(self):
        return FakeSyncPipeline(self._broker)

    def expire(self, key, ttl):
        return True

    def eval(self, script, numkeys, *keys_and_args):
        if not self._with_eval:
            raise AttributeError("eval not supported by this client")
        key = keys_and_args[0]
        now, window, limit, member = keys_and_args[1:5]
        return self._broker.lua_allow(key, float(now), float(window), int(limit), str(member))


class FakeAsyncRedis:
    def __init__(self, broker: FakeBroker, with_eval: bool = True):
        self._broker = broker
        self._with_eval = with_eval

    async def ping(self):
        return True

    def pipeline(self):
        return FakeAsyncPipeline(self._broker)

    async def expire(self, key, ttl):
        return True

    async def eval(self, script, numkeys, *keys_and_args):
        if not self._with_eval:
            raise AttributeError("eval not supported by this client")
        key = keys_and_args[0]
        now, window, limit, member = keys_and_args[1:5]
        return self._broker.lua_allow(key, float(now), float(window), int(limit), str(member))


class _FakeRedisModule(types.ModuleType):
    def __init__(self, broker: FakeBroker, with_eval: bool = True):
        super().__init__("redis")
        self._broker = broker
        self._with_eval = with_eval

    def from_url(self, url, **kw):
        return FakeSyncRedis(self._broker, self._with_eval)


class _FakeAsyncRedisModule(types.ModuleType):
    def __init__(self, broker: FakeBroker, with_eval: bool = True):
        super().__init__("redis.asyncio")
        self._broker = broker
        self._with_eval = with_eval

    def from_url(self, url, **kw):
        return FakeAsyncRedis(self._broker, self._with_eval)


@pytest.fixture
def lockout_env(monkeypatch):
    """Install the fake Redis + a controllable clock; yield (broker, clock)."""
    broker = FakeBroker()
    clock = [1_000_000.0]

    def _install(with_eval: bool = True):
        monkeypatch.setitem(sys.modules, "redis", _FakeRedisModule(broker, with_eval))
        monkeypatch.setitem(sys.modules, "redis.asyncio", _FakeAsyncRedisModule(broker, with_eval))

    _install(True)
    monkeypatch.setenv("REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setenv("AI_BRIDGE_REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(time, "time", lambda: clock[0])
    durable_ratelimit.reset_for_tests()
    yield broker, clock, _install
    durable_ratelimit.reset_for_tests()


def _hammer_and_drain(allow_fn, broker, clock, key="tok", limit=3, window=60.0):
    # Fill the window exactly to the limit.
    assert [allow_fn(key, limit, window) for _ in range(limit)] == [True] * limit
    # Hammer with rejected requests (live probe: 4,718 rejected).
    assert [allow_fn(key, limit, window) for _ in range(20)] == [False] * 20
    # Rejected requests must never have been recorded: the window holds
    # exactly the admitted entries, no more.
    assert len(broker.zsets[f"aibridge:ratelimit:{key}"]) == limit
    # Silence longer than the window, then the next request must be admitted.
    clock[0] += window + 1
    assert allow_fn(key, limit, window) is True


def test_sync_lockout_window_drains_after_rejected_hammer(lockout_env):
    broker, clock, _install = lockout_env
    _hammer_and_drain(durable_ratelimit.allow, broker, clock)
    # The Lua path was used (atomic check-and-add), not the fallback.
    assert broker.eval_calls > 0


def test_async_lockout_window_drains_after_rejected_hammer(lockout_env):
    broker, clock, _install = lockout_env

    def allow_sync(key, limit, window):
        return asyncio.run(durable_ratelimit.allow_async(key, limit, window))

    _hammer_and_drain(allow_sync, broker, clock)
    assert broker.eval_calls > 0


def test_pipeline_fallback_is_admit_only_without_eval(lockout_env, monkeypatch):
    broker, clock, install = lockout_env
    install(False)
    durable_ratelimit.reset_for_tests()
    assert broker.eval_calls == 0
    _hammer_and_drain(durable_ratelimit.allow, broker, clock, key="tok-fallback")
    # No eval was ever possible on this client; the pipeline fallback
    # still drained the window.
    assert broker.eval_calls == 0


def test_async_pipeline_fallback_is_admit_only_without_eval(lockout_env):
    broker, clock, install = lockout_env
    install(False)
    durable_ratelimit.reset_for_tests()

    def allow_sync(key, limit, window):
        return asyncio.run(durable_ratelimit.allow_async(key, limit, window))

    _hammer_and_drain(allow_sync, broker, clock, key="tok-afallback")
    assert broker.eval_calls == 0


def test_window_slides_as_admitted_entries_age_out(lockout_env):
    broker, clock, _install = lockout_env
    assert durable_ratelimit.allow("slide", 2, 60.0) is True
    assert durable_ratelimit.allow("slide", 2, 60.0) is True
    assert durable_ratelimit.allow("slide", 2, 60.0) is False
    clock[0] += 61
    assert durable_ratelimit.allow("slide", 2, 60.0) is True
    assert durable_ratelimit.allow("slide", 2, 60.0) is True
    assert durable_ratelimit.allow("slide", 2, 60.0) is False


def test_db_allow_is_atomic_under_concurrency(tmp_path):
    """F-05-08: concurrent _db_allow calls must never over-admit past limit."""
    import concurrent.futures
    import sqlite3

    db_path = tmp_path / "rl.db"
    setup_conn = sqlite3.connect(str(db_path))
    setup_conn.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
    setup_conn.commit()
    setup_conn.close()

    def _hit(_i):
        c = sqlite3.connect(str(db_path), timeout=30)
        try:
            return durable_ratelimit._db_allow(c, "concurrent-key", limit=5, window=60.0)
        finally:
            c.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(_hit, range(20)))
    assert results.count(True) == 5
    assert results.count(False) == 15


def test_db_allow_postgres_uses_for_update_in_transaction():
    """F-05-08: Postgres path seeds row with ON CONFLICT DO NOTHING and locks with FOR UPDATE."""
    import contextlib
    import json

    class _FakePgConn:
        dialect = "postgres"

        def __init__(self):
            self.sqls = []
            self.store = {}

        @contextlib.contextmanager
        def transaction(self):
            self.sqls.append("BEGIN")
            yield
            self.sqls.append("COMMIT")

        def execute(self, sql, params=()):
            self.sqls.append(sql)
            class _Res:
                def __init__(self, row):
                    self._row = row
                def fetchone(self):
                    return self._row
            if sql.startswith("INSERT INTO settings"):
                self.store.setdefault(params[0], params[1])
                return _Res(None)
            if "FOR UPDATE" in sql:
                return _Res((self.store[params[0]],))
            if sql.startswith("UPDATE settings"):
                self.store[params[1]] = params[0]
                return _Res(None)
            return _Res(None)

    pg = _FakePgConn()
    assert durable_ratelimit._db_allow(pg, "pg-key", limit=2, window=60.0) is True
    assert durable_ratelimit._db_allow(pg, "pg-key", limit=2, window=60.0) is True
    assert durable_ratelimit._db_allow(pg, "pg-key", limit=2, window=60.0) is False
    assert any("FOR UPDATE" in s for s in pg.sqls)
    assert any("ON CONFLICT(key) DO NOTHING" in s for s in pg.sqls)
    assert json.loads(pg.store["ratelimit.pg-key"])["count"] == 2
