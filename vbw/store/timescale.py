"""Tiger Data (TimescaleDB) store: a batched, non-blocking writer for runs, frames and verdicts.

    runner --record_*()--> in-memory queue --background task, every 0.5 s or 500 rows--> COPY into hypertables

`record_*` only appends to a bounded deque, so the runner never waits on the database. If the database stalls,
the queue fills and new ticks are dropped and counted; failed batches are dropped and counted too, while
run start/end records are kept and retried. `store_from_env()` returns this store when DATABASE_URL is set
and the no-op NullStore otherwise. The driver is psycopg async, or sync psycopg on a worker thread where the
running loop can't host the async driver (Windows' ProactorEventLoop, uvicorn's default there).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from vbw.pipeline.jsonsafe import dumps
from vbw.store import queries
from vbw.store.base import NullStore, RunRecord, Store, TickRecord

log = logging.getLogger("vbw.store")

ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = ROOT / "infra" / "sql"


# ===================================================================================== SQL files


def split_sql(text: str) -> list[str]:
    """Split a SQL script into statements on `;`, ignoring semicolons in quotes, $$ bodies and comments."""
    statements: list[str] = []
    buf: list[str] = []
    i, n = 0, len(text)
    quote: str | None = None          # "'" or '"' or a dollar tag like "$$" / "$fn$"
    while i < n:
        ch = text[i]
        if quote is None:
            if text.startswith("--", i):
                end = text.find("\n", i)
                i = n if end == -1 else end
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                i = n if end == -1 else end + 2
                continue
            if ch in ("'", '"'):
                quote = ch
            elif ch == "$":
                end = text.find("$", i + 1)
                tag = text[i:end + 1] if end != -1 else ""
                if tag and (tag == "$$" or tag[1:-1].replace("_", "").isalnum()):
                    quote = tag
                    buf.append(tag)
                    i += len(tag)
                    continue
            elif ch == ";":
                stmt = "".join(buf).strip()
                if stmt:
                    statements.append(stmt)
                buf = []
                i += 1
                continue
        elif len(quote) == 1:
            if ch == quote:
                quote = None
        elif text.startswith(quote, i):
            buf.append(quote)
            i += len(quote)
            quote = None
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def schema_statements(sql_dir: Path = SQL_DIR) -> list[str]:
    """All statements from infra/sql/*.sql, files in name order."""
    statements: list[str] = []
    for path in sorted(sql_dir.glob("*.sql")):
        statements.extend(split_sql(path.read_text()))
    return statements


def apply_schema(dsn: str, sql_dir: Path = SQL_DIR) -> int:
    """Apply the schema synchronously (used by scripts/db_init.py). Returns the number of statements run."""
    import psycopg

    statements = schema_statements(sql_dir)
    with psycopg.connect(dsn, autocommit=True) as conn:
        for stmt in statements:
            conn.execute(stmt)
    return len(statements)


# ===================================================================================== row building


def _num(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def frame_rows(tick: TickRecord) -> list[tuple]:
    return [
        (tick.run.time_at(frame.t), tick.run.run_id, frame.vehicle_id, frame.seq, kind, dumps(frame.channels))
        for kind, frame in (("truth", tick.truth), ("reported", tick.reported))
    ]


def verdict_row(tick: TickRecord) -> tuple:
    v = tick.verdict
    return (
        tick.run.time_at(tick.reported.t), tick.run.run_id, v.seq, float(tick.reported.t),
        _num(v.score), bool(v.alert), _num(v.l1_score), _num(v.l2_score), _num(v.l3_prob),
        v.detector, tick.attack, bool(tick.attack_active), bool(v.l0.ok), v.explanation or "",
        dumps([vars(x) if hasattr(x, "__dict__") else x for x in v.l1_violations]),
    )


def run_params(run: RunRecord) -> dict[str, Any]:
    return {
        "run_id": run.run_id, "started_at": run.started_at, "scenario": run.scenario,
        "vehicle_id": run.vehicle_id, "schema": run.schema, "attack": run.attack,
        "attack_params": dumps(run.attack_params), "detector": run.detector,
        "speed": float(run.speed), "seed": run.seed,
    }


@dataclass
class RunEnd:
    run_id: str
    ended_at: datetime
    frames: int
    alerts: int


@dataclass
class Batch:
    meta: list[RunRecord | RunEnd] = field(default_factory=list)   # in order; written first
    frames: list[tuple] = field(default_factory=list)
    verdicts: list[tuple] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return len(self.frames) + len(self.verdicts)

    def __bool__(self) -> bool:
        return bool(self.meta or self.frames or self.verdicts)


# ===================================================================================== database


class Database(Protocol):
    async def write(self, batch: Batch) -> None: ...

    async def fetch(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...

    async def apply_schema(self, statements: list[str]) -> None: ...

    async def close(self) -> None: ...


Connect = Callable[[], Awaitable[Any]]


class PostgresDatabase:
    """psycopg 3 async. One connection for COPY writes, a second for dashboard reads; both reconnect lazily."""

    def __init__(self, dsn: str | None = None, connect: Connect | None = None):
        if connect is None:
            if not dsn:
                raise ValueError("need a dsn or a connect factory")
            connect = self._psycopg_connect(dsn)
        self._connect = connect
        self._writer: Any = None
        self._reader: Any = None
        self._read_lock = asyncio.Lock()

    @staticmethod
    def _psycopg_connect(dsn: str) -> Connect:
        async def connect() -> Any:
            import psycopg
            from psycopg.rows import dict_row

            return await psycopg.AsyncConnection.connect(dsn, autocommit=True, row_factory=dict_row)

        return connect

    async def _conn(self, attr: str) -> Any:
        conn = getattr(self, attr)
        if conn is None or getattr(conn, "closed", False):
            conn = await self._connect()
            setattr(self, attr, conn)
        return conn

    async def _discard(self, attr: str) -> None:
        conn = getattr(self, attr)
        setattr(self, attr, None)
        if conn is not None:
            try:
                await conn.close()
            except Exception:  # noqa: BLE001
                pass

    async def write(self, batch: Batch) -> None:
        conn = await self._conn("_writer")
        try:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    for item in batch.meta:
                        if isinstance(item, RunRecord):
                            await cur.execute(queries.UPSERT_RUN, run_params(item))
                        else:
                            await cur.execute(queries.END_RUN, vars(item))
                    for sql, rows in ((queries.COPY_FRAMES, batch.frames), (queries.COPY_VERDICTS, batch.verdicts)):
                        if rows:
                            async with cur.copy(sql) as copy:
                                for row in rows:
                                    await copy.write_row(row)
        except BaseException:
            await self._discard("_writer")
            raise

    async def fetch(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        async with self._read_lock:
            conn = await self._conn("_reader")
            try:
                async with conn.cursor() as cur:
                    await cur.execute(sql, params or {})
                    return list(await cur.fetchall())
            except BaseException:
                await self._discard("_reader")
                raise

    async def apply_schema(self, statements: list[str]) -> None:
        conn = await self._conn("_writer")
        try:
            for stmt in statements:
                await conn.execute(stmt)
        except BaseException:
            await self._discard("_writer")
            raise

    async def close(self) -> None:
        await self._discard("_writer")
        await self._discard("_reader")


SyncConnect = Callable[[], Any]


class ThreadedPostgresDatabase:
    """The same database calls with *sync* psycopg on one worker thread.

    psycopg's async driver refuses Windows' ProactorEventLoop, which is the loop uvicorn runs on Windows unless
    --reload is set, so `scripts/serve.py` on a Windows laptop would drop every write. One thread keeps reads and
    writes serialized on their two connections, exactly like the async version.
    """

    def __init__(self, dsn: str | None = None, connect: SyncConnect | None = None):
        if connect is None:
            if not dsn:
                raise ValueError("need a dsn or a connect factory")

            def connect() -> Any:
                import psycopg
                from psycopg.rows import dict_row

                return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)

        self._connect = connect
        self._writer: Any = None
        self._reader: Any = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vbw-db")

    def _conn(self, attr: str) -> Any:
        conn = getattr(self, attr)
        if conn is None or getattr(conn, "closed", False):
            conn = self._connect()
            setattr(self, attr, conn)
        return conn

    def _discard(self, attr: str) -> None:
        conn = getattr(self, attr)
        setattr(self, attr, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    def _write(self, batch: Batch) -> None:
        conn = self._conn("_writer")
        try:
            with conn.transaction():
                with conn.cursor() as cur:
                    for item in batch.meta:
                        if isinstance(item, RunRecord):
                            cur.execute(queries.UPSERT_RUN, run_params(item))
                        else:
                            cur.execute(queries.END_RUN, vars(item))
                    for sql, rows in ((queries.COPY_FRAMES, batch.frames), (queries.COPY_VERDICTS, batch.verdicts)):
                        if rows:
                            with cur.copy(sql) as copy:
                                for row in rows:
                                    copy.write_row(row)
        except BaseException:
            self._discard("_writer")
            raise

    def _fetch(self, sql: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        conn = self._conn("_reader")
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or {})
                return list(cur.fetchall())
        except BaseException:
            self._discard("_reader")
            raise

    def _apply(self, statements: list[str]) -> None:
        conn = self._conn("_writer")
        try:
            for stmt in statements:
                conn.execute(stmt)
        except BaseException:
            self._discard("_writer")
            raise

    async def write(self, batch: Batch) -> None:
        await self._run(self._write, batch)

    async def fetch(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return await self._run(self._fetch, sql, params)

    async def apply_schema(self, statements: list[str]) -> None:
        await self._run(self._apply, statements)

    async def close(self) -> None:
        await self._run(self._discard, "_writer")
        await self._run(self._discard, "_reader")
        self._pool.shutdown(wait=False)


def async_driver_ok(loop: asyncio.AbstractEventLoop | None = None) -> bool:
    """psycopg's async driver works on every loop except Windows' ProactorEventLoop."""
    if sys.platform != "win32":
        return True
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
    proactor = getattr(asyncio, "ProactorEventLoop", None)
    return not (proactor is not None and isinstance(loop, proactor))


def make_database(dsn: str) -> Database:
    """The async driver where it works, else the threaded sync one (Windows + ProactorEventLoop)."""
    return PostgresDatabase(dsn) if async_driver_ok() else ThreadedPostgresDatabase(dsn)


# ===================================================================================== store


class TimescaleStore:
    available = True

    def __init__(
        self,
        db: Database,
        batch_rows: int = 500,
        flush_interval_s: float = 0.5,
        max_queued_ticks: int = 20_000,
        write_timeout_s: float = 5.0,
        query_timeout_s: float = 5.0,
        auto_init: bool = False,
        sql_dir: Path = SQL_DIR,
    ):
        self.db = db
        self.batch_rows = batch_rows
        self.flush_interval_s = flush_interval_s
        self.max_queued_ticks = max_queued_ticks
        self.write_timeout_s = write_timeout_s
        self.query_timeout_s = query_timeout_s
        self.auto_init = auto_init
        self.sql_dir = sql_dir
        self._ticks: deque[TickRecord] = deque()
        self._meta: deque[RunRecord | RunEnd] = deque(maxlen=1000)
        self._kick = asyncio.Event()       # enough rows queued (or closing): flush now
        self._stop = asyncio.Event()       # closing: cut any backoff short
        self._task: asyncio.Task | None = None
        self._closing = False
        self._schema_ok = False
        self.counters = {"written_rows": 0, "batches": 0, "dropped_ticks": 0, "dropped_rows": 0, "errors": 0}
        self.last_error: str | None = None

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._task is None:
            self._closing = False
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="vbw-store-writer")

    async def close(self) -> None:
        self._closing = True
        self._kick.set()
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, self.write_timeout_s + 1.0)
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                log.warning("store writer did not drain cleanly: %s", exc)
                self._task.cancel()
            self._task = None
        await self.db.close()

    # ---------------------------------------------------------------- record (sync, never blocks)

    def record_run_start(self, run: RunRecord) -> None:
        self._meta.append(run)
        self._kick.set()

    def record_tick(self, tick: TickRecord) -> None:
        if len(self._ticks) >= self.max_queued_ticks:
            self.counters["dropped_ticks"] += 1
            return
        self._ticks.append(tick)
        if 3 * len(self._ticks) >= self.batch_rows:
            self._kick.set()

    def record_run_end(self, run_id: str, ended_at: datetime, frames: int, alerts: int) -> None:
        self._meta.append(RunEnd(run_id, ended_at, frames, alerts))
        self._kick.set()

    def stats(self) -> dict[str, Any]:
        return {"available": True, "queued_ticks": len(self._ticks), **self.counters, "last_error": self.last_error}

    # ---------------------------------------------------------------- writer task

    def take_batch(self) -> Batch:
        """Pop up to `batch_rows` rows (3 per tick) plus all pending run records."""
        batch = Batch(meta=list(self._meta))
        while self._ticks and batch.rows + 3 <= max(self.batch_rows, 3):
            tick = self._ticks.popleft()
            batch.frames.extend(frame_rows(tick))
            batch.verdicts.append(verdict_row(tick))
        return batch

    async def flush(self) -> int:
        """Write everything queued, batch by batch. Returns rows written. Stops at the first failed batch."""
        written = 0
        while self._meta or self._ticks:
            batch = self.take_batch()
            try:
                await asyncio.wait_for(self.db.write(batch), self.write_timeout_s)
            except Exception as exc:  # noqa: BLE001 - DB down/slow: drop rows, keep run records, back off
                self.counters["errors"] += 1
                self.counters["dropped_rows"] += batch.rows
                self.last_error = f"{type(exc).__name__}: {exc}"
                if self.counters["errors"] <= 3 or self.counters["errors"] % 50 == 0:
                    log.warning("store write failed (%d rows dropped): %s", batch.rows, self.last_error)
                raise
            for _ in batch.meta:
                self._meta.popleft()
            written += batch.rows
            self.counters["written_rows"] += batch.rows
            self.counters["batches"] += 1
        return written

    async def _run(self) -> None:
        backoff = 0.0
        while True:
            if not self._closing:
                if backoff:
                    await _wait(self._stop, backoff)        # back off after a failure; close() cuts it short
                else:
                    self._kick.clear()
                    await _wait(self._kick, self.flush_interval_s)
            try:
                if self.auto_init and not self._schema_ok:
                    await self._init_schema()
                await self.flush()
                backoff = 0.0
            except Exception:  # noqa: BLE001 - already counted and logged
                if self._closing:
                    return
                backoff = min(max(backoff * 2, 0.5), 10.0)
            if self._closing and not (self._meta or self._ticks):
                return

    async def _init_schema(self) -> None:
        try:
            statements = schema_statements(self.sql_dir)
            await asyncio.wait_for(self.db.apply_schema(statements), 30.0)
        except Exception as exc:
            self.counters["errors"] += 1
            self.last_error = f"schema: {type(exc).__name__}: {exc}"
            log.warning("store: could not apply schema: %s", self.last_error)
            raise
        self._schema_ok = True
        log.info("store: schema applied (%d statements)", len(statements))

    # ---------------------------------------------------------------- queries

    async def _fetch(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return await asyncio.wait_for(self.db.fetch(sql, params), self.query_timeout_s)

    async def list_runs(self, limit: int = 50) -> dict[str, Any]:
        try:
            rows = await self._fetch(queries.LIST_RUNS, {"limit": max(1, min(int(limit), 500))})
        except Exception as exc:  # noqa: BLE001
            return _query_failed(exc)
        return {"available": True, "runs": rows}

    async def timeline(self, run_id: str) -> dict[str, Any]:
        try:
            run_id = str(uuid.UUID(run_id))
        except ValueError:
            return {"available": True, "run": None, "buckets": []}
        try:
            summary = await self._fetch(queries.RUN_SUMMARY, {"run_id": run_id})
            buckets = await self._fetch(queries.TIMELINE, {"run_id": run_id}) if summary else []
        except Exception as exc:  # noqa: BLE001
            return _query_failed(exc)
        return {"available": True, "run": summary[0] if summary else None, "buckets": buckets}

    async def analytics(self) -> dict[str, Any]:
        try:
            latency = await self._fetch(queries.DETECTION_LATENCY)
            scorecard = await self._fetch(queries.SCORECARD)
        except Exception as exc:  # noqa: BLE001
            return _query_failed(exc)
        return {"available": True, "detection_latency": latency, "scorecard": scorecard}


async def _wait(event: asyncio.Event, timeout: float) -> None:
    try:
        async with asyncio.timeout(timeout):
            await event.wait()
    except TimeoutError:
        pass


def _query_failed(exc: Exception) -> dict[str, Any]:
    log.warning("store query failed: %s: %s", type(exc).__name__, exc)
    return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def store_from_env(env: dict[str, str] | None = None) -> Store:
    """TimescaleStore if DATABASE_URL is set, else NullStore. VBW_DB_AUTO_INIT=0 skips applying the schema."""
    env = dict(os.environ) if env is None else env
    dsn = env.get("DATABASE_URL", "").strip()
    if not dsn:
        return NullStore()
    auto_init = env.get("VBW_DB_AUTO_INIT", "1").strip() not in ("0", "false", "no")
    return TimescaleStore(make_database(dsn), auto_init=auto_init)
