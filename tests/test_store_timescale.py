"""Tiger Data store: SQL splitting, row building, batching, stall handling, and the psycopg write path,
all against fake connections (no database needed)."""

import asyncio
import json
import math
import time
from datetime import datetime, timedelta, timezone

from vbw.pipeline.fakes import FakeWorld
from vbw.pipeline.runner import Runner, RunnerConfig
from vbw.schema import FALCON9, Frame, L0Result, Verdict, Violation
from vbw.store import queries
from vbw.store.base import NullStore, RunRecord, TickRecord
from vbw.store.timescale import (
    Batch, PostgresDatabase, RunEnd, ThreadedPostgresDatabase, TimescaleStore, frame_rows, schema_statements,
    split_sql, store_from_env, verdict_row,
)

T0 = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)


def make_run(run_id="0" * 31 + "1") -> RunRecord:
    return RunRecord(run_id, T0, "falcon9:ses-10", "f9", "falcon9", "bias", {"start_t": 60.0}, "base", 5.0)


def make_tick(run: RunRecord, seq: int, score: float = 0.25, active: bool = False) -> TickRecord:
    channels = {c: float(seq) for c in FALCON9.channels}
    truth = Frame("f9", FALCON9.schema_id, seq, seq / 10, channels)
    reported = truth.copy()
    if active:
        reported.channels["altitude"] += 100.0
    violation = Violation("vz_vs_dalt", ["altitude", "vz"], 120.0, 80.0, "altitude disagrees with vz")
    verdict = Verdict("f9", seq, truth.t, L0Result(), [violation] if active else [], l1_score=score,
                      score=score, alert=active, explanation="x", detector="base")
    return TickRecord(run, truth, reported, verdict, "bias", active)


class FakeDatabase:
    """Records batches; can be told to stall or fail."""

    def __init__(self, stall: bool = False, fail: bool = False, rows=None):
        self.batches: list[Batch] = []
        self.stall, self.fail, self.rows = stall, fail, rows or []
        self.statements: list[str] = []
        self.closed = False

    async def write(self, batch: Batch) -> None:
        if self.stall:
            await asyncio.Event().wait()
        if self.fail:
            raise ConnectionError("db down")
        self.batches.append(batch)

    async def fetch(self, sql, params=None):
        if self.fail:
            raise ConnectionError("db down")
        return list(self.rows)

    async def apply_schema(self, statements):
        self.statements.extend(statements)

    async def close(self):
        self.closed = True


# ---------------------------------------------------------------------------------- SQL


def test_split_sql_respects_quotes_dollar_bodies_and_comments():
    sql = """
    -- a comment; with a semicolon
    CREATE TABLE a (x text DEFAULT 'semi;colon');
    DO $$ BEGIN RAISE NOTICE 'one; two: %', 1; END $$;
    /* block; comment */ SELECT 1;
    SELECT $tag$ ; $tag$
    """
    stmts = split_sql(sql)
    assert len(stmts) == 4
    assert stmts[0].startswith("CREATE TABLE a") and "'semi;colon'" in stmts[0]
    assert stmts[1] == "DO $$ BEGIN RAISE NOTICE 'one; two: %', 1; END $$"
    assert stmts[2] == "SELECT 1"
    assert stmts[3] == "SELECT $tag$ ; $tag$"


def test_schema_has_hypertables_cagg_and_policies():
    stmts = schema_statements()
    text = "\n".join(stmts)
    assert sum("create_hypertable(" in s for s in stmts) == 2
    assert "create_hypertable('frames'" in text and "create_hypertable('verdicts'" in text
    cagg = next(s for s in stmts if "CREATE MATERIALIZED VIEW" in s)
    assert "timescaledb.continuous" in cagg and "time_bucket(INTERVAL '1 second', time)" in cagg
    assert "max(score)" in cagg and "count(*)" in cagg
    assert "add_continuous_aggregate_policy('verdicts_1s'" in text
    assert "add_compression_policy('frames'" in text
    do_block = next(s for s in stmts if s.startswith("DO $$"))
    assert do_block.rstrip().endswith("$$") and "timescaledb.compress_segmentby" in do_block


def test_copy_columns_match_row_builders():
    run = make_run()
    tick = make_tick(run, 5, active=True)
    frame_cols = queries.COPY_FRAMES.split("(")[1].split(")")[0].split(",")
    verdict_cols = queries.COPY_VERDICTS.split("(")[1].split(")")[0].split(",")
    assert all(len(r) == len(frame_cols) for r in frame_rows(tick))
    assert len(verdict_row(tick)) == len(verdict_cols)


# ---------------------------------------------------------------------------------- rows


def test_rows_use_run_start_plus_mission_time():
    run = make_run()
    tick = make_tick(run, 30, active=True)
    truth_row, reported_row = frame_rows(tick)
    assert truth_row[0] == T0 + timedelta(seconds=3.0) and truth_row[4] == "truth"
    assert reported_row[4] == "reported"
    assert json.loads(reported_row[5])["altitude"] == 130.0 and json.loads(truth_row[5])["altitude"] == 30.0
    row = verdict_row(tick)
    assert row[0] == T0 + timedelta(seconds=3.0) and row[3] == 3.0
    assert row[5] is True and row[11] is True and row[12] is True          # alert, attack_active, l0_ok
    assert json.loads(row[14])[0]["rule"] == "vz_vs_dalt"


def test_non_finite_scores_become_null():
    tick = make_tick(make_run(), 1, score=math.nan)
    row = verdict_row(tick)
    assert row[4] is None and row[6] is None and row[8] is None             # score, l1_score, l3_prob (None)


# ---------------------------------------------------------------------------------- psycopg path


class FakeCopy:
    def __init__(self, log, sql):
        self.log, self.sql, self.rows = log, sql, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.log.append(("copy", self.sql, self.rows))

    async def write_row(self, row):
        self.rows.append(row)


class FakeCursor:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def execute(self, sql, params=None):
        self.log.append(("execute", sql, params))

    def copy(self, sql):
        return FakeCopy(self.log, sql)

    async def fetchall(self):
        return self.rows


class FakeTransaction:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        self.log.append(("begin",))

    async def __aexit__(self, exc_type, *exc):
        self.log.append(("rollback",) if exc_type else ("commit",))


class FakeConnection:
    def __init__(self, rows=None):
        self.log: list[tuple] = []
        self.rows = rows or []
        self.closed = False

    def transaction(self):
        return FakeTransaction(self.log)

    def cursor(self):
        return FakeCursor(self.log, self.rows)

    async def execute(self, sql, params=None):
        self.log.append(("execute", sql, params))

    async def close(self):
        self.closed = True


def test_postgres_database_writes_one_transaction_with_copy():
    conn = FakeConnection()

    async def connect():
        return conn

    run = make_run()
    batch = Batch(meta=[run, RunEnd(run.run_id, T0, 2, 1)])
    for seq in range(2):
        tick = make_tick(run, seq, active=bool(seq))
        batch.frames.extend(frame_rows(tick))
        batch.verdicts.append(verdict_row(tick))

    asyncio.run(PostgresDatabase(connect=connect).write(batch))
    kinds = [entry[0] for entry in conn.log]
    assert kinds == ["begin", "execute", "execute", "copy", "copy", "commit"]
    assert conn.log[1][1] == queries.UPSERT_RUN and conn.log[1][2]["run_id"] == run.run_id
    assert json.loads(conn.log[1][2]["attack_params"]) == {"start_t": 60.0}
    assert conn.log[2][1] == queries.END_RUN and conn.log[2][2]["frames"] == 2
    assert conn.log[3][1] == queries.COPY_FRAMES and len(conn.log[3][2]) == 4
    assert conn.log[4][1] == queries.COPY_VERDICTS and len(conn.log[4][2]) == 2


def test_postgres_database_reconnects_after_a_failed_write():
    connections = []

    class Broken(FakeConnection):
        def transaction(self):
            raise ConnectionError("server closed the connection")

    async def connect():
        connections.append(Broken() if not connections else FakeConnection())
        return connections[-1]

    async def main():
        db = PostgresDatabase(connect=connect)
        try:
            await db.write(Batch(meta=[make_run()]))
        except ConnectionError:
            pass
        await db.write(Batch(meta=[make_run()]))

    asyncio.run(main())
    assert len(connections) == 2 and connections[0].closed


class SyncCopy:
    def __init__(self, log, sql):
        self.log, self.sql, self.rows = log, sql, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.log.append(("copy", self.sql, self.rows))

    def write_row(self, row):
        self.rows.append(row)


class SyncCursor:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def execute(self, sql, params=None):
        self.log.append(("execute", sql, params))

    def copy(self, sql):
        return SyncCopy(self.log, sql)

    def fetchall(self):
        return self.rows


class SyncTransaction:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        self.log.append(("begin",))

    def __exit__(self, exc_type, *exc):
        self.log.append(("rollback",) if exc_type else ("commit",))


class SyncConnection:
    def __init__(self, rows=None):
        self.log: list[tuple] = []
        self.rows = rows or []
        self.closed = False

    def transaction(self):
        return SyncTransaction(self.log)

    def cursor(self):
        return SyncCursor(self.log, self.rows)

    def execute(self, sql, params=None):
        self.log.append(("execute", sql, params))

    def close(self):
        self.closed = True


def test_threaded_database_writes_reads_and_applies_schema_like_the_async_one():
    conns = []

    def connect():
        conns.append(SyncConnection(rows=[{"run_id": "x"}]))
        return conns[-1]

    run = make_run()
    batch = Batch(meta=[run])
    tick = make_tick(run, 1, active=True)
    batch.frames.extend(frame_rows(tick))
    batch.verdicts.append(verdict_row(tick))

    async def main():
        db = ThreadedPostgresDatabase(connect=connect)
        await db.apply_schema(["SELECT 1"])
        await db.write(batch)
        rows = await db.fetch(queries.LIST_RUNS, {"limit": 5})
        await db.close()
        return rows

    rows = asyncio.run(main())
    writer, reader = conns
    assert [e[0] for e in writer.log] == ["execute", "begin", "execute", "copy", "copy", "commit"]
    assert writer.log[3][1] == queries.COPY_FRAMES and len(writer.log[3][2]) == 2
    assert rows == [{"run_id": "x"}] and reader.log[0][2] == {"limit": 5}
    assert writer.closed and reader.closed


def test_threaded_database_reconnects_after_a_failed_write():
    conns = []

    class Broken(SyncConnection):
        def transaction(self):
            raise ConnectionError("server closed the connection")

    def connect():
        conns.append(Broken() if not conns else SyncConnection())
        return conns[-1]

    async def main():
        db = ThreadedPostgresDatabase(connect=connect)
        try:
            await db.write(Batch(meta=[make_run()]))
        except ConnectionError:
            pass
        await db.write(Batch(meta=[make_run()]))

    asyncio.run(main())
    assert len(conns) == 2 and conns[0].closed


def test_async_driver_is_avoided_only_on_windows_proactor_loops(monkeypatch):
    import vbw.store.timescale as ts

    class Proactor:
        pass

    monkeypatch.setattr(ts.asyncio, "ProactorEventLoop", Proactor, raising=False)
    monkeypatch.setattr(ts.sys, "platform", "win32")
    assert ts.async_driver_ok(Proactor()) is False
    assert ts.async_driver_ok(object()) is True
    monkeypatch.setattr(ts.sys, "platform", "linux")
    assert ts.async_driver_ok(Proactor()) is True


# ---------------------------------------------------------------------------------- batching


def test_flush_splits_into_batches_of_at_most_batch_rows():
    db = FakeDatabase()
    store = TimescaleStore(db, batch_rows=30)
    run = make_run()
    store.record_run_start(run)
    for seq in range(25):
        store.record_tick(make_tick(run, seq))
    written = asyncio.run(store.flush())
    assert written == 75
    assert [b.rows for b in db.batches] == [30, 30, 15]
    assert db.batches[0].meta == [run] and db.batches[1].meta == []
    assert store.stats()["written_rows"] == 75 and store.stats()["queued_ticks"] == 0


def test_background_writer_flushes_on_interval_and_drains_on_close():
    db = FakeDatabase()

    async def main():
        store = TimescaleStore(db, flush_interval_s=0.05, auto_init=True)
        await store.start()
        run = make_run()
        store.record_run_start(run)
        store.record_tick(make_tick(run, 0))
        await asyncio.sleep(0.2)
        flushed_by_timer = sum(b.rows for b in db.batches)
        store.record_tick(make_tick(run, 1))
        store.record_run_end(run.run_id, T0, 2, 0)
        await store.close()
        return flushed_by_timer

    assert asyncio.run(main()) == 3
    assert sum(b.rows for b in db.batches) == 6
    assert isinstance(db.batches[-1].meta[-1], RunEnd)
    assert db.closed and any("create_hypertable" in s for s in db.statements)


def test_stalled_database_never_blocks_the_caller_and_counts_drops():
    db = FakeDatabase(stall=True)

    async def main():
        store = TimescaleStore(db, batch_rows=30, flush_interval_s=0.01, max_queued_ticks=100, write_timeout_s=0.05)
        await store.start()
        run = make_run()
        store.record_run_start(run)
        start = time.perf_counter()
        for seq in range(1000):
            store.record_tick(make_tick(run, seq))
        record_time = time.perf_counter() - start
        await asyncio.sleep(0.2)
        stats = store.stats()
        store._ticks.clear()
        db.stall = False
        await store.close()
        return record_time, stats

    record_time, stats = asyncio.run(main())
    assert record_time < 0.5
    assert stats["dropped_ticks"] == 900
    assert stats["errors"] >= 1 and stats["dropped_rows"] >= 30
    assert "TimeoutError" in stats["last_error"]
    assert any(isinstance(m, RunRecord) for b in db.batches for m in b.meta), "run records survive the stall"


# ---------------------------------------------------------------------------------- queries


def test_queries_report_unavailable_when_the_db_fails():
    store = TimescaleStore(FakeDatabase(fail=True))
    result = asyncio.run(store.list_runs())
    assert result["available"] is False and "db down" in result["error"]
    assert asyncio.run(store.analytics())["available"] is False


def test_queries_return_rows():
    rows = [{"run_id": "0" * 32, "peak_score": 0.97}]
    store = TimescaleStore(FakeDatabase(rows=rows))
    assert asyncio.run(store.list_runs()) == {"available": True, "runs": rows}
    timeline = asyncio.run(store.timeline("0" * 32))
    assert timeline["run"] == rows[0] and timeline["buckets"] == rows
    assert asyncio.run(store.timeline("not-a-uuid")) == {"available": True, "run": None, "buckets": []}
    analytics = asyncio.run(store.analytics())
    assert analytics["detection_latency"] == rows and analytics["scorecard"] == rows


def test_store_from_env():
    assert isinstance(store_from_env({}), NullStore)
    store = store_from_env({"DATABASE_URL": "postgresql://u:p@localhost:5432/vbw", "VBW_DB_AUTO_INIT": "0"})
    assert isinstance(store, TimescaleStore) and store.auto_init is False


# ---------------------------------------------------------------------------------- runner integration


def test_runner_records_a_full_run():
    db = FakeDatabase()
    world = FakeWorld(n_frames=120, rate_hz=100.0)

    async def main():
        store = TimescaleStore(db, flush_interval_s=0.02)
        await store.start()
        runner = Runner(world.registries(), lambda m: None, store, RunnerConfig(speed_max=1e6))
        await runner.control("start", attack="bias", speed=1e6)
        await runner.wait_stopped(timeout=5)
        await store.close()
        return runner

    runner = asyncio.run(main())
    meta = [m for b in db.batches for m in b.meta]
    verdicts = [r for b in db.batches for r in b.verdicts]
    frames = [r for b in db.batches for r in b.frames]
    assert isinstance(meta[0], RunRecord) and meta[0].run_id == runner.status.run_id
    assert isinstance(meta[-1], RunEnd) and meta[-1].frames == 120 and meta[-1].alerts == 1
    assert len(verdicts) == 120 and len(frames) == 240
    assert sum(r[11] for r in verdicts) == runner.status.attacked_frames > 0
    times = [r[0] for r in verdicts]
    assert times == sorted(times) and (times[-1] - times[0]).total_seconds() == 1.19
