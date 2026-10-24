-- Valid-But-Wrong schema for Tiger Data (TimescaleDB). Idempotent: safe to re-run.
-- Applied by `uv run python scripts/db_init.py`, by the server on startup (VBW_DB_AUTO_INIT=1),
-- and by the timescaledb container on first boot (mounted into /docker-entrypoint-initdb.d).
--
-- Time convention: every row's `time` = run wall-clock start + mission time t, so one run occupies a
-- contiguous slice of the hypertable and per-second buckets line up with mission seconds.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- --------------------------------------------------------------------------- runs (plain table)

CREATE TABLE IF NOT EXISTS runs (
    run_id        uuid PRIMARY KEY,
    started_at    timestamptz      NOT NULL,
    ended_at      timestamptz,
    scenario      text             NOT NULL,
    vehicle_id    text             NOT NULL,
    schema_name   text             NOT NULL,
    attack        text             NOT NULL,          -- attack at run start (verdicts.attack tracks swaps)
    attack_params jsonb            NOT NULL DEFAULT '{}'::jsonb,
    detector      text             NOT NULL,          -- detector at run start (verdicts.detector tracks swaps)
    speed         double precision NOT NULL DEFAULT 1,
    seed          integer,
    frames        integer          NOT NULL DEFAULT 0,
    alerts        integer          NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS runs_started_at_idx ON runs (started_at DESC);

-- --------------------------------------------------------------------------- frames (hypertable)
-- Two rows per frame: what the vehicle sent ('truth') and what the ground decoded ('reported').

CREATE TABLE IF NOT EXISTS frames (
    time       timestamptz NOT NULL,
    run_id     uuid        NOT NULL,
    vehicle_id text        NOT NULL,
    seq        integer     NOT NULL,
    kind       text        NOT NULL CHECK (kind IN ('truth', 'reported')),
    channels   jsonb       NOT NULL
);
SELECT create_hypertable('frames', 'time', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS frames_run_time_idx ON frames (run_id, time DESC);

-- Columnar compression: segment by run + kind so a run's truth/reported tracks decompress independently.
DO $$
BEGIN
    ALTER TABLE frames SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'run_id, kind',
        timescaledb.compress_orderby = 'time DESC, seq DESC'
    );
EXCEPTION WHEN others THEN
    RAISE NOTICE 'frames compression settings left as they were: %', SQLERRM;
END
$$;
SELECT add_compression_policy('frames', compress_after => INTERVAL '1 day', if_not_exists => TRUE);

-- --------------------------------------------------------------------------- verdicts (hypertable)

CREATE TABLE IF NOT EXISTS verdicts (
    time          timestamptz      NOT NULL,
    run_id        uuid             NOT NULL,
    seq           integer          NOT NULL,
    t             double precision NOT NULL,          -- mission time, seconds
    score         double precision,
    alert         boolean          NOT NULL,
    l1_score      double precision,
    l2_score      double precision,
    l3_prob       double precision,                   -- NULL when the hardened discriminator is not loaded
    detector      text             NOT NULL,
    attack        text             NOT NULL,          -- attack configured on the link at this frame
    attack_active boolean          NOT NULL,          -- reported frame differed from truth (ground truth label)
    l0_ok         boolean          NOT NULL,          -- passed sync/CRC/seq/time checks
    explanation   text,
    violations    jsonb            NOT NULL DEFAULT '[]'::jsonb
);
SELECT create_hypertable('verdicts', 'time', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS verdicts_run_time_idx ON verdicts (run_id, time DESC);
CREATE INDEX IF NOT EXISTS verdicts_run_alert_idx ON verdicts (run_id, time) WHERE alert;

-- --------------------------------------------------------------------------- verdicts_1s (continuous aggregate)
-- One row per (second, run, detector, attack). Drives the dashboard timeline and the detector scorecard.
-- materialized_only = false: real-time aggregation also covers rows newer than the last refresh, which
-- matters because a run played at 20x writes timestamps ahead of the wall clock.

CREATE MATERIALIZED VIEW IF NOT EXISTS verdicts_1s
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket(INTERVAL '1 second', time)                         AS bucket,
    run_id,
    detector,
    attack,
    max(score)                                                     AS max_score,
    max(l1_score)                                                  AS max_l1,
    max(l2_score)                                                  AS max_l2,
    max(l3_prob)                                                   AS max_l3,
    sum(CASE WHEN alert THEN 1 ELSE 0 END)                         AS alert_frames,
    sum(CASE WHEN attack_active THEN 1 ELSE 0 END)                 AS attacked_frames,
    sum(CASE WHEN alert AND NOT attack_active THEN 1 ELSE 0 END)   AS false_alert_frames,
    count(*)                                                       AS frames
FROM verdicts
GROUP BY bucket, run_id, detector, attack
WITH NO DATA;

SELECT add_continuous_aggregate_policy('verdicts_1s',
    start_offset      => INTERVAL '3 days',
    end_offset        => INTERVAL '1 second',
    schedule_interval => INTERVAL '10 seconds',
    if_not_exists     => TRUE);
