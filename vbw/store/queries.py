"""SQL for the Tiger Data store: write statements and the dashboard/analytics read queries.

Read queries go through the `verdicts_1s` continuous aggregate wherever a per-second resolution is enough,
and touch the raw `verdicts` hypertable only for exact first-alert times.
"""

from __future__ import annotations

# ------------------------------------------------------------------------------------------- writes

UPSERT_RUN = """
INSERT INTO runs (run_id, started_at, scenario, vehicle_id, schema_name, attack, attack_params,
                  detector, speed, seed)
VALUES (%(run_id)s::uuid, %(started_at)s, %(scenario)s, %(vehicle_id)s, %(schema)s, %(attack)s,
        %(attack_params)s::jsonb, %(detector)s, %(speed)s, %(seed)s)
ON CONFLICT (run_id) DO NOTHING
"""

END_RUN = """
UPDATE runs SET ended_at = %(ended_at)s, frames = %(frames)s, alerts = %(alerts)s
WHERE run_id = %(run_id)s::uuid
"""

COPY_FRAMES = "COPY frames (time, run_id, vehicle_id, seq, kind, channels) FROM STDIN"

COPY_VERDICTS = (
    "COPY verdicts (time, run_id, seq, t, score, alert, l1_score, l2_score, l3_prob, detector, attack, "
    "attack_active, l0_ok, explanation, violations) FROM STDIN"
)

# -------------------------------------------------------------------------------------------- reads

# Recent runs with per-run rollups from the continuous aggregate.
LIST_RUNS = """
SELECT r.run_id::text AS run_id, r.scenario, r.vehicle_id, r.schema_name AS schema,
       r.attack, r.detector, r.speed, r.started_at, r.ended_at, r.frames, r.alerts,
       s.peak_score, s.alert_frames, s.attacked_frames, s.false_alert_frames, s.seconds,
       s.attacks, s.detectors
FROM runs r
LEFT JOIN LATERAL (
    SELECT max(v.max_score)                 AS peak_score,
           sum(v.alert_frames)::bigint       AS alert_frames,
           sum(v.attacked_frames)::bigint    AS attacked_frames,
           sum(v.false_alert_frames)::bigint AS false_alert_frames,
           count(DISTINCT v.bucket)::int     AS seconds,
           array_agg(DISTINCT v.attack)      AS attacks,
           array_agg(DISTINCT v.detector)    AS detectors
    FROM verdicts_1s v
    WHERE v.run_id = r.run_id
) s ON TRUE
ORDER BY r.started_at DESC
LIMIT %(limit)s
"""

# Header for one run: when the attack first changed the picture, when the detector first alerted after
# that (same attack + detector), and the detection latency in mission seconds.
RUN_SUMMARY = """
WITH onset AS (
    SELECT min(time) AS attack_start
    FROM verdicts
    WHERE run_id = %(run_id)s::uuid AND attack_active
), detect AS (
    SELECT min(v.time) AS first_alert
    FROM verdicts v, onset o
    WHERE v.run_id = %(run_id)s::uuid AND v.alert AND v.time >= o.attack_start
)
SELECT r.run_id::text AS run_id, r.scenario, r.vehicle_id, r.schema_name AS schema,
       r.attack, r.detector, r.speed, r.started_at, r.ended_at, r.frames, r.alerts,
       extract(epoch FROM o.attack_start - r.started_at)::float8 AS attack_start_t,
       extract(epoch FROM d.first_alert - r.started_at)::float8  AS first_alert_t,
       extract(epoch FROM d.first_alert - o.attack_start)::float8 AS detection_latency_s
FROM runs r, onset o, detect d
WHERE r.run_id = %(run_id)s::uuid
"""

# Peak score and alert/attack counts per mission second, straight from the continuous aggregate.
TIMELINE = """
SELECT v.bucket                                            AS time,
       extract(epoch FROM v.bucket - r.started_at)::float8 AS t,
       max(v.max_score)                AS max_score,
       max(v.max_l1)                   AS max_l1,
       max(v.max_l2)                   AS max_l2,
       max(v.max_l3)                   AS max_l3,
       sum(v.alert_frames)::int        AS alert_frames,
       sum(v.attacked_frames)::int     AS attacked_frames,
       sum(v.false_alert_frames)::int  AS false_alert_frames,
       sum(v.frames)::int              AS frames,
       string_agg(DISTINCT v.attack, ',')   AS attack,
       string_agg(DISTINCT v.detector, ',') AS detector
FROM verdicts_1s v
JOIN runs r ON r.run_id = v.run_id
WHERE v.run_id = %(run_id)s::uuid
GROUP BY v.bucket, r.started_at
ORDER BY v.bucket
"""

# Time to first alert per (attack, detector) across every run: attack onset = first frame the attack
# actually changed; first alert = first alerting frame after it under the same attack and detector.
DETECTION_LATENCY = """
WITH onset AS (
    SELECT run_id, attack, detector, min(time) AS attack_start
    FROM verdicts
    WHERE attack_active
    GROUP BY run_id, attack, detector
), detect AS (
    SELECT o.attack, o.detector, o.attack_start,
           (SELECT min(v.time) FROM verdicts v
             WHERE v.run_id = o.run_id AND v.attack = o.attack AND v.detector = o.detector
               AND v.alert AND v.time >= o.attack_start) AS first_alert
    FROM onset o
), lat AS (
    SELECT attack, detector, extract(epoch FROM first_alert - attack_start)::float8 AS latency_s
    FROM detect
)
SELECT attack, detector,
       count(*)::int                                           AS attacked_runs,
       count(latency_s)::int                                   AS detected_runs,
       avg(latency_s)                                          AS mean_latency_s,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_s)  AS median_latency_s,
       min(latency_s)                                          AS best_latency_s,
       max(latency_s)                                          AS worst_latency_s
FROM lat
GROUP BY attack, detector
ORDER BY attack, detector
"""

# Detector scorecard from the continuous aggregate: frame-level hit rate and false-alert rate.
SCORECARD = """
SELECT detector, attack,
       count(DISTINCT run_id)::int                  AS runs,
       sum(frames)::bigint                          AS frames,
       sum(attacked_frames)::bigint                 AS attacked_frames,
       sum(alert_frames)::bigint                    AS alert_frames,
       sum(false_alert_frames)::bigint              AS false_alert_frames,
       sum(false_alert_frames)::float8
         / nullif(sum(frames) - sum(attacked_frames), 0) AS false_alert_rate,
       (sum(alert_frames) - sum(false_alert_frames))::float8
         / nullif(sum(attacked_frames), 0)          AS attacked_alert_rate,
       max(max_score)                               AS peak_score
FROM verdicts_1s
GROUP BY detector, attack
ORDER BY detector, attack
"""
