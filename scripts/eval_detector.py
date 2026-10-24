"""Evaluate the base and hardened detectors on held-out launches and write results/detector_eval.json.

    uv run python scripts/eval_detector.py --schema falcon9

Everything runs on the holdout launches (never used to train L2 / L3 or to fit L1 thresholds; the val half sets
the alert threshold τ and the L3 logit threshold, the test half is only ever scored), as the ground station decodes
them: sensor noise + float32 wire rounding.

- clean:   every holdout launch × several noise seeds, no attack. Alert episodes (rising edges) and runs with any
           alert, i.e. false alerts per flight.
- attacks: each attack starts at 35% of the flight and runs to the end. Detection rate, latency from onset to the
           first alert, the largest altitude lie the operator saw before that alert, and any alert before onset.
           PPO's shadow detector is always `base`: the policy was trained to beat base and knows nothing about L3.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DETECTORS = ("base", "hardened")
ATTACKS = ("bias", "drift", "replay", "freeze", "coordinated", "ppo")
INTENSITIES = (0.4, 1.0)
CLEAN_SEEDS = (0, 1, 2)
START_FRAC = 0.35
PICTURE = {"falcon9": "altitude", "quad": "alt"}


def run_one(job: dict) -> dict:
    from vbw.attacker.registry import make_attack
    from vbw.detector.data import clean_runs
    from vbw.detector.registry import make_detector
    from vbw.schema import SCHEMAS_BY_NAME, Frame, L0Result

    schema = SCHEMAS_BY_NAME[job["schema"]]
    run = clean_runs(schema, job["split"], (job["seed"],))[job["idx"]]
    t, X = run.t, run.X
    start_t = float(t[int(len(t) * START_FRAC)]) if job["attack"] != "none" else float("inf")
    attack = make_attack(job["attack"], schema, start_t=0.0 if job["attack"] == "none" else start_t,
                         intensity=job["intensity"])
    attack.reset(0)
    det = make_detector(job["detector"], schema)
    pic = schema.channels.index(PICTURE[schema.name])

    alerts = np.zeros(len(t), bool)
    lie = np.zeros(len(t))
    det_s = 0.0
    for i in range(len(t)):
        truth = Frame("eval", schema.schema_id, i & 0xFFFF, float(t[i]),
                      {c: float(X[i, j]) for j, c in enumerate(schema.channels)})
        reported = attack.apply(truth)
        t0 = time.perf_counter()
        alerts[i] = det.update(reported.copy(), L0Result()).alert
        det_s += time.perf_counter() - t0
        lie[i] = abs(reported.channels[schema.channels[pic]] - truth.channels[schema.channels[pic]])

    rising = int(np.sum(alerts[1:] & ~alerts[:-1]) + alerts[0])
    active = t >= start_t
    after = np.where(alerts & active)[0]
    first = int(after[0]) if len(after) else None
    before_alert = active & (np.arange(len(t)) < (first if first is not None else len(t)))
    return {
        **job,
        "scenario": run.scenario,
        "frames": int(len(t)),
        "duration_s": float(t[-1] - t[0]),
        "alert_episodes": rising,
        "alerted_before_onset": bool(np.any(alerts & ~active)),
        "detected": first is not None,
        "latency_s": float(t[first] - start_t) if first is not None else None,
        "max_lie_before_alert": float(lie[before_alert].max()) if before_alert.any() else 0.0,
        "max_lie": float(lie.max()),
        "ms_per_frame": 1000.0 * det_s / len(t),
    }


def summarize_clean(rows: list[dict]) -> dict:
    hours = sum(r["duration_s"] for r in rows) / 3600
    return {
        "runs": len(rows),
        "launches": len({r["scenario"] for r in rows}),
        "flight_hours": round(hours, 2),
        "frames": sum(r["frames"] for r in rows),
        "runs_with_false_alert": sum(r["alert_episodes"] > 0 for r in rows),
        "false_alert_episodes": sum(r["alert_episodes"] for r in rows),
        "false_alerts_per_flight": sum(r["alert_episodes"] for r in rows) / max(len(rows), 1),
        "false_alerts_per_flight_hour": sum(r["alert_episodes"] for r in rows) / max(hours, 1e-9),
        "alerting_scenarios": sorted({r["scenario"] for r in rows if r["alert_episodes"] > 0}),
        "ms_per_frame": float(np.mean([r["ms_per_frame"] for r in rows])),
    }


def summarize_attack(rows: list[dict]) -> dict:
    lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    return {
        "runs": len(rows),
        "detection_rate": float(np.mean([r["detected"] for r in rows])),
        "mean_latency_s": float(np.mean(lat)) if lat else None,
        "median_latency_s": float(np.median(lat)) if lat else None,
        "mean_max_lie_before_alert_m": float(np.mean([r["max_lie_before_alert"] for r in rows])),
        "mean_max_lie_m": float(np.mean([r["max_lie"] for r in rows])),
        "runs_alerting_before_onset": sum(r["alerted_before_onset"] for r in rows),
        "per_run": [{k: r[k] for k in ("scenario", "detected", "latency_s", "max_lie_before_alert", "max_lie")}
                    for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schema", choices=["falcon9"], default="falcon9")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    from vbw.detector.data import clean_runs
    from vbw.schema import SCHEMAS_BY_NAME

    schema = SCHEMAS_BY_NAME[args.schema]
    n_hold = len(clean_runs(schema, "holdout", (0,)))
    test = {r.scenario for r in clean_runs(schema, "test", (0,))}
    attacks = [a for a in ATTACKS if a != "ppo" or (ROOT / "models" / f"ppo_{schema.name}.zip").exists()]
    jobs = []
    for d in DETECTORS:
        jobs += [dict(schema=schema.name, detector=d, attack="none", intensity=0.0, split="holdout", seed=s, idx=i)
                 for s in CLEAN_SEEDS for i in range(n_hold)]
        # The PPO policy ignores intensity, so it only runs once.
        jobs += [dict(schema=schema.name, detector=d, attack=a, intensity=k, split="holdout", seed=0, idx=i)
                 for a in attacks for k in ((1.0,) if a == "ppo" else INTENSITIES) for i in range(n_hold)]
    print(f"{len(jobs)} runs on {args.workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run_one, jobs, chunksize=2))
    print(f"done in {time.time() - t0:.0f}s")

    from vbw.detector.l3 import l3_paths

    out: dict = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema": schema.name,
        "split": "holdout (val launches set tau and the L3 threshold; test launches are scored only)",
        "test_launches": sorted(test),
        "attack_start": f"{int(START_FRAC * 100)}% of flight, runs to the end",
        "picture_channel": PICTURE[schema.name],
        "l3_trained": l3_paths(schema)[0].exists(),
        "ppo_shadow_detector": "base",
        "detectors": {},
    }
    for d in DETECTORS:
        mine = [r for r in rows if r["detector"] == d]
        clean = [r for r in mine if r["attack"] == "none"]
        entry = {"clean": summarize_clean(clean),
                 "clean_test_only": summarize_clean([r for r in clean if r["scenario"] in test]),
                 "attacks": {}}
        for a in attacks:
            by_k = {k: [r for r in mine if r["attack"] == a and r["intensity"] == k] for k in INTENSITIES}
            entry["attacks"][a] = {f"{k:.1f}": summarize_attack(rs) for k, rs in by_k.items() if rs}
        out["detectors"][d] = entry

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "detector_eval.json").write_text(json.dumps(out, indent=1))
    for d, e in out["detectors"].items():
        c = e["clean"]
        print(f"[{d}] clean: {c['runs_with_false_alert']}/{c['runs']} runs alert, {c['false_alert_episodes']} episodes "
              f"over {c['flight_hours']} h, {c['ms_per_frame']:.2f} ms/frame")
        for a, by_k in e["attacks"].items():
            s = by_k["1.0"]
            lat = f"{s['mean_latency_s']:.1f}s" if s["mean_latency_s"] is not None else "-"
            low = f"   (@0.4: detect {by_k['0.4']['detection_rate']:.2f})" if "0.4" in by_k else ""
            print(f"   {a:12s} detect {s['detection_rate']:.2f} latency {lat:>7s} lie-before-alert "
                  f"{s['mean_max_lie_before_alert_m']:9.0f} m{low}")
    print("wrote results/detector_eval.json")


if __name__ == "__main__":
    main()
