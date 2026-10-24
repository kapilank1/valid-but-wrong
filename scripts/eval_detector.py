"""Evaluate the base and hardened detectors on held-out launches and write results/detector_eval.json.

    uv run python scripts/eval_detector.py --schema falcon9

Everything runs on the holdout launches (never used to train L2 / L3 or to fit L1 thresholds; the val half sets
the alert threshold τ and the L3 logit threshold, the test half is only ever scored), as the ground station decodes
them: sensor noise + float32 wire rounding.

- clean:   every holdout launch × several noise seeds, no attack. Alert episodes (rising edges) and runs with any
           alert, i.e. false alerts per flight.
- attacks: each attack starts at 35% of the flight and runs to the end. Detection rate, latency from onset to the
           first alert, the largest altitude lie the operator saw before that alert, and any alert before onset.
           Each learned attacker's shadow detector is the one it trained against: `ppo` (round 1) shadows base,
           `ppo_r2` (round 2) shadows the round-1 hardened detector (models/l3_<schema>_r1.onnx).

Arms-race options: `--l3 models/l3_falcon9_r1.onnx` scores the hardened detector with a specific discriminator,
`--attacks none,ppo_r2` / `--detectors hardened` restrict the runs, `--out` writes somewhere other than
results/detector_eval.json.
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
ATTACKS = ("bias", "drift", "replay", "freeze", "coordinated", "ppo", "ppo_r2")
LEARNED = {"ppo": "", "ppo_r2": "_r2"}   # learned attack -> model suffix; they ignore intensity, so run once
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
    if job["detector"] == "hardened" and job.get("l3"):
        from vbw.detector.l3 import HardenedDetector, L3Model

        det = HardenedDetector(schema, l3=L3Model.from_file(job["l3"]))
    else:
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


def available_attacks(schema_name: str, wanted: list[str]) -> list[str]:
    """`wanted` minus learned attacks whose policy file is missing."""
    return [a for a in wanted
            if a not in LEARNED or (ROOT / "models" / f"ppo_{schema_name}{LEARNED[a]}.zip").exists()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schema", choices=["falcon9"], default="falcon9")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--attacks", default="none," + ",".join(ATTACKS),
                    help="comma list; 'none' means the clean false-alert runs")
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--l3", default=None, help="L3 .onnx (with <stem>_meta.json) for the hardened detector")
    ap.add_argument("--out", default=str(RESULTS / "detector_eval.json"))
    args = ap.parse_args()

    from vbw.detector.data import clean_runs
    from vbw.schema import SCHEMAS_BY_NAME

    schema = SCHEMAS_BY_NAME[args.schema]
    n_hold = len(clean_runs(schema, "holdout", (0,)))
    test = {r.scenario for r in clean_runs(schema, "test", (0,))}
    wanted = [a for a in args.attacks.split(",") if a]
    detectors = [d for d in args.detectors.split(",") if d]
    attacks = available_attacks(schema.name, [a for a in wanted if a != "none"])
    jobs = []
    for d in detectors:
        if "none" in wanted:
            jobs += [dict(schema=schema.name, detector=d, attack="none", intensity=0.0, split="holdout", seed=s, idx=i,
                          l3=args.l3) for s in CLEAN_SEEDS for i in range(n_hold)]
        jobs += [dict(schema=schema.name, detector=d, attack=a, intensity=k, split="holdout", seed=0, idx=i, l3=args.l3)
                 for a in attacks for k in ((1.0,) if a in LEARNED else INTENSITIES) for i in range(n_hold)]
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
        "l3_model": Path(args.l3).name if args.l3 else l3_paths(schema)[0].name,
        "ppo_shadow_detector": {"ppo": "base", "ppo_r2": f"hardened (l3_{schema.name}_r1.onnx)"},
        "detectors": {},
    }
    for d in detectors:
        mine = [r for r in rows if r["detector"] == d]
        clean = [r for r in mine if r["attack"] == "none"]
        entry = {"attacks": {}}
        if clean:
            entry["clean"] = summarize_clean(clean)
            entry["clean_test_only"] = summarize_clean([r for r in clean if r["scenario"] in test])
        for a in attacks:
            by_k = {k: [r for r in mine if r["attack"] == a and r["intensity"] == k] for k in INTENSITIES}
            entry["attacks"][a] = {f"{k:.1f}": summarize_attack(rs) for k, rs in by_k.items() if rs}
        out["detectors"][d] = entry

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    for d, e in out["detectors"].items():
        c = e.get("clean")
        if c:
            print(f"[{d}] clean: {c['runs_with_false_alert']}/{c['runs']} runs alert, {c['false_alert_episodes']} "
                  f"episodes over {c['flight_hours']} h, {c['ms_per_frame']:.2f} ms/frame")
        else:
            print(f"[{d}]")
        for a, by_k in e["attacks"].items():
            s = by_k["1.0"]
            lat = f"{s['mean_latency_s']:.1f}s" if s["mean_latency_s"] is not None else "-"
            low = f"   (@0.4: detect {by_k['0.4']['detection_rate']:.2f})" if "0.4" in by_k else ""
            print(f"   {a:12s} detect {s['detection_rate']:.2f} latency {lat:>7s} lie-before-alert "
                  f"{s['mean_max_lie_before_alert_m']:9.0f} m{low}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
