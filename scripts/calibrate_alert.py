"""Calibrate the alert threshold τ of the k-of-n alerting on clean runs and save models/l1_alert.json.

    uv run python scripts/calibrate_alert.py --schema falcon9
    uv run python scripts/calibrate_alert.py --schema quad --seeds 10

Every layer's score is 0.5 at its own clean-data limit, but the noisy-OR of two layers that are each just under it
can still cross 0.5 (e.g. a webcast-OCR stretch where altitude lags vz, then jumps to catch up: L1 0.28 + L2 0.39).
So τ is set from the combined score itself: the smallest value above which no clean train or val run (several noise
seeds) would raise an alert, i.e. the max over runs of the k-th largest score in any n-frame window, plus a margin.
Test runs are then replayed at that τ to report clean false alerts on launches calibration never saw.
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from vbw.detector.alerting import AlertConfig, alert_series, never_alert_threshold
from vbw.detector.data import clean_runs
from vbw.schema import SCHEMAS_BY_NAME

CAL_SEEDS = (0, 1)
TEST_SEEDS = (0, 1, 2)
MARGIN = 0.005


def _score_run(job: tuple[str, str, int, int]) -> tuple[str, int, np.ndarray]:
    from vbw.detector.registry import make_detector
    from vbw.schema import Frame, L0Result

    schema_name, split, seed, idx = job
    schema = SCHEMAS_BY_NAME[schema_name]
    run = clean_runs(schema, split, (seed,))[idx]
    det = make_detector("base", schema)
    scores = np.empty(len(run.t))
    for i in range(len(run.t)):
        f = Frame("cal", schema.schema_id, i & 0xFFFF, float(run.t[i]),
                  {c: float(run.X[i, j]) for j, c in enumerate(schema.channels)})
        scores[i] = det.update(f, L0Result()).score
    return run.scenario, seed, scores


def score_runs(schema_name: str, split: str, seeds: tuple[int, ...], workers: int) -> list[tuple[str, int, np.ndarray]]:
    n = len(clean_runs(SCHEMAS_BY_NAME[schema_name], split, (seeds[0],)))
    jobs = [(schema_name, split, s, i) for s in seeds for i in range(n)]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_score_run, jobs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schema", choices=["falcon9", "quad"], default="falcon9")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=len(CAL_SEEDS),
                    help="noise seeds per calibration run (the quad sim is cheap, so use more there)")
    args = ap.parse_args()
    base = AlertConfig.load(args.schema)
    cal_seeds = tuple(range(args.seeds))

    cal = score_runs(args.schema, "train", cal_seeds, args.workers) + score_runs(args.schema, "val", cal_seeds, args.workers)
    peaks = sorted(((never_alert_threshold(s, base), scen, seed) for scen, seed, s in cal), reverse=True)
    tau = max(0.5, math.ceil((peaks[0][0] + MARGIN) * 100) / 100)
    cfg = replace(base, tau=tau)
    print(f"{args.schema}: {len(cal)} calibration runs; highest clean peaks:")
    for p, scen, seed in peaks[:5]:
        print(f"  {p:.3f} {scen} seed={seed}")

    test = score_runs(args.schema, "test", TEST_SEEDS, args.workers)
    test_alerts = [scen for scen, _, s in test if any(alert_series(s, cfg))]
    frames = int(sum(len(s) for _, _, s in test))
    print(f"tau={tau:.2f} (k={cfg.k}/n={cfg.n}); clean test runs alerting: {len(test_alerts)}/{len(test)} ({frames} frames)")

    cfg.meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "tau = ceil(max clean k-of-n peak over train+val runs + margin, 2 dp), floor 0.5",
        "margin": MARGIN, "calibration_seeds": list(cal_seeds), "calibration_runs": len(cal),
        "top_clean_peaks": [{"peak": round(p, 4), "scenario": scen, "seed": seed} for p, scen, seed in peaks[:5]],
        "test_seeds": list(TEST_SEEDS), "test_runs": len(test), "test_frames": frames,
        "test_runs_alerting": len(test_alerts), "test_alerting_scenarios": sorted(set(test_alerts)),
    }
    cfg.save(args.schema)
    print("wrote models/l1_alert.json")


if __name__ == "__main__":
    main()
