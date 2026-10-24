"""Assemble the attacker-vs-defender rounds into results/arms_race.json from the evaluation files.

    uv run python scripts/arms_race_report.py

Inputs (all on the 6 holdout launches, attack from 35% of flight, written by scripts/eval_detector.py):
    results/detector_eval.json       round 1: base + hardened r1 (l3_falcon9_r1) vs scripted attacks and PPO r1
    results/detector_eval_r2a.json   PPO r2 vs base and hardened r1  (--attacks ppo_r2 --l3 models/l3_falcon9_r1.onnx)
    results/detector_eval_r2.json    everything vs base and hardened r2 (--l3 models/l3_falcon9_r2.onnx)
PPO r1 was trained against base; PPO r2 against hardened r1, and its shadow stays hardened r1 in every evaluation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _row(attacker: str, defender: str, source: str, ev: dict, detector: str, attack: str) -> dict:
    s = ev["detectors"][detector]["attacks"][attack]["1.0"]
    clean = ev["detectors"][detector].get("clean")
    return {
        "attacker": attacker, "defender": defender, "source": source,
        "detection_rate": s["detection_rate"],
        "mean_latency_s": None if s["mean_latency_s"] is None else round(s["mean_latency_s"], 1),
        "mean_max_lie_before_alert_m": round(s["mean_max_lie_before_alert_m"]),
        "clean_false_alert_runs": None if clean is None else f"{clean['runs_with_false_alert']}/{clean['runs']}",
    }


def main() -> None:
    r1, r2a, r2 = _load("detector_eval.json"), _load("detector_eval_r2a.json"), _load("detector_eval_r2.json")
    meta = json.loads((ROOT / "models" / "l3_falcon9_meta.json").read_text())
    rows = [
        _row("PPO r1 (trained vs base)", "base", "detector_eval_r2.json", r2, "base", "ppo"),
        _row("PPO r1 (trained vs base)", "hardened r1", "detector_eval.json", r1, "hardened", "ppo"),
        _row("PPO r2 (trained vs hardened r1)", "base", "detector_eval_r2a.json", r2a, "base", "ppo_r2"),
        _row("PPO r2 (trained vs hardened r1)", "hardened r1", "detector_eval_r2a.json", r2a, "hardened", "ppo_r2"),
        _row("PPO r2 (trained vs hardened r1)", "hardened r2", "detector_eval_r2.json", r2, "hardened", "ppo_r2"),
        _row("PPO r1 (trained vs base)", "hardened r2", "detector_eval_r2.json", r2, "hardened", "ppo"),
    ]
    # Clean false alerts per defender come from the full runs (the r2a run had no clean flights).
    clean = {"base": r2["detectors"]["base"]["clean"], "hardened r1": r1["detectors"]["hardened"]["clean"],
             "hardened r2": r2["detectors"]["hardened"]["clean"]}
    for row in rows:
        c = clean[row["defender"]]
        row["clean_false_alert_runs"] = f"{c['runs_with_false_alert']}/{c['runs']}"
    out = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "6 holdout launches; attack starts at 35% of flight; clean = 6 launches x 3 noise seeds",
        "rounds": rows,
        "clean_flight_hours": clean["hardened r2"]["flight_hours"],
        "shipped_l3": "r2" if meta.get("logit_threshold") == json.loads(
            (ROOT / "models" / "l3_falcon9_r2_meta.json").read_text()).get("logit_threshold") else "r1",
        "ship_rule": "keep r2 only if clean false alerts stay 0/18 and PPO r1 latency is within ~20% of r1's",
        "l3_files": {"r1": "models/l3_falcon9_r1.onnx", "r2": "models/l3_falcon9_r2.onnx",
                     "deployed": "models/l3_falcon9.onnx"},
    }
    (RESULTS / "arms_race.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        lat = f"{r['mean_latency_s']:.1f}s" if r["mean_latency_s"] is not None else "-"
        print(f"{r['attacker']:34s} vs {r['defender']:12s} detect {r['detection_rate']:.2f} latency {lat:>7s} "
              f"lie {r['mean_max_lie_before_alert_m']:6d} m  clean FA {r['clean_false_alert_runs']}")
    print(f"shipped L3: {out['shipped_l3']}; wrote results/arms_race.json")


if __name__ == "__main__":
    main()
