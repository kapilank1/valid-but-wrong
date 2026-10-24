"""Calibrate L1 physics-rule thresholds on clean training runs and save models/l1_thresholds.json.

    uv run python scripts/calibrate_l1.py --schema falcon9
    uv run python scripts/calibrate_l1.py --schema quad

For each rule: threshold = margin × max |residual| over clean training runs (several noise seeds), so a clean
training flight never trips a rule. Falcon 9 also fits a tabulated atmosphere (log ρ vs altitude, from the q channel)
and a mission-time corridor (altitude/speed envelope), whose threshold comes from leave-one-launch-out excursions.
Held-out launches are then scored to show how often each rule fires on clean data it never saw.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from vbw.detector.data import Run, clean_runs
from vbw.detector.physics import (
    F,
    RULES_BY_SCHEMA,
    THRESHOLDS_PATH,
    L1Config,
    NOMINAL_RATE_HZ,
    f9_corridor,
)
from vbw.schema import SCHEMAS_BY_NAME, ChannelSchema

TRAIN_SEEDS = (0, 1)
EVAL_SEEDS = (100,)
CORRIDOR_BIN_S = 2.0
# The webcast angle readout lags by several degrees during stage-1 flip / entry burns (seen on the val launch
# nrol-76, ratio 1.29 at margin 1.25). Angle is a low-value channel, so it gets a wider margin rather than a gate.
MARGIN_OVERRIDES = {"angle_vs_velocity": 2.0}


def fit_atmosphere(runs: list[Run], step: float = 500.0) -> dict:
    """Median log(2q/v²) per altitude bin from noise-free training data; the q channel encodes the source's atmosphere."""
    alt = np.concatenate([r.truth[:, F["altitude"]] for r in runs])
    v = np.concatenate([r.truth[:, F["speed"]] for r in runs])
    q = np.concatenate([r.truth[:, F["q"]] for r in runs])
    ok = (v > 30) & (q > 0.01)
    log_rho, h = np.log(2 * q[ok] / v[ok] ** 2), alt[ok]
    grid, table = [], []
    for g in np.arange(0.0, h.max() + step, step):
        sel = np.abs(h - g) < step / 2
        if sel.sum() >= 5:
            grid.append(float(g))
            table.append(float(np.median(log_rho[sel])))
    return {"alt": grid, "log_rho": table}


def fit_corridor(runs: list[Run]) -> dict:
    t_end = max(r.t[-1] for r in runs)
    grid = np.arange(0.0, t_end + CORRIDOR_BIN_S, CORRIDOR_BIN_S)
    out: dict[str, list[float] | float] = {"t": grid.tolist()}
    for ch in ("altitude", "speed"):
        lo, hi = np.full(len(grid), np.inf), np.full(len(grid), -np.inf)
        for r in runs:
            x = np.interp(grid, r.t, r.X[:, F[ch]], left=np.nan, right=np.nan)
            ok = np.isfinite(x)
            lo[ok], hi[ok] = np.minimum(lo[ok], x[ok]), np.maximum(hi[ok], x[ok])
        # Bins past every launch's end keep the last envelope.
        last = np.where(np.isfinite(lo))[0][-1]
        lo[last:], hi[last:] = lo[last], hi[last]
        out[f"{ch}_lo"], out[f"{ch}_hi"] = lo.tolist(), hi.tolist()
    out["altitude_floor"], out["speed_floor"] = 5000.0, 200.0
    out["t_all_end"] = float(min(r.t[-1] for r in runs))
    return out


def corridor_loo_max(runs: list[Run]) -> list[float]:
    """Max corridor excursion of each launch against a corridor built from the other launches."""
    by_launch: dict[str, list[Run]] = {}
    for r in runs:
        by_launch.setdefault(r.scenario, []).append(r)
    worst = []
    for name, own in by_launch.items():
        others = [r for r in runs if r.scenario != name]
        params = {"corridor": fit_corridor(others)}
        worst.append(max(float(np.nanmax(f9_corridor(r.X, r.t, params))) for r in own))
    return worst


def residual_stats(values: np.ndarray) -> dict[str, float]:
    a = np.abs(values[np.isfinite(values)])
    if a.size == 0:
        return {"p99": 0.0, "p99_9": 0.0, "p99_99": 0.0, "max": 0.0}
    return {"p99": float(np.percentile(a, 99)), "p99_9": float(np.percentile(a, 99.9)),
            "p99_99": float(np.percentile(a, 99.99)), "max": float(a.max())}


def rule_series(schema: ChannelSchema, runs: list[Run], config: L1Config) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for rule in RULES_BY_SCHEMA[schema.name]:
        out[rule.name] = [rule.residual(r.X, r.t, config.params) for r in runs]
    return out


def calibrate(schema: ChannelSchema, margin: float) -> dict:
    train = clean_runs(schema, "train", TRAIN_SEEDS)
    print(f"{schema.name}: {len(train)} training runs ({len({r.scenario for r in train})} scenarios x {len(TRAIN_SEEDS)} seeds)")
    params: dict = {}
    if schema.name == "falcon9":
        params["atmosphere"] = fit_atmosphere(train)
        params["corridor"] = fit_corridor(train)
    config = L1Config(schema.name, {}, params, True)

    series = rule_series(schema, train, config)
    thresholds, stats = {}, {}
    for rule in RULES_BY_SCHEMA[schema.name]:
        st = residual_stats(np.concatenate(series[rule.name]))
        if rule.name == "corridor":
            loo = corridor_loo_max(train)
            st["loo_max_per_launch_p50"] = float(np.median(loo))
            st["max"] = float(max(loo))
        rule_margin = MARGIN_OVERRIDES.get(rule.name, margin)
        if rule.name == "frozen_channel":
            # Noise makes clean runs of identical values ~1-2 frames; require a couple of seconds.
            thresholds[rule.name] = float(max(rule_margin * st["max"], 2.0 * NOMINAL_RATE_HZ[schema.name]))
        else:
            thresholds[rule.name] = float(max(rule_margin * st["max"], 1e-6))
        stats[rule.name] = st
        print(f"  {rule.name:24s} max={st['max']:9.3f} p99.99={st['p99_99']:9.3f} -> threshold {thresholds[rule.name]:9.3f} {rule.unit}")
    config.thresholds = thresholds

    held = {}
    for split in ("val", "test"):
        runs = clean_runs(schema, split, EVAL_SEEDS)
        held_series = rule_series(schema, runs, config)
        per_rule = {}
        for rule in RULES_BY_SCHEMA[schema.name]:
            frames = [int(np.sum(np.abs(s) > thresholds[rule.name])) for s in held_series[rule.name]]
            per_rule[rule.name] = {"frames_over": int(sum(frames)), "runs_with_any": int(sum(f > 0 for f in frames)),
                                   "max_ratio": float(max(np.nanmax(np.abs(s)) for s in held_series[rule.name]) / thresholds[rule.name])}
        held[split] = {"runs": [r.scenario for r in runs], "frames": int(sum(len(r.t) for r in runs)), "rules": per_rule}
        print(f"  {split}: " + ", ".join(f"{k}={v['frames_over']}" for k, v in per_rule.items() if v["frames_over"]) or "no frames over")

    return {
        "thresholds": thresholds,
        "params": params,
        "rate_hz": NOMINAL_RATE_HZ[schema.name],
        "calibration": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "margin": margin,
            "margin_overrides": MARGIN_OVERRIDES,
            "train_scenarios": sorted({r.scenario for r in train}),
            "train_seeds": list(TRAIN_SEEDS),
            "residual_stats": stats,
            "heldout_clean": held,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schema", choices=sorted(RULES_BY_SCHEMA), default="falcon9")
    ap.add_argument("--margin", type=float, default=1.5)
    args = ap.parse_args()
    schema = SCHEMAS_BY_NAME[args.schema]
    blob = json.loads(THRESHOLDS_PATH.read_text()) if THRESHOLDS_PATH.exists() else {}
    blob[schema.name] = calibrate(schema, args.margin)
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(blob, indent=1))
    print(f"wrote {THRESHOLDS_PATH}")


if __name__ == "__main__":
    main()
