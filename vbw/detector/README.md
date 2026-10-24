# vbw.detector

Streaming detector for CRC-valid but physically wrong telemetry.

| Layer | Module | What it does |
|-------|--------|--------------|
| L1 | `physics.py` | Pure physics-consistency rules over a ring buffer of recent frames. Thresholds in `models/l1_thresholds.json`. |
| L2 | `l2.py` | Small GRU that predicts the next frame from the last 20; ONNX Runtime inference; robust z-scores of the residuals. |
| score / alert | `base.py` | `BaseDetector(schema)`: combines L1 + L2, persistence + hysteresis alerting, warm-up handling. |
| L3 | `l3.py` | `HardenedDetector(schema)`: base + a discriminator P(forged) over a window of frames. Falls back to base when no model. |
| spacecraft | `spacecraft.py` | Rolling robust z-score detector for the single-channel SMAP/MSL schema. |
| registry | `registry.py` | `list_detectors()`, `make_detector("base" \| "hardened", schema)`. |

## Forged-window dataset (`data/cache/forged_*.npz`) — contract with vbw/attacker

The attacker writes one or more `.npz` files to `data/cache/` named `forged_<schema>_<anything>.npz`
(e.g. `forged_falcon9_ppo.npz`). `scripts/train_l3.py` globs them. Arrays:

| key | dtype / shape | meaning |
|-----|---------------|---------|
| `X` | float32 `[N, W, C]` | **reported** channel values (what the ground decoded), schema channel order, SI units, raw (not normalized) |
| `t` | float64 `[N, W]` | mission time of each frame, seconds |
| `y` | int8 `[N]` | `1` if the **last** frame of the window was altered by the attacker (attack active), else `0` |
| `attack` | str `[N]` | attack name per window (`"none"` for clean windows) |
| `scenario` | str `[N]` | scenario id the window came from, e.g. `"falcon9:ses-10"` (used to split by launch, so no leakage) |
| `schema` | str, 0-d | `"falcon9"` or `"quad"` |
| `window` | int, 0-d | `W` |

Rules:
- `W = vbw.detector.features.l3_window(schema)` (**160 frames** for both falcon9 and quad = 16 s of Falcon 9 at 10 Hz).
  The L1 rules integrate over 10 s, so the first ~120 frames are history and the discriminator looks at the per-frame
  features of the last `L3_CONTEXT = 40` frames. Windows are consecutive frames from one run; never span a reset.
  Any stride (10 is a good default).
- Clean windows are optional in your files; `train_l3.py` also makes its own clean windows from the clean sources.
- Generate forged training data on **training** launches only: `vbw.detector.splits.train_slugs()` /
  `holdout_slugs()` (same split the detector is calibrated on). Put holdout runs in a separate file with `holdout` in its name
  (e.g. `forged_falcon9_ppo_holdout.npz`) if you want them used for evaluation instead of training.
- Write with `np.savez_compressed(path, X=..., t=..., y=..., attack=np.array(names), scenario=np.array(ids), schema=np.array("falcon9"), window=np.array(W))`.
  `vbw.detector.features.save_forged_npz(...)` / `load_forged_npz(...)` do this for you and validate shapes.

Shared feature function (both the attacker and the hardened detector use it, so features match exactly):

```python
from vbw.detector.features import l3_features
F = l3_features(X, t, schema)    # float32 [N, L3_CONTEXT, F]: L1 residual ratios, L2 z-scores, normalized diffs
```
