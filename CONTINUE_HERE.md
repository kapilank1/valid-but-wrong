# CONTINUE HERE: Valid-But-Wrong build state (updated 2026-10-24 03:30 PDT)

> **Latest (03:30):** detector eval is DONE and the hardened detector works: PPO caught in 66 s / 1.95 km lie vs
> 190 s / 12.3 km for base, 0 clean false alerts for both (`results/detector_eval.json`). README + DEVPOST results are
> filled. Live console click-through verified, including PPO → Harden. Steps 1–3 of §6 below are done; see the last
> HANDOFF.md entry for details. Remaining: §6 steps 4–6 (human items, ship assets, git commit, nice-to-haves).
> Demo script: SES-9, speed 10×, start; at ~T+190 s click PPO; watch the lie grow with base silent; click Harden.
> Run the Coordinated rung at intensity 1.0 (at 0.5 it evades both detectors on SES-9).
> Changes since the sections below were written: L3 drops the corridor feature and has its own 40-of-50-frame
> persistence (threshold 6.88); 213 tests pass.

Read this first. It says what the project is, what is built and measured, and exactly what to do next.
`HANDOFF.md` is the chronological log (with the reasoning behind every change). `PLAN.md` is the original hackathon
plan. `docs/CONTRACT.md` + `vbw/schema.py` define the module interfaces.

**To resume in Cursor (required for the SpaceXSI track):** open this folder in Cursor, start Claude Code in Cursor's
terminal, and say: *"Read CONTINUE_HERE.md and HANDOFF.md, then continue from 'What's left'."*

---

## 1. What this is
Cal Hacks 13.0 project. Submit by **Sun Oct 25 2026, 9:00 AM PST** (aim for 8:00).

A man-in-the-middle rewrites telemetry frames and recomputes valid CRCs, so every frame passes the link checks
(sync, CRC-16, sequence, time). Valid-But-Wrong catches the lies anyway: **L1** physics consistency rules, **L2** a GRU
next-step predictor, **L3** a discriminator trained on attacker output. A **PPO** red-team agent learns to evade the
detector; the detector is then hardened against it. Data is real: 31 Falcon 9 launches (webcast telemetry) and NASA
SMAP/MSL spacecraft telemetry.

Prize targets: SpaceXSI (built in Cursor, Grok Bot bonus), Armadin Best Security, Most Technically Advanced,
MLH Tiger Data, MLH Vultr, Cartesia.

## 2. This machine (Windows 11, PowerShell) — read before running anything
- **git is NOT installed.** Nothing from the 2026-10-24 session is committed. Install Git for Windows (or use
  Cursor's prompt to install it), then commit everything before doing more work. `.gitignore` already excludes
  `data/raw`, `data/cache`, `.venv`, `node_modules`.
- **uv needs `--system-certs` for anything that touches the network** (TLS interception here), e.g.
  `uv sync --system-certs`, `uv lock --system-certs`. Plain `uv run ...` works offline once synced.
- Shell is PowerShell 5.1: no `&&`; use `;`. Background jobs: Python `print` is buffered when redirected to a file.
- Hardware: Ryzen AI 7 350 (8C/16T), 31 GB RAM, no GPU used. No Docker, no local Postgres.
- `.claude/launch.json` has `api` (uv run python scripts/serve.py, :8000) and `console` (pnpm --dir apps/console dev,
  :3000) for the Claude desktop preview. Nothing is running now.

```powershell
uv sync --system-certs            # Python 3.12 env
uv run pytest -q                  # expect 211 passed
cd apps/console; pnpm install; pnpm test; cd ../..   # expect 57 passed
uv run python scripts/serve.py    # API + WebSocket on :8000
pnpm --dir apps/console dev       # console on :3000 (?mock=1 runs without the server)
```

## 3. State of every module

| Module | State | Notes |
|---|---|---|
| `vbw/schema.py`, `vbw/link/` | DONE | CCSDS-style frame, CRC-16/CCITT-FALSE, LinkMonitor (L0). |
| `vbw/sources/` | DONE | Falcon9Source (seeded sensor noise), QuadSim, SpacecraftSource (SMAP/MSL), SitlSource (never run vs real ArduPilot). **Hero = `falcon9:ses-9`** (holdout launch; clean peak score 0.47 vs τ 0.66 over 6 seeds). |
| `vbw/detector/` L1 | DONE | 11 Falcon 9 rules incl. NEW `vz_vs_altitude_rate_60s` (threshold 20.4 m/s). Thresholds `models/l1_thresholds.json` (falcon9 + quad). |
| L2 | DONE | `models/l2_falcon9.onnx`, `models/l2_quad.onnx` (+ `_norm.json`). |
| Alerting | DONE | τ calibrated by `scripts/calibrate_alert.py` → `models/l1_alert.json`: **falcon9 τ = 0.66, quad τ = 0.53**, k=3 of n=5. |
| L3 / hardened | TRAINED, NOT YET EVALUATED | `models/l3_falcon9.onnx` + `_meta.json` (logit threshold 28.53). Hardened score = max(noisy-OR(L1,L2), l3_score). |
| `vbw/attacker/` | DONE | Scripted ladder + **real PPO** `models/ppo_falcon9.zip` (v3, 1.5M steps vs base). `models/ppo_falcon9_stub.zip` is an obsolete stub; safe to delete. |
| `vbw/pipeline/`, `vbw/store/`, `infra/` | DONE | FastAPI + WS, hot-swap, Timescale writer. SQL has **never run on a real TimescaleDB** (needs DATABASE_URL). Linux torch now CPU-only via `pyproject.toml [tool.uv.sources]` (Docker image not rebuilt/verified). |
| `vbw/bench/` | DONE | SMAP/MSL numbers below; `docs/BENCH.md` filled in. |
| `apps/console/` | BUILDS, PARTLY VERIFIED | `pnpm build` OK, 57 vitest pass. Live-tested: start SES-9, Bias (alert 0.2 s after onset), Clear, Drift. **Not yet clicked live: Replay, Coordinated, PPO, Harden.** Gauge shows calibrated τ (catalog `thresholds` per schema). |
| `README.md`, `docs/DEVPOST.md` | DRAFTED | Contain placeholders `RESULTS_SECTION` (README) and `DEVPOST_RESULTS` (DEVPOST) to fill from `results/detector_eval.json`. |

## 4. Measured numbers (all from files in `results/` or `models/`; quote only these)

**Clean false alerts (base, τ 0.66):** 0 of 9 clean test runs (45,750 frames; calibrate_alert), 0 of 12 holdout runs
(seeds 0,1), 0 of 6 holdout runs in `results/attacks_eval.json`.

**Base detector vs attacks** (`results/attacks_eval.json` / `results/ppo_falcon9_metrics.json`; 6 holdout launches,
attack starts at 35% of flight ≈ 190 s, above the atmosphere, intensity 1.0):

| Attack | Detected | Mean latency | Mean max altitude lie before alert |
|---|---|---|---|
| bias | 6/6 | 0.6 s | 3.0 km (it's a 3 km step) |
| replay | 6/6 | 0.6 s | 5.4 km |
| freeze | 6/6 | 0.6 s | 0.6 km |
| coordinated (scripted) | 6/6 | 5.3 s | 0.13 km |
| drift 30 m/s | 6/6 | 137 s | 4.1 km |
| **PPO v3** | 6/6 eventually | **190 s** | **12.3 km** (3/6 launches: 312–320 s, 23–24 km) |

PPO learned the *coordinated* lie (ramps vz to ~+120 m/s and integrates altitude to match), so L1/L2 stay ≤ 0.04 for
the first ~100 s while the lie reaches ~3.7 km; it is finally caught by speed-vs-components when its speed
bookkeeping drifts. Live demo note: an attack started early (in the atmosphere, e.g. T+70 s) is caught much faster by
the dynamic-pressure-vs-altitude rule (drift caught in 15 s live).

**L3 (hardened) validation, per forged window at the calibrated threshold** (`models/l3_falcon9_meta.json`):
freeze 99%, replay 98%, coordinated 86%, **PPO 59%**, bias 13%, drift 10%; clean val windows over threshold: 0%.
The threshold (28.5) is set by `nrol-76`'s entry-burn OCR glitch (next highest clean flight: 11.3).
**Whether hardened catches PPO on holdout launches end to end, and hardened's clean false alerts, are NOT measured
yet** — that is step 1 below.

**SMAP/MSL benchmark** (event-level, `results/bench_summary.json`): SMAP F1 76.3%, MSL 64.3%, combined
P 78.9 / R 67.6 / F1 72.8%. Telemanom (paper): combined F1 83.6%. We are below it; say so.

## 5. Key findings (don't relearn these)
- **The old "0 of 31 false alerts" claim was wrong.** At τ 0.5 base alerted on abs-2a, jason-3 (train) and nrol-76 (val):
  webcast OCR stretches where altitude lags vz then jumps; L1 ≈ 0.28 and L2 ≈ 0.39 combine (noisy-OR) over 0.5.
  τ is now calibrated on the combined score (train+val only) → 0.66.
- **Never feed noise-free CSVs to the detector** (frozen_channel fires). Use `vbw.detector.data.clean_runs(...)`.
- **PPO reward pitfalls:** penalising raw score every step made "do nothing"/"die fast" optimal. Now: only score above
  a 0.3 margin costs, picture_gain 2000, alert_penalty 100. Letting PPO touch vx/downrange got it caught in seconds
  (downrange-rate rule is ~1 m/s tight), so it controls altitude, vz, speed only. Env warmup 620 frames (longest L1
  lookback is 610).
- **10 s vs 60 s windows:** clean |vz − dalt/dt| max is 81.9 m/s at 10 s but 13.6 m/s at 60 s (OCR glitches cancel).
- **L3 confound:** with all attacks starting at 35%, L3 learned "late flight = forged" and saturated on clean stage-1
  entry/landing burns. Fixed by spreading training onsets (10/40/70%) and phase-balancing (`features.phase_balance`).
  L3 is its own alert stream (max), calibrated by 3-of-5 persistence on clean train+val flights, because L1+L2 already
  sit at 0.655 on abs-2a and any noisy-OR addition would false-alert.
- Quad: `quad_scenarios()` used to include `sitl:udp:14550` (blocks on a socket) — fixed. Quad test split = held mission
  (hover) on seeds +1000.
- SMAP original S3 bucket 403s; `scripts/fetch_smap.py` uses the TranAD mirror.

## 6. What's left (in priority order)
1. **Detector eval** (script ready, never run): `uv run python scripts/eval_detector.py --schema falcon9 --workers 12`
   (~5 min) → `results/detector_eval.json`: clean false alerts per flight (base vs hardened), detection rate / latency /
   lie-before-alert per attack at intensity 0.4 and 1.0, PPO vs both detectors, ms per frame.
   - If **hardened has clean false alerts**: raise `LOGIT_MARGIN` in `scripts/train_l3.py` or check which flight.
   - If **hardened doesn't catch PPO well on holdout**: the honest story still works (PPO lies 12 km on average past
     base); consider a vz corridor or longer L3 context. Don't overfit to holdout.
2. **Fill the results** into `README.md` (`RESULTS_SECTION`) and `docs/DEVPOST.md` (`DEVPOST_RESULTS`) from
   `results/detector_eval.json` + section 4. Never invent numbers.
3. **Console live click-through** against the server: Start (SES-9) → Bias → Drift → Replay → Coordinated → PPO →
   Harden detector. Check the PPO rung is enabled (it reads `available` from the catalog), the L3 P(forged) readout
   updates after Harden, and layout at 1440×900. Restart the API after model changes (no auto-reload).
4. **Human-only items:** `DATABASE_URL` (Tiger Data) then `uv run python scripts/db_init.py --check`;
   `CARTESIA_API_KEY` in `apps/console/.env.local`; Vultr account if deploying (`docs/DEPLOY_VULTR.md`).
5. **Ship:** Devpost image (console screenshot mid-attack), recorded backup demo video, screenshots of Cursor + Grok
   usage for SpaceXSI, commit everything. Update the resume bullet to match what was built (ONNX Runtime, not TensorRT;
   no Kubernetes).
6. Nice to have: verify the CPU-torch Docker image builds; delete `models/ppo_falcon9_stub.zip`; quad PPO.

Cut order if behind: Docker → quad → console polish. Never cut: the live Falcon 9 attack-and-catch demo and the
SMAP/MSL benchmark number.

## 7. Reproduce the whole pipeline (in this order; each step's outputs feed the next)
```powershell
uv run python scripts/calibrate_l1.py --schema falcon9          # models/l1_thresholds.json
uv run python scripts/train_l2.py --schema falcon9              # models/l2_falcon9.onnx
uv run python scripts/calibrate_alert.py --schema falcon9       # models/l1_alert.json (tau)
uv run python scripts/calibrate_l1.py --schema quad
uv run python scripts/train_l2.py --schema quad
uv run python scripts/calibrate_alert.py --schema quad --seeds 10
uv run python scripts/train_ppo.py --schema falcon9 --detector base --timesteps 1500000 --n-envs 12 --episode-len 1500   # ~25 min
uv run python scripts/make_forged_dataset.py --schema falcon9   # data/cache/forged_*.npz + results/attacks_eval.json (~6 min)
uv run python scripts/train_l3.py --schema falcon9              # models/l3_falcon9.onnx (~5 min)
uv run python scripts/eval_detector.py --schema falcon9         # results/detector_eval.json
```
If any L1 rule or τ changes, retrain PPO (it learns the detector), then forge → L3 → eval.
