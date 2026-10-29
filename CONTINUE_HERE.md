# CONTINUE HERE: Valid-But-Wrong build state (updated 2026-10-24 ~15:45 PDT)

> **Docker + Kubernetes (2026-10-28, on the Mac, not committed yet).** Compose stack (`infra/docker-compose.yml`) builds
> and runs on arm64. Kubernetes manifests in `infra/k8s/` (Docker Desktop cluster, `kubectl apply -k infra/k8s`):
> TimescaleDB StatefulSet + PVC, server (1 replica, Recreate, waits for the db), console x2, and the **fleet detector
> workers** (`vbw/fleet/`, StatefulSet): 1,200 Falcon 9 vehicles sharded by rendezvous hashing over the live worker
> pods, each pod scoring its vehicles' frames (CRC link + L1/L2) in real time; `scripts/fleet_report.py` measures
> them. `results/fleet_scale.json` (AC power): 2 pods 4,172 frames/s and 4 pods 7,844 (both behind); 8 pods
> 12,002 of 12,000 needed, real time, p99 1.0 ms/frame; over full flights 115/115 attacked flights detected (median
> 0.8 s after onset, scripted attacks at intensity 1.0) and 0 false alerts in 1,085 clean flights. Battery-throttled
> runs are kept apart in `results/fleet_scale_battery.json`: don't quote them. Workers are scaled to 0 to free the CPU;
> `kubectl apply -k infra/k8s` brings back 8. 33 fleet tests pass.

> **NEXT TASK (in Cursor, with Grok): the project website for judges.** Everything the site needs (story, every
> number with its source file, sections, visual direction, rules) is in **`docs/WEBSITE_BRIEF.md`**. Build it in
> `apps/site/`; don't touch the detector/console/edge code.
>
> **C++ edge monitor (`edge/`): PAUSED by the user, partly done.** Builds with `edge\build.ps1 -Test`; 3/3 CTest suites
> pass; bit-exact parity with the Python detector on 16,766 frames; 30 µs/frame, 32,550 frames/s
> (`results/edge_bench.json`). Remaining work and how to resume: status block at the top of `edge/SPEC.md`.
> Toolchain installed for it: VS Build Tools 2026 C++ + Windows 11 SDK 10.0.26100; ONNX Runtime 1.31.0 C++ and MAVLink
> headers in gitignored `edge/third_party/` (re-download URLs in `edge/SPEC.md`).
>
> **Latest (Sat 14:20):** Tiger Data and Cartesia are live and verified with the user's credentials; the repo is on
> GitHub (https://github.com/kapilank1/valid-but-wrong, branch `main`). Vultr is **skipped** (the user has no credits).
> What's left: the website, ship assets (Devpost image, backup video, Cursor/Grok screenshots), and resuming the C++
> monitor if the user wants.
> - **Tiger Data verified**: `db_init.py --check` passed on Tiger Cloud (TimescaleDB 2.31.0); a live ▶ Demo wrote
>   8,961 rows / 64 batches / 0 errors; `/api/runs`, timeline (300 buckets from `verdicts_1s`) and `/api/analytics`
>   answer from Tiger Cloud. Stop only pauses a run; the run row is closed on Reset / new scenario / end of flight.
> - **Cartesia verified**: all 4 alerts of a live ▶ Demo spoken via `/api/tts` (200). Norton's TLS inspection broke
>   server-side fetch (`UNABLE_TO_VERIFY_LEAF_SIGNATURE`); fixed by trusting the OS cert store at server start
>   (`apps/console/src/instrumentation.ts` → `instrumentation-node.ts`, `src/lib/systemCa.ts`). Restart the console
>   after changing `.env.local`.
> - Credentials live in gitignored `.env` (`DATABASE_URL`) and `apps/console/.env.local` (`CARTESIA_API_KEY`). Never
>   print or commit them.
> - **Hardened r2 ships** (`models/l3_falcon9.onnx` = r2; r1 kept as `models/l3_falcon9_r1.*`). On 6 held-out
>   launches it catches PPO r1 in 41 s / 0.72 km (base: 190 s / 12.3 km) and PPO r2 in 36 s / 0.38 km, 0/18 clean
>   false alerts (`results/detector_eval_r2.json`).
> - **Arms race round 2 done** (`results/arms_race.json`): PPO r2, retrained against hardened r1, survived it 2.5×
>   longer (167 s vs 66 s); retraining L3 on its output gives hardened r2.
> - **TensorRT done** (`docs/TENSORRT.md`, `results/trt_bench.json`, AC power, shipped L3 r2): FP16 scores 16,384
>   streams in ~15.9 ms (L2 6.7 ms vs CPU 232 ms = 35×; L3 9.2 ms vs 80 ms = 8.7×), 0 decisions changed; INT8 measured
>   and not worth it (slower than FP16, flips 3.0% of L3 decisions). Battery run: `results/trt_bench_battery.json`.
> - **Console demo verified live**: ▶ Demo (title row) runs bias → PPO → harden on SES-9 at 10×. With policy
>   preloading at server startup (`vbw/attacker/env.py` `load_policy` / `preload_policies`), PPO goes active at
>   T+190.0 s, the lie reaches 3.70 km with base silent, and hardened r2 alerts 0.1 s after the hot-swap (T+288.6).
>   PPO round 2 rung verified live too: 2.44 km lie at T+349 with base at 0.03; hardened r2 caught it 43 s after swap.
> - **Vultr deploy prep done** (`docs/INTEGRATIONS.md`, `scripts/deploy_vultr.ps1`) but not run: needs
>   `VULTR_API_KEY` in `.env` and the user's OK on the price.
> - **git**: pushed to `origin` = https://github.com/kapilank1/valid-but-wrong (`main`). Commit, then `git push`.
> - Tests: 221 passed + 1 skipped (the GPU test without TensorRT); console 75.

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
- **git** installed 2026-10-24 (winget, 2.56); latest commit `7264e60`. No remote yet: create a GitHub repo and push
  before submitting. If `git` isn't on PATH in an old shell, use `"C:\Program Files\Git\cmd\git.exe"`.
  `models/ppo_falcon9.zip` and `models/ppo_falcon9_r2.zip` are tracked via `.gitignore` exceptions; `models/trt/`
  (engines) is ignored. PowerShell 5.1 mangles here-string commit messages: use `git commit -F <file>`.
- **uv needs `--system-certs` for anything that touches the network** (TLS interception here), e.g.
  `uv sync --system-certs`, `uv lock --system-certs`. Plain `uv run ...` works offline once synced.
- Shell is PowerShell 5.1: no `&&`; use `;`. Background jobs: Python `print` is buffered when redirected to a file.
- Hardware: Ryzen AI 7 350 (8C/16T), 31 GB RAM, **NVIDIA RTX 5060 Laptop GPU** (8 GB, compute 12.0, driver 576.76,
  CUDA 12.9; used only by the TensorRT bench). No Docker, no local Postgres. **On battery the CPU and GPU are throttled**
  (benchmarks ~1.3–1.8× slower): plug in before timing anything.
- GPU packages are not in the default env: `uv run --with tensorrt-cu12 --with "cuda-python>=12.6,<13" --with
  onnxconverter-common ...` (`cuda-python` 13.x fails with this driver).
- `.claude/launch.json` has `api` (uv run python scripts/serve.py, :8000) and `console` (pnpm --dir apps/console dev,
  :3000) for the Claude desktop preview. Restart the API after any model change (no auto-reload).

```powershell
uv sync --system-certs            # Python 3.12 env
uv run pytest -q                  # expect 221 passed, 1 skipped
cd apps/console; pnpm install; pnpm test; cd ../..   # expect 73 passed
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
| L3 / hardened | DONE, r2 SHIPPED | `models/l3_falcon9.onnx` + `_meta.json` = **hardened r2** (logit threshold 4.33; corridor feature excluded; own alert stream, 40 of 50 frames). r1 backup `models/l3_falcon9_r1.*`, r2 copy `models/l3_falcon9_r2.*`. Verdict score = max(noisy-OR(L1,L2), l3_score). |
| TensorRT | DONE | `vbw/detector/trt.py`, `scripts/trt_bench.py`, `docs/TENSORRT.md`. Engines in `models/trt/` (gitignored, rebuilt by the bench). |
| `vbw/attacker/` | DONE | Scripted ladder + **PPO r1** `models/ppo_falcon9.zip` (vs base) + **PPO r2** `models/ppo_falcon9_r2.zip` (vs hardened r1; attack `ppo_r2`, shadow = hardened r1). `models/ppo_falcon9_stub.zip` is an obsolete stub (gitignored). |
| `vbw/pipeline/`, `vbw/store/`, `infra/` | DONE, UNDEPLOYED | FastAPI + WS, hot-swap, Timescale writer (threaded psycopg store on Windows). `scripts/db_init.py --check` is end to end. SQL has **never run on a real TimescaleDB** (needs DATABASE_URL). Docker images not built here (no Docker); `scripts/deploy_vultr.ps1` builds them on the VM. |
| `vbw/bench/` | DONE | SMAP/MSL numbers below; `docs/BENCH.md` filled in. |
| `apps/console/` | DONE, VERIFIED LIVE | 73 vitest, build OK. ▶ Demo button, "LINK 100% VALID / PICTURE OFF BY x km" readout, PPO round 2 rung, fits 1440×900. Bias, Drift, Replay, Coordinated, PPO, Harden and the Demo verified live; **the PPO round 2 rung has not been clicked live yet.** |
| `README.md`, `docs/DEVPOST.md`, `docs/TENSORRT.md` | WRITTEN | Numbers from `results/detector_eval_r2.json`, `results/arms_race.json`, `results/trt_bench*.json`. |

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

**Hardened r2 (shipped) vs attacks** (`results/detector_eval_r2.json`, same protocol): bias 0.6 s, replay 0.6 s,
freeze 0.6 s, coordinated 3.6 s / 0.09 km, drift 18.5 s / 0.55 km, **PPO r1 41 s / 0.72 km, PPO r2 36 s / 0.38 km**, all
6/6. At intensity 0.4: bias and drift 6/6 (base 3/6). Clean: 0/18 runs (2.58 h), test half 0/9 (1.27 h), for both
detectors; no run alerted before its attack. 1.1 ms/frame (base), 1.3 ms/frame (hardened). SES-9 PPO r1: base
315 s / 23.8 km, hardened r2 34 s / 0.59 km.

**Arms race** (`results/arms_race.json`): PPO r1 vs base 190 s / 12.3 km → vs hardened r1 66 s / 1.95 km; PPO r2 vs
base 5/6 caught, 153 s (over the 5) / 2.0 km; PPO r2 vs hardened r1 167 s / 1.97 km; PPO r2 vs hardened r2 36 s / 0.38 km;
PPO r1 vs hardened r2 41 s / 0.72 km. Every row 0/18 clean false alerts.

**TensorRT** (`results/trt_bench.json`, AC power, L3 r2): 16,384 streams per batch — L2 FP16 6.7 ms vs ONNX Runtime
CPU 232 ms (35×), L3 FP16 9.2 ms vs 80 ms (8.7×); FP16 changes 0 decisions; INT8 slower than FP16 and flips 3.0% of L3
decisions. Batch of 1: GPU ~0.2–0.6 ms vs CPU ~0.04–0.06 ms. On battery (`results/trt_bench_battery.json`) everything
is ~1.3–1.8× slower: 4,096 streams ~6 ms, 16,384 ~21.5 ms.

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
- **L3 confounds:** with all attacks starting at 35%, L3 learned "late flight = forged" and saturated on clean stage-1
  entry/landing burns. Fixed by spreading training onsets (10/40/70%) and phase-balancing (`features.phase_balance`).
  Then it learned "near the corridor edge = forged" on held-out stage-1 coasts (zeroing the corridor feature: logit
  23 → 1), so corridor is excluded from L3 features (`features.L3_EXCLUDED_RULES`). L3 is its own alert stream (max,
  not noisy-OR, because L1+L2 already sit at 0.655 on abs-2a) with sustained persistence 40 of 50 frames
  (`l3.L3_PERSIST_K/N`), calibrated on clean train+val flights, so 1–3 s boostback transients don't set its threshold.
- **TensorRT 11 is strongly typed**: no FP16/INT8 builder flags or implicit calibrator. FP16 = ONNX converted with
  onnxconverter-common; INT8 = ORT static QDQ (symmetric, per-channel, `QuantizeBias: False`, first layer float).
  ORT's histogram calibrators (percentile, entropy) crash on these graphs; min/max is used. TF32 makes "FP32"
  engines differ from CPU by ~3e-4. No GRU tactic exists for a 65,536 batch.
- **PPO cold start**: the first SB3 policy load after a server restart takes seconds (≈1 min of flight at 10×).
  Fixed: cached `load_policy` + `preload_policies` in the server lifespan (log line "attacker policies preloaded").
- **Benchmark plugged in**: on battery (Balanced plan) CPU and GPU timings are ~1.3–1.8× slower.
- Quad: `quad_scenarios()` used to include `sitl:udp:14550` (blocks on a socket) — fixed. Quad test split = held mission
  (hover) on seeds +1000.
- SMAP original S3 bucket 403s; `scripts/fetch_smap.py` uses the TranAD mirror.

## 6. What's left (in priority order)
1. ~~PPO preload fix~~ and ~~TensorRT rerun on AC power~~: done and verified (see Latest).
2. ~~Tiger Data~~ and ~~Cartesia~~: done and verified (see Latest). **Vultr (skipped, only if the user gets credits):**
   `VULTR_API_KEY` in `.env`, then `scripts/deploy_vultr.ps1 -DryRun`, show the user the plan + price (default
   `vc2-2c-4gb`, ~$0.027/h), create only after they confirm (`-Yes`); destroy with `scripts/deploy_vultr_destroy.ps1 -Yes`.
3. **Ship:** ~~push to GitHub~~ (done), Devpost image (console screenshot mid-attack), recorded backup demo video, screenshots
   of Cursor + Grok usage for SpaceXSI. Update the resume bullet to match what was built (TensorRT FP16/INT8 was
   benchmarked, the live path is ONNX Runtime; no Kubernetes; no C++ monitor).
4. **Optional (only if the user asks):** C++ MAVLink/UDP edge monitor; quad PPO; delete the stub model.

Cut order if behind: optional items → TensorRT rerun → Cartesia. Never cut: the live Falcon 9 attack-and-catch demo
and the SMAP/MSL benchmark number.

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
# arms race round 2
uv run python scripts/train_ppo.py --schema falcon9 --detector hardened --timesteps 1500000 --n-envs 10 --episode-len 1500 --out models/ppo_falcon9_r2.zip --metrics-out results/ppo_falcon9_r2_metrics.json   # ~40 min
uv run python scripts/make_forged_dataset.py --schema falcon9 --ppo-only ppo_r2
uv run python scripts/train_l3.py --schema falcon9              # hardened r2
uv run python scripts/eval_detector.py --schema falcon9 --l3 models/l3_falcon9_r2.onnx --out results/detector_eval_r2.json
uv run python scripts/arms_race_report.py                       # results/arms_race.json
# GPU (plugged in, servers stopped)
uv run --with tensorrt-cu12 --with "cuda-python>=12.6,<13" --with onnxconverter-common python scripts/trt_bench.py
```
If any L1 rule or τ changes, retrain PPO (it learns the detector), then forge → L3 → eval.
