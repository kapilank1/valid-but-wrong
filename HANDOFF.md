# HANDOFF / build memory

Living log of the build. If a session is cut off, the next one starts here.

## Context
- Hackathon: Cal Hacks 13.0. Submit by **Sun Oct 25 2026, 9:00 AM PST** (aim for 8:00).
- Targets: SpaceXSI (must be built in Cursor, with Grok Bot bonus), Armadin Best Security, Most Technically Advanced, MLH Tiger Data, MLH Vultr, Cartesia.
- Plan: `PLAN.md`. Contract between modules: `docs/CONTRACT.md` + `vbw/schema.py`.
- The user asked for the full build to run autonomously. Only check in with them to connect a DB or API keys.

## Environment
- Python 3.12 via uv (`uv sync`, `uv run pytest`). Node 25, pnpm. Docker installed but the daemon wasn't running at start.
- M4 Pro, 24 GB, so training runs on CPU/MPS.

## Status log
- 2026-10-23 19:30: repo scaffolded, schema + CCSDS-style codec + tests written. Falcon 9 fetch script written.
- 2026-10-23 ~19:50: data fetched. 31 F9 launches → data/processed/falcon9 (committed); SMAP/MSL 82 channels → data/raw/smap_msl (gitignored, re-fetch with scripts/fetch_smap.py; the original S3 returns 403, so it uses the TranAD GitHub mirror).
  Clean F9 consistency: |vz − dalt/dt| p99 ≈ 65–80 m/s, |hypot − speed| p99 ≈ 8–10, |accel − dv/dt| p99 ≈ 10. Pointwise rules are noisy; use windowed ones.
- Commit ae3805f: scaffold.
- Six parallel subagents launched (each owns its own dirs, no git):
  1. sources → vbw/sources (Falcon9Source, QuadSim, SpacecraftSource, SitlSource/MAVLink, registry), docs/SITL.md
  2. detector → vbw/detector (L1 physics, L2 GRU ONNX, scoring/alerts, Hardened + L3 slot, registry), scripts/calibrate_l1|train_l2|train_l3.py, models/
  3. attacker → vbw/attacker (bias/drift/replay/freeze/coordinated/ppo, SpoofEnv, registry), scripts/train_ppo.py, make_forged_dataset.py
  4. console → apps/console (Next.js mission control, mock mode, Cartesia TTS proxy)
  5. pipeline → vbw/pipeline (runner, hub, FastAPI server), vbw/store (Timescale), infra/, docs/DEPLOY_VULTR.md
  6. bench → vbw/bench, scripts/run_bench.py, results/bench_summary.json, docs/BENCH.md
- 2026-10-23 ~23:00: all six agents were cut off by the API usage limit (it reset at 00:10). Their work was saved.
- 2026-10-24 00:12: WIP commit 279668b; 152 tests pass. All six resumed.
  Existing at the checkpoint: sources (all 4 + registry), detector physics/L2/train (missing base.py, l3.py, registry.py),
  attacker scripted/env/registry/evaluate + forged npz in data/cache (PPO only trained vs a stub, and the policy still
  alerts at 0.7 s), pipeline runner/hub/server/session + store, bench (results/bench_summary.json exists), console partially done.
  If agents die again: resume them, or finish by hand in this order: detector registry/base → PPO → L3 → console → integration.
- Sources DONE (55 tests). 43 scenarios; hero = falcon9:thaicom-8 (best data quality, but it's in the detector TRAIN split).
  TODO integration: if the base detector has 0 false alerts on held-out ses-9, make ses-9 the hero for an honest demo.
  SITL bridge has never run against real ArduPilot. Quad has a touchdown accel blip. Yaw is wrapped.
- Pipeline/store/infra DONE (46 tests). `uv run python scripts/serve.py` (port 8000). The full real stack runs end to end:
  thaicom-8 + bias@60 s gives 0 false alerts and an alert at t=60.4 s. Extra endpoints: /api/runs, /api/runs/{id}/timeline, /api/analytics.
  SQL (infra/sql/001_init.sql) has never run on a real TimescaleDB. First thing to do once there's a DATABASE_URL: `scripts/db_init.py --check`.
  Server Docker image is ~6 GB (CUDA torch). NEXT_PUBLIC_VBW_API is fixed at console build time.
- Detector: registry + BaseDetector (L1+L2) live, l2_falcon9.onnx exported. L3 not trained, no detector_eval.json.
- Attacker agent stopped partway. Its forged npz + attacks_eval were built from noise-free CSVs and are INVALID.
  The orchestrator fixed env.py (noisy runs, detector warmup, no stub fallback); make_forged_dataset.py is still TODO.
- 2026-10-24 ~00:40: this machine can't handle the workload, so everything was packed into a zip for another computer.
  199 tests pass. Full state and next steps: **CONTINUE_HERE.md**. Then train quad L1/L2 if pending, PPO for real, forged dataset → train L3 → hardened detector, browser-verify the console against the live server, then README + docs/DEVPOST.md.
- 2026-10-24 01:20 (new machine: Windows 11, Ryzen AI 7 350, 8C/16T, 31 GB; **git is not installed here**, so nothing is committed).
  - make_forged_dataset.py now uses noisy clean_runs (train for datasets, holdout for eval) and the real detector only.
    Forged sets regenerated (6 attacks, ~37k windows each).
  - **The "0 of 31 false alerts" claim was wrong.** At τ = 0.5 the base detector alerted on 3 clean launches
    (abs-2a, jason-3: train; nrol-76: val). Cause: webcast-OCR stretches where altitude lags vz and then jumps; L1 ≈ 0.28
    and L2 ≈ 0.39 are each under their limit, but their noisy-OR crosses 0.5. New `scripts/calibrate_alert.py` sets τ
    from the combined score on train+val runs → **falcon9 τ = 0.66** (models/l1_alert.json), 0/9 clean test runs alert.
    Bias/replay/freeze still caught in < 1 s, coordinated in ~5 s. Drift: 3/6 holdout launches at ~239 s (the L3 gap).
  - PPO reward was broken: "do nothing" (-14) and "get caught at once" (-12) both beat any real lie. Now only score
    above a 0.3 margin is penalised, picture_gain 2000, alert_penalty 50. downrange/vx step bounds cut 10× (exploration
    noise on them tripped the downrange-rate rule in ~150 steps). Retraining: 1M steps, 12 envs, 800-step episodes.
  - Hero is now **ses-9** (holdout; clean peak 0.47 vs τ 0.66 on 6 seeds through Falcon9Source).
  - Quad: data.quad_scenarios() included sitl:udp (blocks on a socket); fixed. Test split = held mission on seeds +1000.
    calibrate_l1 quad, train_l2 quad, calibrate_alert quad --seeds 10 → τ 0.53, 0/3 clean test runs alert.
  - Console: build + 57 vitest pass. Gauge now shows the calibrated τ (catalog detectors carry `thresholds` per schema).
  - Linux torch now comes from the PyTorch CPU index (uv.lock), so the server image drops the CUDA wheels.
    On this machine uv needs `--system-certs` for anything that hits the network (TLS interception).
  - docs/BENCH.md placeholders filled from results/bench_summary*.json.
- 2026-10-24 ~02:00
  - First PPO (5 action dims) was caught by base on 6/6 holdout launches, mostly in 2-16 s: it drifted vx/downrange
    out of step and the downrange-rate rule (~1 m/s limit) caught it. The policy now controls altitude, vz, speed only
    (picture = altitude), alert_penalty 100.
  - **New L1 rule `vz_vs_altitude_rate_60s`** (physics.py): same integrated check over 60 s. Clean max falls from
    81.9 m/s (10 s window) to 13.6 m/s, because OCR lag-then-jump stretches cancel; threshold 20.4 m/s. Drift @30 m/s
    is now caught on 6/6 holdout launches (was 3/6), 73-226 s after onset. Drift @12 m/s: 3/6. τ unchanged (0.66).
    L3 features skip rules with lookback > 150 (features.l3_rules), so L3 windows stay 160 frames.
    SpoofEnv warmup 150 → 620 frames so the new rule is live from the first step of every episode.
  - L3 dry run on scripted sets: flags freeze 99.6%, coordinated 95%, replay 72%, drift 1.6%, bias 4% of forged
    val windows; logit threshold 12.0 (the 99.95% clean-val quantile, pushed up by nrol-76's glitch windows).
  - Callouts now pick the same layer as the written explanation (they disagreed on the live bias alert).
  - PPO v3 training against the new base: 1.5M steps, 12 envs, 1500-step episodes.
- 2026-10-24 02:20–02:50
  - PPO v3 done (`models/ppo_falcon9.zip`, 1424 s). It learned the coordinated lie (vz +120 m/s, altitude integrated to
    match). vs base on 6 holdout launches: all caught eventually, mean 190 s, mean max lie before alert 12.3 km
    (3/6 survive 312–320 s with 23–24 km). Train launches: 11/15 caught, 17 km mean. Scripted drift: 137 s, 4.1 km.
  - L3 v1 with PPO windows: PPO windows flagged 8%, logit threshold 43.7. Streaming check showed the cause: L3 saturated
    on clean first-stage flights (nrol-76, x-37b-otv-5, zuma) during entry/landing = "late flight = forged" confound.
  - Fixes: make_forged_dataset trains on onsets 10/40/70% of flight (eval report still 35%), PPO set built in
    parallel at all three onsets; train_l3 phase-balances windows (features.phase_balance) and sets the logit
    threshold from 3-of-5 persistence on clean train+val flights; hardened score = max(noisy-OR(L1,L2), l3_score).
  - L3 v2: PPO 59%, coordinated 86%, replay 98%, freeze 99%, drift 10%, bias 13% of forged val windows; 0% of clean
    val windows; threshold 28.53 (set by nrol-76 seed 0/1 entry-burn glitch; next clean flight 11.3).
  - 211 tests pass. (Session paused here briefly; the user then chose to keep building in Claude desktop, not Cursor.)
- 2026-10-24 ~03:00–03:30
  - eval_detector run #1: hardened == base on PPO (190 s). Streaming L3 logits showed clean first-stage coasts near
    apogee saturating; ablation proved the **corridor** feature drove it (zeroed: logit 23 → 1). L3 now excludes
    corridor (`features.L3_EXCLUDED_RULES`). Threshold then 19.96, set by 1–3 s boostback-shutdown transients on
    nrol-76, so L3 got its **own alert stream with sustained persistence 40 of 50 frames** (`l3.L3_PERSIST_K/N`);
    base keeps 3 of 5. Threshold now 6.88.
  - **Final eval (`results/detector_eval.json`)**, 6 holdout launches, attack at 35%, intensity 1.0:
    clean 0/18 runs alert for base and hardened (2.58 flight-hours; test-only 0/9, 1.27 h); 1.19 / 1.40 ms per frame.
    PPO: base 190 s / 12.3 km lie → hardened **66 s / 1.95 km**. SES-9: base 315 s / 23.8 km → hardened 95 s / 3.4 km.
    Drift: 137 s → 59 s. Coordinated: 5.3 s → 3.6 s. Bias/replay/freeze 0.6 s both. Intensity 0.4 bias/drift: 3/6.
  - README results + DEVPOST results/challenges filled (by a subagent, numbers checked against the JSON).
  - **Live console verified** (SES-9 at 10×): PPO at T+234 s → 3.7 km lie, base score 0.03, no alert; Harden →
    alert 0.1 s later, "forged-link signatures (P(forged) = 1.00, sustained)". Replay caught in 0.6 s. Bias, drift,
    clear verified earlier. Coordinated at the console's default intensity 0.5 evaded both detectors on SES-9 (2.65 km
    by T+7:07) — use intensity 1.0 for that rung in the demo.
  - 213 tests pass (new tests/test_detector_l3_stream.py).
