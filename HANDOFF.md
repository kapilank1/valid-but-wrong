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
