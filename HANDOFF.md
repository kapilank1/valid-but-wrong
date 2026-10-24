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
- Next for the orchestrator: integrate once reports land. Then train quad L1/L2 if pending, PPO for real, forged dataset → train L3 → hardened detector, browser-verify the console against the live server, then README + docs/DEVPOST.md.
