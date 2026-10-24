# CONTINUE HERE: Valid-But-Wrong build state (packed 2026-10-24, ~00:40 PDT)

Read this first on the new computer. It covers what the project is, what's built, what was in progress
when the build moved machines, and exactly what to do next. `HANDOFF.md` is the chronological log, `PLAN.md`
is the original hackathon plan, and `docs/CONTRACT.md` + `vbw/schema.py` define the module interfaces.

**To resume with Claude Code:** open this folder in Cursor (the SpaceXSI track requires Cursor), start Claude Code
in Cursor's terminal, and say: *"Read CONTINUE_HERE.md and HANDOFF.md, then continue the build from 'What's left'."*

---

## 1. What this is
Cal Hacks 13.0 project. Submit by **Sun Oct 25 2026, 9:00 AM PST** (aim for 8:00).

A man-in-the-middle rewrites telemetry frames and recomputes valid CRCs, so every frame passes the link checks
(sync, CRC-16, sequence, time). Valid-But-Wrong catches the lies anyway:
- **L1:** physics consistency rules
- **L2:** a GRU next-step predictor
- **L3:** a discriminator trained on attacker output
- A PPO red-team agent learns to evade the detector, and then the detector is hardened against it.

Data is real: 31 Falcon 9 launches of webcast telemetry, plus NASA SMAP/MSL spacecraft telemetry.

Prize targets: SpaceXSI (built in Cursor, with a Grok Bot bonus), Armadin Best Security, Most Technically Advanced,
MLH Tiger Data, MLH Vultr, Cartesia.

## 2. Setup on the new machine
```bash
# prerequisites: uv (https://docs.astral.sh/uv/), Node 20+ and pnpm, git
cd valid-but-wrong
uv sync                               # Python 3.12 env with torch, SB3, onnxruntime, fastapi...
uv run pytest -q                      # expect 199 passed
cd apps/console && pnpm install && cd ../..
```
This zip has all the data (`data/processed/falcon9`, `data/raw/smap_msl`), so no re-download is needed.
If anything is missing: `uv run python scripts/fetch_falcon9.py` and `uv run python scripts/fetch_smap.py`.

Run the demo:
```bash
uv run python scripts/serve.py                  # API + WebSocket on :8000
cd apps/console && pnpm dev                     # console on :3000 (add ?mock=1 to run without the server)
```

## 3. What's built (all tests pass: 199)

| Module | State | Notes |
|---|---|---|
| `vbw/schema.py`, `vbw/link/codec.py` | DONE | CCSDS-style frame, CRC-16/CCITT-FALSE, LinkMonitor (L0). |
| `vbw/sources/` | DONE | Falcon9Source (adds seeded sensor noise), QuadSim (physically consistent, 200 Hz internal), SpacecraftSource (SMAP/MSL), SitlSource (MAVLink bridge, **never run against real ArduPilot**, see docs/SITL.md). Registry has 43 scenarios. Hero = `falcon9:thaicom-8`. |
| `vbw/detector/` | MOSTLY DONE | L1 physics (`physics.py`, thresholds in `models/l1_thresholds.json`), L2 GRU exported to `models/l2_falcon9.onnx`, scoring/alerting (`base.py`, `alerting.py`), HardenedDetector + L3 slot (`l3.py`), `scripts/train_l3.py`, registry. **Missing:** a trained L3 model, `results/detector_eval.json`, and quad L1 calibration + quad L2. |
| `vbw/attacker/` | PARTIAL | Scripted attacks (none, bias, drift, replay, freeze, coordinated), registry, `SpoofEnv` (Gymnasium), `PpoAttack`, evaluate.py. **No real PPO model yet** (only `models/ppo_falcon9_stub.zip`, trained against a stub). |
| `vbw/pipeline/`, `vbw/store/`, `infra/` | DONE | FastAPI + WS server per CONTRACT, runner with hot-swap, Timescale writer + SQL (hypertables + `verdicts_1s` continuous aggregate), docker-compose, docs/DEPLOY_VULTR.md. Verified end to end: thaicom-8 + bias at 60 s → 0 false alerts, alert at t=60.4 s. |
| `vbw/bench/` | DONE (numbers below) | Telemanom-style per-channel predictor + Hundman dynamic thresholding, event-level scoring. |
| `apps/console/` | PARTIAL / UNVERIFIED | Next.js 15 mission console. All panels exist (HeroPlot, ChannelStrips, LinkPanel, DetectorPanel, AlertBanner, EventLog, AttackLadder, BenchCard, TopBar), plus mock stream, voice, and `app/api/tts` (Cartesia proxy). The agent was cut off before its final `pnpm build` / vitest pass, so run both first. |

### Measured benchmark (results/bench_summary.json), event-level, NASA SMAP/MSL
| | Precision | Recall | F1 |
|---|---|---|---|
| SMAP | 0.757 | 0.768 | 0.763 |
| MSL | 0.900 | 0.500 | 0.643 |
| Combined | 0.789 | 0.676 | 0.728 |
| Telemanom (Hundman et al. 2018, as reported) | 0.875 | 0.80 | 0.836 |

An ablation (`results/bench_summary_residual.json`) scored lower (combined F1 0.659). Present the main run honestly:
it's below Telemanom on F1. The pitch is the attack-detection story, and the benchmark shows the predictor is real.

## 4. Key findings so far (don't relearn these)
- **The clean Falcon 9 CSVs are only loosely self-consistent** (webcast OCR): |vz − d(alt)/dt| p99 ≈ 65–80 m/s,
  so the L1 rules are windowed and integrated, not pointwise.
- **Never feed noise-free CSVs to the detector.** The `frozen_channel` rule fires on them, giving false alerts on all
  31 launches. Always use noisy, wire-rounded runs: `vbw.detector.data.clean_runs(schema, split, seeds)` or `Falcon9Source`.
  Measured on noisy frames: 0 of 31 launches false-alert (157,430 frames).
- **PPO failed to learn at first** because the step bounds were too large: exploration tripped the alert within about 9 steps.
  Already fixed in `vbw/attacker/env.py` (step_bound (10, 1, 1, 10, 1) per 0.1 s step, picture_gain 200), and
  train_ppo.py should use `log_std_init=-1`. Not retrained since the fix.
- Hero launch: `thaicom-8` has the best data quality, but it's in the detector's TRAINING split. For an honest demo,
  check that `ses-9` (holdout, 545 s, has all events) runs with 0 false alerts and consider making it the hero
  (`HERO_SLUG` in `vbw/sources/registry.py`).
- The bench's original S3 bucket returns 403. `scripts/fetch_smap.py` uses the TranAD GitHub mirror.
- Docker isn't required for anything except the local TimescaleDB.

## 5. What was in progress when packed
I was fixing the attacker so it trains and evaluates on the same noisy frames the detector sees:
- DONE in `vbw/attacker/env.py`: `load_training_trajectories()` now uses `vbw.detector.data.clean_runs` (noisy,
  seeds 0–2). The env warms the detector on 150 clean frames before each episode. `make_detector_factory()` now
  uses the real detector and fails loudly, with no silent stub fallback. Tests still pass.
- NOT DONE in `scripts/make_forged_dataset.py`: `_trajectories()` and `eval_attacks()` still load noise-free CSVs
  via `splits.load_falcon9`. Switch both to `clean_runs(schema, "train")` and `clean_runs(schema, "holdout")`, and
  remove the stub fallback in `eval_attacks`.
- **`data/cache/forged_falcon9_*.npz` and `results/attacks_eval.json` are INVALID** (built from noise-free
  CSVs and evaluated against the stub). Regenerate both before training L3.
- `scripts/train_ppo.py`: its stub fallback branch (saves `ppo_<schema>_stub.zip`) is now unreachable, because
  `make_detector_factory` raises instead of falling back. Simplify it.

## 6. What's left (in priority order)
1. **Fix make_forged_dataset.py** (section 5), then run it:
   `uv run python scripts/make_forged_dataset.py --schema falcon9`
2. **Train PPO for real** (run it in the background; a few minutes on CPU):
   `uv run python scripts/train_ppo.py --schema falcon9 --detector base --timesteps 300000`
   → `models/ppo_falcon9.zip`. Check that it actually lies without alerting. If not, lower the step bounds or raise warmup.
   Then rerun make_forged_dataset.py so it includes `forged_falcon9_ppo.npz`.
3. **Train L3**: `uv run python scripts/train_l3.py --schema falcon9` → `models/l3_falcon9.onnx` + meta.
   The demo moment: PPO evades `base`, and the hot-swap to `hardened` catches it. Measure both.
4. **Detector eval**: write `results/detector_eval.json`: false alerts/flight on holdout, detection rate + latency per
   attack, base vs hardened. These numbers go in the pitch.
5. **Quad**: `uv run python scripts/calibrate_l1.py --schema quad` and `uv run python scripts/train_l2.py --schema quad`.
6. **Console**: `cd apps/console && pnpm build && pnpm test`, fix any errors, then run against the live server and click
   through the attack ladder (Bias → Drift → Replay → Coordinated → PPO → Harden). Fix layout at 1440×900.
7. **Keys and DB, the only things that need the human:**
   - `DATABASE_URL` (Tiger Data / Timescale), then `uv run python scripts/db_init.py --check`
   - `CARTESIA_API_KEY` in `apps/console/.env.local` (voice callouts; falls back to browser speech)
   - Vultr account if deploying (docs/DEPLOY_VULTR.md)
8. **Ship**: README.md (how to run, architecture diagram, numbers), docs/DEVPOST.md write-up, a Devpost image,
   and a recorded backup demo video. Commit often in Cursor, and screenshot Cursor + Grok usage for SpaceXSI.
9. Nice to have: make the server Docker image use CPU-only torch (currently ~6 GB because of CUDA wheels).

Cut order if behind: quad → Docker → L3 (still show PPO evading) → PPO (keep the scripted ladder).
Never cut: the live Falcon 9 attack-and-catch demo, and the SMAP/MSL benchmark number.

## 7. Notes
- Hackathon rule: all work must be done during the event, and SpaceXSI requires Cursor. Keep building and committing in Cursor.
- Resume honesty: when it's done, update the resume bullet to match what was actually built (ONNX Runtime,
  not TensorRT; no Kubernetes).
- `memory/` in this zip has the Claude memory notes about this project and the user, for context only.
