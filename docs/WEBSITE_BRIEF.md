# Website brief: Valid-But-Wrong project site

This is the single source of truth for building the project website that judges will read. Every number here comes
from a file in the repo (listed next to it). If the site needs a number that isn't here, read it from the source file;
never estimate or round in a way that changes meaning. When this brief and a results file disagree, the file wins.

## Audience and goal

Cal Hacks 13.0 judges, skimming in a few minutes, often on a phone. After 60 seconds they should understand:
1. **The problem:** a man-in-the-middle can rewrite telemetry and re-sign it with a valid CRC, so every packet passes
   the link checks, and the operator sees a rocket kilometres from where it really is.
2. **The idea:** check whether the telemetry is *physically possible* (L1), whether it matches *how this vehicle flies*
   (L2), and whether it looks like *what attackers produce* (L3).
3. **The twist:** a reinforcement-learning attacker learned to beat the detector, and the detector was hardened
   against it, for two rounds.
4. **The proof:** real data (31 Falcon 9 launches, NASA SMAP/MSL), held-out evaluation, zero false alerts, honest
   limits.

Tone: confident, precise, mission-control. Not hype. Show limits openly; it builds trust.

## Tagline and one-liners

- Tagline: **"A CRC tells you the bits arrived intact. It doesn't tell you they're true."**
- One-liner: Valid-But-Wrong catches telemetry that passes every integrity check but describes physically impossible
  motion, and fights an AI attacker trained to fool it.

## Key numbers (hero stats) and where they come from

| Claim | Number | Source |
|---|---|---|
| Hero result on SES-9 (held-out launch), PPO attacker | Base: lie of **23.8 km** over **315 s** before alert. Hardened: caught at **34 s**, lie **0.6 km** | `results/detector_eval_r2.json` (per_run for ppo), README |
| PPO agent, 6 held-out launches | Base 190 s / 12.3 km → hardened 41 s / 0.72 km | `results/detector_eval_r2.json`, `results/arms_race.json` |
| False alerts | **0 of 18** clean held-out flights (6 launches × 3 noise seeds, 2.58 flight-hours), both detectors; 0 of 9 on the test half (1.27 h) | `results/detector_eval_r2.json` |
| Forged frames that pass the CRC/link checks | **100%** (by construction: the attacker re-encodes with a valid CRC) | `vbw/link/codec.py`, live console readout |
| Real data | 31 Falcon 9 launches (webcast telemetry, 10 Hz); NASA SMAP/MSL: 82 labeled streams (81 unique channels), 105 labeled anomalies | `data/processed/falcon9/index.json`, `docs/BENCH.md` |
| Arms race | PPO r2 (retrained vs hardened r1) survived 2.5× longer (167 s vs 66 s); hardened r2 catches it in 36 s / 0.38 km | `results/arms_race.json` |
| SMAP/MSL benchmark | Combined F1 **72.8%** (P 78.9 / R 67.6) vs Telemanom 83.6% — **below the published baseline**, stated plainly | `results/bench_summary.json`, `docs/BENCH.md` |
| TensorRT fleet scale | FP16 engines on an RTX 5060 Laptop GPU score **16,384 vehicle streams in ~16 ms** (inside a 50 Hz frame), 35× (L2) / 8.7× (L3) faster than CPU, **0** decisions changed | `results/trt_bench.json`, `docs/TENSORRT.md` |
| C++ edge monitor | Bit-exact with the Python detector on 16,766 frames (max error 0, identical alerts); **30 µs per frame**, 32,550 frames/s on one core (Python: 1.19 ms/frame, ~40× slower) | `results/edge_bench.json`, `edge/` |
| Detector cost (Python) | 1.1 ms (base) / 1.3 ms (hardened) per frame vs a 100 ms frame period | `results/detector_eval_r2.json` |
| Tests | 221 Python tests (+1 GPU test skipped without TensorRT), 75 console tests, 3 C++ test suites | `tests/`, `apps/console`, `edge/tests` |

Full results tables to show (copy from README "Results" and "Arms race" sections, which are generated from the JSON):
the base-vs-hardened table over 6 held-out launches (bias, drift, replay, freeze, coordinated, PPO r1, PPO r2) and the
arms-race table (6 attacker/defender pairings).

## Story structure (suggested page sections)

1. **Hero:** tagline, one-liner, the SES-9 stat (23.8 km undetected vs caught at 34 s), buttons: GitHub, Devpost,
   "Watch the demo" (video link TBD by the team).
2. **The attack:** an animated telemetry packet: sync ✓, CRC ✓, sequence ✓, time ✓ … altitude **wrong by 3.7 km**.
   Explain the man-in-the-middle in 3 sentences. "LINK 100% VALID / PICTURE OFF BY 3.7 km" is the console's own
   readout and a great visual motif.
3. **How it catches lies:** the four layers (L0 link, L1 physics, L2 learned dynamics, L3 discriminator) as cards,
   each with one concrete example from the project:
   - L1: "vertical velocity must match the rate of change of altitude", checked over 10 s and 60 s (over 60 s the
     webcast's own glitches cancel: clean max 13.6 m/s vs 81.9 m/s at 10 s).
   - L1: dynamic pressure must match altitude through the atmosphere (catches altitude lies inside the atmosphere).
   - L2: a GRU predicts the next frame from the last 2 s; robust z-scores catch steps and drifts.
   - L3: a 1D-CNN trained on what attackers actually produce; its own sustained (40 of 50 frames) alert.
   - Alerting: thresholds calibrated so no clean training/validation flight ever alerts.
4. **The attack ladder:** bias → drift → replay → freeze → coordinated (self-consistent fake) → PPO agent → PPO round 2,
   with what catches each (table from README).
5. **The arms race:** a visual timeline/chart of the 6 pairings in `results/arms_race.json` (latency and lie size per
   round). Narrative: PPO r1 lies 12.3 km past base → hardened r1 → PPO r2 learns a quieter lie and survives 2.5× longer
   → hardened r2 catches both. Mention the PPO reward-design lesson (first version learned to do nothing).
6. **Interactive replay (the showpiece):** SES-9 flight, PPO attack from T+190 s. Plot true altitude vs reported
   altitude over time (the gap is the lie), and the base vs hardened detector scores with their alert markers and the
   threshold line. Let the user scrub time. Data: export it with a script (see "Data for the site").
7. **Results:** the full tables, false alerts, cost per frame.
8. **Engineering depth:** TensorRT fleet benchmark (FP32/FP16/INT8; INT8 honestly didn't pay off), the C++ edge
   monitor (bit-exact parity, 30 µs/frame), the NASA benchmark, test counts, calibration story (τ = 0.66 set by the
   worst clean training flight; L3 confounds found and fixed).
9. **Architecture:** the data-flow diagram (Mermaid in README "How it works") rendered as a clean SVG, plus the stack.
10. **Sponsor tech (only what is true):**
    - **Tiger Data / TimescaleDB:** live; hypertables for frames + verdicts, a `verdicts_1s` continuous aggregate,
      refresh/compression policies; verified on Tiger Cloud (a live demo run wrote 8,961 rows with 0 errors; run
      history, per-second timeline and analytics served from it). Source: `docs/INTEGRATIONS.md`, `infra/sql/001_init.sql`.
    - **Cartesia:** live; mission-control voice callouts on every alert ("Flight, Guidance. Altitude disagrees with
      vertical velocity. Possible spoofed telemetry. Confidence 94 percent."), sonic-3.6. Source: `docs/INTEGRATIONS.md`.
    - **Cursor + Grok:** the team builds in Cursor (SpaceXSI track requirement).
    - **NVIDIA TensorRT:** benchmark (not the live path).
    - **Vultr:** deploy scripts are written (`scripts/deploy_vultr.ps1`) but NOT deployed; do not claim a deployment.
11. **Limits (be explicit):** below Telemanom on the NASA benchmark; slow altitude drift under ~20 m/s above the
    atmosphere is the hardest case (hardened r2 still catches it, but after ~132 s at low intensity); webcast data is
    loosely self-consistent; ArduPilot SITL bridge exists but never ran against real ArduPilot; the quadcopter path
    runs on our simulator; the C++ monitor's live UDP end-to-end run is not yet measured.
12. **Team + links:** Cal Hacks 13.0. Team member names: ASK the team (don't invent). GitHub:
    https://github.com/kapilank1/valid-but-wrong. Devpost link: ask the team.

## Data for the site

- Copy (don't retype) these into the site at build time: `results/detector_eval_r2.json`, `results/arms_race.json`,
  `results/trt_bench.json`, `results/bench_summary.json`, `results/edge_bench.json`.
- The interactive replay needs a per-frame export that doesn't exist yet. Write `scripts/export_site_data.py` that runs
  SES-9 with the PPO attack starting at T+190 s and records per frame: t, true altitude, reported altitude, base score
  + alert, hardened score + alert (+ L1/L2/L3 parts if useful). Pattern to copy: `run_one` in `scripts/eval_detector.py`
  (`clean_runs(FALCON9, "holdout", (0,))` → the SES-9 run is the holdout run whose scenario is `falcon9:ses-9`;
  `make_attack("ppo", FALCON9, start_t=190.0)`; `make_detector("base"|"hardened", FALCON9)`; call
  `torch.set_num_threads(1)`). Downsample to ~2 Hz for size. Also optionally the bias attack at T+60 s. Write JSON to
  `apps/site/public/data/`. Run with `uv run python scripts/export_site_data.py`.
- Screenshots of the live console make the site much better (title row, attack ladder, hero plot with the lie, alert
  banner). The console runs with `uv run python scripts/serve.py` + `pnpm --dir apps/console dev` at
  http://localhost:3000 (▶ Demo button runs the whole pitch). Ask the team to capture them, or capture them with a
  browser tool if available. Save to `apps/site/public/images/`.

## Visual direction

Dark mission-control aesthetic matching the console: near-black background, monospaced numerals for telemetry,
**cyan = truth / god view**, **amber = reported / on the link**, red for alerts, green for "link valid". Big, legible
numbers. Smooth but restrained motion (packet animation, chart reveal). Must look great at 1440 px and on a phone.
Respect `prefers-reduced-motion`. Fast: static export, no heavy 3D.

## Rules for whoever builds the site

- Build it in `apps/site/` as its own app. Don't modify `vbw/`, `apps/console/`, `edge/`, `models/`, `results/`
  (except adding `scripts/export_site_data.py`). Don't change any detector or experiment code.
- Every number on the site must be traceable to a results file or this brief. No invented metrics, users, testimonials
  or team members.
- Don't claim: Kubernetes, a live Vultr deployment, live ArduPilot, TensorRT in the live path, state of the art on the
  NASA benchmark.
- Keep secrets out: never read or copy `.env` or `apps/console/.env.local`.
