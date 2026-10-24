# Valid-But-Wrong (Devpost write-up)

**Tagline:** A CRC tells you the bits arrived intact. It doesn't tell you they're true.

## Inspiration

Every telemetry link checks integrity: sync markers, checksums, sequence numbers. None of it checks *truth*. An
attacker who sits on the link and rewrites values can simply recompute the CRC, and every packet still looks
perfect. Spoofed GPS has already walked ships and drones off course this way. We wanted to know: if the bits are
fully under the attacker's control, what is left for the defender? Our answer is physics. A rocket's altitude,
velocity and acceleration have to agree with each other, and with how rockets actually fly.

## What it does

Valid-But-Wrong is a ground station that watches a live telemetry stream and flags frames that pass every link
check but describe impossible or implausible motion.

- It replays **real Falcon 9 ascents** (31 launches of webcast telemetry) and **NASA SMAP/MSL spacecraft telemetry**
  through a CCSDS-style link with CRC-16 and sequence numbers.
- A **man-in-the-middle** rewrites frames and re-encodes them with valid CRCs. Every forged frame passes L0.
- Three detection layers judge each frame: **L1** physics-consistency rules, **L2** a GRU that has learned how
  Falcon 9 flies, and **L3** a discriminator trained on attacker output.
- A **PPO red-team agent** learns, against the live detector, how to grow the operator's altitude error without
  tripping an alert. The detector is then **hardened** against it, and you can hot-swap detectors mid-flight.
- A **mission console** shows the true trajectory next to the reported one, every layer's score and explanation,
  and calls alerts out loud in a mission-control voice (Cartesia).

## Results

- **On SES-9, a launch the detector never trained on, the base detector lets the PPO attacker walk altitude off
  by 23.8 km over 315 s before alerting. The hardened detector catches it at 95 s (3.4 km).**
- Across six held-out launches, PPO is caught at 190 s mean with a 12.3 km lie by the base detector, and at 66 s
  mean with a 2.0 km lie by the hardened detector. Bias, replay and freeze are caught in 0.6 s, and the coordinated
  self-consistent fake in 3.6–5.3 s.
- **Zero false alerts** on 18 clean held-out flights (2.58 flight-hours), for both detectors. Cost is 1.2–1.4 ms per
  frame on CPU.
- NASA SMAP/MSL benchmark (event-level): F1 72.8% combined vs. Telemanom's 83.6%. We are below the published
  baseline and say so.
- **Fleet scale on a laptop GPU:** as TensorRT FP16 engines on an RTX 5060 Laptop GPU, the learned layers score
  16,384 vehicle streams in ~15 ms per frame (inside 50 Hz), 33× / 7.7× faster than the CPU, with zero detection
  decisions changed. INT8 was measured too and didn't pay off on models this small.

## How we built it

- **Python core** (uv, Python 3.12): link codec and L0 monitor; sources for Falcon 9 replay, a 200 Hz quadcopter sim,
  SMAP/MSL replay and an ArduPilot SITL MAVLink bridge; scripted attacks and a Gymnasium environment for the RL
  attacker (Stable-Baselines3 PPO).
- **Detector:** L1 has eleven windowed physics rules for Falcon 9 (vz vs. altitude rate over 10 s and over 60 s, vx vs. downrange rate, speed
  vs. velocity components, dynamic pressure vs. a fitted atmosphere, a flight corridor learned from every launch,
  and more). L2 is a GRU next-step predictor exported to **ONNX Runtime** (under 2 ms per frame on CPU). L3 is a
  1D-CNN over L1/L2 features of the last 4 s. Scores combine with a noisy-OR, and alerts need 3 of 5 frames over a
  threshold calibrated so no clean training or validation flight ever alerts.
- **GPU path:** TensorRT 11 engines (strongly typed, so FP16 via graph conversion and INT8 via explicit Q/DQ from
  calibrated static quantization) benchmarked against ONNX Runtime for fleet-scale scoring.
- **Pipeline:** FastAPI + WebSocket server with live attack/detector hot-swap, writing frames and verdicts to
  **Tiger Data (TimescaleDB)** hypertables, with a `verdicts_1s` continuous aggregate for the run timeline.
- **Console:** Next.js 15 + canvas rendering at 10 Hz, Cartesia TTS proxied through a server route, and a mock
  mode that runs entirely in the browser.
- **Benchmark:** we reimplemented the Telemanom protocol (Hundman et al., KDD 2018) on SMAP/MSL: per-channel LSTM,
  nonparametric dynamic thresholding, event-level scoring.
- **Infra:** Docker images, docker-compose with TimescaleDB, and a Vultr deployment guide.

## Challenges we ran into

- **Webcast telemetry is messy.** The Falcon 9 numbers are read off the broadcast overlay, so |vz − d(alt)/dt|
  reaches 65–80 m/s on clean data. Pointwise physics rules false-alert constantly, so every rule is windowed and
  integrated, and thresholds are calibrated on clean training launches only.
- **Two "half-suspicious" layers made a false alert.** On three clean launches the altitude readout lags and then
  jumps. L1 and L2 each stayed under their own limit, but their noisy-OR crossed the alert threshold. We
  recalibrated the threshold on the combined score itself, on training and validation launches only, and checked it
  on test launches the calibration never saw.
- **The RL attacker first learned to do nothing.** Our first reward penalized the detector's score every step, so
  "never lie" and "get caught immediately" both beat any real lie. Charging only for score above a safety margin,
  and making an alert far more costly than a slow lie is valuable, fixed it. A second policy got caught in seconds
  because it also nudged downrange, which our tightest rule checks, so the final agent lies about altitude and the
  velocities that go with it.
- **The discriminator learned shortcuts.** L3 first learned "late flight = forged", because every attack started
  mid-flight. Then it learned "near the edge of the training flights' envelope = forged": zeroing the corridor feature
  dropped its logit on a clean first-stage coast from 23 to 1. We spread attack onsets across the flight,
  phase-balanced the training windows, removed the corridor feature from L3, and gave L3 its own sustained
  persistence (over threshold on 40 of 50 frames), so 1–3 s boostback transients can't set its threshold.
- **Honest evaluation.** Every 5th launch is held out from all training, the demo runs on a held-out launch, and
  every number we quote is generated by a script into `results/*.json`.

## Accomplishments that we're proud of

- An end-to-end live system: real launch data → link codec → MITM → three detection layers → console with voice,
  all streaming at 10 Hz with hot-swappable attackers and detectors.
- A learned attacker and a learned defender that train against each other, on real flight data.
- A faithful reimplementation of a published spacecraft anomaly benchmark, with the numbers reported honestly even
  where we fall short of it.

## What we learned

- Integrity is not truth. Physics is a strong, cheap prior that attackers have to fight.
- Calibration matters more than model size: most of our work went into thresholds that never cry wolf on real
  flights.
- An RL attacker finds the weakest check fast, and the reward has to be designed as carefully as the detector.

## What's next

- Train the attacker against the hardened detector, and keep alternating rounds.
- Run the MAVLink bridge against real ArduPilot SITL and real drone logs (e.g. CMU ALFA).
- Higher-rate vehicle telemetry, where the physics rules can be much tighter.

## Built with

Python, PyTorch, ONNX Runtime, TensorRT, CUDA, Stable-Baselines3, Gymnasium, FastAPI, TimescaleDB / Tiger Data, Next.js, React,
TypeScript, Tailwind CSS, Cartesia, Docker, Vultr, uv, pnpm, Cursor.
