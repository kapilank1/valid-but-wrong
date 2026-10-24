# Valid-But-Wrong — Cal Hacks 13.0 Plan

> **One-liner:** A CRC check tells you the bits arrived intact. It doesn't tell you they're true.
> Valid-But-Wrong watches spacecraft and drone telemetry links live, catches frames that pass every
> integrity check but describe physically impossible motion, and fights an RL attacker built to fool it.

Deadline: **Sun Oct 25, 9:00 AM PST** (hard). 3-min pitch + 2-min Q&A. Live or recorded demo, image, GitHub repo.
**All code must be written during the hackathon.** This doc is only the design. Don't start the repo early.

---

## 1. Prize targets (ranked by fit)

| # | Prize | Why we fit | What it costs us |
|---|-------|-----------|-----------------|
| 1 | **[SpaceXSI] Make it Legendary** ($400 + Cursor keyboards) | Real space data in (NASA SMAP/MSL spacecraft telemetry, real Falcon 9 ascent data). **Must be built in Cursor**, and more Cursor use scores higher. Using Grok Bot for planning earns bonus points. | Do everything in Cursor (Claude Code in Cursor's terminal counts). Keep the planning in Grok. Screenshot both for the Devpost. |
| 2 | **[Armadin] Best Security Hack** ($996) | Our core idea is a security idea: spoofing a link with frames that still pass CRC, plus an adversarial red-team agent. | Nothing extra. Lean on the attacker-vs-defender story. |
| 3 | **Cal Hacks Most Technically Advanced** | Physics-informed ML, a live link-layer MITM, and RL adversarial training, all in one system. | Nothing extra. This is the main overall-prize shot. |
| 4 | **[MLH] Best Use of Tiger Data** | Telemetry is time-series data by nature. Use hypertables for frames and continuous aggregates to drive the dashboard. | ~2 hrs. Replaces the Postgres we'd use anyway. |
| 5 | **[MLH] Best Use of Vultr** | Train the predictor and PPO agent on a Vultr GPU, and host the demo there. | ~1 hr. Claim the free credits at the start. |
| 6 | **[Cartesia] Best Use of Cartesia** ($1,516) | Voice callouts like mission control: *"Flight, GNC. Velocity channel diverging from altitude rate. Confidence ninety-four percent."* This makes for a very memorable demo. | ~1.5 hrs. Stream TTS when an alert fires. |
| — | **MLH Commit Fellowship**, **GoDaddy domain** | Free entries. Opt in on Devpost and register `validbutwrong.tech` (or similar). | 10 min. |
| stretch | **Best Hardware Hack** | Only if we build a physical man-in-the-middle box (see §6). | 3–4 hrs. Skip unless the core works by H18. |

**Skipping:** the finance tracks (Hyphenate, Visa, CRS, Capital One), Solana/x402, and Figma. Bolting on a sponsor that doesn't fit costs more judging credibility than it earns.

---

## 2. The idea in one picture

```
 Sources (real data)            Link layer (the attack surface)          Defender
 ┌──────────────────┐           ┌────────────────────────────┐          ┌──────────────────────────┐
 │ Falcon 9 ascent  │──frames──▶│ CCSDS-style / MAVLink v2    │──frames─▶│ L0  CRC + sequence check  │ ← passes!
 │ NASA SMAP/MSL    │           │ packets, CRC-16, seq #      │          │ L1  physics consistency   │ (alt' ≈ v·sinγ, etc.)
 │ ArduPilot SITL   │           │                            │          │ L2  learned next-step     │ (GRU/1D-CNN residuals)
 └──────────────────┘           │   ▲  MITM injector          │          │ L3  calibrated score →    │
                                │   │  (scripted + PPO agent) │          │     alert + explanation   │
                                └───┼────────────────────────┘          └────────────┬─────────────┘
                                    │                                                 │
                              Red-team agent                               Tiger Data (hypertables)
                         reward = induced error                                       │
                                − detection penalty                  Mission-control UI + Cartesia voice
```

**Why it's novel enough to pitch:** most anomaly detectors treat telemetry as noisy sensor data.
We treat the *link* as adversarial. The attacker controls the bits, gets every checksum right, and
tries to steer the operator's picture of the vehicle. The defender's edge is that physics constrains
what the frames are allowed to say.

---

## 3. Data (all public, all real)

| Dataset | Use | Notes |
|--------|-----|------|
| **Falcon 9 webcast telemetry** (community-scraped velocity/altitude per launch, e.g. GitHub `shahar603/Telemetry-Data`) | Hero demo: replay a real Falcon 9 ascent, attack it live | Low rate (~1–30 Hz) and only a few channels, which makes the physics checks easy to explain on stage |
| **NASA SMAP/MSL telemetry anomalies** (Hundman et al., KDD '18, `khundman/telemanom`) | Real spacecraft channels with labeled anomalies. Gives us a **benchmark number** to quote against Telemanom. | Strong answer to "does it actually work?" |
| **CMU ALFA** (47 fixed-wing flights with real faults) | Trains the aircraft predictor. Ties back to the resume bullet. | Optional if short on time |
| **ArduPilot SITL** (generated live) | Live MAVLink stream we can MITM in real time | `sim_vehicle.py` + `pymavlink`. Set up Docker at H0, since this install is the most likely to eat time. |

Stretch: ESA-ADB (ESA's 2024 anomaly benchmark from real missions) for a second spacecraft benchmark.

---

## 4. Architecture (modules)

ML lives in Python because that's where the libraries are. Everything the user sees is TypeScript, and all inference runs server-side.

```
valid-but-wrong/
  packages/
    link/        (py)  frame codec: CCSDS-like + MAVLink v2, CRC-16/X.25, seq numbers
    sources/     (py)  replayers: falcon9, smap/msl, alfa; SITL bridge
    attacker/    (py)  scripted attacks + Gymnasium env + PPO (Stable-Baselines3)
    detector/    (py)  FastAPI service: L1 physics rules, L2 GRU predictor (ONNX Runtime), scoring
    bench/       (py)  offline eval on SMAP/MSL → precision/recall/F1, detection latency
  apps/
    console/     (ts)  Next.js mission-control UI: true vs reported trajectory, channel strips,
                       alert timeline, attacker controls, WebSocket feed
  infra/
    docker-compose.yml  (SITL, detector, Tiger Data/Timescale, console)
```

**Interfaces (agree on these at H0 so people can work in parallel):**
- `Frame { t, vehicle_id, seq, crc, channels: {name: value} }`, sent over UDP/WebSocket as JSON (msgpack if time allows)
- `Verdict { frame_seq, l0_ok, l1_violations[], l2_residual, score, alert, explanation }`
- Tiger Data tables: `frames` and `verdicts` hypertables. Use a continuous aggregate `verdicts_1s` for the charts.

**Detector design:**
- **L1 physics (start here, it's the demo backbone):** kinematic consistency (∫v dt vs Δaltitude, attitude rate vs gyro), bounded jerk, cross-channel agreement. Each rule is a pure function that's easy to unit test.
- **L2 learned:** a small GRU that predicts the next step from a 1–2 s window, with per-channel residuals scored against a rolling robust z-score. Export to ONNX. **TensorRT and Kubernetes are cut.** On a laptop they add risk and nothing a judge can see.
- **Score:** a logistic combination of L1 + L2, with the threshold calibrated on clean replays (target FPR < 1%).

**Attacker ladder (each rung is a demo moment):**
1. Bias injection, a constant offset. L1 catches it instantly.
2. Slow drift ramp. Catching it is harder, and this is where L2 earns its keep.
3. Replay of old in-sequence frames. Caught by the dynamics mismatch.
4. **PPO agent** whose action is a per-channel perturbation and whose reward is *drift of the operator's picture − λ·detection*. Show it learning to evade, then retrain the detector on its frames and show the hole closing. This is the "legendary" moment.

---

## 5. Timeline (~36 hrs, assuming 4 people)

| Hours | A: Link + Attacker | B: Detector + Bench | C: Console (TS) | D: Data/Infra/Sponsors |
|------|----|----|----|----|
| H0–2 | Repo, interfaces, frame codec + CRC | Load Falcon 9 + SMAP, explore | Next.js skeleton, WS client | Vultr credits, Tiger Data instance, SITL Docker, Grok planning board |
| H2–8 | Replay → UDP stream, scripted attacks 1–3 | L1 physics rules + tests | Trajectory + channel strip charts | Ingest frames/verdicts to Tiger Data |
| H8–14 | Gymnasium env around detector | GRU predictor, ONNX export, FastAPI | Alert timeline, attacker control panel | Cartesia voice callouts |
| H14–18 | **MILESTONE: end-to-end demo with scripted attacks.** Freeze a working tag. | | | |
| H18–26 | PPO training on Vultr GPU | SMAP/MSL benchmark numbers, adversarial retrain | Polish, before/after "hole closing" view | (Stretch) hardware MITM box |
| H26–30 | Sleep in shifts. Seriously. | | | |
| H30–34 | Bug bash, record backup demo video | Final numbers into slides | Devpost image, README | Devpost write-up, Cursor/Grok screenshots |
| H34–35 | **Submit by 8:00 AM.** One hour of buffer before the 9:00 deadline. | | | |

**Cut order if behind:** hardware → ALFA → Cartesia → PPO (keep the scripted attack ladder) → L2 (L1 alone still demos).
Never cut: the live attack-and-catch on Falcon 9 data, and the SMAP benchmark number.

---

## 6. Stretch: physical MITM box (Best Hardware Hack)
Use an ESP32 or Raspberry Pi as a "compromised radio" between SITL (or a real flight controller over USB) and the ground station.
It forwards MAVLink, recomputes CRC, and injects attacker frames on a button press. On stage it's a physical box
with a red switch: flip it and the voice callout fires. Only attempt it if H18 hits on time.

---

## 7. Demo script (3 min)

1. **0:00 Hook (20 s):** "Every packet on screen passes its checksum. One of them is lying."
2. **0:20 Clean Falcon 9 ascent replay (20 s):** the true track and the reported track overlap.
3. **0:40 Attack ladder (60 s):** bias → caught. Drift → caught by L2 with a voice callout. Replay → caught.
4. **1:40 PPO attacker (40 s):** it evades, the reported track wanders off, and the detector stays quiet. Then hot-swap in the adversarially retrained detector and it catches the attack.
5. **2:20 Proof (25 s):** SMAP/MSL benchmark numbers vs the Telemanom baseline. Detection latency in ms.
6. **2:45 Who it's for (15 s):** satellite and launch ground stations, drone fleet operators, counter-UAS. This is the same engine as the space/airspace domain-awareness idea.

**Q&A prep:** Why not just encrypt or authenticate the link? Because compromised sensors and insiders sign valid lies, and plenty of legacy links have no auth. False-positive rate? How does it generalize to a new vehicle?

---

## 8. Judging criteria → how we score

- **Application:** ground stations already trust CRC. We show the gap with real vehicle data.
- **Intentionality:** the UI looks like a real flight-ops console, with one alert explained in plain language.
- **Functionality:** a frozen H18 tag plus a recorded backup video.
- **Creativity:** an adversarial RL red team against a physics-informed detector, which is rare at hackathons.
- **Technical complexity:** link-layer codec, physics + learned detection, RL, and time-series infrastructure.

---

## 9. Tooling rules for the team
- **Cursor is the IDE for everyone** (required for SpaceXSI). Run Claude Code inside Cursor's terminal. Commit often so the history shows the Cursor work.
- Put this `PLAN.md` in the repo root as shared context for Cursor and Claude Code.
- Do all planning and task tracking in Grok Bot (bonus points). Screenshot it.
- Every physics rule and the frame codec get unit tests (pytest). The CRC/codec tests are what prove an attack frame is "valid".
- **Resume honesty:** after the hackathon, update the Valid-But-Wrong bullet to match what we actually built (ONNX instead of TensorRT, no Kubernetes unless we really do it).
